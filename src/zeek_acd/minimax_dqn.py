"""Minimax-DQN: Littman's (1994) minimax-Q algorithm for zero-sum Markov
games, with the tabular Q-function replaced by a neural network.

Because the game is zero-sum, a single network suffices: it outputs the
defender's payoff Q(s, a_d, a_a) for every joint action pair; the
attacker's payoff is its negation. The state-value used for TD
bootstrapping is the minimax value of the resulting matrix game, found by
solving the linear program:

    maximize_{pi_d, v}  v
    s.t.  sum_i pi_d[i] * Q[i, j] >= v   for every attacker action j
          sum_i pi_d[i] = 1,  pi_d >= 0

``pi_d`` is also the defender's robust (worst-case-optimal) mixed
strategy for that state, used for action selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linprog

from .game import N_ATTACKER_CLASSES, N_DEFENDER_ACTIONS
from .replay_buffer import Batch


def solve_matrix_game(Q: np.ndarray) -> tuple[float, np.ndarray]:
    """Q: (n_d, n_a) payoff matrix for the row (defender) player, who
    maximizes; the column (attacker) player minimizes. Returns
    (game value, defender's optimal mixed strategy over n_d actions)."""
    n_d, n_a = Q.shape
    c = np.zeros(n_d + 1)
    c[-1] = -1.0  # linprog minimizes; we want to maximize v

    A_ub = np.zeros((n_a, n_d + 1))
    A_ub[:, :n_d] = -Q.T
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(n_a)

    A_eq = np.zeros((1, n_d + 1))
    A_eq[0, :n_d] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, 1.0)] * n_d + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        pi = np.ones(n_d) / n_d
        v = float(np.min(pi @ Q))
        return v, pi

    pi = np.clip(res.x[:n_d], 0.0, None)
    total = pi.sum()
    pi = pi / total if total > 0 else np.ones(n_d) / n_d
    v = float(-res.fun)
    return v, pi


def batch_solve_fictitious_play(Q_batch: np.ndarray, iters: int = 150) -> tuple[np.ndarray, np.ndarray]:
    """Approximates the minimax value/strategy for a whole batch of matrix
    games at once via Brown's fictitious play, vectorized over the batch
    with numpy (no per-sample scipy LP call). For zero-sum games the
    average strategies converge to a Nash equilibrium and the average
    payoff converges to the game value (Robinson, 1951); a few dozen
    iterations is enough for the tiny (n_d x n_a) matrices used here, and
    it's what makes solving a value for every sample in a training batch,
    every gradient step, computationally practical.

    Used for the TD-target bootstrap over a training batch, where an
    approximate value is fine. Action selection at deployment/rollout time
    (a single state per call) instead uses the exact LP in
    ``solve_matrix_game``.
    """
    b, n_d, n_a = Q_batch.shape
    row_counts = np.ones((b, n_d), dtype=np.float64)
    col_counts = np.ones((b, n_a), dtype=np.float64)

    for _ in range(iters):
        col_dist = col_counts / col_counts.sum(axis=1, keepdims=True)
        row_payoffs = np.einsum("bij,bj->bi", Q_batch, col_dist)
        row_best = np.argmax(row_payoffs, axis=1)
        row_counts[np.arange(b), row_best] += 1.0

        row_dist = row_counts / row_counts.sum(axis=1, keepdims=True)
        col_payoffs = np.einsum("bi,bij->bj", row_dist, Q_batch)
        col_best = np.argmin(col_payoffs, axis=1)
        col_counts[np.arange(b), col_best] += 1.0

    row_dist = row_counts / row_counts.sum(axis=1, keepdims=True)
    col_dist = col_counts / col_counts.sum(axis=1, keepdims=True)
    values = np.einsum("bi,bij,bj->b", row_dist, Q_batch, col_dist)
    return values, row_dist


def batch_solve(Q_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return batch_solve_fictitious_play(Q_batch)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_d: int = N_DEFENDER_ACTIONS,
                 n_a: int = N_ATTACKER_CLASSES, hidden: int = 128):
        super().__init__()
        self.n_d = n_d
        self.n_a = n_a
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_d * n_a),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        out = self.net(state)
        return out.view(-1, self.n_d, self.n_a)


@dataclass
class MinimaxDQNConfig:
    state_dim: int
    n_defender_actions: int = N_DEFENDER_ACTIONS
    n_attacker_classes: int = N_ATTACKER_CLASSES
    hidden: int = 128
    lr: float = 1e-3
    gamma: float = 0.95
    tau: float = 0.01
    grad_clip: float = 10.0
    device: str = "cpu"


class MinimaxDQNAgent:
    def __init__(self, config: MinimaxDQNConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        self.online = QNetwork(
            config.state_dim, config.n_defender_actions, config.n_attacker_classes, config.hidden
        ).to(self.device)
        self.target = QNetwork(
            config.state_dim, config.n_defender_actions, config.n_attacker_classes, config.hidden
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.lr)

    @torch.no_grad()
    def defender_strategy(self, state: np.ndarray, use_target: bool = False) -> tuple[np.ndarray, float]:
        net = self.target if use_target else self.online
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        Q = net(s).squeeze(0).cpu().numpy()
        v, pi = solve_matrix_game(Q)
        return pi, v

    def select_action(self, state: np.ndarray, epsilon: float = 0.0,
                       rng: Optional[np.random.Generator] = None) -> tuple[int, np.ndarray]:
        rng = rng or np.random.default_rng()
        pi, _ = self.defender_strategy(state)
        if rng.random() < epsilon:
            action = int(rng.integers(0, self.cfg.n_defender_actions))
        else:
            action = int(rng.choice(self.cfg.n_defender_actions, p=pi))
        return action, pi

    def train_step(self, batch: Batch) -> float:
        states = torch.as_tensor(batch.state, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_state, dtype=torch.float32, device=self.device)
        d_actions = torch.as_tensor(batch.defender_action, dtype=torch.long, device=self.device)
        a_classes = torch.as_tensor(batch.attacker_class, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.done, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_Q = self.target(next_states).cpu().numpy()
        next_values, _ = batch_solve(next_Q)
        next_values_t = torch.as_tensor(next_values, dtype=torch.float32, device=self.device)
        targets = rewards + self.cfg.gamma * (1.0 - dones) * next_values_t

        Q = self.online(states)
        idx = torch.arange(Q.shape[0], device=self.device)
        pred = Q[idx, d_actions, a_classes]

        loss = F.smooth_l1_loss(pred, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        with torch.no_grad():
            for online_p, target_p in zip(self.online.parameters(), self.target.parameters()):
                target_p.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * online_p)

        return float(loss.item())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "config": self.cfg.__dict__,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "MinimaxDQNAgent":
        ckpt = torch.load(Path(path), map_location=device)
        cfg = MinimaxDQNConfig(**{**ckpt["config"], "device": device})
        agent = cls(cfg)
        agent.online.load_state_dict(ckpt["online"])
        agent.target.load_state_dict(ckpt["target"])
        return agent
