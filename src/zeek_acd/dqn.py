"""A plain (single-agent) Deep Q-Network, used for the self-play attacker
and, optionally, as an alternative "acts-on-its-own-belief" defender.

This is deliberately separate from ``minimax_dqn.py``: the minimax agent
outputs a full ``(defender_action, attacker_class)`` payoff *matrix* and
solves a game to act, whereas this agent outputs a plain vector of
Q-values over its own actions and acts greedily/epsilon-greedily. In
self-play the two players each optimize their own objective directly
(the attacker maximizes its reward, which is the negation of the
defender's), which is the standard "independent learners" route to an
approximate equilibrium of the zero-sum game.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNTransition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class SimpleReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, seed: int | None = None):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self._size = 0
        self._idx = 0
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    def push(self, t: DQNTransition) -> None:
        i = self._idx
        self.state[i] = t.state
        self.action[i] = t.action
        self.reward[i] = t.reward
        self.next_state[i] = t.next_state
        self.done[i] = float(t.done)
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self._size, size=batch_size)
        return (
            self.state[idx], self.action[idx], self.reward[idx],
            self.next_state[idx], self.done[idx],
        )


@dataclass
class DQNConfig:
    state_dim: int
    n_actions: int
    hidden: int = 128
    lr: float = 1e-3
    gamma: float = 0.95
    tau: float = 0.01
    grad_clip: float = 10.0
    device: str = "cpu"


class DQNAgent:
    def __init__(self, config: DQNConfig):
        self.cfg = config
        self.device = torch.device(config.device)
        self.online = MLP(config.state_dim, config.n_actions, config.hidden).to(self.device)
        self.target = MLP(config.state_dim, config.n_actions, config.hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.lr)

    @torch.no_grad()
    def q_values(self, state: np.ndarray) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.online(s).squeeze(0).cpu().numpy()

    def act(self, state: np.ndarray) -> int:
        return int(self.q_values(state).argmax())

    def select_action(self, state: np.ndarray, epsilon: float = 0.0,
                       rng: Optional[np.random.Generator] = None) -> int:
        rng = rng or np.random.default_rng()
        if rng.random() < epsilon:
            return int(rng.integers(0, self.cfg.n_actions))
        return self.act(state)

    def train_step(self, batch) -> float:
        states, actions, rewards, next_states, dones = batch
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # Double-DQN target: online net picks the action, target net values it.
            next_online = self.online(next_states)
            next_actions = next_online.argmax(dim=1)
            next_q = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.cfg.gamma * (1.0 - dones) * next_q

        pred = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(pred, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        with torch.no_grad():
            for op, tp in zip(self.online.parameters(), self.target.parameters()):
                tp.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * op)

        return float(loss.item())

    def save(self, path: str | Path) -> None:
        torch.save(
            {"online": self.online.state_dict(),
             "target": self.target.state_dict(),
             "config": self.cfg.__dict__},
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "DQNAgent":
        ckpt = torch.load(Path(path), map_location=device)
        cfg = DQNConfig(**{**ckpt["config"], "device": device})
        agent = cls(cfg)
        agent.online.load_state_dict(ckpt["online"])
        agent.target.load_state_dict(ckpt["target"])
        return agent
