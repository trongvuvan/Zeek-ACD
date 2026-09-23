"""Rollout-based evaluation: run a (trained or baseline) policy through the
Markov-game environment and report reward plus detection / false-positive
rates broken down by the true attacker class."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable, Optional

import numpy as np

from .env import ACDMarkovGameEnv
from .game import AttackerClass, DefenderAction

PolicyFn = Callable[[np.ndarray], int]


def rollout(
    env: ACDMarkovGameEnv, policy: PolicyFn, n_episodes: int = 10, seed: int = 0
) -> dict:
    per_class_total = defaultdict(int)
    per_class_acted = defaultdict(int)  # action != ALLOW
    per_class_reward = defaultdict(float)
    per_class_actions = defaultdict(Counter)
    total_reward = 0.0
    total_steps = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            cls = info["attacker_class"]
            per_class_total[cls] += 1
            per_class_reward[cls] += reward
            per_class_actions[cls][int(action)] += 1
            if action != int(DefenderAction.ALLOW):
                per_class_acted[cls] += 1
            total_reward += reward
            total_steps += 1

    result = {
        "avg_reward": total_reward / max(total_steps, 1),
        "total_steps": total_steps,
        "per_class": {},
    }
    for cls in AttackerClass:
        n = per_class_total.get(int(cls), 0)
        if n == 0:
            continue
        result["per_class"][cls.name] = {
            "n": n,
            "action_rate": per_class_acted.get(int(cls), 0) / n,
            "avg_reward": per_class_reward.get(int(cls), 0.0) / n,
            # which actions the policy actually chose, most frequent first
            "actions": {
                DefenderAction(a).name: round(k / n, 3)
                for a, k in per_class_actions[int(cls)].most_common()
            },
        }
    return result


def make_agent_policy(agent, greedy: bool = True) -> PolicyFn:
    def policy(obs: np.ndarray) -> int:
        pi, _ = agent.defender_strategy(obs)
        if greedy:
            return int(np.argmax(pi))
        return int(np.random.default_rng().choice(len(pi), p=pi))
    return policy


def make_constant_policy(action: DefenderAction) -> PolicyFn:
    def policy(_obs: np.ndarray) -> int:
        return int(action)
    return policy


def summarize(name: str, result: dict) -> str:
    lines = [f"== {name} == avg_reward={result['avg_reward']:.3f} steps={result['total_steps']}"]
    for cls_name, stats in result["per_class"].items():
        lines.append(
            f"  {cls_name:16s} n={stats['n']:5d}  action_rate={stats['action_rate']:.2f}  "
            f"avg_reward={stats['avg_reward']:+.3f}  {stats.get('actions', {})}"
        )
    return "\n".join(lines)
