"""Uniform experience replay for joint (defender, attacker) transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    state: np.ndarray
    defender_action: np.ndarray
    attacker_class: np.ndarray
    reward: np.ndarray
    next_state: np.ndarray
    done: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, seed: int | None = None):
        self.capacity = capacity
        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)
        self._size = 0
        self._idx = 0

        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.defender_action = np.zeros(capacity, dtype=np.int64)
        self.attacker_class = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        state: np.ndarray,
        defender_action: int,
        attacker_class: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        i = self._idx
        self.state[i] = state
        self.defender_action[i] = defender_action
        self.attacker_class[i] = attacker_class
        self.reward[i] = reward
        self.next_state[i] = next_state
        self.done[i] = float(done)

        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        idx = self.rng.integers(0, self._size, size=batch_size)
        return Batch(
            state=self.state[idx],
            defender_action=self.defender_action[idx],
            attacker_class=self.attacker_class[idx],
            reward=self.reward[idx],
            next_state=self.next_state[idx],
            done=self.done[idx],
        )
