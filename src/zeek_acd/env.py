"""A Gymnasium-compatible environment for the defender's side of the
zero-sum Markov game.

State transitions are driven by replaying a Zeek connection trace (offline
training) or a live tail (see ``live/``), so the *sequence* of connections
is exogenous. What makes this a genuine Markov game rather than a
per-flow supervised bandit is that ``SUPPRESSING_ACTIONS`` (BLOCK_SRC,
ISOLATE_HOST) remove all subsequent connections from the same source for
the rest of the episode, and RATE_LIMIT probabilistically drops a window
of follow-on connections -- i.e. the defender's action changes which
future states it will see, not just the immediate reward.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .features import FeatureExtractor
from .game import (
    N_ATTACKER_CLASSES,
    N_DEFENDER_ACTIONS,
    RATE_LIMIT_SUPPRESS_PROB,
    AttackerClass,
    DefenderAction,
    PayoffTable,
    classify_label,
)

RATE_LIMIT_WINDOW = 20  # subsequent connections from a rate-limited source


class ACDMarkovGameEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        records: list[dict[str, str]],
        feature_extractor: FeatureExtractor,
        payoff: Optional[PayoffTable] = None,
        episode_length: int = 256,
        training: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if not records:
            raise ValueError("records must be non-empty")
        self.records = records
        self.fx = feature_extractor
        self.payoff = payoff or PayoffTable.default()
        self.episode_length = episode_length
        self.training = training
        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.fx.dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_DEFENDER_ACTIONS)
        self.attacker_action_space = spaces.Discrete(N_ATTACKER_CLASSES)

        self._cursor = 0
        self._steps = 0
        self._permanent_block: set[str] = set()
        self._rate_limited: dict[str, int] = {}
        self._current_record: dict[str, str] | None = None

    def _source(self, record: dict[str, str]) -> str:
        return record.get("id.orig_h", "")

    def _is_suppressed(self, record: dict[str, str]) -> bool:
        src = self._source(record)
        if src in self._permanent_block:
            return True
        remaining = self._rate_limited.get(src)
        if remaining is not None and remaining > 0:
            if self.rng.random() < RATE_LIMIT_SUPPRESS_PROB:
                self._rate_limited[src] = remaining - 1
                return True
            self._rate_limited[src] = remaining - 1
        return False

    def _advance_to_next_valid(self) -> bool:
        """Moves ``_cursor`` forward past any suppressed records. Returns
        False if the trace runs out first."""
        n = len(self.records)
        scanned = 0
        while scanned < n:
            self._cursor += 1
            scanned += 1
            if self._cursor >= n:
                return False
            candidate = self.records[self._cursor]
            if not self._is_suppressed(candidate):
                return True
        return False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        max_start = max(1, len(self.records) - self.episode_length - 1)
        self._cursor = int(self.rng.integers(0, max_start))
        self._steps = 0
        self._permanent_block = set()
        self._rate_limited = {}
        self._current_record = self.records[self._cursor]
        obs = self.fx.transform(self._current_record, training=self.training)
        info = self._info()
        return obs, info

    def _info(self) -> dict[str, Any]:
        rec = self._current_record or {}
        attacker_class = classify_label(rec.get("label"), rec.get("detailed-label"))
        return {
            "attacker_class": int(attacker_class),
            "attacker_class_name": AttackerClass(attacker_class).name,
            "uid": rec.get("uid"),
            "orig_h": rec.get("id.orig_h"),
            "resp_h": rec.get("id.resp_h"),
        }

    def step(self, action: int):
        assert self._current_record is not None, "call reset() first"
        record = self._current_record
        attacker_class = classify_label(record.get("label"), record.get("detailed-label"))
        reward = self.payoff.reward(action, attacker_class)

        defender_action = DefenderAction(action)
        src = self._source(record)
        if src:
            if defender_action in (DefenderAction.BLOCK_SRC, DefenderAction.ISOLATE_HOST):
                self._permanent_block.add(src)
                self._rate_limited.pop(src, None)
            elif defender_action == DefenderAction.RATE_LIMIT:
                self._rate_limited[src] = RATE_LIMIT_WINDOW

        self._steps += 1
        info = self._info()
        info["defender_action"] = int(defender_action)
        info["defender_action_name"] = defender_action.name

        truncated = self._steps >= self.episode_length
        has_next = False
        if not truncated:
            has_next = self._advance_to_next_valid()
            truncated = truncated or not has_next

        terminated = False
        if not truncated and has_next:
            self._current_record = self.records[self._cursor]
            obs = self.fx.transform(self._current_record, training=self.training)
        else:
            obs = np.zeros(self.fx.dim, dtype=np.float32)

        return obs, float(reward), terminated, truncated, info
