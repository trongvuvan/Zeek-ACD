"""Self-play environment: a true two-sided zero-sum Markov game where BOTH
the attacker and the defender are learning agents.

How this differs from ``env.py``. In ``ACDMarkovGameEnv`` the attacker's
"move" at each step is read from the dataset's ground-truth label -- the
defender learns against whatever mix of traffic the trace happens to
contain. Here the attacker *chooses* its move (which class of traffic to
emit, including BENIGN = "lay low"), and the environment then samples a
real Zeek connection record of that class to produce the features the
defender observes. So:

    attacker action  --->  chosen AttackerClass
    env samples a real record of that class  --->  Zeek feature vector
    defender observes features (NOT the class) and chooses a DefenderAction
    reward_defender = payoff(defender_action, attacker_class)
    reward_attacker = -reward_defender          (zero-sum)

The dataset stops being a fixed sequence and becomes a *generative model
of per-class traffic*; the attacker controls the ground truth, the
defender must still infer it from noisy features. This is what lets the
defender train against an adversary that actively learns to exploit its
current policy, rather than against a fixed empirical traffic mix.

Dynamics that make it Markov (not a one-shot matrix game):
- If the defender BLOCK_SRC/ISOLATE_HOST the attacker, the attacker is
  suppressed for the rest of the episode: whatever it emits is dropped
  before reaching the defender, and both sides get 0 for those steps.
  The attacker thus has a real incentive to avoid detection (mixing in
  benign traffic, backing off after being alerted) rather than attacking
  blindly.
- RATE_LIMIT suppresses the attacker probabilistically for a window.

The attacker's own observation is deliberately small and behavioral (it
does NOT get to see the defender's network or the sampled features): how
far into the episode it is, whether it is currently suppressed, the
defender's last action, and a short rolling summary of how often it has
recently been acted upon and how it has recently scored. That is enough
to learn evasive, reactive strategies.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

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

RATE_LIMIT_WINDOW = 20
ATTACKER_HISTORY_LEN = 10

# attacker observation layout:
#   [0]            episode progress in [0, 1]
#   [1]            currently suppressed (blocked or rate-limited) flag
#   [2 .. 2+ND)    defender's previous action, one-hot
#   [-2]           rolling fraction of recent steps the defender acted on it
#   [-1]           rolling mean of the attacker's own recent reward
ATTACKER_OBS_DIM = 1 + 1 + N_DEFENDER_ACTIONS + 1 + 1


class ClassIndexedRecords:
    """Buckets a record list by attacker class so the environment can draw
    a fresh, real connection of a chosen class on demand."""

    def __init__(self, records: list[dict[str, str]], seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.buckets: dict[int, list[dict[str, str]]] = {int(c): [] for c in AttackerClass}
        for r in records:
            cls = classify_label(r.get("label"), r.get("detailed-label"))
            self.buckets[int(cls)].append(r)
        self.available = [c for c, rs in self.buckets.items() if rs]
        if not self.available:
            raise ValueError("no records in any class")
        # If a class is missing from the data (common: some traces have no
        # DoS), fall back to the nearest available class when it's chosen.
        self._fallback = {c: self._nearest_available(c) for c in range(N_ATTACKER_CLASSES)}

    def _nearest_available(self, cls: int) -> int:
        if self.buckets[cls]:
            return cls
        # Prefer any malicious class over benign as a substitute, else benign.
        malicious = [c for c in self.available if c != int(AttackerClass.BENIGN)]
        if cls != int(AttackerClass.BENIGN) and malicious:
            return int(self.rng.choice(malicious))
        return int(self.available[0])

    def sample(self, cls: int) -> tuple[dict[str, str], int]:
        """Returns (record, realized_class). realized_class may differ from
        the requested class only if the requested one has no records."""
        realized = self._fallback[int(cls)]
        bucket = self.buckets[realized]
        rec = bucket[int(self.rng.integers(0, len(bucket)))]
        return rec, realized


class SelfPlayACDEnv:
    """Drives one episode of the two-sided game. Not a Gymnasium env
    because it has two agents with two different observation spaces; the
    training loop in ``train_selfplay.py`` calls into it directly."""

    def __init__(
        self,
        records: list[dict[str, str]],
        feature_extractor: FeatureExtractor,
        payoff: Optional[PayoffTable] = None,
        episode_length: int = 128,
        training: bool = True,
        seed: Optional[int] = None,
    ):
        self.pool = ClassIndexedRecords(records, seed=seed)
        self.fx = feature_extractor
        self.payoff = payoff or PayoffTable.default()
        self.episode_length = episode_length
        self.training = training
        self.rng = np.random.default_rng(seed)

        self.n_defender_actions = N_DEFENDER_ACTIONS
        self.n_attacker_actions = N_ATTACKER_CLASSES
        self.defender_obs_dim = self.fx.dim
        self.attacker_obs_dim = ATTACKER_OBS_DIM

        self._steps = 0
        self._blocked = False
        self._rate_limited_remaining = 0
        self._last_defender_action = int(DefenderAction.ALLOW)
        self._acted_history: deque[float] = deque(maxlen=ATTACKER_HISTORY_LEN)
        self._reward_history: deque[float] = deque(maxlen=ATTACKER_HISTORY_LEN)

    # -- attacker side ----------------------------------------------------
    def _attacker_obs(self) -> np.ndarray:
        obs = np.zeros(self.attacker_obs_dim, dtype=np.float32)
        obs[0] = self._steps / max(1, self.episode_length)
        obs[1] = 1.0 if (self._blocked or self._rate_limited_remaining > 0) else 0.0
        obs[2 + self._last_defender_action] = 1.0
        obs[-2] = float(np.mean(self._acted_history)) if self._acted_history else 0.0
        obs[-1] = float(np.mean(self._reward_history)) if self._reward_history else 0.0
        return obs

    def _is_suppressed(self) -> bool:
        if self._blocked:
            return True
        if self._rate_limited_remaining > 0:
            self._rate_limited_remaining -= 1
            return self.rng.random() < RATE_LIMIT_SUPPRESS_PROB
        return False

    def reset(self) -> np.ndarray:
        self._steps = 0
        self._blocked = False
        self._rate_limited_remaining = 0
        self._last_defender_action = int(DefenderAction.ALLOW)
        self._acted_history.clear()
        self._reward_history.clear()
        return self._attacker_obs()

    def attacker_step(self, attacker_action: int):
        """Phase 1: attacker emits its chosen class. Returns
        (defender_obs, realized_class, suppressed). If suppressed, the
        traffic never reaches the defender and defender_obs is None."""
        suppressed = self._is_suppressed()
        if suppressed:
            return None, None, True
        record, realized_class = self.pool.sample(attacker_action)
        defender_obs = self.fx.transform(record, training=self.training)
        return defender_obs, realized_class, False

    def defender_step(self, defender_action: int, realized_class: int):
        """Phase 2: defender responds to the (unsuppressed) traffic.
        Applies suppression dynamics and returns (defender_reward,
        attacker_reward, next_attacker_obs, done)."""
        d_reward = self.payoff.reward(defender_action, realized_class)
        a_reward = -d_reward

        act = DefenderAction(defender_action)
        if act in (DefenderAction.BLOCK_SRC, DefenderAction.ISOLATE_HOST):
            self._blocked = True
            self._rate_limited_remaining = 0
        elif act == DefenderAction.RATE_LIMIT:
            self._rate_limited_remaining = RATE_LIMIT_WINDOW

        self._last_defender_action = defender_action
        self._acted_history.append(0.0 if act == DefenderAction.ALLOW else 1.0)
        self._reward_history.append(a_reward)

        self._steps += 1
        done = self._steps >= self.episode_length
        return d_reward, a_reward, self._attacker_obs(), done

    def suppressed_step(self):
        """Phase 2 when the attacker's traffic was suppressed: no exchange,
        both get 0. The attacker still advances and observes the outcome."""
        self._acted_history.append(1.0)  # being suppressed counts as "acted upon"
        self._reward_history.append(0.0)
        self._steps += 1
        done = self._steps >= self.episode_length
        # defender took no decision this step; keep its last action as-is.
        return 0.0, 0.0, self._attacker_obs(), done
