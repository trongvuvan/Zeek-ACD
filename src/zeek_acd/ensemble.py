"""Average the Q-values of several plain-DQN checkpoints.

Single training runs are unstable in exactly one way: detection stays at
1.00 but the benign false-positive rate swings between checkpoints and
seeds (0.04 vs 0.37 vs 0.77 on live2026 for three seeds of the same
recipe). Averaging Q over members is the standard cheap fix for that kind
of variance.

Each checkpoint was trained with its own running normalizer, so every
member normalizes the shared raw feature vector itself before its Q pass.
"""

from __future__ import annotations

import json

import numpy as np

from .dqn import DQNAgent
from .features import FEATURE_DIM, RunningNormalizer, raw_feature_vector


class RawExtractor:
    """Stands in for ``FeatureExtractor`` where the policy does its own
    normalization: record -> un-normalized feature vector."""

    dim = FEATURE_DIM

    def transform(self, record: dict[str, str], training: bool = False) -> np.ndarray:
        return raw_feature_vector(record)


def load_ensemble(checkpoints: list[str], normalizers: list[str]):
    """Returns an ``raw_vec -> (action, mean_q, max_mean_q)`` callable, the
    same shape as ``live.run_agent.load_policy`` returns."""
    if len(checkpoints) != len(normalizers):
        raise SystemExit("need one --normalizer per --checkpoint")
    members = []
    for ckpt, norm in zip(checkpoints, normalizers):
        with open(norm) as f:
            members.append((DQNAgent.load(ckpt), RunningNormalizer.from_dict(json.load(f))))

    def policy(raw):
        q = np.mean([agent.q_values(nz.normalize(raw)) for agent, nz in members], axis=0)
        return int(q.argmax()), [round(float(x), 4) for x in q], float(q.max())

    return policy
