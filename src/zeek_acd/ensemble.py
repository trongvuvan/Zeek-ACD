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


# How aggressive each defender action is (how much it suppresses the
# source). Used by the ``vote`` and ``smax`` aggregations to break ties and
# to pick "the most decisive action any member wanted". This is the
# DefenderAction order from game.py by suppression strength -- keeping it
# here (rather than importing) keeps ensemble.py free of the game module.
#   ALLOW < LOG_ALERT < DECEIVE < RATE_LIMIT < BLOCK_SRC < ISOLATE_HOST
#     0        1           5          2            3            4
_SEVERITY = {0: 0, 1: 1, 5: 2, 2: 3, 3: 4, 4: 5}

# The two actions that suppress the source for the rest of the episode
# (game.SUPPRESSING_ACTIONS: BLOCK_SRC=3, ISOLATE_HOST=4). The vote-quorum
# rule keys off how many members would play one of these.
_SUPPRESSING = (3, 4)

AGGREGATIONS = ("mean", "vote", "smax")  # plus "vote<k>", validated in _parse_agg


def _parse_agg(agg: str) -> tuple[str, int]:
    """``agg`` is ``mean``, ``smax``, ``vote`` (plain majority) or ``vote<k>``
    (block if at least k members' argmax is a suppressing action, else fall
    back to majority). Returns (base, quorum); quorum is 0 unless vote<k>."""
    if agg in ("mean", "smax", "vote"):
        return agg, 0
    if agg.startswith("vote") and agg[4:].isdigit() and int(agg[4:]) >= 1:
        return "vote", int(agg[4:])
    raise SystemExit(f"--agg must be mean, vote, vote<k> (e.g. vote2), or smax; got {agg!r}")


def load_ensemble(checkpoints: list[str], normalizers: list[str], agg: str = "mean"):
    """Returns a ``raw_vec -> (action, mean_q, max_mean_q)`` callable, the
    same shape as ``live.run_agent.load_policy`` returns.

    ``agg`` controls how the members' Q-vectors become one action:

    - ``mean``  argmax of the averaged Q. The cheap fix for false-positive
                *variance on fixed real data*, but it softens the decisive
                block: if only some members would BLOCK/ISOLATE evasive
                traffic, averaging pulls that action's score back below
                ALLOW and the attacker is never suppressed (see the README
                "ensemble is more exploitable" entry).
    - ``vote``  each member takes its own argmax; the ensemble plays the
                action with the most votes, ties broken toward the more
                suppressing one. Restores a decisive block when a majority
                of members want it, while a lone member's false positive on
                benign is outvoted.
    - ``smax``  the most suppressing action *any* member's argmax chose.
                Hardest against an adaptive attacker (one member wanting to
                block is enough) at the cost of inheriting every member's
                false positives.

    The reported scores/value are always the mean Q, for a comparable audit
    log; only the chosen action changes with ``agg``.
    """
    if len(checkpoints) != len(normalizers):
        raise SystemExit("need one --normalizer per --checkpoint")
    base, quorum = _parse_agg(agg)
    members = []
    for ckpt, norm in zip(checkpoints, normalizers):
        with open(norm) as f:
            members.append((DQNAgent.load(ckpt), RunningNormalizer.from_dict(json.load(f))))

    def policy(raw):
        qs = [agent.q_values(nz.normalize(raw)) for agent, nz in members]
        mean_q = np.mean(qs, axis=0)
        if base == "mean":
            action = int(mean_q.argmax())
        else:
            votes = [int(q.argmax()) for q in qs]
            if base == "smax":
                action = max(votes, key=lambda a: _SEVERITY[a])
            elif quorum and sum(1 for v in votes if v in _SUPPRESSING) >= quorum:
                # Enough members want to suppress: play the most severe
                # suppressing action they picked, ignoring the softer votes.
                action = max((v for v in votes if v in _SUPPRESSING), key=lambda a: _SEVERITY[a])
            else:  # plain majority: most common action, ties -> more suppressing
                counts: dict[int, int] = {}
                for a in votes:
                    counts[a] = counts.get(a, 0) + 1
                action = max(votes, key=lambda a: (counts[a], _SEVERITY[a]))
        return action, [round(float(x), 4) for x in mean_q], float(mean_q.max())

    return policy
