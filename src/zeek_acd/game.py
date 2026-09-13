"""The zero-sum stochastic game definition: action spaces for both players,
the mapping from dataset ground-truth labels to the attacker's latent
action (used only for computing training rewards, never shown to the
defender's network as an input feature), and the payoff table.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class DefenderAction(IntEnum):
    ALLOW = 0
    LOG_ALERT = 1
    RATE_LIMIT = 2
    BLOCK_SRC = 3
    ISOLATE_HOST = 4
    DECEIVE = 5


class AttackerClass(IntEnum):
    BENIGN = 0
    RECON = 1          # port/vulnerability scanning
    DOS = 2             # (D)DoS / flooding
    C2 = 3              # command-and-control / botnet beaconing
    OTHER_MALICIOUS = 4  # exfil, payload download, generic malware traffic


N_DEFENDER_ACTIONS = len(DefenderAction)
N_ATTACKER_CLASSES = len(AttackerClass)

# Actions that are assumed to suppress further connections from the same
# source within the current episode, simulating real mitigation effect and
# giving the environment genuine action-dependent dynamics.
SUPPRESSING_ACTIONS = {DefenderAction.BLOCK_SRC, DefenderAction.ISOLATE_HOST}
RATE_LIMIT_SUPPRESS_PROB = 0.5  # RATE_LIMIT only partially suppresses traffic


def classify_label(label: str | None, detailed_label: str | None) -> AttackerClass:
    """Maps IoT-23-style ``label`` / ``detailed-label`` ground truth to the
    attacker's latent action class. Unrecognized detailed labels fall back
    to OTHER_MALICIOUS whenever the coarse label says malicious."""
    label = (label or "").strip().lower()
    detailed = (detailed_label or "").strip().lower()

    if "benign" in label or label in ("", "-"):
        return AttackerClass.BENIGN

    if "portscan" in detailed or "scan" in detailed:
        return AttackerClass.RECON
    if "ddos" in detailed or "dos" in detailed:
        return AttackerClass.DOS
    if "c&c" in detailed or "cc" == detailed or "torii" in detailed or "mirai" in detailed:
        return AttackerClass.C2
    return AttackerClass.OTHER_MALICIOUS


@dataclass
class PayoffTable:
    """Defender reward for every (defender_action, attacker_class) pair.
    Zero-sum: the attacker's reward is the negation of this value.
    Rows = DefenderAction, columns = AttackerClass.
    """

    table: list[list[float]]

    @classmethod
    def default(cls) -> "PayoffTable":
        # fmt: off
        #                         BENIGN  RECON   DOS     C2      OTHER
        table = [
            [ 0.00,               -1.00,  -1.00,  -1.00,  -1.00],  # ALLOW
            [-0.05,                0.30,   0.10,   0.20,   0.20],  # LOG_ALERT
            [-0.30,                0.50,   0.60,   0.10,   0.20],  # RATE_LIMIT
            [-0.70,                0.70,   0.90,   0.60,   0.70],  # BLOCK_SRC
            [-1.00,                0.60,   0.50,   1.00,   0.80],  # ISOLATE_HOST
            [-0.40,                0.80,   0.10,   0.40,   0.40],  # DECEIVE
        ]
        # fmt: on
        return cls(table=table)

    def reward(self, defender_action: int, attacker_class: int) -> float:
        return self.table[int(defender_action)][int(attacker_class)]

    def as_matrix(self):
        import numpy as np

        return np.asarray(self.table, dtype=np.float64)

    @classmethod
    def from_dict(cls, d: dict) -> "PayoffTable":
        return cls(table=[[float(x) for x in row] for row in d["table"]])

    def to_dict(self) -> dict:
        return {"table": self.table}
