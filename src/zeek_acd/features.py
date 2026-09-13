"""Turns a raw Zeek ``conn.log`` record into a fixed-size numeric feature
vector, plus an online normalizer so the same code path serves offline
training and live inference without ever refitting on the live stream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

UNSET = "-"

PROTOS = ["tcp", "udp", "icmp"]
CONN_STATES = [
    "S0", "S1", "SF", "REJ", "S2", "S3",
    "RSTO", "RSTR", "RSTOS0", "RSTRH", "SH", "SHR", "OTH",
]
SERVICES = ["-", "http", "dns", "ssl", "irc", "dhcp", "ntp"]
HISTORY_FLAGS = "ShAaDdFfRrCcGgTtIiWwXx"

CONTINUOUS_FEATURES = [
    "duration",
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "resp_pkts",
    "orig_ip_bytes",
    "resp_ip_bytes",
    "missed_bytes",
    "log_orig_bytes",
    "log_resp_bytes",
    "bytes_ratio",
    "history_len",
]

FEATURE_DIM = (
    len(CONTINUOUS_FEATURES)
    + len(PROTOS) + 1          # +1 for "other"
    + len(CONN_STATES) + 1
    + len(SERVICES) + 1
    + len(HISTORY_FLAGS)
)


def _f(value: str | None, default: float = 0.0) -> float:
    if value is None or value == UNSET or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _one_hot(value: str, categories: list[str]) -> list[float]:
    vec = [0.0] * (len(categories) + 1)
    try:
        vec[categories.index(value)] = 1.0
    except ValueError:
        vec[-1] = 1.0
    return vec


def raw_feature_vector(record: dict[str, str]) -> np.ndarray:
    """Deterministic, un-normalized feature vector from a Zeek conn record."""
    duration = _f(record.get("duration"))
    orig_bytes = _f(record.get("orig_bytes"))
    resp_bytes = _f(record.get("resp_bytes"))
    orig_pkts = _f(record.get("orig_pkts"))
    resp_pkts = _f(record.get("resp_pkts"))
    orig_ip_bytes = _f(record.get("orig_ip_bytes"))
    resp_ip_bytes = _f(record.get("resp_ip_bytes"))
    missed_bytes = _f(record.get("missed_bytes"))
    history = record.get("history") or ""
    if history == UNSET:
        history = ""

    continuous = [
        duration,
        orig_bytes,
        resp_bytes,
        orig_pkts,
        resp_pkts,
        orig_ip_bytes,
        resp_ip_bytes,
        missed_bytes,
        math.log2(orig_bytes + 1.0),
        math.log2(resp_bytes + 1.0),
        (orig_bytes + 1.0) / (resp_bytes + 1.0),
        float(len(history)),
    ]

    proto = _one_hot((record.get("proto") or "").lower(), PROTOS)
    conn_state = _one_hot(record.get("conn_state") or "", CONN_STATES)
    service = _one_hot(record.get("service") or UNSET, SERVICES)
    history_counts = [float(history.count(flag)) for flag in HISTORY_FLAGS]

    vec = continuous + proto + conn_state + service + history_counts
    return np.asarray(vec, dtype=np.float64)


@dataclass
class RunningNormalizer:
    """Welford's online mean/variance, applied only to the continuous
    feature block so one-hot indicators stay untouched."""

    dim: int = len(CONTINUOUS_FEATURES)
    count: float = 1e-4
    mean: np.ndarray = field(default=None)
    m2: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.mean is None:
            self.mean = np.zeros(self.dim, dtype=np.float64)
        if self.m2 is None:
            self.m2 = np.ones(self.dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.m2 / self.count, 1e-6))

    def normalize(self, vec: np.ndarray, update: bool = False) -> np.ndarray:
        cont = vec[: self.dim]
        rest = vec[self.dim :]
        if update:
            self.update(cont)
        normed = (cont - self.mean) / self.std()
        normed = np.clip(normed, -8.0, 8.0)
        return np.concatenate([normed, rest]).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "dim": self.dim,
            "count": self.count,
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunningNormalizer":
        return cls(
            dim=d["dim"],
            count=d["count"],
            mean=np.asarray(d["mean"], dtype=np.float64),
            m2=np.asarray(d["m2"], dtype=np.float64),
        )


class FeatureExtractor:
    """Stateful convenience wrapper: raw record -> normalized feature
    vector, updating the normalizer in-place when ``training=True``."""

    def __init__(self, normalizer: RunningNormalizer | None = None):
        self.normalizer = normalizer or RunningNormalizer()

    @property
    def dim(self) -> int:
        return FEATURE_DIM

    def transform(self, record: dict[str, str], training: bool = False) -> np.ndarray:
        raw = raw_feature_vector(record)
        return self.normalizer.normalize(raw, update=training)
