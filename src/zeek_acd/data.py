"""Loading Zeek connection records for training/evaluation from disk."""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

from .zeeklog import open_log

IOT23_EXTRA_FIELDS = ["label", "detailed-label"]


def load_conn_log_labeled(pattern: str) -> list[dict[str, str]]:
    """Loads one or more IoT-23-style ``conn.log.labeled`` files matching a
    glob pattern into a flat list of records."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matched pattern: {pattern}")
    records: list[dict[str, str]] = []
    for path in paths:
        log = open_log(path, extra_fields=IOT23_EXTRA_FIELDS)
        records.extend(list(log))
    if not records:
        raise ValueError(f"no rows parsed from: {paths}")
    return records


def train_eval_split(
    records: list[dict[str, str]], eval_fraction: float = 0.15, seed: int = 0
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Splits into contiguous blocks (not a per-row shuffle) so evaluation
    sees a temporally distinct segment of traffic rather than leaking
    adjacent flows from the same session into both sets."""
    n = len(records)
    n_eval = max(1, int(n * eval_fraction))
    rng = np.random.default_rng(seed)
    n_blocks = max(1, n // max(n_eval, 1))
    block_starts = list(range(0, n, max(n // n_blocks, 1)))
    eval_start = block_starts[int(rng.integers(0, len(block_starts)))]
    eval_end = min(n, eval_start + n_eval)
    eval_records = records[eval_start:eval_end]
    train_records = records[:eval_start] + records[eval_end:]
    return train_records, eval_records
