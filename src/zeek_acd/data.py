"""Loading Zeek connection records for training/evaluation from disk."""

from __future__ import annotations

import glob
from itertools import islice
from pathlib import Path

import numpy as np

from .context import annotate
from .zeeklog import open_log

IOT23_EXTRA_FIELDS = ["label", "detailed-label"]


def load_conn_log_labeled(
    pattern: str, max_records_per_file: int | None = None
) -> list[dict[str, str]]:
    """Loads one or more IoT-23-style ``conn.log.labeled`` files matching a
    glob pattern into a flat list of records. ``max_records_per_file`` keeps
    only the first N rows of each file, so one huge scenario (e.g. a
    port-scan capture) can't dominate episode sampling."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matched pattern: {pattern}")
    records: list[dict[str, str]] = []
    for path in paths:
        log = open_log(path, extra_fields=IOT23_EXTRA_FIELDS)
        # One context pass per file: each file is its own capture, so
        # rolling windows must not run across a scenario boundary.
        records.extend(annotate(list(islice(log, max_records_per_file))))
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


def load_and_split_per_file(
    pattern: str,
    max_records_per_file: int | None = None,
    eval_fraction: float = 0.15,
    seed: int = 0,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Like ``train_eval_split(load_conn_log_labeled(...))``, but carves the
    eval block out of *each* file, so evaluation covers every scenario (and
    therefore every attacker class) instead of whichever file the single
    contiguous block happened to land in."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files matched pattern: {pattern}")
    train_records: list[dict[str, str]] = []
    eval_records: list[dict[str, str]] = []
    for i, path in enumerate(paths):
        log = open_log(path, extra_fields=IOT23_EXTRA_FIELDS)
        records = annotate(list(islice(log, max_records_per_file)))
        if len(records) < 2:
            train_records.extend(records)
            continue
        tr, ev = train_eval_split(records, eval_fraction=eval_fraction, seed=seed + i)
        train_records.extend(tr)
        eval_records.extend(ev)
    if not train_records or not eval_records:
        raise ValueError(f"not enough rows parsed from: {paths}")
    return train_records, eval_records


def load_sources(
    patterns: list[str],
    repeats: list[int] | None = None,
    max_records_per_file: int | None = None,
    eval_fraction: float = 0.15,
    seed: int = 0,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Loads several datasets at once and lets a small one pull its weight.

    Mixing captures of very different sizes (45k rows of IoT-23 against 400
    rows of labeled malware traffic) otherwise means the small one is never
    sampled into an episode. ``repeats[i]`` repeats source ``i``'s *training*
    records that many times; eval records are never repeated, so the eval
    numbers stay honest.
    """
    repeats = list(repeats or [])
    repeats += [1] * (len(patterns) - len(repeats))

    train_records: list[dict[str, str]] = []
    eval_records: list[dict[str, str]] = []
    for i, (pattern, repeat) in enumerate(zip(patterns, repeats)):
        tr, ev = load_and_split_per_file(
            pattern, max_records_per_file, eval_fraction=eval_fraction, seed=seed + 100 * i
        )
        train_records.extend(tr * max(1, int(repeat)))
        eval_records.extend(ev)
    return train_records, eval_records
