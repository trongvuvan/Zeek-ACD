"""Score every checkpoint of a run against several held-out datasets at once.

Training-time eval oscillates: across one 600-episode run the false-positive
rate swung between 1% and 96% with almost no change in average reward, so
"the last checkpoint" is not automatically the one to deploy. This lays the
whole run out side by side on the datasets that matter:

    python -m zeek_acd.compare_checkpoints \\
        --checkpoint-dir checkpoints/dqn_v4 \\
        --data 'MTA-2026=data/mta/2026-*/conn.log.labeled' \\
        --data 'live=data/live2026/conn.log.labeled'

For every checkpoint and dataset it prints average reward, the false-positive
rate (BENIGN acted on) and the detection rate (everything else acted on).

Selecting on these numbers is selecting on the test sets, which flatters the
result. When there is enough data, keep one set for choosing and report the
others; the ``--select`` flag names the one to rank by.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from .data import load_conn_log_labeled
from .eval_checkpoint import flat_eval
from .features import FeatureExtractor, RunningNormalizer
from .game import PayoffTable
from .live.run_agent import load_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--prefix", default="agent",
                    help="checkpoint filename prefix: 'agent' for train_dqn.py runs, "
                         "'defender' for train_selfplay.py runs")
    p.add_argument("--data", action="append", required=True,
                    help="'NAME=glob' per dataset; repeat the flag")
    p.add_argument("--payoff", default=None,
                    help="JSON PayoffTable to score with; the default table is usually "
                         "the right choice even for a model trained with another one, "
                         "so that runs stay comparable")
    p.add_argument("--select", default=None,
                    help="dataset name to rank by (default: the sum over all of them)")
    return p.parse_args()


def episode_of(path: Path, prefix: str) -> int:
    m = re.search(rf"{re.escape(prefix)}_ep(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else -1


def main() -> None:
    torch.set_num_threads(1)
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir)

    datasets = {}
    for spec in args.data:
        if "=" not in spec:
            raise SystemExit(f"--data needs NAME=glob, got: {spec}")
        name, glob = spec.split("=", 1)
        datasets[name] = load_conn_log_labeled(glob)

    if args.select is not None and args.select not in datasets:
        raise SystemExit(f"--select {args.select!r} is not one of the --data names: "
                         + ", ".join(datasets))

    payoff = (PayoffTable.from_dict(json.load(open(args.payoff))) if args.payoff
              else PayoffTable.default())

    checkpoints = sorted(
        (p for p in ckpt_dir.glob(f"{args.prefix}_ep*.pt") if episode_of(p, args.prefix) > 0),
        key=lambda p: episode_of(p, args.prefix),
    )
    if not checkpoints:
        raise SystemExit(f"no {args.prefix}_ep*.pt checkpoints in {ckpt_dir}")

    for name, records in datasets.items():
        print(f"[compare] {name}: {len(records)} records")
    print()
    print(f"{'ckpt':8s} " + " | ".join(f"{n:^24s}" for n in datasets))
    print(f"{'':8s} " + " | ".join(f"{'avg':>7s} {'FP':>5s} {'det':>5s}{'':6s}" for _ in datasets))

    ranked = []
    for ckpt in checkpoints:
        ep = episode_of(ckpt, args.prefix)
        normalizer_path = ckpt_dir / f"normalizer_ep{ep}.json"
        if not normalizer_path.exists():
            normalizer_path = ckpt_dir / "normalizer_latest.json"
        fx = FeatureExtractor(RunningNormalizer.from_dict(json.load(open(normalizer_path))))
        policy, _kind = load_policy(str(ckpt))

        cells, score = [], 0.0
        for name, records in datasets.items():
            r = flat_eval(records, fx, policy, payoff)
            per_class = r["per_class"]
            fp = per_class["BENIGN"]["action_rate"] if "BENIGN" in per_class else float("nan")
            malicious = [c for c in per_class if c != "BENIGN"]
            seen = sum(per_class[c]["n"] for c in malicious)
            det = (sum(per_class[c]["action_rate"] * per_class[c]["n"] for c in malicious) / seen
                   if seen else float("nan"))
            cells.append(f"{r['avg_reward']:+7.3f} {fp:5.2f} {det:5.2f}{'':6s}")
            if args.select is None or args.select == name:
                score += r["avg_reward"]
        print(f"ep{ep:<6d} " + " | ".join(cells))
        ranked.append((score, ep))

    best = max(ranked)[1]
    by = args.select or "the sum over all datasets"
    print(f"\n[compare] best by {by}: ep{best}")
    print(f"[compare] to deploy it:  cp {ckpt_dir}/{args.prefix}_ep{best}.pt "
          f"{ckpt_dir}/{args.prefix}_best.pt && "
          f"cp {ckpt_dir}/normalizer_ep{best}.json {ckpt_dir}/normalizer_best.json")


if __name__ == "__main__":
    main()
