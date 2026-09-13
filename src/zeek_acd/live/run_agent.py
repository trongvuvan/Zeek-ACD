"""CLI: run a trained defender policy against a live Zeek ``conn.log``.

Safe by default: the default executor is ``log_only`` and only ever writes
to the audit log. Real enforcement (``--executor nftables``) additionally
requires ``--live-enforce`` -- an explicit, separate flag -- or it still
runs in dry-run mode and only logs what it would have done.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from ..features import FeatureExtractor, RunningNormalizer
from ..game import DefenderAction
from ..minimax_dqn import MinimaxDQNAgent
from .action_executor import Decision, build_executor
from .zeek_tail import LiveZeekTail


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint from train.py")
    p.add_argument("--normalizer", required=True, help="path to the matching normalizer_*.json")
    p.add_argument("--log-path", required=True, help="path to the live conn.log to tail")
    p.add_argument("--format", choices=["auto", "tsv", "json"], default="auto")
    p.add_argument("--from-start", action="store_true",
                    help="also process lines already in the file, not just new ones")
    p.add_argument("--audit-log", default="acd_audit.jsonl")
    p.add_argument("--executor", choices=["log_only", "nftables"], default="log_only")
    p.add_argument("--live-enforce", action="store_true",
                    help="REQUIRED in addition to --executor nftables to actually run nft; "
                         "without it, nftables actions are logged but not executed")
    p.add_argument("--nft-table", default="inet acd")
    p.add_argument("--nft-blocked-set", default="blocked")
    p.add_argument("--nft-ratelimited-set", default="ratelimited")
    p.add_argument("--deterministic", action="store_true", default=True,
                    help="pick the argmax action instead of sampling the mixed strategy (default)")
    return p.parse_args()


def main() -> None:
    torch.set_num_threads(1)  # tiny network; avoids CPU thread-pool overhead per inference call
    args = parse_args()

    with open(args.normalizer) as f:
        normalizer = RunningNormalizer.from_dict(json.load(f))
    fx = FeatureExtractor(normalizer)

    agent = MinimaxDQNAgent.load(args.checkpoint)

    dry_run = not (args.executor == "nftables" and args.live_enforce)
    if args.executor == "nftables" and not args.live_enforce:
        print("[run_agent] --executor nftables given without --live-enforce: "
              "running in DRY RUN, nothing will be blocked.")
    executor = build_executor(
        args.executor, args.audit_log, dry_run=dry_run,
        table=args.nft_table, blocked_set=args.nft_blocked_set,
        ratelimited_set=args.nft_ratelimited_set,
    )

    tail = LiveZeekTail(args.log_path, format=args.format, extra_fields=["label", "detailed-label"])
    stream = tail.from_start() if args.from_start else iter(tail)

    print(f"[run_agent] watching {args.log_path} with executor={args.executor} dry_run={dry_run}")
    for record in stream:
        vec = fx.transform(record, training=False)
        pi, value = agent.defender_strategy(vec)
        action_idx = int(pi.argmax())
        action = DefenderAction(action_idx)

        decision = Decision(
            ts=time.time(),
            uid=record.get("uid"),
            orig_h=record.get("id.orig_h"),
            resp_h=record.get("id.resp_h"),
            action=action,
            strategy=[round(float(x), 4) for x in pi],
        )
        executor.execute(decision)

        if action != DefenderAction.ALLOW:
            print(f"[run_agent] {decision.uid} {decision.orig_h} -> {action.name} "
                  f"(game_value={value:+.3f}, pi={decision.strategy})")


if __name__ == "__main__":
    main()
