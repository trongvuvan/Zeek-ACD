"""Read back what a live run actually decided.

``live/run_agent.py`` writes one JSON line per connection to its audit log:
uid, source, destination, the chosen action and the policy's score for
every action. This turns that file into something you can read:

* what the policy did overall, and to which destinations;
* with ``--labels``, the breakdown per true class -- i.e. the detection
  rate and the false-positive rate of the run that actually happened,
  rather than of an offline replay;
* with ``--show``, the individual decisions, so a flagged connection can
  be traced back to a uid in Zeek's own logs.

Only connections present in both files are counted when labels are given;
the count of audit entries with no label is printed, since a live log
keeps growing while the agent runs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .data import load_conn_log_labeled
from .game import AttackerClass, DefenderAction, classify_label


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audit-log", required=True, help="the JSONL written by run_agent")
    p.add_argument("--labels", default=None,
                    help="glob for conn.log.labeled files covering the same traffic "
                         "(see pcap_dataset.py --zeek-log-dir); enables the per-class "
                         "detection / false-positive breakdown")
    p.add_argument("--show", type=int, default=0,
                    help="also print this many individual decisions")
    p.add_argument("--action", default=None,
                    help="restrict --show to one action, e.g. DECEIVE")
    p.add_argument("--only-wrong", action="store_true",
                    help="restrict --show to decisions that disagree with the label "
                         "(acting on benign traffic, or allowing an attack)")
    p.add_argument("--top", type=int, default=10,
                    help="how many destinations to list in the summary")
    return p.parse_args()


def load_audit(path: Path) -> list[dict]:
    entries = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def main() -> None:
    args = parse_args()
    entries = load_audit(Path(args.audit_log))
    if not entries:
        raise SystemExit(f"no decisions in {args.audit_log}")

    actions = Counter(e.get("action") for e in entries)
    print(f"[audit] {len(entries)} decisions from {args.audit_log}")
    print("[audit] actions: " + "  ".join(
        f"{a}={n} ({n / len(entries):.1%})" for a, n in actions.most_common()))

    acted = [e for e in entries if e.get("action") != DefenderAction.ALLOW.name]
    by_dst = Counter(e.get("resp_h") for e in acted)
    if by_dst:
        print(f"\n[audit] top {args.top} destinations the policy acted on:")
        for dst, n in by_dst.most_common(args.top):
            mix = Counter(e["action"] for e in acted if e.get("resp_h") == dst)
            print(f"    {str(dst):20s} {n:5d}  {dict(mix)}")

    labels: dict[str, str] = {}
    if args.labels:
        labels = {
            r["uid"]: AttackerClass(
                classify_label(r.get("label"), r.get("detailed-label"))).name
            for r in load_conn_log_labeled(args.labels) if r.get("uid")
        }
        per_class: dict[str, Counter] = defaultdict(Counter)
        unlabeled = 0
        for e in entries:
            cls = labels.get(e.get("uid"))
            if cls is None:
                unlabeled += 1
                continue
            per_class[cls][e.get("action")] += 1

        print(f"\n[audit] per true class ({len(entries) - unlabeled} matched, "
              f"{unlabeled} not in the label file):")
        for cls in AttackerClass:
            counts = per_class.get(cls.name)
            if not counts:
                continue
            n = sum(counts.values())
            acted_n = sum(v for k, v in counts.items()
                          if k != DefenderAction.ALLOW.name)
            rate = "false-positive" if cls == AttackerClass.BENIGN else "detection"
            mix = {k: round(v / n, 3) for k, v in counts.most_common()}
            print(f"    {cls.name:16s} n={n:5d}  {rate} rate={acted_n / n:.3f}  {mix}")

    if args.show:
        print(f"\n[audit] {args.show} individual decisions:")
        shown = 0
        for e in entries:
            if args.action and e.get("action") != args.action:
                continue
            cls = labels.get(e.get("uid"))
            if args.only_wrong:
                if cls is None:
                    continue
                wrong = ((cls == AttackerClass.BENIGN.name
                          and e.get("action") != DefenderAction.ALLOW.name)
                         or (cls != AttackerClass.BENIGN.name
                             and e.get("action") == DefenderAction.ALLOW.name))
                if not wrong:
                    continue
            scores = e.get("strategy") or []
            best = ", ".join(
                f"{DefenderAction(i).name}={s:+.2f}"
                for i, s in sorted(enumerate(scores), key=lambda kv: -kv[1])[:3]
            )
            label = f" label={cls}" if cls else ""
            print(f"    {e.get('uid')}  {e.get('orig_h')} -> {e.get('resp_h')}  "
                  f"{e.get('action')}{label}   [{best}]")
            shown += 1
            if shown >= args.show:
                break


if __name__ == "__main__":
    main()
