#!/usr/bin/env bash
# Score checkpoint dirs on every TEST set (never used for checkpoint selection).
# Prints action_rate per class: FP on BENIGN, detection on the attack classes.
# Usage: tools/test_battery.sh dqn_v6 dqn_v7a ...
# A '+'-joined spec is scored as an ensemble: dqn_v6+dqn_v6_s1+dqn_v6_s2.
# 'dir@ep' picks a specific checkpoint instead of agent_best: dqn_v6@ep550.
# AGG=vote|smax tools/test_battery.sh ...  sets the ensemble aggregation
# (default mean); ignored for a single-checkpoint spec.
cd "$(dirname "$0")/.."
TESTS=(
  "ho-all=data/benign_heldout/conn.log.labeled"
  "ho-web=data/benign_heldout_web/conn.log.labeled"
  "live2026=data/live2026/conn.log.labeled"
  "mta2026=data/mta/2026-*/conn.log.labeled"
  "live0926=data/live0926/conn.log.labeled"
  "brokentest=data/selfbroken_test/conn.log.labeled"
  "iot3-1=data/iot23/CTU-IoT-Malware-Capture-3-1/conn.log.labeled"
  "iot34-1=data/iot23/CTU-IoT-Malware-Capture-34-1/conn.log.labeled"
)
for ck in "$@"; do
  args=()
  IFS=+ read -ra members <<< "$ck"
  for m in "${members[@]}"; do
    d=${m%@*}; tag=best; [[ $m == *@* ]] && tag=${m#*@}
    args+=(--checkpoint "checkpoints/$d/agent_$tag.pt" --normalizer "checkpoints/$d/normalizer_$tag.json")
  done
  for t in "${TESTS[@]}"; do
    name=${t%%=*}; glob=${t#*=}
    PYTHONPATH=src .venv/bin/python -m zeek_acd.eval_checkpoint \
      "${args[@]}" ${AGG:+--agg "$AGG"} \
      --payoff payoffs/low_fp.json --max-records-per-file 15000 --data "$glob" 2>&1 |
    awk -v ck="$ck" -v t="$name" '/ n=/{match($0,/action_rate=[0-9.]+/);printf "%-8s %-11s %-16s %s\n", ck, t, $1, substr($0,RSTART+12,RLENGTH-12)}'
  done
done | .venv/bin/python -c '
import sys, collections
d = collections.OrderedDict(); cks = []
for line in sys.stdin:
    ck, t, c, r = line.split()
    d.setdefault((t, c), {})[ck] = r
    cks += [ck] if ck not in cks else []
print(f"{"test":11} {"class":16} " + " ".join(f"{c:>9}" for c in cks))
for (t, c), v in d.items():
    print(f"{t:11} {c:16} " + " ".join(f"{v.get(k, "-"):>9}" for k in cks))
'
