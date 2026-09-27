#!/usr/bin/env bash
# Train the MITRE ATT&CK-catalog attacker (--attacker attack) against the
# frozen v8.1 6-seed ensemble under a given aggregation, on the 174k bigmix,
# to see which ATT&CK techniques the defender lets through.
# Usage: tools/attack_ensemble_attack.sh <agg> <out-dir>
set -euo pipefail
cd "$(dirname "$0")/.."
AGG=${1:?agg}
OUT=${2:?out-dir}
D=""
for s in 0 1 2 3 4 5; do
  D="$D --defender-checkpoint checkpoints/dqn_g0_s$s/agent_best.pt \
       --defender-normalizer checkpoints/dqn_g0_s$s/normalizer_best.json"
done
mkdir -p "$OUT"
PYTHONPATH=src .venv/bin/python -u -m zeek_acd.train_selfplay \
  --attacker attack \
  --data 'data/iot23/*/conn.log.labeled' \
  --data 'data/mta/2025-*/conn.log.labeled' \
  --data 'data/live2025/conn.log.labeled' \
  --data 'data/benign_web/train/conn.log.labeled' \
  --payoff payoffs/low_fp.json \
  --defender frozen $D --defender-agg "$AGG" \
  --episodes 1000 --eval-every 50 --seed 0 \
  --checkpoint-dir "$OUT" 2>&1 | tee "$OUT/run.out"
