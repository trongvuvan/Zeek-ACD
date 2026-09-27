#!/usr/bin/env bash
# Freeze the v8.1 6-seed ensemble under a given aggregation and train an
# attacker against it for 1000 episodes, to measure how exploitable that
# aggregation is. Usage: tools/attack_ensemble.sh <agg> <out-dir>
#   agg: mean | vote | vote2 | vote3 | smax
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
# "bigmix": entire IoT-23 + MTA-2025 + live2025 + clean benign, no per-file cap,
# repeat 1 -- the 174k-record mix the vote/smax exploitability runs used, so
# these numbers slot into the same table.
PYTHONPATH=src .venv/bin/python -u -m zeek_acd.train_selfplay \
  --data 'data/iot23/*/conn.log.labeled' \
  --data 'data/mta/2025-*/conn.log.labeled' \
  --data 'data/live2025/conn.log.labeled' \
  --data 'data/benign_web/train/conn.log.labeled' \
  --payoff payoffs/low_fp.json \
  --defender frozen $D --defender-agg "$AGG" \
  --episodes 1000 --eval-every 50 --seed 0 \
  --checkpoint-dir "$OUT" 2>&1 | tee "$OUT/run.out"
