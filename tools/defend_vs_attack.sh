#!/usr/bin/env bash
# Fine-tune a defender AGAINST the MITRE ATT&CK attacker, warm-started from a
# v8 seed and with real traffic interleaved into every batch (--real-frac) so
# it does not drift off the real distribution -- the lesson from the earlier
# joint-training failures. Goal: close the T1041 exfil / T1571 odd-port holes
# the ATT&CK attacker found, without wrecking clean-benign FP.
# Usage: tools/defend_vs_attack.sh <seed-dir> <out-dir> [real_frac]
set -euo pipefail
cd "$(dirname "$0")/.."
SEED=${1:?seed checkpoint dir, e.g. dqn_g0_s2}
OUT=${2:?out-dir}
RF=${3:-0.5}
mkdir -p "$OUT"
PYTHONPATH=src .venv/bin/python -u -m zeek_acd.train_selfplay \
  --attacker attack \
  --data 'data/iot23/*/conn.log.labeled' \
  --data 'data/mta/2025-*/conn.log.labeled' \
  --data 'data/live2025/conn.log.labeled' \
  --data 'data/benign_web/train/conn.log.labeled' \
  --payoff payoffs/low_fp.json \
  --defender vanilla \
  --defender-init "checkpoints/$SEED/agent_best.pt" \
  --defender-normalizer "checkpoints/$SEED/normalizer_best.json" \
  --defender-gamma 0 --defender-eps 0.1 --defender-lr 3e-4 \
  --real-frac "$RF" \
  --episodes 1000 --eval-every 50 --seed 0 \
  --checkpoint-dir "$OUT" 2>&1 | tee "$OUT/run.out"
