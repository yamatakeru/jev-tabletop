#!/usr/bin/env bash
# Robustness trials: random layouts, same explicit instruction. Results in outputs/trial_<seed>/
set -uo pipefail
cd "$(dirname "$0")/.."
for seed in "$@"; do
  RUN_DIR=outputs/trial_$seed LAYOUT_SEED=$seed \
  INSTRUCTION="Put the red, blue and green cubes into the tray. Leave the yellow cube alone." \
    uv run dora run dataflow.yml --stop-after 10m > outputs/trial_$seed.log 2>&1
  echo "trial $seed done"
done
