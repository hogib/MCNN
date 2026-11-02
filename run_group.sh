#!/usr/bin/env bash
# Experiment 2: split by pseudo-sequence groups (no near-duplicates across train/test).
set -euo pipefail
export PYTHONUNBUFFERED=1
while pgrep -f run_randsize.sh >/dev/null; do sleep 20; done
uv run sonar-cnn train --split group --model mcnn   --input pad
uv run sonar-cnn train --split group --model varcnn --input native
uv run sonar-cnn train --split group --model varcnn --input randsize --epochs 50
uv run sonar-cnn train --split group --model mcnn   --input randsize-pad --epochs 50
