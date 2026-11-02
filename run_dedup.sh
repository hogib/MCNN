#!/usr/bin/env bash
# Experiment 3: deduplicated dataset (data/dedup_t90, built by `python -m sonar_cnn.dedup`),
# random sizes so size carries no class information.
set -euo pipefail
export PYTHONUNBUFFERED=1
while pgrep -f "^bash ./run_group.sh" >/dev/null; do sleep 20; done
uv run sonar-cnn train --data data/dedup_t90 --model varcnn --input randsize --epochs 50
uv run sonar-cnn train --data data/dedup_t90 --model mcnn   --input randsize-pad --epochs 50
