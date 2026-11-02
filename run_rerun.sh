#!/usr/bin/env bash
# Re-run every experiment so results/*.json carry per-class metrics.
# Waits for sonar_edge's M1 training to finish first (shared GPU).
set -euo pipefail
export PYTHONUNBUFFERED=1
while pgrep -f "[r]un_m1.sh" >/dev/null; do sleep 30; done
./run_all.sh
./run_randsize.sh
./run_group.sh
./run_dedup.sh
