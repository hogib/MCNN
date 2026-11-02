#!/usr/bin/env bash
# Experiment 1: crop size and aspect ratio replaced with random ones, train and test.
set -euo pipefail
export PYTHONUNBUFFERED=1
uv run sonar-cnn train --model varcnn --input randsize-constant --epochs 50
uv run sonar-cnn train --model mcnn   --input randsize-pad-constant --epochs 50
uv run sonar-cnn train --model varcnn --input randsize --epochs 50
uv run sonar-cnn train --model mcnn   --input randsize-pad --epochs 50
