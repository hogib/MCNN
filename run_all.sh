#!/usr/bin/env bash
# Runs every experiment in sequence. Results go to results/*.json.
set -euo pipefail
uv run sonar-cnn size-baseline
uv run sonar-cnn size-baseline --train-frac 0.9
uv run sonar-cnn train --model mcnn --input pad
uv run sonar-cnn train --model mcnn --input pad --train-frac 0.9
uv run sonar-cnn train --model mcnn --input silhouette
uv run sonar-cnn train --model varcnn --input native
uv run sonar-cnn train --model varcnn --input native-constant
