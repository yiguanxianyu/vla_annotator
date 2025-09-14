#!/usr/bin/env bash
set -euo pipefail

python -m src.inference.run_infer --config configs/infer.yaml "$@" || python src/inference/run_infer.py --config configs/infer.yaml "$@"

