#!/usr/bin/env bash
set -euo pipefail

python -m src.training.train_loop --config configs/train.yaml || python -c "from src.training.train_loop import run_train; run_train('configs/train.yaml')"

