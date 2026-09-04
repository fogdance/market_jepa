#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/v/Documents/work/market_jepa
python_bin="$project_dir/.venv/bin/python"
checkpoint=artifacts/checkpoints/market_jepa_v0_default/last.pt
latents_dir=artifacts/latents/market_jepa_v0_6_2
context_dir=artifacts/latents/market_jepa_v0_7_frozen
output_dir=artifacts/evaluation/market_jepa_v0_7_frozen

cd "$project_dir"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

"$python_bin" export_frozen_context.py \
  --checkpoint "$checkpoint" \
  --split train \
  --output-dir "$context_dir" \
  --device cuda

"$python_bin" export_frozen_context.py \
  --checkpoint "$checkpoint" \
  --split validation \
  --output-dir "$context_dir" \
  --device cuda

"$python_bin" run_frozen_predictive_state.py \
  --checkpoint "$checkpoint" \
  --latents-dir "$latents_dir" \
  --context-dir "$context_dir" \
  --output-dir "$output_dir" \
  --device cuda
