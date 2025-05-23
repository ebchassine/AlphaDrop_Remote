#!/usr/bin/env bash
set -euo pipefail

# Always run from this script’s directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Base checkpoint path (relative to analysis/)
BASE_DIR="../../../checkpoints/tensorized/baseline/ptb-adam/bs120/\
tensor_transformer_3layer/head_1/\
max_step40000_max_epoch200_log_interval200/median_xmin_pos2"
# Window size to sweep
WINDOW_SIZE=20
# Where to dump graphs
OUTPUT_ROOT="graphs"

mkdir -p "$OUTPUT_ROOT"

for seed_path in "$BASE_DIR"/seed_*; do
  stats_dir="$seed_path/stats"
  if [ -d "$stats_dir" ]; then
    seed_name="$(basename "$seed_path")"
    out_dir="$OUTPUT_ROOT/$seed_name"
    mkdir -p "$out_dir"
    echo "▶ Processing $seed_name …"
    python esd_window.py \
      --esd_dir "$stats_dir" \
      --window_size "$WINDOW_SIZE" \
      --output_dir "$out_dir"
  else
    echo "⚠️  Skipping $(basename "$seed_path") — no stats/ folder found"
  fi
done
