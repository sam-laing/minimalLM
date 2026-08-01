#!/bin/bash
# Run the full spectral-conditioning study locally (CPU is enough - it's just
# SVDs on a few dozen small matrices per checkpoint, no training involved)
# against whatever checkpoints you've actually downloaded to
# ~/minimalLM_checkpoints. Writes both a flat CSV and a nested JSON
# (per checkpoint -> per layer -> momentum vs gradient side by side).
set -euo pipefail

REPO_DIR="/home/slaing/cloud_computing/minimalLM"
CKPT_DIR="${CKPT_DIR:-$HOME/minimalLM_checkpoints}"
OUT_DIR="${OUT_DIR:-$HOME/minimalLM_checkpoints/results}"

mkdir -p "$OUT_DIR"

set +u  # conda.sh references unset vars (e.g. PS1) in non-interactive shells
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate 310nets
set -u

cd "$REPO_DIR"
PYTHONPATH=. python analysis/spectral_analysis.py \
  --config=config/config_runpod.yaml \
  --ckpt_dir="$CKPT_DIR" \
  --out_path="$OUT_DIR/spectral_results.csv" \
  --json_out_path="$OUT_DIR/spectral_results.json"

echo "Done. Results in $OUT_DIR"
