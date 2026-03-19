#!/bin/bash
# Run orthogonal-init Muon training in an interactive job
# Usage:  
#   ./run_ortho_muon.sh                  # default: use wandb
#   ./run_ortho_muon.sh --no-wandb       # disable wandb for quick testing

CONFIG="config/ortho_muon_config.yaml"

# Check for --no-wandb flag
if [[ "$1" == "--no-wandb" ]]; then
    echo "WandB disabled for this run"
    export WANDB_MODE=disabled
fi

echo "========================================="
echo "  Orthogonal-init Muon Training"
echo "  Config: ${CONFIG}"
echo "========================================="

python train.py --config "${CONFIG}"
