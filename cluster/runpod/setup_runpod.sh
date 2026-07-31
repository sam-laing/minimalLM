#!/bin/bash
# One-time setup on a RunPod pod with a network volume mounted at /workspace.
# Run this INSIDE the pod, after SSH-ing in. Safe to re-run (skips finished steps).
set -euo pipefail

REPO_DIR="/workspace/minimalLM"
DATA_OUT="/workspace/data/lm/sp_1.8B_tokens"
SECRETS_FILE="/workspace/secrets.env"

# --- secrets (WANDB_API_KEY etc.) -----------------------------------------
if [[ -f "$SECRETS_FILE" ]]; then
  echo "Sourcing secrets from $SECRETS_FILE"
  source "$SECRETS_FILE"
else
  echo "No $SECRETS_FILE found."
  echo "Create it once with: echo 'export WANDB_API_KEY=your_key' > $SECRETS_FILE"
  echo "(it lives on the network volume, so this only has to be done once, ever)"
fi

# --- clone repo onto the persistent volume (skip if already there) --------
if [[ ! -d "$REPO_DIR" ]]; then
  echo "Cloning repo into $REPO_DIR"
  git clone https://github.com/sam-laing/minimalLM.git "$REPO_DIR"
fi
cd "$REPO_DIR"

# --- python deps ------------------------------------------------------------
pip install -e . --quiet

# --- HF cache on the volume, not the ephemeral container disk -------------
export HF_HOME="/workspace/huggingface_cache"
mkdir -p "$HF_HOME"

# --- tokenize training data once, reused across every future pod ----------
export PLAINLM_DATA_OUT="$DATA_OUT"
export PLAINLM_NUM_CPUS="${PLAINLM_NUM_CPUS:-$(nproc)}"

if [[ -d "$DATA_OUT/train" ]]; then
  echo "Train data already prepared at $DATA_OUT/train, skipping."
else
  echo "Preparing SlimPajama train split -> $DATA_OUT/train"
  python data/datasets/slim_pajama/prepare_train.py
fi

echo "Setup complete."
echo "Train:  torchrun --standalone --nproc_per_node=<N_GPUS> train.py --config=config/config_runpod.yaml"
