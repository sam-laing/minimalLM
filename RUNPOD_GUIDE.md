# Running minimalLM on RunPod (branch: `orthog_init`)

This repo was originally set up for an HPC cluster (Slurm/Condor, see `cluster/slurm`
and `cluster/condor` — not needed on RunPod, ignore them). This guide covers running
the same training on a rented RunPod GPU instead.

All state (repo, tokenized data, checkpoints) lives on a **Network Volume**, mounted
at `/workspace`, so it survives pod termination. You only pay for GPU time while a
pod is actually running.

**Before any of this**: today's setup changes (`config/config_runpod.yaml`,
`cluster/runpod/setup_runpod.sh`, the WandB-key fix in `utils.py`/`config.yaml`,
the parameterized data-prep scripts) are still local, uncommitted, on `orthog_init`.
Commit and `git push origin orthog_init` before cloning on a pod, or the pod will
pull a stale version without any of this.

## 0. Before you start

- Rotate your WandB API key: https://wandb.ai/authorize (an old key used to be
  committed in `config/config.yaml` — treat it as compromised).
- Commit + push your local `orthog_init` changes (see above).

## 1. Create a Network Volume

RunPod console → Storage → Network Volumes → New Volume.

- Pick a datacenter region that has the GPU type you want (check availability first).
- Size: 150GB is comfortably enough — tokenized SlimPajama is ~7-15GB, and 5 runs
  worth of checkpoints (4 each, per current config) is ~22GB.

## 2. Deploy one pod — training GPU, same pod does data prep first

RunPod console → Pods → Deploy, same region as the volume.

- Template: **RunPod PyTorch** (prebuilt CUDA + PyTorch image).
- GPU: whatever you're training with (e.g. A100).
- Attach the network volume, mount path `/workspace`.
- Deploy, then open the pod's **Connect → Web Terminal** (or SSH).

There's no separate throwaway/CPU-only pod — the GPU sits idle for the ~20-60 min
tokenization takes on first run, which costs a couple dollars and is simpler than
juggling two pods. Every pod after this first one skips tokenization entirely
(see step 4).

## 3. One-time setup + data prep (in the pod's terminal)

```bash
# one-time secret, persisted on the volume so you never re-enter it
echo 'export WANDB_API_KEY=your_new_rotated_key' > /workspace/secrets.env

git clone -b orthog_init https://github.com/sam-laing/minimalLM.git /workspace/minimalLM
cd /workspace/minimalLM
bash cluster/runpod/setup_runpod.sh
```

`setup_runpod.sh` installs the package (`pip install -e .`), points `HF_HOME` at
the volume, and tokenizes SlimPajama straight from the HF Hub (streamed — raw text
is never saved, only the final tokenized dataset) into
`/workspace/data/lm/sp_1.8B_tokens/train`. It skips this step automatically if that
path already exists, so it's safe to re-run.

**If it errors with a 401/gated-repo message**, the dataset needs HF auth:

```bash
huggingface-cli login   # paste a token from https://huggingface.co/settings/tokens
echo 'export HF_TOKEN=your_hf_token' >> /workspace/secrets.env
bash cluster/runpod/setup_runpod.sh   # re-run
```

## 4. Train

```bash
cd /workspace/minimalLM
source /workspace/secrets.env
python train.py --config=config/config_runpod.yaml
```

Single GPU, single process — matches `cluster/slurm/train_muon_90M.sbatch`'s
`gres=gpu:1` setup exactly, just with `lr: 0.008` instead of 0.01 (already set in
`config/config_runpod.yaml`). If you rent a multi-GPU pod instead, use:

```bash
torchrun --standalone --nproc_per_node=<N_GPUS> train.py --config=config/config_runpod.yaml
```

Notes on `config/config_runpod.yaml`:
- Paths point at `/workspace/...` (the volume) instead of the old cluster paths.
- `eval: False` / `validset_path: null` — no validation set.
- `save_every_steps: 6866` → 4 checkpoints per run (~25/50/75/100% of training).
- Everything else (model size, optimizer, schedule) matches the original cluster config.

## 5. Running 5 jobs (sweep)

Same pod, one GPU per job, using the existing `--job_idx` sweep mechanism:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config=config/sweep.yaml --job_idx=0 &
CUDA_VISIBLE_DEVICES=1 python train.py --config=config/sweep.yaml --job_idx=1 &
CUDA_VISIBLE_DEVICES=2 python train.py --config=config/sweep.yaml --job_idx=2 &
CUDA_VISIBLE_DEVICES=3 python train.py --config=config/sweep.yaml --job_idx=3 &
CUDA_VISIBLE_DEVICES=4 python train.py --config=config/sweep.yaml --job_idx=4 &
wait
```

Checkpoints land in `/workspace/checkpoints/muon_90M_1p8B/job_idx_<N>/`, no
collisions between jobs.

## 6. Shutting down between sessions

- **Terminate** the pod when done — stops all billing except the volume's small
  storage cost. Nothing is lost, everything's on the volume.
- Next session: deploy a new pod, attach the same volume, skip straight to step 4
  (no setup/tokenization needed again).

## 7. Checkpoint storage off RunPod (optional)

Checkpoints are ~1.1GB each (weights + optimizer momentum + saved grads, all fp32,
~90M-param model — see `checkpoint_utils.py`). At 4/run × 5 runs that's ~22GB,
comfortably inside the volume, but if you want a copy outside RunPod entirely
(e.g. to run `plot_svs.py`-style analysis locally without eating your 60GB), push
to a private Hugging Face model repo instead of downloading raw checkpoints:

```python
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path="/workspace/checkpoints/muon_90M_1p8B",
    repo_id="sam-laing/muon-90M-checkpoints",
    repo_type="model",
)
```

Or better: run the analysis scripts (`plot_svs.py`, etc.) directly on the pod
against `/workspace/checkpoints/...` and only pull down the small CSV/PNG outputs.
