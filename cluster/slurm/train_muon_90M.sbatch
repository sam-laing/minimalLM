#!/bin/bash
#SBATCH --job-name=muon_90M_1p8B
#SBATCH --account=p_neurasearch
#SBATCH --partition=capella
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/sala597i/minimalLM/cluster/slurm/logs/muon_90M_%j.out
#SBATCH --error=/home/sala597i/minimalLM/cluster/slurm/logs/muon_90M_%j.err

set -e

source /software/genoa/r25.06/Miniconda3/25.5.1-1/etc/profile.d/conda.sh
conda activate /home/sala597i/conda_envs/minimalLM

export HF_HOME=/data/horse/ws/sala597i-slaing/huggingface_cache

cd /home/sala597i/minimalLM
python train.py --config=config/config.yaml
