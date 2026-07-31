#!/bin/bash
#SBATCH --job-name=prep_slimpj_1.8B
#SBATCH --account=p_neurasearch
#SBATCH --partition=capella
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/sala597i/minimalLM/data/datasets/slim_pajama/logs/slurm_%j.out
#SBATCH --error=/home/sala597i/minimalLM/data/datasets/slim_pajama/logs/slurm_%j.err

source /software/genoa/r25.06/Miniconda3/25.5.1-1/etc/profile.d/conda.sh
conda activate minimalLM

export HF_HOME=/data/cat/ws/sala597i-slaing/huggingface_cache

python /home/sala597i/minimalLM/data/datasets/slim_pajama/prepare_train.py
