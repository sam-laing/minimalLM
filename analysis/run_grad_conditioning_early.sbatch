#!/bin/bash
#SBATCH --job-name=gradcond_early
#SBATCH --account=p_neurasearch
#SBATCH --partition=capella
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/home/sala597i/minimalLM/analysis/logs/gradcond_early_%j.out
#SBATCH --error=/home/sala597i/minimalLM/analysis/logs/gradcond_early_%j.err

set -e

source /software/genoa/r25.06/Miniconda3/25.5.1-1/etc/profile.d/conda.sh
conda activate /home/sala597i/conda_envs/minimalLM

cd /home/sala597i/minimalLM
PYTHONPATH=. python analysis/gradient_conditioning.py --group early
