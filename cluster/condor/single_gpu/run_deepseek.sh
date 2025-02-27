#!/bin/bash

export HOME=/lustre/home/slaing

source ~/miniforge3/etc/profile.d/conda.sh
conda activate deepseek

# Job specific vars
config=$1
job_idx=$2 # CONDOR job arrays range from 0 to n-1

# Execute python script
python /lustre/home/slaing/minimalLM/train.py --config=$config --job_idx=$job_idx

