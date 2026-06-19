#!/bin/bash
#SBATCH --job-name=prithvi_NWIA_infer
#SBATCH --partition=epyc
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/prithvi_NWIA_CPU_%j.out
#SBATCH --error=logs/prithvi_NWIA_CPU_%j.err

set -euo pipefail

module purge

cd /bigdata/eldawylab/sdas050/MS_Research
source prithvifmenv/bin/activate

# Set PyTorch to use all allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python infer_prithvi_southCA.py