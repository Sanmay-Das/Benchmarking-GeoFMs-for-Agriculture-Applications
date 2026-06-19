#!/bin/bash
#SBATCH --job-name=gen_csv_satmae_NC
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/gen_csv_satmae_NC_%j.out
#SBATCH --error=logs/gen_csv_satmae_NC_%j.err

set -euo pipefail

cd /bigdata/eldawylab/sdas050/MS_Research
source spectralgptenv/bin/activate

mkdir -p logs

echo "==== Generating SatMAE NC CSVs ===="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

python generate_cd_csvs_satmae_NC.py

echo ""
echo "==== Done ===="
ls -lh change_detection_chips/satmae/NENC_chips.csv
ls -lh change_detection_chips/satmae/ECNC_chips.csv
ls -lh change_detection_chips/satmae/EastNC_chips.csv
