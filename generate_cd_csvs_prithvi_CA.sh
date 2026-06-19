#!/bin/bash
#SBATCH --job-name=gen_csv_prithvi_CA
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gen_csv_prithvi_CA_%j.out
#SBATCH --error=logs/gen_csv_prithvi_CA_%j.err

set -euo pipefail

cd /bigdata/eldawylab/sdas050/MS_Research
source spectralgptenv/bin/activate

mkdir -p logs

echo "==== Generating Prithvi CA CSVs ===="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

python generate_cd_csvs_prithvi_CA.py

echo ""
echo "==== Done ===="
ls -lh change_detection_chips/prithvi/NorthCA_chips.csv
ls -lh change_detection_chips/prithvi/CentCA_chips.csv
ls -lh change_detection_chips/prithvi/SouthCA_chips.csv
