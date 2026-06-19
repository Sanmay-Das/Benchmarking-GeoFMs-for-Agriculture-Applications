#!/bin/bash
#SBATCH --job-name=gen_csv_cd_CA
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gen_csv_cd_CA_%j.out
#SBATCH --error=logs/gen_csv_cd_CA_%j.err

set -euo pipefail

cd /bigdata/eldawylab/sdas050/MS_Research
source spectralgptenv/bin/activate

mkdir -p logs

echo "==== Generating CD CSVs for CA (SatMAE + SpectralGPT) ===="
echo "Job:  $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo ""

echo "--- SatMAE ---"
python generate_cd_csvs_satmae_CA.py

echo ""
echo "--- SpectralGPT ---"
python generate_cd_csvs_spectralgpt_CA.py

echo ""
echo "==== Done ===="
ls -lh change_detection_chips/satmae/*CA*.csv change_detection_chips/spectralgpt/*CA*.csv
