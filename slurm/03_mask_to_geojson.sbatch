#!/bin/bash
#SBATCH --job-name=03_mask_to_geojson
#SBATCH --output=/sci/labs/nirpillar/shared/dania/projects/lung/code/segmentation/slurm/logs/%x_%j.out
#SBATCH --error=/sci/labs/nirpillar/shared/dania/projects/lung/code/segmentation/slurm/logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=dogfish

# Stage 3/10: convert tumor masks (from stage 2) to GeoJSON contours - the
# actual tumor regions used later to filter TRIDENT patches (stage 6). CPU
# only. Skips slides whose GeoJSON already exists and slides not yet
# segmented (logs and continues).
set -euo pipefail

source /sci/labs/nirpillar/shared/dania/projects/lung/env.sh
export CONDA_EXE=/sci/labs/nirpillar/shared/dania/miniconda/bin/conda
source /sci/labs/nirpillar/shared/dania/miniconda/etc/profile.d/conda.sh
conda activate /sci/labs/nirpillar/dania.odeh/.conda/envs/tumour-segmentation

cd "$RESEARCH_ROOT/projects/lung/code/segmentation"

python mask_to_geojson.py \
  --output_dir "$RESEARCH_ROOT/projects/lung/experiments/segmentation/output/" \
  --geojson_dir "$RESEARCH_ROOT/projects/lung/experiments/segmentation/geojson/"
