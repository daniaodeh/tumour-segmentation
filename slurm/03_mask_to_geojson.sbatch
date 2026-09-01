#!/bin/bash
#SBATCH --job-name=03_mask_to_geojson
#SBATCH --output=/sci/labs/nirpillar/dania.odeh/projects/lung/code/segmentation/slurm/logs/%x_%j.out
#SBATCH --error=/sci/labs/nirpillar/dania.odeh/projects/lung/code/segmentation/slurm/logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Stage 3/10: convert tumor masks (from stage 2) to GeoJSON contours - the
# actual tumor regions used later to filter TRIDENT patches (stage 6). CPU
# only. Skips slides whose GeoJSON already exists and slides not yet
# segmented (logs and continues).
#
# --source smooth: use result_probability_smooth.png rather than the raw
# result_segmentation.png - some slides' raw mask came back with zero
# tumor pixels even where the smooth probability map shows real signal.
#
# --overwrite: ONE-TIME ONLY, to regenerate all already-processed slides'
# geojson under the new smooth source instead of the stale raw-mask one
# (process_slide() otherwise skips any slide whose geojson already exists,
# regardless of --source). REMOVE this flag once this run completes, or
# every future stage-3 run reprocesses all slides from scratch instead of
# only newly-finished stage-2 ones.
set -euo pipefail

source /sci/labs/nirpillar/dania.odeh/projects/lung/env.sh
export CONDA_EXE=/sci/labs/nirpillar/shared/dania/miniconda/bin/conda
source /sci/labs/nirpillar/shared/dania/miniconda/etc/profile.d/conda.sh
conda activate /sci/labs/nirpillar/dania.odeh/.conda/envs/tumour-segmentation

cd "$RESEARCH_ROOT/projects/lung/code/segmentation"

python mask_to_geojson.py \
  --output_dir "$RESEARCH_ROOT/projects/lung/experiments/segmentation/output/" \
  --geojson_dir "$RESEARCH_ROOT/projects/lung/experiments/segmentation/geojson/" \
  --source smooth \
  --overwrite
