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
# --threshold 31: 12% of 255, the clinician's cut as of 2026-11-10, replacing
# the script's 127 default (50%). It MUST match stage 6's --threshold. Stage 6
# patches from the raster directly and does not read these GeoJSONs, so a
# mismatch does not fail - it silently leaves the overlays and QuPath
# describing a mask that was never patched.
#
# --overwrite: ONE-TIME ONLY after a source or threshold change, to regenerate
# already-processed slides (process_slide() otherwise skips any slide whose
# geojson exists, regardless of --source or --threshold). REMOVE this flag
# once the run completes, or every future stage-3 run reprocesses all slides
# instead of only newly-finished stage-2 ones.
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
  --threshold 31 \
  --overwrite
