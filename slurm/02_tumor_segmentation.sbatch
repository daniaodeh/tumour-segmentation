#!/bin/bash
#SBATCH --job-name=02_tumor_segmentation
#SBATCH --output=/sci/labs/nirpillar/dania.odeh/projects/lung/code/segmentation/slurm/logs/%x_%j.out
#SBATCH --error=/sci/labs/nirpillar/dania.odeh/projects/lung/code/segmentation/slurm/logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --partition=dogfish

# Stage 2/10: our own tumor-segmentation model, full_scan_segmentation.py.
# Single-GPU only - the script has no multi-GPU support, so requesting more
# than 1 here would just sit idle. Skips slides whose result_segmentation.png
# already exists; skips (rather than crashes on) individually broken slides
# via its own per-slide try/except in full_scan_segmentation.py's main loop.
#
# Requires the `tumour-segmentation` conda env to exist first - it doesn't
# yet on this cluster. Create it once with:
#   conda env create -f ../environment.yml \
#     --prefix /sci/labs/nirpillar/dania.odeh/.conda/envs/tumour-segmentation
set -euo pipefail

source /sci/labs/nirpillar/dania.odeh/projects/lung/env.sh
export CONDA_EXE=/sci/labs/nirpillar/shared/dania/miniconda/bin/conda
source /sci/labs/nirpillar/shared/dania/miniconda/etc/profile.d/conda.sh
conda activate /sci/labs/nirpillar/dania.odeh/.conda/envs/tumour-segmentation

cd "$RESEARCH_ROOT/projects/lung/code/segmentation"

# Rebuild the slide list fresh from new_slides/ each run, so it stays in sync
# even if the symlink set changed since the last run (see stage 1).
python3 -c "
from pathlib import Path
slides_dir = Path('$RESEARCH_ROOT/projects/lung/data/new_slides')
out = Path('$RESEARCH_ROOT/projects/lung/data/lung_partial/metadata/slides.csv')
paths = sorted(str(p) for p in slides_dir.glob('*.svs'))
with open(out, 'w') as f:
    f.write('Path\n')
    for p in paths:
        f.write(p + '\n')
print(f'{len(paths)} slides written to {out}')
"

python full_scan_segmentation.py \
  "$RESEARCH_ROOT/projects/lung/data/lung_partial/metadata/slides.csv" \
  "$RESEARCH_ROOT/projects/lung/experiments/segmentation/output" \
  "$RESEARCH_ROOT/projects/lung/experiments/segmentation/local-cache" \
  "$RESEARCH_ROOT/projects/lung/data/lung_partial/model.tar" \
  --gpu 0 \
  -v
