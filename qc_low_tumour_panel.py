"""Thumbnail + probability map + multi-threshold contours for slides whose
tumour mask is near-empty at the current 50% cut.

Three columns per slide: H&E, the smoothed probability map, and the same H&E
with the mask boundary drawn at four thresholds. The point of the fourth
column is to separate two very different failure modes that look identical in
a patch count:

  - the model found nothing anywhere (no contour at any threshold)
  - the model localised a lesion but scored it under 50% (contours nest
    inside one another, shrinking as the threshold rises)

Patch counts per threshold come from experiments/segmentation/threshold_sweep.csv
so the picture and the numbers cannot drift apart.

Which slides are shown is NOT hardcoded. By default the panel takes the
--n slides with the lowest tumour fraction out of the filter's own QC table,
so it follows the data rather than a list that goes stale the moment the
threshold changes. Pass --slides to override. Slide identifiers are pathology
accession numbers, so they do not belong in a tracked source file.

Writes: <lung>/experiments/segmentation/qc_low_tumour_panel.png
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from openslide import OpenSlide

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]      # projects/lung
OUT = ROOT / "experiments/segmentation/output"
SVS = ROOT / "data/new_slides"

QC_CSV = ROOT / ("experiments/classification/output/lung_tumour_filtered/"
                 "tumour_20x_512px/filter_qc.csv")

# threshold -> (colour, linewidth, label). Ordered low to high so the
# permissive contours are drawn first and the 50% cut sits on top.
LEVELS = [(31,  "#4FC3F7", 0.7, "12%"),
          (63,  "#FF8A3D", 0.8, "25%"),
          (85,  "#FFC83D", 0.9, "33%"),
          (127, "#00E676", 1.3, "50%  (current cut)")]


def pick_slides(qc_csv: Path, n: int):
    """The n slides with the least tumour, from the filter's QC table."""
    q = pd.read_csv(qc_csv)
    return q.sort_values("pct_tumour").head(n).slide.tolist()


def main(SLIDES):
    lab = pd.read_csv(ROOT / "data/lung_partial/metadata/labels_till2024_with_category.csv",
                      dtype=str).set_index("FILENAME")
    sw = pd.read_csv(ROOT / "experiments/segmentation/threshold_sweep.csv").set_index("slide")

    fig, axes = plt.subplots(len(SLIDES), 3, figsize=(14.5, 3.05 * len(SLIDES)))
    for i, s in enumerate(SLIDES):
        sm = cv2.imread(str(OUT / s / "result_probability_smooth.png"), cv2.IMREAD_GRAYSCALE)
        H, W = sm.shape
        with OpenSlide(str(SVS / f"{s}.svs")) as wsi:
            thumb = np.array(wsi.get_thumbnail((W, H)).convert("RGB"))
        th = cv2.resize(thumb, (W, H), interpolation=cv2.INTER_AREA)
        r = lab.loc[s] if s in lab.index else None
        k = sw.loc[s]

        counts = "   ".join(f"{lb.split()[0]}:{int(k[f'keep_{t}'])}" for t, _, _, lb in LEVELS[::-1])
        axes[i, 0].imshow(th)
        axes[i, 0].set_ylabel(f"{s}\n{counts}", fontsize=7.6, rotation=0,
                              ha="right", va="center", labelpad=10)
        cat = f"{r['adenocarcinoma_type']} | {str(r['diagnosis_Type'])[:36]}" if r is not None else ""
        axes[i, 0].set_title(cat, fontsize=8.2, loc="left")

        im = axes[i, 1].imshow(sm, cmap="inferno", vmin=0, vmax=255)
        axes[i, 1].set_title(f"smooth probability    max = {int(sm.max())}/255 "
                             f"({sm.max()/255:.0%})", fontsize=8.2)

        axes[i, 2].imshow(th)
        drawn = []
        for t, colour, lw, label in LEVELS:
            if sm.max() > t:
                axes[i, 2].contour(sm, levels=[t], colors=colour, linewidths=lw)
                drawn.append(label.split()[0])
        axes[i, 2].set_title("mask boundary at: " + (", ".join(drawn) if drawn
                             else "nothing above 12%"), fontsize=8.2)
        for a in axes[i]:
            a.set_xticks([]); a.set_yticks([])

    fig.colorbar(im, ax=axes[:, 1], fraction=0.015, pad=0.01)
    handles = [Line2D([0], [0], color=c, lw=max(lw, 1.4), label=lb)
               for _, c, lw, lb in LEVELS[::-1]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.982),
               ncol=4, frameon=False, fontsize=10, title="probability threshold",
               title_fontsize=10)
    fig.suptitle("Slides with a near-empty tumour mask: H&E, model probability, "
                 "and the mask boundary at four thresholds", fontsize=13, y=0.995)

    out = ROOT / "experiments/segmentation/qc_low_tumour_panel.png"
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    for s in SLIDES:
        k = sw.loc[s]
        print(f"  {s:30s} " + "  ".join(
            f"{lb.split()[0]}={int(k[f'keep_{t}']):4d}" for t, _, _, lb in LEVELS[::-1]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--slides", nargs="+", default=None,
                   help="Slide stems to show. Default: the --n lowest-tumour "
                        "slides in --qc_csv.")
    p.add_argument("--qc_csv", type=Path, default=QC_CSV)
    p.add_argument("-n", type=int, default=9)
    a = p.parse_args()
    main(a.slides or pick_slides(a.qc_csv, a.n))
