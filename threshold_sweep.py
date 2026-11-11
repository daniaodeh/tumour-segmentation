"""Patches kept per smooth-probability threshold, straight from the mask raster.

Avoids contours entirely: a patch is kept if its 1024x1024 level-0 box contains
any mask pixel above the threshold (the approved 'box intersects' rule). Holes
need no special handling here - a hole is simply sub-threshold pixels.

Takes the MAX of each patch box once, so every threshold is read off the same
pass. Appends per slide and skips rows already present, so it survives a restart.

Writes: <lung>/experiments/segmentation/threshold_sweep.csv
"""
import csv, os, sys
from pathlib import Path
import cv2, h5py, numpy as np

ROOT = Path(__file__).resolve().parents[2]      # projects/lung
OUT = ROOT / "experiments/segmentation/output"
PATCH = ROOT / ("experiments/classification/output/lung_0.4_remove_pen_partial_20/"
                "20x_512px_0px_overlap/patches")
CSV = ROOT / "experiments/segmentation/threshold_sweep.csv"
THRESHOLDS = [31, 63, 85, 110, 127, 153, 178, 204, 229]
BOX = 1024
FIELDS = (["slide", "n_tissue", "mask_w", "mask_h", "px_lvl0"]
          + [f"keep_{t}" for t in THRESHOLDS] + ["keep_hysteresis"])


def box_maxima(img, coords, sx, sy):
    H, W = img.shape
    x0 = np.clip((coords[:, 0] / sx).astype(int), 0, W - 1)
    x1 = np.clip(np.ceil((coords[:, 0] + BOX) / sx).astype(int), 1, W)
    y0 = np.clip((coords[:, 1] / sy).astype(int), 0, H - 1)
    y1 = np.clip(np.ceil((coords[:, 1] + BOX) / sy).astype(int), 1, H)
    return np.fromiter((img[a:b, c:d].max() for a, b, c, d in zip(y0, y1, x0, x1)),
                       dtype=np.uint8, count=len(coords))


def main():
    done = set()
    if CSV.exists():
        with open(CSV) as f:
            done = {r["slide"] for r in csv.DictReader(f)}
    dirs = sorted(d for d in OUT.iterdir()
                  if (d / "result_probability_smooth.png").exists()
                  and (PATCH / f"{d.name}_patches.h5").exists())
    todo = [d for d in dirs if d.name not in done]
    print(f"{len(dirs)} slides, {len(done)} already done, {len(todo)} to go", flush=True)

    new = not CSV.exists()
    with open(CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader(); fh.flush()
        for i, d in enumerate(todo, 1):
            sm = cv2.imread(str(d / "result_probability_smooth.png"), cv2.IMREAD_GRAYSCALE)
            if sm is None:
                print(f"  [SKIP] unreadable: {d.name}", flush=True); continue
            with h5py.File(PATCH / f"{d.name}_patches.h5") as f:
                coords = f["coords"][...]
                a = dict(f["coords"].attrs)
            H, W = sm.shape
            sx, sy = a["level0_width"] / W, a["level0_height"] / H
            mx = box_maxima(sm, coords, sx, sy)
            r = dict(slide=d.name, n_tissue=len(coords), mask_w=W, mask_h=H,
                     px_lvl0=round(sx * sy, 3))
            for t in THRESHOLDS:
                r[f"keep_{t}"] = int((mx > t).sum())
            seg = cv2.imread(str(d / "result_segmentation.png"), cv2.IMREAD_GRAYSCALE)
            r["keep_hysteresis"] = (int((box_maxima(seg, coords, sx, sy) > 127).sum())
                                    if seg is not None and seg.shape == sm.shape else -1)
            w.writerow(r); fh.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    print(f"done -> {CSV}", flush=True)


if __name__ == "__main__":
    main()
