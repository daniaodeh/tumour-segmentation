"""
Filter TRIDENT patch coordinates down to the ones overlapping tumour.

The tumour mask is read from the segmentation model's smoothed probability
raster, NOT from segmentation.geojson. The geojsons are only
`result_probability_smooth.png > threshold` pushed through cv2.findContours,
so the raster is the same information without the contour's losses: no
half-pixel shrink, no RETR_CCOMP holes re-emitted as positive polygons (264
of 333 slides had at least one), and no invalid geometries to repair.

Containment rule: a patch is kept if its box contains any mask pixel above
the threshold ("box intersects"). The box is `patch_size_level0` (1024 px),
because coords are in level-0 pixels - `patch_size` (512) is the 20x pixel
count and is the wrong unit in a level-0 coordinate space.

For each slide, writes an H5 holding:
  - coords        : (M, 2) the TUMOUR subset, in level-0 pixels. Named
                    `coords` so TRIDENT's feature extractor reads it directly.
  - all_coords    : (N, 2) every patch the slide started with.
  - in_tumour     : (N,) bool, the mask that turns all_coords into coords.
plus provenance attributes on `coords` recording which rule, which raster and
which threshold produced it (see PROVENANCE_DOC below).

Slides whose mask is empty at this threshold still get a file, with coords of
shape (0, 2), and are listed in the QC csv. They are not silently skipped:
a missing file is indistinguishable from a crashed run.

Usage:
  python filter_coords_by_tumour.py \
    --patches_dir ../../experiments/classification/output/lung_0.4_remove_pen_partial_20/20x_512px_0px_overlap/patches \
    --prob_dir    ../../experiments/segmentation/output \
    --output_dir  ../../experiments/classification/output/lung_tumour_filtered/tumour_20x_512px/patches \
    --threshold   31

  # Single slide
  python filter_coords_by_tumour.py ... --slide <slide-stem>
"""

import argparse
import csv
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np

MASK_NAME = "result_probability_smooth.png"
PROVENANCE_DOC = "box-intersects at patch_size_level0 against smooth probability > threshold"


def box_maxima(mask: np.ndarray, coords: np.ndarray, box: int,
               level0_w: int, level0_h: int) -> np.ndarray:
    """Max mask value under each patch box, in mask-pixel space."""
    H, W = mask.shape
    sx, sy = level0_w / W, level0_h / H
    x0 = np.clip((coords[:, 0] / sx).astype(int), 0, W - 1)
    x1 = np.clip(np.ceil((coords[:, 0] + box) / sx).astype(int), 1, W)
    y0 = np.clip((coords[:, 1] / sy).astype(int), 0, H - 1)
    y1 = np.clip(np.ceil((coords[:, 1] + box) / sy).astype(int), 1, H)
    return np.fromiter((mask[a:b, c:d].max() for a, b, c, d in zip(y0, y1, x0, x1)),
                       dtype=np.uint8, count=len(coords))


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def process_slide(h5_path: Path, mask_path: Path, out_path: Path,
                  threshold: int, overwrite: bool):
    """Returns a QC row, or None if the slide could not be processed."""
    if out_path.exists() and not overwrite:
        print(f"  [SKIP] exists (--overwrite to redo): {out_path.name}")
        return None
    if not mask_path.exists():
        print(f"  [SKIP] no {MASK_NAME}: {mask_path.parent.name}")
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"  [SKIP] unreadable mask: {mask_path}")
        return None

    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][...]
        attrs = dict(f["coords"].attrs)

    box = int(attrs["patch_size_level0"])
    mx = box_maxima(mask, coords, box, int(attrs["level0_width"]),
                    int(attrs["level0_height"]))
    in_tumour = mx > threshold
    tumour_coords = coords[in_tumour]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        ds = f.create_dataset("coords", data=tumour_coords.reshape(-1, 2))
        for k, v in attrs.items():
            ds.attrs[k] = v
        # Provenance: without these, a filtered file cannot be told apart from
        # one made by a different rule, mask or threshold - which cost us a
        # whole investigation once already.
        ds.attrs["filter_rule"] = PROVENANCE_DOC
        ds.attrs["filter_box_level0"] = box
        ds.attrs["filter_mask_source"] = MASK_NAME
        ds.attrs["filter_mask_threshold"] = int(threshold)
        ds.attrs["filter_mask_sha1"] = sha1(mask_path)
        ds.attrs["filter_script"] = Path(__file__).name
        ds.attrs["filter_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        f.create_dataset("all_coords", data=coords)
        f.create_dataset("in_tumour", data=in_tumour)

    n_total, n_tumour = len(coords), int(in_tumour.sum())
    pct = 100 * n_tumour / n_total if n_total else 0.0
    flag = "  [EMPTY MASK]" if n_tumour == 0 else ""
    print(f"  [OK]   {n_tumour:,}/{n_total:,} patches in tumour ({pct:.1f}%){flag}")
    return dict(slide=h5_path.stem.replace("_patches", ""), n_tissue=n_total,
                n_tumour=n_tumour, pct_tumour=round(pct, 2),
                mask_max=int(mask.max()), threshold=int(threshold),
                empty=int(n_tumour == 0))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patches_dir", type=Path, required=True,
                   help="Directory of TRIDENT *_patches.h5 files")
    p.add_argument("--prob_dir", type=Path, required=True,
                   help=f"Segmentation output root holding <slide>/{MASK_NAME}")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Where to write filtered H5 files")
    p.add_argument("--threshold", type=int, default=31,
                   help="Keep a patch if any mask pixel under it exceeds this "
                        "0-255 value (default 31 = 12%%)")
    p.add_argument("--qc_csv", type=Path, default=None,
                   help="Where to write the per-slide QC table "
                        "(default <output_dir>/../filter_qc.csv)")
    p.add_argument("--slide", type=str, default=None, help="Process one slide")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    for d in (args.patches_dir, args.prob_dir):
        if not d.is_dir():
            print(f"Error: not a directory: {d}")
            sys.exit(1)

    pattern = f"{args.slide}_patches.h5" if args.slide else "*_patches.h5"
    h5_files = sorted(args.patches_dir.glob(pattern))
    if not h5_files:
        print(f"Error: no H5 matching {pattern} in {args.patches_dir}")
        sys.exit(1)

    print(f"Processing {len(h5_files)} slide(s) at threshold {args.threshold} "
          f"({args.threshold / 255:.0%})")
    rows = []
    for h5_path in h5_files:
        slide = h5_path.stem.replace("_patches", "")
        print(slide)
        row = process_slide(h5_path, args.prob_dir / slide / MASK_NAME,
                            args.output_dir / f"{slide}_patches.h5",
                            args.threshold, args.overwrite)
        if row:
            rows.append(row)

    if rows:
        qc = args.qc_csv or args.output_dir.parent / "filter_qc.csv"
        qc.parent.mkdir(parents=True, exist_ok=True)
        with open(qc, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        empty = [r["slide"] for r in rows if r["empty"]]
        print(f"\nDone: {len(rows)}/{len(h5_files)} slides written. QC -> {qc}")
        print(f"  tumour patches: {sum(r['n_tumour'] for r in rows):,} "
              f"of {sum(r['n_tissue'] for r in rows):,} tissue patches")
        if empty:
            print(f"  EMPTY tumour mask ({len(empty)}): {', '.join(empty)}")
    else:
        print(f"\nDone: 0/{len(h5_files)} slides written.")


if __name__ == "__main__":
    main()
