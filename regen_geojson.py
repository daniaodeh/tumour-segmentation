"""Regenerate tumour GeoJSONs locally at a chosen threshold.

mask_to_geojson.py derives its pixel scale from <slide>/downsampled_scan/
<slide>.json, which lives on the cluster and was never synced - only the three
result PNGs were. The same scale is recoverable here: scale = level0_width /
mask_width, with level0_width read from the slide's TRIDENT patches H5. This
script supplies that scale and calls mask_to_geojson.py's own contour function,
so the contour logic is shared rather than reimplemented.

  python regen_geojson.py --threshold 31  --out <dir>
  python regen_geojson.py --threshold 127 --validate <dir-built-at-127>

--validate regenerates at the given threshold and diffs against a directory
built by mask_to_geojson.py on the cluster instead of writing. At 127 against
the cluster's own 127 output every slide should come back identical, which is
what proves the derived scale equals the JSON-derived one.
"""
import argparse, json
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]      # projects/lung

from mask_to_geojson import mask_to_geojson   # same directory

OUT = ROOT / "experiments/segmentation/output"
PATCH = ROOT / ("experiments/classification/output/lung_0.4_remove_pen_partial_20/"
                "20x_512px_0px_overlap/patches")


def scale_for(slide, mask_shape):
    h5 = PATCH / f"{slide}_patches.h5"
    if not h5.exists():
        return None
    with h5py.File(h5) as f:
        a = dict(f["coords"].attrs)
    H, W = mask_shape
    return int(a["level0_width"]) / W, int(a["level0_height"]) / H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=int, required=True)
    p.add_argument("--out", type=Path, help="where to write (omit with --validate)")
    p.add_argument("--validate", type=Path, default=None,
                   help="diff against this geojson directory instead of writing")
    a = p.parse_args()
    if not a.validate and not a.out:
        p.error("one of --out or --validate is required")

    dirs = sorted(d for d in OUT.iterdir() if (d / "result_probability_smooth.png").exists())
    print(f"{len(dirs)} slides, threshold {a.threshold} ({a.threshold/255:.0%})")
    same = diff = missing = written = 0
    feats = 0
    for i, d in enumerate(dirs, 1):
        mask = cv2.imread(str(d / "result_probability_smooth.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  [SKIP] unreadable: {d.name}"); continue
        sc = scale_for(d.name, mask.shape)
        if sc is None:
            print(f"  [SKIP] no patches H5: {d.name}"); missing += 1; continue
        gj = mask_to_geojson((mask > a.threshold).astype(np.uint8) * 255, *sc)
        feats += len(gj["features"])
        if a.validate:
            s = a.validate / d.name / "segmentation.geojson"
            if not s.exists():
                missing += 1; continue
            old = json.loads(s.read_text())
            if json.dumps(old, sort_keys=True) == json.dumps(gj, sort_keys=True):
                same += 1
            else:
                diff += 1
                if diff <= 5:
                    print(f"  DIFF {d.name}: stored {len(old['features'])} feats, "
                          f"regen {len(gj['features'])}")
        else:
            o = a.out / d.name
            o.mkdir(parents=True, exist_ok=True)
            (o / "segmentation.geojson").write_text(json.dumps(gj))
            written += 1
        if i % 50 == 0:
            print(f"  {i}/{len(dirs)}", flush=True)
    if a.validate:
        print(f"\nidentical {same}   different {diff}   no stored/H5 {missing}")
    else:
        print(f"\nwrote {written} geojsons -> {a.out}   total features {feats:,}")


if __name__ == "__main__":
    main()
