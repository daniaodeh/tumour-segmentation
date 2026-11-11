"""
Convert tumour segmentation masks to GeoJSON in full-resolution pixel coordinates.

Usage:
  # All slides
  python mask_to_geojson.py --output_dir output/

  # Single slide
  python mask_to_geojson.py --output_dir output/ --slide <slide-stem>

  # Use smooth probability instead of binary segmentation
  python mask_to_geojson.py --output_dir output/ --source smooth --threshold 31

Scale: the pixel scale normally comes from <slide>/downsampled_scan/<slide>.json
(lvl0_width / target_width). Where only the result PNGs were copied off the
cluster and that JSON is absent, pass --patches_dir and the same scale is taken
from the slide's TRIDENT patches H5 instead: the PNGs are written at exactly
target_width x target_height, so level0_width / mask_width is the same number.
Checked on all 330 slides at threshold 127 - every GeoJSON came out identical
to the one built from the JSON.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np


def mask_to_geojson(mask: np.ndarray, scale_x: float, scale_y: float) -> dict:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None or len(contours) == 0:
        return {"type": "FeatureCollection", "features": []}

    features = []
    for i, contour in enumerate(contours):
        if contour.shape[0] < 3:
            continue
        coords = (contour[:, 0, :].astype(float) * [scale_x, scale_y]).tolist()
        coords.append(coords[0])  # close the ring
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {"contour_idx": i},
        })

    return {"type": "FeatureCollection", "features": features}


def scale_from_patches(patches_dir: Path, name: str, mask_shape) -> tuple | None:
    """Pixel scale from the slide's patches H5, for when the JSON is absent."""
    h5 = patches_dir / f"{name}_patches.h5"
    if not h5.exists():
        return None
    with h5py.File(h5) as f:
        a = dict(f["coords"].attrs)
    H, W = mask_shape
    return int(a["level0_width"]) / W, int(a["level0_height"]) / H


def process_slide(slide_dir: Path, geojson_root: Path, source: str,
                  threshold: int, overwrite: bool,
                  metadata_dir: Path | None = None,
                  patches_dir: Path | None = None) -> bool:
    name = slide_dir.name

    meta_root = metadata_dir / name if metadata_dir else slide_dir
    json_path = meta_root / "downsampled_scan" / f"{name}.json"
    seg_path  = slide_dir / "result_segmentation.png"
    prob_path = slide_dir / "result_probability_smooth.png"

    out_dir = geojson_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "segmentation.geojson"

    if out.exists() and not overwrite:
        print(f"  [SKIP] exists (--overwrite to redo): {out}")
        return True

    src_path = seg_path if source == "segmentation" else prob_path
    if not src_path.exists():
        print(f"  [SKIP] mask not found: {src_path}")
        return False

    mask = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"  [SKIP] could not read: {src_path}")
        return False

    if json_path.exists():
        meta = json.loads(json_path.read_text())
        scale_x = meta["lvl0_width"]  / meta["target_width"]
        scale_y = meta["lvl0_height"] / meta["target_height"]
    elif patches_dir is not None:
        sc = scale_from_patches(patches_dir, name, mask.shape)
        if sc is None:
            print(f"  [SKIP] no JSON and no patches H5 for scale: {name}")
            return False
        scale_x, scale_y = sc
    else:
        print(f"  [SKIP] metadata JSON not found: {json_path}  "
              f"(pass --patches_dir to derive the scale instead)")
        return False

    binary = (mask > threshold).astype(np.uint8) * 255
    geojson = mask_to_geojson(binary, scale_x, scale_y)

    out.write_text(json.dumps(geojson))
    n = len(geojson["features"])
    print(f"  [OK]   {out}  ({n} contour{'s' if n != 1 else ''})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert tumour masks to GeoJSON")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Root output directory (one folder per slide)")
    parser.add_argument("--geojson_dir", type=Path, default=None,
                        help="Where to write GeoJSON files (default: <output_dir>/../geojson)")
    parser.add_argument("--slide", type=str, default=None,
                        help="Process a single slide by folder name")
    parser.add_argument("--source", choices=["segmentation", "smooth"], default="segmentation",
                        help="Which mask to convert (default: segmentation)")
    parser.add_argument("--threshold", type=int, default=127,
                        help="Binarisation threshold for smooth probability (default: 127)")
    parser.add_argument("--patches_dir", type=Path, default=None,
                        help="TRIDENT patches directory; used to derive the pixel scale "
                             "for slides whose downsampled_scan JSON is not available")
    parser.add_argument("--metadata_dir", type=Path, default=None,
                        help="Directory containing the original pipeline output with JSON metadata "
                             "(default: same as --output_dir)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing GeoJSON files")
    args = parser.parse_args()

    if not args.output_dir.is_dir():
        print(f"Error: output_dir not found: {args.output_dir}")
        sys.exit(1)

    geojson_root = args.geojson_dir or args.output_dir.parent / "geojson"
    geojson_root.mkdir(parents=True, exist_ok=True)
    print(f"GeoJSON files will be saved to: {geojson_root}")

    slide_dirs = ([args.output_dir / args.slide] if args.slide
                  else sorted(d for d in args.output_dir.iterdir() if d.is_dir()))

    print(f"Processing {len(slide_dirs)} slide(s)")
    n_ok = 0
    for slide_dir in slide_dirs:
        print(slide_dir.name)
        if process_slide(slide_dir, geojson_root, args.source, args.threshold,
                         args.overwrite, args.metadata_dir, args.patches_dir):
            n_ok += 1

    print(f"\nDone: {n_ok}/{len(slide_dirs)} slides converted.")


if __name__ == "__main__":
    main()
