"""
Convert tumour segmentation masks to GeoJSON in full-resolution pixel coordinates.

Usage:
  # All slides
  python mask_to_geojson.py --output_dir output/

  # Single slide
  python mask_to_geojson.py --output_dir output/ --slide 201726608-1-5-1_1041097

  # Use smooth probability instead of binary segmentation
  python mask_to_geojson.py --output_dir output/ --source smooth --threshold 128
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
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


def process_slide(slide_dir: Path, geojson_root: Path, source: str,
                  threshold: int, overwrite: bool,
                  metadata_dir: Path | None = None) -> bool:
    name = slide_dir.name

    meta_root = metadata_dir / name if metadata_dir else slide_dir
    json_path = meta_root / "downsampled_scan" / f"{name}.json"
    seg_path  = slide_dir / "result_segmentation.png"
    prob_path = slide_dir / "result_probability_smooth.png"

    if not json_path.exists():
        print(f"  [SKIP] metadata JSON not found: {json_path}")
        return False

    meta = json.loads(json_path.read_text())
    scale_x = meta["lvl0_width"]  / meta["target_width"]
    scale_y = meta["lvl0_height"] / meta["target_height"]

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
                         args.overwrite, args.metadata_dir):
            n_ok += 1

    print(f"\nDone: {n_ok}/{len(slide_dirs)} slides converted.")


if __name__ == "__main__":
    main()
