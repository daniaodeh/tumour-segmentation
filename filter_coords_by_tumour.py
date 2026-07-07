"""
Flag patch coordinates from TRIDENT H5 files that fall inside tumour GeoJSON regions.

For each slide, reads the existing patches H5 and the tumour GeoJSON, then writes
a new H5 with:
  - coords         : original coords (N, 2) — unchanged
  - in_tumour      : boolean array (N,) — True if patch centre is inside tumour
  - tumour_coords  : filtered subset of coords inside tumour only

Usage:
  python filter_coords_by_tumour.py \
    --patches_dir trident-lung/output/lung_0.4_remove_pen_partial/40x_512px_0px_overlap/patches \
    --geojson_dir automatic-tumour-segmentation-in-WSIs/geojson \
    --output_dir tumour_coords/

  # Single slide
  python filter_coords_by_tumour.py ... --slide 201726608-1-5-1_1041097
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from shapely.geometry import shape, Point
from shapely.ops import unary_union
from shapely.prepared import prep


def load_tumour_region(geojson_path: Path):
    gj = json.loads(geojson_path.read_text())
    if not gj["features"]:
        return None
    polygons = [shape(f["geometry"]) for f in gj["features"]]
    return prep(unary_union(polygons))


def flag_coords(coords: np.ndarray, patch_size: int, tumour_region) -> np.ndarray:
    half = patch_size // 2
    centres = coords + half  # (N, 2) — patch centres in full-res pixels
    return np.array([tumour_region.contains(Point(int(x), int(y))) for x, y in centres],
                    dtype=bool)


def process_slide(h5_path: Path, geojson_path: Path, out_path: Path, overwrite: bool) -> bool:
    if out_path.exists() and not overwrite:
        print(f"  [SKIP] exists (--overwrite to redo): {out_path.name}")
        return True

    if not geojson_path.exists():
        print(f"  [SKIP] tumour GeoJSON not found: {geojson_path}")
        return False

    tumour_region = load_tumour_region(geojson_path)
    if tumour_region is None:
        print(f"  [SKIP] GeoJSON has no features (empty tumour mask): {geojson_path.parent.name}")
        return False

    with h5py.File(h5_path, "r") as f:
        coords     = f["coords"][...]
        attrs      = dict(f["coords"].attrs)
        patch_size = int(attrs.get("patch_size", 512))

    print(f"  Flagging {len(coords):,} patches (patch_size={patch_size})...")
    in_tumour = flag_coords(coords, patch_size, tumour_region)
    tumour_coords = coords[in_tumour]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        # trident-compatible: coords = tumour patches only
        ds = f.create_dataset("coords", data=tumour_coords)
        for k, v in attrs.items():
            ds.attrs[k] = v
        # extras for reference
        f.create_dataset("all_coords", data=coords)
        f.create_dataset("in_tumour", data=in_tumour)

    n_total   = len(coords)
    n_tumour  = in_tumour.sum()
    pct       = 100 * n_tumour / n_total if n_total else 0
    print(f"  [OK]   {n_tumour:,}/{n_total:,} patches in tumour ({pct:.1f}%) → {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Filter TRIDENT patch coords by tumour GeoJSON")
    parser.add_argument("--patches_dir", type=Path, required=True,
                        help="Directory containing *_patches.h5 files from TRIDENT")
    parser.add_argument("--geojson_dir", type=Path, required=True,
                        help="Directory containing <slide>/segmentation.geojson files")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Where to write filtered H5 files")
    parser.add_argument("--slide", type=str, default=None,
                        help="Process a single slide by name")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files")
    args = parser.parse_args()

    if not args.patches_dir.is_dir():
        print(f"Error: patches_dir not found: {args.patches_dir}")
        sys.exit(1)
    if not args.geojson_dir.is_dir():
        print(f"Error: geojson_dir not found: {args.geojson_dir}")
        sys.exit(1)

    if args.slide:
        h5_files = list(args.patches_dir.glob(f"{args.slide}_patches.h5"))
        if not h5_files:
            print(f"Error: no H5 found for slide {args.slide} in {args.patches_dir}")
            sys.exit(1)
    else:
        h5_files = sorted(args.patches_dir.glob("*_patches.h5"))

    print(f"Processing {len(h5_files)} slide(s)")
    n_ok = 0
    for h5_path in h5_files:
        slide_name = h5_path.stem.replace("_patches", "")
        print(slide_name)
        geojson_path = args.geojson_dir / slide_name / "segmentation.geojson"
        out_path     = args.output_dir / f"{slide_name}_patches.h5"
        if process_slide(h5_path, geojson_path, out_path, args.overwrite):
            n_ok += 1

    print(f"\nDone: {n_ok}/{len(h5_files)} slides processed.")


if __name__ == "__main__":
    main()
