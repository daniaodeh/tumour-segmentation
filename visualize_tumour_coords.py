"""
Generate patch visualizations for tumour-filtered coords H5 files.

Usage:
  python visualize_tumour_coords.py \
    --patches_dir trident-lung/output/lung_tumour_coords/tumour_40x_512px/patches \
    --wsi_dir data/lung_partial \
    --output_dir trident-lung/output/lung_tumour_coords/tumour_40x_512px/visualization

  # Single slide
  python visualize_tumour_coords.py ... --slide 201726608-1-5-1_1041097
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trident-lung"))
from trident import load_wsi


def find_wsi(wsi_dir: Path, stem: str) -> Path | None:
    for ext in (".svs", ".tif", ".tiff", ".ndpi", ".scn"):
        p = wsi_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Visualize tumour patch coords on WSI thumbnails")
    parser.add_argument("--patches_dir", type=Path, required=True,
                        help="Directory containing *_patches.h5 files")
    parser.add_argument("--wsi_dir", type=Path, required=True,
                        help="Directory containing WSI files")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory to save visualization JPGs")
    parser.add_argument("--slide", type=str, default=None,
                        help="Process a single slide by name")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.patches_dir.is_dir():
        print(f"Error: patches_dir not found: {args.patches_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.slide:
        h5_files = list(args.patches_dir.glob(f"{args.slide}_patches.h5"))
    else:
        h5_files = sorted(args.patches_dir.glob("*_patches.h5"))

    print(f"Generating visualizations for {len(h5_files)} slide(s)")
    n_ok = 0
    for h5_path in h5_files:
        slide_name = h5_path.stem.replace("_patches", "")
        out_jpg = args.output_dir / f"{slide_name}.jpg"

        if out_jpg.exists() and not args.overwrite:
            print(f"  [SKIP] {slide_name} (--overwrite to redo)")
            continue

        wsi_path = find_wsi(args.wsi_dir, slide_name)
        if wsi_path is None:
            print(f"  [SKIP] WSI not found for {slide_name}")
            continue

        print(f"  {slide_name} ...", end=" ", flush=True)
        try:
            with load_wsi(str(wsi_path), lazy_init=False) as wsi:
                wsi.visualize_coords(
                    coords_path=str(h5_path),
                    save_patch_viz=str(args.output_dir),
                )
            print("OK")
            n_ok += 1
        except Exception as e:
            print(f"FAIL — {e}")

    print(f"\nDone: {n_ok}/{len(h5_files)} visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
