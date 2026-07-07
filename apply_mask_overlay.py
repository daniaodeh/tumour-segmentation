"""
Overlay smooth tumour probability mask and binary segmentation on downsampled WSI PNGs.

Outputs per slide (written next to existing results):
  overlay_smooth.png      — probability heatmap blended onto the scan
  overlay_segmentation.png — binary segmentation painted red onto the scan

Usage:
  # All slides in output dir
  python apply_mask_overlay.py --output_dir output/

  # Single slide
  python apply_mask_overlay.py --output_dir output/ --slide 201721757-1-5-1_1041038

  # Only one overlay type
  python apply_mask_overlay.py --output_dir output/ --mode smooth
  python apply_mask_overlay.py --output_dir output/ --mode segmentation
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def apply_smooth_overlay(image: np.ndarray, smooth_prob: np.ndarray, alpha: float = 0.4,
                         threshold: int = 30) -> np.ndarray:
    mask = smooth_prob
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = image.copy()
    region = mask > threshold
    blended = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    overlay[region] = blended[region]
    return overlay


def apply_segmentation_overlay(image: np.ndarray, seg: np.ndarray,
                                color_bgr: tuple = (0, 0, 255), alpha: float = 0.4) -> np.ndarray:
    mask = seg
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    overlay = image.copy()
    region = mask > 127
    color_layer = np.full_like(image, color_bgr, dtype=np.uint8)
    blended = cv2.addWeighted(image, 1 - alpha, color_layer, alpha, 0)
    overlay[region] = blended[region]
    return overlay


def process_slide(slide_dir: Path, overlays_root: Path, mode: str, alpha: float, overwrite: bool) -> bool:
    name = slide_dir.name

    scan_png     = slide_dir / "downsampled_scan" / f"{name}.png"
    smooth_png   = slide_dir / "result_probability_smooth.png"
    seg_png      = slide_dir / "result_segmentation.png"

    if not scan_png.exists():
        print(f"  [SKIP] downsampled scan not found: {scan_png}")
        return False

    image = cv2.imread(str(scan_png))
    if image is None:
        print(f"  [SKIP] could not read: {scan_png}")
        return False

    # Write overlays to a separate writable directory
    out_dir = overlays_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = True

    if mode in ("smooth", "both"):
        out = out_dir / "overlay_smooth.png"
        if out.exists() and not overwrite:
            print(f"  [SKIP] exists (--overwrite to redo): {out.name}")
        elif not smooth_png.exists():
            print(f"  [SKIP] smooth probability not found: {smooth_png}")
            ok = False
        else:
            prob = cv2.imread(str(smooth_png), cv2.IMREAD_GRAYSCALE)
            overlay = apply_smooth_overlay(image, prob, alpha=alpha)
            if cv2.imwrite(str(out), overlay):
                print(f"  [OK]   {out}")
            else:
                print(f"  [FAIL] could not write {out}")
                ok = False

            # Side-by-side: original | overlay
            out_side = out_dir / "overlay_smooth_sidebyside.png"
            prob_color = cv2.applyColorMap(
                cv2.resize(prob, (image.shape[1], image.shape[0])), cv2.COLORMAP_JET
            )
            side = np.concatenate([image, overlay, prob_color], axis=1)
            if cv2.imwrite(str(out_side), side):
                print(f"  [OK]   {out_side.name}")
            else:
                print(f"  [FAIL] could not write {out_side}")
                ok = False

    if mode in ("segmentation", "both"):
        out = out_dir / "overlay_segmentation.png"
        if out.exists() and not overwrite:
            print(f"  [SKIP] exists (--overwrite to redo): {out.name}")
        elif not seg_png.exists():
            print(f"  [SKIP] segmentation not found: {seg_png}")
            ok = False
        else:
            seg = cv2.imread(str(seg_png), cv2.IMREAD_GRAYSCALE)
            result = apply_segmentation_overlay(image, seg, alpha=alpha)
            if cv2.imwrite(str(out), result):
                print(f"  [OK]   {out}")
            else:
                print(f"  [FAIL] could not write {out}")
                ok = False

    return ok


def main():
    parser = argparse.ArgumentParser(description="Overlay tumour masks on downsampled WSI PNGs")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Root output directory (contains one folder per slide)")
    parser.add_argument("--overlays_dir", type=Path, default=None,
                        help="Directory to write overlay PNGs (default: <output_dir>/../overlays)")
    parser.add_argument("--slide", type=str, default=None,
                        help="Process a single slide by name (folder stem). Default: all slides.")
    parser.add_argument("--mode", choices=["smooth", "segmentation", "both"], default="both",
                        help="Which overlay to produce (default: both)")
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="Overlay opacity 0–1 (default: 0.4)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing overlays")
    args = parser.parse_args()

    if not args.output_dir.is_dir():
        print(f"Error: output_dir does not exist: {args.output_dir}")
        sys.exit(1)

    overlays_root = args.overlays_dir if args.overlays_dir else args.output_dir.parent / "overlays"
    overlays_root.mkdir(parents=True, exist_ok=True)
    print(f"Overlays will be saved to: {overlays_root}")

    if args.slide:
        slide_dirs = [args.output_dir / args.slide]
        if not slide_dirs[0].is_dir():
            print(f"Error: slide folder not found: {slide_dirs[0]}")
            sys.exit(1)
    else:
        slide_dirs = sorted([d for d in args.output_dir.iterdir() if d.is_dir()])

    print(f"Processing {len(slide_dirs)} slide(s) — mode={args.mode}, alpha={args.alpha}")
    n_ok = 0
    for slide_dir in slide_dirs:
        print(f"{slide_dir.name}")
        if process_slide(slide_dir, overlays_root, args.mode, args.alpha, args.overwrite):
            n_ok += 1

    print(f"\nDone: {n_ok}/{len(slide_dirs)} slides processed successfully.")


if __name__ == "__main__":
    main()
