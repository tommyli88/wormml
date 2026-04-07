#!/usr/bin/env python3
"""
Grid-search confidence and IoU thresholds to minimise MAE on validation data.

Usage
-----
  # From a YAML config (auto-discovers best.pt from output_dir)
  python scripts/tune_thresholds.py --config configs/og.yaml

  # Explicit paths
  python scripts/tune_thresholds.py \\
      --model   /runs/og/yolov11_maxacc_11l/weights/best.pt \\
      --dataset /data/og/preprocessed

  # Custom confidence grid
  python scripts/tune_thresholds.py \\
      --config configs/tau.yaml \\
      --conf-grid 0.20 0.25 0.30 0.35 0.40 \\
      --iou-grid  0.20 0.25 0.30 0.35 0.40 0.45
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from wormml.threshold import sweep_thresholds, DEFAULT_CONF_GRID, DEFAULT_IOU_GRID


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WormML threshold optimisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config",  default=None, help="Camera YAML config")
    p.add_argument("--model",   default=None, help="Path to best.pt")
    p.add_argument("--dataset", default=None, help="Dataset root (with images/val)")
    p.add_argument("--split",   default="val", help="Split to evaluate on (default: val)")
    p.add_argument("--conf-grid", nargs="+", type=float, default=None,
                   help="Confidence values to sweep")
    p.add_argument("--iou-grid",  nargs="+", type=float, default=None,
                   help="IoU values to sweep")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_path   = args.model
    dataset_path = args.dataset

    if args.config:
        with open(args.config) as f:
            raw = yaml.safe_load(f)

        dataset_path = dataset_path or raw.get("dataset_base", "")

        if not model_path:
            # Auto-discover best.pt from output_dir
            output_dir = raw.get("output_dir", "")
            if output_dir:
                candidates = list(
                    Path(output_dir).glob("**/yolov11_maxacc_*/weights/best.pt")
                )
                if candidates:
                    model_path = str(sorted(candidates)[-1])
                    print(f"ℹ️  Auto-detected model: {model_path}")

    if not model_path:
        print("❌  Provide --model or --config pointing to a trained output_dir.")
        sys.exit(1)
    if not dataset_path:
        print("❌  Provide --dataset or --config with dataset_base set.")
        sys.exit(1)

    conf_grid = args.conf_grid or DEFAULT_CONF_GRID
    iou_grid  = args.iou_grid  or DEFAULT_IOU_GRID

    best_conf, best_iou, best_mae = sweep_thresholds(
        model_path=model_path,
        dataset_path=dataset_path,
        conf_grid=conf_grid,
        iou_grid=iou_grid,
        split=args.split,
    )

    print(f"\nAdd to your config YAML:")
    print(f"  thresholds:")
    print(f"    conf_thr: {best_conf}")
    print(f"    iou_thr:  {best_iou}")
    print(f"    # MAE at these thresholds: {best_mae:.3f}")


if __name__ == "__main__":
    main()
