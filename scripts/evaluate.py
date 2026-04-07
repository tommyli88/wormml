#!/usr/bin/env python3
"""
Evaluate WormML models using Hungarian-matched precision/recall.

Runs inference on every validation image for one or more cameras and
reports counting metrics (MAE, RMSE, within ±1/2/4) and detection
metrics (precision, recall, F1 — both macro and micro averaged).
Outputs ASCII tables and optionally LaTeX tables for the paper.

Usage
-----
  # Single camera from YAML config
  python scripts/evaluate.py --config configs/og.yaml

  # All four cameras at once
  python scripts/evaluate.py \\
      --config configs/og.yaml \\
      --config configs/tau.yaml \\
      --config configs/lb.yaml \\
      --config configs/uva.yaml \\
      --latex

  # Ad-hoc evaluation without a YAML
  python scripts/evaluate.py \\
      --camera OG \\
      --model  /runs/og/yolov11_maxacc_11l/weights/best.pt \\
      --images /data/og/preprocessed/images/val \\
      --labels /data/og/preprocessed/labels/val \\
      --conf 0.35 --iou 0.30
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from wormml.evaluate import EvalConfig, evaluate, print_latex_tables


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WormML evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", action="append", default=[],
                   metavar="YAML",
                   help="Camera YAML config (may be repeated for multiple cameras)")
    # Ad-hoc flags (single camera without YAML)
    p.add_argument("--camera", default=None, help="Camera name label (e.g. OG)")
    p.add_argument("--model",  default=None, help="Path to best.pt")
    p.add_argument("--images", default=None, help="Path to validation images directory")
    p.add_argument("--labels", default=None, help="Path to validation labels directory")
    p.add_argument("--conf",   type=float, default=0.25)
    p.add_argument("--iou",    type=float, default=0.45)
    p.add_argument("--latex",  action="store_true",
                   help="Print LaTeX tables after evaluation")
    p.add_argument("--show-n", type=int, default=3,
                   help="Number of per-image details to print (default: 3)")
    return p.parse_args()


def eval_config_from_yaml(yaml_path: str, show_first_n: int = 3) -> EvalConfig:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    thr = raw.get("thresholds", {})
    camera = raw.get("camera", Path(yaml_path).stem).upper()

    # Infer val dirs from dataset_base in YAML
    dataset_base = raw.get("dataset_base", "")
    images_dir = str(Path(dataset_base) / "images" / "val") if dataset_base else ""
    labels_dir = str(Path(dataset_base) / "labels" / "val") if dataset_base else ""

    # Allow explicit override keys
    images_dir = raw.get("eval_images_dir", images_dir)
    labels_dir = raw.get("eval_labels_dir", labels_dir)
    model_path  = raw.get("model_path", "")

    # Auto-discover model in output_dir if not explicitly set
    if not model_path and raw.get("output_dir"):
        candidates = list(
            Path(raw["output_dir"]).glob(
                "**/yolov11_maxacc_*/weights/best.pt"
            )
        )
        if candidates:
            model_path = str(sorted(candidates)[-1])
            print(f"  ℹ️  Auto-detected model: {model_path}")

    return EvalConfig(
        camera=camera,
        model_path=model_path,
        images_dir=images_dir,
        labels_dir=labels_dir,
        conf_thr=thr.get("conf_thr", 0.25),
        iou_thr=thr.get("iou_thr", 0.45),
        low_count_threshold=thr.get("low_count_threshold", 10),
        conf_boost=thr.get("conf_boost", 0.0),
        high_count_threshold=thr.get("high_count_threshold", 80),
        high_conf_boost=thr.get("high_conf_boost", 0.0),
        show_first_n=show_first_n,
    )


def main() -> None:
    args = parse_args()
    configs = []

    # ── Load from YAML files ──────────────────────────────────────────────────
    for yaml_path in args.config:
        configs.append(eval_config_from_yaml(yaml_path, show_first_n=args.show_n))

    # ── Ad-hoc single camera ─────────────────────────────────────────────────
    if args.camera and args.model:
        configs.append(
            EvalConfig(
                camera=args.camera,
                model_path=args.model,
                images_dir=args.images or "",
                labels_dir=args.labels or "",
                conf_thr=args.conf,
                iou_thr=args.iou,
                show_first_n=args.show_n,
            )
        )

    if not configs:
        print("❌  Provide at least one --config YAML or --camera/--model flags.")
        sys.exit(1)

    all_results = evaluate(configs)

    if args.latex and all_results:
        print_latex_tables(all_results)


if __name__ == "__main__":
    main()
