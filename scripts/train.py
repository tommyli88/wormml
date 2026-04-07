#!/usr/bin/env python3
"""
Train a YOLOv11 worm-detection model for a given camera.

Usage
-----
  # Train using a config YAML (recommended)
  python scripts/train.py --config configs/og.yaml

  # Override specific fields
  python scripts/train.py --config configs/tau.yaml \\
      --dataset-base /data/tau/preprocessed \\
      --output-dir   /runs/tau_experiment \\
      --epochs 50

  # Quick-start with minimal arguments
  python scripts/train.py \\
      --camera og \\
      --dataset-base /data/og/preprocessed \\
      --output-dir   /runs/og
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from wormml.train import TrainingConfig, train


def load_yaml_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WormML training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default=None,
                   help="Path to a camera YAML config (e.g. configs/og.yaml)")
    p.add_argument("--camera", default=None,
                   choices=["og", "tau", "lb", "uva"],
                   help="Camera profile (overrides config)")
    p.add_argument("--dataset-base", default=None,
                   help="Path to preprocessed dataset root")
    p.add_argument("--output-dir", default=None,
                   help="Where to write training outputs")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--no-progressive-resize", action="store_true",
                   help="Skip optional progressive resizing phases")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Build config ─────────────────────────────────────────────────────────
    if args.config:
        raw = load_yaml_config(args.config)
        training_raw = raw.get("training", {})
        cfg = TrainingConfig(
            camera=raw.get("camera", "og"),
            dataset_base=raw.get("dataset_base", ""),
            output_dir=raw.get("output_dir", ""),
            epochs=training_raw.get("epochs", 100),
            patience=training_raw.get("patience", 25),
            img_size=training_raw.get("img_size", 1344),
            batch_size=training_raw.get("batch_size", 6),
            workers=training_raw.get("workers", 8),
            warmup_epochs=training_raw.get("warmup_epochs", 3),
            mosaic=training_raw.get("mosaic", 1.0),
            mixup=training_raw.get("mixup", 0.3),
            copy_paste=training_raw.get("copy_paste", 0.1),
            degrees=training_raw.get("degrees", 15.0),
            translate=training_raw.get("translate", 0.1),
            scale=training_raw.get("scale", 0.5),
            shear=training_raw.get("shear", 2.0),
            perspective=training_raw.get("perspective", 0.0003),
            flipud=training_raw.get("flipud", 0.5),
            fliplr=training_raw.get("fliplr", 0.5),
            optimizer=training_raw.get("optimizer", "AdamW"),
            lr0=training_raw.get("lr0", 0.001),
            lrf=training_raw.get("lrf", 0.01),
            momentum=training_raw.get("momentum", 0.937),
            weight_decay=training_raw.get("weight_decay", 0.0005),
            cls_loss=training_raw.get("cls_loss", 0.5),
            box_loss=training_raw.get("box_loss", 9.0),
            dfl_loss=training_raw.get("dfl_loss", 1.5),
            rect=training_raw.get("rect", False),
            single_cls=training_raw.get("single_cls", False),
            amp=training_raw.get("amp", True),
            seed=training_raw.get("seed", 42),
            model_types=training_raw.get("model_types", ["11l"]),
        )
    else:
        cfg = TrainingConfig(camera=args.camera or "og")
        cfg.apply_camera_defaults()

    # ── CLI overrides ─────────────────────────────────────────────────────────
    if args.camera:
        cfg.camera = args.camera
    if args.dataset_base:
        cfg.dataset_base = args.dataset_base
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch is not None:
        cfg.batch_size = args.batch
    if args.workers is not None:
        cfg.workers = args.workers

    # ── Validate ──────────────────────────────────────────────────────────────
    if not cfg.dataset_base:
        print("❌  --dataset-base is required (or set dataset_base in the YAML).")
        sys.exit(1)
    if not cfg.output_dir:
        print("❌  --output-dir is required (or set output_dir in the YAML).")
        sys.exit(1)

    # ── Run training ──────────────────────────────────────────────────────────
    # apply_defaults=False because we already applied them above (either from
    # YAML with per-camera values, or via apply_camera_defaults() for bare CLI)
    train(cfg, apply_defaults=False)


if __name__ == "__main__":
    main()
