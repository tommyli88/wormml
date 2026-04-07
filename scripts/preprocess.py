#!/usr/bin/env python3
"""
Preprocess a worm-plate dataset for a given camera.

Expects data already split into train/val (run scripts/split_dataset.py first).

For OG / Tau / LB: applies Gaussian blur, petri-dish Hough-circle crop +
circular mask, optional inversion (LB), optional augmentation (LB train),
then resizes to 1344×1344.

For UVA: copies images as-is into the expected folder structure (no crop needed).

Usage
-----
    # OG
    python scripts/preprocess.py \\
        --camera og \\
        --input  /data/og/split \\
        --output /data/og/preprocessed

    # Tau
    python scripts/preprocess.py \\
        --camera tau \\
        --input  /data/tau/split \\
        --output /data/tau/preprocessed

    # LB
    python scripts/preprocess.py \\
        --camera lb \\
        --input  /data/lb/split \\
        --output /data/lb/preprocessed

    # UVA
    python scripts/preprocess.py \\
        --camera uva \\
        --input  /data/uva/split \\
        --output /data/uva/preprocessed
"""

import argparse
import sys
from pathlib import Path

# Make sure wormml is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wormml.preprocess import preprocess_dataset, get_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="WormML preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--camera", required=True,
                   choices=["og", "tau", "lb", "uva"],
                   help="Camera / imaging system")
    p.add_argument("--input", required=True,
                   help="Pre-split dataset root (images/{train,val} + labels/{train,val})")
    p.add_argument("--output", required=True,
                   help="Output directory for preprocessed dataset")
    p.add_argument("--no-visualize", action="store_true",
                   help="Skip before/after visualisation plot")
    p.add_argument("--workers", type=int, default=None,
                   help="Number of parallel workers (default: from config)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_config(args.camera)

    if args.workers is not None:
        cfg.max_workers = args.workers

    success = preprocess_dataset(
        input_base_dir=args.input,
        output_base_dir=args.output,
        cfg=cfg,
        visualize=not args.no_visualize,
    )

    if success:
        print(f"\n✅ Preprocessed dataset written to: {args.output}")
    else:
        print("\n❌ Preprocessing completed with errors — check log above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
