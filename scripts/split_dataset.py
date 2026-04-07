#!/usr/bin/env python3
"""
Split a flat worm-plate dataset into YOLO train / val folders.

Run this as Step 0 for every camera before preprocessing.

All four cameras — OG, Tau, LB, and UVA — start with a single flat folder
of images and labels and use this script identically.

The split is always deterministic: seed 42, 80 % train / 20 % val.
Every image must have a matching .txt label file; unmatched images are skipped.

Expected input layout (flat, not yet split)
-------------------------------------------
    raw_data/
    ├── images/
    │   ├── plate_001.jpg
    │   ├── plate_002.jpg
    │   └── ...
    └── labels/
        ├── plate_001.txt   ← same stem as the image
        ├── plate_002.txt
        └── ...

Output layout (YOLO-ready, ready for preprocessing)
----------------------------------------------------
    split_data/
    ├── images/
    │   ├── train/
    │   │   ├── plate_001.jpg
    │   │   └── ...
    │   └── val/
    │       ├── plate_002.jpg
    │       └── ...
    └── labels/
        ├── train/
        │   ├── plate_001.txt
        │   └── ...
        └── val/
            ├── plate_002.txt
            └── ...

Usage
-----
    python scripts/split_dataset.py \\
        --input  /data/tau/raw \\
        --output /data/tau/split

    # Custom ratio or seed  (defaults: ratio=0.8, seed=42)
    python scripts/split_dataset.py \\
        --input  /data/og/raw \\
        --output /data/og/split \\
        --train-ratio 0.8 \\
        --seed 42
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def split_dataset(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> None:
    src = Path(input_dir)
    dst = Path(output_dir)

    img_src = src / "images"
    lbl_src = src / "labels"

    if not img_src.exists():
        print(f"❌  images/ folder not found in: {src}")
        sys.exit(1)
    if not lbl_src.exists():
        print(f"❌  labels/ folder not found in: {src}")
        sys.exit(1)

    # Collect matched pairs only
    all_images = sorted(
        f for f in img_src.iterdir()
        if f.suffix.lower() in IMAGE_EXTS
    )
    pairs = [f for f in all_images if (lbl_src / f"{f.stem}.txt").exists()]
    skipped = len(all_images) - len(pairs)

    if not pairs:
        print("❌  No matched image/label pairs found.")
        sys.exit(1)

    # Deterministic shuffle then split
    random.seed(seed)
    random.shuffle(pairs)
    cut = int(len(pairs) * train_ratio)
    splits = {"train": pairs[:cut], "val": pairs[cut:]}

    print(f"Input     : {src}")
    print(f"Total     : {len(pairs)} matched pairs  ({skipped} images skipped — no label)")
    print(f"Seed      : {seed}")
    print(f"Ratio     : {train_ratio:.0%} train / {1 - train_ratio:.0%} val")
    print(f"Train     : {len(splits['train'])} images")
    print(f"Val       : {len(splits['val'])} images")
    print(f"Output    : {dst}")
    print()

    for split, images in splits.items():
        img_out = dst / "images" / split
        lbl_out = dst / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in images:
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(lbl_src / f"{img_path.stem}.txt",
                         lbl_out / f"{img_path.stem}.txt")

        print(f"  ✅ {split}: {len(images)} images + labels → {img_out}")

    print(f"\n✅ Split complete → {dst}")
    print("Next step: run scripts/preprocess.py on the split output.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split a flat dataset into YOLO train/val folders (seed=42, 80/20)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",  required=True,
                   help="Flat dataset root (images/ + labels/)")
    p.add_argument("--output", required=True,
                   help="Output root for train/val split")
    p.add_argument("--train-ratio", type=float, default=0.8,
                   help="Fraction of data for training (default: 0.8)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible split (default: 42)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_dataset(args.input, args.output, args.train_ratio, args.seed)
