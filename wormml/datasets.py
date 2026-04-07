"""
Dataset utilities: validation, statistics, YAML creation, and k-fold splitting.
"""

from __future__ import annotations

import glob
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


# ============================================================
# DATASET VALIDATION
# ============================================================

def check_dataset(dataset_path: str) -> Tuple[int, int]:
    """
    Verify that a pre-split YOLO dataset exists and report counts.

    Parameters
    ----------
    dataset_path : str
        Root directory with images/{train,val} and labels/{train,val}.

    Returns
    -------
    (n_train_images, n_val_images)
    """
    train_imgs = glob.glob(os.path.join(dataset_path, "images/train", "*.*"))
    val_imgs = glob.glob(os.path.join(dataset_path, "images/val", "*.*"))
    train_lbls = glob.glob(os.path.join(dataset_path, "labels/train", "*.txt"))
    val_lbls = glob.glob(os.path.join(dataset_path, "labels/val", "*.txt"))

    print("=" * 60)
    print("Dataset Verification")
    print("=" * 60)
    print(f"  Path:   {dataset_path}")
    print(f"  Train:  {len(train_imgs)} images, {len(train_lbls)} labels")
    print(f"  Val:    {len(val_imgs)} images, {len(val_lbls)} labels")
    print("=" * 60)

    if len(train_imgs) == 0:
        print("❌ ERROR: No training images found!")
    if len(val_imgs) == 0:
        print("⚠️  WARNING: No validation images found!")

    return len(train_imgs), len(val_imgs)


def dataset_stats(dataset_path: str) -> Tuple[List, List, List]:
    """
    Compute annotation statistics (area, aspect ratio, worm count per image).

    Returns
    -------
    (areas, aspect_ratios, counts_per_image)
    """
    areas: List[float] = []
    ratios: List[float] = []
    counts: List[int] = []

    for split in ("train", "val"):
        lbl_dir = os.path.join(dataset_path, "labels", split)
        if not os.path.isdir(lbl_dir):
            continue
        for txt in glob.glob(os.path.join(lbl_dir, "*.txt")):
            c = 0
            with open(txt) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 5:
                        w, h = map(float, parts[3:5])
                        areas.append(w * h)
                        if min(w, h) > 0:
                            ratios.append(max(w, h) / min(w, h))
                        c += 1
            counts.append(c)

    if areas:
        print(f"\nDataset Statistics:")
        print(f"  Total annotations : {len(areas)}")
        print(f"  Images analysed   : {len(counts)}")
        print(f"  Worms per image   : {np.mean(counts):.2f} avg  "
              f"[{min(counts)} – {max(counts)}]")
    else:
        print("⚠️  WARNING: No annotations found.")

    return areas, ratios, counts


# ============================================================
# YAML CREATION
# ============================================================

def create_data_yaml(
    dataset_path: str,
    classes: Optional[List[str]] = None,
    extra: Optional[dict] = None,
) -> str:
    """
    Write a YOLO data.yaml for *dataset_path* and return its path.

    Parameters
    ----------
    dataset_path : str
        Root of the dataset (must contain images/{train,val}).
    classes : list[str], optional
        Class names.  Defaults to ``["worm"]``.
    extra : dict, optional
        Additional keys merged into the YAML (e.g. augmentation flags).
    """
    classes = classes or ["worm"]
    cfg: Dict = {
        "path": dataset_path,
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": classes,
    }
    if extra:
        cfg.update(extra)

    yaml_path = os.path.join(dataset_path, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)

    print(f"✅ data.yaml written: {yaml_path}")
    return yaml_path


# ============================================================
# K-FOLD UTILITIES
# ============================================================

def get_all_image_paths(dataset_path: str) -> List[str]:
    """Collect all image paths from both train and val splits."""
    IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    paths: List[str] = []
    for split in ("train", "val"):
        img_dir = os.path.join(dataset_path, "images", split)
        if os.path.isdir(img_dir):
            for ext in IMAGE_EXTS:
                paths.extend(glob.glob(os.path.join(img_dir, ext)))
                paths.extend(glob.glob(os.path.join(img_dir, ext.upper())))
    return sorted(set(paths))


def find_label_for_image(img_path: str, dataset_path: str) -> Optional[str]:
    """Return the label .txt path that corresponds to *img_path*, or None."""
    stem = Path(img_path).stem
    for split in ("train", "val"):
        lbl = os.path.join(dataset_path, "labels", split, f"{stem}.txt")
        if os.path.exists(lbl):
            return lbl
    return None


def create_fold_dataset(
    fold_idx: int,
    train_images: List[str],
    val_images: List[str],
    source_dataset: str,
    output_base: str,
) -> str:
    """
    Create fold directory structure with symlinked or copied files.

    Returns the fold dataset root path.
    """
    fold_root = os.path.join(output_base, f"fold_{fold_idx}")

    for split, images in [("train", train_images), ("val", val_images)]:
        img_out = os.path.join(fold_root, "images", split)
        lbl_out = os.path.join(fold_root, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for img_path in images:
            dst_img = os.path.join(img_out, Path(img_path).name)
            if not os.path.exists(dst_img):
                shutil.copy2(img_path, dst_img)

            lbl_path = find_label_for_image(img_path, source_dataset)
            if lbl_path:
                dst_lbl = os.path.join(lbl_out, Path(lbl_path).name)
                if not os.path.exists(dst_lbl):
                    shutil.copy2(lbl_path, dst_lbl)

    return fold_root


def setup_kfold_datasets(
    dataset_path: str,
    output_base: str,
    k: int = 5,
    seed: int = 42,
) -> List[str]:
    """
    Create k stratified dataset folds from *dataset_path*.

    Returns
    -------
    list of fold root paths (length k)
    """
    all_images = get_all_image_paths(dataset_path)
    random.seed(seed)
    random.shuffle(all_images)

    fold_size = len(all_images) // k
    folds = [all_images[i * fold_size: (i + 1) * fold_size] for i in range(k)]
    # Append remainder to last fold
    remainder = all_images[k * fold_size:]
    folds[-1].extend(remainder)

    fold_roots: List[str] = []
    for fold_idx in range(k):
        val_images = folds[fold_idx]
        train_images = [img for i, f in enumerate(folds) if i != fold_idx for img in f]
        fold_root = create_fold_dataset(
            fold_idx, train_images, val_images, dataset_path, output_base
        )
        fold_roots.append(fold_root)
        print(f"  Fold {fold_idx}: {len(train_images)} train / {len(val_images)} val → {fold_root}")

    return fold_roots
