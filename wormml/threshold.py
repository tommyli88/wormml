"""
Confidence and IoU threshold optimisation.

``sweep_thresholds`` searches a grid of (confidence, IoU) pairs and returns
the combination that minimises mean absolute error (MAE) on the validation set.
It is intentionally camera-agnostic; the per-camera defaults are stored in the
YAML configs and read by the evaluation script.

Usage
-----
    from wormml.threshold import sweep_thresholds

    best_conf, best_iou, best_mae = sweep_thresholds(
        model_path   = "runs/og/yolov11_maxacc_11l/weights/best.pt",
        dataset_path = "data/og_preprocessed",
    )
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# ============================================================
# GRID DEFAULTS
# ============================================================

DEFAULT_CONF_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
DEFAULT_IOU_GRID  = [0.25, 0.30, 0.35, 0.40, 0.45]


# ============================================================
# HELPERS
# ============================================================

def _count_labels(label_path: str) -> int:
    """Return number of annotated objects in a YOLO .txt label file."""
    if not os.path.exists(label_path):
        return 0
    try:
        with open(label_path) as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def _predict_count(model, img_path: str, conf: float, iou: float) -> int:
    """Run YOLO inference and return the number of predicted boxes."""
    results = model.predict(img_path, conf=conf, iou=iou, verbose=False)
    if results and results[0].boxes is not None:
        return len(results[0].boxes)
    return 0


# ============================================================
# MAIN SWEEP
# ============================================================

def sweep_thresholds(
    model_path: str,
    dataset_path: str,
    conf_grid: Optional[List[float]] = None,
    iou_grid: Optional[List[float]] = None,
    split: str = "val",
    verbose: bool = True,
) -> Tuple[float, float, float]:
    """
    Grid-search (conf, IoU) thresholds to minimise MAE on validation images.

    Parameters
    ----------
    model_path : str
        Path to a YOLO best.pt checkpoint.
    dataset_path : str
        Dataset root containing images/{split} and labels/{split}.
    conf_grid : list[float], optional
        Confidence values to test.  Defaults to ``DEFAULT_CONF_GRID``.
    iou_grid : list[float], optional
        IoU values to test.  Defaults to ``DEFAULT_IOU_GRID``.
    split : str
        Dataset split to evaluate on (``"val"`` or ``"train"``).
    verbose : bool
        Print results for each (conf, iou) combination.

    Returns
    -------
    (best_conf, best_iou, best_mae)
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required.") from exc

    conf_grid = conf_grid or DEFAULT_CONF_GRID
    iou_grid = iou_grid or DEFAULT_IOU_GRID

    print(f"\n{'='*60}")
    print("Threshold Optimisation")
    print(f"{'='*60}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    model = YOLO(model_path)

    IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    val_imgs: List[str] = []
    for ext in IMAGE_EXTS:
        val_imgs.extend(glob.glob(os.path.join(dataset_path, "images", split, ext)))
        val_imgs.extend(glob.glob(os.path.join(dataset_path, "images", split, ext.upper())))
    val_imgs = sorted(set(val_imgs))

    lbl_dir = os.path.join(dataset_path, "labels", split)

    if not val_imgs:
        print(f"⚠️  No {split} images found — skipping threshold sweep.")
        return 0.25, 0.45, float("inf")

    print(f"Evaluating on {len(val_imgs)} {split} images …")

    best = (0.25, 0.45, float("inf"))
    results_table: List[Tuple[float, float, float]] = []

    for conf in conf_grid:
        for iou in iou_grid:
            errs: List[float] = []
            for img_path in val_imgs:
                stem = Path(img_path).stem
                lbl_path = os.path.join(lbl_dir, f"{stem}.txt")
                true_count = _count_labels(lbl_path)
                pred_count = _predict_count(model, img_path, conf, iou)
                errs.append(abs(pred_count - true_count))

            mae = float(np.mean(errs))
            results_table.append((conf, iou, mae))

            if verbose:
                marker = " ← best" if mae < best[2] else ""
                print(f"  conf={conf:.2f}  iou={iou:.2f}  MAE={mae:.3f}{marker}")

            if mae < best[2]:
                best = (conf, iou, mae)

    print(f"\n{'='*60}")
    print("★ BEST THRESHOLDS:")
    print(f"  Confidence : {best[0]}")
    print(f"  IoU        : {best[1]}")
    print(f"  MAE        : {best[2]:.3f}")
    print(f"{'='*60}")

    return best


# ============================================================
# ADAPTIVE CONFIDENCE LOGIC (used during evaluation)
# ============================================================

def adaptive_confidence(
    initial_pred_count: int,
    base_conf: float,
    low_count_threshold: int,
    low_conf_boost: float,
    high_count_threshold: int,
    high_conf_boost: float,
) -> Optional[float]:
    """
    Return a boosted confidence value if the initial prediction count falls
    outside the expected range, otherwise return None (no re-inference needed).

    Parameters
    ----------
    initial_pred_count : int
        Number of boxes predicted at *base_conf*.
    base_conf : float
        The base (primary) confidence threshold.
    low_count_threshold : int
        If pred_count < this, apply *low_conf_boost*.
    low_conf_boost : float
        Confidence boost for sparse predictions (reduce false negatives).
    high_count_threshold : int
        If pred_count > this, apply *high_conf_boost*.
    high_conf_boost : float
        Confidence boost for dense predictions (reduce false positives).

    Returns
    -------
    float or None
        Boosted threshold to re-run inference at, or None.
    """
    if initial_pred_count < low_count_threshold and low_conf_boost > 0:
        return min(base_conf + low_conf_boost, 0.95)
    if initial_pred_count > high_count_threshold and high_conf_boost > 0:
        return min(base_conf + high_conf_boost, 0.95)
    return None
