"""
Model evaluation using Hungarian-matched precision / recall.

The evaluator runs YOLO inference on every validation image, applies
optional adaptive confidence boosting, computes per-image detection
metrics via optimal bipartite matching (scipy ``linear_sum_assignment``),
and aggregates results into counting and detection summary tables.

Images where both precision AND recall are 0.0 are excluded from the
averaged P/R/F1 metrics; they typically indicate annotation errors rather
than genuine model failures.

Usage (Python API)
------------------
    from wormml.evaluate import evaluate, EvalConfig

    result = evaluate(EvalConfig(
        camera        = "OG",
        model_path    = "runs/og/yolov11_maxacc_11l/weights/best.pt",
        images_dir    = "data/og_preprocessed/images/val",
        labels_dir    = "data/og_preprocessed/labels/val",
        conf_thr      = 0.35,
        iou_thr       = 0.30,
    ))
    print(result["mae"], result["mean_precision"])

Usage (CLI)
-----------
    python scripts/evaluate.py --config configs/og.yaml
"""

from __future__ import annotations

import glob
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from wormml.threshold import adaptive_confidence


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================
# EVALUATION CONFIG
# ============================================================

@dataclass
class EvalConfig:
    camera: str = ""
    model_path: str = ""
    images_dir: str = ""
    labels_dir: str = ""
    conf_thr: float = 0.25
    iou_thr: float = 0.50
    low_count_threshold: int = 10
    conf_boost: float = 0.0
    high_count_threshold: int = 80
    high_conf_boost: float = 0.0
    show_first_n: int = 3


# ============================================================
# IoU & MATCHING
# ============================================================

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Compute IoU of two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def calculate_precision_recall(
    true_boxes: List, pred_boxes: List, iou_threshold: float = 0.5
) -> Tuple[float, float, int, int, int]:
    """
    Compute precision and recall via Hungarian bipartite matching.

    Returns
    -------
    (precision, recall, TP, FP, FN)
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError("scipy is required.  Install: pip install scipy") from exc

    if not pred_boxes and not true_boxes:
        return 1.0, 1.0, 0, 0, 0
    if not pred_boxes:
        return 0.0, 0.0, 0, 0, len(true_boxes)
    if not true_boxes:
        return 0.0, 0.0, 0, len(pred_boxes), 0

    iou_matrix = np.zeros((len(true_boxes), len(pred_boxes)), dtype=np.float32)
    for i, tb in enumerate(true_boxes):
        for j, pb in enumerate(pred_boxes):
            iou_matrix[i, j] = calculate_iou(tb, pb)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    tp = sum(1 for r, c in zip(row_ind, col_ind) if iou_matrix[r, c] >= iou_threshold)
    fp = len(pred_boxes) - tp
    fn = len(true_boxes) - tp

    precision = tp / len(pred_boxes) if pred_boxes else 0.0
    recall = tp / len(true_boxes) if true_boxes else 0.0
    return precision, recall, tp, fp, fn


# ============================================================
# LABEL PARSING
# ============================================================

def parse_label_file(
    label_path: str, img_shape: Tuple[int, int]
) -> Tuple[Optional[int], List]:
    """
    Parse a YOLO label file and return (count, pixel-space boxes).
    Returns (None, []) if the file does not exist.
    """
    if not os.path.exists(label_path):
        return None, []

    boxes: List[List[int]] = []
    try:
        img_h, img_w = img_shape
        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                _, cx, cy, w, h = map(float, parts[:5])
                x1 = int((cx - w / 2) * img_w)
                y1 = int((cy - h / 2) * img_h)
                x2 = int((cx + w / 2) * img_w)
                y2 = int((cy + h / 2) * img_h)
                boxes.append([x1, y1, x2, y2])
        return len(lines), boxes
    except Exception:
        return None, []


# ============================================================
# IMAGE COLLECTION
# ============================================================

def get_image_paths(images_dir: str) -> List[str]:
    paths: List[str] = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(images_dir, f"*{ext.upper()}")))
    return sorted(set(paths))


# ============================================================
# FILTER ZERO P/R
# ============================================================

def filter_valid_pr(
    results_list: List[dict],
) -> Tuple[List[dict], int, List[str]]:
    """
    Remove entries where precision=0 AND recall=0 (likely annotation errors).

    Returns
    -------
    (filtered_list, n_excluded, excluded_image_names)
    """
    filtered, excluded = [], []
    for r in results_list:
        if r["precision"] == 0.0 and r["recall"] == 0.0:
            excluded.append(r["image"])
        else:
            filtered.append(r)
    return filtered, len(excluded), excluded


# ============================================================
# MAIN EVALUATION FUNCTION
# ============================================================

def run_evaluation(cfg: EvalConfig) -> Optional[Dict]:
    """
    Run complete evaluation for one camera model.

    Returns a result dict or None on failure.
    """
    try:
        from ultralytics import YOLO
        from PIL import Image as PILImage
    except ImportError as exc:
        raise ImportError(
            "ultralytics and Pillow are required.  "
            "Install: pip install ultralytics Pillow"
        ) from exc

    print(f"\n{'═'*70}")
    print(f"  CAMERA: {cfg.camera}")
    print(f"{'═'*70}")

    for attr, label in [
        ("model_path", "Model"),
        ("images_dir", "Images dir"),
        ("labels_dir", "Labels dir"),
    ]:
        path = getattr(cfg, attr)
        if not os.path.exists(path):
            print(f"  ⚠️  {label} not found: {path}")
            return None

    model = YOLO(cfg.model_path)
    img_paths = get_image_paths(cfg.images_dir)

    if not img_paths:
        print(f"  ❌ No images in {cfg.images_dir}")
        return None

    print(f"  Running inference on {len(img_paths)} images …")
    print(f"  conf={cfg.conf_thr}  iou={cfg.iou_thr}")

    results_list: List[dict] = []
    missing_labels = 0
    failed_preds = 0
    low_boosted = 0
    high_boosted = 0
    start = time.time()

    for i, img_path in enumerate(tqdm(img_paths, desc=f"  {cfg.camera}", unit="img")):
        img_name = os.path.basename(img_path)
        label_path = os.path.join(cfg.labels_dir, Path(img_path).stem + ".txt")

        try:
            with PILImage.open(img_path) as pil_img:
                img_shape = (pil_img.height, pil_img.width)
        except Exception:
            failed_preds += 1
            continue

        true_count, true_boxes = parse_label_file(label_path, img_shape)
        if true_count is None:
            missing_labels += 1
            continue

        try:
            res = model.predict(img_path, conf=cfg.conf_thr, verbose=False)
            if res and res[0].boxes is not None:
                pred_boxes = [
                    [int(x1), int(y1), int(x2), int(y2)]
                    for x1, y1, x2, y2 in res[0].boxes.xyxy.cpu().numpy()
                ]
            else:
                pred_boxes = []

            conf_used = cfg.conf_thr

            # Adaptive confidence boost
            boost_conf = adaptive_confidence(
                len(pred_boxes),
                cfg.conf_thr,
                cfg.low_count_threshold,
                cfg.conf_boost,
                cfg.high_count_threshold,
                cfg.high_conf_boost,
            )
            if boost_conf is not None:
                res2 = model.predict(img_path, conf=boost_conf, verbose=False)
                if res2 and res2[0].boxes is not None:
                    pred_boxes = [
                        [int(x1), int(y1), int(x2), int(y2)]
                        for x1, y1, x2, y2 in res2[0].boxes.xyxy.cpu().numpy()
                    ]
                else:
                    pred_boxes = []
                conf_used = boost_conf
                if len(pred_boxes) < cfg.low_count_threshold:
                    low_boosted += 1
                else:
                    high_boosted += 1

            precision, recall, tp, fp, fn = calculate_precision_recall(
                true_boxes, pred_boxes, cfg.iou_thr
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            results_list.append(
                dict(
                    image=img_name,
                    true_count=true_count,
                    pred_count=len(pred_boxes),
                    error=len(pred_boxes) - true_count,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    conf_used=conf_used,
                )
            )

            if i < cfg.show_first_n:
                print(
                    f"\n    [{i+1}] {img_name}  "
                    f"GT={true_count}  Pred={len(pred_boxes)}  "
                    f"P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}"
                )

        except Exception as exc:
            print(f"    ❌ {img_name}: {exc}")
            failed_preds += 1

    elapsed = time.time() - start

    if not results_list:
        print("  ❌ No valid results.")
        return None

    # ---- Counting metrics ----
    errors = np.array([r["error"] for r in results_list])
    true_counts = np.array([r["true_count"] for r in results_list])
    pred_counts = np.array([r["pred_count"] for r in results_list])

    mae = float(np.mean(np.abs(errors)))
    rmse = math.sqrt(float(np.mean(errors ** 2)))
    within_1 = float(np.mean(np.abs(errors) <= 1))
    within_2 = float(np.mean(np.abs(errors) <= 2))
    within_4 = float(np.mean(np.abs(errors) <= 4))

    # ---- Detection metrics (excluding P=R=0) ----
    valid_pr, n_excluded, excluded_imgs = filter_valid_pr(results_list)

    if valid_pr:
        mean_p = float(np.mean([r["precision"] for r in valid_pr]))
        mean_r = float(np.mean([r["recall"] for r in valid_pr]))
        mean_f1 = float(np.mean([r["f1"] for r in valid_pr]))
        total_tp = sum(r["tp"] for r in valid_pr)
        total_fp = sum(r["fp"] for r in valid_pr)
        total_fn = sum(r["fn"] for r in valid_pr)
        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (
            2 * micro_p * micro_r / (micro_p + micro_r)
            if (micro_p + micro_r) > 0
            else 0.0
        )
    else:
        mean_p = mean_r = mean_f1 = float("nan")
        micro_p = micro_r = micro_f1 = float("nan")
        total_tp = total_fp = total_fn = 0

    return dict(
        camera=cfg.camera,
        mae=mae,
        rmse=rmse,
        within_1=within_1,
        within_2=within_2,
        within_4=within_4,
        mean_precision=mean_p,
        mean_recall=mean_r,
        mean_f1=mean_f1,
        micro_precision=micro_p,
        micro_recall=micro_r,
        micro_f1=micro_f1,
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        total_gt=int(sum(true_counts)),
        total_pred=int(sum(pred_counts)),
        n_images=len(results_list),
        n_valid_pr=len(valid_pr),
        n_excluded_pr=n_excluded,
        excluded_images=excluded_imgs,
        missing_labels=missing_labels,
        failed_preds=failed_preds,
        low_boosted_conf=low_boosted,
        high_boosted_conf=high_boosted,
        conf_thr=cfg.conf_thr,
        iou_thr=cfg.iou_thr,
        low_count_threshold=cfg.low_count_threshold,
        conf_boost=cfg.conf_boost,
        high_count_threshold=cfg.high_count_threshold,
        high_conf_boost=cfg.high_conf_boost,
        inference_time=elapsed,
        per_image_results=results_list,
    )


# ============================================================
# MULTI-CAMERA EVALUATION
# ============================================================

def evaluate(configs: List[EvalConfig]) -> Dict[str, Dict]:
    """
    Evaluate a list of camera configs and return a dict keyed by camera name.

    Parameters
    ----------
    configs : list[EvalConfig]
        One entry per camera/model to evaluate.

    Returns
    -------
    dict  {camera_name: result_dict}
    """
    print("=" * 70)
    print("  WORMML MODEL EVALUATION")
    print("  Hungarian matching  •  P=R=0 images excluded from P/R metrics")
    print("=" * 70)

    all_results: Dict[str, Dict] = {}
    for cfg in configs:
        result = run_evaluation(cfg)
        if result is not None:
            all_results[cfg.camera] = result
            _print_camera_results(result)

    if all_results:
        print_summary_tables(all_results)

    return all_results


# ============================================================
# PRINTING UTILITIES
# ============================================================

def _print_camera_results(r: Dict) -> None:
    cam = r["camera"]
    print(f"\n  ── {cam} RESULTS ──")
    print(f"  COUNTING  (n={r['n_images']})")
    print(f"    MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}  "
          f"±1={r['within_1']:.1%}  ±2={r['within_2']:.1%}  ±4={r['within_4']:.1%}")
    print(f"  DETECTION (Hungarian, IoU≥{r['iou_thr']}, excl {r['n_excluded_pr']} P=R=0)")
    print(f"    P={r['mean_precision']:.4f}  R={r['mean_recall']:.4f}  F1={r['mean_f1']:.4f}")
    print(f"  MICRO     TP={r['total_tp']}  FP={r['total_fp']}  FN={r['total_fn']}")


def print_summary_tables(all_results: Dict[str, Dict]) -> None:
    """Print ASCII summary tables for all evaluated cameras."""
    cams = [c for c in ("OG", "Tau", "LB", "UVA") if c in all_results]
    cams += [c for c in all_results if c not in cams]

    print(f"\n{'═'*70}")
    print("  SUMMARY — COUNTING METRICS")
    print(f"{'═'*70}")
    print(f"  {'Camera':<8} {'MAE':>8} {'RMSE':>8} {'±1':>8} {'±2':>8} {'±4':>8} {'N':>6}")
    print(f"  {'─'*56}")
    for c in cams:
        r = all_results[c]
        print(
            f"  {c:<8} {r['mae']:>8.3f} {r['rmse']:>8.3f} "
            f"{r['within_1']:>7.1%} {r['within_2']:>7.1%} {r['within_4']:>7.1%} "
            f"{r['n_images']:>6}"
        )

    print(f"\n{'═'*70}")
    print("  SUMMARY — DETECTION METRICS (macro-avg, P=R=0 excluded)")
    print(f"{'═'*70}")
    print(f"  {'Camera':<8} {'P':>10} {'R':>10} {'F1':>10} {'Valid':>6} {'Excl':>6}")
    print(f"  {'─'*56}")
    for c in cams:
        r = all_results[c]
        print(
            f"  {c:<8} {r['mean_precision']:>10.4f} {r['mean_recall']:>10.4f} "
            f"{r['mean_f1']:>10.4f} {r['n_valid_pr']:>6} {r['n_excluded_pr']:>6}"
        )


def print_latex_tables(all_results: Dict[str, Dict]) -> None:
    """Print LaTeX table code for the paper."""
    cams = [c for c in ("OG", "Tau", "LB", "UVA") if c in all_results]

    print(r"""
% ── Table 1: Combined metrics ──────────────────────────────────────
\begin{table}[h]
\centering
\caption{In-domain validation results per camera (Hungarian matching, P=R=0 excluded)}
\begin{tabular}{lcccccccc}
\toprule
Camera & P & R & F1 & MAE & RMSE & $\pm$1 & N & Excl \\
\midrule""")
    for c in cams:
        r = all_results[c]
        print(
            f"{c} & {r['mean_precision']:.3f} & {r['mean_recall']:.3f} & "
            f"{r['mean_f1']:.3f} & {r['mae']:.2f} & {r['rmse']:.2f} & "
            f"{r['within_1']:.1%} & {r['n_images']} & {r['n_excluded_pr']} \\\\"
        )
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")

    print(r"""
% ── Table 2: Evaluation configuration ──────────────────────────────
\begin{table}[h]
\centering
\caption{Evaluation configuration per camera}
\begin{tabular}{lccccc}
\toprule
Camera & Conf & IoU & LowThr & LowBoost & HighThr & HighBoost \\
\midrule""")
    for c in cams:
        r = all_results[c]
        print(
            f"{c} & {r['conf_thr']:.2f} & {r['iou_thr']:.2f} & "
            f"{r['low_count_threshold']} & {r['conf_boost']:.3f} & "
            f"{r['high_count_threshold']} & {r['high_conf_boost']:.3f} \\\\"
        )
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")
