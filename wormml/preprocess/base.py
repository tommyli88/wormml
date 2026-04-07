"""
Core preprocessing pipeline shared across all camera types.

Entry point
-----------
    preprocess_dataset(input_base_dir, output_base_dir, cfg, ...)

The function reads a YOLO-format dataset (images/{train,val} +
labels/{train,val}), applies the configured pipeline to every image,
transforms bounding-box coordinates to match the new crop, and writes
the result to output_base_dir.

Pipeline stages (applied per image)
-------------------------------------
1. Gaussian blur  (noise reduction before detection)
2. Petri-dish crop + circular mask  (isolates the worm plate)
3. Optional inversion  (LB domain adaptation)
4. Optional brightness / blur / noise augmentation  (LB train split)
5. Resize to square target with aspect-preserving letterbox padding

Camera-specific behaviour is controlled entirely by the config dataclass
passed in — no camera-specific branches are needed here.
"""

from __future__ import annotations

import concurrent.futures
import glob
import os
import random
import shutil
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from wormml.preprocess.configs import BasePreprocessConfig


# ============================================================
# CIRCLE DETECTION
# ============================================================

def _downscale_for_detection(
    gray: np.ndarray, max_dim: int
) -> Tuple[np.ndarray, float]:
    """Downscale *gray* so its longest side equals *max_dim*."""
    h, w = gray.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        small = cv2.resize(
            gray,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
        return small, scale
    return gray, 1.0


def _score_circles_centrality(
    circles: np.ndarray, img_shape: Tuple[int, int], alpha: float
) -> int:
    """Return index of best circle using radius + centrality scoring."""
    h, w = img_shape[:2]
    img_cx, img_cy = w * 0.5, h * 0.5
    cx, cy, r = circles[:, 0], circles[:, 1], circles[:, 2]
    dist = np.sqrt((cx - img_cx) ** 2 + (cy - img_cy) ** 2)
    scores = r - alpha * dist
    return int(np.argmax(scores))


def _detect_circle_hough(
    gray_small: np.ndarray, cfg: BasePreprocessConfig
) -> Optional[Tuple[float, float, float]]:
    """
    Run HoughCircles with optional histogram equalisation and multi-pass
    fallback.  Returns (cx, cy, r) in small-image coordinates, or None.
    """
    if cfg.use_histeq:
        gray_small = cv2.equalizeHist(gray_small)

    blur = cv2.medianBlur(gray_small, cfg.median_blur_ksize)
    h, w = blur.shape[:2]
    min_dim = min(h, w)

    # Build list of (param1, param2) passes
    passes = [(cfg.HOUGH_PARAM1, cfg.HOUGH_PARAM2)]
    if cfg.hough_fallback_passes:
        passes.extend(cfg.hough_fallback_passes)

    for p1, p2 in passes:
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=cfg.HOUGH_DP,
            minDist=min_dim // 2,
            param1=max(p1, 50),
            param2=max(p2, 15),
            minRadius=int(min_dim * cfg.MIN_R_FRAC),
            maxRadius=int(min_dim * cfg.MAX_R_FRAC),
        )
        if circles is not None:
            circles = circles[0]
            if circles.ndim == 1:
                circles = circles.reshape(1, 3)

            if cfg.circle_selection == "closest_to_center":
                img_cx, img_cy = w / 2.0, h / 2.0
                dists = np.sqrt(
                    (circles[:, 0] - img_cx) ** 2
                    + (circles[:, 1] - img_cy) ** 2
                )
                idx = int(np.argmin(dists))
            elif cfg.circle_selection == "centrality":
                idx = _score_circles_centrality(circles, gray_small.shape, cfg.centrality_alpha)
            else:  # "largest_radius"
                idx = int(np.argmax(circles[:, 2]))

            cx, cy, r = circles[idx]
            return float(cx), float(cy), float(r)

    return None


def _detect_circle_contour(
    gray_small: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    """Fallback contour-based circle detection."""
    blur = cv2.GaussianBlur(gray_small, (9, 9), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = max(9, (min(gray_small.shape[:2]) // 80) * 2 + 1)
    kernel = np.ones((k, k), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    return float(cx), float(cy), float(r)


def get_crop_params(img: np.ndarray, cfg: BasePreprocessConfig) -> dict:
    """
    Detect the petri dish circle and return crop + mask parameters
    expressed in original-image pixel coordinates.

    Returns a dict with keys:
        orig_w, orig_h, x1, y1, x2, y2, new_w, new_h, cx, cy, mask_r
    """
    orig_h, orig_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray_small, scale = _downscale_for_detection(gray, cfg.MAX_DETECT_DIM)

    circ = _detect_circle_hough(gray_small, cfg)
    if circ is None:
        circ = _detect_circle_contour(gray_small)

    if circ is None:
        cx, cy = orig_w / 2.0, orig_h / 2.0
        r = min(orig_w, orig_h) * 0.45
    else:
        cx_s, cy_s, r_s = circ
        cx = cx_s / scale
        cy = cy_s / scale
        r = r_s / scale

    # Optional radius inflation (Tau)
    if cfg.radius_inflate > 0:
        r *= 1.0 + cfg.radius_inflate

    min_dim = min(orig_w, orig_h)
    r = float(np.clip(r, min_dim * 0.20, min_dim * 0.80))

    side = int(round(2.0 * r * (1.0 + cfg.PAD_FRAC)))
    side = int(min(side, orig_w, orig_h))
    side = max(side, 2)

    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x1 = max(0, min(x1, orig_w - side))
    y1 = max(0, min(y1, orig_h - side))

    mask_r = max(int(round(r * (1.0 - cfg.RIM_CUT_FRAC))), 1)

    return {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "x1": x1,
        "y1": y1,
        "x2": x1 + side,
        "y2": y1 + side,
        "new_w": side,
        "new_h": side,
        "cx": float(cx),
        "cy": float(cy),
        "mask_r": mask_r,
    }


def apply_crop_and_mask(img: np.ndarray, params: dict, bg_value: int = 0) -> np.ndarray:
    """Crop to petri dish bounding box and zero-out pixels outside the circle."""
    x1, y1, x2, y2 = params["x1"], params["y1"], params["x2"], params["y2"]
    crop = img[y1:y2, x1:x2].copy()

    cx_crop = params["cx"] - x1
    cy_crop = params["cy"] - y1
    r = params["mask_r"]

    h, w = crop.shape[:2]
    Y, X = np.ogrid[:h, :w]
    inside = (X - cx_crop) ** 2 + (Y - cy_crop) ** 2 <= r ** 2

    out = np.full_like(crop, bg_value)
    out[inside] = crop[inside]
    return out


# ============================================================
# LABEL TRANSFORMATION
# ============================================================

def transform_yolo_labels(
    label_path: str,
    out_label_path: str,
    params: dict,
) -> None:
    """
    Re-express YOLO bounding boxes in the coordinate system of the cropped image.
    Boxes whose centre falls outside the crop are silently discarded.
    """
    if not Path(label_path).exists():
        return

    orig_w, orig_h = params["orig_w"], params["orig_h"]
    x1, y1 = params["x1"], params["y1"]
    new_w, new_h = params["new_w"], params["new_h"]

    Path(out_label_path).parent.mkdir(parents=True, exist_ok=True)
    new_lines = []

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls = parts[0]
            xc = float(parts[1]) * orig_w
            yc = float(parts[2]) * orig_h
            bw = float(parts[3]) * orig_w
            bh = float(parts[4]) * orig_h

            xc_new = (xc - x1) / new_w
            yc_new = (yc - y1) / new_h
            bw_new = float(np.clip(bw / new_w, 0.0, 1.0))
            bh_new = float(np.clip(bh / new_h, 0.0, 1.0))

            if not (0.0 <= xc_new <= 1.0 and 0.0 <= yc_new <= 1.0):
                continue

            new_lines.append(
                f"{cls} {xc_new:.6f} {yc_new:.6f} {bw_new:.6f} {bh_new:.6f}\n"
            )

    with open(out_label_path, "w") as f:
        f.writelines(new_lines)


# ============================================================
# IMAGE UTILITIES
# ============================================================

def blur_and_convert(img: np.ndarray, ksize: Tuple[int, int]) -> np.ndarray:
    """Gaussian-blur and return a 3-channel BGR image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    blurred = cv2.GaussianBlur(gray, ksize, 0)
    return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)


def resize_with_padding(img: np.ndarray, target_size: int) -> np.ndarray:
    """Resize to *target_size* × *target_size* with black letterbox padding."""
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y: pad_y + new_h, pad_x: pad_x + new_w] = resized
    return canvas


def invert_colors(img: np.ndarray) -> np.ndarray:
    """Invert pixel values, preserving the black background (value=0)."""
    out = np.empty_like(img)
    mask = img > 0
    np.subtract(255, img, out=out, where=mask)
    out[~mask] = 0
    return out


# ============================================================
# LB AUGMENTATION HELPERS
# ============================================================

def augment_brightness(
    img: np.ndarray,
    rng: np.random.Generator,
    adjust_range: Tuple[float, float],
    shift_range: Tuple[float, float],
) -> np.ndarray:
    """Multiplicative + additive brightness augmentation on foreground pixels."""
    fg_mask = img > 0 if img.ndim == 2 else np.any(img > 0, axis=-1)
    if not fg_mask.any():
        return img

    out = img.astype(np.float32)
    mult = rng.uniform(*adjust_range)
    shift = rng.uniform(*shift_range)
    out[fg_mask] = out[fg_mask] * mult + shift
    np.clip(out, 0, 255, out=out)
    out = out.astype(np.uint8)
    out[~fg_mask] = 0
    return out


def augment_blur(
    img: np.ndarray,
    rng: np.random.Generator,
    ksize_options: List[int],
    sigma_range: Tuple[float, float],
    prob: float,
) -> np.ndarray:
    """Random Gaussian blur applied with probability *prob*."""
    if rng.random() >= prob:
        return img

    fg_mask = img > 0 if img.ndim == 2 else np.any(img > 0, axis=-1)
    ksize = int(rng.choice(ksize_options))
    sigma = rng.uniform(*sigma_range)
    out = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    out[~fg_mask] = 0
    return out


def augment_noise(
    img: np.ndarray,
    rng: np.random.Generator,
    sigma_range: Tuple[float, float],
    prob: float,
) -> np.ndarray:
    """Subtle Gaussian noise applied with probability *prob*."""
    if rng.random() >= prob:
        return img

    fg_mask = img > 0 if img.ndim == 2 else np.any(img > 0, axis=-1)
    if not fg_mask.any():
        return img

    sigma = rng.uniform(*sigma_range)
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    np.clip(out, 0, 255, out=out)
    out = out.astype(np.uint8)
    out[~fg_mask] = 0
    return out


def apply_lb_augmentations(
    img: np.ndarray, rng: np.random.Generator, cfg: BasePreprocessConfig
) -> np.ndarray:
    """Apply brightness → blur → noise augmentation chain for LB."""
    out = augment_brightness(
        img, rng, cfg.brightness_adjust_range, cfg.brightness_shift_range
    )
    out = augment_blur(
        out, rng, cfg.blur_ksize_options, cfg.blur_sigma_range, cfg.blur_prob
    )
    out = augment_noise(out, rng, cfg.noise_sigma_range, cfg.noise_prob)
    return out


# ============================================================
# FULL SINGLE-IMAGE PIPELINE
# ============================================================

def full_pipeline(
    img: np.ndarray,
    cfg: BasePreprocessConfig,
    is_train: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Apply the complete preprocessing pipeline to a single image.

    Steps:
        1. Gaussian blur
        2. Petri-dish crop + circular mask
        3. Optional image inversion (LB)
        4. Optional augmentation (LB train)
        5. Resize to square target with padding

    Returns
    -------
    (processed_image, crop_params)
    """
    blurred = blur_and_convert(img, cfg.gaussian_ksize)
    params = get_crop_params(blurred, cfg)
    cropped = apply_crop_and_mask(blurred, params, cfg.BG_VALUE)

    if cfg.apply_inversion:
        cropped = invert_colors(cropped)

    if cfg.apply_extra_aug_train and is_train and rng is not None:
        cropped = apply_lb_augmentations(cropped, rng, cfg)

    resized = resize_with_padding(cropped, cfg.target_size)
    return resized, params


# ============================================================
# I/O HELPERS
# ============================================================

def find_image_label_pairs(images_dir: str, labels_dir: str) -> List[Tuple[str, Optional[str]]]:
    """Return list of (image_path, label_path_or_None) pairs."""
    if not os.path.exists(images_dir):
        print(f"  ❌ Images directory missing: {images_dir}")
        return []
    if not os.path.exists(labels_dir):
        print(f"  ❌ Labels directory missing: {labels_dir}")
        return []

    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]
    image_files: List[str] = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        image_files.extend(glob.glob(os.path.join(images_dir, ext.upper())))

    pairs = []
    for img_path in image_files:
        stem = Path(img_path).stem
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        pairs.append(
            (img_path, label_path if os.path.exists(label_path) else None)
        )

    return pairs


def process_single_image(
    pair: Tuple[str, Optional[str]],
    output_images_dir: str,
    output_labels_dir: str,
    cfg: BasePreprocessConfig,
    is_train: bool = False,
    seed: int = 42,
) -> Tuple[str, Optional[str]]:
    """
    End-to-end processing for one (image, label) pair.

    For LB training images, also saves an augmented copy alongside the
    base (inverted) copy, doubling the effective training set size.
    """
    img_path, label_path = pair
    rng = np.random.default_rng(seed + hash(img_path) % (2 ** 31))

    try:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return f"❌ Failed to load: {Path(img_path).name}", None

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        final, crop_params = full_pipeline(img, cfg, is_train=is_train, rng=rng)

        os.makedirs(output_images_dir, exist_ok=True)
        out_img_path = os.path.join(output_images_dir, Path(img_path).name)
        if not cv2.imwrite(out_img_path, final):
            return f"❌ Failed to save: {Path(img_path).name}", None

        if label_path and os.path.exists(label_path):
            os.makedirs(output_labels_dir, exist_ok=True)
            out_label_path = os.path.join(output_labels_dir, Path(label_path).name)
            transform_yolo_labels(label_path, out_label_path, crop_params)

        # LB: save additional augmented copy for training
        if is_train and cfg.apply_extra_aug_train:
            aug_img = full_pipeline(img, cfg, is_train=True, rng=rng)[0]
            stem = Path(img_path).stem
            ext = Path(img_path).suffix
            aug_img_path = os.path.join(output_images_dir, f"{stem}_aug{ext}")
            cv2.imwrite(aug_img_path, aug_img)

            if label_path and os.path.exists(label_path):
                src_lbl = os.path.join(output_labels_dir, Path(label_path).name)
                aug_lbl_path = os.path.join(output_labels_dir, f"{stem}_aug.txt")
                if os.path.exists(src_lbl):
                    shutil.copy2(src_lbl, aug_lbl_path)

        return f"✅ {Path(img_path).name}", img_path

    except Exception as exc:
        return f"❌ Error {Path(img_path).name}: {exc}", None


# ============================================================
# SPLIT PROCESSOR
# ============================================================

def preprocess_split(
    input_images_dir: str,
    input_labels_dir: str,
    output_images_dir: str,
    output_labels_dir: str,
    cfg: BasePreprocessConfig,
    is_train: bool = False,
) -> Tuple[bool, List[str]]:
    """Process one dataset split (train or val)."""
    print(f"\n📂 {'TRAIN' if is_train else 'VAL'} split")
    print(f"  Input:  {input_images_dir}")
    print(f"  Output: {output_images_dir}")

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    pairs = find_image_label_pairs(input_images_dir, input_labels_dir)
    if not pairs:
        print("  ⚠️  No images found — skipping.")
        return False, []

    print(f"  🔄 Processing {len(pairs)} images with {cfg.max_workers} workers...")

    process_func = partial(
        process_single_image,
        output_images_dir=output_images_dir,
        output_labels_dir=output_labels_dir,
        cfg=cfg,
        is_train=is_train,
        seed=cfg.random_seed,
    )

    results, processed_paths = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
        for result, src_path in tqdm(
            ex.map(process_func, pairs),
            total=len(pairs),
            desc=f"  {'train' if is_train else 'val'}",
        ):
            results.append(result)
            if src_path:
                processed_paths.append(src_path)

    n_ok = sum(1 for r in results if r.startswith("✅"))
    n_fail = len(results) - n_ok
    print(f"  ✅ {n_ok} succeeded  ❌ {n_fail} failed")

    return n_ok > 0, processed_paths


# ============================================================
# DATASET SPLITTER (Tau / LB — raw merged datasets)
# ============================================================

def split_dataset(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    classes: Optional[List[str]] = None,
) -> None:
    """
    Split a flat (unsplit) dataset into train/val with matched image-label pairs.

    Expects:
        input_dir/
            images/  *.jpg|png|...
            labels/  *.txt

    Creates:
        output_dir/
            images/train/  images/val/
            labels/train/  labels/val/
            data.yaml
    """
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    classes = classes or ["worm"]

    src = Path(input_dir)
    dst = Path(output_dir)

    assert (src / "images").exists(), f"Missing: {src / 'images'}"
    assert (src / "labels").exists(), f"Missing: {src / 'labels'}"

    images = sorted(
        f
        for f in (src / "images").iterdir()
        if f.suffix.lower() in IMAGE_EXTS
        and (src / "labels" / f"{f.stem}.txt").exists()
    )
    assert images, f"No matched image/label pairs in {src}"

    random.seed(seed)
    random.shuffle(images)
    cut = int(len(images) * train_ratio)
    splits = {"train": images[:cut], "val": images[cut:]}

    print(f"  Matched pairs: {len(images)}")
    print(f"  Train: {len(splits['train'])}  |  Val: {len(splits['val'])}")

    for split, files in splits.items():
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy2(img, dst / "images" / split / img.name)
            shutil.copy2(
                src / "labels" / f"{img.stem}.txt",
                dst / "labels" / split / f"{img.stem}.txt",
            )

    yaml_text = (
        f"path: {output_dir}\n"
        f"train: images/train\n"
        f"val:   images/val\n\n"
        f"nc: {len(classes)}\n"
        f"names: {classes}\n"
    )
    (dst / "data.yaml").write_text(yaml_text)
    print(f"  ✅ Split complete → {output_dir}")


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_before_after(
    original_paths: List[str],
    output_images_dir: str,
    n: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """Side-by-side before/after grid for up to *n* random images."""
    available = [
        p
        for p in original_paths
        if os.path.exists(os.path.join(output_images_dir, Path(p).name))
    ]
    if not available:
        print("⚠️  No processed images found for visualization.")
        return

    sample = random.sample(available, min(n, len(available)))
    n_actual = len(sample)

    fig, axes = plt.subplots(n_actual, 2, figsize=(12, 4 * n_actual), facecolor="#1a1a1a")
    fig.suptitle(
        "Before → After Preprocessing\n(Blur  •  Petri Dish Crop  •  Resize)",
        color="white",
        fontsize=16,
        fontweight="bold",
        y=1.002,
    )
    if n_actual == 1:
        axes = [axes]

    for i, img_path in enumerate(sample):
        orig = cv2.imread(img_path, cv2.IMREAD_COLOR)
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB) if orig is not None else None
        out_path = os.path.join(output_images_dir, Path(img_path).name)
        proc = cv2.imread(out_path, cv2.IMREAD_COLOR)
        proc_rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB) if proc is not None else None

        ax_b, ax_a = axes[i][0], axes[i][1]
        if orig_rgb is not None:
            ax_b.imshow(orig_rgb)
        if proc_rgb is not None:
            ax_a.imshow(proc_rgb)
        ax_b.set_ylabel(Path(img_path).stem, color="white", fontsize=8, rotation=0, labelpad=80, va="center")
        for ax in (ax_b, ax_a):
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0][0].set_title("BEFORE", color="#ff9966", fontsize=11, fontweight="bold")
    axes[0][1].set_title("AFTER", color="#66ccff", fontsize=11, fontweight="bold")
    plt.tight_layout(pad=1.5)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
        print(f"📸 Visualization saved: {save_path}")
    else:
        plt.show()
    plt.close()


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def preprocess_dataset(
    input_base_dir: str,
    output_base_dir: str,
    cfg: BasePreprocessConfig,
    visualize: bool = True,
    vis_save_path: Optional[str] = None,
) -> bool:
    """
    Preprocess an entire YOLO dataset for a given camera config.

    Parameters
    ----------
    input_base_dir : str
        Root directory with images/{train,val} and labels/{train,val}.
    output_base_dir : str
        Where to write the preprocessed dataset.
    cfg : BasePreprocessConfig
        Camera-specific preprocessing configuration.
    visualize : bool
        Whether to generate a before/after comparison figure.
    vis_save_path : str, optional
        Where to save the visualisation.  Defaults to
        ``output_base_dir/preprocess_visualization.png``.

    Returns
    -------
    bool
        True if both train and val splits succeeded.
    """
    # UVA — skip actual preprocessing, just copy
    if getattr(cfg, "skip_preprocessing", False):
        print("ℹ️  UVA config: skipping preprocessing, copying images as-is.")
        for split in ("train", "val"):
            for kind in ("images", "labels"):
                src = os.path.join(input_base_dir, kind, split)
                dst = os.path.join(output_base_dir, kind, split)
                if os.path.exists(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
        return True

    print("=" * 80)
    print("C. elegans Image Preprocessing  (blur → crop → resize)")
    print("=" * 80)
    print(f"Input:       {input_base_dir}")
    print(f"Output:      {output_base_dir}")
    print(f"Target size: {cfg.target_size}×{cfg.target_size}")
    print(f"Inversion:   {cfg.apply_inversion}")
    print(f"Extra aug:   {cfg.apply_extra_aug_train}")
    print("-" * 80)

    # Validate input structure
    for name in ("images/train", "images/val", "labels/train", "labels/val"):
        path = os.path.join(input_base_dir, *name.split("/"))
        if not os.path.exists(path):
            print(f"  ❌ Missing: {path}")
            return False
        count = len(glob.glob(os.path.join(path, "*.*")))
        print(f"  ✅ {name}: {count} files")

    os.makedirs(output_base_dir, exist_ok=True)
    all_processed: List[str] = []

    train_ok, train_paths = preprocess_split(
        input_images_dir=os.path.join(input_base_dir, "images", "train"),
        input_labels_dir=os.path.join(input_base_dir, "labels", "train"),
        output_images_dir=os.path.join(output_base_dir, "images", "train"),
        output_labels_dir=os.path.join(output_base_dir, "labels", "train"),
        cfg=cfg,
        is_train=True,
    )
    all_processed.extend(train_paths)

    val_ok, val_paths = preprocess_split(
        input_images_dir=os.path.join(input_base_dir, "images", "val"),
        input_labels_dir=os.path.join(input_base_dir, "labels", "val"),
        output_images_dir=os.path.join(output_base_dir, "images", "val"),
        output_labels_dir=os.path.join(output_base_dir, "labels", "val"),
        cfg=cfg,
        is_train=False,
    )
    all_processed.extend(val_paths)

    print("\n" + "=" * 80)
    for split, ok in [("TRAIN", train_ok), ("VAL", val_ok)]:
        img_dir = os.path.join(output_base_dir, "images", split.lower())
        lbl_dir = os.path.join(output_base_dir, "labels", split.lower())
        if ok and os.path.exists(img_dir):
            imgs = len(glob.glob(os.path.join(img_dir, "*.*")))
            lbls = len(glob.glob(os.path.join(lbl_dir, "*.txt")))
            print(f"  ✅ {split}: {imgs} images, {lbls} labels")
        else:
            print(f"  ❌ {split}: FAILED")

    overall = train_ok and val_ok
    print("🎉 DONE!" if overall else "⚠️  COMPLETED WITH ERRORS")
    print("=" * 80)

    if visualize and all_processed:
        out_img_dir = os.path.join(output_base_dir, "images", "train")
        if not os.path.exists(out_img_dir):
            out_img_dir = os.path.join(output_base_dir, "images", "val")
        save_path = vis_save_path or os.path.join(
            output_base_dir, "preprocess_visualization.png"
        )
        visualize_before_after(all_processed, out_img_dir, n=10, save_path=save_path)

    return overall
