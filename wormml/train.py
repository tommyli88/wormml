"""
Unified YOLO training wrapper for WormML.

Usage (Python API)
------------------
    from wormml.train import train, TrainingConfig

    cfg = TrainingConfig(
        dataset_base = "/data/og_preprocessed",
        output_dir   = "/runs/og",
        camera       = "og",
    )
    train(cfg)

Usage (CLI)
-----------
    python scripts/train.py --config configs/og.yaml

Design notes
------------
Training hyperparameters are grouped into the TrainingConfig dataclass.
Per-camera defaults encode the empirically-tuned values from the thesis
experiments.  The ``camera`` field selects the right default profile;
individual fields can be overridden for ablation studies.

Camera-specific differences
----------------------------
OG   – warmup=3, copy_paste=0.1, box=9.0, lr0=0.001, rect=False, no single_cls
Tau  – warmup=10, copy_paste=0.3, box=7.5, lr0=0.0008 (×0.8), rect=True, single_cls
LB   – same as Tau (augmentation handled in preprocessing)
UVA  – warmup=10, copy_paste=0.3, box=7.5, lr0=0.0008, rect=True, single_cls, patience=15
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from wormml.datasets import check_dataset, create_data_yaml, dataset_stats


# ============================================================
# TRAINING CONFIG
# ============================================================

@dataclass
class TrainingConfig:
    # ---- Required ----
    dataset_base: str = ""
    output_dir: str = ""

    # Camera profile (selects defaults): "og" | "tau" | "lb" | "uva" | "custom"
    camera: str = "og"

    # ---- Training schedule ----
    epochs: int = 100
    patience: int = 25
    img_size: int = 1344
    batch_size: int = 6
    workers: int = 8

    # ---- Augmentation ----
    warmup_epochs: int = 3
    mosaic: float = 1.0
    mixup: float = 0.3
    copy_paste: float = 0.1
    degrees: float = 15.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 2.0
    perspective: float = 0.0003
    flipud: float = 0.5
    fliplr: float = 0.5

    # ---- Optimizer ----
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005

    # ---- Loss weights ----
    cls_loss: float = 0.5
    box_loss: float = 9.0
    dfl_loss: float = 1.5

    # ---- YOLO-specific flags ----
    rect: bool = False
    single_cls: bool = False
    amp: bool = True
    seed: int = 42
    deterministic: bool = True

    # ---- Model ----
    model_types: List[str] = field(default_factory=lambda: ["11l"])

    def apply_camera_defaults(self) -> "TrainingConfig":
        """
        Apply the per-camera hyperparameter profile in-place.

        OG  → warmup=3,  copy_paste=0.1, box=9.0,  lr0=0.001, rect=False
        Tau → warmup=10, copy_paste=0.3, box=7.5,  lr0=0.0008, rect=True, single_cls
        LB  → same as Tau
        UVA → same as Tau + patience=15
        """
        cam = self.camera.lower()
        if cam == "og":
            self.warmup_epochs = 3
            self.copy_paste = 0.1
            self.box_loss = 9.0
            self.lr0 = 0.001
            self.rect = False
            self.single_cls = False
        elif cam in ("tau", "lb"):
            self.warmup_epochs = 10
            self.copy_paste = 0.3
            self.box_loss = 7.5
            self.lr0 = 0.001 * 0.8  # historical 0.8× scaling
            self.rect = True
            self.single_cls = True
        elif cam == "uva":
            self.warmup_epochs = 10
            self.copy_paste = 0.3
            self.box_loss = 7.5
            self.lr0 = 0.001 * 0.8
            self.rect = True
            self.single_cls = True
            self.patience = 15
            self.workers = 10
        return self


# ============================================================
# CORE TRAINING FUNCTIONS
# ============================================================

def _build_model_name(model_type: str) -> str:
    """Return the Ultralytics model filename for a given type string."""
    if model_type.startswith("11"):
        return f"yolo{model_type}.pt"
    return f"yolo11{model_type}.pt"


def train_single_model(
    data_yaml: str,
    model_type: str,
    cfg: TrainingConfig,
) -> str:
    """
    Train one YOLOv11 model and return the path to best.pt.

    Parameters
    ----------
    data_yaml : str
        Path to the dataset data.yaml file.
    model_type : str
        YOLO size suffix, e.g. ``"11l"``.
    cfg : TrainingConfig
        All training hyperparameters.

    Returns
    -------
    str – path to best.pt
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required.  Install with: pip install ultralytics"
        ) from exc

    model_file = _build_model_name(model_type)
    exp_name = f"yolov11_maxacc_{model_type}"

    print(f"\n{'='*60}")
    print(f"Training {exp_name}")
    print(f"{'='*60}")

    model = YOLO(model_file)

    train_args = dict(
        data=data_yaml,
        epochs=cfg.epochs,
        patience=cfg.patience,
        batch=cfg.batch_size,
        imgsz=cfg.img_size,
        project=cfg.output_dir,
        name=exp_name,
        exist_ok=True,
        device=0,
        workers=cfg.workers,
        optimizer=cfg.optimizer,
        lr0=cfg.lr0,
        lrf=cfg.lrf,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs,
        mosaic=cfg.mosaic,
        mixup=cfg.mixup,
        copy_paste=cfg.copy_paste,
        auto_augment="randaugment",
        rect=cfg.rect,
        amp=cfg.amp,
        seed=cfg.seed,
        deterministic=cfg.deterministic,
    )

    # single_cls only when explicitly enabled (Tau / LB / UVA)
    if cfg.single_cls:
        train_args["single_cls"] = True

    print(f"→ Starting {cfg.epochs}-epoch training run …")
    model.train(**train_args)

    print("\n→ Running final validation …")
    model.val(data=data_yaml, batch=cfg.batch_size, imgsz=cfg.img_size, device=0)

    best_ckpt = os.path.join(cfg.output_dir, exp_name, "weights", "best.pt")
    print(f"\n✅ Best checkpoint: {best_ckpt}")
    return best_ckpt


def progressive_resize(
    data_yaml: str,
    model_type: str,
    base_ckpt: str,
    cfg: TrainingConfig,
    phases: Optional[List] = None,
) -> str:
    """
    Optional progressive-resizing fine-tuning after base training.

    *phases* is a list of ``(imgsz, batch, epochs)`` tuples.
    Falls back to *base_ckpt* if a phase fails (GPU OOM).

    Returns path to best available checkpoint after all phases.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required.") from exc

    if phases is None:
        phases = [(1536, 4, 30), (1728, 2, 20)]

    last_ckpt = base_ckpt
    print(f"\n{'='*60}")
    print("Progressive Resizing Phases")
    print(f"{'='*60}")

    for size, batch, ep in phases:
        exp = f"prog_{model_type}_{size}"
        phase_last = os.path.join(cfg.output_dir, exp, "weights", "last.pt")
        phase_best = os.path.join(cfg.output_dir, exp, "weights", "best.pt")
        resume = os.path.exists(phase_last)
        model_path = phase_last if resume else last_ckpt

        print(f"\nPhase {size}px  (batch={batch}, epochs={ep}, resume={resume})")
        try:
            model = YOLO(model_path)
            model.train(
                data=data_yaml,
                imgsz=size,
                batch=batch,
                epochs=ep,
                project=cfg.output_dir,
                name=exp,
                resume=resume,
                device=0,
                amp=True,
                exist_ok=True,
            )
            model.val(data=data_yaml, batch=batch, imgsz=size, device=0)
            last_ckpt = phase_best if os.path.exists(phase_best) else phase_last
        except Exception as exc:
            print(f"⚠️  Phase {size}px skipped (likely GPU OOM): {exc}")

    return last_ckpt


# ============================================================
# HIGH-LEVEL ENTRY POINT
# ============================================================

def train(cfg: TrainingConfig, apply_defaults: bool = True) -> None:
    """
    Run the full training pipeline for one camera configuration.

    Steps:
        1. Validate dataset
        2. Compute dataset statistics
        3. Write data.yaml
        4. Train each model_type
        5. (Optional) progressive resizing
        6. Threshold sweep

    Parameters
    ----------
    cfg : TrainingConfig
        Training configuration; camera defaults applied if *apply_defaults*.
    apply_defaults : bool
        When True, call ``cfg.apply_camera_defaults()`` before training.
    """
    if apply_defaults:
        cfg.apply_camera_defaults()

    print(f"\n{'='*60}")
    print("WormML — YOLOv11 C. elegans Training Pipeline")
    print(f"Camera profile : {cfg.camera.upper()}")
    print(f"Dataset        : {cfg.dataset_base}")
    print(f"Output         : {cfg.output_dir}")
    print(f"{'='*60}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    n_train, n_val = check_dataset(cfg.dataset_base)
    if n_train == 0:
        raise RuntimeError(
            "No training images found.  "
            "Run preprocessing first or check dataset_base path."
        )

    dataset_stats(cfg.dataset_base)
    data_yaml = create_data_yaml(cfg.dataset_base)

    for m in cfg.model_types:
        print(f"\n{'='*60}")
        print(f"Model: {m}")
        print(f"{'='*60}")

        best_ckpt = train_single_model(data_yaml, m, cfg)

        # Progressive resizing is optional and may fail on limited GPU memory
        try:
            print("\n⚠️  Attempting progressive resizing (may fail on <16 GB GPU) …")
            last_ckpt = progressive_resize(data_yaml, m, best_ckpt, cfg)
        except Exception as exc:
            print(f"⚠️  Progressive resizing skipped: {exc}")

    print(f"\n{'='*60}")
    print("✅ PIPELINE COMPLETE!")
    print(f"Results: {cfg.output_dir}")
    print(f"{'='*60}")
