# WormML — Cross-Camera C. elegans Worm Counting Pipeline

**A reproducible, open-source YOLOv11-based framework for counting *C. elegans* worms across multiple imaging systems.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Worm-counting models trained on one imaging system typically fail to transfer to another. WormML addresses this by providing a **single, unified pipeline** with camera-specific preprocessing and hyperparameter configurations for four imaging platforms:

| Camera | System | Key preprocessing difference |
|--------|--------|------------------------------|
| **OG** | Original lab microscope | Standard Hough crop, no augmentation |
| **Tau** | Tau imaging system | Histogram equalisation, closest-to-centre circle |
| **LB** | LoopBio automated platform | Image inversion + brightness/blur/noise augmentation |
| **UVA** | External UVA dataset | No preprocessing — native YOLO training |

---

## Pretrained Weights

Download all four camera checkpoints with one command:

```bash
pip install huggingface_hub
python scripts/download_weights.py           # all cameras
python scripts/download_weights.py --camera og   # one camera only
```

Weights are hosted at [`litommy88/wormml`](https://huggingface.co/litommy88/wormml) and saved to `weights/`.

| File | Camera |
|------|--------|
| `og_best.pt`  | OG (original) |
| `tau_best.pt` | Tau |
| `lb_best.pt`  | LoopBio |
| `uva_best.pt` | UVA |

---

## Installation

**Python 3.10+ is required.** We recommend creating a virtual environment first.

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/wormml.git
cd wormml

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Mac / Linux
venv\Scripts\activate             # Windows

# 3. Install PyTorch  (visit https://pytorch.org to get the right command for your system)
#    Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#    Example for CPU only (evaluation only, no training):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install remaining dependencies
pip install -r requirements.txt
```

> **GPU note:** a CUDA GPU is required for training (expect ~2–4 hours per camera on a
> modern GPU). Evaluation and preprocessing can run on CPU.

> **All scripts must be run from the repo root directory** (`cd wormml` first), not from
> inside `scripts/`. The imports will break otherwise.

---

## How YOLO Datasets Are Organised

Every script in this repo expects data in **YOLO format** — images and their label files
mirrored across matching folder trees, split into `train/` and `val/`:

```
your_dataset/
├── images/
│   ├── train/
│   │   ├── plate_001.jpg
│   │   ├── plate_002.jpg
│   │   └── ...
│   └── val/
│       ├── plate_101.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── plate_001.txt    ← one line per worm
    │   ├── plate_002.txt
    │   └── ...
    └── val/
        ├── plate_101.txt
        └── ...
```

Each `.txt` label file contains one line per annotated worm in normalised format:

```
0 0.512 0.487 0.023 0.031
0 0.234 0.765 0.021 0.028
```

Fields: `class_id  centre_x  centre_y  width  height` (all 0–1, relative to image size).
`class_id` is always `0` (worm).

---

## Preparing Your Raw Data

### Exporting from CVAT

In CVAT, export your task using **Ultralytics YOLO Detection 1.0** with the
**Save Images** option enabled. This gives you a `.zip` file.

Unzip it — the contents will look like this:

```
cvat_export/
├── images/
│   └── train/          ← all your images land here
│       ├── plate_001.jpg
│       └── ...
├── labels/
│   └── train/          ← all your labels land here
│       ├── plate_001.txt
│       └── ...
└── data.yaml
```

CVAT puts everything inside a `train/` subfolder. You need to flatten it one
level before running the split script.

**Mac / Linux:**
```bash
mkdir -p raw/images raw/labels
mv cvat_export/images/train/* raw/images/
mv cvat_export/labels/train/* raw/labels/
```

**Windows (Command Prompt):**
```bat
mkdir raw\images raw\labels
move cvat_export\images\train\* raw\images\
move cvat_export\labels\train\* raw\labels\
```

Your `raw/` folder should now look like this:

```
raw/
├── images/
│   ├── plate_001.jpg
│   ├── plate_002.jpg
│   └── ...
└── labels/
    ├── plate_001.txt    ← same stem as the image
    ├── plate_002.txt
    └── ...
```

Every image must have a matching label file with the same stem name. Images
without a label file are skipped automatically by the split script.

---

## Step 0 — Split Your Raw Data

All four cameras use the same split script. It always uses seed 42 and an 80/20
train/val ratio so results are reproducible across runs and collaborators.

```bash
python scripts/split_dataset.py \
    --input  /data/tau/raw \
    --output /data/tau/split
```

This produces the `train/` / `val/` structure above, ready for preprocessing:

```
split/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

---

## Step-by-Step: Training Your Own Model

### OG Camera

```bash
# 0. Split raw data (80/20, seed 42)
python scripts/split_dataset.py \
    --input  /data/og/raw \
    --output /data/og/split

# 1. Preprocess  (Gaussian blur → Hough circle crop → resize to 1344×1344)
python scripts/preprocess.py \
    --camera og \
    --input  /data/og/split \
    --output /data/og/preprocessed

# 2. Train  (YOLOv11 Large, 100 epochs)
python scripts/train.py \
    --config configs/og.yaml \
    --dataset-base /data/og/preprocessed \
    --output-dir   /runs/og

# 3. Find optimal confidence + IoU thresholds
python scripts/tune_thresholds.py \
    --config  configs/og.yaml \
    --dataset /data/og/preprocessed

# 4. Evaluate
python scripts/evaluate.py --config configs/og.yaml
```

Trained weights land at `/runs/og/yolov11_maxacc_11l/weights/best.pt`.

---

### Tau Camera

```bash
# 0. Split raw data (80/20, seed 42)
python scripts/split_dataset.py \
    --input  /data/tau/raw \
    --output /data/tau/split

# 1. Preprocess  (adds histogram equalisation + wider crop for Tau optics)
python scripts/preprocess.py \
    --camera tau \
    --input  /data/tau/split \
    --output /data/tau/preprocessed

# 2. Train
python scripts/train.py \
    --config configs/tau.yaml \
    --dataset-base /data/tau/preprocessed \
    --output-dir   /runs/tau

# 3. Threshold sweep
python scripts/tune_thresholds.py \
    --config  configs/tau.yaml \
    --dataset /data/tau/preprocessed

# 4. Evaluate
python scripts/evaluate.py --config configs/tau.yaml
```

---

### LoopBio (LB) Camera

```bash
# 0. Split raw data (80/20, seed 42)
python scripts/split_dataset.py \
    --input  /data/lb/raw \
    --output /data/lb/split

# 1. Preprocess  (inverts images, applies brightness/blur/noise aug on train split)
python scripts/preprocess.py \
    --camera lb \
    --input  /data/lb/split \
    --output /data/lb/preprocessed

# 2. Train
python scripts/train.py \
    --config configs/lb.yaml \
    --dataset-base /data/lb/preprocessed \
    --output-dir   /runs/lb

# 3. Threshold sweep
python scripts/tune_thresholds.py \
    --config  configs/lb.yaml \
    --dataset /data/lb/preprocessed

# 4. Evaluate
python scripts/evaluate.py --config configs/lb.yaml
```

---

### UVA Camera

UVA images are used as-is — no crop, inversion, or resizing.

```bash
# 0. Split raw data (80/20, seed 42)
python scripts/split_dataset.py \
    --input  /data/uva/raw \
    --output /data/uva/split

# 1. "Preprocess"  (copies files into expected folder structure, no image changes)
python scripts/preprocess.py \
    --camera uva \
    --input  /data/uva/split \
    --output /data/uva/ready

# 2. Train
python scripts/train.py \
    --config configs/uva.yaml \
    --dataset-base /data/uva/ready \
    --output-dir   /runs/uva

# 3. Threshold sweep
python scripts/tune_thresholds.py \
    --config  configs/uva.yaml \
    --dataset /data/uva/ready

# 4. Evaluate
python scripts/evaluate.py --config configs/uva.yaml
```

---

## Evaluating with Pretrained Weights (No Training Required)

Download the weights, point the config at your preprocessed validation set, and evaluate:

```bash
python scripts/download_weights.py

python scripts/evaluate.py \
    --camera OG \
    --model  weights/og_best.pt \
    --images /data/og/preprocessed/images/val \
    --labels /data/og/preprocessed/labels/val \
    --conf 0.35 --iou 0.30
```

Run all four cameras with LaTeX output for the paper:

```bash
python scripts/evaluate.py \
    --config configs/og.yaml \
    --config configs/tau.yaml \
    --config configs/lb.yaml \
    --config configs/uva.yaml \
    --latex
```

---

## Python API

```python
# Step 0 — split
from wormml.datasets import split_dataset
split_dataset('/data/tau/raw', '/data/tau/split', train_ratio=0.8, seed=42)

# Step 1 — preprocess
from wormml.preprocess import preprocess_dataset, get_config
cfg = get_config('tau')
preprocess_dataset('/data/tau/split', '/data/tau/preprocessed', cfg)

# Step 2 — train
from wormml.train import TrainingConfig, train
cfg = TrainingConfig(camera='tau', dataset_base='/data/tau/preprocessed', output_dir='/runs/tau')
train(cfg)

# Step 3 — evaluate
from wormml.evaluate import EvalConfig, evaluate
results = evaluate([EvalConfig(
    camera='Tau',
    model_path='/runs/tau/yolov11_maxacc_11l/weights/best.pt',
    images_dir='/data/tau/preprocessed/images/val',
    labels_dir='/data/tau/preprocessed/labels/val',
    conf_thr=0.36, iou_thr=0.25, conf_boost=0.10,
)])
```

---

## Repository Structure

```
wormml/
├── wormml/                     # Core Python package
│   ├── preprocess/
│   │   ├── base.py             # Shared pipeline (blur → crop → mask → resize)
│   │   └── configs.py          # Per-camera config dataclasses
│   ├── train.py                # YOLOv11 training wrapper
│   ├── evaluate.py             # Hungarian-matched evaluation
│   ├── threshold.py            # Confidence/IoU grid search
│   └── datasets.py             # Split, validate, k-fold utilities
├── configs/
│   ├── og.yaml                 # OG hyperparameters + thresholds
│   ├── tau.yaml                # Tau hyperparameters + thresholds
│   ├── lb.yaml                 # LB hyperparameters + thresholds
│   └── uva.yaml                # UVA hyperparameters + thresholds
├── scripts/
│   ├── split_dataset.py        # Step 0: reproducible 80/20 split (seed 42)
│   ├── preprocess.py           # Step 1: camera-specific preprocessing
│   ├── train.py                # Step 2: YOLO training
│   ├── tune_thresholds.py      # Step 3: confidence/IoU sweep
│   ├── evaluate.py             # Step 4: evaluation + LaTeX tables
│   └── download_weights.py     # Download pretrained weights from HuggingFace
├── notebooks/
│   ├── 01_OG_pipeline.ipynb
│   ├── 02_Tau_pipeline.ipynb
│   ├── 03_LB_pipeline.ipynb
│   ├── 04_UVA_pipeline.ipynb
│   └── 05_cross_camera_evaluation.ipynb
├── weights/                    # Pretrained .pt files go here
└── docs/
    ├── dataset_format.md       # YOLO format and folder layout details
    └── camera_configs.md       # Parameter choices explained per camera
```

---

## Using Your Own Dataset

1. Collect plate images with YOLO `.txt` labels (one file per image, one line per worm)
2. Put them in the flat `images/` + `labels/` layout shown above
3. Run `split_dataset.py --seed 42 --train-ratio 0.8`
4. Pick the closest camera config YAML, update `dataset_base` and `output_dir`
5. Run preprocess → train → tune_thresholds → evaluate
6. See [`docs/camera_configs.md`](docs/camera_configs.md) for guidance on tuning
   circle-detection parameters for a new imaging system

---

## Evaluation Methodology

The evaluator runs YOLO inference on every validation image and computes:

- **Counting metrics** (all images): MAE, RMSE, within ±1/±2/±4
- **Detection metrics** (Hungarian matching): per-image precision, recall, F1 — macro and micro averaged
- **Adaptive confidence**: images with unexpectedly sparse or dense predictions are re-run at a boosted confidence threshold (per-camera values stored in the YAML configs)

Images where both P=0 and R=0 are excluded from detection metrics — they typically indicate annotation errors rather than model failures.

---

## Reproducing Paper Results

```bash
python scripts/download_weights.py

python scripts/evaluate.py \
    --config configs/og.yaml \
    --config configs/tau.yaml \
    --config configs/lb.yaml \
    --config configs/uva.yaml \
    --latex
```

---

## Citation

```bibtex
@misc{wormml2024,
  title  = {WormML: A Cross-Camera Pipeline for C. elegans Worm Counting},
  year   = {2024},
  note   = {\url{https://github.com/YOUR_USERNAME/wormml}}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
