---
license: mit
tags:
  - object-detection
  - yolo
  - biology
  - computer-vision
  - c-elegans
  - worm-counting
base_model: ultralytics/assets
---

# WormML — YOLOv11 Worm Counting Weights

Pretrained YOLOv11-Large checkpoints for counting *C. elegans* worms across four imaging systems. Each model was trained on images from a specific camera and should be used with its matching preprocessing pipeline.

## Models

| File | Camera | Preprocessing |
|------|--------|---------------|
| `og_best.pt`  | OG (original lab microscope) | Hough circle crop → resize 1344×1344 |
| `tau_best.pt` | Tau imaging system | Histogram equalisation + circle crop |
| `lb_best.pt`  | LoopBio automated platform | Circle crop + colour inversion + augmentation |
| `uva_best.pt` | UVA external dataset | No preprocessing |

## Usage

Download all weights with one command using the [WormML repository](https://github.com/tommyli88/wormml):

```bash
git clone https://github.com/tommyli88/wormml.git
cd wormml
pip install -r requirements.txt
python scripts/download_weights.py
```

Or download a single camera:

```bash
python scripts/download_weights.py --camera og
```

Run inference on a preprocessed image:

```python
from ultralytics import YOLO

model = YOLO("weights/og_best.pt")
results = model("plate_image.jpg", conf=0.35, iou=0.30)
print(f"Worm count: {len(results[0].boxes)}")
```

## Recommended Thresholds

These confidence and IoU thresholds were tuned on each camera's validation set to minimise mean absolute error:

| Camera | Confidence | IoU  |
|--------|-----------|------|
| OG     | 0.35      | 0.30 |
| Tau    | 0.36      | 0.25 |
| LB     | 0.265     | 0.30 |
| UVA    | 0.32      | 0.30 |

## Training

All models use YOLOv11-Large (`yolo11l.pt`) trained for 100 epochs. Camera-specific hyperparameters (warmup epochs, box loss weight, learning rate, augmentation) are documented in the [configs](https://github.com/tommyli88/wormml/tree/main/configs) folder of the main repository.

## Citation

```bibtex
@misc{wormml2024,
  title  = {WormML: A Cross-Camera Pipeline for C. elegans Worm Counting},
  year   = {2024},
  note   = {\url{https://github.com/tommyli88/wormml}}
}
```
