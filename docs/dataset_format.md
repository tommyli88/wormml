# Dataset Format

WormML expects YOLO-format datasets: one `.txt` label file per image, each line
representing one worm annotation in normalised `[class cx cy w h]` format.

## Standard Directory Layout

```
dataset_root/
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
    │   ├── plate_001.txt
    │   ├── plate_002.txt
    │   └── ...
    └── val/
        ├── plate_101.txt
        └── ...
```

## Label File Format

Each `.txt` file contains one line per annotated worm:

```
0 0.512345 0.487654 0.023456 0.031234
0 0.234567 0.765432 0.021000 0.028000
```

Fields: `class_id  cx  cy  width  height` — all normalised to [0, 1] relative
to image dimensions.  For WormML the class ID is always `0` (worm).

## Starting from a Flat (Unsplit) Dataset

If your raw data has not been split into train/val, provide a flat layout:

```
raw_dataset/
├── images/
│   ├── plate_001.jpg
│   └── ...
└── labels/
    ├── plate_001.txt
    └── ...
```

Then use `--split-first` on the preprocess script:

```bash
python scripts/preprocess.py \
    --camera tau \
    --input  /data/tau/raw \
    --output /data/tau/preprocessed \
    --split-first \
    --train-ratio 0.8 \
    --seed 42
```

This creates a stratified 80/20 split and writes it to the expected structure.

## LB Two-Dataset Merge

LoopBio preprocessing accepts two source roots that are combined before splitting:

```bash
python scripts/preprocess.py \
    --camera lb \
    --input  /data/lb/lbog \
    --input2 /data/lb/lb200 \
    --output /data/lb/preprocessed
```

Both `lbog` and `lb200` must use the flat layout above.

## data.yaml

After preprocessing, a `data.yaml` is automatically created in the output root:

```yaml
path: /data/og/preprocessed
train: images/train
val:   images/val
nc: 1
names: [worm]
```

This file is read by YOLO during training.  You can add custom augmentation
keys here, but the default values are already encoded in the training config.
