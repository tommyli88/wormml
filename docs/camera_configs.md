# Camera Configurations

This document describes the empirically-tuned preprocessing and training
parameters for each supported imaging system, and provides guidance for
adapting the pipeline to a new camera.

---

## OG (Original Camera)

The OG dataset uses the most refined hyperparameter set.  Its preprocessing
parameters were used as the baseline from which all other cameras diverged.

### Key preprocessing parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `PAD_FRAC` | 0.04 | Extra padding around detected dish radius |
| `RIM_CUT_FRAC` | 0.03 | Rim excluded from circular mask |
| `MIN_R_FRAC` | 0.30 | Min dish radius as fraction of image min-dim |
| `MAX_R_FRAC` | 0.60 | Max dish radius |
| Circle selection | Largest radius | Standard — glare artefacts rare on OG |
| Histogram equalisation | Off | Interior brightness is consistent |

### Key training differences vs. previous versions

The OG config incorporates a set of **bug fixes** identified during thesis work:

| Parameter | Old value | Fixed value | Reason |
|-----------|-----------|-------------|--------|
| `warmup_epochs` | 10 | **3** | Long warmup suppressed early feature learning |
| `copy_paste` | 0.3 | **0.1** | High copy-paste can create unrealistic dense clusters |
| `box_loss` | 7.5 | **9.0** | Higher weight improves localisation of small worms |
| `lr0` | `0.001 × 0.8` | **0.001** | No reason to scale down; lr scheduler handles decay |
| `rect` | True | **False** | Rectangular padding harms dense small-object detection |
| `single_cls` | True | **Removed** | Redundant for single-class; can flatten confidence scores |

---

## Tau Camera

Tau images have variable interior brightness (bacterial lawns, condensate patches)
and strong rim reflections that cause the OG circle detector to pick glare artefacts
rather than the dish edge.

### Key preprocessing changes from OG

| Parameter | OG | Tau | Reason |
|-----------|-----|-----|--------|
| `PAD_FRAC` | 0.04 | **0.08** | Wider crop to avoid cutting dish edge |
| `RIM_CUT_FRAC` | 0.03 | **0.01** | Preserve more of the rim (less aggressive masking) |
| `MIN_R_FRAC` | 0.30 | **0.35** | Tighter bounds reduce false detections |
| `MAX_R_FRAC` | 0.60 | **0.55** | |
| Circle selection | Largest radius | **Closest to centre** | Glare artefacts produce off-centre large circles |
| Histogram equalisation | Off | **On** | Normalises dark-interior dishes before Hough |
| Radius inflate | 0 | **+4%** | Hough consistently underestimates radius on Tau |
| Median blur kernel | 5 | **7** | Larger blur for noisier Tau images |

### Tau-specific training note

The Tau training config uses the original (pre-OG-fixes) hyperparameter set.
This means `warmup_epochs=10`, `copy_paste=0.3`, `box_loss=7.5`, and
`lr0=0.0008` (0.001 × 0.8 scaling).  Whether applying the OG fixes would
improve Tau performance is an open ablation study.

---

## LoopBio (LB)

LB images have **inverted polarity** — worms are dark on a bright background,
the opposite of OG/Tau.  The preprocessing pipeline handles this via:

1. **Image inversion** — `255 − pixel` for foreground pixels, background stays 0
2. **Post-inversion augmentation** — after inversion images resemble OG/Tau, so
   extra brightness, blur, and noise augmentation is applied to the training set
   to increase domain coverage

### LB circle detection

The LB Hough detector uses a **centrality-weighted scoring** function rather
than selecting by largest radius:

```
score(circle) = radius − alpha × distance_from_image_centre
```

With `alpha=0.6`, this penalises off-centre detections while still preferring
larger circles.  A **multi-pass fallback** tries progressively looser Hough
parameters if the primary pass fails.

### LB adaptive confidence boosts

LB requires larger confidence boosts than other cameras in both sparse and
dense prediction regimes:

| Situation | Threshold | Boost |
|-----------|-----------|-------|
| Sparse predictions (count < 10) | conf + 0.175 | Reduce false negatives |
| Dense predictions (count > 80) | conf + 0.180 | Reduce false positives |

---

## UVA (External Dataset)

UVA data was provided by an external collaborator and was used for native YOLO
training without custom preprocessing.  The preprocessing step simply copies
images into the expected train/val folder structure.

Key difference: `patience=15` (reduced from 25) because the UVA model converges
faster on the external data distribution.

---

## Adding a New Camera

To adapt WormML to a new imaging system:

1. **Inspect sample images** — determine polarity (dark/bright worms), typical
   plate radius as a fraction of image size, and whether rim reflections are present.

2. **Choose circle selection strategy**
   - Glare artefacts off-centre → use `closest_to_center`
   - Variable brightness interior → enable `use_histeq`
   - Need centrality weighting → use `centrality` with tuned `centrality_alpha`

3. **Tune Hough parameters** — adjust `MIN_R_FRAC`, `MAX_R_FRAC`, `PAD_FRAC`,
   `RIM_CUT_FRAC` by running a few test images through `get_crop_params()` and
   visually inspecting the crop output.

4. **Create a new config** — subclass `BasePreprocessConfig` or copy the closest
   existing config and modify it.  Add to `CAMERA_CONFIGS` in `configs.py`.

5. **Add a YAML** — copy `configs/og.yaml` and update all values.

6. **Run threshold sweep** — after training, always run `tune_thresholds.py`
   rather than guessing; the optimal thresholds vary significantly by camera.
