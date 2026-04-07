"""
Per-camera preprocessing configurations.

Each class encodes the imaging-system-specific tuning values that were
determined empirically during the WormML thesis experiments.  All numeric
defaults are preserved exactly from the original Colab notebooks.

Camera-specific differences
----------------------------
OG   – standard Hough detection, select largest circle, standard crop params.
Tau  – histogram equalisation before Hough, select closest-to-centre circle,
       wider padding, tighter radius bounds, 4 % radius inflation.
LB   – multi-pass Hough with centrality scoring, image inversion + brightness /
       blur / noise augmentation for domain adaptation, two input datasets.
UVA  – no preprocessing required; native YOLO training directly on raw images.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Base config (shared defaults)
# ---------------------------------------------------------------------------

@dataclass
class BasePreprocessConfig:
    # Blur
    gaussian_ksize: Tuple[int, int] = (3, 3)

    # Output size after crop + resize
    target_size: int = 1344

    # Parallel workers for I/O
    max_workers: int = 20

    # ---- Hough circle detection ----
    MAX_DETECT_DIM: int = 1200      # downscale large images for fast detection
    HOUGH_DP: float = 1.2
    HOUGH_PARAM1: int = 120         # Canny high threshold
    HOUGH_PARAM2: int = 35          # accumulator threshold
    MIN_R_FRAC: float = 0.30        # min radius as fraction of min(H, W)
    MAX_R_FRAC: float = 0.60        # max radius as fraction of min(H, W)

    # ---- Crop / mask ----
    PAD_FRAC: float = 0.04          # extra padding around detected radius
    RIM_CUT_FRAC: float = 0.03      # shrink mask radius to exclude dark rim
    BG_VALUE: int = 0               # pixel value outside circular mask

    # Circle-selection strategy: "largest_radius" | "closest_to_center"
    circle_selection: str = "largest_radius"

    # Tau-specific: inflate detected radius by this fraction before cropping
    radius_inflate: float = 0.0

    # Tau-specific: histogram equalisation before Hough
    use_histeq: bool = False

    # LB-specific: invert image colours
    apply_inversion: bool = False

    # LB-specific: extra augmentation (brightness / blur / noise) on train split
    apply_extra_aug_train: bool = False
    apply_extra_aug_val: bool = False

    # LB-specific: centrality weighting in circle scoring  (0 = radius-only)
    centrality_alpha: float = 0.0

    # LB-specific: multi-pass Hough fallback thresholds
    hough_fallback_passes: List[Tuple[int, int]] = field(default_factory=list)

    # LB-specific brightness augmentation
    brightness_adjust_range: Tuple[float, float] = (0.85, 1.15)
    brightness_shift_range: Tuple[float, float] = (-10, 10)

    # LB-specific blur augmentation
    blur_ksize_options: List[int] = field(default_factory=lambda: [3, 5, 7])
    blur_sigma_range: Tuple[float, float] = (0.5, 2.0)
    blur_prob: float = 0.5

    # LB-specific noise augmentation
    noise_prob: float = 0.3
    noise_sigma_range: Tuple[float, float] = (3, 8)

    # LB-specific: median blur kernel for Hough pre-processing
    median_blur_ksize: int = 5

    # Dataset split
    train_ratio: float = 0.8
    random_seed: int = 42

    # JPEG/PNG compression
    jpeg_quality: int = 95
    png_compression: int = 1


# ---------------------------------------------------------------------------
# Per-camera configs
# ---------------------------------------------------------------------------

@dataclass
class OGPreprocessConfig(BasePreprocessConfig):
    """
    OG (original) camera.

    Standard petri-dish crop pipeline with conservative augmentation.
    Circle selection by largest detected radius.  No inversion or extra aug.
    """
    HOUGH_PARAM1: int = 120
    HOUGH_PARAM2: int = 35
    MIN_R_FRAC: float = 0.30
    MAX_R_FRAC: float = 0.60
    PAD_FRAC: float = 0.04
    RIM_CUT_FRAC: float = 0.03
    circle_selection: str = "largest_radius"
    use_histeq: bool = False
    radius_inflate: float = 0.0
    median_blur_ksize: int = 5


@dataclass
class TauPreprocessConfig(BasePreprocessConfig):
    """
    Tau camera.

    Key differences vs OG:
    - Histogram equalisation before Hough (normalises dark-interior dishes).
    - Select closest-to-centre circle rather than largest (avoids glare artefacts).
    - Radius inflated 4 % to compensate for Hough underestimation on Tau images.
    - Wider crop padding (0.08) and tighter radius bounds (0.35–0.55).
    - Smaller rim cut (0.01) to preserve more of the dish edge.
    """
    HOUGH_PARAM1: int = 120
    HOUGH_PARAM2: int = 35
    MIN_R_FRAC: float = 0.35        # tighter lower bound vs OG 0.30
    MAX_R_FRAC: float = 0.55        # tighter upper bound vs OG 0.60
    PAD_FRAC: float = 0.08          # wider padding vs OG 0.04
    RIM_CUT_FRAC: float = 0.01      # preserve more edge vs OG 0.03
    circle_selection: str = "closest_to_center"
    use_histeq: bool = True
    radius_inflate: float = 0.04    # inflate 4 % after detection
    median_blur_ksize: int = 7      # larger blur for Tau vs OG's 5


@dataclass
class LBPreprocessConfig(BasePreprocessConfig):
    """
    LoopBio (LB) camera.

    LB images have a bright background with dark worms — the opposite polarity
    from OG/Tau.  Key differences:
    - Image inversion for domain adaptation (dark-on-bright → bright-on-dark).
    - Brightness, blur, and noise augmentation applied post-inversion.
    - Multi-pass Hough with centrality-weighted scoring.
    - Two source datasets merged and split together.
    - Slightly looser Hough params and wider radius bounds.
    """
    HOUGH_PARAM1: int = 140
    HOUGH_PARAM2: int = 40
    MIN_R_FRAC: float = 0.25
    MAX_R_FRAC: float = 0.80
    PAD_FRAC: float = 0.04
    RIM_CUT_FRAC: float = 0.03
    circle_selection: str = "centrality"
    centrality_alpha: float = 0.6
    use_histeq: bool = False
    apply_inversion: bool = True
    apply_extra_aug_train: bool = True
    apply_extra_aug_val: bool = False
    median_blur_ksize: int = 5
    # Brightness aug (applied AFTER inversion, so images are brighter)
    brightness_adjust_range: Tuple[float, float] = (0.85, 1.15)
    brightness_shift_range: Tuple[float, float] = (-10, 10)

    # Blur aug
    blur_ksize_options: List[int] = field(default_factory=lambda: [3, 5, 7])
    blur_sigma_range: Tuple[float, float] = (0.5, 2.0)
    blur_prob: float = 0.5

    # Noise aug
    noise_prob: float = 0.3
    noise_sigma_range: Tuple[float, float] = (3, 8)

    def __post_init__(self):
        # Override hough_fallback_passes default for LB
        if not self.hough_fallback_passes:
            self.hough_fallback_passes = [(120, 32), (100, 25)]


@dataclass
class UVAPreprocessConfig(BasePreprocessConfig):
    """
    UVA external dataset.

    UVA images were used for native YOLO training without any custom
    preprocessing (cropping, inversion, etc.).  This config exists for
    API completeness; when selected the pipeline simply copies images
    and labels into the expected train/val folder structure.
    """
    # No preprocessing — images are used as-is
    skip_preprocessing: bool = True


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

CAMERA_CONFIGS = {
    "og":  OGPreprocessConfig,
    "tau": TauPreprocessConfig,
    "lb":  LBPreprocessConfig,
    "uva": UVAPreprocessConfig,
}


def get_config(camera: str) -> BasePreprocessConfig:
    """
    Return a default preprocessing config for *camera*.

    Parameters
    ----------
    camera : str
        One of ``"og"``, ``"tau"``, ``"lb"``, ``"uva"`` (case-insensitive).

    Returns
    -------
    BasePreprocessConfig subclass instance
    """
    key = camera.lower()
    if key not in CAMERA_CONFIGS:
        raise ValueError(
            f"Unknown camera '{camera}'.  "
            f"Valid options: {list(CAMERA_CONFIGS.keys())}"
        )
    return CAMERA_CONFIGS[key]()
