#!/usr/bin/env python3
"""
Download pretrained WormML weights from Hugging Face Hub.

Fetches the four camera-specific YOLOv11-Large checkpoints and places them
in the local weights/ directory, ready for evaluation or fine-tuning.

Usage
-----
  # Download all four camera weights
  python scripts/download_weights.py

  # Download a specific camera only
  python scripts/download_weights.py --camera og
  python scripts/download_weights.py --camera tau
  python scripts/download_weights.py --camera lb
  python scripts/download_weights.py --camera uva

Requirements
------------
  pip install huggingface_hub
"""

import argparse
import sys
from pathlib import Path

# ── Hugging Face repo ──────────────────────────────────────────────────────────
HF_REPO_ID = "litommy88/wormml"   # update if repo is moved

WEIGHTS = {
    "og":  "og_best.pt",
    "tau": "tau_best.pt",
    "lb":  "lb_best.pt",
    "uva": "uva_best.pt",
}

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


def download(camera: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is not installed.")
        print("Run: pip install huggingface_hub")
        sys.exit(1)

    filename = WEIGHTS[camera]
    out_path = WEIGHTS_DIR / filename

    if out_path.exists():
        print(f"  ✅ {filename} already exists — skipping download.")
        return out_path

    print(f"  ⬇️  Downloading {filename} from {HF_REPO_ID} …")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    local = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        local_dir=str(WEIGHTS_DIR),
    )
    print(f"  ✅ Saved to {local}")
    return Path(local)


def main() -> None:
    p = argparse.ArgumentParser(description="Download WormML pretrained weights")
    p.add_argument(
        "--camera",
        choices=list(WEIGHTS.keys()),
        default=None,
        help="Download one specific camera (default: all four)",
    )
    args = p.parse_args()

    cameras = [args.camera] if args.camera else list(WEIGHTS.keys())

    print(f"Downloading weights from: https://huggingface.co/{HF_REPO_ID}")
    print(f"Saving to: {WEIGHTS_DIR}\n")

    for cam in cameras:
        download(cam)

    print(f"\nDone. Update configs/<camera>.yaml → model_path to point to weights/.")


if __name__ == "__main__":
    main()
