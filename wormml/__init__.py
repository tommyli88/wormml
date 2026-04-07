"""
WormML — Cross-camera C. elegans worm counting pipeline.

Supports OG, Tau, LoopBio (LB), and UVA imaging systems via a unified
training, evaluation, and preprocessing API built on YOLOv11.
"""

__version__ = "1.0.0"
__author__ = "WormML Contributors"

from wormml.train import train
from wormml.evaluate import evaluate
from wormml.threshold import sweep_thresholds

__all__ = ["train", "evaluate", "sweep_thresholds"]
