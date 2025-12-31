"""
Crackathon Road Damage Detection Pipeline

A complete end-to-end object detection pipeline for the Crackathon challenge
using YOLOv8 on the Road Damage Detection 2022 dataset.

Modules:
    config: Configuration parameters for all pipeline components
    dataset_handler: Dataset download, organization, and validation
    augmentation: Data augmentation strategies for road imagery
    train: YOLOv8 training pipeline
    evaluate: Model evaluation and metrics
    inference: Inference and submission generation
    utils: Utility functions for visualization and file handling

Quick Start:
    from crackathon import main
    # Run full pipeline
    # python main.py --full
"""

from .config import (
    Config,
    get_config,
    get_competition_config,
    CLASS_NAMES,
    NUM_CLASSES,
    PROJECT_ROOT,
    DATASET_ROOT,
    OUTPUT_ROOT
)

__version__ = "1.0.0"
__author__ = "Crackathon Team"

__all__ = [
    'Config',
    'get_config',
    'get_competition_config',
    'CLASS_NAMES',
    'NUM_CLASSES',
    'PROJECT_ROOT',
    'DATASET_ROOT',
    'OUTPUT_ROOT'
]
