"""
Crackathon Road Damage Detection Pipeline - Configuration

This module contains all configuration parameters for the pipeline.
Centralizing configs ensures reproducibility and easy hyperparameter tuning.

Class Mapping (YOLO format):
    0: Longitudinal crack - Linear cracks parallel to road direction
    1: Transverse crack - Linear cracks perpendicular to road direction  
    2: Alligator crack - Interconnected crack patterns resembling alligator skin
    3: Other corruption - Miscellaneous road surface damage
    4: Pothole - Circular/irregular depressions in road surface
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data"
DATASET_ROOT = DATA_ROOT / "rdd2022"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
RUNS_DIR = OUTPUT_ROOT / "runs"
PREDICTIONS_DIR = OUTPUT_ROOT / "predictions"

GDRIVE_FOLDER_ID = "1JpBQ5haJCvPhD-0jUdir3GiGNbBnah93"
GDRIVE_URL = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"

CLASS_NAMES = {
    0: "Longitudinal_crack",
    1: "Transverse_crack", 
    2: "Alligator_crack",
    3: "Other_corruption",
    4: "Pothole"
}

NUM_CLASSES = len(CLASS_NAMES)

CLASS_DESCRIPTIONS = {
    0: "Linear cracks running parallel to the road direction, typically caused by fatigue or reflection cracking",
    1: "Linear cracks running perpendicular to the road direction, often due to thermal contraction",
    2: "Interconnected crack patterns resembling alligator skin, indicating structural failure",
    3: "Miscellaneous road surface damage not fitting other categories",
    4: "Bowl-shaped depressions in the road surface, often caused by water infiltration and traffic"
}

@dataclass
class ModelConfig:
    """
    YOLOv8 Model Configuration
    
    Architecture Justification:
    - YOLOv8 achieves state-of-the-art mAP with efficient inference
    - Anchor-free design better handles varying crack aspect ratios
    - CSPDarknet backbone provides strong feature extraction
    - Path Aggregation Network (PAN) enables multi-scale detection
    - Decoupled head improves classification and localization
    
    Model Selection Rationale:
    - YOLOv8m balances accuracy and training speed
    - Sufficient capacity for 5-class detection
    - Pretrained COCO weights provide transfer learning benefits
    """
    model_name: str = "yolov8m.pt"
    model_variant: str = "yolov8m"
    
    input_size: int = 640
    
    num_classes: int = NUM_CLASSES
    
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_det: int = 300
    
    use_pretrained: bool = True
    freeze_backbone: int = 0


@dataclass
class TrainingConfig:
    """
    Training Pipeline Configuration
    
    Hyperparameter Selection Rationale:
    - Batch size 8: Optimized for 8GB VRAM (RTX 4070) with YOLOv8m
    - 100 epochs: Sufficient for convergence with early stopping
    - AdamW optimizer: Better weight decay handling than Adam
    - Cosine LR schedule: Smooth decay prevents training instability
    - Warmup: Prevents early gradient explosion
    
    VRAM Note: YOLOv8m @ batch=8, imgsz=640 fits in 8GB VRAM safely.
               Batch=16 risks OOM crash around epoch 3-10.
    """
    epochs: int = 100
    patience: int = 20
    
    batch_size: int = 8
    workers: int = 8
    
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    
    box_loss_weight: float = 7.5
    cls_loss_weight: float = 0.5
    dfl_loss_weight: float = 1.5
    
    dropout: float = 0.0
    label_smoothing: float = 0.0
    
    amp: bool = True
    
    save_period: int = 10
    
    device: str = "0"
    
    seed: int = 42
    deterministic: bool = True


@dataclass  
class AugmentationConfig:
    """
    Data Augmentation Configuration for Road Imagery
    
    Strategy Rationale:
    Road damage detection faces unique challenges:
    1. Varying lighting conditions (day/night, shadows, overcast)
    2. Different camera angles and perspectives
    3. Motion blur from vehicle-mounted cameras
    4. Scale variation (close-up vs distant cracks)
    5. Weather effects (wet roads, reflections)
    
    Augmentations are tuned to simulate these real-world conditions
    while avoiding unrealistic transforms that could hurt generalization.
    """
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    
    degrees: float = 10.0
    
    translate: float = 0.1
    
    scale: float = 0.5
    
    shear: float = 5.0
    
    perspective: float = 0.0005
    
    flipud: float = 0.0
    fliplr: float = 0.5
    
    mosaic: float = 1.0
    
    mixup: float = 0.1
    
    copy_paste: float = 0.0
    
    motion_blur_prob: float = 0.2
    motion_blur_limit: int = 7
    
    gaussian_blur_prob: float = 0.1
    gaussian_blur_limit: tuple = (3, 7)
    
    brightness_contrast_prob: float = 0.3
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    
    shadow_prob: float = 0.2
    
    rain_prob: float = 0.1
    fog_prob: float = 0.1
    
    clahe_prob: float = 0.2
    clahe_clip_limit: float = 4.0
    
    compression_prob: float = 0.1
    compression_quality_lower: int = 70
    
    noise_prob: float = 0.1
    noise_var_limit: tuple = (10.0, 50.0)


@dataclass
class EvaluationConfig:
    """
    Evaluation and Validation Configuration
    
    mAP Calculation:
    - Uses COCO-style mAP with IoU thresholds from 0.5 to 0.95
    - Primary metric: mAP@0.5 for competition ranking
    - Also tracks mAP@0.5:0.95 for comprehensive evaluation
    
    NMS Strategy:
    - IoU threshold 0.45 balances duplicate removal and valid detection retention
    - Class-agnostic NMS disabled to preserve multi-class detections
    """
    val_batch_size: int = 32
    
    iou_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75])
    
    conf_thresholds: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75])
    
    nms_iou: float = 0.45
    agnostic_nms: bool = False
    
    plot_confusion_matrix: bool = True
    plot_pr_curve: bool = True
    save_val_predictions: bool = True
    
    per_class_metrics: bool = True


@dataclass
class InferenceConfig:
    """
    Inference Pipeline Configuration
    
    Submission Format:
    Each prediction file contains lines with:
    <class_id> <x_center> <y_center> <width> <height> <confidence>
    
    All coordinates are normalized (0-1 range).
    """
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    
    batch_size: int = 32
    
    use_tta: bool = False
    tta_scales: List[float] = field(default_factory=lambda: [0.83, 1.0, 1.17])
    tta_flips: bool = True
    
    use_ensemble: bool = False
    ensemble_weights: List[float] = field(default_factory=lambda: [1.0])
    
    save_visualizations: bool = True
    visualization_conf: float = 0.3
    

@dataclass
class ExperimentConfig:
    """Experiment tracking and logging configuration"""
    experiment_name: str = "crackathon_baseline"
    run_name: Optional[str] = None
    
    log_metrics_every: int = 1
    verbose: bool = True
    
    save_best_only: bool = True
    save_last: bool = True
    

@dataclass
class Config:
    """Master configuration combining all sub-configs"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    # Path attributes for compatibility with inference.py
    DATASET_DIR: Path = field(default_factory=lambda: DATASET_ROOT)
    
    def __post_init__(self):
        """Create necessary directories"""
        for path in [DATA_ROOT, DATASET_ROOT, OUTPUT_ROOT, RUNS_DIR, PREDICTIONS_DIR]:
            path.mkdir(parents=True, exist_ok=True)


# Create default configuration instance
def get_config() -> Config:
    """Returns the default configuration"""
    return Config()


# Competition-optimized configuration
def get_competition_config() -> Config:
    """
    Returns configuration optimized for competition performance.
    
    Key differences from baseline:
    - YOLOv8m (best mAP/time ratio - what strong teams run)
    - More epochs with patience
    - More aggressive augmentation
    - Test-time augmentation enabled
    
    Model Choice Rationale (non-negotiable):
    - v8n: too weak
    - v8l: overkill, slower, diminishing returns  
    - v8m: best mAP / time ratio
    """
    config = Config()
    
    config.model.model_name = "yolov8m.pt"
    config.model.model_variant = "yolov8m"
    config.model.input_size = 640
    
    config.training.epochs = 150
    config.training.patience = 30
    config.training.batch_size = 8
    
    # Lower confidence for higher recall
    config.model.conf_threshold = 0.2
    config.inference.conf_threshold = 0.2
    
    # Enable TTA for inference
    config.inference.use_tta = True
    
    return config


if __name__ == "__main__":
    # Print configuration for verification
    config = get_config()
    print("=" * 60)
    print("Crackathon Pipeline Configuration")
    print("=" * 60)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Dataset Root: {DATASET_ROOT}")
    print(f"Output Root: {OUTPUT_ROOT}")
    print(f"\nClasses ({NUM_CLASSES}):")
    for idx, name in CLASS_NAMES.items():
        print(f"  {idx}: {name}")
    print(f"\nModel: {config.model.model_variant}")
    print(f"Input Size: {config.model.input_size}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch Size: {config.training.batch_size}")
    print(f"Learning Rate: {config.training.lr0}")
