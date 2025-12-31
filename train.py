"""
Crackathon Road Damage Detection Pipeline - Training Module

This module implements the complete training pipeline for YOLOv8-based
road damage detection. It handles model initialization, training
configuration, and training execution.

Architecture Justification - Why YOLOv8:
=========================================
1. State-of-the-art Performance:
   - YOLOv8 achieves superior mAP compared to previous YOLO versions
   - Competitive with two-stage detectors while being much faster
   
2. Anchor-free Design:
   - Better handles varying aspect ratios of cracks (long/thin vs compact)
   - Reduces hyperparameter tuning (no anchor box optimization needed)
   
3. CSPDarknet Backbone:
   - Cross-Stage Partial connections improve gradient flow
   - Efficient feature extraction with reduced computation
   
4. Path Aggregation Network (PAN):
   - Multi-scale feature fusion for detecting both small and large damage
   - Crucial for detecting both hairline cracks and large potholes
   
5. Decoupled Head:
   - Separate branches for classification and localization
   - Improves both classification accuracy and bounding box precision
   
6. Production Ready:
   - Extensive documentation and community support
   - Easy export to various deployment formats (ONNX, TensorRT, etc.)
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, List
import shutil

import yaml
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    Config, get_config, get_competition_config,
    PROJECT_ROOT, DATASET_ROOT, RUNS_DIR, OUTPUT_ROOT,
    CLASS_NAMES, NUM_CLASSES
)
from augmentation import get_yolo_augmentation_config


class YOLOTrainer:
    """
    YOLOv8 Training Pipeline for Road Damage Detection.
    
    This class encapsulates the complete training workflow:
    1. Model initialization with pretrained weights
    2. Training configuration setup
    3. Training execution with logging
    4. Model checkpointing and best model selection
    
    mAP Improvement Techniques Used:
    ================================
    1. Transfer Learning: Initialize from COCO pretrained weights
    2. Progressive Resizing: Start with smaller images, increase during training
    3. Cosine Learning Rate Schedule: Smooth decay prevents instability
    4. Heavy Augmentation: Mosaic, MixUp for regularization
    5. Early Stopping: Prevent overfitting with patience-based stopping
    6. Multi-scale Training: Random input sizes during training
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize YOLOv8 Trainer.
        
        Args:
            config: Pipeline configuration. Uses defaults if None.
        """
        self.config = config or get_config()
        
        # Paths
        self.data_yaml = DATASET_ROOT / "data.yaml"
        self.run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Model
        self.model = None
        self.results = None
        
        # Device
        self.device = self._setup_device()
        
        # Set random seeds for reproducibility
        self._set_seeds()
    
    def _setup_device(self) -> str:
        """
        Configure compute device (GPU/CPU).
        
        Returns:
            Device string for YOLO
        """
        if torch.cuda.is_available():
            device = self.config.training.device
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            return device
        else:
            print("No GPU available, using CPU (training will be slow)")
            return "cpu"
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        seed = self.config.training.seed
        
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        if self.config.training.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        print(f"Random seed set to {seed}")
    
    def load_model(self, weights: Optional[str] = None) -> 'YOLO':
        """
        Load YOLOv8 model.
        
        Model Selection Strategy:
        - yolov8n: Fastest, lowest mAP (~37% COCO)
        - yolov8s: Fast, good for real-time (~45% COCO)
        - yolov8m: Balanced speed/accuracy (~50% COCO) [DEFAULT]
        - yolov8l: High accuracy, slower (~53% COCO)
        - yolov8x: Highest accuracy, slowest (~54% COCO)
        
        Args:
            weights: Path to weights file or model name
            
        Returns:
            Loaded YOLO model
        """
        from ultralytics import YOLO
        
        weights = weights or self.config.model.model_name
        
        print("=" * 60)
        print("Loading YOLOv8 Model")
        print("=" * 60)
        print(f"Model: {weights}")
        print(f"Pretrained: {self.config.model.use_pretrained}")
        
        # Load model
        self.model = YOLO(weights)
        
        # Print model info
        print(f"\nModel Architecture:")
        print(f"  Parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
        print(f"  GFLOPs: {self.model.info(verbose=False)}")
        
        return self.model
    
    def _get_training_args(self) -> Dict[str, Any]:
        """
        Build training arguments dictionary for YOLO trainer.
        
        Hyperparameter Justification:
        =============================
        - epochs=100: Sufficient for convergence with early stopping
        - batch=16: Balances memory usage and gradient stability
        - imgsz=640: Standard YOLO input, good balance of speed/detail
        - optimizer=AdamW: Better weight decay handling than SGD
        - lr0=0.001: Conservative initial LR for pretrained model
        - lrf=0.01: Final LR = 0.00001 (smooth cosine decay)
        - patience=20: Early stopping if no improvement for 20 epochs
        
        Loss Function Weights:
        - box=7.5: Emphasize localization for precise bounding boxes
        - cls=0.5: Classification is easier with distinct damage types
        - dfl=1.5: Distribution focal loss for better box regression
        
        Returns:
            Dictionary of training arguments
        """
        tc = self.config.training
        mc = self.config.model
        aug = self.config.augmentation
        
        # Get augmentation configuration
        aug_config = get_yolo_augmentation_config(aug)
        
        training_args = {
            # Paths
            'data': str(self.data_yaml),
            'project': str(RUNS_DIR),
            'name': self.config.experiment.experiment_name,
            
            # Model
            'imgsz': mc.input_size,
            
            # Training
            'epochs': tc.epochs,
            'patience': tc.patience,
            'batch': tc.batch_size,
            'workers': tc.workers,
            
            # Optimizer
            'optimizer': tc.optimizer,
            'lr0': tc.lr0,
            'lrf': tc.lrf,
            'momentum': tc.momentum,
            'weight_decay': tc.weight_decay,
            
            # LR schedule
            'warmup_epochs': tc.warmup_epochs,
            'warmup_momentum': tc.warmup_momentum,
            'warmup_bias_lr': tc.warmup_bias_lr,
            'cos_lr': True,
            
            # Loss weights
            'box': tc.box_loss_weight,
            'cls': tc.cls_loss_weight,
            'dfl': tc.dfl_loss_weight,
            
            # Regularization
            'dropout': tc.dropout,
            'label_smoothing': tc.label_smoothing,
            
            # Augmentation
            **aug_config,
            
            # Training options
            'amp': tc.amp,
            'device': self.device,
            'seed': tc.seed,
            'deterministic': tc.deterministic,
            
            # Validation
            'val': True,
            'plots': True,
            
            # Checkpointing
            'save': True,
            'save_period': tc.save_period,
            
            # Logging
            'verbose': self.config.experiment.verbose,
        }
        
        return training_args
    
    def train(self, resume: bool = False, weights: Optional[str] = None) -> Dict:
        """
        Execute training pipeline.
        
        Training Workflow:
        1. Load model (pretrained or custom weights)
        2. Configure training parameters
        3. Run training with validation
        4. Save best model and training artifacts
        
        Args:
            resume: Resume from last checkpoint
            weights: Path to weights for initialization
            
        Returns:
            Training results dictionary
        """
        print("\n" + "=" * 60)
        print("STARTING TRAINING PIPELINE")
        print("=" * 60)
        
        # Verify data.yaml exists
        if not self.data_yaml.exists():
            raise FileNotFoundError(
                f"data.yaml not found at {self.data_yaml}. "
                "Run dataset_handler.py first to prepare the dataset."
            )
        
        # Load model
        self.load_model(weights)
        
        # Get training arguments
        train_args = self._get_training_args()
        
        # Print configuration
        print("\nTraining Configuration:")
        print("-" * 40)
        for key, value in sorted(train_args.items()):
            if not key.startswith('_'):
                print(f"  {key}: {value}")
        
        # Start training
        print("\n" + "=" * 60)
        print("TRAINING STARTED")
        print("=" * 60)
        
        self.results = self.model.train(**train_args, resume=resume)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        
        # Save training summary
        self._save_training_summary()
        
        return self.results
    
    def _save_training_summary(self):
        """Save training summary and metrics."""
        if self.results is None:
            return
        
        summary = {
            'config': {
                'model': self.config.model.model_variant,
                'epochs': self.config.training.epochs,
                'batch_size': self.config.training.batch_size,
                'learning_rate': self.config.training.lr0,
                'input_size': self.config.model.input_size,
            },
            'results': {
                'best_map50': float(self.results.results_dict.get('metrics/mAP50(B)', 0)),
                'best_map50_95': float(self.results.results_dict.get('metrics/mAP50-95(B)', 0)),
                'best_precision': float(self.results.results_dict.get('metrics/precision(B)', 0)),
                'best_recall': float(self.results.results_dict.get('metrics/recall(B)', 0)),
            },
            'paths': {
                'best_weights': str(self.results.save_dir / 'weights' / 'best.pt'),
                'last_weights': str(self.results.save_dir / 'weights' / 'last.pt'),
            }
        }
        
        summary_path = self.results.save_dir / 'training_summary.yaml'
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        print(f"\nTraining Summary:")
        print(f"  mAP@0.5: {summary['results']['best_map50']:.4f}")
        print(f"  mAP@0.5:0.95: {summary['results']['best_map50_95']:.4f}")
        print(f"  Precision: {summary['results']['best_precision']:.4f}")
        print(f"  Recall: {summary['results']['best_recall']:.4f}")
        print(f"\nBest weights saved to: {summary['paths']['best_weights']}")
    
    def validate(self, weights: Optional[str] = None) -> Dict:
        """
        Run validation on the validation set.
        
        Args:
            weights: Path to model weights. Uses best.pt if None.
            
        Returns:
            Validation results dictionary
        """
        from ultralytics import YOLO
        
        if weights is None:
            # Try to find best.pt from last training
            if self.results is not None:
                weights = str(self.results.save_dir / 'weights' / 'best.pt')
            else:
                raise ValueError("No weights specified and no training results available")
        
        print("\n" + "=" * 60)
        print("RUNNING VALIDATION")
        print("=" * 60)
        print(f"Weights: {weights}")
        
        model = YOLO(weights)
        results = model.val(
            data=str(self.data_yaml),
            imgsz=self.config.model.input_size,
            batch=self.config.evaluation.val_batch_size,
            conf=self.config.model.conf_threshold,
            iou=self.config.evaluation.nms_iou,
            device=self.device,
            plots=True,
            save_json=True
        )
        
        return results


class TrainingScheduler:
    """
    Advanced training scheduler for multi-stage training.
    
    Implements progressive training strategies:
    1. Progressive resizing: 320 -> 640 -> 1280
    2. Multi-scale training
    3. Learning rate restart
    """
    
    def __init__(self, config: Config):
        """Initialize scheduler with configuration."""
        self.config = config
        self.stages = self._define_stages()
    
    def _define_stages(self) -> List[Dict[str, Any]]:
        """
        Define training stages for progressive training.
        
        Progressive Training Strategy:
        - Stage 1: Small images (320px) for fast initial learning
        - Stage 2: Medium images (640px) for main training
        - Stage 3: Large images (960px) for fine details (optional)
        
        Returns:
            List of stage configurations
        """
        stages = [
            {
                'name': 'stage_1_warmup',
                'epochs': 10,
                'imgsz': 320,
                'lr0': 0.001,
                'description': 'Fast warmup with small images'
            },
            {
                'name': 'stage_2_main',
                'epochs': 80,
                'imgsz': 640,
                'lr0': 0.0001,
                'description': 'Main training at standard resolution'
            },
            {
                'name': 'stage_3_finetune',
                'epochs': 10,
                'imgsz': 960,
                'lr0': 0.00001,
                'description': 'Fine-tuning at high resolution'
            }
        ]
        return stages
    
    def run_progressive_training(self) -> Dict:
        """
        Execute progressive multi-stage training.
        
        Returns:
            Final training results
        """
        print("\n" + "=" * 60)
        print("PROGRESSIVE TRAINING SCHEDULE")
        print("=" * 60)
        
        for i, stage in enumerate(self.stages):
            print(f"\nStage {i+1}: {stage['name']}")
            print(f"  {stage['description']}")
        
        trainer = YOLOTrainer(self.config)
        weights = None
        
        for stage in self.stages:
            print(f"\n\n{'=' * 60}")
            print(f"STAGE: {stage['name']}")
            print(f"{'=' * 60}")
            
            # Update config for this stage
            self.config.training.epochs = stage['epochs']
            self.config.training.lr0 = stage['lr0']
            self.config.model.input_size = stage['imgsz']
            self.config.experiment.experiment_name = stage['name']
            
            # Train
            trainer = YOLOTrainer(self.config)
            results = trainer.train(weights=weights)
            
            # Use this stage's best weights for next stage
            weights = str(results.save_dir / 'weights' / 'best.pt')
        
        return results


def find_best_weights() -> Optional[Path]:
    """
    Find the best trained weights in the runs directory.
    
    Returns:
        Path to best.pt or None if not found
    """
    runs = sorted(RUNS_DIR.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if runs:
        return runs[-1]
    return None


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Road Damage Detection")
    parser.add_argument('--config', type=str, choices=['default', 'competition'],
                        default='default', help='Configuration preset')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to pretrained weights')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    parser.add_argument('--progressive', action='store_true',
                        help='Use progressive training schedule')
    
    args = parser.parse_args()
    
    # Select configuration
    if args.config == 'competition':
        config = get_competition_config()
        print("Using competition configuration (optimized for mAP)")
    else:
        config = get_config()
        print("Using default configuration")
    
    # Run training
    if args.progressive:
        scheduler = TrainingScheduler(config)
        results = scheduler.run_progressive_training()
    else:
        trainer = YOLOTrainer(config)
        results = trainer.train(resume=args.resume, weights=args.weights)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest weights: {find_best_weights()}")


if __name__ == "__main__":
    main()
