"""
Crackathon Road Damage Detection Pipeline - Main Runner

This is the main entry point for the complete end-to-end pipeline.
It orchestrates all stages: dataset preparation, training, evaluation,
inference, and submission generation.

Usage Examples:
===============
# Full pipeline (download, train, evaluate, generate submission)
python main.py --full

# Prepare dataset only
python main.py --prepare

# Train model only (requires prepared dataset)
python main.py --train

# Evaluate model only (requires trained model)
python main.py --evaluate

# Generate submission only (requires trained model)
python main.py --submit

# Competition mode (optimized settings)
python main.py --full --competition
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    Config, get_config, get_competition_config,
    PROJECT_ROOT, DATASET_ROOT, OUTPUT_ROOT, RUNS_DIR, PREDICTIONS_DIR,
    CLASS_NAMES, NUM_CLASSES
)


def print_banner():
    """Print the pipeline banner."""
    print("\n" + "=" * 60)
    print("CRACKATHON - Road Damage Detection Pipeline")
    print("YOLOv8 + Ultralytics")
    print("=" * 60)


def print_config_summary(config: Config):
    """Print configuration summary."""
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Dataset Root: {DATASET_ROOT}")
    print(f"Output Root: {OUTPUT_ROOT}")
    
    print(f"\nClasses ({NUM_CLASSES}):")
    for idx, name in CLASS_NAMES.items():
        print(f"  {idx}: {name}")
    
    print(f"\nModel Configuration:")
    print(f"  Architecture: {config.model.model_variant}")
    print(f"  Input Size: {config.model.input_size}x{config.model.input_size}")
    print(f"  Pretrained: {config.model.use_pretrained}")
    print(f"  Conf Threshold: {config.model.conf_threshold}")
    print(f"  IoU Threshold: {config.model.iou_threshold}")
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Batch Size: {config.training.batch_size}")
    print(f"  Learning Rate: {config.training.lr0}")
    print(f"  Optimizer: {config.training.optimizer}")
    print(f"  Early Stopping Patience: {config.training.patience}")
    
    print(f"\nInference Configuration:")
    print(f"  TTA Enabled: {config.inference.use_tta}")
    print(f"  Conf Threshold: {config.inference.conf_threshold}")


def prepare_dataset(config: Config) -> bool:
    """
    Stage 1: Dataset Preparation
    
    Downloads, organizes, and validates the RDD2022 dataset.
    """
    print("\n" + "=" * 60)
    print("STAGE 1: DATASET PREPARATION")
    print("=" * 60)
    
    try:
        from dataset_handler import DatasetHandler
        
        handler = DatasetHandler()
        data_yaml = handler.prepare_dataset(
            download=True,
            verify=True,
            create_split=True,
            val_ratio=0.2
        )
        
        print(f"\n✓ Dataset prepared successfully!")
        print(f"  data.yaml: {data_yaml}")
        return True
        
    except Exception as e:
        print(f"\n✗ Dataset preparation failed: {e}")
        traceback.print_exc()
        return False


def train_model(config: Config, 
                weights: str = None, 
                resume: bool = False) -> str:
    """
    Stage 2: Model Training
    
    Trains YOLOv8 model on the prepared dataset.
    
    Returns:
        Path to best trained weights
    """
    print("\n" + "=" * 60)
    print("STAGE 2: MODEL TRAINING")
    print("=" * 60)
    
    try:
        from train import YOLOTrainer, find_best_weights
        
        trainer = YOLOTrainer(config)
        results = trainer.train(resume=resume, weights=weights)
        
        # Get best weights path
        best_weights = str(results.save_dir / 'weights' / 'best.pt')
        
        print(f"\n✓ Training completed successfully!")
        print(f"  Best weights: {best_weights}")
        print(f"  mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
        
        return best_weights
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        traceback.print_exc()
        return None


def evaluate_model(config: Config, weights: str) -> dict:
    """
    Stage 3: Model Evaluation
    
    Evaluates the trained model on validation set.
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "=" * 60)
    print("STAGE 3: MODEL EVALUATION")
    print("=" * 60)
    
    try:
        from evaluate import Evaluator
        
        evaluator = Evaluator(config)
        metrics = evaluator.evaluate_model(
            weights=weights,
            conf_threshold=config.model.conf_threshold,
            iou_threshold=config.evaluation.nms_iou
        )
        
        print(f"\n✓ Evaluation completed successfully!")
        return metrics
        
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        traceback.print_exc()
        return None


def generate_submission(config: Config, 
                        weights: str,
                        use_tta: bool = False) -> str:
    """
    Stage 4: Submission Generation
    
    Runs inference on test set and generates submission.zip.
    
    Returns:
        Path to submission.zip
    """
    print("\n" + "=" * 60)
    print("STAGE 4: SUBMISSION GENERATION")
    print("=" * 60)
    
    try:
        from inference import SubmissionGenerator
        
        generator = SubmissionGenerator(config)
        submission_path = generator.generate_submission(
            weights=weights,
            use_tta=use_tta,
            conf_threshold=config.inference.conf_threshold,
            iou_threshold=config.inference.iou_threshold
        )
        
        print(f"\n✓ Submission generated successfully!")
        print(f"  Submission: {submission_path}")
        return str(submission_path)
        
    except Exception as e:
        print(f"\n✗ Submission generation failed: {e}")
        traceback.print_exc()
        return None


def find_latest_weights() -> str:
    """Find the latest best.pt weights file."""
    from train import find_best_weights
    
    weights = find_best_weights()
    if weights:
        return str(weights)
    return None


def run_full_pipeline(config: Config, 
                      use_tta: bool = False,
                      skip_download: bool = False) -> dict:
    """
    Run the complete pipeline end-to-end.
    
    Stages:
    1. Dataset preparation
    2. Model training
    3. Model evaluation
    4. Submission generation
    
    Args:
        config: Pipeline configuration
        use_tta: Use Test-Time Augmentation for submission
        skip_download: Skip dataset download if already available
        
    Returns:
        Dictionary with paths to outputs and metrics
    """
    print_banner()
    print_config_summary(config)
    
    results = {
        'success': False,
        'weights': None,
        'metrics': None,
        'submission': None,
        'errors': []
    }
    
    start_time = datetime.now()
    
    # Stage 1: Dataset Preparation
    if not skip_download or not DATASET_ROOT.exists():
        if not prepare_dataset(config):
            results['errors'].append("Dataset preparation failed")
            return results
    else:
        print("\n[SKIP] Dataset already exists, skipping preparation")
    
    # Stage 2: Model Training
    weights = train_model(config)
    if weights is None:
        results['errors'].append("Model training failed")
        return results
    results['weights'] = weights
    
    # Stage 3: Model Evaluation
    metrics = evaluate_model(config, weights)
    if metrics:
        results['metrics'] = metrics
    else:
        results['errors'].append("Model evaluation failed (non-critical)")
    
    # Stage 4: Submission Generation
    submission = generate_submission(config, weights, use_tta)
    if submission:
        results['submission'] = submission
    else:
        results['errors'].append("Submission generation failed")
        return results
    
    results['success'] = True
    
    # Print final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nDuration: {duration}")
    print(f"\nOutputs:")
    print(f"  Best Weights: {results['weights']}")
    if results['metrics']:
        print(f"  mAP@0.5: {results['metrics']['overall']['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95: {results['metrics']['overall']['mAP50_95']:.4f}")
    print(f"  Submission: {results['submission']}")
    
    if results['errors']:
        print(f"\nWarnings: {len(results['errors'])}")
        for err in results['errors']:
            print(f"  - {err}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Crackathon Road Damage Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --full              # Run complete pipeline
  python main.py --full --competition  # Competition mode (optimized)
  python main.py --prepare           # Prepare dataset only
  python main.py --train             # Train model only
  python main.py --evaluate          # Evaluate model only
  python main.py --submit            # Generate submission only
  python main.py --submit --tta      # Submission with TTA
        """
    )
    
    # Pipeline stages
    parser.add_argument('--full', action='store_true',
                        help='Run complete pipeline (prepare, train, evaluate, submit)')
    parser.add_argument('--prepare', action='store_true',
                        help='Prepare dataset only')
    parser.add_argument('--train', action='store_true',
                        help='Train model only')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate model only')
    parser.add_argument('--submit', action='store_true',
                        help='Generate submission only')
    
    # Configuration
    parser.add_argument('--competition', action='store_true',
                        help='Use competition-optimized configuration')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to model weights (for evaluate/submit)')
    
    # Training options
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of training epochs')
    parser.add_argument('--batch', type=int, default=None,
                        help='Override batch size')
    
    # Inference options
    parser.add_argument('--tta', action='store_true',
                        help='Use Test-Time Augmentation')
    parser.add_argument('--conf', type=float, default=None,
                        help='Override confidence threshold')
    
    # Utility
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip dataset download if already exists')
    
    args = parser.parse_args()
    
    # Select configuration
    if args.competition:
        config = get_competition_config()
        print("Using COMPETITION configuration (optimized for mAP)")
    else:
        config = get_config()
        print("Using DEFAULT configuration")
    
    # Apply overrides
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch:
        config.training.batch_size = args.batch
    if args.conf:
        config.model.conf_threshold = args.conf
        config.inference.conf_threshold = args.conf
    if args.tta:
        config.inference.use_tta = True
    
    # Run requested stages
    if args.full:
        run_full_pipeline(config, use_tta=args.tta, skip_download=args.skip_download)
        
    elif args.prepare:
        print_banner()
        prepare_dataset(config)
        
    elif args.train:
        print_banner()
        train_model(config, weights=args.weights, resume=args.resume)
        
    elif args.evaluate:
        print_banner()
        weights = args.weights or find_latest_weights()
        if weights:
            evaluate_model(config, weights)
        else:
            print("ERROR: No weights found. Train a model first or specify --weights")
            
    elif args.submit:
        print_banner()
        weights = args.weights or find_latest_weights()
        if weights:
            generate_submission(config, weights, use_tta=args.tta)
        else:
            print("ERROR: No weights found. Train a model first or specify --weights")
            
    else:
        # Default: print help
        parser.print_help()
        print("\n\nQuick start: python main.py --full")


if __name__ == "__main__":
    main()
