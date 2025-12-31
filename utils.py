"""
Crackathon Road Damage Detection Pipeline - Utilities

Visualization, file handling, metric calculations, and logging helpers.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import random
import json

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import CLASS_NAMES, NUM_CLASSES, OUTPUT_ROOT


# Visualization
CLASS_COLORS = {
    0: (255, 100, 100),   # Light Blue - Longitudinal
    1: (100, 255, 100),   # Light Green - Transverse
    2: (100, 100, 255),   # Light Red - Alligator
    3: (255, 255, 100),   # Cyan - Other
    4: (255, 100, 255),   # Magenta - Pothole
}


def draw_bboxes(image: np.ndarray,
                detections: List[Dict],
                class_names: Dict[int, str] = None,
                colors: Dict[int, Tuple] = None,
                thickness: int = 2,
                font_scale: float = 0.5) -> np.ndarray:
    """
    Draw bounding boxes and labels on image.
    
    Args:
        image: Input image (BGR format)
        detections: List of detection dicts with keys:
                   class_id, x_center, y_center, width, height, confidence
        class_names: Mapping of class_id to name
        colors: Mapping of class_id to BGR color
        thickness: Box line thickness
        font_scale: Label font scale
        
    Returns:
        Image with drawn bounding boxes
    """
    class_names = class_names or CLASS_NAMES
    colors = colors or CLASS_COLORS
    
    img = image.copy()
    h, w = img.shape[:2]
    
    for det in detections:
        # Convert normalized coordinates to pixels
        x_center = det['x_center'] * w
        y_center = det['y_center'] * h
        box_w = det['width'] * w
        box_h = det['height'] * h
        
        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)
        
        class_id = det['class_id']
        confidence = det.get('confidence', 1.0)
        
        # Get color
        color = colors.get(class_id, (128, 128, 128))
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label background
        label = f"{class_names.get(class_id, f'cls_{class_id}')}: {confidence:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        
        cv2.rectangle(
            img, 
            (x1, y1 - label_h - baseline - 5), 
            (x1 + label_w, y1), 
            color, 
            -1
        )
        
        # Draw label text
        cv2.putText(
            img, label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1, cv2.LINE_AA
        )
    
    return img


def visualize_comparison(image_path: str,
                         ground_truth: List[Dict],
                         predictions: List[Dict],
                         output_path: Optional[str] = None,
                         show: bool = True) -> np.ndarray:
    """
    Create side-by-side comparison of ground truth and predictions.
    
    Args:
        image_path: Path to input image
        ground_truth: List of ground truth detections
        predictions: List of predicted detections
        output_path: Optional path to save visualization
        show: Whether to display the image
        
    Returns:
        Combined visualization image
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    # Draw ground truth
    gt_img = draw_bboxes(image.copy(), ground_truth)
    cv2.putText(gt_img, "Ground Truth", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw predictions
    pred_img = draw_bboxes(image.copy(), predictions)
    cv2.putText(pred_img, "Predictions", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Combine side by side
    combined = np.hstack([gt_img, pred_img])
    
    if output_path:
        cv2.imwrite(output_path, combined)
    
    if show:
        cv2.imshow("Comparison", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return combined


def plot_training_curves(results_csv_path: str,
                         output_path: Optional[str] = None,
                         show: bool = True):
    """
    Plot training curves from YOLO results.csv file.
    
    Args:
        results_csv_path: Path to results.csv from training
        output_path: Optional path to save plot
        show: Whether to display the plot
    """
    import pandas as pd
    
    df = pd.read_csv(results_csv_path)
    df.columns = df.columns.str.strip()  # Clean column names
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Box Loss
    if 'train/box_loss' in df.columns:
        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train')
        axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val')
        axes[0, 0].set_title('Box Loss')
        axes[0, 0].legend()
        axes[0, 0].set_xlabel('Epoch')
    
    # Classification Loss
    if 'train/cls_loss' in df.columns:
        axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train')
        axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val')
        axes[0, 1].set_title('Classification Loss')
        axes[0, 1].legend()
        axes[0, 1].set_xlabel('Epoch')
    
    # DFL Loss
    if 'train/dfl_loss' in df.columns:
        axes[0, 2].plot(df['epoch'], df['train/dfl_loss'], label='Train')
        axes[0, 2].plot(df['epoch'], df['val/dfl_loss'], label='Val')
        axes[0, 2].set_title('DFL Loss')
        axes[0, 2].legend()
        axes[0, 2].set_xlabel('Epoch')
    
    # mAP
    if 'metrics/mAP50(B)' in df.columns:
        axes[1, 0].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
        if 'metrics/mAP50-95(B)' in df.columns:
            axes[1, 0].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
        axes[1, 0].set_title('mAP')
        axes[1, 0].legend()
        axes[1, 0].set_xlabel('Epoch')
    
    # Precision
    if 'metrics/precision(B)' in df.columns:
        axes[1, 1].plot(df['epoch'], df['metrics/precision(B)'])
        axes[1, 1].set_title('Precision')
        axes[1, 1].set_xlabel('Epoch')
    
    # Recall
    if 'metrics/recall(B)' in df.columns:
        axes[1, 2].plot(df['epoch'], df['metrics/recall(B)'])
        axes[1, 2].set_title('Recall')
        axes[1, 2].set_xlabel('Epoch')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_class_distribution(label_dir: Path,
                            output_path: Optional[str] = None,
                            show: bool = True):
    """
    Plot class distribution from label files.
    
    Args:
        label_dir: Directory containing YOLO label files
        output_path: Optional path to save plot
        show: Whether to display the plot
    """
    from collections import Counter
    
    class_counts = Counter()
    
    for label_file in label_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
    
    # Prepare data
    classes = list(range(NUM_CLASSES))
    counts = [class_counts.get(c, 0) for c in classes]
    names = [CLASS_NAMES.get(c, f'class_{c}') for c in classes]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, counts, color=[
        f'C{i}' for i in range(NUM_CLASSES)
    ])
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution')
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
    
    if show:
        plt.show()
    else:
        plt.close()


# File handling
def load_yolo_labels(label_path: Path) -> List[Dict]:
    """
    Load YOLO format labels from file.
    
    Args:
        label_path: Path to label file
        
    Returns:
        List of detection dictionaries
    """
    detections = []
    
    if not label_path.exists():
        return detections
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                detections.append({
                    'class_id': int(parts[0]),
                    'x_center': float(parts[1]),
                    'y_center': float(parts[2]),
                    'width': float(parts[3]),
                    'height': float(parts[4]),
                    'confidence': float(parts[5]) if len(parts) > 5 else 1.0,
                    'class_name': CLASS_NAMES.get(int(parts[0]), 'unknown')
                })
    
    return detections


def save_yolo_labels(detections: List[Dict], 
                     label_path: Path,
                     include_confidence: bool = False):
    """
    Save detections to YOLO format file.
    
    Args:
        detections: List of detection dictionaries
        label_path: Path to output file
        include_confidence: Whether to include confidence scores
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(label_path, 'w') as f:
        for det in detections:
            if include_confidence:
                line = f"{det['class_id']} {det['x_center']:.6f} {det['y_center']:.6f} " \
                       f"{det['width']:.6f} {det['height']:.6f} {det.get('confidence', 1.0):.6f}\n"
            else:
                line = f"{det['class_id']} {det['x_center']:.6f} {det['y_center']:.6f} " \
                       f"{det['width']:.6f} {det['height']:.6f}\n"
            f.write(line)


def get_image_files(directory: Path, 
                    extensions: List[str] = None) -> List[Path]:
    """
    Get all image files in a directory.
    
    Args:
        directory: Directory to search
        extensions: List of extensions to include
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    files = []
    for ext in extensions:
        files.extend(directory.glob(f"*{ext}"))
        files.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(files)


# Metrics
def calculate_iou(box1: Dict, box2: Dict) -> float:
    """
    Calculate IoU between two YOLO format boxes.
    
    Args:
        box1, box2: Detection dictionaries with x_center, y_center, width, height
        
    Returns:
        IoU value (0-1)
    """
    # Convert center format to corner format
    x1_1 = box1['x_center'] - box1['width'] / 2
    y1_1 = box1['y_center'] - box1['height'] / 2
    x2_1 = box1['x_center'] + box1['width'] / 2
    y2_1 = box1['y_center'] + box1['height'] / 2
    
    x1_2 = box2['x_center'] - box2['width'] / 2
    y1_2 = box2['y_center'] - box2['height'] / 2
    x2_2 = box2['x_center'] + box2['width'] / 2
    y2_2 = box2['y_center'] + box2['height'] / 2
    
    # Calculate intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    
    # Calculate union
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_precision_recall(predictions: List[Dict],
                               ground_truths: List[Dict],
                               iou_threshold: float = 0.5) -> Dict:
    """
    Calculate precision and recall for a set of predictions.
    
    Args:
        predictions: List of predicted detections
        ground_truths: List of ground truth detections
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dictionary with precision, recall, f1, tp, fp, fn counts
    """
    # Sort predictions by confidence
    predictions = sorted(predictions, 
                         key=lambda x: x.get('confidence', 1.0), 
                         reverse=True)
    
    matched_gts = set()
    tp = 0
    fp = 0
    
    for pred in predictions:
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt in enumerate(ground_truths):
            if i in matched_gts:
                continue
            if pred['class_id'] != gt['class_id']:
                continue
            
            iou = calculate_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gts.add(best_gt_idx)
        else:
            fp += 1
    
    fn = len(ground_truths) - len(matched_gts)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


# Sampling
def sample_random_images(images_dir: Path, 
                         n: int = 10,
                         seed: int = None) -> List[Path]:
    """
    Sample random images from a directory.
    
    Args:
        images_dir: Directory containing images
        n: Number of images to sample
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled image paths
    """
    if seed is not None:
        random.seed(seed)
    
    image_files = get_image_files(images_dir)
    n = min(n, len(image_files))
    
    return random.sample(image_files, n)


def create_sample_predictions_visualization(images_dir: Path,
                                            labels_dir: Path,
                                            output_dir: Path,
                                            n_samples: int = 10):
    """
    Create visualization of random sample predictions.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing predictions
        output_dir: Directory for output visualizations
        n_samples: Number of samples to visualize
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples = sample_random_images(images_dir, n_samples)
    
    for img_path in samples:
        label_path = labels_dir / f"{img_path.stem}.txt"
        detections = load_yolo_labels(label_path)
        
        # Load and annotate image
        image = cv2.imread(str(img_path))
        annotated = draw_bboxes(image, detections)
        
        # Save
        output_path = output_dir / f"{img_path.stem}_annotated{img_path.suffix}"
        cv2.imwrite(str(output_path), annotated)
    
    print(f"Saved {len(samples)} visualizations to {output_dir}")


if __name__ == "__main__":
    # Example usage
    print("Utility functions for Crackathon pipeline")
    print(f"\nClass names: {CLASS_NAMES}")
    print(f"Number of classes: {NUM_CLASSES}")
