# Crackathon Road Damage Detection Pipeline

End-to-end pipeline for the Crackathon challenge using YOLOv8 on the RDD2022 dataset.

## Overview

Object detection pipeline for road damage detection using YOLOv8, trained on the Road Damage Detection 2022 dataset.

### Class Mapping
| Class ID | Name | Description |
|----------|------|-------------|
| 0 | Longitudinal crack | Linear cracks parallel to road direction |
| 1 | Transverse crack | Linear cracks perpendicular to road direction |
| 2 | Alligator crack | Interconnected crack patterns |
| 3 | Other corruption | Miscellaneous road surface damage |
| 4 | Pothole | Circular/irregular depressions |

## Installation

### Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ GPU memory for training

### Setup

```bash
# Clone or navigate to project directory
cd crackathon

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Full Pipeline
```bash
python main.py --full
```

### Competition Mode
```bash
python main.py --full --competition
```

### Individual Stages
```bash
python main.py --prepare

python main.py --train

python main.py --evaluate

python main.py --submit
```

## Dataset

### Source
The dataset is downloaded from Google Drive:
https://drive.google.com/drive/folders/1JpBQ5haJCvPhD-0jUdir3GiGNbBnah93

## Model Architecture

YOLOv8 was chosen for its anchor-free design, which handles varying aspect ratios of road damage well. The CSPDarknet backbone with Path Aggregation Network provides multi-scale feature fusion for detecting both small cracks and large potholes.

### Model Variants
| Variant | Parameters | mAP (COCO) | Use Case |
|---------|------------|------------|----------|
| YOLOv8n | 3.2M | 37.3% | Real-time, edge devices |
| YOLOv8s | 11.2M | 44.9% | Balanced speed/accuracy |
| YOLOv8m | 25.9M | 50.2% | Default |
| YOLOv8l | 43.7M | 52.9% | Higher accuracy |
| YOLOv8x | 68.2M | 53.9% | Maximum accuracy |

### Training Command
```bash
python main.py --train

python main.py --train --epochs 150 --batch 8

python main.py --train --resume
```

## Evaluation

### Metrics
- **mAP@0.5**: Primary metric - Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: COCO-style mAP averaged over IoU thresholds
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)

### NMS (Non-Maximum Suppression)
IoU threshold 0.45 (default) for removing redundant overlapping detections.

### Evaluation Command
```bash
python main.py --evaluate

python evaluate.py --weights path/to/best.pt --optimize-conf
```

## Inference & Submission

### Prediction Format
Each test image generates a `.txt` file:
```
<class_id> <x_center> <y_center> <width> <height> <confidence>
```

### Test-Time Augmentation (TTA)
TTA aggregates predictions from multiple augmented versions (multi-scale + horizontal flip). Slower inference but typically improves mAP by 1-3%.

### Submission Generation
```bash
python main.py --submit

# slower but better mAP
python main.py --submit --tta
```
### References
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
