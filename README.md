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

## Project Structure

```
crackathon/
├── main.py                 # Main entry point - runs complete pipeline
├── config.py               # All configuration parameters
├── dataset_handler.py      # Dataset download, organization, validation
├── augmentation.py         # Data augmentation strategies
├── train.py                # YOLOv8 training pipeline
├── evaluate.py             # Model evaluation and metrics
├── inference.py            # Inference and submission generation
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── data/                   # Dataset storage (created automatically)
│   └── rdd2022/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── val/
│       │   ├── images/
│       │   └── labels/
│       └── test/
│           └── images/
│
└── outputs/                # Training outputs (created automatically)
    ├── runs/               # Training runs and checkpoints
    └── predictions/        # Inference results
```

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
# Download, train, evaluate, and generate submission
python main.py --full
```

### Competition Mode
```bash
# Uses optimized settings for higher mAP
python main.py --full --competition
```

### Individual Stages
```bash
# 1. Prepare dataset only
python main.py --prepare

# 2. Train model only
python main.py --train

# 3. Evaluate model only
python main.py --evaluate

# 4. Generate submission only
python main.py --submit
```

## Dataset

### Source
The dataset is downloaded from Google Drive:
https://drive.google.com/drive/folders/1JpBQ5haJCvPhD-0jUdir3GiGNbBnah93

### Format
- Images: JPG format
- Labels: YOLO TXT format (`<class_id> <x_center> <y_center> <width> <height>`)
- All coordinates are normalized (0-1 range)

### Preparation
The `dataset_handler.py` module handles:
1. Downloading from Google Drive using `gdown`
2. Organizing into YOLO directory structure
3. Verifying image-label consistency
4. Creating train/validation splits if needed
5. Generating `data.yaml` configuration file

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

## Training

### Configuration
All training parameters are in `config.py`:

```python
# Key hyperparameters
epochs = 100              # Training duration
batch_size = 16           # Batch size
lr0 = 0.001               # Initial learning rate
optimizer = "AdamW"       # Optimizer
patience = 20             # Early stopping patience
```

### Data Augmentation
Augmentations for road imagery: rotation, scale, flip, perspective, brightness, contrast, HSV shifts, motion blur, and simulated weather effects.

### Loss Functions
CIoU loss for bounding box regression, BCE for classification, and Distribution Focal Loss for improved localization.

### Training Command
```bash
# Default training
python main.py --train

# Custom epochs and batch size
python main.py --train --epochs 150 --batch 8

# Resume from checkpoint
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
# Evaluate with default settings
python main.py --evaluate

# Find optimal confidence threshold
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
# Generate submission.zip
python main.py --submit

# With TTA (slower but better mAP)
python main.py --submit --tta
```

### Output Structure
```
outputs/
└── predictions/
    ├── image001.txt
    ├── image002.txt
    └── ...
submission.zip  # Contains predictions/ folder
```

## Technical Notes

### Techniques Used

1. **Transfer Learning** - COCO pretrained weights
2. **Data Augmentation** - Mosaic, MixUp, motion blur, shadows
3. **Learning Rate Schedule** - Warmup + cosine annealing with AdamW
4. **Test-Time Augmentation** - Multi-scale inference and horizontal flip
5. **Confidence Threshold Tuning** - Grid search for precision/recall balance

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 16 |
| Learning rate | 0.001 |
| Epochs | 100 |
| Input size | 640 |
| Patience | 20 |
| Box loss weight | 7.5 |

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size: `--batch 8`
   - Use smaller model variant in `config.py`

2. **Dataset download fails**
   - Manually download from Google Drive
   - Place in `data/rdd2022/` directory

3. **Training not converging**
   - Check dataset integrity: `python dataset_handler.py`
   - Lower learning rate in `config.py`

4. **Low mAP scores**
   - Increase epochs
   - Enable TTA for inference
   - Try competition config: `--competition`

## License

For educational and competition use.

## References

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Road Damage Detection Challenge](https://rdd2022.sekilab.global/)
