Road Damage Detection

End-to-end computer vision pipeline for detecting road surface damage (cracks and potholes) using YOLOv8 and the RDD2022 dataset.

This project was built for the Crackathon challenge and focuses on a clean, reproducible training and inference workflow rather than leaderboard-only optimization.

Overview

The system detects and classifies five types of road damage from images:

Longitudinal crack

Transverse crack

Alligator crack

Other surface corruption

Pothole

It supports dataset preparation, model training, evaluation, and generation of competition-ready prediction files.

Model

Architecture: YOLOv8m

Framework: Ultralytics YOLOv8

Input size: 640 × 640

Optimizer: AdamW

Mixed Precision (AMP): Enabled

YOLOv8m was chosen as a balance between accuracy and inference speed, performing well on both thin cracks and larger potholes.

Dataset

Dataset: Road Damage Detection 2022 (RDD2022)

Format: YOLO (class x_center y_center width height)

Split:

Train / Validation (with labels)

Test (images only, used for submission)

Dataset handling includes verification of image–label consistency and automatic configuration generation.

Project Structure
crackathon/
├── main.py              # Pipeline entry point
├── config.py            # Training configuration
├── dataset_handler.py   # Dataset preparation and validation
├── train.py             # Training logic
├── evaluate.py          # Evaluation metrics
├── inference.py         # Test inference and submission generation
├── data/                # Dataset directory
└── outputs/             # Model checkpoints and predictions

Quick Start

Install dependencies:

pip install -r requirements.txt


Run the full pipeline:

python main.py --full


Competition-optimized run:

python main.py --full --competition

Output

Trained model checkpoint (best.pt)

Per-image prediction .txt files for the test set

submission.zip ready for upload

Results (Validation)

mAP@50: ~45–50%

mAP@50–95: ~21–26%

Inference speed: ~77 FPS on RTX 4070

Performance varies by class, with potholes being the most challenging due to size and frequency.

Notes

Designed for reproducibility and correctness

Uses only the provided dataset

Suitable for real-time or near–real-time inference

References

Ultralytics YOLOv8


