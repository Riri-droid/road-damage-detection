# Crackathon Road Damage Detection Pipeline

End-to-end object detection pipeline for road damage detection using YOLOv8, built for the Crackathon challenge on the RDD2022 dataset.

## Overview

This project implements a computer vision pipeline to detect and classify road surface damage from images. The model is trained on the Road Damage Detection 2022 (RDD2022) dataset and supports training, evaluation, and generation of competition-ready predictions.

### Class Mapping

| Class ID | Name                | Description                                  |
|----------|---------------------|----------------------------------------------|
| 0        | Longitudinal crack  | Linear cracks parallel to road direction     |
| 1        | Transverse crack    | Linear cracks perpendicular to road direction|
| 2        | Alligator crack     | Interconnected crack patterns                |
| 3        | Other corruption    | Miscellaneous road surface damage            |
| 4        | Pothole             | Circular or irregular depressions            |

## Installation

### Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ GPU memory for training

### Setup

```bash
cd crackathon

python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux / Mac

pip install -r requirements.txt
