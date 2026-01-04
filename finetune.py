"""
Short finetuning run to refine bounding box precision and class confidence.
Starts from existing trained checkpoint with conservative hyperparameters.
"""

from pathlib import Path
from ultralytics import YOLO
import torch

WEIGHTS_PATH = Path(r"D:\Hackathons\crackathon\outputs\runs\crackathon_baseline4\weights\yolov8m_rdd2022_best.pt")
DATA_YAML = Path(r"D:\Hackathons\crackathon\data\rdd2022\data.yaml")
OUTPUT_DIR = Path(r"D:\Hackathons\crackathon\outputs\runs")

EPOCHS = 15
LR0 = 0.0003
LRF = 0.01
IMGSZ = 640
BATCH_SIZE = 8


def main():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {WEIGHTS_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("FINETUNING RUN")
    print("=" * 70)
    print(f"Checkpoint: {WEIGHTS_PATH}")
    print(f"Device: {device}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LR0}")
    print(f"Image size: {IMGSZ}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Mosaic: DISABLED (close_mosaic=0)")
    print("=" * 70)

    model = YOLO(str(WEIGHTS_PATH))

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        device=device,
        project=str(OUTPUT_DIR),
        name="finetune_v1",
        exist_ok=True,
        lr0=LR0,
        lrf=LRF,
        mosaic=0.0,
        close_mosaic=0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        patience=10,
        save=True,
        save_period=5,
        val=True,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("FINETUNING COMPLETE")
    print("=" * 70)

    best_weights = OUTPUT_DIR / "finetune_v1" / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\nEvaluating finetuned model...")
        finetuned_model = YOLO(str(best_weights))
        metrics = finetuned_model.val(data=str(DATA_YAML), imgsz=IMGSZ, conf=0.25, iou=0.4)

        print("\n" + "=" * 70)
        print("FINETUNED MODEL RESULTS")
        print("=" * 70)
        print(f"mAP@50:      {metrics.box.map50:.4f}")
        print(f"mAP@50-95:   {metrics.box.map:.4f}")
        print(f"Precision:   {metrics.box.mp:.4f}")
        print(f"Recall:      {metrics.box.mr:.4f}")
        print(f"\nWeights saved to: {best_weights}")

        print("\n" + "=" * 70)
        print("BASELINE COMPARISON")
        print("=" * 70)
        print("Original (80 epochs): mAP@50=50.9%, mAP@50-95=26.4%")
        print(f"Finetuned (+15 epochs): mAP@50={metrics.box.map50*100:.2f}%, mAP@50-95={metrics.box.map*100:.2f}%")


if __name__ == "__main__":
    main()
