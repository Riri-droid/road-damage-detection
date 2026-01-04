"""
Resolution sweep for inference evaluation.
Tests different input resolutions without retraining.
"""

from pathlib import Path
from ultralytics import YOLO
import pandas as pd

WEIGHTS_PATH = Path(r"D:\Hackathons\crackathon\outputs\runs\crackathon_baseline4\weights\yolov8m_rdd2022_best.pt")
DATA_YAML = Path(r"D:\Hackathons\crackathon\data\rdd2022\data.yaml")
OUTPUT_DIR = Path(r"D:\Hackathons\crackathon\outputs\runs")

RESOLUTIONS = [640, 768, 832]
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.40

def main():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")

    print("=" * 70)
    print("RESOLUTION SWEEP")
    print("=" * 70)
    print(f"Model: {WEIGHTS_PATH}")
    print(f"Resolutions: {RESOLUTIONS}")
    print(f"Confidence: {CONF_THRESHOLD}, NMS IoU: {IOU_THRESHOLD}")
    print("=" * 70)

    model = YOLO(str(WEIGHTS_PATH))
    results_list = []

    for imgsz in RESOLUTIONS:
        print(f"\nTesting resolution {imgsz}x{imgsz}...")
        
        try:
            metrics = model.val(
                data=str(DATA_YAML),
                imgsz=imgsz,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                batch=8 if imgsz <= 768 else 4,
                verbose=False,
            )

            result = {
                "resolution": imgsz,
                "mAP50": metrics.box.map50,
                "mAP50_95": metrics.box.map,
                "precision": metrics.box.mp,
                "recall": metrics.box.mr,
            }
            results_list.append(result)

            print(f"  mAP50={result['mAP50']:.4f}  mAP50-95={result['mAP50_95']:.4f}  P={result['precision']:.4f}  R={result['recall']:.4f}")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  SKIPPED - GPU out of memory at {imgsz}x{imgsz}")
                continue
            raise

    df = pd.DataFrame(results_list)
    
    print("\n" + "=" * 70)
    print("FULL RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))

    baseline = df[df["resolution"] == 640].iloc[0]
    
    print("\n" + "=" * 70)
    print("COMPARISON VS BASELINE (640)")
    print("=" * 70)
    
    for _, row in df.iterrows():
        res = int(row["resolution"])
        if res == 640:
            continue
        
        map50_delta = (row["mAP50"] - baseline["mAP50"]) * 100
        map50_95_delta = (row["mAP50_95"] - baseline["mAP50_95"]) * 100
        p_delta = (row["precision"] - baseline["precision"]) * 100
        r_delta = (row["recall"] - baseline["recall"]) * 100
        
        print(f"\n{res} vs 640:")
        print(f"  mAP@50:    {row['mAP50']*100:.2f}% ({map50_delta:+.2f}%)")
        print(f"  mAP@50-95: {row['mAP50_95']*100:.2f}% ({map50_95_delta:+.2f}%)")
        print(f"  Precision: {row['precision']*100:.2f}% ({p_delta:+.2f}%)")
        print(f"  Recall:    {row['recall']*100:.2f}% ({r_delta:+.2f}%)")

    best_idx = df["mAP50_95"].idxmax()
    best = df.loc[best_idx]

    print("\n" + "=" * 70)
    print("BEST RESOLUTION (by mAP@50-95)")
    print("=" * 70)
    print(f"Resolution:  {int(best['resolution'])}x{int(best['resolution'])}")
    print(f"mAP@50:      {best['mAP50']:.4f}")
    print(f"mAP@50-95:   {best['mAP50_95']:.4f}")
    print(f"Precision:   {best['precision']:.4f}")
    print(f"Recall:      {best['recall']:.4f}")

    csv_path = OUTPUT_DIR / "resolution_sweep_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
