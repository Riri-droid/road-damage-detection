"""
Threshold sweep for optimal confidence and NMS IoU selection.
Tests combinations without retraining the model.
"""
import sys
from pathlib import Path
from itertools import product
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
from config import DATASET_ROOT, RUNS_DIR

WEIGHTS_PATH = RUNS_DIR / "crackathon_baseline4" / "weights" / "yolov8m_rdd2022_best.pt"
DATA_YAML = DATASET_ROOT / "data.yaml"

CONF_THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35]
IOU_THRESHOLDS = [0.40, 0.45, 0.50]

def run_sweep():
    print("=" * 70)
    print("THRESHOLD SWEEP")
    print("=" * 70)
    print(f"Model: {WEIGHTS_PATH}")
    print(f"Confidence thresholds: {CONF_THRESHOLDS}")
    print(f"NMS IoU thresholds: {IOU_THRESHOLDS}")
    print(f"Total combinations: {len(CONF_THRESHOLDS) * len(IOU_THRESHOLDS)}")
    print("=" * 70)
    
    model = YOLO(str(WEIGHTS_PATH))
    
    results_list = []
    
    for conf, iou in product(CONF_THRESHOLDS, IOU_THRESHOLDS):
        print(f"\nTesting conf={conf:.2f}, iou={iou:.2f}...")
        
        results = model.val(
            data=str(DATA_YAML),
            imgsz=640,
            batch=32,
            conf=conf,
            iou=iou,
            device="0",
            plots=False,
            verbose=False
        )
        
        metrics = {
            'conf': conf,
            'iou': iou,
            'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
            'mAP50_95': results.results_dict.get('metrics/mAP50-95(B)', 0),
            'precision': results.results_dict.get('metrics/precision(B)', 0),
            'recall': results.results_dict.get('metrics/recall(B)', 0),
        }
        
        results_list.append(metrics)
        print(f"  mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50_95']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")
    
    df = pd.DataFrame(results_list)
    
    print("\n" + "=" * 70)
    print("FULL RESULTS")
    print("=" * 70)
    print(df.to_string(index=False))
    
    df['pr_balance'] = 2 * df['precision'] * df['recall'] / (df['precision'] + df['recall'] + 1e-6)
    
    best_idx = df['mAP50_95'].idxmax()
    best = df.loc[best_idx]
    
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION (by mAP@50-95)")
    print("=" * 70)
    print(f"Confidence threshold: {best['conf']:.2f}")
    print(f"NMS IoU threshold:    {best['iou']:.2f}")
    print(f"mAP@50:               {best['mAP50']:.4f}")
    print(f"mAP@50-95:            {best['mAP50_95']:.4f}")
    print(f"Precision:            {best['precision']:.4f}")
    print(f"Recall:               {best['recall']:.4f}")
    print(f"F1 Score:             {best['pr_balance']:.4f}")
    
    df.to_csv(RUNS_DIR / "threshold_sweep_results.csv", index=False)
    print(f"\nResults saved to: {RUNS_DIR / 'threshold_sweep_results.csv'}")
    
    return df, best

if __name__ == "__main__":
    run_sweep()
