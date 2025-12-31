"""
Quick smoke test - 5 epochs to verify pipeline works end-to-end.
"""
from ultralytics import YOLO
import torch

print('='*60)
print('SMOKE TEST: 5 epochs on CPU with YOLOv8n')
print('='*60)

# Use nano model for speed
model = YOLO('yolov8n.pt')

results = model.train(
    data='D:/Hackathons/crackathon/data/rdd2022/data.yaml',
    epochs=5,
    imgsz=640,
    batch=8,
    device='cpu',
    workers=0,
    project='D:/Hackathons/crackathon/outputs/runs',
    name='smoke_test',
    verbose=True,
    patience=50
)

print('\n' + '='*60)
print('SMOKE TEST COMPLETE')
print('='*60)

# Print results
if hasattr(results, 'results_dict'):
    rd = results.results_dict
    print(f"mAP50:    {rd.get('metrics/mAP50(B)', 'N/A')}")
    print(f"mAP50-95: {rd.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"Precision: {rd.get('metrics/precision(B)', 'N/A')}")
    print(f"Recall:    {rd.get('metrics/recall(B)', 'N/A')}")
