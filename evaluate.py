import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import json

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    Config, get_config,
    DATASET_ROOT, OUTPUT_ROOT, RUNS_DIR,
    CLASS_NAMES, NUM_CLASSES
)


class Evaluator:

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.data_yaml = DATASET_ROOT / "data.yaml"
        self.predictions = []
        self.ground_truths = []
        self.metrics = {}

    def evaluate_model(self, 
                       weights: str,
                       split: str = 'val',
                       conf_threshold: float = 0.001,
                       iou_threshold: float = 0.45) -> Dict:
        if split != 'val':
            print(f"WARNING: split='{split}' is not allowed. Using 'val' split.")
            print("         Test set has no labels and cannot be evaluated.")
            split = 'val'

        from ultralytics import YOLO

        print("=" * 60)
        print(f"EVALUATING MODEL")
        print("=" * 60)
        print(f"Weights: {weights}")
        print(f"Split: {split}")
        print(f"Conf threshold: {conf_threshold}")
        print(f"IoU threshold: {iou_threshold}")

        model = YOLO(weights)

        results = model.val(
            data=str(self.data_yaml),
            split=split,
            imgsz=self.config.model.input_size,
            batch=self.config.evaluation.val_batch_size,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.config.training.device,
            plots=self.config.evaluation.plot_pr_curve,
            save_json=True,
            verbose=True
        )

        self.metrics = self._extract_metrics(results)
        self._print_evaluation_results()
        return self.metrics

    def _extract_metrics(self, results) -> Dict:
        metrics = {
            'overall': {
                'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                'mAP50_95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
                'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
                'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
            },
            'per_class': {},
            'confusion_matrix': None
        }

        if hasattr(results, 'maps'):
            for i, ap in enumerate(results.maps):
                if i < NUM_CLASSES:
                    class_name = CLASS_NAMES.get(i, f'class_{i}')
                    metrics['per_class'][class_name] = {
                        'AP50': float(ap),
                        'class_id': i
                    }

        return metrics

    def _print_evaluation_results(self):
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print("\nOverall Metrics:")
        print("-" * 40)
        print(f"  mAP@0.5:        {self.metrics['overall']['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95:   {self.metrics['overall']['mAP50_95']:.4f}")
        print(f"  Precision:      {self.metrics['overall']['precision']:.4f}")
        print(f"  Recall:         {self.metrics['overall']['recall']:.4f}")

        if self.metrics['per_class']:
            print("\nPer-Class AP@0.5:")
            print("-" * 40)
            for class_name, class_metrics in self.metrics['per_class'].items():
                print(f"  {class_name}: {class_metrics['AP50']:.4f}")

    def find_optimal_confidence(self,
                                weights: str,
                                conf_range: Tuple[float, float] = (0.1, 0.9),
                                num_steps: int = 9) -> Tuple[float, Dict]:
        print("\n" + "=" * 60)
        print("CONFIDENCE THRESHOLD OPTIMIZATION")
        print("=" * 60)

        from ultralytics import YOLO
        model = YOLO(weights)

        conf_values = np.linspace(conf_range[0], conf_range[1], num_steps)
        results_list = []

        for conf in tqdm(conf_values, desc="Testing thresholds"):
            results = model.val(
                data=str(self.data_yaml),
                imgsz=self.config.model.input_size,
                batch=self.config.evaluation.val_batch_size,
                conf=conf,
                iou=self.config.evaluation.nms_iou,
                device=self.config.training.device,
                plots=False,
                verbose=False
            )

            results_list.append({
                'conf': conf,
                'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
                'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
            })

        best_result = max(results_list, key=lambda x: x['mAP50'])

        print("\nResults by Confidence Threshold:")
        print("-" * 60)
        print(f"{'Conf':>8} {'mAP50':>10} {'Precision':>10} {'Recall':>10}")
        print("-" * 60)

        for r in results_list:
            marker = " *" if r['conf'] == best_result['conf'] else ""
            print(f"{r['conf']:>8.2f} {r['mAP50']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f}{marker}")

        print(f"\nOptimal confidence threshold: {best_result['conf']:.2f}")
        print(f"  mAP@0.5: {best_result['mAP50']:.4f}")

        return best_result['conf'], results_list

    def find_optimal_nms_iou(self,
                             weights: str,
                             iou_range: Tuple[float, float] = (0.3, 0.7),
                             num_steps: int = 5) -> Tuple[float, Dict]:
        print("\n" + "=" * 60)
        print("NMS IoU THRESHOLD OPTIMIZATION")
        print("=" * 60)

        from ultralytics import YOLO
        model = YOLO(weights)

        iou_values = np.linspace(iou_range[0], iou_range[1], num_steps)
        results_list = []

        for iou in tqdm(iou_values, desc="Testing NMS thresholds"):
            results = model.val(
                data=str(self.data_yaml),
                imgsz=self.config.model.input_size,
                batch=self.config.evaluation.val_batch_size,
                conf=self.config.model.conf_threshold,
                iou=iou,
                device=self.config.training.device,
                plots=False,
                verbose=False
            )

            results_list.append({
                'iou': iou,
                'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
                'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
                'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
            })

        best_result = max(results_list, key=lambda x: x['mAP50'])

        print("\nResults by NMS IoU Threshold:")
        print("-" * 60)
        print(f"{'IoU':>8} {'mAP50':>10} {'Precision':>10} {'Recall':>10}")
        print("-" * 60)

        for r in results_list:
            marker = " *" if r['iou'] == best_result['iou'] else ""
            print(f"{r['iou']:>8.2f} {r['mAP50']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f}{marker}")

        print(f"\nOptimal NMS IoU threshold: {best_result['iou']:.2f}")
        print(f"  mAP@0.5: {best_result['mAP50']:.4f}")

        return best_result['iou'], results_list


class ErrorAnalyzer:

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()

    def analyze_predictions(self,
                           predictions_dir: Path,
                           ground_truth_dir: Path,
                           iou_threshold: float = 0.5) -> Dict:
        print("=" * 60)
        print("ERROR ANALYSIS")
        print("=" * 60)

        analysis = {
            'total_predictions': 0,
            'total_ground_truths': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'classification_errors': 0,
            'per_class_errors': defaultdict(lambda: {
                'tp': 0, 'fp': 0, 'fn': 0, 'cls_err': 0
            })
        }

        pred_files = list(predictions_dir.glob("*.txt"))

        for pred_file in tqdm(pred_files, desc="Analyzing predictions"):
            gt_file = ground_truth_dir / pred_file.name
            preds = self._load_detections(pred_file)
            analysis['total_predictions'] += len(preds)

            if gt_file.exists():
                gts = self._load_detections(gt_file, has_confidence=False)
                analysis['total_ground_truths'] += len(gts)
            else:
                gts = []

            self._match_detections(preds, gts, iou_threshold, analysis)

        tp = analysis['true_positives']
        fp = analysis['false_positives']
        fn = analysis['false_negatives']

        analysis['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        analysis['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        analysis['f1'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        self._print_error_analysis(analysis)
        return analysis

    def _load_detections(self, 
                         file_path: Path, 
                         has_confidence: bool = True) -> List[Dict]:
        detections = []
        if not file_path.exists():
            return detections

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if has_confidence and len(parts) == 6:
                    detections.append({
                        'class_id': int(parts[0]),
                        'x_center': float(parts[1]),
                        'y_center': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4]),
                        'confidence': float(parts[5])
                    })
                elif not has_confidence and len(parts) == 5:
                    detections.append({
                        'class_id': int(parts[0]),
                        'x_center': float(parts[1]),
                        'y_center': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4]),
                        'confidence': 1.0
                    })

        return detections

    def _match_detections(self,
                          preds: List[Dict],
                          gts: List[Dict],
                          iou_threshold: float,
                          analysis: Dict):
        matched_gts = set()
        preds = sorted(preds, key=lambda x: x['confidence'], reverse=True)

        for pred in preds:
            best_iou = 0
            best_gt_idx = -1

            for i, gt in enumerate(gts):
                if i in matched_gts:
                    continue
                iou = self._calculate_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                matched_gts.add(best_gt_idx)
                gt = gts[best_gt_idx]

                if pred['class_id'] == gt['class_id']:
                    analysis['true_positives'] += 1
                    analysis['per_class_errors'][pred['class_id']]['tp'] += 1
                else:
                    analysis['classification_errors'] += 1
                    analysis['per_class_errors'][pred['class_id']]['cls_err'] += 1
            else:
                analysis['false_positives'] += 1
                analysis['per_class_errors'][pred['class_id']]['fp'] += 1

        for i, gt in enumerate(gts):
            if i not in matched_gts:
                analysis['false_negatives'] += 1
                analysis['per_class_errors'][gt['class_id']]['fn'] += 1

    def _calculate_iou(self, box1: Dict, box2: Dict) -> float:
        x1_1 = box1['x_center'] - box1['width'] / 2
        y1_1 = box1['y_center'] - box1['height'] / 2
        x2_1 = box1['x_center'] + box1['width'] / 2
        y2_1 = box1['y_center'] + box1['height'] / 2

        x1_2 = box2['x_center'] - box2['width'] / 2
        y1_2 = box2['y_center'] - box2['height'] / 2
        x2_2 = box2['x_center'] + box2['width'] / 2
        y2_2 = box2['y_center'] + box2['height'] / 2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = box1['width'] * box1['height']
        area2 = box2['width'] * box2['height']
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _print_error_analysis(self, analysis: Dict):
        print("\n" + "-" * 60)
        print("ERROR ANALYSIS SUMMARY")
        print("-" * 60)
        print(f"\nOverall Statistics:")
        print(f"  Total Predictions:    {analysis['total_predictions']}")
        print(f"  Total Ground Truths:  {analysis['total_ground_truths']}")
        print(f"  True Positives:       {analysis['true_positives']}")
        print(f"  False Positives:      {analysis['false_positives']}")
        print(f"  False Negatives:      {analysis['false_negatives']}")
        print(f"  Classification Errors: {analysis['classification_errors']}")
        print(f"\nMetrics:")
        print(f"  Precision: {analysis['precision']:.4f}")
        print(f"  Recall:    {analysis['recall']:.4f}")
        print(f"  F1 Score:  {analysis['f1']:.4f}")
        print(f"\nPer-Class Errors:")
        print(f"{'Class':<25} {'TP':>6} {'FP':>6} {'FN':>6} {'Cls Err':>8}")
        print("-" * 55)

        for class_id in sorted(analysis['per_class_errors'].keys()):
            errors = analysis['per_class_errors'][class_id]
            class_name = CLASS_NAMES.get(class_id, f'class_{class_id}')
            print(f"{class_name:<25} {errors['tp']:>6} {errors['fp']:>6} {errors['fn']:>6} {errors['cls_err']:>8}")


def find_best_weights() -> Optional[Path]:
    runs = sorted(RUNS_DIR.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate road damage detection model")
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to model weights (uses latest if not specified)')
    parser.add_argument('--optimize-conf', action='store_true',
                        help='Find optimal confidence threshold')
    parser.add_argument('--optimize-nms', action='store_true',
                        help='Find optimal NMS IoU threshold')

    args = parser.parse_args()

    weights = args.weights
    if weights is None:
        weights = find_best_weights()
        if weights is None:
            print("ERROR: No trained weights found. Train a model first.")
            return
        weights = str(weights)

    evaluator = Evaluator()
    evaluator.evaluate_model(weights)

    if args.optimize_conf:
        evaluator.find_optimal_confidence(weights)

    if args.optimize_nms:
        evaluator.find_optimal_nms_iou(weights)


if __name__ == "__main__":
    main()
