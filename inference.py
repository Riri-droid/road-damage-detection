import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import zipfile
import shutil

import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    Config, get_config, get_competition_config,
    DATASET_ROOT, OUTPUT_ROOT, PREDICTIONS_DIR, RUNS_DIR,
    CLASS_NAMES, NUM_CLASSES
)


class Inferencer:

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.model = None
        self.test_images_dir = DATASET_ROOT / "test" / "images"
        self.predictions_dir = PREDICTIONS_DIR
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self, weights: str) -> 'YOLO':
        from ultralytics import YOLO
        print("=" * 60)
        print("LOADING MODEL FOR INFERENCE")
        print("=" * 60)
        print(f"Weights: {weights}")
        self.model = YOLO(weights)
        print("Model loaded successfully!")
        return self.model

    def predict_single(self, 
                       image_path: str,
                       conf_threshold: Optional[float] = None,
                       iou_threshold: Optional[float] = None,
                       save_visualization: bool = False) -> List[Dict]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        conf = conf_threshold or self.config.inference.conf_threshold
        iou = iou_threshold or self.config.inference.iou_threshold
        results = self.model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            imgsz=self.config.model.input_size,
            device=self.config.training.device,
            verbose=False,
            save=save_visualization
        )
        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    xywhn = boxes.xywhn[i].cpu().numpy()
                    detections.append({
                        'class_id': int(boxes.cls[i].cpu().numpy()),
                        'x_center': float(xywhn[0]),
                        'y_center': float(xywhn[1]),
                        'width': float(xywhn[2]),
                        'height': float(xywhn[3]),
                        'confidence': float(boxes.conf[i].cpu().numpy()),
                        'class_name': CLASS_NAMES.get(int(boxes.cls[i].cpu().numpy()), 'unknown')
                    })
        return detections

    def predict_with_tta(self,
                         image_path: str,
                         conf_threshold: Optional[float] = None,
                         iou_threshold: Optional[float] = None) -> List[Dict]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        conf = conf_threshold or self.config.inference.conf_threshold
        iou = iou_threshold or self.config.inference.iou_threshold
        results = self.model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            imgsz=self.config.model.input_size,
            device=self.config.training.device,
            augment=True,
            verbose=False
        )
        detections = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    xywhn = boxes.xywhn[i].cpu().numpy()
                    detections.append({
                        'class_id': int(boxes.cls[i].cpu().numpy()),
                        'x_center': float(xywhn[0]),
                        'y_center': float(xywhn[1]),
                        'width': float(xywhn[2]),
                        'height': float(xywhn[3]),
                        'confidence': float(boxes.conf[i].cpu().numpy()),
                        'class_name': CLASS_NAMES.get(int(boxes.cls[i].cpu().numpy()), 'unknown')
                    })
        return detections

    def predict_batch(self,
                      images_dir: Optional[Path] = None,
                      conf_threshold: Optional[float] = None,
                      iou_threshold: Optional[float] = None,
                      use_tta: Optional[bool] = None,
                      output_dir: Optional[Path] = None) -> Dict[str, List[Dict]]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        images_dir = images_dir or self.test_images_dir
        output_dir = output_dir or self.predictions_dir
        conf = conf_threshold or self.config.inference.conf_threshold
        iou = iou_threshold or self.config.inference.iou_threshold
        use_tta = use_tta if use_tta is not None else self.config.inference.use_tta
        output_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 60)
        print("BATCH INFERENCE")
        print("=" * 60)
        print(f"Images directory: {images_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Confidence threshold: {conf}")
        print(f"IoU threshold: {iou}")
        print(f"TTA enabled: {use_tta}")
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
            image_files.extend(images_dir.glob(f"*{ext.upper()}"))
        print(f"\nFound {len(image_files)} images")
        if not image_files:
            print("WARNING: No images found in directory!")
            return {}
        all_predictions = {}
        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                if use_tta:
                    detections = self.predict_with_tta(str(image_path), conf, iou)
                else:
                    detections = self.predict_single(str(image_path), conf, iou)
                all_predictions[image_path.stem] = detections
                self._save_prediction_file(detections, output_dir / f"{image_path.stem}.txt")
            except Exception as e:
                print(f"\nError processing {image_path.name}: {e}")
                all_predictions[image_path.stem] = []
                self._save_prediction_file([], output_dir / f"{image_path.stem}.txt")
        total_detections = sum(len(d) for d in all_predictions.values())
        images_with_detections = sum(1 for d in all_predictions.values() if d)
        print(f"\nInference complete!")
        print(f"  Total images processed: {len(all_predictions)}")
        print(f"  Images with detections: {images_with_detections}")
        print(f"  Total detections: {total_detections}")
        print(f"  Predictions saved to: {output_dir}")
        return all_predictions

    def _save_prediction_file(self, detections: List[Dict], output_path: Path):
        with open(output_path, 'w') as f:
            for det in detections:
                line = f"{det['class_id']} {det['x_center']:.6f} {det['y_center']:.6f} " \
                       f"{det['width']:.6f} {det['height']:.6f} {det['confidence']:.6f}\n"
                f.write(line)


class EnsembleInferencer:

    def __init__(self, 
                 model_weights: List[str],
                 model_weights_weights: Optional[List[float]] = None,
                 config: Optional[Config] = None):
        self.model_paths = model_weights
        self.weights = model_weights_weights or [1.0] * len(model_weights)
        self.config = config or get_config()
        self.models = []

    def load_models(self):
        from ultralytics import YOLO
        print("=" * 60)
        print(f"LOADING ENSEMBLE ({len(self.model_paths)} models)")
        print("=" * 60)
        for i, path in enumerate(self.model_paths):
            print(f"  Loading model {i+1}: {path}")
            self.models.append(YOLO(path))
        print("All models loaded!")

    def predict_ensemble(self,
                         image_path: str,
                         conf_threshold: float = 0.25,
                         iou_threshold: float = 0.45,
                         wbf_iou_threshold: float = 0.55) -> List[Dict]:
        if not self.models:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        all_boxes = []
        all_scores = []
        all_labels = []
        for model in self.models:
            results = model.predict(
                source=image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=self.config.model.input_size,
                device=self.config.training.device,
                verbose=False
            )
            for result in results:
                if result.boxes is not None:
                    xyxyn = result.boxes.xyxyn.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    labels = result.boxes.cls.cpu().numpy().astype(int)
                    all_boxes.append(xyxyn.tolist())
                    all_scores.append(scores.tolist())
                    all_labels.append(labels.tolist())
                else:
                    all_boxes.append([])
                    all_scores.append([])
                    all_labels.append([])
        fused_boxes, fused_scores, fused_labels = self._weighted_box_fusion(
            all_boxes, all_scores, all_labels,
            self.weights, wbf_iou_threshold
        )
        detections = []
        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            x1, y1, x2, y2 = box
            detections.append({
                'class_id': int(label),
                'x_center': (x1 + x2) / 2,
                'y_center': (y1 + y2) / 2,
                'width': x2 - x1,
                'height': y2 - y1,
                'confidence': float(score),
                'class_name': CLASS_NAMES.get(int(label), 'unknown')
            })
        return detections

    def _weighted_box_fusion(self,
                              boxes_list: List[List],
                              scores_list: List[List],
                              labels_list: List[List],
                              weights: List[float],
                              iou_threshold: float) -> Tuple[List, List, List]:
        all_boxes = []
        all_scores = []
        all_labels = []
        all_weights = []
        for boxes, scores, labels, weight in zip(boxes_list, scores_list, labels_list, weights):
            for box, score, label in zip(boxes, scores, labels):
                all_boxes.append(box)
                all_scores.append(score * weight)
                all_labels.append(label)
                all_weights.append(weight)
        if not all_boxes:
            return [], [], []
        class_groups = {}
        for box, score, label, weight in zip(all_boxes, all_scores, all_labels, all_weights):
            if label not in class_groups:
                class_groups[label] = []
            class_groups[label].append((box, score, weight))
        fused_boxes = []
        fused_scores = []
        fused_labels = []
        for label, predictions in class_groups.items():
            predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
            while predictions:
                best = predictions.pop(0)
                box, score, weight = best
                cluster = [best]
                remaining = []
                for other in predictions:
                    other_box = other[0]
                    iou = self._calculate_iou_xyxy(box, other_box)
                    if iou > iou_threshold:
                        cluster.append(other)
                    else:
                        remaining.append(other)
                predictions = remaining
                if len(cluster) >= 1:
                    total_weight = sum(c[2] for c in cluster)
                    fused_box = [0, 0, 0, 0]
                    fused_score = 0
                    for c in cluster:
                        w = c[2] / total_weight
                        for i in range(4):
                            fused_box[i] += c[0][i] * w
                        fused_score += c[1] * w
                    fused_boxes.append(fused_box)
                    fused_scores.append(fused_score)
                    fused_labels.append(label)
        return fused_boxes, fused_scores, fused_labels

    def _calculate_iou_xyxy(self, box1: List, box2: List) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0


class SubmissionGenerator:

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.predictions_dir = PREDICTIONS_DIR
        self.output_dir = OUTPUT_ROOT

    def generate_submission(self,
                            weights: str,
                            use_tta: bool = False,
                            conf_threshold: float = 0.25,
                            iou_threshold: float = 0.45) -> Path:
        print("\n" + "=" * 60)
        print("GENERATING SUBMISSION")
        print("=" * 60)
        if self.predictions_dir.exists():
            shutil.rmtree(self.predictions_dir)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        inferencer = Inferencer(self.config)
        inferencer.load_model(weights)
        predictions = inferencer.predict_batch(
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            use_tta=use_tta
        )
        validation_passed, errors = self._validate_predictions()
        if not validation_passed:
            print("\n" + "=" * 60)
            print("SUBMISSION VALIDATION FAILED")
            print("=" * 60)
            for error in errors[:10]:
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
            print("\nFix these issues before submitting!")
        submission_path = self._create_submission_zip()
        return submission_path

    def _validate_predictions(self) -> Tuple[bool, List[str]]:
        errors = []
        pred_files = list(self.predictions_dir.glob("*.txt"))
        if not pred_files:
            errors.append("No prediction files found!")
            return False, errors
        test_images_dir = self.config.DATASET_DIR / "test" / "images"
        if test_images_dir.exists():
            test_images = list(test_images_dir.glob("*"))
            test_images = [f for f in test_images if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            num_test = len(test_images)
            num_preds = len(pred_files)
            if num_test != num_preds:
                errors.append(f"FILE COUNT MISMATCH: {num_test} test images vs {num_preds} prediction files")
                test_stems = {img.stem for img in test_images}
                pred_stems = {pf.stem for pf in pred_files}
                missing = test_stems - pred_stems
                if missing:
                    for m in list(missing)[:5]:
                        errors.append(f"  Missing prediction for: {m}")
                    if len(missing) > 5:
                        errors.append(f"  ... and {len(missing) - 5} more missing")
        for pred_file in pred_files:
            try:
                with open(pred_file, 'r') as f:
                    lines = f.readlines()
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 6:
                        errors.append(f"{pred_file.name}:{line_num}: Expected 6 values, got {len(parts)}")
                        continue
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        confidence = float(parts[5])
                        if class_id < 0 or class_id > 4:
                            errors.append(f"{pred_file.name}:{line_num}: Invalid class_id {class_id} (must be 0-4)")
                        for val, name in [(x_center, 'x'), (y_center, 'y'), (width, 'w'), (height, 'h')]:
                            if val < 0 or val > 1:
                                errors.append(f"{pred_file.name}:{line_num}: {name}={val} not normalized (must be 0-1)")
                        if confidence < 0 or confidence > 1:
                            errors.append(f"{pred_file.name}:{line_num}: confidence={confidence} out of range (0-1)")
                        if width <= 0 or height <= 0:
                            errors.append(f"{pred_file.name}:{line_num}: Invalid box dimensions w={width}, h={height}")
                    except ValueError as e:
                        errors.append(f"{pred_file.name}:{line_num}: Parse error - {e}")
            except Exception as e:
                errors.append(f"{pred_file.name}: Could not read file - {e}")
        return len(errors) == 0, errors

    def _create_submission_zip(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = self.output_dir / f"submission_{timestamp}.zip"
        print(f"\nCreating submission archive...")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for pred_file in self.predictions_dir.glob("*.txt"):
                arcname = f"predictions/{pred_file.name}"
                zf.write(pred_file, arcname)
        num_files = len(list(self.predictions_dir.glob("*.txt")))
        print(f"\nSubmission created!")
        print(f"  Files included: {num_files}")
        print(f"  Archive path: {zip_path}")
        print(f"  Archive size: {zip_path.stat().st_size / 1024:.1f} KB")
        standard_path = self.output_dir / "submission.zip"
        shutil.copy(zip_path, standard_path)
        print(f"  Standard copy: {standard_path}")
        return standard_path


def find_best_weights() -> Optional[Path]:
    runs = sorted(RUNS_DIR.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def visualize_predictions(image_path: str, 
                         detections: List[Dict],
                         output_path: Optional[str] = None) -> np.ndarray:
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]
    for det in detections:
        x_center = det['x_center'] * w
        y_center = det['y_center'] * h
        box_w = det['width'] * w
        box_h = det['height'] * h
        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)
        class_id = det['class_id']
        color = colors[class_id % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']}: {det['confidence']:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if output_path:
        cv2.imwrite(output_path, image)
    return image


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run inference for road damage detection")
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to model weights (uses latest if not specified)')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image for inference')
    parser.add_argument('--tta', action='store_true',
                        help='Use Test-Time Augmentation')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU threshold')
    parser.add_argument('--submission', action='store_true',
                        help='Generate submission.zip')
    args = parser.parse_args()
    weights = args.weights
    if weights is None:
        weights = find_best_weights()
        if weights is None:
            print("ERROR: No trained weights found. Train a model first.")
            return
        weights = str(weights)
    if args.submission:
        generator = SubmissionGenerator()
        submission_path = generator.generate_submission(
            weights=weights,
            use_tta=args.tta,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        print(f"\nSubmission ready: {submission_path}")
    elif args.image:
        inferencer = Inferencer()
        inferencer.load_model(weights)
        if args.tta:
            detections = inferencer.predict_with_tta(args.image, args.conf, args.iou)
        else:
            detections = inferencer.predict_single(args.image, args.conf, args.iou)
        print(f"\nDetections ({len(detections)}):")
        for det in detections:
            print(f"  {det['class_name']}: {det['confidence']:.3f}")
        output_path = Path(args.image).stem + "_predicted.jpg"
        visualize_predictions(args.image, detections, output_path)
        print(f"\nVisualization saved to: {output_path}")
    else:
        inferencer = Inferencer()
        inferencer.load_model(weights)
        inferencer.predict_batch(use_tta=args.tta)


if __name__ == "__main__":
    main()
