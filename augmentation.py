"""
Crackathon Road Damage Detection Pipeline - Data Augmentation

Custom augmentation strategies for road damage detection using Albumentations
and YOLO native augmentations.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import random

import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import AugmentationConfig

# Try to import albumentations
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("WARNING: albumentations not installed. Custom augmentations disabled.")


class RoadDamageAugmentation:
    """
    Custom augmentation pipeline for road damage detection.
    
    This class provides two augmentation strategies:
    1. YOLO native augmentations (via ultralytics config)
    2. Custom Albumentations pipeline for advanced transforms
    
    Architecture Decision:
    - Use YOLO native augmentations during training (integrated with trainer)
    - Apply Albumentations for offline augmentation or specialized transforms
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize augmentation pipeline.
        
        Args:
            config: Augmentation configuration. Uses defaults if None.
        """
        self.config = config or AugmentationConfig()
        
        if ALBUMENTATIONS_AVAILABLE:
            self.train_transform = self._build_train_transform()
            self.val_transform = self._build_val_transform()
        else:
            self.train_transform = None
            self.val_transform = None
    
    def _build_train_transform(self) -> A.Compose:
        """
        Build training augmentation pipeline using Albumentations.
        
        Augmentation Categories:
        1. Geometric: Rotation, flip, scale, perspective
        2. Photometric: Brightness, contrast, hue, saturation
        3. Blur: Motion blur, Gaussian blur, defocus
        4. Weather: Rain, fog, shadows
        5. Quality: Compression, noise
        
        Returns:
            Albumentations Compose object
        """
        transforms = []
        
        # Geometric transforms
        transforms.append(
            A.HorizontalFlip(p=self.config.fliplr)
        )
        
        # Rotation - limited range since roads are typically level
        transforms.append(
            A.Rotate(
                limit=self.config.degrees,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            )
        )
        
        # Scale/Zoom - important for varying damage sizes
        transforms.append(
            A.RandomScale(
                scale_limit=self.config.scale,
                p=0.5
            )
        )
        
        # Perspective transform - simulates different camera angles
        transforms.append(
            A.Perspective(
                scale=(0.05, 0.1),
                p=0.3
            )
        )
        
        # Affine transforms (shift, shear)
        transforms.append(
            A.Affine(
                translate_percent={'x': (-self.config.translate, self.config.translate),
                                  'y': (-self.config.translate, self.config.translate)},
                shear={'x': (-self.config.shear, self.config.shear),
                      'y': (-self.config.shear, self.config.shear)},
                p=0.3
            )
        )
        
        # Photometric transforms
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=self.config.brightness_limit,
                contrast_limit=self.config.contrast_limit,
                p=self.config.brightness_contrast_prob
            )
        )
        
        # HSV adjustments - color variations
        transforms.append(
            A.HueSaturationValue(
                hue_shift_limit=int(self.config.hsv_h * 180),
                sat_shift_limit=int(self.config.hsv_s * 100),
                val_shift_limit=int(self.config.hsv_v * 100),
                p=0.4
            )
        )
        
        # CLAHE - adaptive histogram equalization for low contrast images
        transforms.append(
            A.CLAHE(
                clip_limit=self.config.clahe_clip_limit,
                tile_grid_size=(8, 8),
                p=self.config.clahe_prob
            )
        )
        
        # Random shadow - simulates shadows from trees, buildings, vehicles
        transforms.append(
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),  # Lower half of image
                num_shadows_lower=1,
                num_shadows_upper=3,
                shadow_dimension=5,
                p=self.config.shadow_prob
            )
        )
        
        # Blur transforms
        transforms.append(
            A.MotionBlur(
                blur_limit=self.config.motion_blur_limit,
                p=self.config.motion_blur_prob
            )
        )
        
        # Gaussian blur - slight defocus
        transforms.append(
            A.GaussianBlur(
                blur_limit=self.config.gaussian_blur_limit,
                p=self.config.gaussian_blur_prob
            )
        )
        
        # Weather transforms
        transforms.append(
            A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=15,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=3,
                brightness_coefficient=0.8,
                rain_type='drizzle',
                p=self.config.rain_prob
            )
        )
        
        # Fog effect
        transforms.append(
            A.RandomFog(
                fog_coef_lower=0.1,
                fog_coef_upper=0.3,
                alpha_coef=0.1,
                p=self.config.fog_prob
            )
        )
        
        # Image quality transforms
        transforms.append(
            A.GaussNoise(
                var_limit=self.config.noise_var_limit,
                p=self.config.noise_prob
            )
        )
        
        # JPEG compression artifacts
        transforms.append(
            A.ImageCompression(
                quality_lower=self.config.compression_quality_lower,
                quality_upper=100,
                p=self.config.compression_prob
            )
        )
        
        # Compose all transforms with bounding box support
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3,  # Keep bbox if 30% visible
                min_area=100  # Minimum pixel area
            )
        )
    
    def _build_val_transform(self) -> A.Compose:
        """
        Build validation/test augmentation pipeline.
        
        Validation transforms are minimal to ensure fair evaluation.
        Only normalization and resizing are applied.
        
        Returns:
            Albumentations Compose object
        """
        transforms = [
            # No augmentation for validation - just resize if needed
        ]
        
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.1
            )
        )
    
    def apply_train_augmentation(self, 
                                  image: np.ndarray, 
                                  bboxes: List[List[float]], 
                                  class_labels: List[int]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Apply training augmentations to image and bounding boxes.
        
        Args:
            image: Input image as numpy array (H, W, C)
            bboxes: List of bounding boxes in YOLO format [x_center, y_center, w, h]
            class_labels: List of class IDs
            
        Returns:
            Tuple of (augmented_image, augmented_bboxes, filtered_labels)
        """
        if self.train_transform is None:
            return image, bboxes, class_labels
        
        try:
            result = self.train_transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            return (
                result['image'],
                result['bboxes'],
                result['class_labels']
            )
        except Exception as e:
            # Return original if augmentation fails
            print(f"Augmentation failed: {e}")
            return image, bboxes, class_labels
    
    def apply_val_augmentation(self,
                               image: np.ndarray,
                               bboxes: List[List[float]],
                               class_labels: List[int]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Apply validation augmentations (minimal transforms).
        
        Args:
            image: Input image as numpy array
            bboxes: List of bounding boxes in YOLO format
            class_labels: List of class IDs
            
        Returns:
            Tuple of (image, bboxes, labels)
        """
        if self.val_transform is None:
            return image, bboxes, class_labels
        
        try:
            result = self.val_transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            
            return (
                result['image'],
                result['bboxes'],
                result['class_labels']
            )
        except Exception:
            return image, bboxes, class_labels


def get_yolo_augmentation_config(config: Optional[AugmentationConfig] = None) -> Dict[str, Any]:
    """
    Generate YOLO-native augmentation parameters for ultralytics training.
    
    These parameters are passed directly to the YOLO trainer and leverage
    the built-in augmentation pipeline (mosaic, mixup, etc.).
    
    Args:
        config: Augmentation configuration
        
    Returns:
        Dictionary of YOLO augmentation parameters
    """
    if config is None:
        config = AugmentationConfig()
    
    return {
        # HSV augmentation
        'hsv_h': config.hsv_h,
        'hsv_s': config.hsv_s,
        'hsv_v': config.hsv_v,
        
        # Geometric augmentation
        'degrees': config.degrees,
        'translate': config.translate,
        'scale': config.scale,
        'shear': config.shear,
        'perspective': config.perspective,
        
        # Flip augmentation
        'flipud': config.flipud,
        'fliplr': config.fliplr,
        
        # Mosaic & MixUp
        'mosaic': config.mosaic,
        'mixup': config.mixup,
        'copy_paste': config.copy_paste,
        
        # Additional
        'erasing': 0.4,
        'crop_fraction': 1.0,
    }


def get_test_time_augmentation_config() -> Dict[str, Any]:
    """
    Generate Test-Time Augmentation (TTA) configuration.
    
    TTA applies multiple augmentations during inference and aggregates
    predictions to improve robustness. This can boost mAP at the cost
    of slower inference.
    
    Augmentations for TTA:
    - Multi-scale inference (0.83x, 1x, 1.17x)
    - Horizontal flip
    
    Returns:
        Dictionary of TTA parameters
    """
    return {
        'augment': True,  # Enable TTA
        'scales': [0.83, 1.0, 1.17],  # Multi-scale
        'flips': [False, True],  # Original + horizontal flip
    }


class OfflineAugmentor:
    """
    Offline data augmentation to expand training dataset.
    
    Use this class to generate additional training samples before training.
    This is particularly useful for:
    - Balancing class distributions
    - Creating hard examples
    - Expanding limited datasets
    
    Note: YOLO's online augmentation is usually sufficient. Use this
    only if you need specific offline augmentation strategies.
    """
    
    def __init__(self, 
                 augmentor: RoadDamageAugmentation,
                 num_augmentations: int = 3):
        """
        Initialize offline augmentor.
        
        Args:
            augmentor: RoadDamageAugmentation instance
            num_augmentations: Number of augmented versions per image
        """
        self.augmentor = augmentor
        self.num_augmentations = num_augmentations
    
    def augment_image(self, 
                      image_path: Path,
                      label_path: Path,
                      output_dir: Path) -> List[Tuple[Path, Path]]:
        """
        Generate augmented versions of a single image.
        
        Args:
            image_path: Path to input image
            label_path: Path to label file
            output_dir: Directory for augmented outputs
            
        Returns:
            List of (image_path, label_path) tuples for generated files
        """
        from PIL import Image
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        bboxes = []
        class_labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_labels.append(int(parts[0]))
                        bboxes.append([float(x) for x in parts[1:5]])
        
        # Generate augmentations
        output_files = []
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(self.num_augmentations):
            # Apply augmentation
            aug_image, aug_bboxes, aug_labels = self.augmentor.apply_train_augmentation(
                image, bboxes, class_labels
            )
            
            # Save augmented image
            aug_name = f"{image_path.stem}_aug{i}{image_path.suffix}"
            aug_image_path = output_dir / "images" / aug_name
            aug_image_path.parent.mkdir(parents=True, exist_ok=True)
            
            aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(aug_image_path), aug_image_bgr)
            
            # Save augmented labels
            aug_label_path = output_dir / "labels" / f"{image_path.stem}_aug{i}.txt"
            aug_label_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(aug_label_path, 'w') as f:
                for bbox, label in zip(aug_bboxes, aug_labels):
                    f.write(f"{label} {' '.join(map(str, bbox))}\n")
            
            output_files.append((aug_image_path, aug_label_path))
        
        return output_files


def visualize_augmentations(image_path: Path, 
                            label_path: Path,
                            num_samples: int = 6) -> None:
    """
    Visualize augmentation effects on a sample image.
    
    Useful for debugging and tuning augmentation parameters.
    
    Args:
        image_path: Path to input image
        label_path: Path to label file
        num_samples: Number of augmented samples to generate
    """
    import matplotlib.pyplot as plt
    
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load labels
    bboxes = []
    class_labels = []
    
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_labels.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:5]])
    
    # Create augmentor
    augmentor = RoadDamageAugmentation()
    
    # Generate and visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Original
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Augmented versions
    for i in range(1, min(num_samples, 6)):
        aug_image, _, _ = augmentor.apply_train_augmentation(
            image.copy(), bboxes.copy(), class_labels.copy()
        )
        axes[i].imshow(aug_image)
        axes[i].set_title(f"Augmented {i}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("augmentation_samples.png", dpi=150)
    plt.show()
    print("Saved augmentation samples to augmentation_samples.png")


if __name__ == "__main__":
    # Test augmentation configuration
    config = AugmentationConfig()
    
    print("=" * 60)
    print("Road Damage Augmentation Configuration")
    print("=" * 60)
    
    print("\nYOLO Native Augmentations:")
    yolo_config = get_yolo_augmentation_config(config)
    for key, value in yolo_config.items():
        print(f"  {key}: {value}")
    
    print("\nTest-Time Augmentation:")
    tta_config = get_test_time_augmentation_config()
    for key, value in tta_config.items():
        print(f"  {key}: {value}")
    
    if ALBUMENTATIONS_AVAILABLE:
        print("\n✓ Albumentations available for custom augmentations")
        augmentor = RoadDamageAugmentation(config)
        print(f"  Training transforms: {len(augmentor.train_transform.transforms)} operations")
    else:
        print("\n✗ Albumentations not available")
