"""
Crackathon Road Damage Detection Pipeline - Dataset Handler

This module handles:
1. Dataset download from Google Drive
2. Dataset organization and validation
3. Image-label consistency verification
4. Train/validation split creation
5. YOLO format data.yaml generation

Dataset Structure Expected:
    rdd2022/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        └── images/
"""

import os
import sys
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import random

import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PROJECT_ROOT, DATA_ROOT, DATASET_ROOT, 
    GDRIVE_FOLDER_ID, GDRIVE_URL,
    CLASS_NAMES, NUM_CLASSES
)


class DatasetHandler:
    """
    Handles dataset download, organization, and validation for RDD2022.
    
    Key Responsibilities:
    - Download dataset from Google Drive using gdown
    - Verify dataset integrity (image-label pairs)
    - Validate label format (YOLO TXT format)
    - Create or verify train/val splits
    - Generate YOLO-compatible data.yaml
    """
    
    # Supported image formats
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    def __init__(self, dataset_root: Path = DATASET_ROOT):
        """
        Initialize DatasetHandler.
        
        Args:
            dataset_root: Root directory for dataset storage
        """
        self.dataset_root = Path(dataset_root)
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        
        # Dataset split paths
        self.train_dir = self.dataset_root / "train"
        self.val_dir = self.dataset_root / "val"
        self.test_dir = self.dataset_root / "test"
        
        # Statistics
        self.stats = {
            'train': {'images': 0, 'labels': 0, 'annotations': 0},
            'val': {'images': 0, 'labels': 0, 'annotations': 0},
            'test': {'images': 0, 'labels': 0, 'annotations': 0}
        }
        
        # Class distribution
        self.class_distribution = {
            'train': defaultdict(int),
            'val': defaultdict(int),
            'test': defaultdict(int)
        }
    
    def download_dataset(self, force: bool = False) -> bool:
        """
        Download dataset from Google Drive.
        
        The RDD2022 dataset is stored as a folder on Google Drive.
        We use gdown to download it with folder support.
        
        Args:
            force: If True, re-download even if dataset exists
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            import gdown
        except ImportError:
            print("ERROR: gdown not installed. Run: pip install gdown")
            return False
        
        # Check if dataset already exists
        if not force and self._verify_dataset_exists():
            print("Dataset already exists. Use force=True to re-download.")
            return True
        
        print("=" * 60)
        print("Downloading RDD2022 Dataset from Google Drive")
        print("=" * 60)
        print(f"Source: {GDRIVE_URL}")
        print(f"Destination: {self.dataset_root}")
        print()
        
        # NOTE: gdown has a 50-file limit per folder
        # For large datasets, we need to handle this limitation
        
        try:
            # First try direct folder download (works for small folders)
            gdown.download_folder(
                url=GDRIVE_URL,
                output=str(self.dataset_root.parent),
                quiet=False,
                use_cookies=False,
                remaining_ok=True  # Continue even if some files fail
            )
            
            # Check if download was complete enough
            if self._verify_dataset_exists():
                print("\nDownload completed!")
                return True
            else:
                # Partial download - provide manual instructions
                raise Exception("Partial download - dataset incomplete")
            
        except Exception as e:
            print(f"\nAutomatic download failed: {e}")
            print("\nManual download required:")
            print(f"  1. Open: {GDRIVE_URL}")
            print(f"  2. Download and extract to: {self.dataset_root}")
            print("  3. Re-run the pipeline")
            print("=" * 60)
            return False
    
    def _verify_dataset_exists(self) -> bool:
        """Check if dataset directories exist and contain data."""
        required_dirs = [
            self.train_dir / "images",
            self.train_dir / "labels",
            self.test_dir / "images"
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                return False
            if not any(dir_path.iterdir()):
                return False
        
        return True
    
    def organize_dataset(self) -> bool:
        """
        Organize dataset into standard YOLO structure.
        
        Expected final structure:
            dataset_root/
            ├── train/
            │   ├── images/
            │   └── labels/
            ├── val/
            │   ├── images/
            │   └── labels/
            └── test/
                └── images/
        
        Returns:
            True if organization successful
        """
        print("=" * 60)
        print("Organizing Dataset Structure")
        print("=" * 60)
        
        # Create required directories
        for split in ['train', 'val', 'test']:
            split_dir = self.dataset_root / split
            (split_dir / "images").mkdir(parents=True, exist_ok=True)
            if split != 'test':
                (split_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        # Check for various possible structures after download
        self._handle_nested_structure()
        
        print("Dataset organization complete!")
        return True
    
    def _handle_nested_structure(self):
        """
        Handle various nested structures that might occur after download.
        
        Google Drive downloads can create nested folders. This method
        flattens them to the expected structure.
        """
        # Look for nested rdd2022 or similar folders
        for item in self.dataset_root.parent.iterdir():
            if item.is_dir() and item != self.dataset_root:
                # Check if this is the downloaded folder
                if (item / "train").exists() or (item / "test").exists():
                    print(f"Found dataset in: {item}")
                    # Move contents to expected location
                    for sub in item.iterdir():
                        dest = self.dataset_root / sub.name
                        if sub.is_dir() and not dest.exists():
                            shutil.move(str(sub), str(dest))
                        elif sub.is_dir() and dest.exists():
                            # Merge contents
                            for file in sub.rglob("*"):
                                if file.is_file():
                                    rel_path = file.relative_to(sub)
                                    dest_file = dest / rel_path
                                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                                    if not dest_file.exists():
                                        shutil.move(str(file), str(dest_file))
    
    def verify_image_label_consistency(self) -> Dict[str, List[str]]:
        """
        Verify that each image has a corresponding label file.
        
        YOLO format requires:
        - Each image file has a corresponding .txt label file
        - Label file has same base name as image
        - Label files contain valid YOLO annotations
        
        Returns:
            Dictionary of issues found per split
        """
        print("=" * 60)
        print("Verifying Image-Label Consistency")
        print("=" * 60)
        
        issues = {
            'missing_labels': [],
            'missing_images': [],
            'invalid_labels': [],
            'empty_labels': []
        }
        
        for split in ['train', 'val']:
            split_dir = self.dataset_root / split
            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"
            
            if not images_dir.exists():
                print(f"WARNING: {images_dir} does not exist")
                continue
            
            print(f"\nChecking {split} split...")
            
            # Get all image files
            image_files = set()
            for ext in self.IMAGE_EXTENSIONS:
                image_files.update(f.stem for f in images_dir.glob(f"*{ext}"))
                image_files.update(f.stem for f in images_dir.glob(f"*{ext.upper()}"))
            
            # Get all label files
            label_files = set(f.stem for f in labels_dir.glob("*.txt"))
            
            # Check for missing labels
            missing_labels = image_files - label_files
            if missing_labels:
                issues['missing_labels'].extend(
                    [f"{split}/{name}" for name in list(missing_labels)[:10]]
                )
                print(f"  WARNING: {len(missing_labels)} images without labels")
            
            # Check for orphan labels
            orphan_labels = label_files - image_files
            if orphan_labels:
                issues['missing_images'].extend(
                    [f"{split}/{name}" for name in list(orphan_labels)[:10]]
                )
                print(f"  WARNING: {len(orphan_labels)} labels without images")
            
            # Validate label format
            valid_count = 0
            invalid_count = 0
            empty_count = 0
            
            for label_file in tqdm(labels_dir.glob("*.txt"), desc=f"Validating {split} labels"):
                is_valid, error = self._validate_label_file(label_file)
                if is_valid:
                    valid_count += 1
                elif error == 'empty':
                    # Empty labels mean no objects - could be valid negative sample
                    # or could indicate missing annotations. Track but don't fail.
                    empty_count += 1
                    issues['empty_labels'].append(f"{split}/{label_file.stem}")
                    # Note: YOLO handles empty .txt files (negative samples)
                else:
                    invalid_count += 1
                    issues['invalid_labels'].append(f"{split}/{label_file.stem}: {error}")
            
            print(f"  Valid labels: {valid_count}")
            print(f"  Empty labels: {empty_count}")
            print(f"  Invalid labels: {invalid_count}")
        
        # Summary
        total_issues = sum(len(v) for v in issues.values())
        if total_issues == 0:
            print("\n✓ All images and labels are consistent!")
        else:
            print(f"\n⚠ Found {total_issues} issues (see details above)")
        
        return issues
    
    def _validate_label_file(self, label_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate a single YOLO format label file.
        
        YOLO format: <class_id> <x_center> <y_center> <width> <height>
        - class_id: integer in range [0, NUM_CLASSES-1]
        - x_center, y_center, width, height: floats in range [0, 1]
        
        Args:
            label_path: Path to label file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if not lines or all(line.strip() == '' for line in lines):
                return False, 'empty'  # Empty labels = no annotations, flag for review
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    return False, f"Line {i+1}: Expected 5 values, got {len(parts)}"
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                except ValueError:
                    return False, f"Line {i+1}: Invalid number format"
                
                # Validate class ID
                if class_id < 0 or class_id >= NUM_CLASSES:
                    return False, f"Line {i+1}: Invalid class_id {class_id}"
                
                # Validate coordinates: must be in (0, 1] range
                # Centers can be anywhere in image, but must be positive
                for val, name in [(x_center, 'x_center'), (y_center, 'y_center')]:
                    if val < 0.0 or val > 1.0:
                        return False, f"Line {i+1}: {name}={val} out of range (0,1]"
                
                # Width/height must be strictly positive and <= 1
                for val, name in [(width, 'width'), (height, 'height')]:
                    if val <= 0.0 or val > 1.0:
                        return False, f"Line {i+1}: {name}={val} must be in (0,1]"
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def fix_label_file(self, label_path: Path) -> Tuple[bool, int]:
        """
        Fix out-of-range values in a label file by clamping.
        
        This is a repair function - use after validation identifies issues.
        
        Args:
            label_path: Path to label file
            
        Returns:
            Tuple of (was_modified, num_lines_fixed)
        """
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            fixed_lines = []
            num_fixed = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    continue  # Skip malformed lines
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Skip invalid class IDs
                    if class_id < 0 or class_id >= NUM_CLASSES:
                        continue
                    
                    # Clamp coordinates to valid range
                    original = (x_center, y_center, width, height)
                    
                    x_center = max(0.001, min(0.999, x_center))
                    y_center = max(0.001, min(0.999, y_center))
                    width = max(0.001, min(1.0, width))
                    height = max(0.001, min(1.0, height))
                    
                    # Ensure box doesn't exceed image bounds
                    if x_center - width/2 < 0:
                        width = x_center * 2
                    if x_center + width/2 > 1:
                        width = (1 - x_center) * 2
                    if y_center - height/2 < 0:
                        height = y_center * 2
                    if y_center + height/2 > 1:
                        height = (1 - y_center) * 2
                    
                    # Check if we modified anything
                    if (x_center, y_center, width, height) != original:
                        num_fixed += 1
                    
                    # Skip degenerate boxes
                    if width <= 0.001 or height <= 0.001:
                        continue
                    
                    fixed_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    
                except ValueError:
                    continue
            
            # Write back if we have valid lines
            if fixed_lines:
                with open(label_path, 'w') as f:
                    f.write('\n'.join(fixed_lines) + '\n')
                return True, num_fixed
            
            return False, 0
            
        except Exception as e:
            print(f"Error fixing {label_path}: {e}")
            return False, 0
    
    def compute_statistics(self) -> Dict:
        """
        Compute comprehensive dataset statistics.
        
        Returns:
            Dictionary containing:
            - Image counts per split
            - Label counts per split
            - Total annotations
            - Class distribution
            - Image size statistics
        """
        print("=" * 60)
        print("Computing Dataset Statistics")
        print("=" * 60)
        
        stats = {
            'splits': {},
            'class_distribution': {},
            'image_sizes': {'widths': [], 'heights': []},
            'bbox_sizes': {'widths': [], 'heights': []},
            'total_images': 0,
            'total_annotations': 0
        }
        
        for split in ['train', 'val', 'test']:
            split_dir = self.dataset_root / split
            images_dir = split_dir / "images"
            labels_dir = split_dir / "labels"
            
            if not images_dir.exists():
                continue
            
            # Count images
            image_files = []
            for ext in self.IMAGE_EXTENSIONS:
                image_files.extend(images_dir.glob(f"*{ext}"))
                image_files.extend(images_dir.glob(f"*{ext.upper()}"))
            
            num_images = len(image_files)
            # Test set should NOT have labels - don't count them
            num_labels = len(list(labels_dir.glob("*.txt"))) if (labels_dir.exists() and split != 'test') else 0
            
            # Count annotations and class distribution
            # IMPORTANT: Only for train/val - test labels must not exist
            num_annotations = 0
            class_counts = defaultdict(int)
            
            if labels_dir.exists() and split != 'test':
                for label_file in labels_dir.glob("*.txt"):
                    with open(label_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    class_counts[class_id] += 1
                                    num_annotations += 1
                                    
                                    # Store bbox sizes
                                    if split == 'train':
                                        stats['bbox_sizes']['widths'].append(float(parts[3]))
                                        stats['bbox_sizes']['heights'].append(float(parts[4]))
            
            stats['splits'][split] = {
                'images': num_images,
                'labels': num_labels,
                'annotations': num_annotations
            }
            stats['class_distribution'][split] = dict(class_counts)
            stats['total_images'] += num_images
            stats['total_annotations'] += num_annotations
            
            print(f"\n{split.upper()} Split:")
            print(f"  Images: {num_images}")
            print(f"  Labels: {num_labels}")
            print(f"  Annotations: {num_annotations}")
            
            if class_counts:
                print("  Class distribution:")
                for class_id in sorted(class_counts.keys()):
                    class_name = CLASS_NAMES.get(class_id, f"Unknown_{class_id}")
                    print(f"    {class_id} ({class_name}): {class_counts[class_id]}")
        
        # Sample image sizes
        if image_files:
            print("\nSampling image sizes...")
            sample_size = min(100, len(image_files))
            for img_path in random.sample(image_files, sample_size):
                try:
                    with Image.open(img_path) as img:
                        stats['image_sizes']['widths'].append(img.width)
                        stats['image_sizes']['heights'].append(img.height)
                except Exception:
                    pass
            
            if stats['image_sizes']['widths']:
                avg_w = np.mean(stats['image_sizes']['widths'])
                avg_h = np.mean(stats['image_sizes']['heights'])
                print(f"\nAverage image size: {avg_w:.0f} x {avg_h:.0f}")
        
        return stats
    
    def create_train_val_split(self, val_ratio: float = 0.2, 
                               stratify: bool = True,
                               seed: int = 42) -> bool:
        """
        Create train/validation split if not already provided.
        
        Uses stratified sampling to maintain class distribution.
        
        Args:
            val_ratio: Fraction of training data to use for validation
            stratify: Whether to stratify by class distribution
            seed: Random seed for reproducibility
            
        Returns:
            True if split created/verified successfully
        """
        # Check if val split already has data
        val_images = self.val_dir / "images"
        if val_images.exists() and any(val_images.iterdir()):
            print("Validation split already exists. Skipping creation.")
            return True
        
        print("=" * 60)
        print("Creating Train/Validation Split")
        print("=" * 60)
        print(f"Validation ratio: {val_ratio}")
        print(f"Stratified: {stratify}")
        print(f"Random seed: {seed}")
        
        random.seed(seed)
        np.random.seed(seed)
        
        train_images_dir = self.train_dir / "images"
        train_labels_dir = self.train_dir / "labels"
        
        if not train_images_dir.exists():
            print("ERROR: Training images directory not found!")
            return False
        
        # Get all training images
        image_files = []
        for ext in self.IMAGE_EXTENSIONS:
            image_files.extend(train_images_dir.glob(f"*{ext}"))
            image_files.extend(train_images_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print("ERROR: No training images found!")
            return False
        
        print(f"\nTotal training images: {len(image_files)}")
        
        if stratify:
            # Group images by primary class
            class_to_images = defaultdict(list)
            
            for img_path in image_files:
                label_path = train_labels_dir / f"{img_path.stem}.txt"
                primary_class = self._get_primary_class(label_path)
                class_to_images[primary_class].append(img_path)
            
            # Split each class proportionally
            val_images_list = []
            for class_id, images in class_to_images.items():
                n_val = max(1, int(len(images) * val_ratio))
                random.shuffle(images)
                val_images_list.extend(images[:n_val])
            
            val_set = set(val_images_list)
        else:
            # Random split
            random.shuffle(image_files)
            n_val = int(len(image_files) * val_ratio)
            val_set = set(image_files[:n_val])
        
        # Create validation directories
        (self.val_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.val_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        # Move validation files
        moved_count = 0
        for img_path in tqdm(val_set, desc="Moving validation files"):
            # Move image
            dest_img = self.val_dir / "images" / img_path.name
            shutil.copy2(str(img_path), str(dest_img))
            
            # Move label
            label_path = train_labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                dest_label = self.val_dir / "labels" / label_path.name
                shutil.copy2(str(label_path), str(dest_label))
            
            moved_count += 1
        
        print(f"\nMoved {moved_count} images to validation set")
        
        return True
    
    def _get_primary_class(self, label_path: Path) -> int:
        """
        Get the primary (most frequent) class in a label file.
        
        Used for stratified splitting.
        
        Args:
            label_path: Path to label file
            
        Returns:
            Primary class ID (or -1 for empty/missing labels)
        """
        if not label_path.exists():
            return -1
        
        class_counts = defaultdict(int)
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_counts[int(parts[0])] += 1
        except Exception:
            return -1
        
        if not class_counts:
            return -1
        
        return max(class_counts.keys(), key=lambda k: class_counts[k])
    
    def generate_data_yaml(self, output_path: Optional[Path] = None) -> Path:
        """
        Generate YOLO-compatible data.yaml configuration file.
        
        The data.yaml file specifies:
        - Dataset paths (train, val, test)
        - Number of classes
        - Class names
        
        Args:
            output_path: Optional custom output path
            
        Returns:
            Path to generated data.yaml file
        """
        if output_path is None:
            output_path = self.dataset_root / "data.yaml"
        
        print("=" * 60)
        print("Generating data.yaml")
        print("=" * 60)
        
        # Use absolute paths for reliability
        data_config = {
            'path': str(self.dataset_root.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': NUM_CLASSES,
            'names': {i: name for i, name in CLASS_NAMES.items()}
        }
        
        # Write YAML file
        with open(output_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Generated: {output_path}")
        print("\nContents:")
        print("-" * 40)
        with open(output_path, 'r') as f:
            print(f.read())
        
        return output_path
    
    def prepare_dataset(self, 
                       download: bool = True,
                       verify: bool = True,
                       create_split: bool = True,
                       val_ratio: float = 0.2) -> Path:
        """
        Complete dataset preparation pipeline.
        
        Steps:
        1. Download dataset from Google Drive
        2. Organize into YOLO structure
        3. Verify image-label consistency
        4. Create train/val split if needed
        5. Generate data.yaml
        6. Compute and display statistics
        
        Args:
            download: Whether to download dataset
            verify: Whether to verify consistency
            create_split: Whether to create train/val split
            val_ratio: Validation split ratio
            
        Returns:
            Path to data.yaml file
        """
        print("=" * 60)
        print("CRACKATHON DATASET PREPARATION")
        print("=" * 60)
        print()
        
        # Step 1: Download
        if download:
            self.download_dataset()
        
        # Step 2: Organize
        self.organize_dataset()
        
        # Step 3: Verify
        if verify:
            issues = self.verify_image_label_consistency()
        
        # Step 4: Create split
        if create_split:
            self.create_train_val_split(val_ratio=val_ratio)
        
        # Step 5: Generate data.yaml
        data_yaml_path = self.generate_data_yaml()
        
        # Step 6: Statistics
        self.compute_statistics()
        
        print("\n" + "=" * 60)
        print("DATASET PREPARATION COMPLETE")
        print("=" * 60)
        print(f"\nData YAML: {data_yaml_path}")
        
        return data_yaml_path


def main():
    """Main function for dataset preparation."""
    handler = DatasetHandler()
    data_yaml = handler.prepare_dataset(
        download=True,
        verify=True,
        create_split=True,
        val_ratio=0.2
    )
    print(f"\nReady for training! Use: {data_yaml}")


if __name__ == "__main__":
    main()
