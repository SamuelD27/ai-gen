"""
Dataset cleaning utilities: face detection, deduplication, quality filtering.
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import imagehash
from tqdm import tqdm

try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None

try:
    import pyiqa
except ImportError:
    pyiqa = None


class DatasetCleaner:
    """Clean and filter datasets for LoRA training."""

    def __init__(
        self,
        min_resolution: int = 512,
        max_aspect_ratio: float = 2.0,
        enable_face_detection: bool = True,
        enable_deduplication: bool = True,
        enable_quality_filter: bool = True,
        quality_threshold: float = 0.3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.min_resolution = min_resolution
        self.max_aspect_ratio = max_aspect_ratio
        self.enable_face_detection = enable_face_detection
        self.enable_deduplication = enable_deduplication
        self.enable_quality_filter = enable_quality_filter
        self.quality_threshold = quality_threshold
        self.device = device

        # Initialize face detector
        self.face_detector = None
        if self.enable_face_detection and MTCNN is not None:
            self.face_detector = MTCNN(
                keep_all=False,
                device=self.device,
                post_process=False
            )

        # Initialize quality assessor
        self.quality_assessor = None
        if self.enable_quality_filter and pyiqa is not None:
            # Using NIQE (no-reference quality assessment)
            self.quality_assessor = pyiqa.create_metric(
                'niqe', device=self.device
            )

    def check_resolution(self, image_path: str) -> Tuple[bool, str]:
        """Check if image meets minimum resolution requirements."""
        try:
            img = Image.open(image_path)
            width, height = img.size

            if width < self.min_resolution or height < self.min_resolution:
                return False, f"Resolution too low: {width}x{height}"

            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > self.max_aspect_ratio:
                return False, f"Aspect ratio too extreme: {aspect_ratio:.2f}"

            return True, "OK"
        except Exception as e:
            return False, f"Error reading image: {e}"

    def check_face_detection(self, image_path: str) -> Tuple[bool, str]:
        """Check if image contains a detectable face."""
        if not self.enable_face_detection or self.face_detector is None:
            return True, "Face detection disabled"

        try:
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)

            boxes, probs = self.face_detector.detect(img_array)

            if boxes is None or len(boxes) == 0:
                return False, "No face detected"

            # Check if face is prominent enough (at least 20% of image)
            img_area = img.size[0] * img.size[1]
            box = boxes[0]
            face_area = (box[2] - box[0]) * (box[3] - box[1])
            face_ratio = face_area / img_area

            if face_ratio < 0.05:
                return False, f"Face too small: {face_ratio:.2%}"

            return True, f"Face detected ({probs[0]:.2f} confidence)"
        except Exception as e:
            return False, f"Face detection error: {e}"

    def compute_image_hash(self, image_path: str) -> Optional[str]:
        """Compute perceptual hash for deduplication."""
        try:
            img = Image.open(image_path)
            # Using average hash (fast and effective)
            hash_value = imagehash.average_hash(img, hash_size=16)
            return str(hash_value)
        except Exception as e:
            print(f"Error computing hash for {image_path}: {e}")
            return None

    def assess_quality(self, image_path: str) -> Tuple[bool, float, str]:
        """Assess image quality (blur, noise, compression artifacts)."""
        if not self.enable_quality_filter or self.quality_assessor is None:
            return True, 1.0, "Quality filter disabled"

        try:
            img = Image.open(image_path).convert('RGB')

            # NIQE score (lower is better, typically 0-100)
            score = self.quality_assessor(img).item()

            # Normalize and invert (higher is better)
            # Typical NIQE range: 0-10 (excellent) to 50+ (poor)
            normalized_score = 1.0 / (1.0 + score / 10.0)

            if normalized_score < self.quality_threshold:
                return False, normalized_score, f"Low quality: {score:.2f} NIQE"

            return True, normalized_score, f"Quality OK: {score:.2f} NIQE"
        except Exception as e:
            return True, 0.5, f"Quality assessment error: {e}"

    def clean_dataset(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        copy_files: bool = True,
    ) -> dict:
        """
        Clean a dataset by filtering out invalid images.

        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for cleaned dataset (if None, creates filtered list)
            copy_files: If True, copy valid files to output_dir

        Returns:
            dict: Statistics about the cleaning process
        """
        input_path = Path(input_dir)
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        stats = {
            "total": len(image_files),
            "valid": 0,
            "invalid_resolution": 0,
            "invalid_face": 0,
            "invalid_quality": 0,
            "duplicates": 0,
        }

        valid_files = []
        seen_hashes = set()

        print(f"Cleaning dataset: {len(image_files)} images")

        for img_file in tqdm(image_files, desc="Filtering images"):
            # Check resolution
            resolution_ok, resolution_msg = self.check_resolution(str(img_file))
            if not resolution_ok:
                stats["invalid_resolution"] += 1
                continue

            # Check face detection
            face_ok, face_msg = self.check_face_detection(str(img_file))
            if not face_ok:
                stats["invalid_face"] += 1
                continue

            # Check quality
            quality_ok, quality_score, quality_msg = self.assess_quality(str(img_file))
            if not quality_ok:
                stats["invalid_quality"] += 1
                continue

            # Check for duplicates
            if self.enable_deduplication:
                img_hash = self.compute_image_hash(str(img_file))
                if img_hash and img_hash in seen_hashes:
                    stats["duplicates"] += 1
                    continue
                if img_hash:
                    seen_hashes.add(img_hash)

            # Image passed all checks
            stats["valid"] += 1
            valid_files.append(img_file)

            # Copy to output directory if requested
            if copy_files and output_dir:
                import shutil
                shutil.copy2(img_file, output_path / img_file.name)

        print(f"\nCleaning complete:")
        print(f"  Total: {stats['total']}")
        print(f"  Valid: {stats['valid']}")
        print(f"  Invalid resolution: {stats['invalid_resolution']}")
        print(f"  Invalid face: {stats['invalid_face']}")
        print(f"  Invalid quality: {stats['invalid_quality']}")
        print(f"  Duplicates: {stats['duplicates']}")

        return stats


def quick_clean(
    input_dir: str,
    output_dir: str,
    min_resolution: int = 512,
    require_faces: bool = True,
    quality_filter: bool = True,
) -> dict:
    """
    Quick dataset cleaning with common defaults.

    Args:
        input_dir: Directory containing images
        output_dir: Output directory for cleaned images
        min_resolution: Minimum image resolution
        require_faces: Only keep images with detected faces
        quality_filter: Enable quality assessment

    Returns:
        dict: Cleaning statistics
    """
    cleaner = DatasetCleaner(
        min_resolution=min_resolution,
        enable_face_detection=require_faces,
        enable_quality_filter=quality_filter,
    )

    return cleaner.clean_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        copy_files=True,
    )
