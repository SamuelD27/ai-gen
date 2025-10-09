"""
Dataset preprocessing: smart cropping, resizing, augmentation.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Literal
from PIL import Image
from tqdm import tqdm
import torch

try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None


class DatasetPreprocessor:
    """Preprocess images for LoRA training."""

    def __init__(
        self,
        target_resolution: int = 1024,
        center_crop: bool = False,
        smart_crop: bool = True,
        enable_augmentation: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.target_resolution = target_resolution
        self.center_crop = center_crop
        self.smart_crop = smart_crop
        self.enable_augmentation = enable_augmentation
        self.device = device

        # Initialize face detector for smart cropping
        self.face_detector = None
        if self.smart_crop and MTCNN is not None:
            self.face_detector = MTCNN(
                keep_all=False,
                device=self.device,
                post_process=False
            )

    def detect_face_region(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face region in image."""
        if self.face_detector is None:
            return None

        try:
            boxes, probs = self.face_detector.detect(image)
            if boxes is None or len(boxes) == 0:
                return None

            # Return the first (most confident) face box
            box = boxes[0]
            return tuple(map(int, box))  # (x1, y1, x2, y2)
        except Exception as e:
            print(f"Face detection error: {e}")
            return None

    def smart_crop_image(
        self,
        image: Image.Image,
        target_size: int,
    ) -> Image.Image:
        """
        Smart crop image around detected face.
        Falls back to center crop if no face detected.
        """
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        # Detect face
        face_box = self.detect_face_region(img_array)

        if face_box is not None:
            # Expand face box to include context
            x1, y1, x2, y2 = face_box
            face_w = x2 - x1
            face_h = y2 - y1

            # Expand by 1.5x
            expansion = 0.5
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            new_w = int(face_w * (1 + expansion))
            new_h = int(face_h * (1 + expansion))

            # Make it square
            crop_size = max(new_w, new_h)

            x1 = max(0, cx - crop_size // 2)
            y1 = max(0, cy - crop_size // 2)
            x2 = min(w, x1 + crop_size)
            y2 = min(h, y1 + crop_size)

            # Adjust if we hit boundaries
            if x2 - x1 < crop_size:
                x1 = max(0, x2 - crop_size)
            if y2 - y1 < crop_size:
                y1 = max(0, y2 - crop_size)

            crop_box = (x1, y1, x2, y2)
        else:
            # Fall back to center crop
            crop_size = min(w, h)
            x1 = (w - crop_size) // 2
            y1 = (h - crop_size) // 2
            crop_box = (x1, y1, x1 + crop_size, y1 + crop_size)

        # Crop and resize
        cropped = image.crop(crop_box)
        resized = cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)

        return resized

    def center_crop_image(
        self,
        image: Image.Image,
        target_size: int,
    ) -> Image.Image:
        """Center crop and resize image."""
        w, h = image.size
        crop_size = min(w, h)

        x1 = (w - crop_size) // 2
        y1 = (h - crop_size) // 2

        cropped = image.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        resized = cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)

        return resized

    def resize_image(
        self,
        image: Image.Image,
        target_size: int,
    ) -> Image.Image:
        """Resize image to target size without cropping."""
        return image.resize((target_size, target_size), Image.Resampling.LANCZOS)

    def augment_image(self, image: Image.Image) -> List[Image.Image]:
        """
        Create augmented versions of an image.
        Returns list including original + augmentations.
        """
        if not self.enable_augmentation:
            return [image]

        augmented = [image]

        # Horizontal flip
        augmented.append(image.transpose(Image.FLIP_LEFT_RIGHT))

        # Slight color variations
        from PIL import ImageEnhance

        # Brightness variation
        enhancer = ImageEnhance.Brightness(image)
        augmented.append(enhancer.enhance(1.1))
        augmented.append(enhancer.enhance(0.9))

        # Contrast variation
        enhancer = ImageEnhance.Contrast(image)
        augmented.append(enhancer.enhance(1.1))

        return augmented

    def preprocess_image(
        self,
        image_path: str,
        output_path: str,
        create_augmentations: bool = False,
    ) -> List[str]:
        """
        Preprocess a single image.

        Returns:
            List of output paths (original + augmentations if enabled)
        """
        try:
            image = Image.open(image_path).convert('RGB')

            # Crop/resize
            if self.smart_crop:
                processed = self.smart_crop_image(image, self.target_resolution)
            elif self.center_crop:
                processed = self.center_crop_image(image, self.target_resolution)
            else:
                processed = self.resize_image(image, self.target_resolution)

            # Save main image
            processed.save(output_path, quality=95)
            saved_paths = [output_path]

            # Create augmentations if requested
            if create_augmentations and self.enable_augmentation:
                augmented_images = self.augment_image(processed)

                output_dir = Path(output_path).parent
                output_name = Path(output_path).stem
                output_ext = Path(output_path).suffix

                for i, aug_img in enumerate(augmented_images[1:], 1):
                    aug_path = output_dir / f"{output_name}_aug{i}{output_ext}"
                    aug_img.save(aug_path, quality=95)
                    saved_paths.append(str(aug_path))

            return saved_paths

        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return []

    def preprocess_dataset(
        self,
        input_dir: str,
        output_dir: str,
        create_augmentations: bool = False,
    ) -> dict:
        """
        Preprocess entire dataset.

        Args:
            input_dir: Input directory containing raw images
            output_dir: Output directory for preprocessed images
            create_augmentations: Whether to create augmented versions

        Returns:
            dict: Processing statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        stats = {
            "total_input": len(image_files),
            "total_output": 0,
            "failed": 0,
        }

        print(f"Preprocessing {len(image_files)} images...")

        for img_file in tqdm(image_files, desc="Processing images"):
            output_file = output_path / img_file.name

            saved_paths = self.preprocess_image(
                str(img_file),
                str(output_file),
                create_augmentations=create_augmentations,
            )

            if saved_paths:
                stats["total_output"] += len(saved_paths)
            else:
                stats["failed"] += 1

        print(f"\nPreprocessing complete:")
        print(f"  Input images: {stats['total_input']}")
        print(f"  Output images: {stats['total_output']}")
        print(f"  Failed: {stats['failed']}")

        return stats


def quick_preprocess(
    input_dir: str,
    output_dir: str,
    resolution: int = 1024,
    smart_crop: bool = True,
    augment: bool = False,
) -> dict:
    """
    Quick dataset preprocessing with common defaults.

    Args:
        input_dir: Directory containing raw images
        output_dir: Output directory for processed images
        resolution: Target resolution
        smart_crop: Enable face-aware cropping
        augment: Create augmented versions

    Returns:
        dict: Processing statistics
    """
    preprocessor = DatasetPreprocessor(
        target_resolution=resolution,
        smart_crop=smart_crop,
        enable_augmentation=augment,
    )

    return preprocessor.preprocess_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        create_augmentations=augment,
    )
