"""
BLIP2 image captioning backend.
"""

import torch
from pathlib import Path
from typing import List, Optional
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class BLIP2Captioner:
    """BLIP2-based image captioning."""

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir

        self.processor = None
        self.model = None
        self.is_loaded = False

    def load(self):
        """Load model and processor."""
        if self.is_loaded:
            return

        print(f"Loading BLIP2 model: {self.model_name}")

        self.processor = Blip2Processor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )

        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            cache_dir=self.cache_dir,
        ).to(self.device)

        self.is_loaded = True
        print("BLIP2 model loaded successfully")

    def caption_image(
        self,
        image_path: str,
        prompt: str = "A photo of",
        max_length: int = 50,
        num_beams: int = 5,
    ) -> str:
        """
        Generate caption for a single image.

        Args:
            image_path: Path to image file
            prompt: Optional prompt to guide captioning
            max_length: Maximum caption length
            num_beams: Number of beams for beam search

        Returns:
            str: Generated caption
        """
        if not self.is_loaded:
            self.load()

        try:
            image = Image.open(image_path).convert('RGB')

            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                )

            caption = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()

            return caption

        except Exception as e:
            print(f"Error captioning {image_path}: {e}")
            return ""

    def caption_batch(
        self,
        image_paths: List[str],
        prompt: str = "A photo of",
        max_length: int = 50,
        batch_size: int = 4,
    ) -> List[str]:
        """
        Caption multiple images in batches.

        Args:
            image_paths: List of image paths
            prompt: Optional prompt to guide captioning
            max_length: Maximum caption length
            batch_size: Batch size for processing

        Returns:
            List[str]: Generated captions
        """
        if not self.is_loaded:
            self.load()

        captions = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            try:
                images = [Image.open(p).convert('RGB') for p in batch_paths]

                inputs = self.processor(
                    images=images,
                    text=[prompt] * len(images),
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=5,
                    )

                batch_captions = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )

                captions.extend([c.strip() for c in batch_captions])

            except Exception as e:
                print(f"Error captioning batch: {e}")
                captions.extend([""] * len(batch_paths))

        return captions

    def unload(self):
        """Unload model to free memory."""
        if self.is_loaded:
            del self.model
            del self.processor
            torch.cuda.empty_cache()
            self.is_loaded = False
