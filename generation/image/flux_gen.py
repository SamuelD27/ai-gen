"""
Flux image generation - No content restrictions.
"""

import os
import torch
from pathlib import Path
from typing import List, Optional, Union
from PIL import Image
from diffusers import FluxPipeline
from tqdm import tqdm


class FluxGenerator:
    """
    Flux image generation with LoRA support.
    No safety checkers, no content restrictions.
    """

    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.1-dev",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
        enable_optimization: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir or os.getenv("HF_HOME")
        self.enable_optimization = enable_optimization

        self.pipeline = None
        self.is_loaded = False
        self.current_lora = None

    def load(self):
        """Load Flux pipeline."""
        if self.is_loaded:
            return

        print(f"Loading Flux pipeline: {self.model_name}")

        self.pipeline = FluxPipeline.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            cache_dir=self.cache_dir,
        ).to(self.device)

        if self.enable_optimization:
            # Apply memory optimizations
            self.pipeline.enable_model_cpu_offload()

            # Fuse QKV projections for speed
            if hasattr(self.pipeline, 'transformer'):
                self.pipeline.transformer.fuse_qkv_projections()
            if hasattr(self.pipeline, 'vae'):
                self.pipeline.vae.fuse_qkv_projections()

            # Use channels-last memory format
            if hasattr(self.pipeline, 'transformer'):
                self.pipeline.transformer.to(memory_format=torch.channels_last)
            if hasattr(self.pipeline, 'vae'):
                self.pipeline.vae.to(memory_format=torch.channels_last)

        self.is_loaded = True
        print("Flux pipeline loaded successfully")

    def load_lora(self, lora_path: str, adapter_name: str = "default"):
        """Load LoRA weights."""
        if not self.is_loaded:
            self.load()

        print(f"Loading LoRA from: {lora_path}")

        # Unload previous LoRA if exists
        if self.current_lora:
            self.pipeline.unload_lora_weights()

        self.pipeline.load_lora_weights(lora_path, adapter_name=adapter_name)
        self.current_lora = lora_path
        print("LoRA loaded successfully")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        lora_scale: float = 0.8,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Generate images with Flux.

        NO SAFETY CHECKS. NO CONTENT FILTERING.

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt (not used in Flux but kept for API compatibility)
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            num_images: Number of images to generate
            lora_scale: LoRA strength (0.0 to 1.0)
            seed: Random seed for reproducibility

        Returns:
            List of PIL Images
        """
        if not self.is_loaded:
            self.load()

        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"Generating {num_images} image(s) with prompt: {prompt[:100]}...")

        with torch.inference_mode():
            result = self.pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
                generator=generator,
                joint_attention_kwargs={"scale": lora_scale} if self.current_lora else None,
            )

        print(f"Generated {len(result.images)} images successfully")
        return result.images

    def generate_batch(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[Image.Image]:
        """
        Generate images for multiple prompts.

        Args:
            prompts: List of text prompts
            **kwargs: Additional generation parameters

        Returns:
            List of PIL Images (one per prompt)
        """
        all_images = []

        for prompt in tqdm(prompts, desc="Generating images"):
            images = self.generate(prompt=prompt, num_images=1, **kwargs)
            all_images.extend(images)

        return all_images

    def save_images(
        self,
        images: List[Image.Image],
        output_dir: str,
        prefix: str = "flux",
        format: str = "png",
    ) -> List[str]:
        """
        Save generated images to disk.

        Args:
            images: List of PIL Images
            output_dir: Output directory
            prefix: Filename prefix
            format: Image format (png, jpg, etc.)

        Returns:
            List of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for i, image in enumerate(images):
            filename = f"{prefix}_{i:04d}.{format}"
            filepath = output_path / filename
            image.save(filepath, quality=95 if format == "jpg" else None)
            saved_paths.append(str(filepath))

        print(f"Saved {len(saved_paths)} images to {output_dir}")
        return saved_paths

    def unload(self):
        """Unload pipeline to free memory."""
        if self.is_loaded:
            del self.pipeline
            torch.cuda.empty_cache()
            self.is_loaded = False
            self.current_lora = None


def quick_generate(
    prompt: str,
    lora_path: Optional[str] = None,
    output_dir: str = "./output",
    num_images: int = 4,
    resolution: int = 1024,
    steps: int = 30,
    lora_strength: float = 0.8,
) -> List[str]:
    """
    Quick image generation with common defaults.

    Args:
        prompt: Text prompt
        lora_path: Path to LoRA weights (optional)
        output_dir: Output directory
        num_images: Number of images to generate
        resolution: Image resolution
        steps: Number of inference steps
        lora_strength: LoRA weight

    Returns:
        List of saved image paths
    """
    generator = FluxGenerator()
    generator.load()

    if lora_path:
        generator.load_lora(lora_path)

    images = generator.generate(
        prompt=prompt,
        width=resolution,
        height=resolution,
        num_inference_steps=steps,
        num_images=num_images,
        lora_scale=lora_strength,
    )

    return generator.save_images(images, output_dir)
