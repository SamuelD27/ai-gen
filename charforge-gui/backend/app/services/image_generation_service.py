"""
MASUKA Image Generation Service
Direct image generation using Diffusers library (Flux.1, SD3.5, SDXL)
"""

import torch
from diffusers import (
    FluxPipeline,
    StableDiffusion3Pipeline,
    DiffusionPipeline,
    AutoPipelineForText2Image
)
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
from PIL import Image
import json

logger = logging.getLogger(__name__)

class ImageGenerationService:
    """Service for generating images using modern diffusion models"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Cache for loaded pipelines (to avoid reloading)
        self._pipeline_cache: Dict[str, Any] = {}

        logger.info(f"ImageGenerationService initialized on {self.device}")

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models with metadata"""
        return [
            {
                "id": "flux-dev",
                "name": "Flux.1 Dev",
                "provider": "Black Forest Labs",
                "model_id": "black-forest-labs/FLUX.1-dev",
                "type": "flux",
                "description": "Best-in-class image generation with excellent prompt following",
                "recommended": True,
                "requires_token": True,
                "vram_required": "16GB+",
                "speed": "slow",
                "quality": "excellent"
            },
            {
                "id": "flux-schnell",
                "name": "Flux.1 Schnell",
                "provider": "Black Forest Labs",
                "model_id": "black-forest-labs/FLUX.1-schnell",
                "type": "flux",
                "description": "Fast version of Flux.1 for quick generations",
                "recommended": True,
                "requires_token": False,
                "vram_required": "12GB+",
                "speed": "fast",
                "quality": "very good"
            },
            {
                "id": "sd35-large",
                "name": "Stable Diffusion 3.5 Large",
                "provider": "Stability AI",
                "model_id": "stabilityai/stable-diffusion-3.5-large",
                "type": "sd3",
                "description": "Latest Stable Diffusion with improved quality",
                "recommended": True,
                "requires_token": True,
                "vram_required": "16GB+",
                "speed": "medium",
                "quality": "excellent"
            },
            {
                "id": "sd35-medium",
                "name": "Stable Diffusion 3.5 Medium",
                "provider": "Stability AI",
                "model_id": "stabilityai/stable-diffusion-3.5-medium",
                "type": "sd3",
                "description": "Balanced SD3.5 model for good quality and speed",
                "recommended": False,
                "requires_token": True,
                "vram_required": "12GB+",
                "speed": "fast",
                "quality": "very good"
            },
            {
                "id": "sdxl-base",
                "name": "Stable Diffusion XL",
                "provider": "Stability AI",
                "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "type": "sdxl",
                "description": "Previous generation high-quality model",
                "recommended": False,
                "requires_token": False,
                "vram_required": "10GB+",
                "speed": "medium",
                "quality": "very good"
            },
            {
                "id": "playground-v25",
                "name": "Playground v2.5",
                "provider": "Playground AI",
                "model_id": "playgroundai/playground-v2.5-1024px-aesthetic",
                "type": "sdxl",
                "description": "Highly aesthetic SDXL-based model",
                "recommended": True,
                "requires_token": False,
                "vram_required": "10GB+",
                "speed": "medium",
                "quality": "excellent"
            }
        ]

    def _load_pipeline(self, model_id: str, model_type: str) -> Any:
        """Load and cache a diffusion pipeline"""

        # Check cache first
        if model_id in self._pipeline_cache:
            logger.info(f"Using cached pipeline for {model_id}")
            return self._pipeline_cache[model_id]

        logger.info(f"Loading pipeline: {model_id} (type: {model_type})")

        try:
            # Load appropriate pipeline based on model type
            if model_type == "flux":
                pipeline = FluxPipeline.from_pretrained(
                    model_id,
                    torch_dtype=self.dtype,
                    use_safetensors=True
                )
            elif model_type == "sd3":
                pipeline = StableDiffusion3Pipeline.from_pretrained(
                    model_id,
                    torch_dtype=self.dtype,
                    use_safetensors=True
                )
            elif model_type == "sdxl":
                pipeline = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                    variant="fp16" if self.dtype == torch.float16 else None
                )
            else:
                # Generic pipeline loader
                pipeline = AutoPipelineForText2Image.from_pretrained(
                    model_id,
                    torch_dtype=self.dtype,
                    use_safetensors=True
                )

            # Move to device
            pipeline = pipeline.to(self.device)

            # Enable memory optimizations
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()

            if hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()

            # For CUDA, enable xformers if available
            if self.device == "cuda":
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("xFormers memory efficient attention enabled")
                except Exception as e:
                    logger.warning(f"Could not enable xFormers: {e}")

            # Cache the pipeline
            self._pipeline_cache[model_id] = pipeline

            logger.info(f"Pipeline loaded successfully: {model_id}")
            return pipeline

        except Exception as e:
            logger.error(f"Failed to load pipeline {model_id}: {e}")
            raise RuntimeError(f"Failed to load model {model_id}: {str(e)}")

    def generate(
        self,
        prompt: str,
        model_id: str,
        model_type: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        num_images: int = 1,
        seed: Optional[int] = None,
        lora_path: Optional[str] = None,
        lora_scale: float = 0.8,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate images using specified model

        Args:
            prompt: Text description of desired image
            model_id: HuggingFace model ID
            model_type: Type of model (flux, sd3, sdxl)
            negative_prompt: What to avoid in generation
            width: Image width (must be divisible by 8)
            height: Image height (must be divisible by 8)
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            num_images: Number of images to generate
            seed: Random seed for reproducibility
            lora_path: Optional path to LoRA weights
            lora_scale: LoRA influence strength

        Returns:
            List of dicts with 'image' (PIL Image) and 'metadata'
        """

        # Validate dimensions
        width = (width // 8) * 8
        height = (height // 8) * 8

        # Load pipeline
        pipeline = self._load_pipeline(model_id, model_type)

        # Load LoRA if specified
        if lora_path and Path(lora_path).exists():
            try:
                pipeline.load_lora_weights(lora_path)
                logger.info(f"Loaded LoRA from {lora_path}")
            except Exception as e:
                logger.warning(f"Failed to load LoRA: {e}")

        # Set seed for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Prepare generation parameters
        gen_params = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images,
            "generator": generator,
        }

        # Add negative prompt if supported
        if negative_prompt and hasattr(pipeline, "negative_prompt"):
            gen_params["negative_prompt"] = negative_prompt

        # Add any additional kwargs
        gen_params.update(kwargs)

        # Generate images
        logger.info(f"Generating {num_images} image(s) with {model_id}")
        start_time = datetime.now()

        try:
            result = pipeline(**gen_params)
            images = result.images
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Image generation failed: {str(e)}")

        generation_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Generation completed in {generation_time:.2f}s")

        # Prepare results with metadata
        results = []
        for idx, image in enumerate(images):
            results.append({
                "image": image,
                "metadata": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "model_id": model_id,
                    "model_type": model_type,
                    "width": width,
                    "height": height,
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed + idx if seed is not None else None,
                    "lora_path": lora_path,
                    "lora_scale": lora_scale if lora_path else None,
                    "generation_time": generation_time,
                    "timestamp": datetime.now().isoformat()
                }
            })

        return results

    def unload_pipeline(self, model_id: str):
        """Unload a pipeline from cache to free memory"""
        if model_id in self._pipeline_cache:
            del self._pipeline_cache[model_id]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded pipeline: {model_id}")

    def clear_cache(self):
        """Clear all cached pipelines"""
        self._pipeline_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleared all pipeline caches")

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        if torch.cuda.is_available():
            return {
                "device": "cuda",
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
        else:
            return {"device": "cpu", "message": "CUDA not available"}


# Global service instance
_image_service: Optional[ImageGenerationService] = None

def get_image_service() -> ImageGenerationService:
    """Get or create the global image generation service instance"""
    global _image_service
    if _image_service is None:
        _image_service = ImageGenerationService()
    return _image_service
