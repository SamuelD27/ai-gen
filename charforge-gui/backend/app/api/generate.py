"""
Image Generation API Endpoints
Direct text-to-image generation using modern diffusion models
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import uuid
import logging

from app.core.database import get_db, User
from app.core.auth import get_current_user_optional
from app.core.config import settings
from app.services.image_generation_service import get_image_service

router = APIRouter()
logger = logging.getLogger(__name__)

# Pydantic Models

class ImageGenerationRequest(BaseModel):
    """Request model for image generation"""
    prompt: str = Field(..., min_length=3, max_length=2000, description="Text description of the image")
    negative_prompt: Optional[str] = Field(None, max_length=1000, description="What to avoid in the image")
    model_id: str = Field(default="flux-schnell", description="Model to use for generation")
    width: int = Field(default=1024, ge=256, le=2048, description="Image width (divisible by 8)")
    height: int = Field(default=1024, ge=256, le=2048, description="Image height (divisible by 8)")
    num_inference_steps: int = Field(default=30, ge=1, le=100, description="Number of denoising steps")
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0, description="Prompt adherence strength")
    num_images: int = Field(default=1, ge=1, le=4, description="Number of images to generate")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    lora_path: Optional[str] = Field(None, description="Path to LoRA weights (optional)")
    lora_scale: float = Field(default=0.8, ge=0.0, le=2.0, description="LoRA influence strength")

class ImageMetadata(BaseModel):
    """Metadata for a generated image"""
    prompt: str
    negative_prompt: Optional[str]
    model_id: str
    model_type: str
    width: int
    height: int
    steps: int
    guidance_scale: float
    seed: Optional[int]
    lora_path: Optional[str]
    lora_scale: Optional[float]
    generation_time: float
    timestamp: str

class GeneratedImage(BaseModel):
    """A single generated image with metadata"""
    url: str
    filename: str
    metadata: ImageMetadata

class ImageGenerationResponse(BaseModel):
    """Response model for image generation"""
    job_id: str
    status: str
    images: List[GeneratedImage]
    total_generation_time: float
    model_info: Dict[str, Any]

class ModelInfo(BaseModel):
    """Information about an available model"""
    id: str
    name: str
    provider: str
    model_id: str
    type: str
    description: str
    recommended: bool
    requires_token: bool
    vram_required: str
    speed: str
    quality: str

class ModelListResponse(BaseModel):
    """List of available models"""
    models: List[ModelInfo]

class GenerationPreset(BaseModel):
    """Pre-configured generation preset"""
    id: str
    name: str
    description: str
    model_id: str
    width: int
    height: int
    steps: int
    guidance_scale: float
    example_prompt: str


# API Endpoints

@router.get("/models", response_model=ModelListResponse)
async def list_available_models(
    current_user: User = Depends(get_current_user_optional)
):
    """Get list of available image generation models"""
    service = get_image_service()
    models = service.get_available_models()

    return ModelListResponse(
        models=[ModelInfo(**model) for model in models]
    )

@router.post("/generate", response_model=ImageGenerationResponse, status_code=status.HTTP_201_CREATED)
async def generate_images(
    request: ImageGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Generate images from text prompt using specified model

    This endpoint generates images synchronously (blocking).
    For long-running jobs, consider using the /generate-async endpoint.
    """
    logger.info(f"=== IMAGE GENERATION REQUEST ===")
    logger.info(f"User ID: {current_user.id}")
    logger.info(f"Model: {request.model_id}")
    logger.info(f"Prompt: {request.prompt[:100]}...")

    # Get service
    service = get_image_service()

    # Validate model exists
    available_models = service.get_available_models()
    model_info = next((m for m in available_models if m["id"] == request.model_id), None)

    if not model_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{request.model_id}' not found"
        )

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Create output directory for this job
    output_dir = settings.RESULTS_DIR / "generated" / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()

    try:
        # Generate images
        results = service.generate(
            prompt=request.prompt,
            model_id=model_info["model_id"],
            model_type=model_info["type"],
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            num_images=request.num_images,
            seed=request.seed,
            lora_path=request.lora_path,
            lora_scale=request.lora_scale
        )

        # Save images and prepare response
        generated_images = []

        for idx, result in enumerate(results):
            image = result["image"]
            metadata = result["metadata"]

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{request.model_id}_{timestamp}_{idx}.png"
            filepath = output_dir / filename

            # Save image
            image.save(filepath, format="PNG", optimize=True)

            # Create URL (relative to results mount)
            image_url = f"/results/generated/{job_id}/{filename}"

            generated_images.append(GeneratedImage(
                url=image_url,
                filename=filename,
                metadata=ImageMetadata(**metadata)
            ))

        total_time = (datetime.now() - start_time).total_seconds()

        logger.info(f"Generated {len(generated_images)} images in {total_time:.2f}s")

        return ImageGenerationResponse(
            job_id=job_id,
            status="completed",
            images=generated_images,
            total_generation_time=total_time,
            model_info=model_info
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image generation failed: {str(e)}"
        )

@router.get("/presets", response_model=List[GenerationPreset])
async def list_generation_presets(
    current_user: User = Depends(get_current_user_optional)
):
    """Get pre-configured generation presets"""
    presets = [
        GenerationPreset(
            id="portrait-fast",
            name="Fast Portrait",
            description="Quick portrait generation with Flux Schnell",
            model_id="flux-schnell",
            width=768,
            height=1024,
            steps=20,
            guidance_scale=3.5,
            example_prompt="portrait of a person, detailed face, professional lighting, 8k uhd"
        ),
        GenerationPreset(
            id="portrait-hq",
            name="High Quality Portrait",
            description="Best quality portrait with Flux Dev",
            model_id="flux-dev",
            width=768,
            height=1024,
            steps=40,
            guidance_scale=3.5,
            example_prompt="professional portrait photography, detailed face, soft studio lighting, bokeh background"
        ),
        GenerationPreset(
            id="landscape",
            name="Landscape",
            description="Wide landscape scenes",
            model_id="flux-schnell",
            width=1344,
            height=768,
            steps=25,
            guidance_scale=3.5,
            example_prompt="beautiful landscape, mountains, dramatic sky, golden hour, highly detailed"
        ),
        GenerationPreset(
            id="square-hq",
            name="Square High Quality",
            description="Square format for social media",
            model_id="playground-v25",
            width=1024,
            height=1024,
            steps=30,
            guidance_scale=7.0,
            example_prompt="highly detailed, professional quality, 8k resolution"
        ),
        GenerationPreset(
            id="character-design",
            name="Character Design",
            description="Character concept art and design",
            model_id="sd35-large",
            width=768,
            height=1024,
            steps=35,
            guidance_scale=7.5,
            example_prompt="character concept art, full body, white background, detailed design"
        ),
        GenerationPreset(
            id="cinematic",
            name="Cinematic",
            description="Movie-like wide shots",
            model_id="flux-dev",
            width=1536,
            height=640,
            steps=40,
            guidance_scale=3.5,
            example_prompt="cinematic shot, dramatic lighting, film grain, anamorphic lens"
        )
    ]

    return presets

@router.get("/memory")
async def get_memory_usage(
    current_user: User = Depends(get_current_user_optional)
):
    """Get current GPU/CPU memory usage"""
    service = get_image_service()
    return service.get_memory_usage()

@router.post("/clear-cache")
async def clear_model_cache(
    current_user: User = Depends(get_current_user_optional)
):
    """Clear all cached models to free memory"""
    service = get_image_service()
    service.clear_cache()
    return {"message": "Model cache cleared successfully"}

@router.get("/health")
async def generation_health_check():
    """Check if generation service is healthy"""
    try:
        service = get_image_service()
        memory = service.get_memory_usage()
        models_count = len(service.get_available_models())

        return {
            "status": "healthy",
            "device": memory.get("device", "unknown"),
            "available_models": models_count,
            "memory": memory
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Generation service unhealthy: {str(e)}"
        )
