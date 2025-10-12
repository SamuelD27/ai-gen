"""
Video Generation API Endpoints for MASUKA
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
import logging

from app.core.database import get_db, User
from app.core.auth import get_current_user_optional
from app.services.video_service import (
    video_service,
    VideoProvider,
    VideoResolution,
    VideoGenerationRequest as ServiceVideoRequest
)

# Configure logger
logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models for API
class VideoGenerationRequest(BaseModel):
    """Request model for video generation"""
    prompt: str = Field(..., min_length=10, max_length=1000, description="Text description of the video to generate")
    provider: str = Field(default="veo3-fast", description="Video generation provider")
    duration: int = Field(default=5, ge=3, le=60, description="Video duration in seconds")
    resolution: str = Field(default="1080p", description="Video resolution")
    motion_intensity: float = Field(default=0.7, ge=0.0, le=1.0, description="Motion intensity (0-1)")
    aspect_ratio: str = Field(default="16:9", description="Aspect ratio")
    fps: int = Field(default=24, ge=24, le=60, description="Frames per second")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class VideoGenerationResponse(BaseModel):
    """Response model for video generation"""
    video_url: str
    thumbnail_url: Optional[str] = None
    duration: float
    resolution: str
    file_size: int
    provider: str
    generation_time: float
    metadata: dict = {}


class VideoProviderInfo(BaseModel):
    """Information about a video provider"""
    id: str
    name: str
    description: str
    max_duration: int
    resolutions: List[str]
    available: bool
    speed: str
    quality: str


@router.post("/generate", response_model=VideoGenerationResponse, status_code=status.HTTP_201_CREATED)
async def generate_video(
    request: VideoGenerationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    """
    Generate a video from text prompt

    This endpoint creates a video using the specified AI provider.
    Generation may take several minutes depending on duration and quality.

    **Supported Providers:**
    - sora-2: OpenAI Sora 2 (best quality, slower)
    - sora-2-hd: Sora 2 HD variant
    - veo3-pro: Google VEO3 Professional
    - veo3-fast: Google VEO3 Fast (recommended)
    - runway-gen4: Runway Gen-4

    **Example:**
    ```json
    {
        "prompt": "A serene lake at sunset with mountains in the background",
        "provider": "veo3-fast",
        "duration": 5,
        "resolution": "1080p"
    }
    ```
    """
    logger.info(f"=== VIDEO GENERATION REQUEST ===")
    logger.info(f"User ID: {current_user.id}")
    logger.info(f"Prompt: {request.prompt}")
    logger.info(f"Provider: {request.provider}")
    logger.info(f"Duration: {request.duration}s")
    logger.info(f"Resolution: {request.resolution}")

    try:
        # Validate provider
        try:
            provider_enum = VideoProvider(request.provider)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid provider: {request.provider}. Use /video/providers to see available options."
            )

        # Validate resolution
        try:
            resolution_enum = VideoResolution(request.resolution)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid resolution: {request.resolution}. Supported: 480p, 720p, 1080p, 2160p"
            )

        # Create service request
        service_request = ServiceVideoRequest(
            prompt=request.prompt,
            provider=provider_enum,
            duration=request.duration,
            resolution=resolution_enum,
            motion_intensity=request.motion_intensity,
            aspect_ratio=request.aspect_ratio,
            fps=request.fps,
            seed=request.seed
        )

        # Generate video
        logger.info("Starting video generation...")
        result = await video_service.generate(service_request, current_user.id)
        logger.info(f"Video generated successfully in {result.generation_time:.2f}s")

        return VideoGenerationResponse(
            video_url=result.video_url,
            thumbnail_url=result.thumbnail_url,
            duration=result.duration,
            resolution=result.resolution,
            file_size=result.file_size,
            provider=result.provider,
            generation_time=result.generation_time,
            metadata=result.metadata
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Video generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video generation failed: {str(e)}"
        )


@router.get("/providers", response_model=List[VideoProviderInfo])
async def get_video_providers(
    current_user: User = Depends(get_current_user_optional)
):
    """
    Get list of available video generation providers

    Returns information about each provider including:
    - Availability (based on API key configuration)
    - Capabilities (max duration, resolutions)
    - Performance characteristics (speed, quality)
    """
    logger.info("Fetching available video providers")

    try:
        providers = await video_service.get_available_providers()
        return providers
    except Exception as e:
        logger.error(f"Failed to fetch providers: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch providers: {str(e)}"
        )


@router.get("/health")
async def video_health_check():
    """
    Health check for video generation service

    Returns service status and configuration info
    """
    has_comet = bool(video_service.comet_api_key)
    has_openai = bool(video_service.openai_api_key)

    return {
        "status": "healthy",
        "service": "video_generation",
        "providers_configured": has_comet or has_openai,
        "comet_api": "configured" if has_comet else "not_configured",
        "openai_api": "configured" if has_openai else "not_configured"
    }
