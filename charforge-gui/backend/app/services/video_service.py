"""
Video Generation Service for MASUKA
Supports multiple API providers: Sora 2, VEO3, Runway Gen-4, etc.
"""

from typing import Optional, Dict, Any, List
from enum import Enum
from pathlib import Path
import httpx
import asyncio
from datetime import datetime

from app.core.config import settings


class VideoProvider(str, Enum):
    """Supported video generation providers"""
    SORA_2 = "sora-2"
    SORA_2_HD = "sora-2-hd"
    VEO3_PRO = "veo3-pro"
    VEO3_FAST = "veo3-fast"
    RUNWAY_GEN4 = "runway-gen4"
    HAIPER = "haiper"
    KLING = "kling"


class VideoResolution(str, Enum):
    """Supported video resolutions"""
    SD = "480p"
    HD = "720p"
    FULL_HD = "1080p"
    ULTRA_HD = "2160p"


class VideoGenerationRequest:
    """Video generation request parameters"""
    def __init__(
        self,
        prompt: str,
        provider: VideoProvider = VideoProvider.VEO3_FAST,
        duration: int = 5,
        resolution: VideoResolution = VideoResolution.FULL_HD,
        motion_intensity: float = 0.7,
        aspect_ratio: str = "16:9",
        fps: int = 24,
        seed: Optional[int] = None,
        lora_path: Optional[str] = None,
        lora_scale: float = 0.8,
        character_id: Optional[int] = None,
    ):
        self.prompt = prompt
        self.provider = provider
        self.duration = max(3, min(duration, 60))  # 3-60 seconds
        self.resolution = resolution
        self.motion_intensity = max(0.0, min(motion_intensity, 1.0))
        self.aspect_ratio = aspect_ratio
        self.fps = fps
        self.seed = seed
        self.lora_path = lora_path
        self.lora_scale = max(0.0, min(lora_scale, 2.0))
        self.character_id = character_id


class VideoGenerationResult:
    """Video generation result"""
    def __init__(
        self,
        video_url: str,
        thumbnail_url: Optional[str] = None,
        duration: float = 0,
        resolution: str = "",
        file_size: int = 0,
        provider: str = "",
        generation_time: float = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.video_url = video_url
        self.thumbnail_url = thumbnail_url
        self.duration = duration
        self.resolution = resolution
        self.file_size = file_size
        self.provider = provider
        self.generation_time = generation_time
        self.metadata = metadata or {}


class VideoGenerationService:
    """Main video generation service"""

    def __init__(self):
        self.comet_api_key = getattr(settings, 'COMET_API_KEY', None)
        self.openai_api_key = getattr(settings, 'OPENAI_API_KEY', None)
        self.base_url = "https://api.comet.ai/v1"  # CometAPI unified endpoint

    async def generate(
        self,
        request: VideoGenerationRequest,
        user_id: int
    ) -> VideoGenerationResult:
        """
        Generate video using specified provider

        Args:
            request: Video generation request parameters
            user_id: User ID for tracking and storage

        Returns:
            VideoGenerationResult with video URL and metadata
        """
        start_time = datetime.now()

        # Route to appropriate provider
        if request.provider in [VideoProvider.SORA_2, VideoProvider.SORA_2_HD]:
            result = await self._generate_sora(request)
        elif request.provider in [VideoProvider.VEO3_PRO, VideoProvider.VEO3_FAST]:
            result = await self._generate_veo3(request)
        elif request.provider == VideoProvider.RUNWAY_GEN4:
            result = await self._generate_runway(request)
        elif request.provider == VideoProvider.HAIPER:
            result = await self._generate_haiper(request)
        elif request.provider == VideoProvider.KLING:
            result = await self._generate_kling(request)
        else:
            raise ValueError(f"Unsupported provider: {request.provider}")

        # Calculate generation time
        generation_time = (datetime.now() - start_time).total_seconds()
        result.generation_time = generation_time

        return result

    async def _generate_sora(
        self,
        request: VideoGenerationRequest
    ) -> VideoGenerationResult:
        """Generate video using Sora 2 via CometAPI"""
        if not self.comet_api_key:
            raise ValueError("COMET_API_KEY not configured")

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{self.base_url}/video/generate",
                headers={"Authorization": f"Bearer {self.comet_api_key}"},
                json={
                    "model": request.provider.value,
                    "prompt": request.prompt,
                    "duration": request.duration,
                    "resolution": request.resolution.value,
                    "aspect_ratio": request.aspect_ratio,
                    "fps": request.fps,
                    "seed": request.seed,
                }
            )
            response.raise_for_status()
            data = response.json()

        return VideoGenerationResult(
            video_url=data.get("video_url"),
            thumbnail_url=data.get("thumbnail_url"),
            duration=data.get("duration", request.duration),
            resolution=request.resolution.value,
            file_size=data.get("file_size", 0),
            provider="Sora 2",
            metadata=data.get("metadata", {})
        )

    async def _generate_veo3(
        self,
        request: VideoGenerationRequest
    ) -> VideoGenerationResult:
        """Generate video using Google VEO3 via CometAPI"""
        if not self.comet_api_key:
            raise ValueError("COMET_API_KEY not configured")

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{self.base_url}/video/generate",
                headers={"Authorization": f"Bearer {self.comet_api_key}"},
                json={
                    "model": request.provider.value,
                    "prompt": request.prompt,
                    "duration": request.duration,
                    "resolution": request.resolution.value,
                    "motion_scale": request.motion_intensity,
                    "aspect_ratio": request.aspect_ratio,
                    "seed": request.seed,
                }
            )
            response.raise_for_status()
            data = response.json()

        return VideoGenerationResult(
            video_url=data.get("video_url"),
            thumbnail_url=data.get("thumbnail_url"),
            duration=data.get("duration", request.duration),
            resolution=request.resolution.value,
            file_size=data.get("file_size", 0),
            provider="Google VEO3",
            metadata=data.get("metadata", {})
        )

    async def _generate_runway(
        self,
        request: VideoGenerationRequest
    ) -> VideoGenerationResult:
        """Generate video using Runway Gen-4 via CometAPI"""
        if not self.comet_api_key:
            raise ValueError("COMET_API_KEY not configured")

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{self.base_url}/video/generate",
                headers={"Authorization": f"Bearer {self.comet_api_key}"},
                json={
                    "model": "runway-gen4",
                    "prompt": request.prompt,
                    "duration": request.duration,
                    "resolution": request.resolution.value,
                    "seed": request.seed,
                }
            )
            response.raise_for_status()
            data = response.json()

        return VideoGenerationResult(
            video_url=data.get("video_url"),
            thumbnail_url=data.get("thumbnail_url"),
            duration=data.get("duration", request.duration),
            resolution=request.resolution.value,
            file_size=data.get("file_size", 0),
            provider="Runway Gen-4",
            metadata=data.get("metadata", {})
        )

    async def _generate_haiper(
        self,
        request: VideoGenerationRequest
    ) -> VideoGenerationResult:
        """Generate video using Haiper AI"""
        # Placeholder for Haiper implementation
        raise NotImplementedError("Haiper integration coming soon")

    async def _generate_kling(
        self,
        request: VideoGenerationRequest
    ) -> VideoGenerationResult:
        """Generate video using Kling AI"""
        # Placeholder for Kling implementation
        raise NotImplementedError("Kling integration coming soon")

    async def get_available_providers(self) -> List[Dict[str, Any]]:
        """Get list of available video providers with their capabilities"""
        providers = [
            {
                "id": "sora-2",
                "name": "Sora 2",
                "description": "OpenAI's latest video generation model",
                "max_duration": 60,
                "resolutions": ["720p", "1080p", "2160p"],
                "available": bool(self.comet_api_key or self.openai_api_key),
                "speed": "medium",
                "quality": "excellent"
            },
            {
                "id": "sora-2-hd",
                "name": "Sora 2 HD",
                "description": "Higher quality Sora 2 variant",
                "max_duration": 60,
                "resolutions": ["1080p", "2160p"],
                "available": bool(self.comet_api_key or self.openai_api_key),
                "speed": "slow",
                "quality": "best"
            },
            {
                "id": "veo3-pro",
                "name": "Google VEO3 Pro",
                "description": "Google's professional video generation",
                "max_duration": 30,
                "resolutions": ["720p", "1080p", "2160p"],
                "available": bool(self.comet_api_key),
                "speed": "medium",
                "quality": "excellent"
            },
            {
                "id": "veo3-fast",
                "name": "Google VEO3 Fast",
                "description": "Faster VEO3 variant",
                "max_duration": 20,
                "resolutions": ["720p", "1080p"],
                "available": bool(self.comet_api_key),
                "speed": "fast",
                "quality": "good"
            },
            {
                "id": "runway-gen4",
                "name": "Runway Gen-4",
                "description": "Latest Runway generation model",
                "max_duration": 15,
                "resolutions": ["720p", "1080p"],
                "available": bool(self.comet_api_key),
                "speed": "fast",
                "quality": "very good"
            }
        ]
        return providers


# Singleton instance
video_service = VideoGenerationService()
