"""Video generation backends."""

from .animatediff import AnimateDiffGenerator, quick_video

__all__ = [
    "AnimateDiffGenerator",
    "quick_video",
]
