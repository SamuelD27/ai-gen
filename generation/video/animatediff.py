"""
AnimateDiff video generation pipeline.
Supports prompt-to-video and LoRA-to-video with motion modules.
No content restrictions.
"""

import os
import torch
import imageio
from pathlib import Path
from typing import List, Optional, Union
from PIL import Image
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_video
from tqdm import tqdm


class AnimateDiffGenerator:
    """
    AnimateDiff video generation.
    NO SAFETY CHECKS. NO CONTENT FILTERING.
    """

    def __init__(
        self,
        base_model: str = "emilianJR/epiCRealism",
        motion_adapter: str = "guoyww/animatediff-motion-adapter-v1-5-2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16,
        cache_dir: Optional[str] = None,
    ):
        self.base_model = base_model
        self.motion_adapter_name = motion_adapter
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir or os.getenv("HF_HOME")

        self.pipeline = None
        self.motion_adapter = None
        self.is_loaded = False
        self.current_lora = None

    def load(self):
        """Load AnimateDiff pipeline."""
        if self.is_loaded:
            return

        print(f"Loading AnimateDiff pipeline...")
        print(f"  Base model: {self.base_model}")
        print(f"  Motion adapter: {self.motion_adapter_name}")

        # Load motion adapter
        self.motion_adapter = MotionAdapter.from_pretrained(
            self.motion_adapter_name,
            torch_dtype=self.dtype,
            cache_dir=self.cache_dir,
        )

        # Load pipeline
        self.pipeline = AnimateDiffPipeline.from_pretrained(
            self.base_model,
            motion_adapter=self.motion_adapter,
            torch_dtype=self.dtype,
            cache_dir=self.cache_dir,
        ).to(self.device)

        # Use DDIM scheduler for better quality
        self.pipeline.scheduler = DDIMScheduler.from_config(
            self.pipeline.scheduler.config,
            beta_schedule="linear",
            steps_offset=1,
        )

        # Enable memory optimizations
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_model_cpu_offload()

        self.is_loaded = True
        print("AnimateDiff pipeline loaded successfully")

    def load_lora(self, lora_path: str, adapter_name: str = "default"):
        """Load LoRA weights for video generation."""
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
        negative_prompt: str = "bad quality, worst quality, low resolution, blurry",
        width: int = 512,
        height: int = 512,
        num_frames: int = 16,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        lora_scale: float = 0.8,
        fps: int = 8,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Generate video frames with AnimateDiff.

        NO SAFETY CHECKS. NO CONTENT FILTERING.

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            width: Frame width
            height: Frame height
            num_frames: Number of frames to generate
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            lora_scale: LoRA strength (0.0 to 1.0)
            fps: Frames per second (for reference, not used in generation)
            seed: Random seed for reproducibility

        Returns:
            List of PIL Images (frames)
        """
        if not self.is_loaded:
            self.load()

        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"Generating {num_frames} frame video with prompt: {prompt[:100]}...")

        with torch.inference_mode():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                cross_attention_kwargs={"scale": lora_scale} if self.current_lora else None,
            )

        frames = result.frames[0]  # Get first (and only) video
        print(f"Generated {len(frames)} frames successfully")

        return frames

    def save_video(
        self,
        frames: List[Image.Image],
        output_path: str,
        fps: int = 8,
        format: str = "mp4",
        quality: int = 9,
    ) -> str:
        """
        Save frames as video file.

        Args:
            frames: List of PIL Images
            output_path: Output file path
            fps: Frames per second
            format: Video format (mp4, gif, webm)
            quality: Video quality (0-10, higher is better)

        Returns:
            Path to saved video file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure correct extension
        if not output_path.suffix:
            output_path = output_path.with_suffix(f".{format}")

        print(f"Saving video to {output_path} ({len(frames)} frames @ {fps} fps)...")

        if format == "gif":
            # Save as GIF
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000 // fps,
                loop=0,
            )
        else:
            # Save as video using export_to_video
            export_to_video(frames, str(output_path), fps=fps)

        print(f"Video saved successfully")
        return str(output_path)

    def generate_and_save(
        self,
        prompt: str,
        output_path: str,
        **kwargs,
    ) -> str:
        """
        Generate video and save in one step.

        Args:
            prompt: Text prompt
            output_path: Output file path
            **kwargs: Additional generation parameters

        Returns:
            Path to saved video file
        """
        # Extract fps from kwargs or use default
        fps = kwargs.pop("fps", 8)

        # Generate frames
        frames = self.generate(prompt=prompt, fps=fps, **kwargs)

        # Save video
        return self.save_video(frames, output_path, fps=fps)

    def generate_batch(
        self,
        prompts: List[str],
        output_dir: str,
        prefix: str = "animatediff",
        **kwargs,
    ) -> List[str]:
        """
        Generate videos for multiple prompts.

        Args:
            prompts: List of text prompts
            output_dir: Output directory
            prefix: Filename prefix
            **kwargs: Additional generation parameters

        Returns:
            List of saved video paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for i, prompt in enumerate(tqdm(prompts, desc="Generating videos")):
            output_file = output_path / f"{prefix}_{i:04d}.mp4"
            video_path = self.generate_and_save(
                prompt=prompt,
                output_path=str(output_file),
                **kwargs,
            )
            saved_paths.append(video_path)

        return saved_paths

    def interpolate_frames(
        self,
        frames: List[Image.Image],
        target_fps: int = 24,
        method: str = "linear",
    ) -> List[Image.Image]:
        """
        Interpolate frames to increase FPS.

        Args:
            frames: Input frames
            target_fps: Target frames per second
            method: Interpolation method (linear, cubic, rife)

        Returns:
            Interpolated frames
        """
        # TODO: Implement RIFE or other advanced interpolation
        # For now, use simple frame duplication
        current_fps = 8  # Default AnimateDiff FPS
        multiplier = target_fps // current_fps

        if multiplier <= 1:
            return frames

        interpolated = []
        for frame in frames:
            for _ in range(multiplier):
                interpolated.append(frame.copy())

        return interpolated

    def unload(self):
        """Unload pipeline to free memory."""
        if self.is_loaded:
            del self.pipeline
            del self.motion_adapter
            torch.cuda.empty_cache()
            self.is_loaded = False
            self.current_lora = None


def quick_video(
    prompt: str,
    lora_path: Optional[str] = None,
    output_path: str = "./output/video.mp4",
    num_frames: int = 16,
    resolution: int = 512,
    fps: int = 8,
    lora_strength: float = 0.8,
) -> str:
    """
    Quick video generation with common defaults.

    Args:
        prompt: Text prompt
        lora_path: Path to LoRA weights (optional)
        output_path: Output video path
        num_frames: Number of frames
        resolution: Frame resolution
        fps: Frames per second
        lora_strength: LoRA weight

    Returns:
        Path to saved video
    """
    generator = AnimateDiffGenerator()
    generator.load()

    if lora_path:
        generator.load_lora(lora_path)

    return generator.generate_and_save(
        prompt=prompt,
        output_path=output_path,
        width=resolution,
        height=resolution,
        num_frames=num_frames,
        fps=fps,
        lora_scale=lora_strength,
    )
