"""
Core configuration system for CharForgeX.
Supports local and cloud compute modes with flexible model/dataset management.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum


class ComputeMode(str, Enum):
    """Compute execution mode."""
    LOCAL = "local"
    CLOUD = "cloud"
    HYBRID = "hybrid"


class ModelBackend(str, Enum):
    """Supported model backends."""
    FLUX = "flux"
    FLUX_DEV = "flux-dev"
    FLUX_SCHNELL = "flux-schnell"
    SDXL = "sdxl"
    SD3 = "sd3"
    SD15 = "sd15"


class VideoBackend(str, Enum):
    """Supported video generation backends."""
    ANIMATEDIFF = "animatediff"
    HOTSHOTXL = "hotshotxl"
    SVD = "svd"  # Stable Video Diffusion


class CaptioningBackend(str, Enum):
    """Supported captioning backends."""
    BLIP2 = "blip2"
    CLIP_INTERROGATOR = "clip_interrogator"
    COGVLM = "cogvlm"
    LLAVA = "llava"
    ENSEMBLE = "ensemble"  # Use multiple backends


@dataclass
class ComputeConfig:
    """Compute configuration."""
    mode: ComputeMode = ComputeMode.LOCAL
    device: str = "cuda"
    precision: Literal["fp32", "fp16", "bf16"] = "bf16"
    use_xformers: bool = True
    use_torch_compile: bool = False
    max_memory_gb: Optional[float] = None


@dataclass
class CloudConfig:
    """Cloud compute configuration (RunPod)."""
    enabled: bool = False
    provider: str = "runpod"
    api_key: Optional[str] = None
    gpu_type: str = "RTX4090"
    region: str = "US-OR"
    max_cost_per_hour: float = 1.0
    auto_shutdown_minutes: int = 30
    storage_gb: int = 100
    template_id: Optional[str] = None


@dataclass
class DatasetConfig:
    """Dataset processing configuration."""
    # Cleaning
    enable_face_detection: bool = True
    enable_deduplication: bool = True
    enable_quality_filter: bool = True
    min_resolution: int = 512
    max_aspect_ratio: float = 2.0

    # Preprocessing
    target_resolution: int = 1024
    center_crop: bool = False
    smart_crop: bool = True  # Crop around detected faces

    # Captioning
    captioning_backend: CaptioningBackend = CaptioningBackend.BLIP2
    caption_prefix: str = ""
    caption_suffix: str = ""
    max_caption_length: int = 200

    # Augmentation
    enable_augmentation: bool = False
    flip_horizontal: bool = True
    color_jitter: bool = False


@dataclass
class TrainingConfig:
    """LoRA training configuration."""
    # Model
    model_backend: ModelBackend = ModelBackend.FLUX_DEV
    base_model_path: Optional[str] = None

    # LoRA settings
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Training hyperparameters
    steps: int = 1000
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    lr_scheduler: str = "constant"
    warmup_steps: int = 0

    # Optimization
    optimizer: str = "adamw8bit"
    weight_decay: float = 0.01
    gradient_checkpointing: bool = True
    mixed_precision: bool = True

    # Regularization
    noise_offset: float = 0.0
    min_snr_gamma: Optional[float] = None

    # Saving
    save_every: int = 250
    max_checkpoints: int = 5
    output_dir: str = "./output/loras"

    # Advanced
    train_text_encoder: bool = False
    cache_latents: bool = True
    resolution: int = 512


@dataclass
class ImageGenerationConfig:
    """Image generation configuration."""
    model_backend: ModelBackend = ModelBackend.FLUX_DEV

    # Generation parameters
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5

    # LoRA
    lora_path: Optional[str] = None
    lora_weight: float = 0.8

    # Advanced features
    use_controlnet: bool = False
    controlnet_type: Optional[str] = None
    controlnet_weight: float = 1.0

    use_ipadapter: bool = False
    ipadapter_weight: float = 0.8

    use_regional_prompting: bool = False

    # Post-processing
    upscale: bool = False
    upscale_factor: float = 2.0
    face_restoration: bool = False

    # Batch generation
    batch_size: int = 1
    output_dir: str = "./output/images"


@dataclass
class VideoGenerationConfig:
    """Video generation configuration."""
    backend: VideoBackend = VideoBackend.ANIMATEDIFF

    # Video parameters
    width: int = 512
    height: int = 512
    num_frames: int = 16
    fps: int = 8
    num_inference_steps: int = 25
    guidance_scale: float = 7.5

    # LoRA
    lora_path: Optional[str] = None
    lora_weight: float = 0.8

    # Motion control
    use_facial_motion: bool = False
    facial_motion_source: Optional[str] = None  # Path to driving video

    use_audio_sync: bool = False
    audio_path: Optional[str] = None

    # Temporal smoothing
    use_interpolation: bool = True
    interpolation_method: Literal["rife", "amt", "raft"] = "rife"
    target_fps: int = 24

    # Output
    output_dir: str = "./output/videos"
    output_format: Literal["mp4", "gif", "webm"] = "mp4"


@dataclass
class CharForgeXConfig:
    """Main CharForgeX configuration."""
    # Core settings
    project_name: str = "charforgex"
    work_dir: str = "./workspace"
    cache_dir: str = "./cache"

    # No content restrictions
    enable_safety_checker: bool = False  # Disabled by default
    enable_watermark: bool = False
    enable_telemetry: bool = False

    # Sub-configurations
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    image_generation: ImageGenerationConfig = field(default_factory=ImageGenerationConfig)
    video_generation: VideoGenerationConfig = field(default_factory=VideoGenerationConfig)

    # Environment variables
    hf_token: Optional[str] = None
    civitai_api_key: Optional[str] = None

    def __post_init__(self):
        """Load environment variables."""
        self.hf_token = os.getenv("HF_TOKEN")
        self.civitai_api_key = os.getenv("CIVITAI_API_KEY")

        # Create directories
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str) -> "CharForgeXConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse nested configurations
        config_dict = {}

        if "compute" in data:
            config_dict["compute"] = ComputeConfig(**data["compute"])
        if "cloud" in data:
            config_dict["cloud"] = CloudConfig(**data["cloud"])
        if "dataset" in data:
            config_dict["dataset"] = DatasetConfig(**data["dataset"])
        if "training" in data:
            config_dict["training"] = TrainingConfig(**data["training"])
        if "image_generation" in data:
            config_dict["image_generation"] = ImageGenerationConfig(**data["image_generation"])
        if "video_generation" in data:
            config_dict["video_generation"] = VideoGenerationConfig(**data["video_generation"])

        # Top-level settings
        for key in ["project_name", "work_dir", "cache_dir", "enable_safety_checker",
                    "enable_watermark", "enable_telemetry"]:
            if key in data:
                config_dict[key] = data[key]

        return cls(**config_dict)

    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        data = {}

        # Top-level settings
        for key in ["project_name", "work_dir", "cache_dir", "enable_safety_checker",
                    "enable_watermark", "enable_telemetry"]:
            data[key] = getattr(self, key)

        # Nested configurations
        data["compute"] = asdict(self.compute)
        data["cloud"] = asdict(self.cloud)
        data["dataset"] = asdict(self.dataset)
        data["training"] = asdict(self.training)
        data["image_generation"] = asdict(self.image_generation)
        data["video_generation"] = asdict(self.video_generation)

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)

    def get_model_path(self, backend: ModelBackend) -> str:
        """Get the model path for a given backend."""
        model_paths = {
            ModelBackend.FLUX_DEV: "black-forest-labs/FLUX.1-dev",
            ModelBackend.FLUX_SCHNELL: "black-forest-labs/FLUX.1-schnell",
            ModelBackend.SDXL: "stabilityai/stable-diffusion-xl-base-1.0",
            ModelBackend.SD3: "stabilityai/stable-diffusion-3-medium",
            ModelBackend.SD15: "runwayml/stable-diffusion-v1-5",
        }
        return model_paths.get(backend, "")

    def is_cloud_mode(self) -> bool:
        """Check if running in cloud mode."""
        return self.compute.mode in [ComputeMode.CLOUD, ComputeMode.HYBRID] and self.cloud.enabled


# Global configuration instance
_global_config: Optional[CharForgeXConfig] = None


def get_config() -> CharForgeXConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = CharForgeXConfig()
    return _global_config


def set_config(config: CharForgeXConfig):
    """Set the global configuration instance."""
    global _global_config
    _global_config = config


def load_config(path: str = "config.yaml") -> CharForgeXConfig:
    """Load and set global configuration from file."""
    if os.path.exists(path):
        config = CharForgeXConfig.from_yaml(path)
    else:
        config = CharForgeXConfig()
        config.to_yaml(path)  # Create default config

    set_config(config)
    return config
