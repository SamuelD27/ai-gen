from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import os
from pathlib import Path

from app.core.database import get_db, User
from app.core.auth import get_current_active_user, get_current_user_optional
from app.core.config import settings

router = APIRouter()

# Pydantic models
class ModelInfo(BaseModel):
    name: str
    path: str
    size: Optional[int] = None
    type: str
    description: Optional[str] = None

class ModelListResponse(BaseModel):
    checkpoints: List[ModelInfo]
    vaes: List[ModelInfo]
    loras: List[ModelInfo]
    controlnets: List[ModelInfo]
    adapters: List[ModelInfo]

class TrainerInfo(BaseModel):
    name: str
    type: str
    description: str
    parameters: Dict[str, Any]

class TrainerListResponse(BaseModel):
    trainers: List[TrainerInfo]

@router.get("/models", response_model=ModelListResponse)
async def list_available_models(
    current_user: User = Depends(get_current_user_optional)
):
    """Get all available models from ComfyUI and HuggingFace."""

    # Default HuggingFace models (always available)
    default_checkpoints = [
        ModelInfo(
            name="Flux.1 Dev",
            path="black-forest-labs/FLUX.1-dev",
            type="checkpoint",
            description="Best quality from Black Forest Labs (requires HF token)"
        ),
        ModelInfo(
            name="Flux.1 Schnell",
            path="black-forest-labs/FLUX.1-schnell",
            type="checkpoint",
            description="Fast version of Flux.1"
        ),
        ModelInfo(
            name="SD 3.5 Large",
            path="stabilityai/stable-diffusion-3.5-large",
            type="checkpoint",
            description="Latest Stability AI flagship model"
        ),
        ModelInfo(
            name="SD 3.5 Medium",
            path="stabilityai/stable-diffusion-3.5-medium",
            type="checkpoint",
            description="Balanced SD 3.5 model"
        ),
        ModelInfo(
            name="SDXL Base",
            path="stabilityai/stable-diffusion-xl-base-1.0",
            type="checkpoint",
            description="Stable Diffusion XL base model"
        ),
        ModelInfo(
            name="Playground v2.5",
            path="playgroundai/playground-v2.5-1024px-aesthetic",
            type="checkpoint",
            description="Highly aesthetic SDXL-based model"
        )
    ]

    default_vaes = [
        ModelInfo(
            name="SDXL VAE FP16",
            path="madebyollin/sdxl-vae-fp16-fix",
            type="vae",
            description="Fixed SDXL VAE for FP16"
        ),
        ModelInfo(
            name="SD 1.5 VAE",
            path="stabilityai/sd-vae-ft-mse",
            type="vae",
            description="Standard SD 1.5 VAE"
        )
    ]

    # ComfyUI model paths - use settings (optional)
    comfyui_path = Path(settings.COMFYUI_PATH)
    print(f"🔍 Scanning for models in ComfyUI path: {comfyui_path}")
    print(f"   ComfyUI path exists: {comfyui_path.exists()}")

    def scan_model_directory(directory: Path, model_type: str) -> List[ModelInfo]:
        """Scan a directory for model files."""
        models = []
        if not directory.exists():
            print(f"   Directory does not exist: {directory}")
            return models
            
        for file_path in directory.rglob("*.safetensors"):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    models.append(ModelInfo(
                        name=file_path.stem,
                        path=str(file_path.relative_to(comfyui_path)),
                        size=stat.st_size,
                        type=model_type,
                        description=f"{model_type.title()} model"
                    ))
                except Exception:
                    continue
                    
        # Also scan for .ckpt files
        for file_path in directory.rglob("*.ckpt"):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    models.append(ModelInfo(
                        name=file_path.stem,
                        path=str(file_path.relative_to(comfyui_path)),
                        size=stat.st_size,
                        type=model_type,
                        description=f"{model_type.title()} model"
                    ))
                except Exception:
                    continue
                    
        return models
    
    # Scan different model directories (ComfyUI local files)
    comfyui_checkpoints = scan_model_directory(comfyui_path / "models" / "checkpoints", "checkpoint")
    comfyui_vaes = scan_model_directory(comfyui_path / "models" / "vae", "vae")
    loras = scan_model_directory(comfyui_path / "models" / "loras", "lora")
    controlnets = scan_model_directory(comfyui_path / "models" / "controlnet", "controlnet")

    print(f"   Found {len(comfyui_checkpoints)} local checkpoints, {len(comfyui_vaes)} local VAEs")

    # Combine ComfyUI models with HuggingFace defaults
    # Defaults are listed first so they appear at the top of dropdowns
    checkpoints = default_checkpoints + comfyui_checkpoints
    vaes = default_vaes + comfyui_vaes

    # Add MV adapters
    adapters = []
    mv_adapter_path = Path("./MV_Adapter")
    if mv_adapter_path.exists():
        adapters.append(ModelInfo(
            name="MV Adapter",
            path="huanngzh/mv-adapter",
            type="adapter",
            description="Multi-view adapter for generating multiple viewpoints"
        ))

    print(f"   Returning: {len(checkpoints)} total checkpoints, {len(vaes)} total VAEs, {len(loras)} LoRAs")

    return ModelListResponse(
        checkpoints=checkpoints,
        vaes=vaes,
        loras=loras,
        controlnets=controlnets,
        adapters=adapters
    )

@router.get("/trainers", response_model=TrainerListResponse)
async def list_available_trainers(
    current_user: User = Depends(get_current_user_optional)
):
    """Get all available training methods and their parameters."""
    
    trainers = [
        TrainerInfo(
            name="LoRA Training",
            type="lora",
            description="Low-Rank Adaptation training for efficient fine-tuning",
            parameters={
                "rank_dim": {"type": "int", "min": 4, "max": 128, "default": 8},
                "alpha": {"type": "int", "min": 4, "max": 128, "default": 8},
                "learning_rate": {"type": "float", "min": 1e-6, "max": 1e-2, "default": 8e-4},
                "optimizer": {"type": "select", "options": ["adamw", "adam", "sgd"], "default": "adamw"},
                "scheduler": {"type": "select", "options": ["constant", "cosine", "linear"], "default": "constant"}
            }
        ),
        TrainerInfo(
            name="Full Fine-tuning",
            type="full",
            description="Full model fine-tuning for maximum customization",
            parameters={
                "learning_rate": {"type": "float", "min": 1e-7, "max": 1e-3, "default": 1e-5},
                "weight_decay": {"type": "float", "min": 0, "max": 1e-1, "default": 1e-2},
                "gradient_checkpointing": {"type": "bool", "default": True},
                "train_text_encoder": {"type": "bool", "default": False}
            }
        ),
        TrainerInfo(
            name="Textual Inversion",
            type="textual_inversion",
            description="Learn new tokens for specific concepts",
            parameters={
                "tokens": {"type": "int", "min": 1, "max": 20, "default": 12},
                "init_words": {"type": "text", "default": "person"},
                "learning_rate": {"type": "float", "min": 1e-6, "max": 1e-3, "default": 5e-5}
            }
        ),
        TrainerInfo(
            name="MV Adapter Training",
            type="mv_adapter",
            description="Multi-view adapter training for 3D-aware generation",
            parameters={
                "num_views": {"type": "int", "min": 4, "max": 12, "default": 6},
                "guidance_scale": {"type": "float", "min": 1.0, "max": 10.0, "default": 3.0},
                "reference_conditioning_scale": {"type": "float", "min": 0.1, "max": 2.0, "default": 1.0},
                "azimuth_degrees": {"type": "list", "default": [0, 45, 90, 180, 270, 315]}
            }
        )
    ]
    
    return TrainerListResponse(trainers=trainers)

@router.get("/schedulers")
async def list_schedulers(
    current_user: User = Depends(get_current_user_optional)
):
    """Get available noise schedulers."""
    
    schedulers = [
        {"name": "DDPM", "value": "ddpm", "description": "Denoising Diffusion Probabilistic Models"},
        {"name": "DDIM", "value": "ddim", "description": "Denoising Diffusion Implicit Models"},
        {"name": "LMS", "value": "lms", "description": "Linear Multi-Step scheduler"},
        {"name": "Euler", "value": "euler", "description": "Euler scheduler"},
        {"name": "Euler Ancestral", "value": "euler_a", "description": "Euler Ancestral scheduler"},
        {"name": "DPM++ 2M", "value": "dpmpp_2m", "description": "DPM++ 2M scheduler"},
        {"name": "UniPC", "value": "uni_pc", "description": "UniPC scheduler"}
    ]
    
    return {"schedulers": schedulers}

@router.get("/optimizers")
async def list_optimizers(
    current_user: User = Depends(get_current_user_optional)
):
    """Get available optimizers."""
    
    optimizers = [
        {"name": "AdamW", "value": "adamw", "description": "Adam with weight decay"},
        {"name": "Adam", "value": "adam", "description": "Adaptive Moment Estimation"},
        {"name": "SGD", "value": "sgd", "description": "Stochastic Gradient Descent"},
        {"name": "RMSprop", "value": "rmsprop", "description": "Root Mean Square Propagation"},
        {"name": "Adagrad", "value": "adagrad", "description": "Adaptive Gradient Algorithm"}
    ]
    
    return {"optimizers": optimizers}

@router.post("/validate-model")
async def validate_model_path(
    model_path: str,
    current_user: User = Depends(get_current_user_optional)
):
    """Validate if a model path exists and is accessible."""
    
    # Check if it's a HuggingFace model path
    if "/" in model_path and not model_path.startswith("/"):
        return {"valid": True, "type": "huggingface", "message": "HuggingFace model path"}
    
    # Check if it's a local file path
    path = Path(model_path)
    if path.exists() and path.is_file():
        return {"valid": True, "type": "local", "message": "Local model file found"}
    
    # Check relative to ComfyUI
    comfyui_path = Path(settings.COMFYUI_PATH)
    full_path = comfyui_path / model_path
    if full_path.exists() and full_path.is_file():
        return {"valid": True, "type": "comfyui", "message": "ComfyUI model found"}

    return {"valid": False, "type": "unknown", "message": "Model not found"}
