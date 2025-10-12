from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import json
from datetime import datetime
from pathlib import Path

from app.core.database import get_db, Character, TrainingSession, User
from app.core.auth import get_current_active_user, get_current_user_optional
from app.core.config import settings
from app.services.charforge_integration import (
    CharForgeIntegration,
    CharacterConfig,
    ModelConfig as CharForgeModelConfig,
    MVAdapterConfig as ChargeForgeMVAdapterConfig,
    AdvancedTrainingConfig as CharForgeAdvancedTrainingConfig
)
from app.services.settings_service import get_user_env_vars

router = APIRouter()

# Pydantic models
class ModelConfig(BaseModel):
    base_model: str = "RunDiffusion/Juggernaut-XL-v9"
    vae_model: str = "madebyollin/sdxl-vae-fp16-fix"
    unet_model: Optional[str] = None
    adapter_path: str = "huanngzh/mv-adapter"
    scheduler: str = "ddpm"
    dtype: str = "float16"

class MVAdapterConfig(BaseModel):
    enabled: bool = False
    num_views: int = 6
    height: int = 768
    width: int = 768
    guidance_scale: float = 3.0
    reference_conditioning_scale: float = 1.0
    azimuth_degrees: List[int] = [0, 45, 90, 180, 270, 315]
    remove_background: bool = True

class AdvancedTrainingConfig(BaseModel):
    optimizer: str = "adamw"
    weight_decay: float = 1e-2
    lr_scheduler: str = "constant"
    gradient_checkpointing: bool = True
    train_text_encoder: bool = False
    noise_scheduler: str = "ddpm"
    gradient_accumulation: int = 1
    mixed_precision: str = "fp16"
    save_every: int = 250
    max_saves: int = 5

class TrainingRequest(BaseModel):
    character_id: int
    # Basic parameters
    steps: Optional[int] = 800
    batch_size: Optional[int] = 1
    learning_rate: Optional[float] = 8e-4
    train_dim: Optional[int] = 512
    rank_dim: Optional[int] = 8
    pulidflux_images: Optional[int] = 0

    # Model configuration
    model_config: Optional[ModelConfig] = ModelConfig()

    # MV Adapter configuration
    mv_adapter_config: Optional[MVAdapterConfig] = MVAdapterConfig()

    # Advanced training options
    advanced_config: Optional[AdvancedTrainingConfig] = AdvancedTrainingConfig()

    # ComfyUI model selection
    comfyui_checkpoint: Optional[str] = None
    comfyui_vae: Optional[str] = None
    comfyui_lora: Optional[str] = None

class TrainingResponse(BaseModel):
    id: int
    character_id: int
    character_name: Optional[str] = None
    status: str
    progress: float
    steps: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    rank_dim: Optional[int] = None
    train_dim: Optional[int] = None
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True

class CharacterResponse(BaseModel):
    id: int
    name: str
    input_image_path: Optional[str]
    dataset_id: Optional[int]
    trigger_word: Optional[str]
    status: str
    work_dir: str
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True

class CharacterCreateRequest(BaseModel):
    name: str
    input_image_path: Optional[str] = None
    dataset_id: Optional[int] = None
    trigger_word: Optional[str] = None

# Global integration instance
charforge = CharForgeIntegration()

@router.post("/characters", response_model=CharacterResponse)
async def create_character(
    request: CharacterCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_optional)
):
    """Create a new character."""

    # Validate input
    if not request.name or len(request.name.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Character name is required"
        )

    # Sanitize character name
    sanitized_name = ''.join(c for c in request.name if c.isalnum() or c in '_-').strip()
    if not sanitized_name or len(sanitized_name) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid character name. Use only letters, numbers, underscores, and hyphens (max 100 chars)"
        )

    # Validate that either image path OR dataset is provided
    if not request.input_image_path and not request.dataset_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either input_image_path or dataset_id must be provided"
        )

    # Validate image path if provided
    if request.input_image_path and not Path(request.input_image_path).exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid input image path"
        )

    # Validate dataset if provided
    if request.dataset_id:
        from app.core.database import Dataset
        dataset = db.query(Dataset).filter(
            Dataset.id == request.dataset_id,
            Dataset.user_id == current_user.id
        ).first()
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found"
            )
        if dataset.status != "ready":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dataset is not ready for training"
            )

    # Check if character name already exists for this user
    existing = db.query(Character).filter(
        Character.name == sanitized_name,
        Character.user_id == current_user.id
    ).first()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Character with this name already exists"
        )

    # Create character record
    character = Character(
        name=sanitized_name,
        user_id=current_user.id,
        input_image_path=request.input_image_path,
        dataset_id=request.dataset_id,
        trigger_word=request.trigger_word,
        work_dir=str(charforge.scratch_dir / sanitized_name),
        status="created"
    )

    db.add(character)
    db.commit()
    db.refresh(character)

    return character

@router.get("/characters", response_model=List[CharacterResponse])
async def list_characters(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_optional)
):
    """List all characters for the current user."""
    characters = db.query(Character).filter(Character.user_id == current_user.id).all()
    return characters

@router.get("/characters/{character_id}", response_model=CharacterResponse)
async def get_character(
    character_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_optional)
):
    """Get a specific character."""
    character = db.query(Character).filter(
        Character.id == character_id,
        Character.user_id == current_user.id
    ).first()
    
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )
    
    return character

@router.delete("/characters/{character_id}")
async def delete_character(
    character_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_optional)
):
    """Delete a character and cancel any active training sessions."""
    import shutil

    # Get character
    character = db.query(Character).filter(
        Character.id == character_id,
        Character.user_id == current_user.id
    ).first()

    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )

    # Cancel any active training sessions for this character
    active_sessions = db.query(TrainingSession).filter(
        TrainingSession.character_id == character_id,
        TrainingSession.status.in_(["pending", "running"])
    ).all()

    for session in active_sessions:
        session.status = "cancelled"
        session.completed_at = datetime.utcnow()

    # Delete work directory if it exists
    work_dir = Path(character.work_dir)
    if work_dir.exists():
        try:
            shutil.rmtree(work_dir)
        except Exception as e:
            # Log but don't fail if directory cleanup fails
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to delete work directory {work_dir}: {e}")

    # Delete character record
    db.delete(character)
    db.commit()

    return {"message": "Character deleted successfully", "character_id": character_id}

@router.post("/characters/{character_id}/train", response_model=TrainingResponse)
async def start_training(
    character_id: int,
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_optional)
):
    """Start training for a character."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"=== TRAINING REQUEST RECEIVED ===")
    logger.info(f"Character ID: {character_id}")
    logger.info(f"User ID: {current_user.id}")
    logger.info(f"Request data: steps={request.steps}, batch_size={request.batch_size}, learning_rate={request.learning_rate}")

    # Get character
    character = db.query(Character).filter(
        Character.id == character_id,
        Character.user_id == current_user.id
    ).first()

    if not character:
        logger.error(f"Character {character_id} not found for user {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )
    
    # Check if there's already a running training session
    existing_session = db.query(TrainingSession).filter(
        TrainingSession.character_id == character_id,
        TrainingSession.status.in_(["pending", "running"])
    ).first()

    if existing_session:
        logger.warning(f"Training already in progress for character {character_id}, session {existing_session.id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Training session already in progress"
        )

    # Enhanced validation for training parameters
    logger.info("Validating training parameters...")
    steps = request.steps or 800
    learning_rate = request.learning_rate or 8e-4
    batch_size = request.batch_size or 1
    rank_dim = request.rank_dim or 8
    train_dim = request.train_dim or 512

    logger.info(f"Parsed values: steps={steps}, lr={learning_rate}, batch={batch_size}, rank={rank_dim}, train_dim={train_dim}")

    if steps < 100 or steps > 10000:
        logger.error(f"Invalid steps: {steps}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Steps must be between 100 and 10000"
        )

    if learning_rate < 1e-6 or learning_rate > 1e-2:
        logger.error(f"Invalid learning rate: {learning_rate}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Learning rate must be between 1e-6 and 1e-2"
        )

    if batch_size < 1 or batch_size > 16:
        logger.error(f"Invalid batch size: {batch_size}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size must be between 1 and 16"
        )

    if rank_dim < 4 or rank_dim > 256:
        logger.error(f"Invalid rank dimension: {rank_dim}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Rank dimension must be between 4 and 256"
        )

    if train_dim < 256 or train_dim > 2048:
        logger.error(f"Invalid train dimension: {train_dim}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Train dimension must be between 256 and 2048"
        )

    logger.info("✓ All training parameters validated successfully")

    # Validate model configuration if provided
    if request.model_config:
        # Handle both dict and ModelConfig object
        dtype = request.model_config.dtype if hasattr(request.model_config, 'dtype') else request.model_config.get('dtype', 'float16')
        if dtype not in ["float16", "float32", "bfloat16"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid data type. Must be float16, float32, or bfloat16"
            )

    # Validate MV Adapter configuration if enabled
    if request.mv_adapter_config:
        # Handle both dict and MVAdapterConfig object
        enabled = request.mv_adapter_config.enabled if hasattr(request.mv_adapter_config, 'enabled') else request.mv_adapter_config.get('enabled', False)
        if enabled:
            num_views = request.mv_adapter_config.num_views if hasattr(request.mv_adapter_config, 'num_views') else request.mv_adapter_config.get('num_views', 6)
            guidance_scale = request.mv_adapter_config.guidance_scale if hasattr(request.mv_adapter_config, 'guidance_scale') else request.mv_adapter_config.get('guidance_scale', 3.0)

            if num_views < 4 or num_views > 12:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Number of views must be between 4 and 12"
                )

            if guidance_scale < 1.0 or guidance_scale > 20.0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Guidance scale must be between 1.0 and 20.0"
                )

    # Create training session
    training_session = TrainingSession(
        character_id=character_id,
        user_id=current_user.id,
        steps=request.steps,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        train_dim=request.train_dim,
        rank_dim=request.rank_dim,
        pulidflux_images=request.pulidflux_images,
        status="pending"
    )
    
    db.add(training_session)
    db.commit()
    db.refresh(training_session)
    
    # Start training in background
    background_tasks.add_task(
        run_training_background,
        training_session.id,
        character.id,  # Pass character ID instead of object
        request,
        current_user.id
    )

    return training_session

async def run_training_background(
    session_id: int,
    character_id: int,  # Changed from Character object to int ID
    request: TrainingRequest,
    user_id: int
):
    """Background task to run training."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"=== STARTING TRAINING ===")
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Character ID: {character_id}")
    logger.info(f"User ID: {user_id}")

    db = next(get_db())

    try:
        # Re-query the character in this session
        character = db.query(Character).filter(Character.id == character_id).first()
        if not character:
            raise ValueError(f"Character {character_id} not found")

        logger.info(f"Character: {character.name}")
        logger.info(f"Dataset ID: {character.dataset_id}")
        logger.info(f"Input image: {character.input_image_path}")

        # Update session status
        session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        session.status = "running"
        session.started_at = datetime.utcnow()
        db.commit()
        logger.info(f"✓ Session status updated to 'running'")

        # Update character status
        character.status = "training"
        db.commit()
        logger.info(f"✓ Character status updated to 'training'")
        
        # Get user environment variables
        logger.info("Getting user environment variables...")
        env_vars = await get_user_env_vars(user_id, db)
        logger.info(f"✓ Environment variables loaded")

        # Prepare input images
        logger.info("Preparing input images...")
        input_images = []
        if character.dataset_id:
            logger.info(f"Loading images from dataset {character.dataset_id}")
            # Get images from dataset
            from app.core.database import DatasetImage
            dataset_images = db.query(DatasetImage).filter(
                DatasetImage.dataset_id == character.dataset_id
            ).all()
            logger.info(f"Found {len(dataset_images)} images in dataset")

            # Get paths to actual image files
            user_media_dir = settings.MEDIA_DIR / str(user_id)
            logger.info(f"User media directory: {user_media_dir}")
            input_images = [str(user_media_dir / img.filename) for img in dataset_images]

            # Verify files exist
            existing_images = [img for img in input_images if Path(img).exists()]
            logger.info(f"✓ {len(existing_images)}/{len(input_images)} images found on disk")

            if not input_images:
                raise ValueError(f"No images found in dataset {character.dataset_id}")
            if len(existing_images) < len(input_images):
                logger.warning(f"Some images are missing: {len(input_images) - len(existing_images)} files not found")

        elif character.input_image_path:
            logger.info(f"Using single image: {character.input_image_path}")
            # Single image training
            input_images = [character.input_image_path]
            if not Path(character.input_image_path).exists():
                raise ValueError(f"Input image not found: {character.input_image_path}")
        else:
            raise ValueError("Character has no input images or dataset")

        logger.info(f"✓ Prepared {len(input_images)} images for training")

        # Create CharForge config
        logger.info("Creating training configuration...")
        try:
            # Helper function to get value from dict or object
            def get_attr(obj, key, default=None):
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)

            # Convert Pydantic models to dataclasses for CharForge
            charforge_model_config = None
            if request.model_config:
                mc = request.model_config
                charforge_model_config = CharForgeModelConfig(
                    base_model=get_attr(mc, 'base_model', 'RunDiffusion/Juggernaut-XL-v9'),
                    vae_model=get_attr(mc, 'vae_model', 'madebyollin/sdxl-vae-fp16-fix'),
                    unet_model=get_attr(mc, 'unet_model', None),
                    adapter_path=get_attr(mc, 'adapter_path', 'huanngzh/mv-adapter'),
                    scheduler=get_attr(mc, 'scheduler', 'ddpm'),
                    dtype=get_attr(mc, 'dtype', 'float16')
                )

            charforge_mv_config = None
            if request.mv_adapter_config:
                mvc = request.mv_adapter_config
                charforge_mv_config = ChargeForgeMVAdapterConfig(
                    enabled=get_attr(mvc, 'enabled', False),
                    num_views=get_attr(mvc, 'num_views', 6),
                    height=get_attr(mvc, 'height', 768),
                    width=get_attr(mvc, 'width', 768),
                    guidance_scale=get_attr(mvc, 'guidance_scale', 3.0),
                    reference_conditioning_scale=get_attr(mvc, 'reference_conditioning_scale', 1.0),
                    azimuth_degrees=get_attr(mvc, 'azimuth_degrees', [0, 45, 90, 180, 270, 315]),
                    remove_background=get_attr(mvc, 'remove_background', True)
                )

            charforge_advanced_config = None
            if request.advanced_config:
                ac = request.advanced_config
                charforge_advanced_config = CharForgeAdvancedTrainingConfig(
                    optimizer=get_attr(ac, 'optimizer', 'adamw'),
                    weight_decay=get_attr(ac, 'weight_decay', 1e-2),
                    lr_scheduler=get_attr(ac, 'lr_scheduler', 'constant'),
                    gradient_checkpointing=get_attr(ac, 'gradient_checkpointing', True),
                    train_text_encoder=get_attr(ac, 'train_text_encoder', False),
                    noise_scheduler=get_attr(ac, 'noise_scheduler', 'ddpm'),
                    gradient_accumulation=get_attr(ac, 'gradient_accumulation', 1),
                    mixed_precision=get_attr(ac, 'mixed_precision', 'fp16'),
                    save_every=get_attr(ac, 'save_every', 250),
                    max_saves=get_attr(ac, 'max_saves', 5)
                )

            config = CharacterConfig(
                name=character.name,
                input_image=input_images[0] if len(input_images) == 1 else None,
                input_images=input_images if len(input_images) > 1 else None,
                work_dir=character.work_dir,
                steps=request.steps or 800,
                batch_size=request.batch_size or 1,
                learning_rate=request.learning_rate or 8e-4,
                train_dim=request.train_dim or 512,
                rank_dim=request.rank_dim or 8,
                pulidflux_images=request.pulidflux_images or 0,

                # Model configuration (converted to dataclasses)
                model_config=charforge_model_config,
                mv_adapter_config=charforge_mv_config,
                advanced_config=charforge_advanced_config,

                # ComfyUI model paths
                comfyui_checkpoint=request.comfyui_checkpoint or "",
                comfyui_vae=request.comfyui_vae or "",
                comfyui_lora=request.comfyui_lora or ""
            )
            logger.info(f"✓ Configuration created:")
            logger.info(f"  Steps: {config.steps}")
            logger.info(f"  Batch size: {config.batch_size}")
            logger.info(f"  Learning rate: {config.learning_rate}")
            logger.info(f"  Train dim: {config.train_dim}")
            logger.info(f"  Work dir: {config.work_dir}")
        except Exception as e:
            logger.error(f"Failed to create config: {e}")
            raise

        # Progress callback
        def update_progress(progress: float, message: str):
            logger.info(f"Training progress: {progress:.1f}% - {message}")
            session.progress = progress
            db.commit()

        # Run training
        logger.info("Starting CharForge training process...")
        result = await charforge.run_training(config, env_vars, update_progress)
        logger.info(f"Training completed with result: {result}")
        
        # Update session with results
        session.status = "completed" if result["success"] else "failed"
        session.completed_at = datetime.utcnow()
        session.progress = 100.0 if result["success"] else session.progress
        
        # Update character status
        character.status = "completed" if result["success"] else "failed"
        if result["success"]:
            character.completed_at = datetime.utcnow()
        
        db.commit()
        
    except Exception as e:
        # Handle errors with detailed logging
        import logging
        import traceback
        logger = logging.getLogger(__name__)

        error_details = traceback.format_exc()
        logger.error(f"Training failed for session {session_id}: {str(e)}")
        logger.error(f"Full traceback:\n{error_details}")

        session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if session:
            session.status = "failed"
            session.completed_at = datetime.utcnow()
            # Store error message in session (we'll add this field to the model)
            logger.error(f"Error type: {type(e).__name__}, Message: {str(e)}")

        # Update character status
        char = db.query(Character).filter(Character.id == character.id).first()
        if char:
            char.status = "failed"

        db.commit()

        # Re-raise the exception so it appears in logs
        raise

    finally:
        db.close()

@router.get("/characters/{character_id}/training", response_model=List[TrainingResponse])
async def get_training_sessions(
    character_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_optional)
):
    """Get training sessions for a character."""
    
    # Verify character ownership
    character = db.query(Character).filter(
        Character.id == character_id,
        Character.user_id == current_user.id
    ).first()
    
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found"
        )
    
    sessions = db.query(TrainingSession).filter(
        TrainingSession.character_id == character_id
    ).order_by(TrainingSession.created_at.desc()).all()
    
    return sessions

@router.get("/training/{session_id}", response_model=TrainingResponse)
async def get_training_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_optional)
):
    """Get a specific training session."""

    # Validate session_id
    if session_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID"
        )

    session = db.query(TrainingSession).filter(
        TrainingSession.id == session_id,
        TrainingSession.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training session not found"
        )

    return session

@router.get("/training", response_model=List[TrainingResponse])
async def get_all_training_sessions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_optional)
):
    """Get all training sessions for the current user."""
    # Query training sessions with character names
    sessions = db.query(TrainingSession, Character.name.label('character_name')).join(
        Character, TrainingSession.character_id == Character.id
    ).filter(
        TrainingSession.user_id == current_user.id
    ).order_by(TrainingSession.created_at.desc()).all()

    # Convert to response format
    result = []
    for session, character_name in sessions:
        session_dict = {
            "id": session.id,
            "character_id": session.character_id,
            "character_name": character_name,
            "status": session.status,
            "progress": session.progress,
            "steps": session.steps,
            "batch_size": session.batch_size,
            "learning_rate": session.learning_rate,
            "rank_dim": session.rank_dim,
            "train_dim": session.train_dim,
            "created_at": session.created_at,
            "started_at": session.started_at,
            "completed_at": session.completed_at
        }
        result.append(session_dict)

    return result

@router.post("/training/{session_id}/cancel")
async def cancel_training_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user_optional)
):
    """Cancel a stuck or running training session."""
    import logging
    logger = logging.getLogger(__name__)

    session = db.query(TrainingSession).filter(
        TrainingSession.id == session_id,
        TrainingSession.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training session not found"
        )

    if session.status in ["completed", "failed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel session with status: {session.status}"
        )

    logger.info(f"Cancelling training session {session_id} (status: {session.status})")

    # Update session status
    session.status = "failed"
    session.completed_at = datetime.utcnow()

    # Update character status
    character = db.query(Character).filter(Character.id == session.character_id).first()
    if character and character.status == "training":
        character.status = "failed"

    db.commit()

    logger.info(f"✓ Training session {session_id} cancelled")

    return {"message": "Training session cancelled", "session_id": session_id}
