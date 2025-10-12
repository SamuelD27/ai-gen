from pydantic_settings import BaseSettings
from pydantic import field_validator, model_validator
from typing import List
import os
from pathlib import Path

def _parse_bool_env(var_name: str, default: str = "false") -> bool:
    """Parse boolean environment variable with multiple valid representations."""
    val = os.getenv(var_name, default)
    return str(val).strip().lower() in {"1", "true", "yes", "on"}

class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"  # development, staging, production

    # API Configuration
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production-min-32-chars")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Authentication (Optional - disabled by default)
    ENABLE_AUTH: bool = _parse_bool_env("ENABLE_AUTH", "false")
    ALLOW_REGISTRATION: bool = _parse_bool_env("ALLOW_REGISTRATION", "false")
    DEFAULT_USER_ID: int = int(os.getenv("DEFAULT_USER_ID", "1"))  # Used when auth is disabled
    
    # Database
    DATABASE_URL: str = "sqlite:///./database.db"
    
    # CORS - Enhanced for remote access
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://0.0.0.0:5173",
        "http://0.0.0.0:3000"
    ]

    # Server Configuration
    HOST: str = "0.0.0.0"  # Allow external connections
    PORT: int = 8000
    FRONTEND_HOST: str = "0.0.0.0"
    FRONTEND_PORT: int = 5173
    
    # File Storage
    BASE_DIR: Path = Path(__file__).parent.parent.parent.parent.parent  # 5 parents up to /content/ai-gen
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    MEDIA_DIR: Path = BASE_DIR / "media"
    RESULTS_DIR: Path = BASE_DIR / "results"

    # CharForge Integration
    CHARFORGE_ROOT: Path = BASE_DIR  # Points to the main ai-gen directory
    CHARFORGE_SCRATCH_DIR: Path = CHARFORGE_ROOT / "scratch"
    COMFYUI_PATH: str = str(BASE_DIR / "ComfyUI")  # Default ComfyUI path

    # Environment Variables for MASUKA
    HF_TOKEN: str = ""
    HF_HOME: str = ""
    CIVITAI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    FAL_KEY: str = ""

    # Video Generation APIs
    COMET_API_KEY: str = ""  # CometAPI for Sora 2, VEO3, Runway
    OPENAI_API_KEY: str = ""  # Direct OpenAI Sora access
    
    # Training Defaults
    DEFAULT_STEPS: int = 800
    DEFAULT_BATCH_SIZE: int = 1
    DEFAULT_LEARNING_RATE: float = 8e-4
    DEFAULT_TRAIN_DIM: int = 512
    DEFAULT_RANK_DIM: int = 8
    
    # Inference Defaults
    DEFAULT_LORA_WEIGHT: float = 0.73
    DEFAULT_TEST_DIM: int = 1024
    DEFAULT_INFERENCE_STEPS: int = 30
    DEFAULT_BATCH_SIZE_INFERENCE: int = 4
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Create required directories after settings initialization
settings.MEDIA_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set secure permissions on media directory (Unix only)
try:
    import stat
    settings.MEDIA_DIR.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)  # 750
except (OSError, AttributeError):
    pass  # Windows or permission error

print(f"✓ Media directories created:")
print(f"  MEDIA_DIR: {settings.MEDIA_DIR}")
print(f"  UPLOAD_DIR: {settings.UPLOAD_DIR}")
print(f"  RESULTS_DIR: {settings.RESULTS_DIR}")
