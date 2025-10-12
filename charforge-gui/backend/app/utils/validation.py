"""
Validation utilities for MASUKA
Provides comprehensive validation and sanitization functions
"""

import re
from pathlib import Path
from typing import Optional, List, Union


class ValidationError(Exception):
    """Raised when validation fails"""
    pass


def validate_character_name(name: str) -> str:
    """
    Validate and sanitize character name

    Args:
        name: Character name to validate

    Returns:
        Sanitized name

    Raises:
        ValidationError: If name is invalid
    """
    if not name or not isinstance(name, str):
        raise ValidationError("Character name must be a non-empty string")

    # Sanitize: keep only alphanumeric, underscores, and hyphens
    sanitized = ''.join(c for c in name if c.isalnum() or c in '_-')
    sanitized = sanitized.strip()

    if not sanitized:
        raise ValidationError("Character name must contain at least one alphanumeric character")

    if len(sanitized) > 100:
        raise ValidationError("Character name too long (max 100 characters)")

    # Prevent path traversal
    if '..' in sanitized or '/' in sanitized or '\\' in sanitized:
        raise ValidationError("Character name contains invalid characters")

    return sanitized


def validate_file_path(
    path: Union[str, Path],
    must_exist: bool = True,
    allowed_extensions: Optional[List[str]] = None
) -> Path:
    """
    Validate file path for safety

    Args:
        path: Path to validate
        must_exist: Whether file must exist
        allowed_extensions: List of allowed file extensions (e.g., ['.jpg', '.png'])

    Returns:
        Validated Path object

    Raises:
        ValidationError: If path is invalid or unsafe
    """
    if not path:
        raise ValidationError("Path cannot be empty")

    try:
        path_obj = Path(path).resolve()
    except (ValueError, OSError) as e:
        raise ValidationError(f"Invalid path: {e}")

    # Check for path traversal
    if '..' in str(path) or not path_obj.is_absolute():
        raise ValidationError("Path traversal detected")

    # Check existence
    if must_exist and not path_obj.exists():
        raise ValidationError(f"File does not exist: {path}")

    # Check extension
    if allowed_extensions:
        if path_obj.suffix.lower() not in allowed_extensions:
            raise ValidationError(
                f"Invalid file extension. Allowed: {', '.join(allowed_extensions)}"
            )

    return path_obj


def validate_training_params(
    steps: int,
    batch_size: int,
    learning_rate: float,
    train_dim: int,
    rank_dim: int
) -> dict:
    """
    Validate training parameters

    Returns:
        Dict of validated parameters

    Raises:
        ValidationError: If any parameter is invalid
    """
    errors = []

    if not (100 <= steps <= 10000):
        errors.append("Steps must be between 100 and 10000")

    if not (1 <= batch_size <= 16):
        errors.append("Batch size must be between 1 and 16")

    if not (1e-6 <= learning_rate <= 1e-2):
        errors.append("Learning rate must be between 1e-6 and 1e-2")

    if not (256 <= train_dim <= 2048):
        errors.append("Train dimension must be between 256 and 2048")

    if not (4 <= rank_dim <= 256):
        errors.append("Rank dimension must be between 4 and 256")

    if errors:
        raise ValidationError("; ".join(errors))

    return {
        'steps': int(steps),
        'batch_size': int(batch_size),
        'learning_rate': float(learning_rate),
        'train_dim': int(train_dim),
        'rank_dim': int(rank_dim)
    }


def sanitize_prompt(prompt: str, max_length: int = 2000) -> str:
    """
    Sanitize user prompt while preserving readability

    Args:
        prompt: User input prompt
        max_length: Maximum allowed length

    Returns:
        Sanitized prompt

    Raises:
        ValidationError: If prompt is invalid
    """
    if not prompt or not isinstance(prompt, str):
        raise ValidationError("Prompt must be a non-empty string")

    # Remove potentially dangerous characters
    dangerous_chars = ['`', '$', '\\', ';', '|', '&']
    sanitized = prompt
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')

    # Normalize whitespace
    sanitized = ' '.join(sanitized.split())

    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    if not sanitized:
        raise ValidationError("Prompt cannot be empty after sanitization")

    return sanitized


def validate_api_key(key: str, key_type: str) -> bool:
    """
    Validate API key format

    Args:
        key: API key to validate
        key_type: Type of key (e.g., 'hf', 'openai', 'civitai')

    Returns:
        True if valid

    Raises:
        ValidationError: If key is invalid
    """
    if not key or not isinstance(key, str):
        raise ValidationError(f"{key_type} API key must be a non-empty string")

    # Check for common patterns
    patterns = {
        'hf': r'^hf_[a-zA-Z0-9]{20,}$',
        'openai': r'^sk-[a-zA-Z0-9]{40,}$',
        'civitai': r'^[a-f0-9]{32}$',
    }

    if key_type in patterns:
        if not re.match(patterns[key_type], key):
            raise ValidationError(f"Invalid {key_type} API key format")

    return True


def validate_dataset_config(
    name: str,
    trigger_word: str,
    caption_template: str,
    image_count: int
) -> dict:
    """
    Validate dataset configuration

    Returns:
        Dict of validated parameters

    Raises:
        ValidationError: If any parameter is invalid
    """
    errors = []

    # Validate name
    if not name or len(name) > 100:
        errors.append("Dataset name must be 1-100 characters")

    # Validate trigger word
    if not trigger_word or len(trigger_word) > 50:
        errors.append("Trigger word must be 1-50 characters")

    # Validate caption template
    if caption_template and len(caption_template) > 500:
        errors.append("Caption template too long (max 500 characters)")

    # Validate image count
    if image_count < 1 or image_count > 1000:
        errors.append("Image count must be between 1 and 1000")

    if errors:
        raise ValidationError("; ".join(errors))

    return {
        'name': sanitize_character_name(name),
        'trigger_word': sanitize_character_name(trigger_word),
        'caption_template': caption_template,
        'image_count': int(image_count)
    }


def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """
    Sanitize filename to prevent path traversal and other attacks

    Args:
        filename: Filename to sanitize
        max_length: Maximum allowed length

    Returns:
        Safe filename

    Raises:
        ValidationError: If filename is invalid
    """
    if not filename:
        raise ValidationError("Filename cannot be empty")

    # Remove path components
    filename = Path(filename).name

    # Keep only safe characters
    safe_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-')
    sanitized = ''.join(c for c in filename if c in safe_chars)

    if not sanitized:
        raise ValidationError("Filename must contain at least one alphanumeric character")

    # Truncate if too long
    if len(sanitized) > max_length:
        # Preserve extension
        parts = sanitized.rsplit('.', 1)
        if len(parts) == 2:
            name, ext = parts
            sanitized = name[:max_length - len(ext) - 1] + '.' + ext
        else:
            sanitized = sanitized[:max_length]

    return sanitized


def sanitize_character_name(name: str) -> str:
    """Alias for validate_character_name for consistency"""
    return validate_character_name(name)
