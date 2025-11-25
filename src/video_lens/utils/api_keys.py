"""Secure API key management for external services.

This module provides utilities for safely loading API keys from environment
variables and .env files, with automatic fallback behavior.
"""

import logging
import os
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# Supported API providers
APIProvider = Literal["anthropic", "openai", "google", "openrouter"]

# Standard environment variable names for each provider
STANDARD_ENV_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def load_env_file(env_path: Path | None = None) -> bool:
    """
    Load environment variables from .env file.

    Args:
        env_path: Optional path to .env file. If None, searches current directory.

    Returns:
        True if .env file was loaded, False otherwise
    """
    if env_path is None:
        # Search for .env in current directory and parents
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            env_file = parent / ".env"
            if env_file.exists():
                env_path = env_file
                break

    if env_path and Path(env_path).exists():
        try:
            # Simple .env parser (doesn't require python-dotenv dependency)
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        # Remove quotes if present
                        value = value.strip().strip("\"'")
                        os.environ[key.strip()] = value
            logger.debug(f"Loaded environment variables from {env_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load .env file from {env_path}: {e}")
            return False

    return False


def get_api_key(
    provider: APIProvider,
    env_var_name: str | None = None,
    auto_load_env: bool = True,
) -> str | None:
    """
    Get API key for a provider using secure fallback chain.

    Priority order:
    1. Custom environment variable (if specified)
    2. Standard environment variable for provider
    3. .env file (if auto_load_env=True)
    4. Return None (caller should handle fallback)

    Args:
        provider: API provider name ('anthropic', 'openai', 'google')
        env_var_name: Optional custom environment variable name
        auto_load_env: Whether to automatically load from .env file

    Returns:
        API key string if found, None otherwise

    Example:
        >>> key = get_api_key("anthropic")
        >>> if key:
        ...     # Use API
        ... else:
        ...     # Fall back to local model
    """
    # 1. Try custom environment variable
    if env_var_name and (key := os.getenv(env_var_name)):
        logger.debug(f"Found API key in {env_var_name}")
        return key

    # 2. Try standard environment variable
    standard_var = STANDARD_ENV_VARS.get(provider)
    if standard_var and (key := os.getenv(standard_var)):
        logger.debug(f"Found API key in {standard_var}")
        return key

    # 3. Try loading from .env file
    if auto_load_env and load_env_file():
        # Retry after loading .env
        if env_var_name and (key := os.getenv(env_var_name)):
            logger.debug(f"Found API key in {env_var_name} (from .env)")
            return key
        if standard_var and (key := os.getenv(standard_var)):
            logger.debug(f"Found API key in {standard_var} (from .env)")
            return key

    # 4. No key found
    logger.debug(f"No API key found for provider '{provider}'")
    return None


def validate_api_key(provider: APIProvider, api_key: str) -> bool:
    """
    Validate API key format for a provider.

    Args:
        provider: API provider name
        api_key: API key to validate

    Returns:
        True if key format is valid, False otherwise
    """
    if not api_key:
        return False

    # Basic format validation
    if provider == "anthropic":
        # Anthropic keys start with sk-ant-
        return api_key.startswith("sk-ant-") and len(api_key) > 20
    elif provider == "openai":
        # OpenAI keys start with sk-
        return api_key.startswith("sk-") and len(api_key) > 20
    elif provider == "google":
        # Google API keys are typically 39 characters
        return len(api_key) > 20
    elif provider == "openrouter":
        # OpenRouter keys start with sk-or-
        return api_key.startswith("sk-or-") and len(api_key) > 20

    return False


def mask_api_key(api_key: str, visible_chars: int = 4) -> str:
    """
    Mask an API key for safe logging.

    Args:
        api_key: API key to mask
        visible_chars: Number of characters to show at start/end

    Returns:
        Masked API key string

    Example:
        >>> mask_api_key("sk-ant-1234567890abcdef")
        'sk-a...cdef'
    """
    if not api_key or len(api_key) <= visible_chars * 2:
        return "***"

    return f"{api_key[:visible_chars]}...{api_key[-visible_chars:]}"


def get_api_key_with_validation(
    provider: APIProvider,
    env_var_name: str | None = None,
    auto_load_env: bool = True,
    log_result: bool = True,
) -> tuple[str | None, str]:
    """
    Get and validate API key, returning key and status message.

    Args:
        provider: API provider name
        env_var_name: Optional custom environment variable name
        auto_load_env: Whether to automatically load from .env file
        log_result: Whether to log the result

    Returns:
        Tuple of (api_key or None, status_message)
    """
    key = get_api_key(provider, env_var_name, auto_load_env)

    if not key:
        msg = f"No API key found for {provider}. Set {STANDARD_ENV_VARS.get(provider)} or create .env file."
        if log_result:
            logger.warning(msg)
        return None, msg

    if not validate_api_key(provider, key):
        masked = mask_api_key(key)
        msg = f"Invalid API key format for {provider}: {masked}"
        if log_result:
            logger.error(msg)
        return None, msg

    masked = mask_api_key(key)
    msg = f"Valid API key found for {provider}: {masked}"
    if log_result:
        logger.info(msg)

    return key, msg
