"""Configuration management for Video Analyser."""

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProcessingConfig(BaseModel):
    """Video processing configuration."""

    max_video_size_mb: int = Field(default=500, ge=1, le=5000)
    supported_formats: list[str] = Field(
        default=["mp4", "mov", "avi", "webm"], min_length=1
    )
    temp_dir: Path = Field(default=Path("temp"))
    cleanup_temp_files: bool = Field(default=True)

    @field_validator("supported_formats")
    @classmethod
    def validate_formats(cls, v: list[str]) -> list[str]:
        """Validate supported video formats."""
        valid_formats = {"mp4", "mov", "avi", "webm", "mkv", "flv"}
        for fmt in v:
            if fmt.lower() not in valid_formats:
                raise ValueError(f"Unsupported format: {fmt}")
        return [fmt.lower() for fmt in v]


class SceneDetectionConfig(BaseModel):
    """Scene detection configuration."""

    method: str = Field(default="threshold", pattern="^(threshold|adaptive)$")
    threshold: float = Field(default=0.4, ge=0.1, le=0.9)
    min_scene_duration: float = Field(default=2.0, ge=0.5, le=30.0)
    fallback_interval: float = Field(default=30.0, ge=5.0, le=300.0)


class AudioConfig(BaseModel):
    """Audio processing configuration."""

    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    channels: int = Field(default=1, ge=1, le=2)
    noise_reduction: bool = Field(default=False)
    normalize_audio: bool = Field(default=True)


class TranscriptionConfig(BaseModel):
    """Speech transcription configuration."""

    model: str = Field(default="whisper-base")
    language: str = Field(default="auto")
    word_timestamps: bool = Field(default=True)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    device: str = Field(default="auto")  # auto, cpu, cuda

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate Whisper model name."""
        valid_models = {
            "whisper-tiny",
            "whisper-base",
            "whisper-small",
            "whisper-medium",
            "whisper-large",
            "whisper-large-v2",
            "whisper-large-v3",
        }
        if v not in valid_models:
            raise ValueError(f"Invalid model: {v}. Valid options: {valid_models}")
        return v


class AnalysisConfig(BaseModel):
    """Analysis configuration."""

    filler_words: list[str] = Field(
        default=["um", "uh", "like", "you know", "so", "actually", "basically"]
    )
    target_wpm_range: list[int] = Field(default=[140, 160], min_length=2, max_length=2)
    sentiment_analysis: bool = Field(default=True)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    @field_validator("target_wpm_range")
    @classmethod
    def validate_wpm_range(cls, v: list[int]) -> list[int]:
        """Validate WPM range."""
        if len(v) != 2:
            raise ValueError("WPM range must have exactly 2 values")
        if v[0] >= v[1]:
            raise ValueError("WPM min must be less than max")
        if v[0] < 50 or v[1] > 300:
            raise ValueError("WPM values must be between 50 and 300")
        return v


class VisualAnalysisConfig(BaseModel):
    """Visual analysis configuration."""

    # Frame extraction settings
    frames_per_scene: int = Field(default=3, ge=1, le=10)
    frame_quality: int = Field(default=85, ge=50, le=100)
    frame_format: str = Field(default="JPEG", pattern="^(JPEG|PNG|WEBP)$")
    max_frame_width: int = Field(default=1920, ge=480, le=4096)
    max_frame_height: int = Field(default=1080, ge=360, le=2160)

    # Quality assessment thresholds
    blur_threshold: float = Field(default=100.0, ge=10.0, le=1000.0)
    contrast_threshold: float = Field(default=20.0, ge=5.0, le=100.0)
    brightness_min: float = Field(default=50.0, ge=0.0, le=255.0)
    brightness_max: float = Field(default=200.0, ge=0.0, le=255.0)

    # Quality scoring weights
    blur_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    contrast_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    brightness_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    # Processing settings
    enable_quality_filtering: bool = Field(default=True)
    min_quality_score: float = Field(default=0.3, ge=0.0, le=1.0)
    save_extracted_frames: bool = Field(default=False)

    # Image captioning settings
    enable_captioning: bool = Field(default=True)
    captioning_backend: str = Field(default="api")  # local, api

    # Local captioning model settings
    captioning_model: str = Field(default="Salesforce/blip2-opt-2.7b")
    captioning_device: str = Field(default="auto")  # auto, cpu, cuda, mps
    max_caption_length: int = Field(default=50, ge=10, le=200)
    caption_temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    caption_batch_size: int = Field(default=1, ge=1, le=8)

    # API captioning settings
    api_provider: str = Field(default="anthropic")  # anthropic, openai, google, openrouter
    api_model: str = Field(
        default="claude-haiku-4-5-20251001"
    )  # Provider-specific default models
    api_key_env_var: str = Field(default="ANTHROPIC_API_KEY")
    api_max_concurrent: int = Field(default=5, ge=1, le=20)
    api_timeout: float = Field(default=30.0, ge=5.0, le=120.0)
    api_max_retries: int = Field(default=3, ge=0, le=10)

    # OCR settings
    enable_ocr: bool = Field(default=True)
    ocr_engine: str = Field(default="tesseract")  # tesseract, easyocr
    ocr_languages: list[str] = Field(default=["eng"])
    ocr_confidence_threshold: float = Field(default=60.0, ge=0.0, le=100.0)
    ocr_text_min_length: int = Field(default=3, ge=1, le=50)
    detect_slide_text: bool = Field(default=True)
    detect_ui_text: bool = Field(default=False)

    # Object detection settings
    enable_object_detection: bool = Field(default=True)
    object_detection_model: str = Field(default="heuristic")  # heuristic, yolov5, etc.
    object_detection_device: str = Field(default="auto")  # auto, cpu, cuda, mps
    object_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("blur_weight", "contrast_weight", "brightness_weight")
    @classmethod
    def validate_weights_sum(cls, v: float, info: ValidationInfo) -> float:
        """Ensure quality weights are reasonable."""
        if "blur_weight" in info.data and "contrast_weight" in info.data:
            total = (
                info.data.get("blur_weight", 0.0)
                + info.data.get("contrast_weight", 0.0)
                + v
            )
            if abs(total - 1.0) > 0.01:  # Allow small floating point errors
                raise ValueError("Quality weights must sum to 1.0")
        return v

    @field_validator("captioning_model")
    @classmethod
    def validate_captioning_model(cls, v: str) -> str:
        """Validate image captioning model name."""
        valid_models = {
            "Salesforce/blip2-opt-2.7b",
            "Salesforce/blip2-opt-6.7b",
            "Salesforce/blip2-flan-t5-xl",
            "Salesforce/blip-image-captioning-base",
            "Salesforce/blip-image-captioning-large",
        }
        if v not in valid_models:
            # Allow custom models but log a warning
            import logging

            logging.getLogger(__name__).warning(
                f"Using custom captioning model: {v}. Supported models: {valid_models}"
            )
        return v

    @field_validator("captioning_backend")
    @classmethod
    def validate_captioning_backend(cls, v: str) -> str:
        """Validate captioning backend."""
        valid_backends = {"local", "api"}
        if v.lower() not in valid_backends:
            raise ValueError(
                f"Invalid captioning backend: {v}. Valid options: {valid_backends}"
            )
        return v.lower()

    @field_validator("api_provider")
    @classmethod
    def validate_api_provider(cls, v: str) -> str:
        """Validate API provider."""
        valid_providers = {"anthropic", "openai", "google", "openrouter"}
        if v.lower() not in valid_providers:
            raise ValueError(
                f"Invalid API provider: {v}. Valid options: {valid_providers}"
            )
        return v.lower()

    @field_validator("api_model")
    @classmethod
    def validate_api_model(cls, v: str, info: ValidationInfo) -> str:
        """Validate and set default API model based on provider."""
        # Provider-specific default models
        default_models = {
            "anthropic": "claude-haiku-4-5-20251001",
            "openai": "gpt-4o",
            "google": "gemini-2.5-flash",
            "openrouter": "openai/gpt-4-turbo",
        }

        # If model is the global default and provider is set, use provider default
        if v == "claude-haiku-4-5-20251001" and "api_provider" in info.data:
            provider = info.data.get("api_provider", "anthropic")
            if provider in default_models and provider != "anthropic":
                return default_models[provider]

        return v

    @field_validator("ocr_engine")
    @classmethod
    def validate_ocr_engine(cls, v: str) -> str:
        """Validate OCR engine name."""
        valid_engines = {"tesseract", "easyocr"}
        if v.lower() not in valid_engines:
            raise ValueError(f"Invalid OCR engine: {v}. Valid options: {valid_engines}")
        return v.lower()

    @field_validator("ocr_languages")
    @classmethod
    def validate_ocr_languages(cls, v: list[str]) -> list[str]:
        """Validate OCR language codes."""
        # Common language codes for Tesseract
        common_languages = {
            "eng",
            "spa",
            "fra",
            "deu",
            "ita",
            "por",
            "rus",
            "jpn",
            "kor",
            "chi_sim",
            "chi_tra",
        }
        validated: list[str] = []
        for lang in v:
            if lang.lower() in common_languages:
                validated.append(lang.lower())
            else:
                # Allow custom language codes but log a warning
                import logging

                logging.getLogger(__name__).warning(
                    f"Using custom OCR language: {lang}. Common languages: {common_languages}"
                )
                validated.append(lang.lower())
        return validated


class OutputConfig(BaseModel):
    """Output configuration."""

    formats: list[str] = Field(default=["json", "html"], min_length=1)
    include_frames: bool = Field(default=True)
    frame_quality: int = Field(default=80, ge=1, le=100)
    report_template: str = Field(default="default")

    @field_validator("formats")
    @classmethod
    def validate_formats(cls, v: list[str]) -> list[str]:
        """Validate output formats."""
        valid_formats = {"json", "html", "pdf", "csv", "markdown"}
        for fmt in v:
            if fmt.lower() not in valid_formats:
                raise ValueError(f"Unsupported format: {fmt}")
        return [fmt.lower() for fmt in v]


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_enabled: bool = Field(default=True)
    file_path: Path = Field(default=Path("logs/video_analyser.log"))
    max_bytes: int = Field(default=10_000_000)  # 10MB
    backup_count: int = Field(default=5)
    console_enabled: bool = Field(default=True)

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()


class VideoAnalyserConfig(BaseSettings):
    """Main configuration for Video Analyser."""

    model_config = SettingsConfigDict(
        env_prefix="VIDEO_ANALYSER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
    )

    # Application settings
    app_name: str = Field(default="Video Analyser")
    version: str = Field(default_factory=lambda: __import__("importlib.metadata", fromlist=["version"]).version("video-analyser"))
    debug: bool = Field(default=False)

    # Component configurations
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    scene_detection: SceneDetectionConfig = Field(default_factory=SceneDetectionConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    visual_analysis: VisualAnalysisConfig = Field(default_factory=VisualAnalysisConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: Path | None = None) -> VideoAnalyserConfig:
    """
    Load configuration from file and environment variables.

    Args:
        config_path: Path to YAML config file. If None, uses default locations.

    Returns:
        Loaded configuration object.
    """
    config_data: dict[str, Any] = {}

    # Try to load from YAML file
    if config_path is None:
        # Check default locations
        possible_paths = [
            Path("config/config.yaml"),
            Path("config.yaml"),
            Path.home() / ".video-analyser" / "config.yaml",
        ]
        for path in possible_paths:
            if path.exists():
                config_path = path
                break

    if config_path and config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")

    # Create config object (environment variables will override file values)
    return VideoAnalyserConfig(**config_data)


def save_config(config: VideoAnalyserConfig, config_path: Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration object to save.
        config_path: Path where to save the config file.
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict, excluding computed fields and converting Path objects to strings
    config_dict = config.model_dump(exclude={"version"}, mode="json")

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(
            config_dict,
            f,
            default_flow_style=False,
            indent=2,
            sort_keys=False,
        )


def setup_logging(config: LoggingConfig) -> None:
    """
    Set up logging based on configuration.

    Args:
        config: Logging configuration.
    """
    import logging.handlers

    # Create logger
    logger = logging.getLogger("video_analyser")
    logger.setLevel(getattr(logging, config.level))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(config.format)

    # Console handler
    if config.console_enabled:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if config.file_enabled:
        config.file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            config.file_path,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Set library loggers to WARNING to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


# Global configuration cache
_global_config: VideoAnalyserConfig | None = None


def get_config() -> VideoAnalyserConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
        setup_logging(_global_config.logging)
    return _global_config


def reload_config(_config_path: Path | None = None) -> VideoAnalyserConfig:
    """Reload configuration from file."""
    if hasattr(get_config, "_config"):
        delattr(get_config, "_config")
    return get_config()


def export_config_to_json(config: VideoAnalyserConfig, output_path: Path) -> None:
    """
    Export configuration to JSON format.

    Args:
        config: Configuration object to export.
        output_path: Path where to save the JSON file.
    """
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict with proper serialization
    config_dict = config.model_dump(mode="json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    logging.getLogger(__name__).info(f"Configuration exported to {output_path}")


def export_config_to_env(config: VideoAnalyserConfig, output_path: Path) -> None:
    """
    Export configuration to .env file format.

    Args:
        config: Configuration object to export.
        output_path: Path where to save the .env file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def flatten_config(
        data: dict[str, Any], prefix: str = "VIDEO_ANALYSER"
    ) -> dict[str, str]:
        """Flatten nested config to env var format."""
        result: dict[str, str] = {}

        for key, value in data.items():
            env_key = f"{prefix}__{key.upper()}"

            if isinstance(value, dict):
                # Recursively flatten nested dicts
                nested = flatten_config(value, env_key)  # type: ignore[arg-type]
                result.update(nested)
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                result[env_key] = ",".join(str(v) for v in value)  # type: ignore[arg-type]
            elif isinstance(value, bool):
                # Convert bools to string
                result[env_key] = "true" if value else "false"
            elif isinstance(value, Path):
                # Convert Path to string
                result[env_key] = str(value)
            else:
                result[env_key] = str(value)

        return result

    config_dict: dict[str, Any] = config.model_dump(exclude={"version"}, mode="json")
    env_vars = flatten_config(config_dict)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Video Analyser Configuration\n")
        f.write("# Auto-generated .env file\n\n")
        for key, value in sorted(env_vars.items()):
            f.write(f'{key}="{value}"\n')

    logging.getLogger(__name__).info(f"Configuration exported to {output_path}")


def create_default_config_file(output_path: Path) -> None:
    """
    Create a default configuration file at the specified path.

    Args:
        output_path: Path where to create the config file.
    """
    default_config = VideoAnalyserConfig()
    save_config(default_config, output_path)
    logging.getLogger(__name__).info(f"Default config created at {output_path}")


def validate_config(config: VideoAnalyserConfig) -> tuple[bool, list[str]]:
    """
    Validate configuration and return validation results.

    Args:
        config: Configuration to validate.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    errors: list[str] = []

    # Validate directory paths can be created
    try:
        temp_path = config.processing.temp_dir
        if not temp_path.exists():
            temp_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot create temp directory: {e}")

    # Validate logging directory can be created
    try:
        log_path = config.logging.file_path.parent
        if not log_path.exists():
            log_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot create log directory: {e}")

    # Validate WPM range
    if config.analysis.target_wpm_range[0] >= config.analysis.target_wpm_range[1]:
        errors.append("WPM range min must be less than max")

    # Validate quality weights sum to approximately 1.0
    weights_sum = (
        config.visual_analysis.blur_weight
        + config.visual_analysis.contrast_weight
        + config.visual_analysis.brightness_weight
    )
    if abs(weights_sum - 1.0) > 0.01:
        errors.append(f"Quality weights must sum to 1.0, got {weights_sum}")

    # Validate output formats
    if not config.output.formats:
        errors.append("At least one output format must be specified")

    return len(errors) == 0, errors


def get_config_schema() -> dict[str, Any]:
    """
    Get JSON schema for the configuration.

    Returns:
        JSON schema for VideoAnalyserConfig.
    """
    return VideoAnalyserConfig.model_json_schema()  # type: ignore


def reset_config_to_defaults() -> VideoAnalyserConfig:
    """
    Reset configuration to default values.

    Returns:
        New VideoAnalyserConfig with default values.
    """
    global _global_config
    _global_config = VideoAnalyserConfig()
    setup_logging(_global_config.logging)
    return _global_config
