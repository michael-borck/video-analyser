"""Tests for configuration management."""

import json
from pathlib import Path

import pytest
import yaml

from video_analyser.utils.config import (
    VideoAnalyserConfig,
    create_default_config_file,
    export_config_to_env,
    export_config_to_json,
    get_config,
    get_config_schema,
    load_config,
    reset_config_to_defaults,
    save_config,
    setup_logging,
    validate_config,
)


def test_default_config() -> None:
    """Test that default configuration loads correctly."""
    config = VideoAnalyserConfig()

    assert config.app_name == "Video Analyser"
    assert config.debug is False
    assert config.processing.max_video_size_mb == 500
    assert config.scene_detection.threshold == 0.4
    assert config.transcription.model == "whisper-base"
    assert config.logging.level == "INFO"


def test_config_validation() -> None:
    """Test configuration validation."""
    # Test invalid video format
    with pytest.raises(ValueError, match="Unsupported format"):
        VideoAnalyserConfig(processing={"supported_formats": ["invalid"]})

    # Test invalid WPM range
    with pytest.raises(ValueError, match="WPM min must be less than max"):
        VideoAnalyserConfig(analysis={"target_wpm_range": [160, 140]})

    # Test invalid log level
    with pytest.raises(ValueError, match="Invalid log level"):
        VideoAnalyserConfig(logging={"level": "INVALID"})


def test_load_config_from_file(temp_dir: Path) -> None:
    """Test loading configuration from YAML file."""
    config_file = temp_dir / "test_config.yaml"
    config_data = {
        "app_name": "Test App",
        "debug": True,
        "processing": {"max_video_size_mb": 100},
        "logging": {"level": "DEBUG"},
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = load_config(config_file)

    assert config.app_name == "Test App"
    assert config.debug is True
    assert config.processing.max_video_size_mb == 100
    assert config.logging.level == "DEBUG"


def test_save_config(temp_dir: Path) -> None:
    """Test saving configuration to file."""
    config = VideoAnalyserConfig(
        app_name="Test Save", debug=True, processing={"max_video_size_mb": 250}
    )

    config_file = temp_dir / "saved_config.yaml"
    save_config(config, config_file)

    assert config_file.exists()

    # Load and verify
    loaded_config = load_config(config_file)
    assert loaded_config.app_name == "Test Save"
    assert loaded_config.debug is True
    assert loaded_config.processing.max_video_size_mb == 250


def test_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test environment variable overrides."""
    monkeypatch.setenv("VIDEO_ANALYSER_APP_NAME", "Env Override")
    monkeypatch.setenv("VIDEO_ANALYSER_DEBUG", "true")
    monkeypatch.setenv("VIDEO_ANALYSER_PROCESSING__MAX_VIDEO_SIZE_MB", "750")

    config = VideoAnalyserConfig()

    assert config.app_name == "Env Override"
    assert config.debug is True
    assert config.processing.max_video_size_mb == 750


def test_nested_config_validation() -> None:
    """Test validation of nested configuration objects."""
    config = VideoAnalyserConfig()

    # Test scene detection bounds
    assert 0.1 <= config.scene_detection.threshold <= 0.9
    assert config.scene_detection.min_scene_duration >= 0.5

    # Test audio settings
    assert 8000 <= config.audio.sample_rate <= 48000
    assert 1 <= config.audio.channels <= 2

    # Test analysis settings
    assert len(config.analysis.target_wpm_range) == 2
    assert config.analysis.target_wpm_range[0] < config.analysis.target_wpm_range[1]


def test_setup_logging(temp_dir: Path) -> None:
    """Test logging setup."""
    import logging

    log_file = temp_dir / "test.log"

    from video_analyser.utils.config import LoggingConfig

    logging_config = LoggingConfig(
        level="DEBUG",
        file_path=log_file,
        console_enabled=False,  # Don't spam console during tests
    )

    setup_logging(logging_config)

    logger = logging.getLogger("video_analyser")
    logger.debug("Test debug message")
    logger.info("Test info message")

    assert log_file.exists()
    log_content = log_file.read_text()
    assert "Test debug message" in log_content
    assert "Test info message" in log_content


def test_config_file_precedence(temp_dir: Path) -> None:
    """Test configuration file precedence."""
    # Create config in temp dir
    config_file = temp_dir / "config.yaml"
    config_data = {"app_name": "File Config", "debug": False}

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = load_config(config_file)
    assert config.app_name == "File Config"
    assert config.debug is False


def test_missing_config_file() -> None:
    """Test handling of missing config file."""
    nonexistent_file = Path("nonexistent_config.yaml")
    config = load_config(nonexistent_file)

    # Should still work with defaults
    assert config.app_name == "Video Analyser"
    assert config.debug is False


def test_malformed_config_file(temp_dir: Path) -> None:
    """Test handling of malformed config file."""
    config_file = temp_dir / "malformed.yaml"
    config_file.write_text("invalid: yaml: content: [")

    # Should not crash, should use defaults
    config = load_config(config_file)
    assert config.app_name == "Video Analyser"


@pytest.mark.parametrize(
    "model_name",
    [
        "whisper-tiny",
        "whisper-base",
        "whisper-small",
        "whisper-medium",
        "whisper-large",
    ],
)
def test_valid_whisper_models(model_name: str) -> None:
    """Test that all valid Whisper models are accepted."""
    config = VideoAnalyserConfig(transcription={"model": model_name})
    assert config.transcription.model == model_name


def test_invalid_whisper_model() -> None:
    """Test that invalid Whisper models are rejected."""
    with pytest.raises(ValueError, match="Invalid model"):
        VideoAnalyserConfig(transcription={"model": "invalid-model"})


class TestConfigExport:
    """Tests for configuration export functionality."""

    def test_export_to_json(self, temp_dir: Path) -> None:
        """Test exporting configuration to JSON format."""
        config = VideoAnalyserConfig(app_name="Test Export", debug=True)
        export_path = temp_dir / "config.json"

        export_config_to_json(config, export_path)

        assert export_path.exists()
        with open(export_path) as f:
            exported_data = json.load(f)

        assert exported_data["app_name"] == "Test Export"
        assert exported_data["debug"] is True

    def test_export_to_env(self, temp_dir: Path) -> None:
        """Test exporting configuration to .env format."""
        config = VideoAnalyserConfig(app_name="Test Env", debug=True)
        export_path = temp_dir / ".env"

        export_config_to_env(config, export_path)

        assert export_path.exists()
        content = export_path.read_text()

        assert "VIDEO_ANALYSER__APP_NAME" in content
        assert "Test Env" in content
        assert "VIDEO_ANALYSER__DEBUG" in content
        assert "true" in content

    def test_export_env_handles_nested_config(self, temp_dir: Path) -> None:
        """Test that env export properly flattens nested config."""
        config = VideoAnalyserConfig(
            processing={"max_video_size_mb": 250},
            transcription={"model": "whisper-small"},
        )
        export_path = temp_dir / ".env"

        export_config_to_env(config, export_path)

        content = export_path.read_text()
        assert "VIDEO_ANALYSER__PROCESSING__MAX_VIDEO_SIZE_MB" in content
        assert "VIDEO_ANALYSER__TRANSCRIPTION__MODEL" in content

    def test_export_env_handles_lists(self, temp_dir: Path) -> None:
        """Test that env export properly handles list values."""
        config = VideoAnalyserConfig(
            processing={"supported_formats": ["mp4", "mov", "avi"]}
        )
        export_path = temp_dir / ".env"

        export_config_to_env(config, export_path)

        content = export_path.read_text()
        assert "VIDEO_ANALYSER__PROCESSING__SUPPORTED_FORMATS" in content
        assert "mp4,mov,avi" in content

    def test_create_default_config_file(self, temp_dir: Path) -> None:
        """Test creating a default configuration file."""
        config_path = temp_dir / "default_config.yaml"

        create_default_config_file(config_path)

        assert config_path.exists()
        loaded_config = load_config(config_path)
        assert loaded_config.app_name == "Video Analyser"


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_validate_valid_config(self, temp_dir: Path) -> None:
        """Test that valid config passes validation."""
        config = VideoAnalyserConfig(
            processing={"temp_dir": temp_dir / "temp"},
            logging={"file_path": temp_dir / "logs" / "test.log"},
        )

        is_valid, errors = validate_config(config)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_invalid_wpm_range(self) -> None:
        """Test validation catches invalid WPM range."""
        # Should fail at creation due to validator
        with pytest.raises(ValueError, match="WPM min must be less than max"):
            VideoAnalyserConfig(analysis={"target_wpm_range": [160, 140]})

    def test_validate_weights_sum(self) -> None:
        """Test validation of quality weights."""
        # Valid: weights sum to 1.0
        config = VideoAnalyserConfig(
            visual_analysis={
                "blur_weight": 0.4,
                "contrast_weight": 0.3,
                "brightness_weight": 0.3,
            }
        )
        is_valid, errors = validate_config(config)
        assert is_valid is True

    def test_validate_empty_output_formats(self) -> None:
        """Test validation catches empty output formats."""
        # Should fail at creation due to validator
        with pytest.raises(ValueError, match="List should have at least 1 item"):
            VideoAnalyserConfig(output={"formats": []})


class TestConfigSchema:
    """Tests for configuration schema functionality."""

    def test_get_config_schema(self) -> None:
        """Test that config schema can be retrieved."""
        schema = get_config_schema()

        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "app_name" in schema["properties"]
        assert "processing" in schema["properties"]
        assert "logging" in schema["properties"]

    def test_schema_includes_all_sections(self) -> None:
        """Test that schema includes all config sections."""
        schema = get_config_schema()
        properties = schema["properties"]

        required_sections = [
            "app_name",
            "version",
            "debug",
            "processing",
            "scene_detection",
            "audio",
            "transcription",
            "analysis",
            "visual_analysis",
            "output",
            "logging",
        ]

        for section in required_sections:
            assert section in properties


class TestConfigGlobalInstance:
    """Tests for global config instance management."""

    def test_get_config_returns_instance(self) -> None:
        """Test that get_config returns a config instance."""
        config = get_config()

        assert isinstance(config, VideoAnalyserConfig)
        assert config.app_name == "Video Analyser"

    def test_get_config_caches_instance(self) -> None:
        """Test that get_config caches the instance."""
        config1 = get_config()
        config2 = get_config()

        # Should be the same instance
        assert config1 is config2

    def test_reset_config_to_defaults(self) -> None:
        """Test resetting configuration to defaults."""
        # Reset configuration
        reset_config = reset_config_to_defaults()

        assert reset_config.app_name == "Video Analyser"
        assert reset_config.debug is False


class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_round_trip_save_and_load(self, temp_dir: Path) -> None:
        """Test saving and loading configuration preserves values."""
        original_config = VideoAnalyserConfig(
            app_name="Integration Test",
            debug=True,
            processing={"max_video_size_mb": 300},
            transcription={"model": "whisper-small"},
        )

        config_path = temp_dir / "integration_test.yaml"
        save_config(original_config, config_path)
        loaded_config = load_config(config_path)

        assert loaded_config.app_name == "Integration Test"
        assert loaded_config.debug is True
        assert loaded_config.processing.max_video_size_mb == 300
        assert loaded_config.transcription.model == "whisper-small"

    def test_export_and_reimport_json(self, temp_dir: Path) -> None:
        """Test exporting to JSON and reimporting."""
        original_config = VideoAnalyserConfig(
            app_name="JSON Test", debug=True, processing={"max_video_size_mb": 200}
        )

        # Export to JSON
        json_path = temp_dir / "config.json"
        export_config_to_json(original_config, json_path)

        # Load JSON and create config from it
        with open(json_path) as f:
            data = json.load(f)

        reimported_config = VideoAnalyserConfig(**data)

        assert reimported_config.app_name == "JSON Test"
        assert reimported_config.debug is True
        assert reimported_config.processing.max_video_size_mb == 200

    def test_yaml_and_json_format_consistency(self, temp_dir: Path) -> None:
        """Test that YAML and JSON exports contain equivalent data."""
        config = VideoAnalyserConfig(
            app_name="Format Test",
            debug=True,
            processing={"max_video_size_mb": 250},
        )

        yaml_path = temp_dir / "config.yaml"
        json_path = temp_dir / "config.json"

        save_config(config, yaml_path)
        export_config_to_json(config, json_path)

        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)

        with open(json_path) as f:
            json_data = json.load(f)

        # Both should have same basic structure
        assert yaml_data["app_name"] == json_data["app_name"]
        assert yaml_data["debug"] == json_data["debug"]
        assert (
            yaml_data["processing"]["max_video_size_mb"]
            == json_data["processing"]["max_video_size_mb"]
        )
