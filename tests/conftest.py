"""Pytest configuration and shared fixtures."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

# Reset global config before tests
from video_analyser.utils.config import reload_config

reload_config()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_video_path() -> Path:
    """Path to a sample video file for testing."""
    # Return path to minimal test video
    fixtures_path = Path(__file__).parent / "fixtures"
    video_path = fixtures_path / "minimal_test.mp4"

    if not video_path.exists():
        pytest.skip(
            f"Test video not found: {video_path}. Run scripts/generate_test_videos.py first."
        )

    return video_path


@pytest.fixture
def sample_video_no_audio() -> Path:
    """Path to a sample video file without audio."""
    fixtures_path = Path(__file__).parent / "fixtures"
    video_path = fixtures_path / "no_audio_test.mp4"

    if not video_path.exists():
        pytest.skip(
            f"Test video not found: {video_path}. Run scripts/generate_test_videos.py first."
        )

    return video_path


@pytest.fixture
def sample_audio_path() -> Path:
    """Path to a sample audio file for testing."""
    # For now, we'll extract audio from our test video in tests that need it
    return Path("tests/fixtures/sample_audio.wav")


@pytest.fixture
def video_formats_for_testing() -> list[Path]:
    """List of video files in different formats for testing."""
    fixtures_path = Path(__file__).parent / "fixtures"
    formats = ["mp4", "webm", "avi"]
    video_paths = []

    for fmt in formats:
        video_path = fixtures_path / f"different_format_test.{fmt}"
        if video_path.exists():
            video_paths.append(video_path)

    if not video_paths:
        pytest.skip("No test videos found. Run scripts/generate_test_videos.py first.")

    return video_paths


@pytest.fixture
def large_video_sample() -> Path:
    """Path to a larger video sample for development testing."""
    samples_path = Path(__file__).parent.parent / "samples"
    video_path = samples_path / "dev_sample_good.mp4"

    if not video_path.exists():
        pytest.skip(
            f"Development sample not found: {video_path}. Run scripts/generate_test_videos.py first."
        )

    return video_path


@pytest.fixture
def config_dict() -> dict:
    """Sample configuration dictionary for testing."""
    return {
        "processing": {
            "max_video_size_mb": 500,
            "supported_formats": ["mp4", "mov", "avi", "webm"],
        },
        "scene_detection": {
            "method": "threshold",
            "threshold": 0.4,
            "min_scene_duration": 2.0,
            "fallback_interval": 30.0,
        },
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
        },
        "transcription": {
            "model": "whisper-base",
            "language": "auto",
            "word_timestamps": True,
        },
        "analysis": {
            "filler_words": ["um", "uh", "like", "you know", "so"],
            "target_wpm_range": [140, 160],
        },
        "output": {
            "formats": ["json", "html"],
            "include_frames": True,
            "frame_quality": 80,
        },
    }


@pytest.fixture
def mock_video_metadata() -> dict:
    """Mock video metadata for testing."""
    return {
        "filename": "test_video.mp4",
        "duration": 10.0,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "audio_channels": 2,
        "audio_sample_rate": 48000,
        "file_size": 1024 * 1024,  # 1MB
    }


@pytest.fixture
def mock_analysis_result() -> dict:
    """Mock analysis result for testing."""
    return {
        "metadata": {
            "filename": "test_video.mp4",
            "duration": 10.0,
            "processed_at": "2025-06-24T10:30:00Z",
        },
        "scenes": [
            {
                "id": 1,
                "start_time": 0.0,
                "end_time": 5.0,
                "transcript": "Hello everyone, welcome to our presentation.",
                "speaking_rate": 145,
                "filler_words": {"um": 0, "uh": 1},
                "sentiment": "positive",
            },
            {
                "id": 2,
                "start_time": 5.0,
                "end_time": 10.0,
                "transcript": "Today we'll discuss our quarterly results.",
                "speaking_rate": 150,
                "filler_words": {"um": 0, "uh": 0},
                "sentiment": "neutral",
            },
        ],
        "summary": {
            "total_words": 156,
            "average_wpm": 147.5,
            "total_filler_words": 1,
            "overall_sentiment": "positive",
        },
    }


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    """Configure pytest.

    Marker registration lives in pyproject.toml's [tool.pytest.ini_options]
    section so a single source of truth governs the family-wide --strict-markers
    contract. This hook is preserved as a no-op so future per-session config
    has a place to land without re-introducing the duplicate-registration trap.
    """


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:  # noqa: ARG001
    """Automatically mark tests based on their location and content."""
    for item in items:
        # Mark tests in certain directories
        if "video" in str(item.fspath):
            item.add_marker(pytest.mark.video)
        if "audio" in str(item.fspath):
            item.add_marker(pytest.mark.audio)
        if "analysis" in str(item.fspath):
            item.add_marker(pytest.mark.ai)

        # Mark tests that use certain fixtures as requiring video files
        if hasattr(item, "fixturenames") and any(
            fixture in item.fixturenames
            for fixture in [
                "sample_video_path",
                "video_formats_for_testing",
                "large_video_sample",
            ]
        ):
            item.add_marker(pytest.mark.video)


# Pytest plugins for better video testing
pytest_plugins = [
    # Add any video-specific pytest plugins here in the future
]
