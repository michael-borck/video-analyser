"""Enhanced comprehensive tests for audio extraction functionality.

This test suite focuses on improving test coverage for core audio extraction
functionality and edge cases not covered by existing tests.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import ffmpeg
import pytest

from video_analyser.core.audio_extractor import AudioExtractor, AudioInfo
from video_analyser.core.exceptions import (
    AudioProcessingError,
    ErrorCode,
)
from video_analyser.core.video_processor import VideoInfo
from video_analyser.utils.config import AudioConfig, VideoAnalyserConfig, ProcessingConfig


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = VideoAnalyserConfig(
        processing=ProcessingConfig(
            temp_dir=Path(tempfile.mkdtemp()), cleanup_temp_files=True
        ),
        audio=AudioConfig(
            sample_rate=16000, channels=1, noise_reduction=False, normalize_audio=True
        ),
    )
    return config


@pytest.fixture
def audio_extractor(mock_config):
    """Create AudioExtractor instance for testing."""
    return AudioExtractor(config=mock_config)


@pytest.fixture
def mock_video_info(tmp_path):
    """Create mock VideoInfo for testing."""
    video_file = tmp_path / "test_video.mp4"
    video_file.write_text("fake video content")

    return VideoInfo(
        file_path=video_file,
        duration=120.0,
        width=1920,
        height=1080,
        fps=30.0,
        format="mp4",
        size_mb=50.0,
        codec="h264",
    )


@pytest.fixture
def mock_probe_data_with_audio():
    """Mock ffmpeg probe data with audio stream."""
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30/1",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "44100",
                "channels": "2",
            },
        ],
        "format": {
            "duration": "120.0",
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        },
    }


@pytest.fixture
def mock_probe_data_no_audio():
    """Mock ffmpeg probe data without audio stream."""
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30/1",
            }
        ],
        "format": {
            "duration": "120.0",
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        },
    }


@pytest.fixture
def mock_audio_probe_data():
    """Mock ffmpeg probe data for extracted audio file."""
    return {
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "pcm_s16le",
                "sample_rate": "16000",
                "channels": "1",
            }
        ],
        "format": {"duration": "120.0", "format_name": "wav"},
    }


class TestAudioExtractorInitialization:
    """Test AudioExtractor initialization and configuration."""

    def test_initialization_with_default_config(self):
        """Test extractor initialization with default config."""
        extractor = AudioExtractor()
        assert extractor.config is not None
        assert extractor.temp_dir.exists()

    def test_initialization_with_custom_config(self, mock_config):
        """Test extractor initialization with custom config."""
        extractor = AudioExtractor(config=mock_config)
        assert extractor.config == mock_config
        assert extractor.temp_dir == mock_config.processing.temp_dir

    def test_temp_directory_creation(self, mock_config):
        """Test that temp directory is created on initialization."""
        # Use a non-existent temp dir
        temp_path = Path(tempfile.mkdtemp()) / "nested" / "temp"
        mock_config.processing.temp_dir = temp_path

        extractor = AudioExtractor(config=mock_config)
        assert extractor.temp_dir.exists()
        assert extractor.temp_dir == temp_path

    def test_get_audio_config(self, audio_extractor):
        """Test getting audio configuration."""
        config = audio_extractor.get_audio_config()

        assert config["sample_rate"] == 16000
        assert config["channels"] == 1
        assert config["noise_reduction"] is False
        assert config["normalize_audio"] is True


class TestAudioExtractionCore:
    """Test core audio extraction functionality."""

    @patch("ffmpeg.probe")
    @patch("ffmpeg.run")
    def test_extract_audio_success_default_path(
        self,
        _mock_run,
        mock_probe,
        audio_extractor,
        mock_video_info,
        mock_probe_data_with_audio,
        mock_audio_probe_data,
    ):
        """Test successful audio extraction with default output path."""
        # Mock ffmpeg.probe calls (video probe, then audio probe)
        mock_probe.side_effect = [mock_probe_data_with_audio, mock_audio_probe_data]

        # Mock the output file creation
        expected_output = (
            audio_extractor.temp_dir / f"{mock_video_info.file_path.stem}_audio.wav"
        )
        expected_output.write_text("fake audio data")

        result = audio_extractor.extract_audio(mock_video_info)

        # Verify result
        assert isinstance(result, AudioInfo)
        assert result.file_path == expected_output
        assert result.duration == 120.0
        assert result.sample_rate == 16000
        assert result.channels == 1
        assert result.format == "wav"

    @patch("ffmpeg.probe")
    @patch("ffmpeg.run")
    def test_extract_audio_custom_output_path(
        self,
        _mock_run,
        mock_probe,
        audio_extractor,
        mock_video_info,
        mock_probe_data_with_audio,
        mock_audio_probe_data,
        tmp_path,
    ):
        """Test audio extraction with custom output path."""
        custom_output = tmp_path / "custom_audio.wav"
        mock_probe.side_effect = [mock_probe_data_with_audio, mock_audio_probe_data]
        custom_output.write_text("fake audio data")

        result = audio_extractor.extract_audio(
            mock_video_info, output_path=custom_output
        )

        assert result.file_path == custom_output

    @patch("ffmpeg.probe")
    def test_extract_audio_no_audio_stream(
        self, mock_probe, audio_extractor, mock_video_info, mock_probe_data_no_audio
    ):
        """Test extraction from video with no audio stream."""
        mock_probe.return_value = mock_probe_data_no_audio

        with pytest.raises(AudioProcessingError) as exc_info:
            audio_extractor.extract_audio(mock_video_info)

        assert exc_info.value.error_code == ErrorCode.NO_AUDIO_STREAM

    @patch("ffmpeg.probe")
    def test_extract_audio_invalid_metadata(
        self, mock_probe, audio_extractor, mock_video_info
    ):
        """Test extraction with invalid audio metadata."""
        # Mock probe data with invalid audio metadata
        mock_probe.return_value = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "invalid",  # Invalid sample rate
                    "channels": "also_invalid",  # Invalid channels
                },
            ],
            "format": {"duration": "120.0"},
        }

        with pytest.raises(AudioProcessingError) as exc_info:
            audio_extractor.extract_audio(mock_video_info)

        assert exc_info.value.error_code == ErrorCode.AUDIO_CODEC_ERROR

    @patch("ffmpeg.probe")
    def test_extract_audio_zero_audio_params(
        self, mock_probe, audio_extractor, mock_video_info
    ):
        """Test extraction with zero audio parameters."""
        # Mock probe data with zero sample rate/channels
        mock_probe.return_value = {
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "0",  # Zero sample rate
                    "channels": "0",  # Zero channels
                }
            ],
            "format": {"duration": "120.0"},
        }

        with pytest.raises(AudioProcessingError) as exc_info:
            audio_extractor.extract_audio(mock_video_info)

        assert exc_info.value.error_code == ErrorCode.AUDIO_CODEC_ERROR

    @patch("ffmpeg.probe")
    def test_extract_audio_probe_error(
        self, mock_probe, audio_extractor, mock_video_info
    ):
        """Test handling of ffmpeg probe errors."""
        import ffmpeg

        # Create proper ffmpeg.Error mock
        error = ffmpeg.Error("ffmpeg", "stdout", "stderr")
        error.stderr = b"Invalid data found"
        error.stdout = b""
        error.returncode = 1
        error.cmd = ["ffmpeg", "-i", "input.mp4"]
        mock_probe.side_effect = error

        # The outer exception handler wraps FFmpegError in AudioProcessingError
        with pytest.raises(AudioProcessingError) as exc_info:
            audio_extractor.extract_audio(mock_video_info)

        # Verify it's wrapping an ffmpeg error
        assert "ffmpeg" in str(exc_info.value).lower()

    @patch("ffmpeg.probe")
    @patch("ffmpeg.run")
    def test_extract_audio_output_directory_creation_error(
        self,
        _mock_run,
        mock_probe,
        audio_extractor,
        mock_video_info,
        mock_probe_data_with_audio,
        tmp_path,
    ):
        """Test handling of output directory creation errors."""
        mock_probe.return_value = mock_probe_data_with_audio

        # Use invalid output path (file exists where directory should be)
        invalid_output = tmp_path / "file.txt" / "audio.wav"
        (tmp_path / "file.txt").write_text("blocking file")

        with pytest.raises(AudioProcessingError) as exc_info:
            audio_extractor.extract_audio(mock_video_info, output_path=invalid_output)

        # Should fail on directory creation
        assert "Cannot create output directory" in str(exc_info.value)


class TestAudioExtractionAdvanced:
    """Test advanced audio extraction features."""

    @patch("ffmpeg.probe")
    @patch("ffmpeg.run")
    def test_extract_audio_with_noise_reduction(
        self,
        mock_run,
        mock_probe,
        mock_config,
        mock_video_info,
        mock_probe_data_with_audio,
        mock_audio_probe_data,
    ):
        """Test audio extraction with noise reduction enabled."""
        mock_config.audio.noise_reduction = True
        extractor = AudioExtractor(config=mock_config)

        mock_probe.side_effect = [mock_probe_data_with_audio, mock_audio_probe_data]

        # Mock the output file creation
        expected_output = (
            extractor.temp_dir / f"{mock_video_info.file_path.stem}_audio.wav"
        )
        expected_output.write_text("fake audio data")

        result = extractor.extract_audio(mock_video_info)

        # Verify ffmpeg was called (noise reduction should be applied)
        assert mock_run.called
        assert isinstance(result, AudioInfo)

    @patch("ffmpeg.probe")
    @patch("ffmpeg.run")
    def test_extract_audio_with_normalization_disabled(
        self,
        _mock_run,
        mock_probe,
        mock_config,
        mock_video_info,
        mock_probe_data_with_audio,
        mock_audio_probe_data,
    ):
        """Test audio extraction with normalization disabled."""
        mock_config.audio.normalize_audio = False
        extractor = AudioExtractor(config=mock_config)

        mock_probe.side_effect = [mock_probe_data_with_audio, mock_audio_probe_data]

        # Mock the output file creation
        expected_output = (
            extractor.temp_dir / f"{mock_video_info.file_path.stem}_audio.wav"
        )
        expected_output.write_text("fake audio data")

        result = extractor.extract_audio(mock_video_info)

        # Verify extraction succeeded without normalization
        assert isinstance(result, AudioInfo)

    @patch("ffmpeg.probe")
    @patch("ffmpeg.run")
    def test_extract_audio_timeout_handling(
        self,
        mock_run,
        mock_probe,
        audio_extractor,
        mock_video_info,
        mock_probe_data_with_audio,
    ):
        """Test handling of extraction timeout."""
        import subprocess

        mock_probe.return_value = mock_probe_data_with_audio
        mock_run.side_effect = subprocess.TimeoutExpired("ffmpeg", 300)

        with pytest.raises(AudioProcessingError) as exc_info:
            audio_extractor.extract_audio(mock_video_info)

        assert "timed out" in str(exc_info.value)


class TestAudioSegmentExtraction:
    """Test audio segment extraction functionality."""

    @patch("ffmpeg.probe")
    @patch("ffmpeg.run")
    def test_extract_audio_segment_success(
        self,
        _mock_run,
        mock_probe,
        audio_extractor,
        mock_video_info,
        mock_probe_data_with_audio,
        mock_audio_probe_data,
    ):
        """Test successful audio segment extraction."""
        mock_probe.side_effect = [mock_probe_data_with_audio, mock_audio_probe_data]

        # Mock the output file creation (use correct filename format from implementation)
        expected_output = (
            audio_extractor.temp_dir
            / f"{mock_video_info.file_path.stem}_segment_30.0s.wav"
        )
        expected_output.write_text("fake audio data")

        result = audio_extractor.extract_audio_segment(
            video_info=mock_video_info, start_time=30.0, duration=30.0
        )

        assert isinstance(result, AudioInfo)
        assert result.duration == 120.0  # From mock_audio_probe_data

    def test_extract_audio_segment_invalid_start_time(
        self, audio_extractor, mock_video_info
    ):
        """Test segment extraction with invalid start time."""
        with pytest.raises(ValueError, match="Start time cannot be negative"):
            audio_extractor.extract_audio_segment(
                video_info=mock_video_info, start_time=-5.0, duration=30.0
            )

    def test_extract_audio_segment_invalid_duration(
        self, audio_extractor, mock_video_info
    ):
        """Test segment extraction with invalid duration."""
        with pytest.raises(ValueError, match="Duration must be positive"):
            audio_extractor.extract_audio_segment(
                video_info=mock_video_info, start_time=30.0, duration=-10.0
            )

    def test_extract_audio_segment_extends_beyond_video(
        self, audio_extractor, mock_video_info
    ):
        """Test segment extraction that extends beyond video duration."""
        # This should be handled gracefully - either clip to video end or raise error
        # The exact behavior depends on implementation
        with pytest.raises((ValueError, AudioProcessingError)):
            audio_extractor.extract_audio_segment(
                video_info=mock_video_info,
                start_time=100.0,
                duration=50.0,  # Extends beyond 120s video
            )


class TestAudioInfoExtraction:
    """Test AudioInfo extraction and metadata handling."""

    @patch("ffmpeg.probe")
    def test_get_audio_info_success(
        self, mock_probe, audio_extractor, mock_audio_probe_data, tmp_path
    ):
        """Test successful audio info extraction."""
        audio_file = tmp_path / "test_audio.wav"
        audio_file.write_text("fake audio content")

        mock_probe.return_value = mock_audio_probe_data

        result = audio_extractor._get_audio_info(audio_file)

        assert isinstance(result, AudioInfo)
        assert result.file_path == audio_file
        assert result.duration == 120.0
        assert result.sample_rate == 16000
        assert result.channels == 1
        assert result.format == "wav"

    @patch("ffmpeg.probe")
    def test_get_audio_info_no_audio_stream(
        self, mock_probe, audio_extractor, tmp_path
    ):
        """Test audio info when file has no audio stream."""
        audio_file = tmp_path / "no_audio.wav"
        audio_file.write_text("fake content")

        mock_probe.return_value = {
            "streams": [],  # No streams
            "format": {"duration": "0.0", "format_name": "wav"},
        }

        # _get_audio_info raises RuntimeError, not AudioProcessingError
        with pytest.raises(RuntimeError) as exc_info:
            audio_extractor._get_audio_info(audio_file)

        assert "No audio stream" in str(exc_info.value)

    @patch("ffmpeg.probe")
    def test_get_audio_info_probe_error(self, mock_probe, audio_extractor, tmp_path):
        """Test audio info when probe fails."""
        import ffmpeg

        audio_file = tmp_path / "corrupted_audio.wav"
        audio_file.write_text("fake content")

        error = ffmpeg.Error("ffmpeg", "stdout", "stderr")
        error.stderr = b"Invalid data"
        mock_probe.side_effect = error

        # _get_audio_info raises RuntimeError, not FFmpegError
        with pytest.raises(RuntimeError) as exc_info:
            audio_extractor._get_audio_info(audio_file)

        assert "Failed to get audio metadata" in str(exc_info.value)


class TestAudioFileCleanup:
    """Test temporary audio file cleanup functionality."""

    def test_cleanup_temp_audio_files_enabled(self, audio_extractor):
        """Test cleanup when enabled."""
        # Create some temp audio files
        temp_file1 = audio_extractor.temp_dir / "temp1.wav"
        temp_file2 = audio_extractor.temp_dir / "temp2.wav"
        temp_file1.write_text("temp content")
        temp_file2.write_text("temp content")

        # Ensure cleanup is enabled
        audio_extractor.config.processing.cleanup_temp_files = True

        audio_extractor.cleanup_temp_audio_files()

        # Only audio files should be removed
        assert not temp_file1.exists()
        assert not temp_file2.exists()

    def test_cleanup_temp_audio_files_disabled(self, audio_extractor):
        """Test cleanup when disabled."""
        temp_file = audio_extractor.temp_dir / "temp.wav"
        temp_file.write_text("temp content")

        # Disable cleanup
        audio_extractor.config.processing.cleanup_temp_files = False

        audio_extractor.cleanup_temp_audio_files()

        # File should still exist
        assert temp_file.exists()

    def test_cleanup_nonexistent_temp_dir(self, audio_extractor):
        """Test cleanup when temp directory doesn't exist."""
        import shutil

        shutil.rmtree(audio_extractor.temp_dir)

        # Should not raise an error
        audio_extractor.cleanup_temp_audio_files()

    def test_cleanup_handles_permission_errors(self, audio_extractor):
        """Test that cleanup handles permission errors gracefully."""
        temp_file = audio_extractor.temp_dir / "temp.wav"
        temp_file.write_text("temp content")

        # Mock permission error
        with patch.object(Path, "unlink", side_effect=OSError("Permission denied")):
            # Should not raise an error, just log warning
            audio_extractor.cleanup_temp_audio_files()


class TestProgressTracking:
    """Test progress tracking functionality."""

    @patch("ffmpeg.run_async")
    @patch("ffmpeg.probe")
    def test_run_with_progress_success(
        self, mock_probe, mock_run_async, audio_extractor, mock_probe_data_with_audio
    ):
        """Test progress tracking with successful execution."""
        mock_probe.return_value = mock_probe_data_with_audio

        # Mock the ffmpeg process
        mock_process = MagicMock()

        # Use a generator to handle multiple readline calls
        def readline_generator():
            yield b"time=00:01:00.00 bitrate=N/A speed= 0.5x\n"
            yield b"time=00:02:00.00 bitrate=N/A speed= 1.0x\n"
            # After real data, keep returning empty bytes
            while True:
                yield b""

        # poll() returns None while running, then 0 (success) when process completes
        mock_process.poll.side_effect = [None, None, 0]
        mock_process.stderr.readline.side_effect = readline_generator()
        mock_process.wait.return_value = 0
        mock_process.returncode = 0  # Success exit code
        mock_run_async.return_value = mock_process

        progress_callback = MagicMock()

        # Create a mock stream
        mock_stream = MagicMock()

        audio_extractor._run_with_progress(mock_stream, 120.0, progress_callback)

        # Verify progress callback was called
        assert progress_callback.called

    @patch("ffmpeg.run_async")
    def test_run_with_progress_ffmpeg_error(self, mock_run_async, audio_extractor):
        """Test progress tracking with ffmpeg error."""
        mock_process = MagicMock()
        # poll() should return None while running, then 1 (error) when complete
        mock_process.poll.side_effect = [None, None, 1]
        mock_process.stderr.readline.return_value = b""
        mock_process.wait.return_value = 1  # Error return code
        mock_process.returncode = 1  # Set return code for error check
        mock_run_async.return_value = mock_process

        progress_callback = MagicMock()
        mock_stream = MagicMock()

        with pytest.raises(ffmpeg.Error):
            audio_extractor._run_with_progress(mock_stream, 120.0, progress_callback)


class TestAudioExtractionIntegration:
    """Test integration scenarios for AudioExtractor."""

    def test_full_extraction_workflow(self, audio_extractor):
        """Test complete workflow from validation to extraction."""
        # This would test the integration between validation and extraction
        # but requires more complex mocking of ffmpeg operations
        pass

    def test_error_recovery_scenarios(self, audio_extractor):
        """Test error recovery in various scenarios."""
        # Test that extractor can recover from various error conditions
        pass
