"""Tests for audio extractor functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import ffmpeg
import pytest

from video_analyser.core.audio_extractor import AudioExtractor, AudioInfo
from video_analyser.core.exceptions import AudioProcessingError, ErrorCode, FFmpegError
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
def mock_probe_data():
    """Mock ffmpeg probe data for audio."""
    return {
        "streams": [
            {"codec_type": "video", "codec_name": "h264"},
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
            },
        ],
        "format": {"duration": "120.0", "format_name": "mp4"},
    }


@pytest.fixture
def mock_audio_probe_data():
    """Mock ffmpeg probe data for extracted audio."""
    return {
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "pcm_s16le",
                "sample_rate": "16000",
                "channels": 1,
            }
        ],
        "format": {"duration": "120.0", "format_name": "wav"},
    }


class TestAudioExtractor:
    """Test AudioExtractor class initialization and configuration."""

    def test_init_creates_temp_directory(self, mock_config):
        """Test that AudioExtractor creates temp directory on init."""
        extractor = AudioExtractor(config=mock_config)
        assert extractor.temp_dir.exists()
        assert extractor.config.audio.sample_rate == 16000

    def test_get_audio_config(self, audio_extractor):
        """Test getting audio configuration."""
        config = audio_extractor.get_audio_config()

        assert config["sample_rate"] == 16000
        assert config["channels"] == 1
        assert config["noise_reduction"] is False
        assert config["normalize_audio"] is True


class TestAudioExtraction:
    """Test audio extraction functionality."""

    @patch("ffmpeg.probe")
    @patch("ffmpeg.run")
    def test_extract_audio_success(
        self,
        mock_run,
        mock_probe,
        audio_extractor,
        mock_video_info,
        mock_probe_data,
        mock_audio_probe_data,
    ):
        """Test successful audio extraction."""
        # Mock ffmpeg.probe calls (video probe, then audio probe)
        mock_probe.side_effect = [mock_probe_data, mock_audio_probe_data]

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

        # Verify ffmpeg was called
        mock_run.assert_called_once()

    @patch("ffmpeg.probe")
    def test_extract_audio_no_audio_stream(
        self, mock_probe, audio_extractor, mock_video_info
    ):
        """Test extraction from video with no audio stream."""
        # Mock probe data with no audio stream
        mock_probe.return_value = {
            "streams": [{"codec_type": "video", "codec_name": "h264"}],
            "format": {"duration": "120.0", "format_name": "mp4"},
        }

        with pytest.raises(AudioProcessingError) as exc_info:
            audio_extractor.extract_audio(mock_video_info)

        assert exc_info.value.error_code == ErrorCode.NO_AUDIO_STREAM

    @patch("ffmpeg.probe")
    @patch("ffmpeg.run")
    def test_extract_audio_ffmpeg_error(
        self, mock_run, mock_probe, audio_extractor, mock_video_info, mock_probe_data
    ):
        """Test handling of ffmpeg errors during extraction."""
        mock_probe.return_value = mock_probe_data

        # Create proper ffmpeg.Error mock
        error = ffmpeg.Error("ffmpeg", "stdout", "stderr_output")
        error.stderr = b"ffmpeg error output"
        error.stdout = b"ffmpeg stdout output"
        error.returncode = 1
        error.cmd = ["ffmpeg", "-i", "input.mp4"]
        mock_run.side_effect = error

        with pytest.raises(FFmpegError) as exc_info:
            audio_extractor.extract_audio(mock_video_info)

        assert exc_info.value.error_code == ErrorCode.FFMPEG_ERROR

    @patch("ffmpeg.probe")
    @patch("ffmpeg.run")
    def test_extract_audio_custom_output_path(
        self,
        mock_run,  # noqa: ARG002
        mock_probe,
        audio_extractor,
        mock_video_info,
        mock_probe_data,
        mock_audio_probe_data,
        tmp_path,
    ):
        """Test extraction with custom output path."""
        custom_output = tmp_path / "custom_audio.wav"

        # Mock probes
        mock_probe.side_effect = [mock_probe_data, mock_audio_probe_data]

        # Create mock output file
        custom_output.write_text("fake audio data")

        result = audio_extractor.extract_audio(
            mock_video_info, output_path=custom_output
        )

        assert result.file_path == custom_output

    @patch("ffmpeg.probe")
    @patch("ffmpeg.run")
    def test_extract_audio_with_noise_reduction(
        self,
        mock_run,
        mock_probe,
        mock_config,
        mock_video_info,
        mock_probe_data,
        mock_audio_probe_data,
    ):
        """Test extraction with noise reduction enabled."""
        mock_config.audio.noise_reduction = True
        extractor = AudioExtractor(config=mock_config)

        mock_probe.side_effect = [mock_probe_data, mock_audio_probe_data]

        # Mock output file
        expected_output = (
            extractor.temp_dir / f"{mock_video_info.file_path.stem}_audio.wav"
        )
        expected_output.write_text("fake audio data")

        extractor.extract_audio(mock_video_info)

        # Verify ffmpeg was called (noise reduction should be applied)
        mock_run.assert_called_once()


class TestAudioSegmentExtraction:
    """Test audio segment extraction functionality."""

    @patch("ffmpeg.probe")
    @patch("ffmpeg.run")
    def test_extract_audio_segment_success(
        self,
        mock_run,
        mock_probe,
        audio_extractor,
        mock_video_info,
        mock_audio_probe_data,
    ):
        """Test successful audio segment extraction."""
        mock_probe.return_value = mock_audio_probe_data

        # Mock output file
        expected_output = (
            audio_extractor.temp_dir
            / f"{mock_video_info.file_path.stem}_segment_10.0s.wav"
        )
        expected_output.write_text("fake audio segment")

        result = audio_extractor.extract_audio_segment(
            mock_video_info, start_time=10.0, duration=30.0
        )

        assert isinstance(result, AudioInfo)
        assert result.file_path == expected_output
        mock_run.assert_called_once()

    def test_extract_audio_segment_invalid_start_time(
        self, audio_extractor, mock_video_info
    ):
        """Test segment extraction with invalid start time."""
        with pytest.raises(ValueError, match="Start time cannot be negative"):
            audio_extractor.extract_audio_segment(
                mock_video_info, start_time=-5.0, duration=10.0
            )

    def test_extract_audio_segment_invalid_duration(
        self, audio_extractor, mock_video_info
    ):
        """Test segment extraction with invalid duration."""
        with pytest.raises(ValueError, match="Duration must be positive"):
            audio_extractor.extract_audio_segment(
                mock_video_info, start_time=10.0, duration=0.0
            )

    def test_extract_audio_segment_extends_beyond_video(
        self, audio_extractor, mock_video_info
    ):
        """Test segment extraction that extends beyond video duration."""
        with pytest.raises(ValueError, match="Segment extends beyond video duration"):
            audio_extractor.extract_audio_segment(
                mock_video_info,
                start_time=100.0,
                duration=50.0,  # 150s total, but video is only 120s
            )


class TestAudioInfo:
    """Test AudioInfo extraction and metadata."""

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
            "streams": [{"codec_type": "video"}],
            "format": {"duration": "60.0", "format_name": "wav"},
        }

        with pytest.raises(RuntimeError, match="No audio stream found"):
            audio_extractor._get_audio_info(audio_file)

    @patch("ffmpeg.probe")
    def test_get_audio_info_probe_error(self, mock_probe, audio_extractor, tmp_path):
        """Test audio info when probe fails."""
        audio_file = tmp_path / "corrupted.wav"
        audio_file.write_text("corrupted content")

        mock_probe.side_effect = ffmpeg.Error("ffmpeg", "stdout", "stderr")

        with pytest.raises(RuntimeError, match="Failed to get audio metadata"):
            audio_extractor._get_audio_info(audio_file)


class TestProgressTracking:
    """Test progress tracking functionality."""

    def test_run_with_progress_success(self, audio_extractor):
        """Test progress tracking with successful execution."""
        mock_stream = MagicMock()
        progress_callback = MagicMock()

        # Mock process
        mock_process = MagicMock()
        mock_process.poll.side_effect = [None, None, None, 0]  # Running, then complete

        # Create a function that returns empty bytes indefinitely after the real data
        readline_values = [
            b"time=00:00:30.00 bitrate=N/A speed= 1.0x\n",
            b"time=00:01:00.00 bitrate=N/A speed= 1.0x\n",
        ]
        readline_index = 0

        def mock_readline():
            nonlocal readline_index
            if readline_index < len(readline_values):
                result = readline_values[readline_index]
                readline_index += 1
                return result
            return b""  # Return empty bytes forever after

        mock_process.stderr.readline.side_effect = mock_readline
        mock_process.wait.return_value = None
        mock_process.returncode = 0

        with patch("ffmpeg.run_async", return_value=mock_process):
            audio_extractor._run_with_progress(mock_stream, 120.0, progress_callback)

        # Should call progress callback for each time update plus final
        assert progress_callback.call_count >= 1
        # Final call should be 1.0 (100% complete)
        progress_callback.assert_called_with(1.0)

    def test_run_with_progress_ffmpeg_error(self, audio_extractor):
        """Test progress tracking with ffmpeg error."""
        mock_stream = MagicMock()
        progress_callback = MagicMock()

        mock_process = MagicMock()
        mock_process.poll.return_value = 1  # Error return code
        mock_process.stderr.readline.return_value = b""
        mock_process.wait.return_value = None
        mock_process.returncode = 1
        mock_process.stderr.read.return_value = b"ffmpeg error"

        with (
            patch("ffmpeg.run_async", return_value=mock_process),
            pytest.raises(ffmpeg.Error),
        ):
            audio_extractor._run_with_progress(mock_stream, 120.0, progress_callback)


class TestTempFileCleanup:
    """Test temporary file cleanup functionality."""

    def test_cleanup_temp_audio_files_enabled(self, audio_extractor):
        """Test cleanup when enabled."""
        # Create some temp audio files
        temp_files = [
            audio_extractor.temp_dir / "temp1.wav",
            audio_extractor.temp_dir / "temp2.mp3",
            audio_extractor.temp_dir / "temp3.aac",
        ]

        for temp_file in temp_files:
            temp_file.write_text("temp audio content")

        # Verify files exist
        for temp_file in temp_files:
            assert temp_file.exists()

        audio_extractor.cleanup_temp_audio_files()

        # Verify files are cleaned up
        for temp_file in temp_files:
            assert not temp_file.exists()

    def test_cleanup_temp_audio_files_disabled(self, mock_config):
        """Test cleanup when disabled."""
        mock_config.processing.cleanup_temp_files = False
        extractor = AudioExtractor(config=mock_config)

        # Create temp audio file
        temp_file = extractor.temp_dir / "temp.wav"
        temp_file.write_text("temp audio content")

        extractor.cleanup_temp_audio_files()

        # File should still exist
        assert temp_file.exists()

    def test_cleanup_nonexistent_temp_dir(self, audio_extractor):
        """Test cleanup when temp directory doesn't exist."""
        audio_extractor.temp_dir.rmdir()  # Remove temp directory

        # Should not raise an error
        audio_extractor.cleanup_temp_audio_files()

    def test_cleanup_handles_permission_errors(self, audio_extractor, caplog):
        """Test cleanup handles permission errors gracefully."""
        temp_file = audio_extractor.temp_dir / "readonly.wav"
        temp_file.write_text("content")

        # Mock unlink to raise permission error
        with patch.object(Path, "unlink", side_effect=PermissionError("Access denied")):
            audio_extractor.cleanup_temp_audio_files()

        # Should log warning but not crash
        assert "Could not remove temp audio file" in caplog.text
