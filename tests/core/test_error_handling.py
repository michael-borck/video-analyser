"""Tests for comprehensive error handling system."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from video_analyser.core.exceptions import (
    AudioProcessingError,
    ErrorCode,
    FFmpegError,
    FileValidationError,
    FrameExtractionError,
    VideoProcessingError,
    get_user_friendly_message,
    handle_ffmpeg_error,
)
from video_analyser.core.video_processor import VideoProcessor


class TestCustomExceptions:
    """Test custom exception classes."""

    def test_video_processing_error_creation(self):
        """Test creating VideoProcessingError."""
        error = VideoProcessingError(
            message="Test error",
            error_code=ErrorCode.FILE_NOT_FOUND,
            file_path="/test/video.mp4",
            details={"key": "value"},
            cause=ValueError("Original error"),
        )

        assert error.message == "Test error"
        assert error.error_code == ErrorCode.FILE_NOT_FOUND
        assert str(error.file_path) == "/test/video.mp4"
        assert error.details["key"] == "value"
        assert isinstance(error.cause, ValueError)

    def test_video_processing_error_to_dict(self):
        """Test converting error to dictionary."""
        error = VideoProcessingError(
            message="Test error",
            error_code=ErrorCode.INVALID_FORMAT,
            file_path="/test/video.mp4",
        )

        error_dict = error.to_dict()

        assert error_dict["message"] == "Test error"
        assert error_dict["error_code"] == "INVALID_FORMAT"
        assert error_dict["file_path"] == "/test/video.mp4"
        assert error_dict["details"] == {}
        assert error_dict["cause"] is None

    def test_video_processing_error_str(self):
        """Test string representation of error."""
        error = VideoProcessingError(
            message="Test error",
            error_code=ErrorCode.FILE_CORRUPTED,
            file_path="/test/video.mp4",
            details={"size": "10MB"},
            cause=RuntimeError("Root cause"),
        )

        error_str = str(error)

        assert "FILE_CORRUPTED: Test error" in error_str
        assert "File: /test/video.mp4" in error_str
        assert "size=10MB" in error_str
        assert "Caused by: Root cause" in error_str

    def test_file_validation_error(self):
        """Test FileValidationError creation."""
        error = FileValidationError(
            message="File not found",
            file_path="/missing/video.mp4",
            error_code=ErrorCode.FILE_NOT_FOUND,
        )

        assert isinstance(error, VideoProcessingError)
        assert error.error_code == ErrorCode.FILE_NOT_FOUND

    def test_ffmpeg_error(self):
        """Test FFmpegError creation."""
        error = FFmpegError(
            message="FFmpeg failed",
            command="ffmpeg -i input.mp4 output.wav",
            stderr="Error message",
            return_code=1,
            file_path="/test/video.mp4",
        )

        assert error.error_code == ErrorCode.FFMPEG_ERROR
        assert error.details["command"] == "ffmpeg -i input.mp4 output.wav"
        assert error.details["stderr"] == "Error message"
        assert error.details["return_code"] == 1

    def test_audio_processing_error(self):
        """Test AudioProcessingError creation."""
        error = AudioProcessingError(
            message="No audio stream",
            error_code=ErrorCode.NO_AUDIO_STREAM,
            file_path="/test/video.mp4",
        )

        assert isinstance(error, VideoProcessingError)
        assert error.error_code == ErrorCode.NO_AUDIO_STREAM

    def test_frame_extraction_error(self):
        """Test FrameExtractionError creation."""
        error = FrameExtractionError(
            message="Frame extraction failed",
            timestamp=15.5,
            scene_number=2,
            file_path="/test/video.mp4",
        )

        assert error.error_code == ErrorCode.FRAME_EXTRACTION_FAILED
        assert error.details["timestamp"] == 15.5
        assert error.details["scene_number"] == 2


class TestErrorHandleUtilities:
    """Test error handling utility functions."""

    def test_handle_ffmpeg_error(self):
        """Test ffmpeg error handling utility."""
        import ffmpeg

        # Create mock ffmpeg error
        mock_error = ffmpeg.Error("ffmpeg", "stdout", "stderr")
        mock_error.cmd = ["ffmpeg", "-i", "input.mp4"]
        mock_error.stdout = b"stdout output"
        mock_error.stderr = b"stderr output"
        mock_error.returncode = 1

        result = handle_ffmpeg_error(mock_error, "test operation", "/test/video.mp4")

        assert isinstance(result, FFmpegError)
        assert result.message == "FFmpeg test operation failed"
        assert result.details["command"] == ["ffmpeg", "-i", "input.mp4"]
        assert "stdout output" in result.details["stdout"]
        assert "stderr output" in result.details["stderr"]

    def test_get_user_friendly_message(self):
        """Test user-friendly message generation."""
        error = VideoProcessingError(
            message="Technical error message",
            error_code=ErrorCode.FILE_NOT_FOUND,
            file_path="/test/video.mp4",
        )

        friendly_msg = get_user_friendly_message(error)

        assert "could not be found" in friendly_msg.lower()
        assert "video.mp4" in friendly_msg

    def test_get_user_friendly_message_unknown_error(self):
        """Test user-friendly message for unknown error."""
        error = VideoProcessingError(
            message="Unknown technical error",
            error_code=ErrorCode.UNKNOWN_ERROR,
            file_path="/test/video.mp4",
        )

        friendly_msg = get_user_friendly_message(error)

        assert "Unknown technical error" in friendly_msg
        assert "video.mp4" in friendly_msg


class TestVideoProcessorErrorHandling:
    """Test error handling in VideoProcessor."""

    def test_validate_file_not_found(self):
        """Test file not found error."""
        processor = VideoProcessor()

        with pytest.raises(FileValidationError) as exc_info:
            processor.validate_file("/nonexistent/video.mp4")

        assert exc_info.value.error_code == ErrorCode.FILE_NOT_FOUND

    def test_validate_file_access_denied(self, tmp_path):
        """Test file access denied error."""
        processor = VideoProcessor()

        # Create a file and remove read permissions
        test_file = tmp_path / "test.mp4"
        test_file.write_text("test")

        # Mock os.access to return False
        with (
            patch("os.access", return_value=False),
            pytest.raises(FileValidationError) as exc_info,
        ):
            processor.validate_file(test_file)

        assert exc_info.value.error_code == ErrorCode.FILE_ACCESS_DENIED

    def test_validate_file_unsupported_format(self, tmp_path):
        """Test unsupported format error."""
        processor = VideoProcessor()

        test_file = tmp_path / "test.xyz"
        test_file.write_text("test")

        with pytest.raises(FileValidationError) as exc_info:
            processor.validate_file(test_file)

        assert exc_info.value.error_code == ErrorCode.INVALID_FORMAT
        assert "xyz" in exc_info.value.details["detected_format"]

    def test_validate_file_too_large(self, tmp_path):
        """Test file too large error."""
        processor = VideoProcessor()

        # Mock file size to be very large
        test_file = tmp_path / "test.mp4"
        test_file.write_text("test")

        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value.st_size = 1024 * 1024 * 1024 * 10  # 10GB

            with pytest.raises(FileValidationError) as exc_info:
                processor.validate_file(test_file)

        assert exc_info.value.error_code == ErrorCode.FILE_TOO_LARGE

    def test_validate_file_corrupted_ffmpeg_error(self, tmp_path):
        """Test corrupted file detection via ffmpeg error."""
        processor = VideoProcessor()

        test_file = tmp_path / "test.mp4"
        test_file.write_bytes(b"invalid video data")

        with patch("ffmpeg.probe") as mock_probe:
            import ffmpeg

            mock_error = ffmpeg.Error("ffmpeg", "", "")
            mock_error.stderr = b"Invalid data found"
            mock_probe.side_effect = mock_error

            with pytest.raises(FileValidationError) as exc_info:
                processor.validate_file(test_file)

            assert exc_info.value.error_code == ErrorCode.FILE_CORRUPTED

    def test_validate_file_no_video_stream(self, tmp_path):
        """Test no video stream error."""
        processor = VideoProcessor()

        test_file = tmp_path / "test.mp4"
        test_file.write_text("test")

        with patch("ffmpeg.probe") as mock_probe:
            mock_probe.return_value = {
                "streams": [
                    {"codec_type": "audio"}  # Only audio, no video
                ],
                "format": {"duration": "10.0"},
            }

            with pytest.raises(FileValidationError) as exc_info:
                processor.validate_file(test_file)

            assert exc_info.value.error_code == ErrorCode.NO_VIDEO_STREAM

    def test_extract_frame_invalid_scene_times(self):
        """Test frame extraction with invalid scene times."""
        processor = VideoProcessor()

        # Mock video info
        video_info = MagicMock()
        video_info.file_path = Path("/test/video.mp4")
        video_info.duration = 120.0

        with pytest.raises(FrameExtractionError):
            processor.extract_frame_from_scene(
                video_info=video_info,
                scene_start=30.0,
                scene_end=20.0,  # End before start
                scene_number=1,
            )

    def test_extract_frame_output_directory_error(self, tmp_path):
        """Test frame extraction with output directory creation error."""
        processor = VideoProcessor()

        video_info = MagicMock()
        video_info.file_path = Path("/test/video.mp4")
        video_info.duration = 120.0

        # Mock directory creation to fail
        invalid_output = tmp_path / "nonexistent" / "deep" / "path"

        with (
            patch.object(Path, "mkdir", side_effect=OSError("Permission denied")),
            pytest.raises(FrameExtractionError),
        ):
            processor.extract_frame_from_scene(
                video_info=video_info,
                scene_start=10.0,
                scene_end=20.0,
                scene_number=1,
                output_dir=invalid_output,
            )


class TestAudioExtractorErrorHandling:
    """Test error handling in AudioExtractor."""

    def test_extract_audio_no_audio_stream(self):
        """Test audio extraction with no audio stream."""
        from video_analyser.core.audio_extractor import AudioExtractor

        extractor = AudioExtractor()

        video_info = MagicMock()
        video_info.file_path = Path("/test/video.mp4")

        with patch("ffmpeg.probe") as mock_probe:
            mock_probe.return_value = {
                "streams": [
                    {"codec_type": "video"}  # Only video, no audio
                ]
            }

            with pytest.raises(AudioProcessingError) as exc_info:
                extractor.extract_audio(video_info)

            assert exc_info.value.error_code == ErrorCode.NO_AUDIO_STREAM

    def test_extract_audio_invalid_metadata(self):
        """Test audio extraction with invalid metadata."""
        from video_analyser.core.audio_extractor import AudioExtractor

        extractor = AudioExtractor()

        video_info = MagicMock()
        video_info.file_path = Path("/test/video.mp4")

        with patch("ffmpeg.probe") as mock_probe:
            mock_probe.return_value = {
                "streams": [
                    {
                        "codec_type": "audio",
                        "sample_rate": "invalid",  # Should be int
                        "channels": "also_invalid",
                    }
                ]
            }

            with pytest.raises(AudioProcessingError) as exc_info:
                extractor.extract_audio(video_info)

            assert exc_info.value.error_code == ErrorCode.AUDIO_CODEC_ERROR


class TestIntegratedErrorHandling:
    """Test integrated error handling across components."""

    def test_pipeline_coordinator_error_aggregation(self):
        """Test that PipelineCoordinator properly aggregates errors."""
        from video_analyser.core.pipeline_coordinator import VideoAnalysisResult

        # Create a video analysis result
        result = VideoAnalysisResult(video_info=MagicMock())

        # Add some errors
        error1 = AudioProcessingError(
            message="No audio stream", error_code=ErrorCode.NO_AUDIO_STREAM
        )
        error2 = FrameExtractionError(message="Frame extraction failed", scene_number=1)

        result.add_error(error1)
        result.add_error(error2)

        # Check error summary
        summary = result.get_error_summary()
        assert summary["total_errors"] == 2
        assert ErrorCode.NO_AUDIO_STREAM.value in summary["error_types"]
        assert ErrorCode.FRAME_EXTRACTION_FAILED.value in summary["error_types"]

        # Check dictionary serialization
        result_dict = result.to_dict()
        assert result_dict["error_summary"] is not None
        assert result_dict["error_summary"]["total_errors"] == 2

    def test_error_message_priority(self):
        """Test that first error sets the main error message."""
        from video_analyser.core.pipeline_coordinator import VideoAnalysisResult

        result = VideoAnalysisResult(video_info=MagicMock())

        error1 = FileValidationError(
            message="First error",
            file_path="/test/video.mp4",
            error_code=ErrorCode.FILE_NOT_FOUND,
        )
        error2 = AudioProcessingError(
            message="Second error", error_code=ErrorCode.NO_AUDIO_STREAM
        )

        result.add_error(error1)
        result.add_error(error2)

        # First error should set the main message
        assert "could not be found" in result.error_message.lower()

    @patch("video_analyser.core.video_processor.VideoProcessor.validate_file")
    def test_pipeline_validation_error_handling(self, mock_validate):
        """Test pipeline coordinator handling validation errors."""
        from video_analyser.core.pipeline_coordinator import PipelineCoordinator

        # Mock validation to raise error
        validation_error = FileValidationError(
            message="File not found",
            file_path="/test/video.mp4",
            error_code=ErrorCode.FILE_NOT_FOUND,
        )
        mock_validate.side_effect = validation_error

        coordinator = PipelineCoordinator()

        result = coordinator.analyze_video("/test/video.mp4")

        assert result.success is False
        assert len(result.errors) == 1
        assert result.errors[0].error_code == ErrorCode.FILE_NOT_FOUND
        assert "could not be found" in result.error_message.lower()


class TestErrorRecovery:
    """Test error recovery and partial processing scenarios."""

    def test_audio_extraction_failure_continues_processing(self):
        """Test that audio extraction failure doesn't stop video processing."""
        from pathlib import Path

        from video_analyser.core.pipeline_coordinator import PipelineCoordinator
        from video_analyser.core.scene_detector import SceneDetectionResult
        from video_analyser.core.video_processor import VideoInfo

        coordinator = PipelineCoordinator()

        # Create proper mock objects instead of using MagicMock for complex objects
        mock_video_info = VideoInfo(
            file_path=Path("/test/video.mp4"),
            duration=120.0,
            width=1920,
            height=1080,
            fps=30.0,
            format="mp4",
            size_mb=50.0,
            codec="h264",
        )

        mock_scene_result = SceneDetectionResult(
            scenes=[],
            total_scenes=0,
            detection_method="threshold",
            threshold_used=0.3,
            video_duration=120.0,
            average_scene_duration=0.0,
        )

        # Mock successful video validation but failed audio extraction
        with (
            patch.object(coordinator.video_processor, "validate_file") as mock_validate,
            patch.object(coordinator.audio_extractor, "extract_audio") as mock_audio,
            patch.object(coordinator.scene_detector, "detect_scenes") as mock_scenes,
        ):
            # Set up mocks with real objects
            mock_validate.return_value = mock_video_info
            mock_audio.side_effect = AudioProcessingError(
                message="Audio extraction failed",
                error_code=ErrorCode.AUDIO_EXTRACTION_FAILED,
            )
            mock_scenes.return_value = mock_scene_result

            result = coordinator.analyze_video(
                "/test/video.mp4",
                extract_audio=True,
                detect_scenes=True,
                extract_frames=False,
            )

            # Processing should continue despite audio error
            assert result.success is True  # Video and scenes processed successfully
            assert result.audio_info is None  # Audio failed
            assert len(result.errors) == 1  # But error recorded
            assert result.errors[0].error_code == ErrorCode.AUDIO_EXTRACTION_FAILED

    def test_graceful_degradation_multiple_errors(self):
        """Test graceful degradation with multiple errors."""
        from video_analyser.core.pipeline_coordinator import VideoAnalysisResult

        result = VideoAnalysisResult(video_info=MagicMock())

        # Simulate multiple non-fatal errors
        audio_error = AudioProcessingError(
            message="Audio quality poor", error_code=ErrorCode.AUDIO_CODEC_ERROR
        )
        frame_error = FrameExtractionError(message="Some frames failed", scene_number=5)

        result.add_error(audio_error)
        result.add_error(frame_error)

        # Result can still be successful with errors
        result.success = True

        assert result.success is True
        assert len(result.errors) == 2

        # Should provide comprehensive error information
        summary = result.get_error_summary()
        assert summary["total_errors"] == 2
        assert len(summary["errors"]) == 2
