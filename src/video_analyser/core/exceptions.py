"""Custom exceptions for video processing operations."""

from enum import Enum
from pathlib import Path
from typing import Any


class ErrorCode(Enum):
    """Error codes for video processing operations."""

    # File-related errors
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    FILE_ACCESS_DENIED = "FILE_ACCESS_DENIED"
    FILE_CORRUPTED = "FILE_CORRUPTED"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_FORMAT = "INVALID_FORMAT"

    # Processing errors
    FFMPEG_ERROR = "FFMPEG_ERROR"
    FFMPEG_NOT_FOUND = "FFMPEG_NOT_FOUND"
    PROCESSING_TIMEOUT = "PROCESSING_TIMEOUT"
    INSUFFICIENT_MEMORY = "INSUFFICIENT_MEMORY"
    INSUFFICIENT_DISK_SPACE = "INSUFFICIENT_DISK_SPACE"

    # Audio-specific errors
    NO_AUDIO_STREAM = "NO_AUDIO_STREAM"
    AUDIO_CODEC_ERROR = "AUDIO_CODEC_ERROR"
    AUDIO_EXTRACTION_FAILED = "AUDIO_EXTRACTION_FAILED"

    # Video-specific errors
    NO_VIDEO_STREAM = "NO_VIDEO_STREAM"
    VIDEO_CODEC_ERROR = "VIDEO_CODEC_ERROR"
    VIDEO_READ_ERROR = "VIDEO_READ_ERROR"
    SCENE_DETECTION_FAILED = "SCENE_DETECTION_FAILED"
    FRAME_EXTRACTION_FAILED = "FRAME_EXTRACTION_FAILED"

    # Configuration errors
    INVALID_CONFIGURATION = "INVALID_CONFIGURATION"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    INVALID_INPUT = "INVALID_INPUT"

    # Model-related errors
    MODEL_LOADING_FAILED = "MODEL_LOADING_FAILED"
    MODEL_INFERENCE_FAILED = "MODEL_INFERENCE_FAILED"

    # Generic errors
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    OPERATION_CANCELLED = "OPERATION_CANCELLED"


class VideoProcessingError(Exception):
    """Base exception for video processing operations."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        file_path: Path | str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        """
        Initialize video processing error.

        Args:
            message: Human-readable error message
            error_code: Specific error code for programmatic handling
            file_path: Path to the file that caused the error
            details: Additional error details
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.file_path = Path(file_path) if file_path else None
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary format for serialization."""
        return {
            "message": self.message,
            "error_code": self.error_code.value,
            "file_path": str(self.file_path) if self.file_path else None,
            "details": self.details,
            "cause": str(self.cause) if self.cause else None,
        }

    def __str__(self) -> str:
        """Return string representation of error."""
        parts = [f"{self.error_code.value}: {self.message}"]

        if self.file_path:
            parts.append(f"File: {self.file_path}")

        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"Details: {details_str}")

        if self.cause:
            parts.append(f"Caused by: {self.cause}")

        return " | ".join(parts)


class FileValidationError(VideoProcessingError):
    """Error during file validation."""

    def __init__(
        self,
        message: str,
        file_path: Path | str,
        error_code: ErrorCode = ErrorCode.FILE_CORRUPTED,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            file_path=file_path,
            details=details,
            cause=cause,
        )


class FFmpegError(VideoProcessingError):
    """Error during ffmpeg operations."""

    def __init__(
        self,
        message: str,
        command: str | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        return_code: int | None = None,
        file_path: Path | str | None = None,
        cause: Exception | None = None,
    ):
        """
        Initialize ffmpeg error.

        Args:
            message: Human-readable error message
            command: FFmpeg command that failed
            stdout: FFmpeg stdout output
            stderr: FFmpeg stderr output
            return_code: FFmpeg process return code
            file_path: File being processed when error occurred
            cause: Original exception
        """
        details: dict[str, str | int] = {}
        if command:
            details["command"] = command
        if stdout:
            details["stdout"] = stdout
        if stderr:
            details["stderr"] = stderr
        if return_code is not None:
            details["return_code"] = return_code

        super().__init__(
            message=message,
            error_code=ErrorCode.FFMPEG_ERROR,
            file_path=file_path,
            details=details,
            cause=cause,
        )


class AudioProcessingError(VideoProcessingError):
    """Error during audio processing operations."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.AUDIO_EXTRACTION_FAILED,
        file_path: Path | str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            file_path=file_path,
            details=details,
            cause=cause,
        )


class SceneDetectionError(VideoProcessingError):
    """Error during scene detection operations."""

    def __init__(
        self,
        message: str,
        file_path: Path | str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.SCENE_DETECTION_FAILED,
            file_path=file_path,
            details=details,
            cause=cause,
        )


class FrameExtractionError(VideoProcessingError):
    """Error during frame extraction operations."""

    def __init__(
        self,
        message: str,
        timestamp: float | None = None,
        scene_number: int | None = None,
        file_path: Path | str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        """
        Initialize frame extraction error.

        Args:
            message: Human-readable error message
            timestamp: Timestamp where extraction failed
            scene_number: Scene number where extraction failed
            file_path: Video file being processed
            details: Additional error details
            cause: Original exception
        """
        frame_details = details or {}
        if timestamp is not None:
            frame_details["timestamp"] = timestamp
        if scene_number is not None:
            frame_details["scene_number"] = scene_number

        super().__init__(
            message=message,
            error_code=ErrorCode.FRAME_EXTRACTION_FAILED,
            file_path=file_path,
            details=frame_details,
            cause=cause,
        )


class ConfigurationError(VideoProcessingError):
    """Error in configuration or setup."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        config_details = details or {}
        if config_key:
            config_details["config_key"] = config_key

        super().__init__(
            message=message,
            error_code=ErrorCode.INVALID_CONFIGURATION,
            details=config_details,
            cause=cause,
        )


class ProcessingTimeoutError(VideoProcessingError):
    """Error when processing operation times out."""

    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        operation: str | None = None,
        file_path: Path | str | None = None,
        cause: Exception | None = None,
    ):
        details: dict[str, float | str] = {"timeout_seconds": timeout_seconds}
        if operation:
            details["operation"] = operation

        super().__init__(
            message=message,
            error_code=ErrorCode.PROCESSING_TIMEOUT,
            file_path=file_path,
            details=details,
            cause=cause,
        )


def handle_ffmpeg_error(
    error: Exception, operation: str, file_path: Path | str | None = None
) -> FFmpegError:
    """
    Convert generic ffmpeg errors to structured FFmpegError.

    Args:
        error: Original ffmpeg exception
        operation: Description of operation that failed
        file_path: File being processed

    Returns:
        Structured FFmpegError
    """
    import ffmpeg

    if isinstance(error, ffmpeg.Error):
        return FFmpegError(
            message=f"FFmpeg {operation} failed",
            command=getattr(error, "cmd", None),
            stdout=getattr(error, "stdout", b"").decode("utf-8", errors="ignore")
            if hasattr(error, "stdout")
            else None,
            stderr=getattr(error, "stderr", b"").decode("utf-8", errors="ignore")
            if hasattr(error, "stderr")
            else None,
            return_code=getattr(error, "returncode", None),
            file_path=file_path,
            cause=error,
        )
    else:
        return FFmpegError(
            message=f"FFmpeg {operation} failed: {error}",
            file_path=file_path,
            cause=error,
        )


def get_user_friendly_message(error: VideoProcessingError) -> str:
    """
    Get user-friendly error message for common error types.

    Args:
        error: Video processing error

    Returns:
        User-friendly error message
    """
    error_messages = {
        ErrorCode.FILE_NOT_FOUND: "The video file could not be found. Please check the file path.",
        ErrorCode.FILE_ACCESS_DENIED: "Cannot access the video file. Please check file permissions.",
        ErrorCode.FILE_CORRUPTED: "The video file appears to be corrupted or damaged.",
        ErrorCode.FILE_TOO_LARGE: "The video file is too large for processing.",
        ErrorCode.INVALID_FORMAT: "The video format is not supported. Supported formats: MP4, MOV, AVI, WebM.",
        ErrorCode.NO_AUDIO_STREAM: "The video file does not contain an audio stream.",
        ErrorCode.NO_VIDEO_STREAM: "The file does not contain a valid video stream.",
        ErrorCode.FFMPEG_NOT_FOUND: "FFmpeg is not installed or not found in system PATH.",
        ErrorCode.PROCESSING_TIMEOUT: "Video processing timed out. The file may be too large or complex.",
        ErrorCode.INSUFFICIENT_MEMORY: "Not enough memory available for processing this video.",
        ErrorCode.INSUFFICIENT_DISK_SPACE: "Not enough disk space available for processing.",
        ErrorCode.MISSING_DEPENDENCY: "Required software dependency is missing.",
    }

    user_message = error_messages.get(error.error_code, error.message)

    if error.file_path:
        filename = error.file_path.name
        user_message = f"{user_message} (File: {filename})"

    return user_message
