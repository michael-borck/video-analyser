"""Video processing pipeline for Video Analyser."""

# NOTE: ffmpeg-python library lacks comprehensive type annotations
# See: https://github.com/kkroening/ffmpeg-python/issues/247
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import logging
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import ffmpeg
from pydantic import BaseModel

from video_analyser.core.exceptions import (
    ErrorCode,
    FileValidationError,
    FrameExtractionError,
    VideoProcessingError,
    handle_ffmpeg_error,
)
from video_analyser.utils.config import get_config

logger = logging.getLogger(__name__)


class VideoInfo(BaseModel):
    """Video metadata information."""

    file_path: Path
    duration: float
    width: int
    height: int
    fps: float
    format: str
    size_mb: float
    codec: str


class FrameInfo(BaseModel):
    """Frame extraction metadata."""

    frame_path: Path
    timestamp: float
    scene_number: int
    width: int
    height: int
    size_kb: float
    format: str = "jpg"


class VideoProcessor:
    """Main video processing class with file validation and format support."""

    def __init__(self, config: Any = None):
        """Initialize VideoProcessor with configuration."""
        self.config = config or get_config()
        self.supported_formats = [
            fmt.lower() for fmt in self.config.processing.supported_formats
        ]

        # Ensure temp directory exists
        self.temp_dir = Path(self.config.processing.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"VideoProcessor initialized with formats: {self.supported_formats}"
        )
        logger.debug(f"Temp directory: {self.temp_dir}")

    def validate_file(self, file_path: Path | str) -> VideoInfo:
        """
        Validate video file format, size, and accessibility with comprehensive error handling.

        Args:
            file_path: Path to video file

        Returns:
            VideoInfo object with file metadata

        Raises:
            FileValidationError: For all file validation issues
            VideoProcessingError: For processing-related errors
        """
        file_path = Path(file_path)

        try:
            # Check if file exists
            if not file_path.exists():
                raise FileValidationError(
                    message=f"Video file not found: {file_path.name}",
                    file_path=file_path,
                    error_code=ErrorCode.FILE_NOT_FOUND,
                )

            # Check if file is accessible
            if not os.access(file_path, os.R_OK):
                raise FileValidationError(
                    message=f"Cannot read video file: {file_path.name}",
                    file_path=file_path,
                    error_code=ErrorCode.FILE_ACCESS_DENIED,
                )

            # Check file extension
            file_extension = file_path.suffix.lower().lstrip(".")
            if file_extension not in self.supported_formats:
                raise FileValidationError(
                    message=f"Unsupported video format '.{file_extension}'",
                    file_path=file_path,
                    error_code=ErrorCode.INVALID_FORMAT,
                    details={
                        "detected_format": file_extension,
                        "supported_formats": self.supported_formats,
                    },
                )

            # Check file size
            try:
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
            except OSError as e:
                raise FileValidationError(
                    message=f"Cannot access file information: {file_path.name}",
                    file_path=file_path,
                    error_code=ErrorCode.FILE_ACCESS_DENIED,
                    cause=e,
                ) from e

            if file_size_mb > self.config.processing.max_video_size_mb:
                raise FileValidationError(
                    message=f"Video file is too large ({file_size_mb:.1f}MB)",
                    file_path=file_path,
                    error_code=ErrorCode.FILE_TOO_LARGE,
                    details={
                        "file_size_mb": file_size_mb,
                        "max_size_mb": self.config.processing.max_video_size_mb,
                    },
                )

            # Check available disk space
            if not self._check_disk_space(file_size_mb):
                raise VideoProcessingError(
                    message="Insufficient disk space for processing",
                    error_code=ErrorCode.INSUFFICIENT_DISK_SPACE,
                    file_path=file_path,
                    details={
                        "required_mb": file_size_mb * 2
                    },  # Estimate 2x space needed
                )

            # Probe video file with ffmpeg to get metadata
            try:
                probe = ffmpeg.probe(str(file_path))
            except ffmpeg.Error as e:
                # Handle ffmpeg probe errors
                stderr_str = ""
                if hasattr(e, "stderr") and e.stderr:
                    stderr_str = e.stderr.decode("utf-8", errors="ignore")

                if (
                    "Invalid data found" in stderr_str
                    or "moov atom not found" in stderr_str
                ):
                    raise FileValidationError(
                        message=f"Video file appears to be corrupted: {file_path.name}",
                        file_path=file_path,
                        error_code=ErrorCode.FILE_CORRUPTED,
                        details={"ffmpeg_error": stderr_str},
                        cause=e,
                    ) from e
                else:
                    raise handle_ffmpeg_error(e, "probe", file_path) from e
            except Exception as e:
                raise VideoProcessingError(
                    message=f"Failed to analyze video file: {file_path.name}",
                    error_code=ErrorCode.UNKNOWN_ERROR,
                    file_path=file_path,
                    cause=e,
                ) from e

            # Validate probe results
            if not probe or "streams" not in probe:
                raise FileValidationError(
                    message=f"Invalid video file structure: {file_path.name}",
                    file_path=file_path,
                    error_code=ErrorCode.FILE_CORRUPTED,
                )

            # Find video stream
            video_stream = None
            for stream in probe["streams"]:
                if stream["codec_type"] == "video":
                    video_stream = stream
                    break

            if not video_stream:
                raise FileValidationError(
                    message=f"No video stream found in file: {file_path.name}",
                    file_path=file_path,
                    error_code=ErrorCode.NO_VIDEO_STREAM,
                )

            # Extract and validate metadata
            try:
                duration = float(probe["format"]["duration"])
                width = int(video_stream["width"])
                height = int(video_stream["height"])

                # Validate video dimensions
                if width <= 0 or height <= 0:
                    raise FileValidationError(
                        message=f"Invalid video dimensions: {width}x{height}",
                        file_path=file_path,
                        error_code=ErrorCode.FILE_CORRUPTED,
                        details={"width": width, "height": height},
                    )

                # Calculate FPS (handle different fps representations)
                fps_str = video_stream.get("r_frame_rate", "0/1")
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    fps = float(num) / float(den) if float(den) != 0 else 0
                else:
                    fps = float(fps_str)

                # Validate FPS
                if fps <= 0 or fps > 1000:  # Reasonable bounds check
                    logger.warning(f"Unusual FPS value: {fps} for {file_path.name}")

                format_name = probe["format"]["format_name"]
                codec = video_stream["codec_name"]

                logger.info(
                    f"Video validated: {file_path.name} ({duration:.1f}s, {width}x{height}, {fps:.1f}fps)"
                )

                return VideoInfo(
                    file_path=file_path,
                    duration=duration,
                    width=width,
                    height=height,
                    fps=fps,
                    format=format_name,
                    size_mb=file_size_mb,
                    codec=codec,
                )

            except (KeyError, ValueError, TypeError) as e:
                raise FileValidationError(
                    message=f"Invalid video metadata in file: {file_path.name}",
                    file_path=file_path,
                    error_code=ErrorCode.FILE_CORRUPTED,
                    details={"metadata_error": str(e)},
                    cause=e,
                ) from e

        except VideoProcessingError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Catch any other unexpected errors
            raise VideoProcessingError(
                message=f"Unexpected error validating video file: {file_path.name}",
                error_code=ErrorCode.UNKNOWN_ERROR,
                file_path=file_path,
                cause=e,
            ) from e

    def _check_disk_space(self, required_mb: float) -> bool:
        """
        Check if sufficient disk space is available for processing.

        Args:
            required_mb: Required disk space in MB

        Returns:
            True if sufficient space available, False otherwise
        """
        try:
            # Check space in temp directory
            disk_usage = shutil.disk_usage(self.temp_dir)
            available_mb = disk_usage.free / (1024 * 1024)

            # Require 2x the file size plus 1GB buffer
            needed_mb = (required_mb * 2) + 1024

            return available_mb >= needed_mb
        except OSError:
            # If we can't check disk space, assume it's available
            logger.warning("Could not check disk space")
            return True

    def _check_ffmpeg_available(self) -> bool:
        """
        Check if FFmpeg is available in the system.

        Returns:
            True if FFmpeg is available, False otherwise
        """
        try:
            subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, check=True, timeout=5
            )
            return True
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return False

    def is_format_supported(self, file_path: Path | str) -> bool:
        """
        Check if video file format is supported.

        Args:
            file_path: Path to video file

        Returns:
            True if format is supported, False otherwise
        """
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower().lstrip(".")
        return file_extension in self.supported_formats

    def get_supported_formats(self) -> list[str]:
        """
        Get list of supported video formats.

        Returns:
            List of supported file extensions
        """
        return self.supported_formats.copy()

    def extract_frame_from_scene(
        self,
        video_info: VideoInfo,
        scene_start: float,
        scene_end: float,
        scene_number: int,
        output_dir: Path | str | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> FrameInfo:
        """
        Extract a representative frame from a scene with comprehensive error handling.

        Args:
            video_info: VideoInfo object from validated video file
            scene_start: Scene start time in seconds
            scene_end: Scene end time in seconds
            scene_number: Scene number for naming
            output_dir: Optional custom output directory for frame
            progress_callback: Optional callback function for progress updates

        Returns:
            FrameInfo object with extracted frame metadata

        Raises:
            FrameExtractionError: If frame extraction fails
        """
        # Validate inputs
        if scene_start < 0 or scene_end <= scene_start:
            raise FrameExtractionError(
                message=f"Invalid scene times: start={scene_start:.2f}s, end={scene_end:.2f}s",
                scene_number=scene_number,
                file_path=video_info.file_path,
            )

        if scene_end > video_info.duration:
            logger.warning(
                f"Scene end time ({scene_end:.2f}s) exceeds video duration ({video_info.duration:.2f}s)"
            )
            scene_end = video_info.duration

        # Set up output directory
        if output_dir is None:
            output_dir = self.temp_dir / "frames"
        else:
            output_dir = Path(output_dir)

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise FrameExtractionError(
                message=f"Cannot create output directory: {output_dir}",
                scene_number=scene_number,
                file_path=video_info.file_path,
                cause=e,
            ) from e

        # Check disk space
        if not self._check_disk_space(1):  # 1MB should be enough for a frame
            raise FrameExtractionError(
                message="Insufficient disk space for frame extraction",
                scene_number=scene_number,
                file_path=video_info.file_path,
                details={"error_code": ErrorCode.INSUFFICIENT_DISK_SPACE.value},
            )

        # Use middle of scene as representative timestamp
        timestamp = scene_start + (scene_end - scene_start) / 2

        # Create output filename
        output_filename = f"scene_{scene_number:03d}_frame_{timestamp:.2f}s.jpg"
        output_path = output_dir / output_filename

        logger.info(f"Extracting frame from scene {scene_number} at {timestamp:.2f}s")

        try:
            # Check if FFmpeg is available
            if not self._check_ffmpeg_available():
                raise FrameExtractionError(
                    message="FFmpeg not found or not available",
                    timestamp=timestamp,
                    scene_number=scene_number,
                    file_path=video_info.file_path,
                    details={"error_code": ErrorCode.FFMPEG_NOT_FOUND.value},
                )

            # Build ffmpeg command for frame extraction
            stream = ffmpeg.input(str(video_info.file_path), ss=timestamp)

            # Configure frame extraction
            frame_args = {
                "vframes": 1,  # Extract only 1 frame
                "q:v": self._get_quality_value(),  # Quality setting
                "f": "image2",  # Image format
            }

            # Apply any scaling if needed (maintain aspect ratio)
            max_frame_width = getattr(self.config.processing, "max_frame_width", None)
            if max_frame_width and video_info.width > max_frame_width:
                stream = ffmpeg.filter(stream, "scale", f"{max_frame_width}:-1")

            # Configure output
            stream = ffmpeg.output(stream, str(output_path), **frame_args)
            stream = ffmpeg.overwrite_output(stream)

            # Execute extraction (ffmpeg.run doesn't support timeout parameter)
            try:
                ffmpeg.run(
                    stream,
                    quiet=True,
                    capture_stdout=True,
                    capture_stderr=True,
                )
            except subprocess.TimeoutExpired as e:
                raise FrameExtractionError(
                    message="Frame extraction timed out",
                    timestamp=timestamp,
                    scene_number=scene_number,
                    file_path=video_info.file_path,
                    cause=e,
                ) from e

            # Verify output file was created
            if not output_path.exists():
                raise FrameExtractionError(
                    message="Frame extraction completed but output file not found",
                    timestamp=timestamp,
                    scene_number=scene_number,
                    file_path=video_info.file_path,
                )

            # Verify file has content
            try:
                file_size_kb = output_path.stat().st_size / 1024
                if file_size_kb == 0:
                    raise FrameExtractionError(
                        message="Frame extraction produced empty file",
                        timestamp=timestamp,
                        scene_number=scene_number,
                        file_path=video_info.file_path,
                    )
            except OSError as e:
                raise FrameExtractionError(
                    message="Cannot access extracted frame file",
                    timestamp=timestamp,
                    scene_number=scene_number,
                    file_path=video_info.file_path,
                    cause=e,
                ) from e

            logger.info(f"Frame extracted: {output_path.name} ({file_size_kb:.1f}KB)")

            if progress_callback:
                progress_callback(1.0)

            return FrameInfo(
                frame_path=output_path,
                timestamp=timestamp,
                scene_number=scene_number,
                width=video_info.width,
                height=video_info.height,
                size_kb=file_size_kb,
                format="jpg",
            )

        except ffmpeg.Error as e:
            # Handle ffmpeg-specific errors
            raise handle_ffmpeg_error(
                e, "frame extraction", video_info.file_path
            ) from e

        except FrameExtractionError:
            # Re-raise our custom exceptions
            raise

        except Exception as e:
            # Handle any other unexpected errors
            raise FrameExtractionError(
                message=f"Unexpected error during frame extraction: {str(e)}",
                timestamp=timestamp,
                scene_number=scene_number,
                file_path=video_info.file_path,
                cause=e,
            ) from e

    def extract_frames_from_scenes(
        self,
        video_info: VideoInfo,
        scenes: list[tuple[float, float, int]],  # (start, end, scene_number)
        output_dir: Path | str | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[FrameInfo]:
        """
        Extract representative frames from multiple scenes.

        Args:
            video_info: VideoInfo object from validated video file
            scenes: List of (start_time, end_time, scene_number) tuples
            output_dir: Optional custom output directory for frames
            progress_callback: Optional callback function for progress updates

        Returns:
            List of FrameInfo objects for extracted frames

        Raises:
            RuntimeError: If frame extraction fails
        """
        if not scenes:
            logger.warning("No scenes provided for frame extraction")
            return []

        logger.info(f"Extracting frames from {len(scenes)} scenes")

        extracted_frames = []
        total_scenes = len(scenes)

        for i, (start_time, end_time, scene_number) in enumerate(scenes):
            try:
                # Extract frame from this scene
                frame_info = self.extract_frame_from_scene(
                    video_info,
                    start_time,
                    end_time,
                    scene_number,
                    output_dir,
                    None,  # Don't pass progress callback to individual extractions
                )
                extracted_frames.append(frame_info)

                # Update overall progress
                if progress_callback:
                    progress = (i + 1) / total_scenes
                    progress_callback(progress)

            except Exception as e:
                logger.error(f"Failed to extract frame from scene {scene_number}: {e}")
                # Continue with other scenes rather than failing completely
                continue

        logger.info(
            f"Successfully extracted {len(extracted_frames)} frames from {total_scenes} scenes"
        )
        return extracted_frames

    def _get_quality_value(self) -> int:
        """
        Convert quality percentage to ffmpeg quality value.

        Returns:
            Quality value for ffmpeg (lower is better quality)
        """
        # Convert quality percentage (80% = good) to ffmpeg q:v scale (2-31, lower is better)
        # 100% quality = q:v 2, 50% quality = q:v 15, 1% quality = q:v 31
        quality_percent = getattr(self.config.output, "frame_quality", 80)
        # Map 100-1% to 2-31 scale
        quality_value = 2 + int((100 - quality_percent) * 29 / 99)
        return max(2, min(31, quality_value))

    def cleanup_temp_files(self) -> None:
        """Clean up temporary files if configured to do so."""
        if not self.config.processing.cleanup_temp_files:
            logger.debug("Temp file cleanup disabled")
            return

        if not self.temp_dir.exists():
            return

        temp_files = list(self.temp_dir.glob("*"))
        if temp_files:
            logger.info(f"Cleaning up {len(temp_files)} temporary files")
            for temp_file in temp_files:
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                    elif temp_file.is_dir() and not any(temp_file.iterdir()):
                        # Remove empty directories
                        temp_file.rmdir()
                except OSError as e:
                    logger.warning(f"Could not remove temp file {temp_file}: {e}")
