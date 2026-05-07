"""Audio extraction and preprocessing utilities for Video Analyser."""

# NOTE: ffmpeg-python library lacks comprehensive type annotations
# See: https://github.com/kkroening/ffmpeg-python/issues/247
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import logging
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import ffmpeg
from pydantic import BaseModel

from video_analyser.core.exceptions import (
    AudioProcessingError,
    ErrorCode,
    handle_ffmpeg_error,
)
from video_analyser.core.video_processor import VideoInfo
from video_analyser.utils.config import get_config

logger = logging.getLogger(__name__)


class AudioInfo(BaseModel):
    """Audio metadata information."""

    file_path: Path
    duration: float
    sample_rate: int
    channels: int
    size_mb: float
    format: str


class AudioExtractor:
    """Audio extraction and preprocessing using ffmpeg."""

    def __init__(self, config: Any = None):
        """Initialize AudioExtractor with configuration."""
        self.config = config or get_config()
        self.temp_dir = Path(self.config.processing.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"AudioExtractor initialized with sample rate: {self.config.audio.sample_rate}Hz"
        )

    def extract_audio(
        self,
        video_info: VideoInfo,
        output_path: Path | str | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> AudioInfo:
        """
        Extract audio from video file with comprehensive error handling.

        Args:
            video_info: VideoInfo object from validated video file
            output_path: Optional custom output path for audio file
            progress_callback: Optional callback function for progress updates (0.0 to 1.0)

        Returns:
            AudioInfo object with extracted audio metadata

        Raises:
            AudioProcessingError: If audio extraction fails or no audio stream found
        """
        # Set up output path
        if output_path is None:
            output_path = self.temp_dir / f"{video_info.file_path.stem}_audio.wav"
        else:
            output_path = Path(output_path)

        # Ensure output directory exists
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise AudioProcessingError(
                message=f"Cannot create output directory: {output_path.parent}",
                file_path=video_info.file_path,
                cause=e,
            ) from e

        logger.info(
            f"Extracting audio from {video_info.file_path.name} to {output_path.name}"
        )

        try:
            # First, probe the video to check for audio streams
            try:
                probe = ffmpeg.probe(str(video_info.file_path))  # type: ignore[reportUnknownMemberType]
            except ffmpeg.Error as e:
                raise handle_ffmpeg_error(e, "audio probe", video_info.file_path) from e

            # Check if video has audio stream
            audio_streams = [s for s in probe["streams"] if s["codec_type"] == "audio"]
            if not audio_streams:
                raise AudioProcessingError(
                    message="Video file contains no audio stream",
                    error_code=ErrorCode.NO_AUDIO_STREAM,
                    file_path=video_info.file_path,
                )

            # Get original audio info and validate
            original_audio = audio_streams[0]
            try:
                original_sample_rate = int(original_audio.get("sample_rate", 0))
                original_channels = int(original_audio.get("channels", 0))
                original_codec = original_audio.get("codec_name", "unknown")
            except (ValueError, TypeError) as e:
                raise AudioProcessingError(
                    message="Invalid audio stream metadata",
                    error_code=ErrorCode.AUDIO_CODEC_ERROR,
                    file_path=video_info.file_path,
                    details={"audio_stream": original_audio},
                    cause=e,
                ) from e

            if original_sample_rate <= 0 or original_channels <= 0:
                raise AudioProcessingError(
                    message=f"Invalid audio parameters: {original_sample_rate}Hz, {original_channels} channels",
                    error_code=ErrorCode.AUDIO_CODEC_ERROR,
                    file_path=video_info.file_path,
                    details={
                        "sample_rate": original_sample_rate,
                        "channels": original_channels,
                        "codec": original_codec,
                    },
                )

            logger.debug(
                f"Original audio: {original_sample_rate}Hz, {original_channels} channels, codec: {original_codec}"
            )

            # Build ffmpeg pipeline for audio extraction
            stream = ffmpeg.input(str(video_info.file_path))  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]

            # Configure audio processing
            audio_args = {
                "acodec": "pcm_s16le",  # 16-bit PCM for compatibility
                "ar": self.config.audio.sample_rate,  # Sample rate
                "ac": self.config.audio.channels,  # Channel count
                "f": "wav",  # Output format
            }

            # Add audio normalization if enabled
            if self.config.audio.normalize_audio:
                stream = ffmpeg.filter(stream, "loudnorm")  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                logger.debug("Audio normalization enabled")

            # Add noise reduction if enabled (simple high-pass filter)
            if self.config.audio.noise_reduction:
                stream = ffmpeg.filter(  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                    stream, "highpass", f=200
                )  # Remove low-frequency noise
                logger.debug("Noise reduction enabled")

            # Configure output
            stream = ffmpeg.output(stream, str(output_path), **audio_args)  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            stream = ffmpeg.overwrite_output(stream)  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]

            # Execute extraction with progress tracking
            try:
                if progress_callback:
                    self._run_with_progress(
                        stream, video_info.duration, progress_callback
                    )
                else:
                    # Run without timeout (ffmpeg.run doesn't support timeout parameter)
                    ffmpeg.run(
                        stream,
                        quiet=True,
                        capture_stdout=True,
                        capture_stderr=True,
                    )  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            except subprocess.TimeoutExpired as e:
                raise AudioProcessingError(
                    message="Audio extraction timed out",
                    file_path=video_info.file_path,
                    cause=e,
                ) from e

            # Verify output file was created
            if not output_path.exists():
                raise AudioProcessingError(
                    message="Audio extraction completed but output file not found",
                    file_path=video_info.file_path,
                )

            # Verify file has content
            try:
                file_size = output_path.stat().st_size
                if file_size == 0:
                    raise AudioProcessingError(
                        message="Audio extraction produced empty file",
                        file_path=video_info.file_path,
                    )
            except OSError as e:
                raise AudioProcessingError(
                    message="Cannot access extracted audio file",
                    file_path=video_info.file_path,
                    cause=e,
                ) from e

            # Get extracted audio info
            try:
                audio_info = self._get_audio_info(output_path)
            except Exception as e:
                raise AudioProcessingError(
                    message="Failed to validate extracted audio file",
                    file_path=video_info.file_path,
                    cause=e,
                ) from e

            logger.info(
                f"Audio extracted successfully: {audio_info.duration:.1f}s, "
                f"{audio_info.sample_rate}Hz, {audio_info.channels} channels"
            )

            return audio_info

        except AudioProcessingError:
            # Re-raise our custom exceptions
            raise
        except ffmpeg.Error as e:
            # Handle ffmpeg-specific errors
            raise handle_ffmpeg_error(
                e, "audio extraction", video_info.file_path
            ) from e
        except Exception as e:
            # Handle any other unexpected errors
            raise AudioProcessingError(
                message=f"Unexpected error during audio extraction: {str(e)}",
                file_path=video_info.file_path,
                cause=e,
            ) from e

    def extract_audio_segment(
        self,
        video_info: VideoInfo,
        start_time: float,
        duration: float,
        output_path: Path | str | None = None,
    ) -> AudioInfo:
        """
        Extract a specific segment of audio from video.

        Args:
            video_info: VideoInfo object from validated video file
            start_time: Start time in seconds
            duration: Duration in seconds
            output_path: Optional custom output path for audio file

        Returns:
            AudioInfo object with extracted audio segment metadata

        Raises:
            ValueError: If start_time or duration are invalid
            RuntimeError: If audio extraction fails
        """
        if start_time < 0:
            raise ValueError("Start time cannot be negative")
        if duration <= 0:
            raise ValueError("Duration must be positive")
        if start_time + duration > video_info.duration:
            raise ValueError("Segment extends beyond video duration")

        if output_path is None:
            output_path = (
                self.temp_dir
                / f"{video_info.file_path.stem}_segment_{start_time:.1f}s.wav"
            )
        else:
            output_path = Path(output_path)

        logger.info(
            f"Extracting audio segment: {start_time:.1f}s - {start_time + duration:.1f}s"
        )

        try:
            # Build ffmpeg pipeline for segment extraction
            stream = ffmpeg.input(str(video_info.file_path), ss=start_time, t=duration)  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]

            # Configure audio processing
            audio_args = {
                "acodec": "pcm_s16le",
                "ar": self.config.audio.sample_rate,
                "ac": self.config.audio.channels,
                "f": "wav",
            }

            # Add normalization if enabled
            if self.config.audio.normalize_audio:
                stream = ffmpeg.filter(stream, "loudnorm")  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]

            # Configure output
            stream = ffmpeg.output(stream, str(output_path), **audio_args)  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            stream = ffmpeg.overwrite_output(stream)  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]

            # Execute extraction
            ffmpeg.run(stream, quiet=True, capture_stdout=True, capture_stderr=True)  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]

            # Get extracted audio info
            audio_info = self._get_audio_info(output_path)

            logger.info(f"Audio segment extracted: {audio_info.duration:.1f}s")
            return audio_info

        except ffmpeg.Error as e:
            # Handle stderr which can be bytes or string
            if e.stderr:
                error_msg = (
                    e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)
                )  # type: ignore[reportUnknownMemberType,reportUnknownVariableType]
            else:
                error_msg = str(e)
            logger.error(f"FFmpeg error during segment extraction: {error_msg}")
            raise RuntimeError(f"Audio segment extraction failed: {error_msg}") from e

    def _get_audio_info(self, audio_path: Path) -> AudioInfo:
        """
        Get metadata information for extracted audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            AudioInfo object with metadata
        """
        try:
            probe = ffmpeg.probe(str(audio_path))

            # Find audio stream
            audio_stream = None
            for stream in probe["streams"]:
                if stream["codec_type"] == "audio":
                    audio_stream = stream
                    break

            if not audio_stream:
                raise RuntimeError("No audio stream found in extracted file")

            duration = float(probe["format"]["duration"])
            sample_rate = int(audio_stream["sample_rate"])
            channels = int(audio_stream["channels"])
            size_mb = audio_path.stat().st_size / (1024 * 1024)
            format_name = probe["format"]["format_name"]

            return AudioInfo(
                file_path=audio_path,
                duration=duration,
                sample_rate=sample_rate,
                channels=channels,
                size_mb=size_mb,
                format=format_name,
            )

        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            raise RuntimeError(f"Failed to get audio metadata: {e}") from e

    def _run_with_progress(
        self,
        stream: Any,
        total_duration: float,
        progress_callback: Callable[[float], None],
    ) -> None:
        """
        Run ffmpeg with progress tracking.

        Args:
            stream: ffmpeg stream object
            total_duration: Total duration in seconds for progress calculation
            progress_callback: Callback function for progress updates
        """
        try:
            process = ffmpeg.run_async(stream, pipe_stderr=True, quiet=True)

            while True:
                if process.stderr is None:
                    break
                output = process.stderr.readline()
                if output == b"" and process.poll() is not None:
                    break

                if output:
                    line = output.decode("utf-8").strip()

                    # Parse ffmpeg progress output
                    if "time=" in line:
                        try:
                            # Extract time from ffmpeg output (format: time=00:01:23.45)
                            time_str = line.split("time=")[1].split()[0]

                            # Convert time to seconds
                            if ":" in time_str:
                                parts = time_str.split(":")
                                if len(parts) == 3:
                                    hours = float(parts[0])
                                    minutes = float(parts[1])
                                    seconds = float(parts[2])
                                    current_time = hours * 3600 + minutes * 60 + seconds

                                    # Calculate progress (0.0 to 1.0)
                                    progress = min(current_time / total_duration, 1.0)
                                    progress_callback(progress)
                        except (ValueError, IndexError):
                            # Ignore parsing errors
                            pass

            # Wait for process to complete
            process.wait()

            # Final progress update
            progress_callback(1.0)

            if process.returncode != 0:
                stderr_content = process.stderr.read() if process.stderr else b""
                raise ffmpeg.Error("ffmpeg", "", stderr_content)

        except Exception as e:
            logger.error(f"Error during progress tracking: {e}")
            raise

    def cleanup_temp_audio_files(self) -> None:
        """Clean up temporary audio files."""
        if not self.config.processing.cleanup_temp_files:
            logger.debug("Audio temp file cleanup disabled")
            return

        if not self.temp_dir.exists():
            return

        # Clean up audio files (wav, mp3, etc.)
        audio_extensions = ["*.wav", "*.mp3", "*.aac", "*.flac", "*.ogg"]
        temp_files = []

        for ext in audio_extensions:
            temp_files.extend(self.temp_dir.glob(ext))

        if temp_files:
            logger.info(f"Cleaning up {len(temp_files)} temporary audio files")
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                except OSError as e:
                    logger.warning(f"Could not remove temp audio file {temp_file}: {e}")

    def get_audio_config(self) -> dict[str, Any]:
        """
        Get current audio processing configuration.

        Returns:
            Dictionary with audio configuration settings
        """
        return {
            "sample_rate": self.config.audio.sample_rate,
            "channels": self.config.audio.channels,
            "noise_reduction": self.config.audio.noise_reduction,
            "normalize_audio": self.config.audio.normalize_audio,
        }
