"""Scene detection using ffmpeg scene filter with configurable thresholds."""

# NOTE: ffmpeg-python library lacks comprehensive type annotations
# See: https://github.com/kkroening/ffmpeg-python/issues/247
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import ffmpeg
from pydantic import BaseModel

from video_analyser.core.video_processor import VideoInfo
from video_analyser.utils.config import get_config

logger = logging.getLogger(__name__)


class Scene(BaseModel):
    """Scene information with timestamps and metadata."""

    start_time: float
    end_time: float
    duration: float
    scene_number: int
    confidence: float = 0.0  # Scene change confidence (0.0 to 1.0)

    @property
    def start_time_str(self) -> str:
        """Get start time as formatted string (HH:MM:SS.mmm)."""
        return self._format_time(self.start_time)

    @property
    def end_time_str(self) -> str:
        """Get end time as formatted string (HH:MM:SS.mmm)."""
        return self._format_time(self.end_time)

    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS.mmm."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


class SceneDetectionResult(BaseModel):
    """Result of scene detection analysis."""

    scenes: list[Scene]
    total_scenes: int
    detection_method: str
    threshold_used: float
    video_duration: float
    average_scene_duration: float

    @property
    def scene_boundaries(self) -> list[float]:
        """Get list of scene boundary timestamps."""
        boundaries = [0.0]  # Always include start
        for scene in self.scenes[:-1]:  # Exclude last scene
            boundaries.append(scene.end_time)
        return boundaries


class SceneDetector:
    """Scene detection system using ffmpeg scene filter with configurable thresholds."""

    def __init__(self, config: Any = None):
        """Initialize SceneDetector with configuration."""
        self.config = config or get_config()
        self.temp_dir = Path(self.config.processing.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"SceneDetector initialized: method={self.config.scene_detection.method}, "
            f"threshold={self.config.scene_detection.threshold}"
        )

    def detect_scenes(
        self,
        video_info: VideoInfo,
        progress_callback: Callable[[float], None] | None = None,
    ) -> SceneDetectionResult:
        """
        Detect scenes in video using configured method and thresholds.

        Args:
            video_info: VideoInfo object from validated video file
            progress_callback: Optional callback function for progress updates (0.0 to 1.0)

        Returns:
            SceneDetectionResult with detected scenes

        Raises:
            RuntimeError: If scene detection fails
        """
        logger.info(f"Detecting scenes in {video_info.file_path.name}")

        try:
            if self.config.scene_detection.method == "threshold":
                return self._detect_scenes_threshold(video_info, progress_callback)
            elif self.config.scene_detection.method == "adaptive":
                return self._detect_scenes_adaptive(video_info, progress_callback)
            else:
                raise ValueError(
                    f"Unknown scene detection method: {self.config.scene_detection.method}"
                )

        except Exception as e:
            logger.error(f"Scene detection failed: {e}")
            raise RuntimeError(f"Scene detection failed: {e}") from e

    def _detect_scenes_threshold(
        self,
        video_info: VideoInfo,
        progress_callback: Callable[[float], None] | None = None,
    ) -> SceneDetectionResult:
        """
        Detect scenes using fixed threshold method.

        Args:
            video_info: VideoInfo object
            progress_callback: Optional progress callback

        Returns:
            SceneDetectionResult with detected scenes
        """
        threshold = self.config.scene_detection.threshold
        logger.debug(f"Using threshold scene detection: {threshold}")

        try:
            # Use ffmpeg scene filter to detect scene changes
            scene_times = self._run_scene_detection(
                video_info, threshold, progress_callback
            )

            # If no scenes detected or too few, use fallback
            if len(scene_times) <= 1:
                logger.warning(
                    "Threshold detection found no/few scenes, using fallback"
                )
                return self._fallback_scene_detection(video_info, threshold)

            # Create scenes from detected timestamps
            scenes = self._create_scenes_from_timestamps(
                scene_times, video_info.duration
            )

            # Filter scenes by minimum duration
            scenes = self._filter_scenes_by_duration(scenes)

            logger.info(f"Detected {len(scenes)} scenes using threshold method")

            return SceneDetectionResult(
                scenes=scenes,
                total_scenes=len(scenes),
                detection_method="threshold",
                threshold_used=threshold,
                video_duration=video_info.duration,
                average_scene_duration=sum(s.duration for s in scenes) / len(scenes)
                if scenes
                else 0.0,
            )

        except Exception as e:
            logger.warning(f"Threshold detection failed: {e}, using fallback")
            return self._fallback_scene_detection(video_info, threshold)

    def _detect_scenes_adaptive(
        self,
        video_info: VideoInfo,
        progress_callback: Callable[[float], None] | None = None,
    ) -> SceneDetectionResult:
        """
        Detect scenes using adaptive threshold method.

        Args:
            video_info: VideoInfo object
            progress_callback: Optional progress callback

        Returns:
            SceneDetectionResult with detected scenes
        """
        base_threshold = self.config.scene_detection.threshold
        logger.debug(f"Using adaptive scene detection: base_threshold={base_threshold}")

        # Try multiple thresholds to find optimal scene count
        thresholds_to_try = [
            base_threshold * 0.5,  # More sensitive
            base_threshold,  # Default
            base_threshold * 1.5,  # Less sensitive
            base_threshold * 2.0,  # Much less sensitive
        ]

        best_result = None
        best_scene_count = 0

        for i, threshold in enumerate(thresholds_to_try):
            try:
                # Update progress for adaptive attempts
                if progress_callback:
                    progress_callback(
                        i / len(thresholds_to_try) * 0.8
                    )  # 80% for trying thresholds

                scene_times = self._run_scene_detection(video_info, threshold, None)

                if scene_times:
                    scenes = self._create_scenes_from_timestamps(
                        scene_times, video_info.duration
                    )
                    scenes = self._filter_scenes_by_duration(scenes)

                    # Prefer results with reasonable scene count (2-20 scenes for most videos)
                    scene_count = len(scenes)
                    if 2 <= scene_count <= 20 and scene_count > best_scene_count:
                        best_result = SceneDetectionResult(
                            scenes=scenes,
                            total_scenes=scene_count,
                            detection_method="adaptive",
                            threshold_used=threshold,
                            video_duration=video_info.duration,
                            average_scene_duration=sum(s.duration for s in scenes)
                            / scene_count,
                        )
                        best_scene_count = scene_count

            except Exception as e:
                logger.debug(f"Adaptive threshold {threshold} failed: {e}")
                continue

        # Final progress update
        if progress_callback:
            progress_callback(1.0)

        # Return best result or fallback
        if best_result:
            logger.info(
                f"Adaptive detection found {best_result.total_scenes} scenes with threshold {best_result.threshold_used}"
            )
            return best_result
        else:
            logger.warning("Adaptive detection failed, using fallback")
            return self._fallback_scene_detection(video_info, base_threshold)

    def _run_scene_detection(
        self,
        video_info: VideoInfo,
        threshold: float,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[float]:
        """
        Run ffmpeg scene detection and parse timestamps.

        Args:
            video_info: VideoInfo object
            threshold: Scene detection threshold
            progress_callback: Optional progress callback

        Returns:
            List of scene change timestamps

        Raises:
            RuntimeError: If ffmpeg scene detection fails
        """
        try:
            # Build ffmpeg command for scene detection
            # Use select filter to detect scene changes
            stream = ffmpeg.input(str(video_info.file_path))
            stream = ffmpeg.filter(stream, "select", f"gt(scene,{threshold})")
            stream = ffmpeg.output(
                stream,
                "-",
                f="null",
                **{"show_entries": "packet=pts_time", "of": "csv=p=0"},
            )

            # Alternative approach using scene detection filter
            # This gives us scene change information
            scene_stream = ffmpeg.input(str(video_info.file_path))
            scene_stream = ffmpeg.filter(scene_stream, "scdet", threshold=threshold)
            scene_stream = ffmpeg.output(
                scene_stream,
                "-",
                f="null",
                loglevel="info",
            )

            # Run scene detection
            if progress_callback:
                # Progress version doesn't capture stderr, so return empty
                self._run_with_progress(
                    scene_stream, video_info.duration, progress_callback
                )
                stderr_output = ""
            else:
                _, stderr = ffmpeg.run(
                    scene_stream, capture_stdout=True, capture_stderr=True
                )
                stderr_output = stderr.decode() if stderr else ""

            # Parse scene change timestamps from stderr output
            scene_times = self._parse_scene_timestamps(stderr_output)

            logger.debug(
                f"Found {len(scene_times)} scene changes with threshold {threshold}"
            )
            return scene_times

        except ffmpeg.Error as e:
            # Handle stderr which can be bytes or string
            if e.stderr:
                error_msg = (
                    e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr)
                )
            else:
                error_msg = str(e)
            logger.error(f"FFmpeg scene detection error: {error_msg}")
            raise RuntimeError(f"Scene detection failed: {error_msg}") from e

    def _parse_scene_timestamps(self, ffmpeg_output: str) -> list[float]:
        """
        Parse scene detection timestamps from ffmpeg output.

        Args:
            ffmpeg_output: FFmpeg stderr output containing scene detection info

        Returns:
            List of scene change timestamps in seconds
        """
        scene_times = [0.0]  # Always start with beginning of video

        # Look for scene detection patterns in ffmpeg output
        # Pattern like: [scdet @ 0x...] lavfi.scene_score=0.123456 pts_time:12.345
        scene_pattern = r"lavfi\.scene_score=[\d.]+.*?pts_time:([\d.]+)"

        for match in re.finditer(scene_pattern, ffmpeg_output):
            try:
                timestamp = float(match.group(1))
                scene_times.append(timestamp)
            except (ValueError, IndexError):
                continue

        # Alternative pattern for scene changes
        # Pattern like: [Parsed_scdet_0 @ 0x...] scene_score=0.123 time=12.345
        alt_pattern = r"scene_score=[\d.]+.*?time=([\d.]+)"

        for match in re.finditer(alt_pattern, ffmpeg_output):
            try:
                timestamp = float(match.group(1))
                if timestamp not in scene_times:
                    scene_times.append(timestamp)
            except (ValueError, IndexError):
                continue

        return sorted(set(scene_times))  # Remove duplicates and sort

    def _create_scenes_from_timestamps(
        self, timestamps: list[float], total_duration: float
    ) -> list[Scene]:
        """
        Create Scene objects from timestamps.

        Args:
            timestamps: List of scene start timestamps
            total_duration: Total video duration

        Returns:
            List of Scene objects
        """
        scenes = []

        for i, start_time in enumerate(timestamps):
            # Determine end time
            end_time = timestamps[i + 1] if i + 1 < len(timestamps) else total_duration

            duration = end_time - start_time

            # Skip very short scenes (likely detection errors)
            if duration < 0.5:
                continue

            scene = Scene(
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                scene_number=len(scenes) + 1,
                confidence=0.8,  # Default confidence for detected scenes
            )
            scenes.append(scene)

        return scenes

    def _filter_scenes_by_duration(self, scenes: list[Scene]) -> list[Scene]:
        """
        Filter scenes by minimum duration requirement.

        Args:
            scenes: List of Scene objects

        Returns:
            Filtered list of scenes
        """
        min_duration = self.config.scene_detection.min_scene_duration
        filtered_scenes = []

        for scene in scenes:
            if scene.duration >= min_duration:
                filtered_scenes.append(scene)
            else:
                logger.debug(
                    f"Filtered out short scene: {scene.duration:.1f}s < {min_duration}s"
                )

        # Renumber scenes after filtering
        for i, scene in enumerate(filtered_scenes):
            scene.scene_number = i + 1

        return filtered_scenes

    def _fallback_scene_detection(
        self, video_info: VideoInfo, threshold: float
    ) -> SceneDetectionResult:
        """
        Fallback scene detection using fixed intervals.

        Args:
            video_info: VideoInfo object
            threshold: Threshold used (for reporting)

        Returns:
            SceneDetectionResult with fixed-interval scenes
        """
        interval = self.config.scene_detection.fallback_interval
        logger.info(f"Using fallback scene detection with {interval}s intervals")

        # Create scenes at fixed intervals
        timestamps = []
        current_time = 0.0

        while current_time < video_info.duration:
            timestamps.append(current_time)
            current_time += interval

        # Ensure we don't exceed video duration
        if (
            timestamps[-1] < video_info.duration - 1.0
        ):  # Leave at least 1s for final scene
            timestamps.append(video_info.duration)

        scenes = self._create_scenes_from_timestamps(timestamps, video_info.duration)

        # Set lower confidence for fallback scenes
        for scene in scenes:
            scene.confidence = 0.3

        logger.info(f"Created {len(scenes)} fallback scenes with {interval}s intervals")

        return SceneDetectionResult(
            scenes=scenes,
            total_scenes=len(scenes),
            detection_method="fallback",
            threshold_used=threshold,
            video_duration=video_info.duration,
            average_scene_duration=sum(s.duration for s in scenes) / len(scenes)
            if scenes
            else 0.0,
        )

    def _run_with_progress(
        self,
        stream: Any,
        total_duration: float,
        progress_callback: Callable[[float], None],
    ) -> Any:
        """
        Run ffmpeg with progress tracking for scene detection.

        Args:
            stream: ffmpeg stream object
            total_duration: Total duration for progress calculation
            progress_callback: Callback function for progress updates

        Returns:
            ffmpeg result object
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

                    # Parse progress from ffmpeg output
                    if "time=" in line:
                        try:
                            time_str = line.split("time=")[1].split()[0]

                            if ":" in time_str:
                                parts = time_str.split(":")
                                if len(parts) == 3:
                                    hours = float(parts[0])
                                    minutes = float(parts[1])
                                    seconds = float(parts[2])
                                    current_time = hours * 3600 + minutes * 60 + seconds

                                    progress = min(current_time / total_duration, 1.0)
                                    progress_callback(progress)
                        except (ValueError, IndexError):
                            pass

            process.wait()
            progress_callback(1.0)  # Final progress update

            # Create result object with stderr
            class Result:
                def __init__(self, stderr: bytes):
                    self.stderr = stderr

            return Result(process.stderr.read() if process.stderr else b"")

        except Exception as e:
            logger.error(f"Error during scene detection progress tracking: {e}")
            raise

    def get_scene_summary(self, result: SceneDetectionResult) -> dict[str, Any]:
        """
        Get summary statistics for scene detection result.

        Args:
            result: SceneDetectionResult object

        Returns:
            Dictionary with scene statistics
        """
        if not result.scenes:
            return {
                "total_scenes": 0,
                "average_duration": 0.0,
                "shortest_scene": 0.0,
                "longest_scene": 0.0,
                "detection_method": result.detection_method,
                "threshold_used": result.threshold_used,
            }

        durations = [scene.duration for scene in result.scenes]

        return {
            "total_scenes": result.total_scenes,
            "average_duration": result.average_scene_duration,
            "shortest_scene": min(durations),
            "longest_scene": max(durations),
            "detection_method": result.detection_method,
            "threshold_used": result.threshold_used,
            "scene_boundaries": result.scene_boundaries,
        }
