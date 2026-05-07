"""Pipeline coordinator for orchestrating complete video analysis workflows."""

import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from video_analyser.analysis.frame_analyzer import (
    ExtractedFrame,
    FrameAnalysisPipeline,
)
from video_analyser.analysis.speech_analyzer import SpeechAnalyzer
from video_analyser.analysis.transcriber import WhisperTranscriber
from video_analyser.core.audio_extractor import AudioExtractor, AudioInfo
from video_analyser.core.exceptions import (
    AudioProcessingError,
    ErrorCode,
    VideoProcessingError,
    get_user_friendly_message,
)
from video_analyser.core.progress_tracker import (
    CompositeProgressTracker,
    ProgressTracker,
)
from video_analyser.core.scene_detector import Scene, SceneDetectionResult, SceneDetector
from video_analyser.core.video_processor import FrameInfo, VideoInfo, VideoProcessor
from video_analyser.reports.html_renderer import HTMLRenderer
from video_analyser.reports.report_generator import ReportGenerator
from video_analyser.utils.config import get_config

logger = logging.getLogger(__name__)


class VideoAnalysisResult:
    """Complete video analysis result containing all processing outputs."""

    def __init__(
        self,
        video_info: VideoInfo,
        audio_info: AudioInfo | None = None,
        scene_result: SceneDetectionResult | None = None,
        frame_infos: list[FrameInfo] | None = None,
        processing_time: float = 0.0,
        success: bool = True,
        error_message: str | None = None,
    ):
        """
        Initialize video analysis result.

        Args:
            video_info: Video file information
            audio_info: Extracted audio information
            scene_result: Scene detection results
            frame_infos: Extracted frame information
            processing_time: Total processing time in seconds
            success: Whether analysis completed successfully
            error_message: Error message if analysis failed
        """
        self.video_info = video_info
        self.audio_info = audio_info
        self.scene_result = scene_result
        self.frame_infos = frame_infos or []
        self.processing_time = processing_time
        self.success = success
        self.error_message = error_message
        self.errors: list[VideoProcessingError] = []  # Detailed error information

    def add_error(self, error: VideoProcessingError) -> None:
        """Add an error to the result."""
        self.errors.append(error)
        if not self.error_message:
            self.error_message = get_user_friendly_message(error)

    def get_error_summary(self) -> dict[str, Any]:
        """Get summary of all errors that occurred."""
        return {
            "total_errors": len(self.errors),
            "error_types": [error.error_code.value for error in self.errors],
            "errors": [error.to_dict() for error in self.errors],
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "video_info": {
                "file_path": str(self.video_info.file_path),
                "duration": self.video_info.duration,
                "width": self.video_info.width,
                "height": self.video_info.height,
                "fps": self.video_info.fps,
                "format": self.video_info.format,
                "size_mb": self.video_info.size_mb,
                "codec": self.video_info.codec,
            },
            "audio_info": {
                "file_path": str(self.audio_info.file_path)
                if self.audio_info
                else None,
                "duration": self.audio_info.duration if self.audio_info else None,
                "sample_rate": self.audio_info.sample_rate if self.audio_info else None,
                "channels": self.audio_info.channels if self.audio_info else None,
                "size_mb": self.audio_info.size_mb if self.audio_info else None,
                "format": self.audio_info.format if self.audio_info else None,
            }
            if self.audio_info
            else None,
            "scene_result": {
                "total_scenes": self.scene_result.total_scenes
                if self.scene_result
                else 0,
                "detection_method": self.scene_result.detection_method
                if self.scene_result
                else None,
                "threshold_used": self.scene_result.threshold_used
                if self.scene_result
                else None,
                "video_duration": self.scene_result.video_duration
                if self.scene_result
                else None,
                "average_scene_duration": self.scene_result.average_scene_duration
                if self.scene_result
                else None,
                "scenes": [
                    {
                        "start_time": scene.start_time,
                        "end_time": scene.end_time,
                        "duration": scene.duration,
                        "scene_number": scene.scene_number,
                        "confidence": scene.confidence,
                    }
                    for scene in self.scene_result.scenes
                ]
                if self.scene_result
                else [],
            }
            if self.scene_result
            else None,
            "frame_infos": [
                {
                    "frame_path": str(frame.frame_path),
                    "timestamp": frame.timestamp,
                    "scene_number": frame.scene_number,
                    "width": frame.width,
                    "height": frame.height,
                    "size_kb": frame.size_kb,
                    "format": frame.format,
                }
                for frame in self.frame_infos
            ],
            "processing_time": self.processing_time,
            "success": self.success,
            "error_message": self.error_message,
            "error_summary": self.get_error_summary() if self.errors else None,
        }


class PipelineCoordinator:
    """Coordinates the complete video analysis pipeline with progress tracking."""

    def __init__(
        self, config: Any = None, progress_tracker: ProgressTracker | None = None
    ):
        """
        Initialize pipeline coordinator.

        Args:
            config: Configuration object
            progress_tracker: Progress tracker instance
        """
        self.config = config or get_config()
        self.progress_tracker = progress_tracker

        # Initialize processing components
        self.video_processor = VideoProcessor(self.config)
        self.audio_extractor = AudioExtractor(self.config)
        self.scene_detector = SceneDetector(self.config)

        logger.info("PipelineCoordinator initialized")

    def analyze_video(
        self,
        video_path: Path | str,
        extract_audio: bool = True,
        detect_scenes: bool = True,
        extract_frames: bool = True,
        output_dir: Path | str | None = None,
    ) -> VideoAnalysisResult:
        """
        Perform complete video analysis with progress tracking.

        Args:
            video_path: Path to video file
            extract_audio: Whether to extract audio
            detect_scenes: Whether to detect scenes
            extract_frames: Whether to extract frames
            output_dir: Optional output directory for extracted files

        Returns:
            VideoAnalysisResult with all analysis outputs
        """
        video_path = Path(video_path)
        workflow_id = f"video_analysis_{uuid.uuid4().hex[:8]}"

        # Set up progress tracking
        composite_tracker = None
        if self.progress_tracker:
            composite_tracker = CompositeProgressTracker(self.progress_tracker)

            # Define workflow operations with weights
            operations = [("validate", "Validating video file", 0.05)]

            if extract_audio:
                operations.append(("audio", "Extracting audio", 0.25))

            if detect_scenes:
                operations.append(("scenes", "Detecting scenes", 0.35))

            if extract_frames and detect_scenes:
                operations.append(("frames", "Extracting frames", 0.35))

            # Normalize weights
            total_weight = sum(weight for _, _, weight in operations)
            operations = [
                (op_id, name, weight / total_weight)
                for op_id, name, weight in operations
            ]

            composite_tracker.start_workflow(
                workflow_id=workflow_id,
                workflow_name=f"Analyzing {video_path.name}",
                operations=operations,
            )

        video_info = None
        try:
            # Step 1: Validate video file
            progress_callback = (
                composite_tracker.start_next_operation() if composite_tracker else None
            )
            logger.info(f"Starting video analysis: {video_path}")

            if progress_callback:
                progress_callback(0.5)

            video_info = self.video_processor.validate_file(video_path)

            if progress_callback:
                progress_callback(1.0)

            if composite_tracker:
                composite_tracker.complete_current_operation()

            logger.info(
                f"Video validated: {video_info.duration:.1f}s, {video_info.width}x{video_info.height}"
            )

            # Initialize result
            result = VideoAnalysisResult(video_info=video_info)

            # Step 2: Extract audio (if requested)
            audio_info = None
            if extract_audio:
                progress_callback = (
                    composite_tracker.start_next_operation()
                    if composite_tracker
                    else None
                )

                try:
                    audio_info = self.audio_extractor.extract_audio(
                        video_info=video_info,
                        output_path=Path(output_dir) / f"{video_path.stem}_audio.wav"
                        if output_dir
                        else None,
                        progress_callback=progress_callback,
                    )
                    result.audio_info = audio_info

                    if composite_tracker:
                        composite_tracker.complete_current_operation()

                    logger.info(
                        f"Audio extracted: {audio_info.duration:.1f}s, {audio_info.sample_rate}Hz"
                    )

                except AudioProcessingError as e:
                    # Handle specific audio processing errors
                    if e.error_code == ErrorCode.NO_AUDIO_STREAM:
                        # No audio stream - this is okay, continue without audio
                        logger.warning(f"No audio stream found: {e.message}")
                        if progress_callback:
                            progress_callback(1.0)
                        if composite_tracker:
                            composite_tracker.complete_current_operation()
                    else:
                        # Other audio errors should be logged but not stop processing
                        logger.error(f"Audio extraction failed: {e}")
                        result.add_error(e)
                        if progress_callback:
                            progress_callback(1.0)
                        if composite_tracker:
                            composite_tracker.complete_current_operation()

            # Step 3: Detect scenes (if requested)
            scene_result = None
            if detect_scenes:
                progress_callback = (
                    composite_tracker.start_next_operation()
                    if composite_tracker
                    else None
                )

                scene_result = self.scene_detector.detect_scenes(
                    video_info=video_info, progress_callback=progress_callback
                )
                result.scene_result = scene_result

                if composite_tracker:
                    composite_tracker.complete_current_operation()

                logger.info(
                    f"Scenes detected: {scene_result.total_scenes} scenes using {scene_result.detection_method}"
                )

            # Step 4: Extract frames (if requested and we have scenes)
            frame_infos = []
            if extract_frames and scene_result and scene_result.scenes:
                progress_callback = (
                    composite_tracker.start_next_operation()
                    if composite_tracker
                    else None
                )

                # Convert scenes to format expected by frame extraction
                scene_tuples = [
                    (scene.start_time, scene.end_time, scene.scene_number)
                    for scene in scene_result.scenes
                ]

                frame_infos = self.video_processor.extract_frames_from_scenes(
                    video_info=video_info,
                    scenes=scene_tuples,
                    output_dir=Path(output_dir) / "frames" if output_dir else None,
                    progress_callback=progress_callback,
                )
                result.frame_infos = frame_infos

                if composite_tracker:
                    composite_tracker.complete_current_operation()

                logger.info(
                    f"Frames extracted: {len(frame_infos)} frames from {len(scene_result.scenes)} scenes"
                )

            # Mark workflow as complete
            if composite_tracker and self.progress_tracker:
                # Final progress update
                self.progress_tracker.update_progress(
                    operation_id=workflow_id,
                    progress=1.0,
                    current_step="Analysis complete",
                )
                self.progress_tracker.complete_operation(
                    operation_id=workflow_id,
                    details={
                        "total_scenes": scene_result.total_scenes
                        if scene_result
                        else 0,
                        "total_frames": len(frame_infos),
                        "has_audio": audio_info is not None,
                        "video_duration": video_info.duration,
                    },
                )

            logger.info(f"Video analysis complete: {video_path.name}")
            return result

        except VideoProcessingError as e:
            # Handle our custom video processing errors
            error_msg = get_user_friendly_message(e)
            logger.error(f"Video analysis failed: {e}")

            if composite_tracker:
                composite_tracker.fail_workflow(error_msg)

            # If video_info is None, create minimal video info from path
            if video_info is None:
                video_info = VideoInfo(
                    file_path=video_path,
                    duration=0.0,
                    width=0,
                    height=0,
                    fps=0.0,
                    format="unknown",
                    size_mb=0.0,
                    codec="unknown",
                )

            result = VideoAnalysisResult(
                video_info=video_info,
                success=False,
                error_message=error_msg,
            )
            result.add_error(e)
            return result

        except Exception as e:
            # Handle unexpected errors
            error_msg = f"Unexpected error during video analysis: {str(e)}"
            logger.error(error_msg)

            if composite_tracker:
                composite_tracker.fail_workflow(error_msg)

            # Create a generic video processing error
            generic_error = VideoProcessingError(
                message=str(e), file_path=video_path, cause=e
            )

            # If video_info is None, create minimal video info from path
            if video_info is None:
                video_info = VideoInfo(
                    file_path=video_path,
                    duration=0.0,
                    width=0,
                    height=0,
                    fps=0.0,
                    format="unknown",
                    size_mb=0.0,
                    codec="unknown",
                )

            result = VideoAnalysisResult(
                video_info=video_info,
                success=False,
                error_message=error_msg,
            )
            result.add_error(generic_error)
            return result

    def analyze_video_batch(
        self,
        video_paths: list[Path | str],
        extract_audio: bool = True,
        detect_scenes: bool = True,
        extract_frames: bool = True,
        output_dir: Path | str | None = None,
    ) -> list[VideoAnalysisResult]:
        """
        Analyze multiple videos with progress tracking.

        Args:
            video_paths: List of paths to video files
            extract_audio: Whether to extract audio
            detect_scenes: Whether to detect scenes
            extract_frames: Whether to extract frames
            output_dir: Optional output directory for extracted files

        Returns:
            List of VideoAnalysisResult objects
        """
        batch_id = f"batch_analysis_{uuid.uuid4().hex[:8]}"
        results: list[VideoAnalysisResult] = []

        # Set up batch progress tracking
        if self.progress_tracker:
            self.progress_tracker.start_operation(
                operation_id=batch_id,
                operation_name=f"Analyzing {len(video_paths)} videos",
                total_steps=len(video_paths),
                details={"video_count": len(video_paths)},
            )

        try:
            for i, video_path in enumerate(video_paths):
                logger.info(
                    f"Processing video {i + 1}/{len(video_paths)}: {video_path}"
                )

                # Update batch progress
                if self.progress_tracker:
                    self.progress_tracker.update_progress(
                        operation_id=batch_id,
                        progress=i / len(video_paths),
                        current_step=f"Processing {Path(video_path).name}",
                        current_step_number=i + 1,
                    )

                # Analyze individual video
                result = self.analyze_video(
                    video_path=video_path,
                    extract_audio=extract_audio,
                    detect_scenes=detect_scenes,
                    extract_frames=extract_frames,
                    output_dir=Path(output_dir) / Path(video_path).stem
                    if output_dir
                    else None,
                )
                results.append(result)

                if not result.success:
                    logger.error(
                        f"Failed to analyze {video_path}: {result.error_message}"
                    )

            # Complete batch processing
            if self.progress_tracker:
                successful_results: list[VideoAnalysisResult] = [
                    r for r in results if r.success
                ]
                self.progress_tracker.complete_operation(
                    operation_id=batch_id,
                    details={
                        "total_videos": len(video_paths),
                        "successful": len(successful_results),
                        "failed": len(video_paths) - len(successful_results),
                    },
                )

            logger.info(f"Batch analysis complete: {len(results)} videos processed")
            return results

        except Exception as e:
            error_msg = f"Batch analysis failed: {e}"
            logger.error(error_msg)

            if self.progress_tracker:
                self.progress_tracker.fail_operation(batch_id, error_msg)

            return results  # Return partial results

    def analyze_speech(
        self,
        audio_path: Path,
        scene_result: SceneDetectionResult | None = None,
        _progress_callback: Callable[[float], None] | None = None,
    ) -> dict[str, Any]:
        """
        Perform speech analysis on extracted audio.

        Args:
            audio_path: Path to extracted audio file
            scene_result: Optional scene detection result for per-scene analysis
            progress_callback: Optional progress callback

        Returns:
            Dictionary containing transcription and speech analysis results
        """
        logger.info(f"Starting speech analysis: {audio_path}")

        try:
            # Step 1: Transcribe audio
            transcriber = WhisperTranscriber(config=self.config)
            transcription_result = transcriber.transcribe_from_path(
                audio_path=audio_path, language=self.config.transcription.language
            )

            logger.info(
                f"Transcription complete: {len(transcription_result.segments)} segments"
            )

            # Step 2: Analyze speech patterns (only if we have scene_result)
            speech_analysis = None
            if scene_result:
                speech_analyzer = SpeechAnalyzer(config=self.config)
                speech_analysis = speech_analyzer.analyze_speech(
                    transcription_result=transcription_result, scene_result=scene_result
                )
            else:
                logger.warning("No scene result provided, skipping speech analysis")

            logger.info("Speech analysis complete")

            return {
                "transcription": transcription_result,
                "speech_analysis": speech_analysis,
            }

        except Exception as e:
            logger.error(f"Speech analysis failed: {e}")
            raise

    def analyze_frames(
        self,
        frame_paths: list[Path],
        _scenes: list[Scene] | None = None,
        _progress_callback: Callable[[float], None] | None = None,
    ) -> dict[str, Any]:
        """
        Perform visual analysis on extracted frames.

        Args:
            frame_paths: List of paths to extracted frame images
            scenes: Optional list of scene information
            progress_callback: Optional progress callback

        Returns:
            Dictionary containing visual analysis results
        """
        logger.info(f"Starting visual analysis: {len(frame_paths)} frames")

        try:
            # Initialize frame analyzer
            frame_analyzer = FrameAnalysisPipeline(config=self.config)

            # Analyze all frames
            frame_results: list[ExtractedFrame] = []
            for i, frame_path in enumerate(frame_paths):
                logger.debug(
                    f"Analyzing frame {i + 1}/{len(frame_paths)}: {frame_path}"
                )

                result = frame_analyzer.analyze_frame_from_path(
                    frame_path=frame_path, frame_number=i + 1
                )
                frame_results.append(result)

                if _progress_callback:
                    _progress_callback((i + 1) / len(frame_paths))

            logger.info(f"Visual analysis complete: {len(frame_results)} frames")

            return {"frame_analyses": frame_results}

        except Exception as e:
            logger.error(f"Visual analysis failed: {e}")
            raise

    def generate_reports(
        self,
        video_info: VideoInfo,
        audio_info: AudioInfo | None,
        scene_result: SceneDetectionResult | None,
        speech_analysis: dict[str, Any] | None,
        visual_analysis: dict[str, Any] | None,
        output_dir: Path,
    ) -> dict[str, Path]:
        """
        Generate analysis reports in JSON and HTML formats.

        Args:
            video_info: Video file information
            audio_info: Audio extraction information
            scene_result: Scene detection results
            speech_analysis: Speech analysis results
            visual_analysis: Visual analysis results
            output_dir: Directory to save reports

        Returns:
            Dictionary mapping format names to report file paths
        """
        logger.info(f"Generating reports in {output_dir}")

        try:
            # Initialize report generator
            report_generator = ReportGenerator(config=self.config)

            # Build complete analysis data
            analysis_data = {
                "video_info": video_info,
                "audio_info": audio_info,
                "scenes": scene_result.scenes if scene_result else [],
                "transcription": speech_analysis.get("transcription")
                if speech_analysis
                else None,
                "speech_metrics": speech_analysis.get("speech_analysis")
                if speech_analysis
                else None,
                "frame_analyses": visual_analysis.get("frame_analyses")
                if visual_analysis
                else [],
            }

            # Generate report
            report = report_generator.generate_report(analysis_data)

            # Save JSON report
            json_path = output_dir / "analysis.json"
            report_generator.save_json(report, json_path)
            logger.info(f"JSON report saved: {json_path}")

            # Generate and save HTML report
            html_renderer = HTMLRenderer()
            html_content = html_renderer.render_report(report)
            html_path = output_dir / "analysis.html"
            html_renderer.save_html(html_content, html_path)
            logger.info(f"HTML report saved: {html_path}")

            return {"json": json_path, "html": html_path}

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise


def create_pipeline_coordinator(
    progress_tracker: ProgressTracker | None = None,
) -> PipelineCoordinator:
    """
    Create a new pipeline coordinator instance.

    Args:
        progress_tracker: Optional progress tracker instance

    Returns:
        Configured PipelineCoordinator
    """
    return PipelineCoordinator(progress_tracker=progress_tracker)
