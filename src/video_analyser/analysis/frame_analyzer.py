"""Frame analysis pipeline for comprehensive video frame processing.

This module provides a unified pipeline that processes video frames through
all available analysis components including quality assessment, captioning,
OCR, and object detection.
"""

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from video_analyser.analysis.visual_analyzer import (
    ExtractedFrame,
    FrameExtractor,
    VisualAnalysisResult,
)
from video_analyser.core.exceptions import ErrorCode, VideoProcessingError
from video_analyser.core.progress_tracker import ProgressTracker
from video_analyser.core.scene_detector import SceneDetectionResult
from video_analyser.utils.config import get_config

logger = logging.getLogger(__name__)


class FrameAnalysisStep(BaseModel):
    """Configuration for a single analysis step."""

    name: str
    enabled: bool
    weight: float = 1.0  # Weight for progress tracking

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


class PipelineMetrics(BaseModel):
    """Metrics collected during pipeline execution."""

    total_frames_processed: int
    successful_frames: int
    failed_frames: int

    # Timing metrics
    total_processing_time: float
    average_frame_time: float

    # Step-specific metrics
    step_timings: dict[str, float]
    step_success_rates: dict[str, float]

    # Quality metrics
    average_quality_score: float
    quality_distribution: dict[str, int]

    # Analysis coverage
    frames_with_captions: int
    frames_with_ocr: int
    frames_with_objects: int

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of pipeline metrics."""
        return {
            "success_rate": self.successful_frames / self.total_frames_processed
            if self.total_frames_processed > 0
            else 0,
            "average_processing_time": self.average_frame_time,
            "total_processing_time": self.total_processing_time,
            "quality_score": self.average_quality_score,
            "analysis_coverage": {
                "captions": self.frames_with_captions / self.total_frames_processed
                if self.total_frames_processed > 0
                else 0,
                "ocr": self.frames_with_ocr / self.total_frames_processed
                if self.total_frames_processed > 0
                else 0,
                "objects": self.frames_with_objects / self.total_frames_processed
                if self.total_frames_processed > 0
                else 0,
            },
        }


class FrameAnalysisPipeline:
    """Unified pipeline for analyzing video frames."""

    def __init__(self, config: Any = None):
        """Initialize the frame analysis pipeline."""
        self.config = config or get_config()
        self.frame_extractor = FrameExtractor(config=self.config)

        # Define analysis steps
        self.analysis_steps = [
            FrameAnalysisStep(
                name="quality_assessment",
                enabled=True,  # Always enabled
                weight=0.2,
            ),
            FrameAnalysisStep(
                name="image_captioning",
                enabled=self.config.visual_analysis.enable_captioning,
                weight=0.3,
            ),
            FrameAnalysisStep(
                name="text_extraction",
                enabled=self.config.visual_analysis.enable_ocr,
                weight=0.3,
            ),
            FrameAnalysisStep(
                name="object_detection",
                enabled=self.config.visual_analysis.enable_object_detection,
                weight=0.2,
            ),
        ]

        # Calculate total weight for enabled steps
        self.total_weight = sum(
            step.weight for step in self.analysis_steps if step.enabled
        )

        logger.info(
            f"FrameAnalysisPipeline initialized with steps: "
            f"{[step.name for step in self.analysis_steps if step.enabled]}"
        )

    def analyze_video_frames(
        self,
        video_path: Path,
        scene_result: SceneDetectionResult,
        output_dir: Path | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> tuple[VisualAnalysisResult, PipelineMetrics]:
        """
        Analyze frames from video scenes through the complete pipeline.

        Args:
            video_path: Path to the video file
            scene_result: Scene detection results
            output_dir: Optional directory to save extracted frames
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (VisualAnalysisResult, PipelineMetrics)

        Raises:
            VideoProcessingError: If analysis fails
        """
        start_time = time.time()

        # Create progress tracker
        progress_tracker = ProgressTracker()

        # Create wrapper for the callback to match expected signature
        if progress_callback:

            def wrapper(update: Any):
                # Extract progress and message from update if available
                if hasattr(update, "progress") and hasattr(update, "current_step"):
                    progress_callback(update.progress, update.current_step)
                else:
                    progress_callback(0.0, "Processing...")

            progress_tracker.add_callback(wrapper)

        try:
            # Extract and analyze frames
            logger.info(
                f"Starting frame analysis for {len(scene_result.scenes)} scenes"
            )
            # Start tracking the operation
            progress_tracker.start_operation(
                operation_id="frame_analysis",
                operation_name="Frame Analysis Pipeline",
                total_steps=len(scene_result.scenes),
            )

            # Use frame extractor which already includes all analysis
            visual_result = self.frame_extractor.extract_frames_from_scenes(
                video_path=video_path,
                scene_result=scene_result,
                output_dir=output_dir,
            )

            # Collect metrics from the results
            metrics = self._collect_pipeline_metrics(
                visual_result, time.time() - start_time
            )

            # Complete the operation
            progress_tracker.complete_operation(
                operation_id="frame_analysis",
                details={"frames_extracted": visual_result.total_frames_extracted},
            )

            return visual_result, metrics

        except Exception as e:
            error_msg = f"Frame analysis pipeline failed: {str(e)}"
            logger.error(error_msg)
            progress_tracker.fail_operation(
                operation_id="frame_analysis",
                error=error_msg,
            )

            raise VideoProcessingError(
                message=error_msg,
                error_code=ErrorCode.FRAME_EXTRACTION_FAILED,
                file_path=video_path,
                cause=e,
            ) from e

    def analyze_single_frame(
        self,
        frame: npt.NDArray[np.uint8],
        frame_info: dict[str, Any] | None = None,
    ) -> ExtractedFrame:
        """
        Analyze a single frame through the pipeline.

        Args:
            frame: Frame as numpy array (BGR format)
            frame_info: Optional metadata about the frame

        Returns:
            ExtractedFrame with all analysis results
        """
        frame_info = frame_info or {}

        # Quality assessment (always performed)
        # Access protected methods through the instance - type checker will accept this
        # since we're delegating to the internal frame_extractor
        quality_metrics = self.frame_extractor._assess_frame_quality(frame)  # type: ignore[reportPrivateUsage]

        # Image captioning
        caption_result = None
        if self.config.visual_analysis.enable_captioning:
            caption_result = self.frame_extractor._caption_frame(frame)  # type: ignore[reportPrivateUsage]

        # OCR text extraction
        ocr_result = None
        if self.config.visual_analysis.enable_ocr:
            ocr_result = self.frame_extractor._extract_text_from_frame(frame)  # type: ignore[reportPrivateUsage]

        # Object detection
        object_detection_result = None
        if self.config.visual_analysis.enable_object_detection:
            object_detection_result = self.frame_extractor._detect_objects_in_frame(  # type: ignore[reportPrivateUsage]
                frame
            )

        # Create ExtractedFrame
        height: int
        width: int
        height, width = frame.shape[:2]
        extracted_frame = ExtractedFrame(
            frame_number=frame_info.get("frame_number", 0),
            timestamp=frame_info.get("timestamp", 0.0),
            scene_number=frame_info.get("scene_number", 0),
            width=width,
            height=height,
            quality_metrics=quality_metrics,
            caption_result=caption_result,
            ocr_result=ocr_result,
            object_detection_result=object_detection_result,
        )

        return extracted_frame

    def analyze_frame_from_path(
        self,
        frame_path: Path | str,
        frame_number: int | None = None,
        timestamp: float | None = None,
        scene_number: int | None = None,
    ) -> ExtractedFrame:
        """
        Convenience method to analyze a frame from a file path.

        Args:
            frame_path: Path to the frame image file
            frame_number: Optional frame number
            timestamp: Optional timestamp in video
            scene_number: Optional scene number

        Returns:
            ExtractedFrame with all analysis results
        """
        import cv2

        frame_path = Path(frame_path)
        logger.info(f"Analyzing frame from path: {frame_path}")

        # Load image - cv2.imread returns None on failure despite type stubs
        loaded_frame = cv2.imread(str(frame_path))
        if loaded_frame is None:  # type: ignore[reportUnnecessaryComparison]
            raise VideoProcessingError(
                message=f"Failed to load frame from {frame_path}",
                error_code=ErrorCode.FRAME_EXTRACTION_FAILED,
                file_path=frame_path,
            )
        # cv2.imread returns BGR uint8 image, cast for type checker
        frame: npt.NDArray[np.uint8] = loaded_frame  # type: ignore[assignment]

        # Build frame info
        frame_info = {
            "frame_number": frame_number or 0,
            "timestamp": timestamp or 0.0,
            "scene_number": scene_number or 0,
        }

        # Call the main analysis method
        return self.analyze_single_frame(frame, frame_info)

    def get_enabled_analyses(self) -> list[str]:
        """Get list of enabled analysis types."""
        enabled = ["quality_assessment"]  # Always enabled

        if self.config.visual_analysis.enable_captioning:
            enabled.append("image_captioning")
        if self.config.visual_analysis.enable_ocr:
            enabled.append("text_extraction")
        if self.config.visual_analysis.enable_object_detection:
            enabled.append("object_detection")

        return enabled

    def _collect_pipeline_metrics(
        self,
        visual_result: VisualAnalysisResult,
        total_time: float,
    ) -> PipelineMetrics:
        """Collect metrics from analysis results."""
        total_frames = visual_result.total_frames_extracted
        successful_frames = visual_result.total_frames_extracted
        failed_frames = (
            visual_result.total_frames_processed - visual_result.total_frames_extracted
        )

        # Collect quality metrics
        quality_distribution = visual_result.overall_quality_distribution
        average_quality = visual_result.average_quality_score

        # Count frames with different analysis types
        frames_with_captions = 0
        frames_with_ocr = 0
        frames_with_objects = 0

        for scene_analysis in visual_result.scene_analyses:
            for frame in scene_analysis.frames:
                if frame.caption_result and frame.caption_result.text:
                    frames_with_captions += 1
                if frame.ocr_result and frame.ocr_result.text:
                    frames_with_ocr += 1
                if (
                    frame.object_detection_result
                    and frame.object_detection_result.total_objects > 0
                ):
                    frames_with_objects += 1

        # Estimate step timings (rough approximation)
        average_frame_time = total_time / total_frames if total_frames > 0 else 0
        step_timings = {
            "quality_assessment": average_frame_time * 0.1,
            "image_captioning": average_frame_time * 0.4
            if self.config.visual_analysis.enable_captioning
            else 0,
            "text_extraction": average_frame_time * 0.3
            if self.config.visual_analysis.enable_ocr
            else 0,
            "object_detection": average_frame_time * 0.2
            if self.config.visual_analysis.enable_object_detection
            else 0,
        }

        # Calculate success rates for each step
        step_success_rates = {
            "quality_assessment": 1.0,  # Always succeeds
            "image_captioning": frames_with_captions / total_frames
            if total_frames > 0
            else 0,
            "text_extraction": frames_with_ocr / total_frames
            if total_frames > 0
            else 0,
            "object_detection": frames_with_objects / total_frames
            if total_frames > 0
            else 0,
        }

        return PipelineMetrics(
            total_frames_processed=visual_result.total_frames_processed,
            successful_frames=successful_frames,
            failed_frames=failed_frames,
            total_processing_time=total_time,
            average_frame_time=average_frame_time,
            step_timings=step_timings,
            step_success_rates=step_success_rates,
            average_quality_score=average_quality,
            quality_distribution=quality_distribution,
            frames_with_captions=frames_with_captions,
            frames_with_ocr=frames_with_ocr,
            frames_with_objects=frames_with_objects,
        )

    def generate_analysis_summary(
        self,
        visual_result: VisualAnalysisResult,
        metrics: PipelineMetrics,
    ) -> dict[str, Any]:
        """
        Generate a comprehensive summary of the frame analysis.

        Args:
            visual_result: Visual analysis results
            metrics: Pipeline execution metrics

        Returns:
            Dictionary with analysis summary
        """
        # Get quality report
        quality_report = visual_result.generate_quality_report()

        # Get metrics summary
        metrics_summary = metrics.get_summary()

        # Collect key insights
        insights = []

        # Quality insights
        if metrics.average_quality_score < 0.6:
            insights.append("Overall video quality is below optimal levels")
        if (
            quality_report["summary"]["scenes_with_issues"]
            > visual_result.total_scenes * 0.5
        ):
            insights.append("More than half of the scenes have quality issues")

        # Content insights
        if metrics.frames_with_ocr > 0:
            text_heavy_scenes = sum(
                1
                for scene in visual_result.scene_analyses
                if any(
                    frame.ocr_result and len(frame.ocr_result.blocks) > 5
                    for frame in scene.frames
                )
            )
            if text_heavy_scenes > visual_result.total_scenes * 0.3:
                insights.append(
                    "Video contains significant text content (likely presentation slides)"
                )

        if metrics.frames_with_objects > 0:
            # Layout types come from presentation_classifier; "single-presenter"
            # and "slide-only" both indicate a presentation/lecture frame.
            presentation_layouts = {"single-presenter", "slide-only", "multi-presenter"}
            presentation_scenes = sum(
                1
                for scene in visual_result.scene_analyses
                if any(
                    frame.object_detection_result
                    and frame.object_detection_result.layout_type
                    in presentation_layouts
                    for frame in scene.frames
                )
            )
            if presentation_scenes > visual_result.total_scenes * 0.5:
                insights.append("Video appears to be a presentation or lecture")

        # Create summary with explicit type annotations
        summary: dict[str, Any] = {
            "video_overview": {
                "duration": visual_result.video_duration,
                "total_scenes": visual_result.total_scenes,
                "frames_analyzed": visual_result.total_frames_extracted,
                "analysis_time": metrics.total_processing_time,
            },
            "quality_summary": {
                "average_score": metrics.average_quality_score,
                "distribution": quality_report["quality_distribution"],
                "top_issues": quality_report["top_recommendations"][:3]
                if quality_report["top_recommendations"]
                else [],
            },
            "content_analysis": {
                "has_text_content": metrics.frames_with_ocr > 0,
                "has_visual_elements": metrics.frames_with_objects > 0,
                "has_descriptions": metrics.frames_with_captions > 0,
                "content_type": self._determine_content_type(visual_result, metrics),
            },
            "analysis_coverage": metrics_summary["analysis_coverage"],
            "key_insights": insights,
            "performance": {
                "success_rate": metrics_summary["success_rate"],
                "average_frame_time": metrics_summary["average_processing_time"],
                "enabled_analyses": self.get_enabled_analyses(),
            },
        }

        return summary

    def _determine_content_type(
        self,
        visual_result: VisualAnalysisResult,
        _metrics: PipelineMetrics,
    ) -> str:
        """Determine the primary content type of the video."""
        # Count different layout types detected
        layout_counts: dict[str, int] = {}
        for scene in visual_result.scene_analyses:
            for frame in scene.frames:
                if frame.object_detection_result:
                    layout = frame.object_detection_result.layout_type
                    layout_counts[layout] = layout_counts.get(layout, 0) + 1

        if not layout_counts:
            return "unknown"

        # Find dominant layout type
        dominant_layout: str = max(layout_counts.items(), key=lambda x: x[1])[0]

        # Map presentation_classifier layout names to high-level content types.
        content_type_mapping: dict[str, str] = {
            "slide-only": "presentation",
            "single-presenter": "presentation",
            "multi-presenter": "lecture",
            "audience-only": "audience",
            "empty": "unknown",
            "mixed": "mixed_content",
        }

        return content_type_mapping.get(dominant_layout, "mixed_content")

    def cleanup(self):
        """Clean up resources."""
        if self.frame_extractor:
            self.frame_extractor.cleanup()
        logger.info("FrameAnalysisPipeline resources cleaned up")


def create_frame_analysis_pipeline(config: Any = None) -> FrameAnalysisPipeline:
    """
    Create a new FrameAnalysisPipeline instance.

    Args:
        config: Optional configuration object

    Returns:
        Configured FrameAnalysisPipeline instance
    """
    return FrameAnalysisPipeline(config=config)
