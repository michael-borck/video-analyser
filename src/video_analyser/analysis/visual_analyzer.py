"""Visual analysis for frame extraction and quality assessment.

This module provides frame extraction capabilities with quality assessment
including blur detection, contrast analysis, and lighting evaluation.

Per-frame image analysis (captioning, OCR, object detection) is delegated to
the ``image-analyser`` package (>=0.1.3, which surfaces real backend signals
for caption tokens / cost and OCR average confidence). Captions and OCR are
returned as image-analyser's native ``Caption`` / ``Ocr`` types directly;
object detection is post-processed by ``presentation_classifier.classify_objects``
into a presentation-domain ``PresentationLayout`` (presenter / screen / slide /
audience aggregations).
"""

import contextlib
import logging
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from image_analyser import ImageAnalyser
from image_analyser.schemas import AnalysisResult as ImageAnalysisResult
from image_analyser.schemas import Caption, Ocr
from numpy.typing import NDArray
from PIL import Image as PILImage
from pydantic import BaseModel

from video_analyser.analysis.error_handling import (
    ErrorRecoveryContext,
    handle_corrupt_frame,
    validate_image,
)
from video_analyser.analysis.presentation_classifier import (
    PresentationLayout,
    classify_objects,
)
from video_analyser.core.exceptions import ErrorCode, VideoProcessingError
from video_analyser.core.scene_detector import SceneDetectionResult
from video_analyser.utils.config import get_config

logger = logging.getLogger(__name__)


class FrameQualityMetrics(BaseModel):
    """Quality metrics for a single frame."""

    # Blur assessment
    blur_score: float  # Laplacian variance (higher = sharper)
    blur_category: str  # excellent, good, fair, poor

    # Contrast assessment
    contrast_score: float  # RMS contrast (0-100+)
    contrast_category: str  # excellent, good, fair, poor

    # Brightness assessment
    brightness_score: float  # Mean brightness (0-255)
    brightness_category: str  # excellent, good, fair, poor

    # Overall quality
    overall_quality_score: float  # Weighted combination (0-1)
    overall_quality_category: str  # excellent, good, fair, poor

    # Technical details
    histogram_metrics: dict[str, Any]  # Detailed histogram analysis
    sharpness_details: dict[str, Any]  # Edge detection details

    # Additional quality metrics
    color_metrics: dict[str, Any]  # Color distribution and saturation
    noise_metrics: dict[str, Any]  # Noise level estimation
    composition_metrics: dict[str, Any]  # Rule of thirds, symmetry

    # Quality report summary
    quality_report: dict[str, Any]  # Comprehensive quality report


class ExtractedFrame(BaseModel):
    """Information about an extracted frame."""

    frame_number: int
    timestamp: float  # Time in video (seconds)
    scene_number: int
    file_path: Path | None = None  # Path to saved frame file
    width: int
    height: int
    quality_metrics: FrameQualityMetrics

    # Image captioning results (image-analyser native type)
    caption_result: Caption | None = None

    # OCR results (image-analyser native type)
    ocr_result: Ocr | None = None

    # Object detection results (presentation-domain classification over COCO labels)
    object_detection_result: PresentationLayout | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "scene_number": self.scene_number,
            "file_path": str(self.file_path) if self.file_path else None,
            "width": self.width,
            "height": self.height,
            "quality_metrics": self.quality_metrics.model_dump(),
            "caption_result": self.caption_result.model_dump()
            if self.caption_result
            else None,
            "ocr_result": self.ocr_result.model_dump() if self.ocr_result else None,
            "object_detection_result": self.object_detection_result.model_dump()
            if self.object_detection_result
            else None,
        }

    def get_quality_summary(self) -> dict[str, Any]:
        """Get a summary of quality metrics for reporting."""
        return {
            "overall_score": self.quality_metrics.overall_quality_score,
            "overall_category": self.quality_metrics.overall_quality_category,
            "blur_category": self.quality_metrics.blur_category,
            "contrast_category": self.quality_metrics.contrast_category,
            "brightness_category": self.quality_metrics.brightness_category,
            "timestamp": self.timestamp,
            "frame_number": self.frame_number,
        }


class SceneFrameAnalysis(BaseModel):
    """Frame analysis results for a single scene."""

    scene_number: int
    start_time: float
    end_time: float
    duration: float

    # Extracted frames
    frames: list[ExtractedFrame]
    total_frames_extracted: int

    # Quality summary
    best_frame: ExtractedFrame | None = None  # Highest quality frame
    average_quality_score: float
    quality_distribution: dict[str, int]  # Count by quality category

    # Technical summary
    total_frames_processed: int
    frames_filtered_by_quality: int
    extraction_success_rate: float

    def get_frames_by_quality(self, category: str) -> list[ExtractedFrame]:
        """Get frames matching a specific quality category."""
        return [
            frame
            for frame in self.frames
            if frame.quality_metrics.overall_quality_category == category
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            "scene_number": self.scene_number,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "frames": [frame.to_dict() for frame in self.frames],
            "total_frames_extracted": self.total_frames_extracted,
            "best_frame": self.best_frame.to_dict() if self.best_frame else None,
            "average_quality_score": self.average_quality_score,
            "quality_distribution": self.quality_distribution,
            "total_frames_processed": self.total_frames_processed,
            "frames_filtered_by_quality": self.frames_filtered_by_quality,
            "extraction_success_rate": self.extraction_success_rate,
        }

    def get_quality_report(self) -> dict[str, Any]:
        """Generate a quality report for this scene."""
        return {
            "scene_number": self.scene_number,
            "duration": self.duration,
            "frames_analyzed": self.total_frames_extracted,
            "average_quality": {
                "score": self.average_quality_score,
                "category": self._score_to_category(self.average_quality_score),
            },
            "quality_breakdown": self.quality_distribution,
            "best_frame_timestamp": self.best_frame.timestamp
            if self.best_frame
            else None,
            "issues": self._identify_quality_issues(),
            "recommendations": self._generate_recommendations(),
        }

    def _score_to_category(self, score: float) -> str:
        """Convert quality score to category."""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"

    def _identify_quality_issues(self) -> list[str]:
        """Identify quality issues in the scene."""
        issues: list[str] = []

        # Check for poor quality frames
        poor_frames = int(self.quality_distribution.get("poor", 0) or 0)
        if poor_frames > 0:
            issues.append(f"{poor_frames} frame(s) with poor quality")

        # Check for fair quality frames
        fair_frames = int(self.quality_distribution.get("fair", 0) or 0)
        if fair_frames > self.total_frames_extracted * 0.5:
            issues.append("Majority of frames have only fair quality")

        # Check average quality
        if self.average_quality_score < 0.4:
            issues.append("Overall scene quality is below acceptable threshold")

        # Check extraction success rate
        if self.extraction_success_rate < 0.8:
            issues.append(
                f"Low extraction success rate: {self.extraction_success_rate:.1%}"
            )

        return issues

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on quality analysis."""
        recommendations: list[str] = []

        if self.average_quality_score < 0.6:
            recommendations.append("Consider improving video recording quality")

        # Analyze specific quality issues from frames
        if self.frames:
            blur_issues = sum(
                1
                for f in self.frames
                if f.quality_metrics.blur_category in ["poor", "fair"]
            )
            contrast_issues = sum(
                1
                for f in self.frames
                if f.quality_metrics.contrast_category in ["poor", "fair"]
            )
            brightness_issues = sum(
                1
                for f in self.frames
                if f.quality_metrics.brightness_category in ["poor", "fair"]
            )

            if blur_issues > len(self.frames) * 0.5:
                recommendations.append("Focus on camera stability to reduce blur")
            if contrast_issues > len(self.frames) * 0.5:
                recommendations.append(
                    "Improve lighting contrast in recording environment"
                )
            if brightness_issues > len(self.frames) * 0.5:
                recommendations.append("Adjust lighting levels for better visibility")

        return recommendations


class VisualAnalysisResult(BaseModel):
    """Complete visual analysis results."""

    # Overall metrics
    total_scenes: int
    total_frames_extracted: int
    total_frames_processed: int
    overall_success_rate: float

    # Scene-by-scene analysis
    scene_analyses: list[SceneFrameAnalysis]

    # Quality summary across all frames
    overall_quality_distribution: dict[str, int]
    average_quality_score: float
    best_frames_per_scene: list[ExtractedFrame]

    # Processing metadata
    video_duration: float
    extraction_method: str
    processing_time: float

    def get_scene_analysis(self, scene_number: int) -> SceneFrameAnalysis | None:
        """Get analysis for a specific scene."""
        for analysis in self.scene_analyses:
            if analysis.scene_number == scene_number:
                return analysis
        return None

    def get_all_frames(self) -> list[ExtractedFrame]:
        """Get all extracted frames across all scenes."""
        all_frames: list[ExtractedFrame] = []
        for scene_analysis in self.scene_analyses:
            all_frames.extend(scene_analysis.frames)
        return all_frames

    def get_frames_by_quality(self, category: str) -> list[ExtractedFrame]:
        """Get all frames matching a specific quality category."""
        matching_frames: list[ExtractedFrame] = []
        for scene_analysis in self.scene_analyses:
            matching_frames.extend(scene_analysis.get_frames_by_quality(category))
        return matching_frames

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            "total_scenes": self.total_scenes,
            "total_frames_extracted": self.total_frames_extracted,
            "total_frames_processed": self.total_frames_processed,
            "overall_success_rate": self.overall_success_rate,
            "scene_analyses": [analysis.to_dict() for analysis in self.scene_analyses],
            "overall_quality_distribution": self.overall_quality_distribution,
            "average_quality_score": self.average_quality_score,
            "best_frames_per_scene": [
                frame.to_dict() for frame in self.best_frames_per_scene
            ],
            "video_duration": self.video_duration,
            "extraction_method": self.extraction_method,
            "processing_time": self.processing_time,
        }

    def generate_quality_report(self) -> dict[str, Any]:
        """Generate comprehensive quality report for the entire video."""
        scene_reports = [
            analysis.get_quality_report() for analysis in self.scene_analyses
        ]

        # Calculate aggregate metrics
        total_issues = sum(len(report["issues"]) for report in scene_reports)
        scenes_with_issues = sum(1 for report in scene_reports if report["issues"])

        # Generate overall recommendations
        all_recommendations: list[str] = []
        recommendation_counts: dict[str, int] = {}
        for report in scene_reports:
            for rec in report["recommendations"]:
                recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
                if rec not in all_recommendations:
                    all_recommendations.append(rec)

        # Sort recommendations by frequency
        sorted_recommendations: list[str] = sorted(
            all_recommendations,
            key=lambda x: recommendation_counts.get(x, 0),
            reverse=True,
        )[:5]  # Top 5 recommendations

        return {
            "summary": {
                "total_scenes": self.total_scenes,
                "total_frames_analyzed": self.total_frames_extracted,
                "overall_quality_score": self.average_quality_score,
                "overall_quality_category": self._score_to_category(
                    self.average_quality_score
                ),
                "processing_success_rate": self.overall_success_rate,
                "scenes_with_issues": scenes_with_issues,
                "total_quality_issues": total_issues,
            },
            "quality_distribution": self.overall_quality_distribution,
            "scene_quality_reports": scene_reports,
            "top_recommendations": sorted_recommendations,
            "technical_details": {
                "video_duration": self.video_duration,
                "extraction_method": self.extraction_method,
                "processing_time": self.processing_time,
                "frames_per_scene": self.total_frames_extracted / self.total_scenes
                if self.total_scenes > 0
                else 0,
            },
            "best_frames": [
                {
                    "scene": frame.scene_number,
                    "timestamp": frame.timestamp,
                    "quality_score": frame.quality_metrics.overall_quality_score,
                }
                for frame in self.best_frames_per_scene
            ],
        }

    def _score_to_category(self, score: float) -> str:
        """Convert quality score to category."""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"


class FrameExtractor:
    """Frame extraction and quality assessment for video analysis."""

    def __init__(self, config: Any = None):
        """Initialize frame extractor with configuration."""
        self.config = config or get_config()
        # Single delegated image analyser; lazily constructed so callers that
        # never enable any image analysis pay no startup cost.
        self.image_analyser: ImageAnalyser | None = None
        # Legacy attributes preserved for tests that still patch them; they no
        # longer hold per-engine state but allow `cleanup()` and assertions to
        # behave the same way they did before the image-analyser delegation.
        self.captioner: Any = None
        self.ocr_detector: Any = None
        self.object_detector: Any = None
        logger.info("FrameExtractor initialized")

    # ------------------------------------------------------------------
    # image-analyser delegation
    # ------------------------------------------------------------------

    def _get_image_analyser(self) -> ImageAnalyser:
        """Return the lazily-constructed shared ImageAnalyser instance."""
        if self.image_analyser is None:
            visual = self.config.visual_analysis
            # Build the `only` list dynamically so we never run analyses the
            # user has disabled; the always-on identity / quality / colour /
            # hashing / metadata / barcode tiers are skipped to keep per-frame
            # cost low (those are useful for static images, less so for
            # already-quality-assessed video frames).
            only: list[str] = []
            if visual.enable_captioning:
                only.append("caption")
            if visual.enable_ocr:
                only.append("ocr")
            if visual.enable_object_detection:
                only.append("objects")
            if not only:
                # ImageAnalyser requires at least one tier; pass a harmless
                # selection so construction doesn't fail. Callers will simply
                # never invoke `.analyse()` in this case.
                only = ["objects"]
            # Map the configured caption backend to image-analyser's env-driven
            # selector ("local" or "api").
            backend = "api" if visual.captioning_backend == "api" else "local"
            self.image_analyser = ImageAnalyser(only=only, caption_backend=backend)
        return self.image_analyser

    def _analyse_frame(self, frame: NDArray[np.uint8]) -> ImageAnalysisResult:
        """Run image-analyser on a BGR OpenCV frame and return its AnalysisResult."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(rgb_frame)
        return self._get_image_analyser().analyse(pil)

    def extract_frames_from_scenes(
        self,
        video_path: Path,
        scene_result: SceneDetectionResult,
        output_dir: Path | None = None,
    ) -> VisualAnalysisResult:
        """
        Extract representative frames from each scene with quality assessment.

        Args:
            video_path: Path to the video file
            scene_result: Scene detection results
            output_dir: Directory to save extracted frames (optional)

        Returns:
            VisualAnalysisResult with extracted frames and quality metrics

        Raises:
            VideoProcessingError: If frame extraction fails
        """
        import time

        start_time = time.time()

        # Validate input
        if not video_path.exists():
            raise VideoProcessingError(
                message=f"Video file not found: {video_path}",
                error_code=ErrorCode.FILE_NOT_FOUND,
                file_path=video_path,
            )

        logger.info(
            f"Extracting frames from {len(scene_result.scenes)} scenes: {video_path.name}"
        )

        cap: cv2.VideoCapture | None = None
        try:
            # Open video capture
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise VideoProcessingError(
                    message=f"Failed to open video file: {video_path}",
                    error_code=ErrorCode.VIDEO_READ_ERROR,
                    file_path=video_path,
                )

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0  # Default fallback
                logger.warning(f"Invalid FPS detected, using fallback: {fps}")

            scene_analyses: list[SceneFrameAnalysis] = []
            total_frames_extracted = 0
            total_frames_processed = 0
            best_frames_per_scene: list[ExtractedFrame] = []

            # Process each scene
            for scene in scene_result.scenes:
                logger.debug(
                    f"Processing scene {scene.scene_number}: {scene.start_time:.1f}s - {scene.end_time:.1f}s"
                )

                scene_analysis = self._extract_frames_from_scene(
                    cap, scene, fps, output_dir, video_path.stem
                )

                scene_analyses.append(scene_analysis)
                total_frames_extracted += scene_analysis.total_frames_extracted
                total_frames_processed += scene_analysis.total_frames_processed

                if scene_analysis.best_frame:
                    best_frames_per_scene.append(scene_analysis.best_frame)

            cap.release()

            # Calculate overall statistics
            overall_success_rate = (
                total_frames_extracted / total_frames_processed
                if total_frames_processed > 0
                else 0.0
            )

            # Aggregate quality distribution
            overall_quality_distribution = {
                "excellent": 0,
                "good": 0,
                "fair": 0,
                "poor": 0,
            }
            total_quality_score = 0.0
            total_frames_for_avg = 0

            for scene_analysis in scene_analyses:
                for category, count in scene_analysis.quality_distribution.items():
                    overall_quality_distribution[category] = (
                        overall_quality_distribution.get(category, 0) + count
                    )

                for frame in scene_analysis.frames:
                    total_quality_score += frame.quality_metrics.overall_quality_score
                    total_frames_for_avg += 1

            average_quality_score: float = (
                total_quality_score / total_frames_for_avg
                if total_frames_for_avg > 0
                else 0.0
            )

            processing_time = time.time() - start_time

            result = VisualAnalysisResult(
                total_scenes=len(scene_result.scenes),
                total_frames_extracted=total_frames_extracted,
                total_frames_processed=total_frames_processed,
                overall_success_rate=overall_success_rate,
                scene_analyses=scene_analyses,
                overall_quality_distribution=overall_quality_distribution,
                average_quality_score=average_quality_score,
                best_frames_per_scene=best_frames_per_scene,
                video_duration=scene_result.video_duration,
                extraction_method="scene_based",
                processing_time=processing_time,
            )

            logger.info(
                f"Frame extraction complete: {total_frames_extracted} frames from {len(scene_result.scenes)} scenes "
                f"(avg quality: {average_quality_score:.3f}, processing time: {processing_time:.1f}s)"
            )

            return result

        except Exception as e:
            if cap is not None:
                cap.release()

            error_msg = f"Frame extraction failed: {str(e)}"
            logger.error(error_msg)

            raise VideoProcessingError(
                message=error_msg,
                error_code=ErrorCode.FRAME_EXTRACTION_FAILED,
                file_path=video_path,
                cause=e,
            ) from e

    def _extract_frames_from_scene(
        self,
        cap: cv2.VideoCapture,
        scene: Any,
        fps: float,
        output_dir: Path | None,
        video_name: str,
    ) -> SceneFrameAnalysis:
        """Extract frames from a single scene with quality assessment."""
        frames_per_scene = self.config.visual_analysis.frames_per_scene
        min_quality_score = self.config.visual_analysis.min_quality_score
        enable_quality_filtering = self.config.visual_analysis.enable_quality_filtering

        # Calculate frame positions within the scene
        scene_duration = scene.end_time - scene.start_time
        frame_positions = []

        if frames_per_scene == 1:
            # Extract middle frame
            frame_positions = [scene.start_time + scene_duration / 2]
        else:
            # Extract evenly spaced frames
            for i in range(frames_per_scene):
                position = scene.start_time + (
                    scene_duration * (i + 1) / (frames_per_scene + 1)
                )
                frame_positions.append(position)

        extracted_frames: list[ExtractedFrame] = []
        total_frames_processed = 0
        frames_filtered_by_quality = 0

        for i, timestamp in enumerate(frame_positions):
            total_frames_processed += 1

            # Seek to frame position
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # Read frame
            ret, frame = cap.read()
            if not ret:
                logger.warning(
                    f"Failed to read frame at {timestamp:.1f}s in scene {scene.scene_number}"
                )
                continue

            # Validate and handle corrupt frames
            try:
                validated_frame = validate_image(frame, f"frame at {timestamp:.1f}s")
                processed_frame = handle_corrupt_frame(
                    validated_frame,
                    {
                        "timestamp": timestamp,
                        "scene_number": scene.scene_number,
                        "frame_number": frame_number,
                    },
                )
                if processed_frame is None:
                    logger.warning(
                        f"Corrupt frame detected at {timestamp:.1f}s in scene {scene.scene_number}, skipping"
                    )
                    continue
                frame = processed_frame
            except Exception as e:
                logger.error(f"Frame validation failed at {timestamp:.1f}s: {e}")
                continue

            # Assess frame quality with error recovery
            quality_metrics: FrameQualityMetrics | None = None
            with ErrorRecoveryContext(
                f"quality assessment for frame at {timestamp:.1f}s",
                suppress_errors=True,
            ) as ctx:
                quality_metrics = self._assess_frame_quality(frame)
                if ctx.error:
                    logger.warning(f"Quality assessment failed: {ctx.error}")
                    # Create default quality metrics
                    quality_metrics = self._create_default_quality_metrics()

            # Ensure quality_metrics is assigned
            if quality_metrics is None:
                quality_metrics = self._create_default_quality_metrics()

            # Apply quality filtering if enabled
            if (
                enable_quality_filtering
                and quality_metrics.overall_quality_score < min_quality_score
            ):
                frames_filtered_by_quality += 1
                logger.debug(
                    f"Frame at {timestamp:.1f}s filtered due to low quality: {quality_metrics.overall_quality_score:.3f}"
                )
                continue

            # Generate caption for frame with error recovery
            caption_result: Caption | None = None
            with ErrorRecoveryContext(
                f"caption generation for frame at {timestamp:.1f}s",
                suppress_errors=True,
            ) as ctx:
                caption_result = self._caption_frame(frame)
                if ctx.error:
                    logger.warning(f"Caption generation failed: {ctx.error}")
                    caption_result = None

            # Perform OCR on frame with error recovery
            ocr_result: Ocr | None = None
            with ErrorRecoveryContext(
                f"OCR for frame at {timestamp:.1f}s", suppress_errors=True
            ) as ctx:
                ocr_result = self._extract_text_from_frame(frame)
                if ctx.error:
                    logger.warning(f"OCR failed: {ctx.error}")
                    ocr_result = None

            # Perform object detection on frame with error recovery
            object_detection_result: PresentationLayout | None = None
            with ErrorRecoveryContext(
                f"object detection for frame at {timestamp:.1f}s", suppress_errors=True
            ) as ctx:
                object_detection_result = self._detect_objects_in_frame(frame)
                if ctx.error:
                    logger.warning(f"Object detection failed: {ctx.error}")
                    object_detection_result = None

            # Save frame if output directory specified
            file_path = None
            if output_dir and self.config.visual_analysis.save_extracted_frames:
                output_dir.mkdir(parents=True, exist_ok=True)
                file_path = (
                    output_dir
                    / f"{video_name}_scene{scene.scene_number:03d}_frame{i + 1:02d}.jpg"
                )

                # Resize frame if needed
                height: int
                width: int
                height, width = frame.shape[:2]
                max_width = self.config.visual_analysis.max_frame_width
                max_height = self.config.visual_analysis.max_frame_height

                if width > max_width or height > max_height:
                    scale: float = min(max_width / width, max_height / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(
                        frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
                    )

                # Save frame
                quality = self.config.visual_analysis.frame_quality
                cv2.imwrite(str(file_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

            # Create ExtractedFrame object
            final_height: int
            final_width: int
            final_height, final_width = frame.shape[:2]
            extracted_frame = ExtractedFrame(
                frame_number=frame_number,
                timestamp=timestamp,
                scene_number=scene.scene_number,
                file_path=file_path,
                width=final_width,
                height=final_height,
                quality_metrics=quality_metrics,
                caption_result=caption_result,
                ocr_result=ocr_result,
                object_detection_result=object_detection_result,
            )

            extracted_frames.append(extracted_frame)

        # Find best frame (highest quality)
        best_frame: ExtractedFrame | None = None
        if extracted_frames:
            best_frame = max(
                extracted_frames, key=lambda f: f.quality_metrics.overall_quality_score
            )

        # Calculate quality distribution
        quality_distribution: dict[str, int] = {
            "excellent": 0,
            "good": 0,
            "fair": 0,
            "poor": 0,
        }
        total_quality_score = 0.0

        for frame in extracted_frames:
            category = frame.quality_metrics.overall_quality_category
            quality_distribution[category] = quality_distribution.get(category, 0) + 1
            total_quality_score += frame.quality_metrics.overall_quality_score

        average_quality_score: float = (
            total_quality_score / len(extracted_frames) if extracted_frames else 0.0
        )

        extraction_success_rate = (
            len(extracted_frames) / total_frames_processed
            if total_frames_processed > 0
            else 0.0
        )

        return SceneFrameAnalysis(
            scene_number=scene.scene_number,
            start_time=scene.start_time,
            end_time=scene.end_time,
            duration=scene_duration,
            frames=extracted_frames,
            total_frames_extracted=len(extracted_frames),
            best_frame=best_frame,
            average_quality_score=average_quality_score,
            quality_distribution=quality_distribution,
            total_frames_processed=total_frames_processed,
            frames_filtered_by_quality=frames_filtered_by_quality,
            extraction_success_rate=extraction_success_rate,
        )

    def _assess_frame_quality(self, frame: NDArray[np.uint8]) -> FrameQualityMetrics:
        """Assess the quality of a single frame."""
        # Convert to grayscale for analysis
        gray = cast("NDArray[np.uint8]", cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # 1. Blur Assessment (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_category = self._categorize_blur(blur_score)

        # 2. Contrast Assessment (RMS contrast)
        contrast_score = gray.std()
        contrast_category = self._categorize_contrast(contrast_score)

        # 3. Brightness Assessment
        brightness_score = gray.mean()
        brightness_category = self._categorize_brightness(brightness_score)

        # 4. Additional histogram analysis
        histogram_metrics = self._analyze_histogram(gray)

        # 5. Sharpness details using edge detection
        sharpness_details = self._analyze_sharpness(gray)

        # 6. Color metrics analysis
        color_metrics = self._analyze_color_metrics(frame)

        # 7. Noise metrics estimation
        noise_metrics = self._analyze_noise_metrics(gray)

        # 8. Composition metrics
        composition_metrics = self._analyze_composition_metrics(frame)

        # 9. Calculate overall quality score
        overall_quality_score = self._calculate_overall_quality(
            blur_score, contrast_score, brightness_score
        )
        overall_quality_category = self._categorize_overall_quality(
            overall_quality_score
        )

        # 10. Generate quality report
        quality_report = self._generate_frame_quality_report(
            blur_score,
            blur_category,
            contrast_score,
            contrast_category,
            brightness_score,
            brightness_category,
            overall_quality_score,
            overall_quality_category,
            histogram_metrics,
            sharpness_details,
            color_metrics,
            noise_metrics,
            composition_metrics,
        )

        return FrameQualityMetrics(
            blur_score=blur_score,
            blur_category=blur_category,
            contrast_score=contrast_score,
            contrast_category=contrast_category,
            brightness_score=brightness_score,
            brightness_category=brightness_category,
            overall_quality_score=overall_quality_score,
            overall_quality_category=overall_quality_category,
            histogram_metrics=histogram_metrics,
            sharpness_details=sharpness_details,
            color_metrics=color_metrics,
            noise_metrics=noise_metrics,
            composition_metrics=composition_metrics,
            quality_report=quality_report,
        )

    def _categorize_blur(self, blur_score: float) -> str:
        """Categorize blur score into quality levels."""
        threshold = self.config.visual_analysis.blur_threshold

        if blur_score >= threshold * 2:
            return "excellent"
        elif blur_score >= threshold:
            return "good"
        elif blur_score >= threshold * 0.5:
            return "fair"
        else:
            return "poor"

    def _categorize_contrast(self, contrast_score: float) -> str:
        """Categorize contrast score into quality levels."""
        threshold = self.config.visual_analysis.contrast_threshold

        if contrast_score >= threshold * 2:
            return "excellent"
        elif contrast_score >= threshold:
            return "good"
        elif contrast_score >= threshold * 0.5:
            return "fair"
        else:
            return "poor"

    def _categorize_brightness(self, brightness_score: float) -> str:
        """Categorize brightness score into quality levels."""
        min_brightness = self.config.visual_analysis.brightness_min
        max_brightness = self.config.visual_analysis.brightness_max

        optimal_range = (max_brightness - min_brightness) * 0.3
        center = (min_brightness + max_brightness) / 2

        distance_from_optimal = abs(brightness_score - center)

        if distance_from_optimal <= optimal_range * 0.5:
            return "excellent"
        elif distance_from_optimal <= optimal_range:
            return "good"
        elif min_brightness <= brightness_score <= max_brightness:
            return "fair"
        else:
            return "poor"

    def _analyze_histogram(self, gray: NDArray[np.uint8]) -> dict[str, Any]:
        """Analyze histogram properties for additional quality metrics."""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()

        # Calculate histogram statistics
        total_pixels = gray.size
        hist_normalized = hist / total_pixels

        # Dynamic range (difference between max and min intensity)
        non_zero_indices = np.nonzero(hist)[0]
        if len(non_zero_indices) > 0:
            dynamic_range = non_zero_indices[-1] - non_zero_indices[0]
        else:
            dynamic_range = 0

        # Histogram entropy (measure of information content)
        hist_normalized = hist_normalized[hist_normalized > 0]  # Remove zeros for log
        entropy = (
            -np.sum(hist_normalized * np.log2(hist_normalized))
            if len(hist_normalized) > 0
            else 0
        )

        # Peak analysis
        peaks: list[int] = []
        for i in range(1, len(hist) - 1):
            if (
                hist[i] > hist[i - 1]
                and hist[i] > hist[i + 1]
                and hist[i] > total_pixels * 0.01
            ):
                peaks.append(i)

        return {
            "dynamic_range": float(dynamic_range),
            "entropy": float(entropy),
            "num_peaks": len(peaks),
            "peaks": peaks,
            "mean_intensity": float(gray.mean()),
            "std_intensity": float(gray.std()),
        }

    def _analyze_sharpness(self, gray: NDArray[np.uint8]) -> dict[str, Any]:
        """Analyze sharpness using multiple edge detection methods."""
        # Sobel edge detection
        sobel_x = cast(
            "NDArray[np.float64]", cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        )
        sobel_y = cast(
            "NDArray[np.float64]", cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        )
        sobel_magnitude: NDArray[np.float64] = np.sqrt(
            sobel_x**2 + sobel_y**2  # type: ignore[arg-type]
        )
        sobel_mean = sobel_magnitude.mean()

        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Gradient variance
        gradient_variance = sobel_magnitude.var()

        return {
            "sobel_mean": float(sobel_mean),
            "edge_density": float(edge_density),
            "gradient_variance": float(gradient_variance),
        }

    def _calculate_overall_quality(
        self, blur_score: float, contrast_score: float, brightness_score: float
    ) -> float:
        """Calculate weighted overall quality score (0-1)."""
        # Normalize individual scores to 0-1 range
        blur_normalized = min(
            blur_score / (self.config.visual_analysis.blur_threshold * 2), 1.0
        )
        contrast_normalized = min(
            contrast_score / (self.config.visual_analysis.contrast_threshold * 2), 1.0
        )

        # Brightness normalization (distance from optimal range)
        min_brightness = self.config.visual_analysis.brightness_min
        max_brightness = self.config.visual_analysis.brightness_max
        center = (min_brightness + max_brightness) / 2
        optimal_range = (max_brightness - min_brightness) * 0.3

        distance_from_optimal = abs(brightness_score - center)
        brightness_normalized = max(
            0, 1 - (distance_from_optimal / (optimal_range * 2))
        )

        # Apply weights
        weights = self.config.visual_analysis
        overall_score = (
            blur_normalized * weights.blur_weight
            + contrast_normalized * weights.contrast_weight
            + brightness_normalized * weights.brightness_weight
        )

        return min(overall_score, 1.0)

    def _categorize_overall_quality(self, quality_score: float) -> str:
        """Categorize overall quality score into quality levels."""
        if quality_score >= 0.8:
            return "excellent"
        elif quality_score >= 0.6:
            return "good"
        elif quality_score >= 0.4:
            return "fair"
        else:
            return "poor"

    def _analyze_color_metrics(self, frame: NDArray[np.uint8]) -> dict[str, Any]:
        """Analyze color distribution and saturation metrics."""
        # Convert to HSV for saturation analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, _v = cv2.split(hsv)

        # Color distribution in RGB
        b, g, r = cv2.split(frame)

        # Calculate color metrics
        color_metrics = {
            "saturation": {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max()),
            },
            "hue_distribution": {
                "mean": float(h.mean()),
                "std": float(h.std()),
                "dominant_hue": int(
                    np.argmax(cv2.calcHist([h], [0], None, [180], [0, 180]))
                ),
            },
            "color_balance": {
                "red_mean": float(r.mean()),
                "green_mean": float(g.mean()),
                "blue_mean": float(b.mean()),
                "color_cast": self._detect_color_cast(r.mean(), g.mean(), b.mean()),
            },
            "color_diversity": float(np.std([r.std(), g.std(), b.std()])),
        }

        return color_metrics

    def _detect_color_cast(self, r_mean: float, g_mean: float, b_mean: float) -> str:
        """Detect if image has a color cast."""
        # Calculate ratios
        total = r_mean + g_mean + b_mean
        if total == 0:
            return "neutral"

        r_ratio = r_mean / total
        g_ratio = g_mean / total
        b_ratio = b_mean / total

        # Check for color cast
        if r_ratio > 0.4:
            return "red_cast"
        elif g_ratio > 0.4:
            return "green_cast"
        elif b_ratio > 0.4:
            return "blue_cast"
        else:
            return "neutral"

    def _analyze_noise_metrics(self, gray: NDArray[np.uint8]) -> dict[str, Any]:
        """Estimate noise levels in the image."""
        # Method 1: High-pass filter approach
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        laplacian = cast("NDArray[np.float32]", cv2.filter2D(gray, cv2.CV_32F, kernel))
        noise_std: float = float(np.std(laplacian))

        # Method 2: Difference from Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = gray.astype(np.float32) - blurred.astype(np.float32)
        noise_estimate = np.std(diff)

        # Signal-to-noise ratio estimation
        signal_mean = float(np.mean(gray))
        snr: float = signal_mean / noise_std if noise_std > 0 else float("inf")

        noise_metrics = {
            "noise_std_laplacian": noise_std,
            "noise_estimate_gaussian": float(noise_estimate),
            "signal_to_noise_ratio": snr if snr != float("inf") else 1000.0,
            "noise_level": self._categorize_noise_level(noise_std),
        }

        return noise_metrics

    def _categorize_noise_level(self, noise_std: float) -> str:
        """Categorize noise level."""
        if noise_std < 5:
            return "very_low"
        elif noise_std < 10:
            return "low"
        elif noise_std < 20:
            return "moderate"
        elif noise_std < 40:
            return "high"
        else:
            return "very_high"

    def _analyze_composition_metrics(self, frame: NDArray[np.uint8]) -> dict[str, Any]:
        """Analyze frame composition including rule of thirds and symmetry."""
        height: int
        width: int
        height, width = frame.shape[:2]
        gray = cast("NDArray[np.uint8]", cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # Rule of thirds analysis
        thirds_x = [width // 3, 2 * width // 3]
        thirds_y = [height // 3, 2 * height // 3]

        # Extract regions around rule of thirds lines
        line_width = max(3, int(min(width, height) * 0.01))
        thirds_regions: list[float] = []

        # Vertical lines
        for x in thirds_x:
            region = gray[:, max(0, x - line_width) : min(width, x + line_width)]
            sobel_result = cast(
                "NDArray[np.float64]", cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
            )
            thirds_regions.append(float(np.mean(sobel_result)))

        # Horizontal lines
        for y in thirds_y:
            region = gray[max(0, y - line_width) : min(height, y + line_width), :]
            sobel_result = cast(
                "NDArray[np.float64]", cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
            )
            thirds_regions.append(float(np.mean(sobel_result)))

        # Symmetry analysis
        # Vertical symmetry
        left_half = gray[:, : width // 2]
        right_half = cv2.flip(gray[:, width // 2 :], 1)
        min_width = min(left_half.shape[1], right_half.shape[1])
        vertical_symmetry = (
            1.0
            - np.mean(np.abs(left_half[:, :min_width] - right_half[:, :min_width]))
            / 255.0
        )

        # Horizontal symmetry
        top_half = gray[: height // 2, :]
        bottom_half = cv2.flip(gray[height // 2 :, :], 0)
        min_height = min(top_half.shape[0], bottom_half.shape[0])
        horizontal_symmetry = (
            1.0
            - np.mean(np.abs(top_half[:min_height, :] - bottom_half[:min_height, :]))
            / 255.0
        )

        # Focus point detection (simplified)
        # Find the region with highest gradient magnitude
        grad_x = cast("NDArray[np.float64]", cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
        grad_y = cast("NDArray[np.float64]", cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
        grad_mag: NDArray[np.float64] = np.sqrt(grad_x**2 + grad_y**2)  # type: ignore[arg-type]

        # Find center of mass of high gradient regions
        threshold = np.percentile(grad_mag, 90)
        y_coords, x_coords = np.where(grad_mag > threshold)
        focus_x: float
        focus_y: float
        if len(x_coords) > 0:
            focus_x = float(np.mean(x_coords)) / width
            focus_y = float(np.mean(y_coords)) / height
        else:
            focus_x, focus_y = 0.5, 0.5

        composition_metrics = {
            "rule_of_thirds": {
                "edge_strength_mean": float(np.mean(thirds_regions)),
                "edge_strength_std": float(np.std(thirds_regions)),
                "alignment_score": float(
                    np.mean(thirds_regions) / (np.mean(grad_mag) + 1e-6)
                ),
            },
            "symmetry": {
                "vertical": float(vertical_symmetry),
                "horizontal": float(horizontal_symmetry),
                "overall": float((vertical_symmetry + horizontal_symmetry) / 2),
            },
            "focus_point": {
                "x": float(focus_x),
                "y": float(focus_y),
                "distance_from_center": float(
                    np.sqrt((focus_x - 0.5) ** 2 + (focus_y - 0.5) ** 2)
                ),
            },
            "balance": self._assess_visual_balance(gray),
        }

        return composition_metrics

    def _assess_visual_balance(self, gray: NDArray[np.uint8]) -> dict[str, float]:
        """Assess visual balance of the image."""
        height: int
        width: int
        height, width = gray.shape

        # Divide into quadrants
        mid_x: int
        mid_y: int
        mid_x, mid_y = width // 2, height // 2
        quadrants: list[NDArray[np.uint8]] = [
            gray[:mid_y, :mid_x],  # Top-left
            gray[:mid_y, mid_x:],  # Top-right
            gray[mid_y:, :mid_x],  # Bottom-left
            gray[mid_y:, mid_x:],  # Bottom-right
        ]

        # Calculate weight (brightness) of each quadrant
        weights = [np.mean(q) for q in quadrants]

        # Calculate balance metrics
        horizontal_balance = 1.0 - abs(
            weights[0] + weights[2] - weights[1] - weights[3]
        ) / (sum(weights) + 1e-6)
        vertical_balance = 1.0 - abs(
            weights[0] + weights[1] - weights[2] - weights[3]
        ) / (sum(weights) + 1e-6)
        diagonal_balance = 1.0 - abs(
            weights[0] + weights[3] - weights[1] - weights[2]
        ) / (sum(weights) + 1e-6)

        return {
            "horizontal": float(horizontal_balance),
            "vertical": float(vertical_balance),
            "diagonal": float(diagonal_balance),
            "overall": float(
                (horizontal_balance + vertical_balance + diagonal_balance) / 3
            ),
        }

    def _generate_frame_quality_report(
        self,
        blur_score: float,
        blur_category: str,
        contrast_score: float,
        contrast_category: str,
        brightness_score: float,
        brightness_category: str,
        overall_quality_score: float,
        overall_quality_category: str,
        histogram_metrics: dict[str, Any],
        sharpness_details: dict[str, Any],
        color_metrics: dict[str, Any],
        noise_metrics: dict[str, Any],
        composition_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate comprehensive quality report for a frame."""
        # Identify issues
        issues = []
        recommendations = []

        # Blur issues
        if blur_category in ["poor", "fair"]:
            issues.append(
                f"Image sharpness is {blur_category} (score: {blur_score:.1f})"
            )
            recommendations.append("Ensure camera is stable and properly focused")

        # Contrast issues
        if contrast_category in ["poor", "fair"]:
            issues.append(
                f"Image contrast is {contrast_category} (score: {contrast_score:.1f})"
            )
            recommendations.append(
                "Improve lighting contrast or adjust camera settings"
            )

        # Brightness issues
        if brightness_category == "poor":
            if brightness_score < 50:
                issues.append(f"Image is too dark (brightness: {brightness_score:.1f})")
                recommendations.append("Increase lighting or adjust exposure settings")
            else:
                issues.append(
                    f"Image is overexposed (brightness: {brightness_score:.1f})"
                )
                recommendations.append("Reduce lighting or adjust exposure settings")

        # Color cast issues
        color_cast = color_metrics["color_balance"]["color_cast"]
        if color_cast != "neutral":
            issues.append(f"Image has a {color_cast.replace('_', ' ')}")
            recommendations.append("Adjust white balance settings")

        # Noise issues
        noise_level = noise_metrics["noise_level"]
        if noise_level in ["high", "very_high"]:
            issues.append(f"Image has {noise_level.replace('_', ' ')} noise levels")
            recommendations.append("Use better lighting or reduce ISO settings")

        # Composition issues
        if composition_metrics["focus_point"]["distance_from_center"] > 0.4:
            issues.append("Main subject appears off-center")
            recommendations.append(
                "Consider repositioning subject using rule of thirds"
            )

        # Generate strengths
        strengths = []
        if blur_category == "excellent":
            strengths.append("Excellent image sharpness")
        if contrast_category == "excellent":
            strengths.append("Excellent contrast")
        if brightness_category == "excellent":
            strengths.append("Optimal brightness levels")
        if noise_level == "very_low":
            strengths.append("Very low noise levels")
        if composition_metrics["symmetry"]["overall"] > 0.8:
            strengths.append("Good visual symmetry")

        return {
            "overall_score": overall_quality_score,
            "overall_category": overall_quality_category,
            "strengths": strengths,
            "issues": issues,
            "recommendations": recommendations,
            "technical_details": {
                "blur": {"score": blur_score, "category": blur_category},
                "contrast": {"score": contrast_score, "category": contrast_category},
                "brightness": {
                    "score": brightness_score,
                    "category": brightness_category,
                },
                "dynamic_range": histogram_metrics["dynamic_range"],
                "entropy": histogram_metrics["entropy"],
                "edge_density": sharpness_details["edge_density"],
                "saturation_mean": color_metrics["saturation"]["mean"],
                "noise_snr": noise_metrics["signal_to_noise_ratio"],
                "symmetry_score": composition_metrics["symmetry"]["overall"],
                "balance_score": composition_metrics["balance"]["overall"],
            },
        }

    def _caption_frame(self, frame: NDArray[np.uint8]) -> Caption | None:
        """Generate caption for a frame via the delegated image-analyser."""
        if not self.config.visual_analysis.enable_captioning:
            return None

        try:
            result = self._analyse_frame(frame)
            return result.caption

        except Exception as e:
            logger.warning(f"Failed to generate caption for frame: {e}")
            # Preserve the historical "return a placeholder, not None" contract.
            return Caption(
                text="Caption generation failed",
                backend="local",
                model=self.config.visual_analysis.captioning_model,
                tokens_generated=None,
                cost_estimate_usd=None,
            )

    def _extract_text_from_frame(self, frame: NDArray[np.uint8]) -> Ocr | None:
        """Extract text from a frame via the delegated image-analyser."""
        if not self.config.visual_analysis.enable_ocr:
            return None

        try:
            result = self._analyse_frame(frame)
            return result.ocr

        except Exception as e:
            logger.warning(f"Failed to extract text from frame: {e}")
            engine = self.config.visual_analysis.ocr_engine
            # Caller config may use values outside image-analyser's strict
            # ("tesseract" | "easyocr") set; default to tesseract on mismatch
            # so the placeholder still validates.
            if engine not in ("tesseract", "easyocr"):
                engine = "tesseract"
            return Ocr(
                text="OCR extraction failed",
                blocks=[],
                engine=engine,
                average_confidence=0.0,
            )

    def _detect_objects_in_frame(
        self, frame: NDArray[np.uint8]
    ) -> PresentationLayout | None:
        """Detect objects in a frame via the delegated image-analyser.

        The returned ``PresentationLayout`` post-processes image-analyser's
        raw COCO labels into the presentation-domain taxonomy (presenter,
        screen, slide, audience, etc.) — see ``presentation_classifier``.
        """
        if not self.config.visual_analysis.enable_object_detection:
            return None

        try:
            result = self._analyse_frame(frame)
            return classify_objects(result.objects or [])

        except Exception as e:
            logger.warning(f"Failed to detect objects in frame: {e}")
            # Empty layout sentinel — preserves the historical "return a
            # placeholder, not None" contract on backend failure.
            return classify_objects([])

    def _create_default_quality_metrics(self) -> FrameQualityMetrics:
        """Create default quality metrics when assessment fails."""
        return FrameQualityMetrics(
            blur_score=0.0,
            blur_category="unknown",
            contrast_score=0.0,
            contrast_category="unknown",
            brightness_score=0.0,
            brightness_category="unknown",
            overall_quality_score=0.0,
            overall_quality_category="unknown",
            histogram_metrics={},
            sharpness_details={},
            color_metrics={},
            noise_metrics={},
            composition_metrics={},
            quality_report={"status": "error", "message": "Quality assessment failed"},
        )

    def cleanup(self):
        """Clean up model resources.

        ImageAnalyser does not expose a cleanup API; we simply drop the
        reference and let GC reclaim any cached models. Legacy attributes
        (``captioner``, ``ocr_detector``, ``object_detector``) are also reset
        so tests that injected mocks see them cleared.
        """
        if self.image_analyser is not None:
            self.image_analyser = None

        for attr in ("captioner", "ocr_detector", "object_detector"):
            obj = getattr(self, attr, None)
            if obj is not None and hasattr(obj, "cleanup"):
                # Best-effort during teardown — cleanup failures shouldn't propagate.
                with contextlib.suppress(Exception):
                    obj.cleanup()
            setattr(self, attr, None)

        logger.info("FrameExtractor model resources cleaned up")


def create_frame_extractor(config: Any = None) -> FrameExtractor:
    """Create a new FrameExtractor instance.

    Args:
        config: Optional configuration object

    Returns:
        Configured FrameExtractor instance
    """
    return FrameExtractor(config=config)
