"""Report generator for creating structured analysis reports."""

import csv
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ReportFormat(str, Enum):
    """Supported report export formats."""

    JSON = "json"
    CSV = "csv"
    HTML = "html"
    PLAIN_TEXT = "txt"


class ReportCustomization(BaseModel):
    """Configuration for report customization."""

    include_video_metadata: bool = True
    include_audio_metadata: bool = True
    include_scenes: bool = True
    include_frames: bool = True
    include_transcription: bool = True
    include_speech_metrics: bool = True
    include_api_costs: bool = True
    include_visual_analysis: bool = True
    include_ocr: bool = True
    include_object_detection: bool = True

    # Frame detail level
    include_frame_files: bool = False  # Don't include full file paths by default
    include_detected_objects: bool = True

    # Verbosity
    max_frames_in_summary: int = 50  # Limit frames in summary reports


class VideoMetadata(BaseModel):
    """Video file metadata."""

    file_path: str
    duration: float
    width: int
    height: int
    fps: float
    format: str | None = None


class AudioMetadata(BaseModel):
    """Audio metadata."""

    duration: float
    sample_rate: int
    channels: int


class SceneReport(BaseModel):
    """Report for a single scene."""

    scene_number: int
    start_time: float
    end_time: float
    duration: float
    num_frames: int


class FrameReport(BaseModel):
    """Report for a single analyzed frame."""

    frame_number: int
    timestamp: float
    scene_number: int
    file_path: str | None
    width: int
    height: int

    # Caption
    caption: str | None = None
    caption_confidence: float | None = None
    caption_model: str | None = None
    caption_tokens: int | None = None  # Total tokens used for caption
    caption_cost: float | None = None  # Cost in USD for caption

    # OCR
    ocr_text: str | None = None
    ocr_confidence: float | None = None
    num_text_regions: int = 0

    # Object detection
    num_objects_detected: int = 0
    detected_objects: list[str] = []

    # Quality
    quality_score: float | None = None
    sharpness: float | None = None
    brightness: float | None = None


class TranscriptionSegment(BaseModel):
    """Transcription segment."""

    id: int
    text: str
    start_time: float
    end_time: float
    confidence: float | None = None
    num_words: int = 0


class SpeechMetrics(BaseModel):
    """Speech analysis metrics."""

    total_words: int
    total_speech_duration: float
    speaking_rate_wpm: float
    average_confidence: float


class APICostSummary(BaseModel):
    """Summary of API usage costs."""

    total_frames_processed: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    provider: str | None = None
    model: str | None = None


class AnalysisReport(BaseModel):
    """Complete video analysis report."""

    # Video metadata
    video: VideoMetadata

    # Audio metadata
    audio: AudioMetadata | None = None

    # Scenes
    scenes: list[SceneReport] = []
    total_scenes: int = 0

    # Frames
    frames: list[FrameReport] = []
    total_frames: int = 0

    # Transcription
    transcription_segments: list[TranscriptionSegment] = []
    full_transcription_text: str = ""
    language: str | None = None

    # Speech metrics
    speech_metrics: SpeechMetrics | None = None

    # API cost summary
    api_cost_summary: APICostSummary | None = None

    # Analysis metadata
    has_transcription: bool = False
    has_visual_analysis: bool = False
    has_captions: bool = False
    has_ocr: bool = False
    has_object_detection: bool = False


class ReportGenerator:
    """Generate analysis reports from video analysis data."""

    def __init__(self, config: Any = None):
        """Initialize report generator."""
        self.config = config
        logger.info("ReportGenerator initialized")

    def generate_report(self, analysis_data: dict[str, Any]) -> dict[str, Any]:
        """Generate analysis report from data."""
        logger.info("Generating analysis report")

        # Extract video info
        video_info = analysis_data.get("video_info")
        if not video_info:
            raise ValueError("Missing video_info in analysis data")

        video_metadata = VideoMetadata(
            file_path=str(video_info.file_path),
            duration=video_info.duration,
            width=video_info.width,
            height=video_info.height,
            fps=video_info.fps,
            format=getattr(video_info, "format", None),
        )

        # Extract audio info
        audio_info = analysis_data.get("audio_info")
        audio_metadata = None
        if audio_info:
            audio_metadata = AudioMetadata(
                duration=audio_info.duration,
                sample_rate=audio_info.sample_rate,
                channels=audio_info.channels,
            )

        # Extract scenes
        scenes = analysis_data.get("scenes", [])
        scene_reports: list[SceneReport] = []
        for scene in scenes:
            scene_reports.append(
                SceneReport(
                    scene_number=scene.scene_number,
                    start_time=scene.start_time,
                    end_time=scene.end_time,
                    duration=scene.duration,
                    num_frames=0,  # Will be updated from frame analysis
                )
            )

        # Extract frame analyses
        frame_analyses = analysis_data.get("frame_analyses", [])
        frame_reports: list[FrameReport] = []
        has_captions = False
        has_ocr = False
        has_object_detection = False

        # frame_analyses is a flat list of frame results
        for frame in frame_analyses:
            # Build frame report
            frame_report = FrameReport(
                frame_number=frame.frame_number,
                timestamp=frame.timestamp,
                scene_number=frame.scene_number,
                file_path=str(frame.file_path) if frame.file_path else None,
                width=frame.width,
                height=frame.height,
            )

            # Add caption if available
            if frame.caption_result:
                frame_report.caption = frame.caption_result.caption
                frame_report.caption_confidence = frame.caption_result.confidence
                frame_report.caption_model = frame.caption_result.model_used
                frame_report.caption_tokens = frame.caption_result.tokens_generated
                frame_report.caption_cost = frame.caption_result.cost_estimate
                has_captions = True

            # Add OCR if available
            if frame.ocr_result:
                # Use full_text if available, otherwise combine text regions
                frame_report.ocr_text = frame.ocr_result.full_text
                frame_report.ocr_confidence = frame.ocr_result.average_confidence
                frame_report.num_text_regions = len(frame.ocr_result.text_regions)
                has_ocr = True

            # Add object detection if available
            if frame.object_detection_result:
                frame_report.num_objects_detected = len(
                    frame.object_detection_result.detected_objects
                )
                frame_report.detected_objects = [
                    str(
                        obj.element_type.value
                        if hasattr(obj.element_type, "value")
                        else obj.element_type
                    )
                    for obj in frame.object_detection_result.detected_objects
                ]
                has_object_detection = True

            # Add quality metrics if available
            if frame.quality_metrics:
                frame_report.quality_score = frame.quality_metrics.overall_quality_score
                frame_report.sharpness = frame.quality_metrics.blur_score
                frame_report.brightness = frame.quality_metrics.brightness_score

            frame_reports.append(frame_report)

            # Update scene frame count
            scene_report: SceneReport
            for scene_report in scene_reports:
                if scene_report.scene_number == frame.scene_number:
                    scene_report.num_frames += 1
                    break

        # Extract transcription
        transcription = analysis_data.get("transcription")
        transcription_segments: list[TranscriptionSegment] = []
        full_text = ""
        language = None

        if transcription:
            language = transcription.language
            if transcription.segments:
                for seg in transcription.segments:
                    transcription_segments.append(
                        TranscriptionSegment(
                            id=seg.id,
                            text=seg.text,
                            start_time=seg.start,
                            end_time=seg.end,
                            confidence=getattr(seg, "confidence", None),
                            num_words=len(seg.words) if hasattr(seg, "words") else 0,
                        )
                    )
                full_text = " ".join(seg.text for seg in transcription.segments)

        # Extract speech metrics
        speech_analysis = analysis_data.get("speech_metrics")
        speech_metrics = None

        if speech_analysis and hasattr(speech_analysis, "overall_metrics"):
            overall = speech_analysis.overall_metrics
            speech_metrics = SpeechMetrics(
                total_words=overall.total_word_count,
                total_speech_duration=overall.total_speaking_time,
                speaking_rate_wpm=overall.average_wpm,
                average_confidence=getattr(overall, "average_confidence", 0.0),
            )

        # Calculate API cost summary
        api_cost_summary: APICostSummary | None = None
        total_tokens: int = 0
        total_cost: float = 0.0
        frames_with_cost = 0
        provider: str | None = None
        model: str | None = None

        frame_report: FrameReport
        for frame_report in frame_reports:
            if frame_report.caption_cost is not None and frame_report.caption_cost > 0:
                total_cost += frame_report.caption_cost
                frames_with_cost += 1
            if frame_report.caption_tokens is not None:
                total_tokens += frame_report.caption_tokens
            # Extract provider and model from first frame with caption
            if provider is None and frame_report.caption_model:
                provider = (
                    frame_report.caption_model.split(":")[0]
                    if ":" in frame_report.caption_model
                    else None
                )
                model = (
                    frame_report.caption_model.split(":", 1)[1]
                    if ":" in frame_report.caption_model
                    else frame_report.caption_model
                )

        if frames_with_cost > 0:
            api_cost_summary = APICostSummary(
                total_frames_processed=frames_with_cost,
                total_tokens_used=total_tokens,
                total_cost_usd=total_cost,
                provider=provider,
                model=model,
            )

        # Build final report
        report = AnalysisReport(
            video=video_metadata,
            audio=audio_metadata,
            scenes=scene_reports,
            total_scenes=len(scene_reports),
            frames=frame_reports,
            total_frames=len(frame_reports),
            transcription_segments=transcription_segments,
            full_transcription_text=full_text,
            language=language,
            speech_metrics=speech_metrics,
            api_cost_summary=api_cost_summary,
            has_transcription=len(transcription_segments) > 0,
            has_visual_analysis=len(frame_reports) > 0,
            has_captions=has_captions,
            has_ocr=has_ocr,
            has_object_detection=has_object_detection,
        )

        # Convert to dict for JSON serialization
        return report.model_dump()

    def save_json(self, report: dict[str, Any], output_path: Path) -> None:
        """Save report as JSON file."""
        logger.info(f"Saving JSON report to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"JSON report saved: {output_path}")

    def export_report(
        self,
        report: dict[str, Any],
        output_path: Path,
        format: ReportFormat = ReportFormat.JSON,
        customization: ReportCustomization | None = None,
    ) -> None:
        """Export report in specified format with customization options.

        Args:
            report: The analysis report dictionary
            output_path: Path where to save the exported report
            format: Export format (json, csv, html, txt)
            customization: Report customization options
        """
        if customization is None:
            customization = ReportCustomization()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == ReportFormat.JSON:
            self._export_json(report, output_path, customization)
        elif format == ReportFormat.CSV:
            self._export_csv(report, output_path, customization)
        elif format == ReportFormat.PLAIN_TEXT:
            self._export_text(report, output_path, customization)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_json(
        self,
        report: dict[str, Any],
        output_path: Path,
        customization: ReportCustomization,
    ) -> None:
        """Export report as JSON with customization."""
        filtered_report = self._filter_report(report, customization)
        with open(output_path, "w") as f:
            json.dump(filtered_report, f, indent=2)
        logger.info(f"JSON report exported to {output_path}")

    def _export_csv(
        self,
        report: dict[str, Any],
        output_path: Path,
        customization: ReportCustomization,
    ) -> None:
        """Export frame data as CSV for analysis in spreadsheet applications."""
        frames = report.get("frames", [])
        if not frames:
            logger.warning("No frames to export as CSV")
            return

        # Determine CSV columns based on customization
        fieldnames = ["frame_number", "timestamp", "scene_number"]

        if customization.include_visual_analysis:
            fieldnames.extend(["caption", "caption_confidence", "quality_score"])

        if customization.include_ocr:
            fieldnames.extend(["ocr_text", "ocr_confidence", "num_text_regions"])

        if (
            customization.include_object_detection
            and customization.include_detected_objects
        ):
            fieldnames.extend(["num_objects_detected", "detected_objects"])

        # Limit frames if requested
        frames_to_export = frames[: customization.max_frames_in_summary]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for frame in frames_to_export:
                row: dict[str, Any] = {
                    "frame_number": frame.get("frame_number"),
                    "timestamp": frame.get("timestamp"),
                    "scene_number": frame.get("scene_number"),
                }

                if customization.include_visual_analysis:
                    row.update(
                        {
                            "caption": frame.get("caption", ""),
                            "caption_confidence": frame.get("caption_confidence", ""),
                            "quality_score": frame.get("quality_score", ""),
                        }
                    )

                if customization.include_ocr:
                    row.update(
                        {
                            "ocr_text": (frame.get("ocr_text", "") or "")[
                                :100
                            ],  # Limit text
                            "ocr_confidence": frame.get("ocr_confidence", ""),
                            "num_text_regions": frame.get("num_text_regions", ""),
                        }
                    )

                if (
                    customization.include_object_detection
                    and customization.include_detected_objects
                ):
                    row.update(
                        {
                            "num_objects_detected": frame.get(
                                "num_objects_detected", ""
                            ),
                            "detected_objects": ", ".join(
                                frame.get("detected_objects", [])
                            ),
                        }
                    )

                writer.writerow(row)

        logger.info(f"CSV report exported to {output_path}")

    def _export_text(
        self,
        report: dict[str, Any],
        output_path: Path,
        customization: ReportCustomization,
    ) -> None:
        """Export report as plain text for easy reading."""
        lines: list[str] = []

        # Header
        lines.append("=" * 80)
        lines.append("VIDEO ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Video metadata
        if customization.include_video_metadata:
            video = report.get("video", {})
            lines.append("VIDEO INFORMATION")
            lines.append("-" * 40)
            lines.append(f"File: {video.get('file_path', 'Unknown')}")
            lines.append(f"Duration: {video.get('duration', 0):.2f} seconds")
            lines.append(
                f"Resolution: {video.get('width', 0)}x{video.get('height', 0)}"
            )
            lines.append(f"FPS: {video.get('fps', 0):.2f}")
            lines.append("")

        # Speech metrics
        if customization.include_speech_metrics:
            metrics = report.get("speech_metrics")
            if metrics:
                lines.append("SPEECH METRICS")
                lines.append("-" * 40)
                lines.append(f"Total Words: {metrics.get('total_words', 0)}")
                lines.append(
                    f"Speaking Time: {metrics.get('total_speech_duration', 0):.2f}s"
                )
                lines.append(
                    f"Speaking Rate: {metrics.get('speaking_rate_wpm', 0):.1f} WPM"
                )
                lines.append("")

        # API costs
        if customization.include_api_costs:
            api_costs = report.get("api_cost_summary")
            if api_costs:
                lines.append("API USAGE COSTS")
                lines.append("-" * 40)
                lines.append(f"Provider: {api_costs.get('provider', 'Unknown')}")
                lines.append(f"Model: {api_costs.get('model', 'Unknown')}")
                lines.append(f"Total Tokens: {api_costs.get('total_tokens_used', 0)}")
                lines.append(f"Total Cost: ${api_costs.get('total_cost_usd', 0):.4f}")
                lines.append("")

        # Transcription summary
        if customization.include_transcription:
            full_text = report.get("full_transcription_text", "")
            if full_text:
                lines.append("TRANSCRIPTION SUMMARY")
                lines.append("-" * 40)
                lines.append(
                    full_text[: customization.max_frames_in_summary * 10] + "..."
                )
                lines.append("")

        # Frame summary
        if customization.include_frames:
            frames = report.get("frames", [])
            lines.append(f"FRAME ANALYSIS SUMMARY ({len(frames)} frames)")
            lines.append("-" * 40)
            for frame in frames[: customization.max_frames_in_summary]:
                lines.append(
                    f"Frame {frame.get('frame_number')}: {frame.get('timestamp'):.2f}s"
                )
                if customization.include_visual_analysis and frame.get("caption"):
                    lines.append(f"  Caption: {frame.get('caption')}")
                if customization.include_ocr and frame.get("ocr_text"):
                    lines.append(f"  OCR: {frame.get('ocr_text')[:60]}...")
            lines.append("")

        lines.append("=" * 80)

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Text report exported to {output_path}")

    def _filter_report(
        self,
        report: dict[str, Any],
        customization: ReportCustomization,
    ) -> dict[str, Any]:
        """Filter report based on customization options."""
        filtered = dict(report)

        # Remove unwanted sections
        if not customization.include_audio_metadata:
            filtered.pop("audio", None)

        if not customization.include_scenes:
            filtered.pop("scenes", None)
            filtered["total_scenes"] = 0

        if not customization.include_frames:
            filtered.pop("frames", None)
            filtered["total_frames"] = 0
        else:
            # Limit frames if requested
            frames = filtered.get("frames", [])
            if len(frames) > customization.max_frames_in_summary:
                filtered["frames"] = frames[: customization.max_frames_in_summary]
                filtered["total_frames"] = customization.max_frames_in_summary

            # Remove file paths if not requested
            if not customization.include_frame_files:
                for frame in filtered.get("frames", []):
                    frame.pop("file_path", None)

            # Remove detected objects if not requested
            if not customization.include_detected_objects:
                for frame in filtered.get("frames", []):
                    frame.pop("detected_objects", None)

        if not customization.include_transcription:
            filtered.pop("transcription_segments", None)
            filtered["full_transcription_text"] = ""
            filtered["language"] = None

        if not customization.include_speech_metrics:
            filtered.pop("speech_metrics", None)

        if not customization.include_api_costs:
            filtered.pop("api_cost_summary", None)

        # Update analysis metadata flags
        filtered["has_captions"] = (
            customization.include_visual_analysis
            and filtered.get("has_captions", False)
        )
        filtered["has_ocr"] = customization.include_ocr and filtered.get(
            "has_ocr", False
        )
        filtered["has_object_detection"] = (
            customization.include_object_detection
            and filtered.get("has_object_detection", False)
        )

        return filtered

    def export_to_formats(
        self,
        report: dict[str, Any],
        output_dir: Path,
        formats: list[ReportFormat] | None = None,
        customization: ReportCustomization | None = None,
    ) -> dict[str, Path]:
        """Export report to multiple formats at once.

        Args:
            report: The analysis report dictionary
            output_dir: Directory to save exported reports
            formats: List of formats to export (default: JSON, CSV, TXT)
            customization: Report customization options

        Returns:
            Dictionary mapping format names to output file paths
        """
        if formats is None:
            formats = [ReportFormat.JSON, ReportFormat.CSV, ReportFormat.PLAIN_TEXT]

        if customization is None:
            customization = ReportCustomization()

        output_paths: dict[str, Path] = {}

        for fmt in formats:
            ext = fmt.value
            output_path = output_dir / f"report.{ext}"

            self.export_report(report, output_path, fmt, customization)
            output_paths[fmt.value] = output_path

        logger.info(f"Report exported to {len(output_paths)} formats")
        return output_paths
