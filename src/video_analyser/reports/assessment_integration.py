"""Integration between AnalysisReport and AssessmentReport."""

import json
import logging
from pathlib import Path
from typing import Any

from .assessment_report import AssessmentReport
from .assessment_session import AssessmentSession
from .report_generator import AnalysisReport

logger = logging.getLogger(__name__)


class AssessmentIntegration:
    """Handle integration between analysis and assessment reports."""

    @staticmethod
    def load_analysis_report(report_path: str | Path) -> AnalysisReport | None:
        """
        Load an AnalysisReport from JSON file.

        Args:
            report_path: Path to analysis report JSON

        Returns:
            AnalysisReport or None if failed to load
        """
        try:
            with open(report_path) as f:
                data = json.load(f)
            return AnalysisReport(**data)
        except Exception as e:
            logger.error(f"Failed to load analysis report: {e}")
            return None

    @staticmethod
    def get_transcription_for_segment(
        analysis_report: AnalysisReport, start_time: float, end_time: float
    ) -> str:
        """
        Get transcription text for a time segment.

        Args:
            analysis_report: AnalysisReport with transcription
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds

        Returns:
            Transcribed text for the segment
        """
        segment_text: list[str] = []

        for trans_seg in analysis_report.transcription_segments:
            # Check if segment overlaps with time range
            if trans_seg.start_time < end_time and trans_seg.end_time > start_time:
                segment_text.append(trans_seg.text)

        return " ".join(segment_text)

    @staticmethod
    def enrich_assessment_with_analysis(
        assessment: AssessmentReport, analysis_report: AnalysisReport
    ) -> AssessmentReport:
        """
        Enrich assessment with data from analysis report.

        Args:
            assessment: AssessmentReport to enrich
            analysis_report: AnalysisReport with source data

        Returns:
            Updated AssessmentReport with transcriptions added to segments
        """
        # Add transcriptions to diarization segments
        for segment in assessment.diarization_segments:
            if not segment.transcription:
                segment.transcription = (
                    AssessmentIntegration.get_transcription_for_segment(
                        analysis_report, segment.start_time, segment.end_time
                    )
                )

        # Store reference to analysis data for reporting
        assessment.analysis_id = (
            getattr(analysis_report, "id", None) or assessment.analysis_id
        )

        logger.info("Enriched assessment with transcription data from analysis report")
        return assessment

    @staticmethod
    def create_assessment_from_analysis(
        analysis_report: AnalysisReport,
        assessor_name: str,
        storage_dir: str | Path = "assessments",
    ) -> AssessmentSession:
        """
        Create an AssessmentSession pre-populated with analysis data.

        Args:
            analysis_report: AnalysisReport to base assessment on
            assessor_name: Name of assessor
            storage_dir: Directory for storing assessments

        Returns:
            AssessmentSession with analysis data loaded
        """
        # Create session
        analysis_id = getattr(analysis_report, "id", "analysis_unknown")
        video_path = str(analysis_report.video.file_path)

        session = AssessmentSession(
            analysis_id=analysis_id,
            video_file_path=video_path,
            assessor_name=assessor_name,
            storage_dir=storage_dir,
        )

        # Store reference to analysis report
        session.assessment.analysis_id = analysis_id

        logger.info(
            f"Created assessment session from analysis {analysis_id} "
            f"for assessor {assessor_name}"
        )

        return session

    @staticmethod
    def get_speaker_transcription(
        assessment: AssessmentReport, analysis_report: AnalysisReport, speaker_id: str
    ) -> str:
        """
        Get full transcription for a specific speaker.

        Args:
            assessment: AssessmentReport with diarization
            analysis_report: AnalysisReport with transcription
            speaker_id: Speaker ID to get transcription for

        Returns:
            Full transcribed text for speaker
        """
        speaker_text: list[str] = []

        for segment in assessment.diarization_segments:
            if segment.speaker_id == speaker_id:
                # Get transcription for this segment if not already there
                if segment.transcription:
                    speaker_text.append(segment.transcription)
                else:
                    trans = AssessmentIntegration.get_transcription_for_segment(
                        analysis_report, segment.start_time, segment.end_time
                    )
                    if trans:
                        speaker_text.append(trans)

        return " ".join(speaker_text)

    @staticmethod
    def get_assessment_context(
        assessment: AssessmentReport, analysis_report: AnalysisReport
    ) -> dict[str, Any]:
        """
        Get context information combining analysis and assessment.

        Args:
            assessment: AssessmentReport
            analysis_report: AnalysisReport

        Returns:
            Context dictionary for reporting
        """
        return {
            "video_file": analysis_report.video.file_path,
            "video_duration": analysis_report.video.duration,
            "video_resolution": f"{analysis_report.video.width}x{analysis_report.video.height}",
            "fps": analysis_report.video.fps,
            "language": analysis_report.language or "unknown",
            "has_transcription": analysis_report.has_transcription,
            "has_visual_analysis": analysis_report.has_visual_analysis,
            "total_transcription_words": len(
                analysis_report.full_transcription_text.split()
            ),
            "total_scenes": analysis_report.total_scenes,
            "total_frames": analysis_report.total_frames,
            "assessment_id": assessment.id,
            "assessor": assessment.assessed_by,
            "num_speakers": len(assessment.speaker_profiles),
            "has_rubric": bool(assessment.rubric_application),
            "overall_score": (
                assessment.rubric_application.overall_percentage
                if assessment.rubric_application
                else None
            ),
            "assessment_status": "complete" if assessment.is_complete else "draft",
        }
