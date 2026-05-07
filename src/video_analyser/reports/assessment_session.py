"""Assessment session management for marking workflow."""

import logging
from pathlib import Path
from typing import Any

from ..analysis.rubric_system import Rubric, RubricScorer
from ..analysis.speaker_diarization import DiarizationResult
from .assessment_report import (
    AssessmentDiarizationSegment,
    AssessmentReport,
    AssessmentSpeakerProfile,
    CategoryAssessment,
    CriterionFeedback,
    RubricApplicationResult,
)
from .assessment_storage import AssessmentStorage

logger = logging.getLogger(__name__)


class AssessmentSession:
    """Manage a marking/assessment session for a single video analysis."""

    def __init__(
        self,
        analysis_id: str,
        video_file_path: str | Path,
        assessor_name: str,
        storage_dir: str | Path = "assessments",
    ):
        """
        Initialize an assessment session.

        Args:
            analysis_id: Reference to parent AnalysisReport ID
            video_file_path: Path to the video being assessed
            assessor_name: Name/ID of the assessor
            storage_dir: Directory for storing assessment data
        """
        self.analysis_id = analysis_id
        self.video_file_path = str(video_file_path)
        self.assessor_name = assessor_name
        self.storage = AssessmentStorage(storage_dir)

        # Try to load existing draft assessment or create new one
        draft = self.storage.get_draft_assessment(analysis_id)
        if draft:
            self.assessment = draft
            logger.info(f"Loaded draft assessment {draft.id}")
        else:
            self.assessment = AssessmentReport(
                analysis_id=analysis_id,
                video_file_path=self.video_file_path,
                assessed_by=assessor_name,
                assessment_notes=None,
                rubric_application=None,
            )
            logger.info(f"Created new assessment session {self.assessment.id}")

    def add_diarization_result(self, diarization: DiarizationResult) -> None:
        """
        Add diarization results to the assessment.

        Args:
            diarization: DiarizationResult from speaker diarization
        """
        # Convert segments
        self.assessment.diarization_segments = [
            AssessmentDiarizationSegment(
                speaker_id=seg.speaker_id,
                speaker_label=seg.label,
                start_time=seg.start_time,
                end_time=seg.end_time,
                duration=seg.duration,
                confidence=seg.confidence,
                transcription=None,  # Will be filled in with transcription integration
            )
            for seg in diarization.segments
        ]

        # Convert profiles
        self.assessment.speaker_profiles = [
            AssessmentSpeakerProfile(
                speaker_id=prof.speaker_id,
                speaker_label=prof.label,
                num_segments=prof.num_segments,
                total_speaking_time=prof.total_speaking_time,
                percentage_of_total=prof.percentage_of_total,
                first_appearance=prof.first_appearance,
                last_appearance=prof.last_appearance,
                avg_segment_duration=prof.avg_segment_duration,
            )
            for prof in diarization.speakers
        ]

        self.assessment.diarization_status = "completed"

        # Flag overlapping speech
        if diarization.overlapping_speech_segments:
            self.assessment.add_quality_flag("overlapping_speech")

        logger.info(f"Added diarization with {len(diarization.speakers)} speakers")

    def set_speaker_label(self, speaker_id: str, label: str) -> bool:
        """
        Set or update human-readable label for a speaker.

        Args:
            speaker_id: Speaker ID to label
            label: Human-readable label (e.g., "Alice", "Bob")

        Returns:
            True if successful, False if speaker not found
        """
        # Update in segments
        for segment in self.assessment.diarization_segments:
            if segment.speaker_id == speaker_id:
                segment.speaker_label = label

        # Update in profiles
        for profile in self.assessment.speaker_profiles:
            if profile.speaker_id == speaker_id:
                profile.speaker_label = label
                return True

        logger.warning(f"Speaker {speaker_id} not found")
        return False

    def add_criterion_feedback(
        self,
        criterion_id: str,
        score: float,
        feedback: str | None = None,
        evidence_timestamps: list[tuple[float, float]] | None = None,
    ) -> None:
        """
        Add feedback for a rubric criterion.

        Args:
            criterion_id: Rubric criterion ID
            score: Score given (0-1 or as per rubric scale)
            feedback: Detailed feedback text
            evidence_timestamps: List of (start, end) time tuples supporting this score
        """
        if not self.assessment.rubric_application:
            logger.warning("No rubric applied yet; cannot add criterion feedback")
            return

        # Find the category containing this criterion
        for category in self.assessment.rubric_application.category_assessments:
            existing = next(
                (
                    f
                    for f in category.criterion_feedbacks
                    if f.criterion_id == criterion_id
                ),
                None,
            )
            if existing:
                # Update existing feedback
                existing.score = score
                existing.feedback = feedback
                existing.evidence_timestamps = evidence_timestamps or []
                logger.info(f"Updated feedback for criterion {criterion_id}")
                return

        logger.warning(f"Criterion {criterion_id} not found in applied rubric")

    def apply_rubric(
        self, rubric: Rubric, criterion_scores: dict[str, dict[str, int | float]]
    ) -> None:
        """
        Apply a rubric and calculate scores.

        Args:
            rubric: Rubric to apply
            criterion_scores: Dict mapping category_id -> criterion_id -> score
                Example: {"cat1": {"crit1": 4, "crit2": 5}}
        """
        scorer = RubricScorer(rubric)
        # Cast to match RubricScorer's expected int type (it handles floats internally)
        int_scores: dict[str, dict[str, int]] = {  # type: ignore[assignment]
            cat_id: {crit_id: int(score) for crit_id, score in crit_scores.items()}
            for cat_id, crit_scores in criterion_scores.items()
        }
        assessment_result = scorer.score_all_categories(int_scores)

        # Convert to AssessmentReport structure
        category_assessments: list[CategoryAssessment] = []

        for category in rubric.categories:
            # Collect criterion feedbacks for this category
            criterion_feedbacks: list[CriterionFeedback] = []

            for criterion in category.criteria:
                score = criterion_scores.get(category.id, {}).get(criterion.id, 0)
                criterion_feedbacks.append(
                    CriterionFeedback(
                        criterion_id=criterion.id,
                        criterion_name=criterion.name,
                        score=score,
                        feedback=None,  # Can be added later via add_criterion_feedback
                        evidence_timestamps=[],
                    )
                )

            # Find category result
            cat_score = next(
                (
                    cs
                    for cs in assessment_result.category_scores
                    if cs.category_id == category.id
                ),
                None,
            )

            if cat_score:
                category_assessments.append(
                    CategoryAssessment(
                        category_id=category.id,
                        category_name=category.name,
                        category_weight=category.weight,
                        criterion_feedbacks=criterion_feedbacks,
                        category_score=cat_score.category_total,
                        category_percentage=cat_score.category_percentage,
                    )
                )

        self.assessment.rubric_application = RubricApplicationResult(
            rubric_id=rubric.id,
            rubric_name=rubric.name,
            rubric_version=rubric.version,
            category_assessments=category_assessments,
            overall_score=assessment_result.overall_score,
            overall_percentage=assessment_result.overall_percentage,
            general_feedback=None,
        )

        logger.info(
            f"Applied rubric {rubric.name} with overall score {assessment_result.overall_percentage:.1f}%"
        )

    def set_assessment_notes(self, notes: str) -> None:
        """
        Set overall assessment notes.

        Args:
            notes: Assessment notes/comments
        """
        self.assessment.assessment_notes = notes
        logger.info("Assessment notes updated")

    def set_general_feedback(self, feedback: str) -> None:
        """
        Set general feedback for the rubric application.

        Args:
            feedback: General feedback text
        """
        if self.assessment.rubric_application:
            self.assessment.rubric_application.general_feedback = feedback
            logger.info("General feedback updated")
        else:
            logger.warning("No rubric applied; cannot set general feedback")

    def add_quality_flag(self, flag: str) -> None:
        """
        Add a quality/concern flag.

        Args:
            flag: Flag name (e.g., 'poor_audio', 'multiple_speakers_overlapping')
        """
        self.assessment.add_quality_flag(flag)
        logger.info(f"Added quality flag: {flag}")

    def save_draft(self) -> str:
        """
        Save current assessment as draft.

        Returns:
            Assessment ID
        """
        self.assessment.is_draft = True
        self.storage.save(self.assessment)
        logger.info(f"Assessment {self.assessment.id} saved as draft")
        return self.assessment.id

    def finalize(self) -> str:
        """
        Finalize the assessment (mark as complete).

        Returns:
            Assessment ID
        """
        # Validate that rubric is applied
        if not self.assessment.rubric_application:
            raise ValueError("Cannot finalize assessment without applied rubric")

        # Validate that speakers are labeled if diarization was done
        if self.assessment.diarization_segments:
            for segment in self.assessment.diarization_segments:
                if not segment.speaker_label:
                    logger.warning(
                        f"Speaker {segment.speaker_id} not labeled; "
                        "consider adding labels before finalizing"
                    )

        self.assessment.finalize()
        self.storage.save(self.assessment)
        logger.info(f"Assessment {self.assessment.id} finalized")
        return self.assessment.id

    def get_assessment(self) -> AssessmentReport:
        """Get the current assessment object."""
        return self.assessment

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary of current assessment.

        Returns:
            Summary dictionary
        """
        return {
            "assessment_id": self.assessment.id,
            "analysis_id": self.analysis_id,
            "assessor": self.assessor_name,
            "status": "complete" if self.assessment.is_complete else "draft",
            "speakers_identified": len(self.assessment.speaker_profiles),
            "rubric_applied": bool(self.assessment.rubric_application),
            "overall_score": (
                self.assessment.rubric_application.overall_percentage
                if self.assessment.rubric_application
                else None
            ),
            "quality_flags": self.assessment.quality_flags,
            "assessed_at": str(self.assessment.assessed_at),
        }
