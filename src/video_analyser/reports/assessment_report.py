"""Assessment and marking report generation for presentations."""

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class AssessmentDiarizationSegment(BaseModel):
    """Speaker segment for assessment report."""

    speaker_id: str = Field(..., description="Speaker identifier")
    speaker_label: str | None = Field(None, description="Human-readable speaker label")
    start_time: float = Field(..., description="Segment start time in seconds")
    end_time: float = Field(..., description="Segment end time in seconds")
    duration: float = Field(..., description="Segment duration in seconds")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Diarization confidence")
    transcription: str | None = Field(
        None, description="Transcribed text for this segment"
    )


class AssessmentSpeakerProfile(BaseModel):
    """Speaker profile for assessment."""

    speaker_id: str = Field(..., description="Speaker identifier")
    speaker_label: str | None = Field(None, description="Human-assigned label")
    num_segments: int = Field(..., description="Number of speaking segments")
    total_speaking_time: float = Field(
        ..., description="Total speaking time in seconds"
    )
    percentage_of_total: float = Field(
        ..., ge=0.0, le=100.0, description="Percentage of total speaking time"
    )
    first_appearance: float | None = Field(None, description="First appearance time")
    last_appearance: float | None = Field(None, description="Last appearance time")
    avg_segment_duration: float = Field(..., description="Average segment duration")


class CriterionFeedback(BaseModel):
    """Feedback for a specific rubric criterion."""

    criterion_id: str = Field(..., description="Rubric criterion ID")
    criterion_name: str = Field(..., description="Criterion name")
    score: float = Field(..., description="Score given for this criterion")
    feedback: str | None = Field(None, description="Detailed feedback on criterion")
    evidence_timestamps: list[tuple[float, float]] = Field(
        default_factory=list,
        description="List of (start, end) time tuples supporting this score",
    )


class CategoryAssessment(BaseModel):
    """Assessment for a rubric category."""

    category_id: str = Field(..., description="Category ID")
    category_name: str = Field(..., description="Category name")
    category_weight: float = Field(..., description="Category weight in overall score")
    criterion_feedbacks: list[CriterionFeedback] = Field(
        ..., description="Feedbacks for each criterion"
    )
    category_score: float = Field(..., description="Weighted score for category")
    category_percentage: float = Field(
        ..., ge=0.0, le=100.0, description="Percentage for category"
    )


class RubricApplicationResult(BaseModel):
    """Result of applying a rubric to an assessment."""

    rubric_id: str = Field(..., description="Rubric ID used")
    rubric_name: str = Field(..., description="Rubric name")
    rubric_version: int = Field(default=1, description="Rubric version used")
    category_assessments: list[CategoryAssessment] = Field(
        ..., description="Assessment for each category"
    )
    overall_score: float = Field(..., description="Overall weighted score")
    overall_percentage: float = Field(
        ..., ge=0.0, le=100.0, description="Overall percentage"
    )
    general_feedback: str | None = Field(
        None, description="Overall assessment feedback"
    )


class AssessmentReport(BaseModel):
    """Complete assessment and marking report for a presentation."""

    # Identifiers
    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique assessment ID"
    )
    analysis_id: str = Field(..., description="Reference to parent AnalysisReport ID")
    video_file_path: str = Field(..., description="Path to assessed video")

    # Assessment metadata
    assessed_by: str = Field(..., description="Name/ID of assessor")
    assessed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Assessment creation time",
    )
    last_modified_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last modification time",
    )
    assessment_notes: str | None = Field(None, description="Overall assessment notes")

    # Diarization results
    diarization_status: str = Field(
        default="completed",
        description="Status of diarization (completed/pending/failed)",
    )
    diarization_segments: list[AssessmentDiarizationSegment] = Field(
        default_factory=list, description="Speaker segments detected"
    )
    speaker_profiles: list[AssessmentSpeakerProfile] = Field(
        default_factory=list, description="Speaker summary information"
    )

    # Rubric application
    rubric_application: RubricApplicationResult | None = Field(
        None, description="Applied rubric and scores"
    )

    # Status and flags
    is_complete: bool = Field(
        default=False, description="Whether assessment is finalized"
    )
    is_draft: bool = Field(
        default=True, description="Whether this is a draft assessment"
    )
    quality_flags: list[str] = Field(
        default_factory=list,
        description="Flags for quality issues (e.g., 'poor_audio_quality', 'multiple_speakers_overlapping')",
    )

    # Versioning
    version: int = Field(default=1, ge=1, description="Assessment version")

    @field_validator("video_file_path", mode="before")
    @classmethod
    def validate_path(cls, v: Any) -> str:
        """Ensure path is string."""
        return str(v) if v else ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssessmentReport":
        """Create from dictionary."""
        # Parse datetime strings if needed
        if isinstance(data.get("assessed_at"), str):
            data["assessed_at"] = datetime.fromisoformat(data["assessed_at"])
        if isinstance(data.get("last_modified_at"), str):
            data["last_modified_at"] = datetime.fromisoformat(data["last_modified_at"])

        return cls(**data)

    def get_speaker_by_id(self, speaker_id: str) -> AssessmentSpeakerProfile | None:
        """Get speaker profile by ID."""
        for profile in self.speaker_profiles:
            if profile.speaker_id == speaker_id:
                return profile
        return None

    def get_speaker_by_label(self, label: str) -> AssessmentSpeakerProfile | None:
        """Get speaker profile by human label."""
        for profile in self.speaker_profiles:
            if profile.speaker_label == label:
                return profile
        return None

    def has_overlapping_speech(self) -> bool:
        """Check if there are overlapping speaker segments."""
        segments = sorted(self.diarization_segments, key=lambda s: s.start_time)
        for i, seg1 in enumerate(segments):
            for seg2 in segments[i + 1 :]:
                if seg1.end_time > seg2.start_time:
                    return True
        return False

    def get_segments_for_speaker(
        self, speaker_id: str
    ) -> list[AssessmentDiarizationSegment]:
        """Get all segments for a specific speaker."""
        return [s for s in self.diarization_segments if s.speaker_id == speaker_id]

    def add_quality_flag(self, flag: str) -> None:
        """Add a quality flag."""
        if flag not in self.quality_flags:
            self.quality_flags.append(flag)

    def finalize(self) -> None:
        """Mark assessment as complete and finalized."""
        self.is_complete = True
        self.is_draft = False
        self.last_modified_at = datetime.now(UTC)
        self.version += 1
