"""Tests for assessment session module."""

import tempfile

import pytest

from video_analyser.analysis.rubric_system import (
    RubricBuilder,
)
from video_analyser.analysis.speaker_diarization import (
    DiarizationResult,
    SpeakerProfile,
    SpeakerSegment,
)
from video_analyser.reports.assessment_session import AssessmentSession


class TestAssessmentSession:
    """Tests for AssessmentSession."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_rubric(self):
        """Create a sample rubric for testing."""
        builder = RubricBuilder("Test Rubric")
        cat_builder = builder.add_category("Content", weight=1.0)
        cat_builder.add_criterion("Clarity", weight=1.0)
        cat_builder.add_criterion("Completeness", weight=1.0)
        return builder.build()

    @pytest.fixture
    def sample_diarization(self):
        """Create sample diarization result."""
        seg1 = SpeakerSegment(
            speaker_id="speaker_1",
            start_time=0.0,
            end_time=30.0,
            confidence=0.95,
            duration=30.0,
        )
        seg2 = SpeakerSegment(
            speaker_id="speaker_2",
            start_time=30.0,
            end_time=60.0,
            confidence=0.92,
            duration=30.0,
        )
        prof1 = SpeakerProfile(
            speaker_id="speaker_1",
            num_segments=1,
            total_speaking_time=30.0,
            percentage_of_total=50.0,
            avg_segment_duration=30.0,
        )
        prof2 = SpeakerProfile(
            speaker_id="speaker_2",
            num_segments=1,
            total_speaking_time=30.0,
            percentage_of_total=50.0,
            avg_segment_duration=30.0,
        )
        return DiarizationResult(
            audio_path="test.wav",
            num_speakers=2,
            segments=[seg1, seg2],
            speakers=[prof1, prof2],
            total_duration=60.0,
            processing_time=5.0,
        )

    def test_session_initialization(self, temp_storage):
        """Test session initialization."""
        session = AssessmentSession(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessor_name="Dr. Smith",
            storage_dir=temp_storage,
        )
        assert session.analysis_id == "analysis_123"
        assert session.assessor_name == "Dr. Smith"
        assert session.assessment.is_draft

    def test_session_loads_existing_draft(self, temp_storage, sample_diarization):
        """Test that session loads existing draft assessment."""
        # Create first session and save draft
        session1 = AssessmentSession(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessor_name="Dr. Smith",
            storage_dir=temp_storage,
        )
        session1.add_diarization_result(sample_diarization)
        draft_id = session1.save_draft()

        # Create second session - should load the draft
        session2 = AssessmentSession(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessor_name="Dr. Smith",
            storage_dir=temp_storage,
        )
        assert session2.assessment.id == draft_id
        assert len(session2.assessment.speaker_profiles) == 2

    def test_add_diarization_result(self, temp_storage, sample_diarization):
        """Test adding diarization results."""
        session = AssessmentSession(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessor_name="Dr. Smith",
            storage_dir=temp_storage,
        )

        session.add_diarization_result(sample_diarization)

        assert len(session.assessment.diarization_segments) == 2
        assert len(session.assessment.speaker_profiles) == 2
        assert session.assessment.diarization_status == "completed"

    def test_set_speaker_label(self, temp_storage, sample_diarization):
        """Test setting speaker labels."""
        session = AssessmentSession(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessor_name="Dr. Smith",
            storage_dir=temp_storage,
        )
        session.add_diarization_result(sample_diarization)

        success = session.set_speaker_label("speaker_1", "Alice")
        assert success

        # Check update in segments
        assert session.assessment.diarization_segments[0].speaker_label == "Alice"

        # Check update in profiles
        assert session.assessment.speaker_profiles[0].speaker_label == "Alice"

    def test_set_speaker_label_nonexistent(self, temp_storage):
        """Test setting label for non-existent speaker."""
        session = AssessmentSession(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessor_name="Dr. Smith",
            storage_dir=temp_storage,
        )
        success = session.set_speaker_label("nonexistent", "Alice")
        assert not success

    def test_apply_rubric(self, temp_storage, sample_diarization, sample_rubric):
        """Test applying a rubric."""
        session = AssessmentSession(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessor_name="Dr. Smith",
            storage_dir=temp_storage,
        )
        session.add_diarization_result(sample_diarization)

        # Apply rubric with scores
        category_id = sample_rubric.categories[0].id
        criterion_ids = [c.id for c in sample_rubric.categories[0].criteria]
        criterion_scores = {category_id: {criterion_ids[0]: 4, criterion_ids[1]: 5}}

        session.apply_rubric(sample_rubric, criterion_scores)

        assert session.assessment.rubric_application is not None
        assert session.assessment.rubric_application.rubric_name == "Test Rubric"
        assert session.assessment.rubric_application.overall_percentage > 0

    def test_add_criterion_feedback(
        self, temp_storage, sample_diarization, sample_rubric
    ):
        """Test adding criterion feedback."""
        session = AssessmentSession(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessor_name="Dr. Smith",
            storage_dir=temp_storage,
        )
        session.add_diarization_result(sample_diarization)

        # Apply rubric first
        category_id = sample_rubric.categories[0].id
        criterion_ids = [c.id for c in sample_rubric.categories[0].criteria]
        criterion_scores = {category_id: {criterion_ids[0]: 4, criterion_ids[1]: 5}}
        session.apply_rubric(sample_rubric, criterion_scores)

        # Add feedback
        session.add_criterion_feedback(
            criterion_ids[0],
            4,
            feedback="Very clear presentation",
            evidence_timestamps=[(10.0, 20.0)],
        )

        # Find the feedback in the assessment
        feedback = None
        for cat in session.assessment.rubric_application.category_assessments:
            for crit_fb in cat.criterion_feedbacks:
                if crit_fb.criterion_id == criterion_ids[0]:
                    feedback = crit_fb
                    break

        assert feedback is not None
        assert feedback.feedback == "Very clear presentation"
        assert len(feedback.evidence_timestamps) == 1

    def test_set_assessment_notes(self, temp_storage):
        """Test setting assessment notes."""
        session = AssessmentSession(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessor_name="Dr. Smith",
            storage_dir=temp_storage,
        )
        notes = "Overall good presentation with minor issues"
        session.set_assessment_notes(notes)

        assert session.assessment.assessment_notes == notes

    def test_set_general_feedback(
        self, temp_storage, sample_diarization, sample_rubric
    ):
        """Test setting general feedback."""
        session = AssessmentSession(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessor_name="Dr. Smith",
            storage_dir=temp_storage,
        )
        session.add_diarization_result(sample_diarization)

        # Apply rubric first
        category_id = sample_rubric.categories[0].id
        criterion_ids = [c.id for c in sample_rubric.categories[0].criteria]
        criterion_scores = {category_id: {criterion_ids[0]: 4, criterion_ids[1]: 5}}
        session.apply_rubric(sample_rubric, criterion_scores)

        feedback = "Great job overall!"
        session.set_general_feedback(feedback)

        assert session.assessment.rubric_application.general_feedback == feedback

    def test_add_quality_flag(self, temp_storage):
        """Test adding quality flags."""
        session = AssessmentSession(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessor_name="Dr. Smith",
            storage_dir=temp_storage,
        )
        session.add_quality_flag("poor_audio")

        assert "poor_audio" in session.assessment.quality_flags

    def test_save_draft(self, temp_storage):
        """Test saving draft assessment."""
        session = AssessmentSession(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessor_name="Dr. Smith",
            storage_dir=temp_storage,
        )
        draft_id = session.save_draft()

        assert session.assessment.is_draft
        assert draft_id == session.assessment.id

    def test_finalize_without_rubric(self, temp_storage):
        """Test that finalizing without rubric raises error."""
        session = AssessmentSession(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessor_name="Dr. Smith",
            storage_dir=temp_storage,
        )

        with pytest.raises(ValueError):
            session.finalize()

    def test_finalize_with_rubric(
        self, temp_storage, sample_diarization, sample_rubric
    ):
        """Test finalizing assessment."""
        session = AssessmentSession(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessor_name="Dr. Smith",
            storage_dir=temp_storage,
        )
        session.add_diarization_result(sample_diarization)

        # Apply rubric
        category_id = sample_rubric.categories[0].id
        criterion_ids = [c.id for c in sample_rubric.categories[0].criteria]
        criterion_scores = {category_id: {criterion_ids[0]: 4, criterion_ids[1]: 5}}
        session.apply_rubric(sample_rubric, criterion_scores)

        # Finalize
        assessment_id = session.finalize()

        assert not session.assessment.is_draft
        assert session.assessment.is_complete
        assert assessment_id == session.assessment.id

    def test_get_summary(self, temp_storage, sample_diarization, sample_rubric):
        """Test getting assessment summary."""
        session = AssessmentSession(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessor_name="Dr. Smith",
            storage_dir=temp_storage,
        )
        session.add_diarization_result(sample_diarization)

        category_id = sample_rubric.categories[0].id
        criterion_ids = [c.id for c in sample_rubric.categories[0].criteria]
        criterion_scores = {category_id: {criterion_ids[0]: 4, criterion_ids[1]: 5}}
        session.apply_rubric(sample_rubric, criterion_scores)

        summary = session.get_summary()

        assert summary["assessor"] == "Dr. Smith"
        assert summary["speakers_identified"] == 2
        assert summary["rubric_applied"]
        assert summary["overall_score"] is not None
