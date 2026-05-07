"""Tests for assessment report module."""

from video_analyser.reports.assessment_report import (
    AssessmentDiarizationSegment,
    AssessmentReport,
    AssessmentSpeakerProfile,
    CategoryAssessment,
    CriterionFeedback,
    RubricApplicationResult,
)


class TestAssessmentDiarizationSegment:
    """Tests for AssessmentDiarizationSegment."""

    def test_create_segment(self):
        """Test creating a diarization segment."""
        segment = AssessmentDiarizationSegment(
            speaker_id="speaker_1",
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            confidence=0.95,
        )
        assert segment.speaker_id == "speaker_1"
        assert segment.start_time == 0.0
        assert segment.end_time == 10.0
        assert segment.confidence == 0.95
        assert segment.speaker_label is None

    def test_segment_with_label_and_transcription(self):
        """Test segment with label and transcription."""
        segment = AssessmentDiarizationSegment(
            speaker_id="speaker_1",
            speaker_label="Alice",
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            confidence=0.95,
            transcription="Hello everyone",
        )
        assert segment.speaker_label == "Alice"
        assert segment.transcription == "Hello everyone"


class TestAssessmentSpeakerProfile:
    """Tests for AssessmentSpeakerProfile."""

    def test_create_profile(self):
        """Test creating a speaker profile."""
        profile = AssessmentSpeakerProfile(
            speaker_id="speaker_1",
            num_segments=5,
            total_speaking_time=120.0,
            percentage_of_total=60.0,
            avg_segment_duration=24.0,
        )
        assert profile.speaker_id == "speaker_1"
        assert profile.num_segments == 5
        assert profile.total_speaking_time == 120.0
        assert profile.percentage_of_total == 60.0

    def test_profile_with_label_and_appearances(self):
        """Test profile with speaker label and appearance times."""
        profile = AssessmentSpeakerProfile(
            speaker_id="speaker_1",
            speaker_label="Bob",
            num_segments=3,
            total_speaking_time=90.0,
            percentage_of_total=45.0,
            first_appearance=5.0,
            last_appearance=195.0,
            avg_segment_duration=30.0,
        )
        assert profile.speaker_label == "Bob"
        assert profile.first_appearance == 5.0
        assert profile.last_appearance == 195.0


class TestCriterionFeedback:
    """Tests for CriterionFeedback."""

    def test_create_feedback(self):
        """Test creating criterion feedback."""
        feedback = CriterionFeedback(
            criterion_id="crit_1",
            criterion_name="Clear Speaking",
            score=4.0,
        )
        assert feedback.criterion_id == "crit_1"
        assert feedback.criterion_name == "Clear Speaking"
        assert feedback.score == 4.0
        assert feedback.feedback is None

    def test_feedback_with_details(self):
        """Test feedback with detailed information."""
        feedback = CriterionFeedback(
            criterion_id="crit_1",
            criterion_name="Clear Speaking",
            score=4.0,
            feedback="Good clarity but some mumbling in middle section",
            evidence_timestamps=[(10.0, 20.0), (45.0, 55.0)],
        )
        assert feedback.feedback is not None
        assert len(feedback.evidence_timestamps) == 2


class TestRubricApplicationResult:
    """Tests for RubricApplicationResult."""

    def test_create_application_result(self):
        """Test creating rubric application result."""
        criterion_feedback = CriterionFeedback(
            criterion_id="crit_1",
            criterion_name="Content",
            score=4.0,
        )
        category = CategoryAssessment(
            category_id="cat_1",
            category_name="Content Quality",
            category_weight=1.0,
            criterion_feedbacks=[criterion_feedback],
            category_score=4.0,
            category_percentage=80.0,
        )
        result = RubricApplicationResult(
            rubric_id="rubric_1",
            rubric_name="Presentation Rubric",
            rubric_version=1,
            category_assessments=[category],
            overall_score=4.0,
            overall_percentage=80.0,
        )
        assert result.rubric_id == "rubric_1"
        assert result.overall_percentage == 80.0
        assert len(result.category_assessments) == 1


class TestAssessmentReport:
    """Tests for AssessmentReport."""

    def test_create_assessment_report(self):
        """Test creating an assessment report."""
        report = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/presentation.mp4",
            assessed_by="Dr. Smith",
        )
        assert report.analysis_id == "analysis_123"
        assert report.video_file_path == "/videos/presentation.mp4"
        assert report.assessed_by == "Dr. Smith"
        assert report.is_draft
        assert not report.is_complete

    def test_report_with_diarization(self):
        """Test assessment report with diarization data."""
        segment = AssessmentDiarizationSegment(
            speaker_id="speaker_1",
            speaker_label="Alice",
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            confidence=0.95,
        )
        profile = AssessmentSpeakerProfile(
            speaker_id="speaker_1",
            speaker_label="Alice",
            num_segments=1,
            total_speaking_time=10.0,
            percentage_of_total=100.0,
            avg_segment_duration=10.0,
        )
        report = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/presentation.mp4",
            assessed_by="Dr. Smith",
            diarization_segments=[segment],
            speaker_profiles=[profile],
        )
        assert len(report.diarization_segments) == 1
        assert len(report.speaker_profiles) == 1

    def test_report_serialization(self):
        """Test report to/from dict conversion."""
        segment = AssessmentDiarizationSegment(
            speaker_id="speaker_1",
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            confidence=0.95,
        )
        report = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/presentation.mp4",
            assessed_by="Dr. Smith",
            diarization_segments=[segment],
        )

        # Convert to dict and back
        data = report.to_dict()
        restored = AssessmentReport.from_dict(data)

        assert restored.analysis_id == report.analysis_id
        assert restored.video_file_path == report.video_file_path
        assert restored.assessed_by == report.assessed_by
        assert len(restored.diarization_segments) == 1

    def test_get_speaker_by_id(self):
        """Test retrieving speaker by ID."""
        profile1 = AssessmentSpeakerProfile(
            speaker_id="speaker_1",
            num_segments=1,
            total_speaking_time=10.0,
            percentage_of_total=50.0,
            avg_segment_duration=10.0,
        )
        profile2 = AssessmentSpeakerProfile(
            speaker_id="speaker_2",
            num_segments=1,
            total_speaking_time=10.0,
            percentage_of_total=50.0,
            avg_segment_duration=10.0,
        )
        report = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/presentation.mp4",
            assessed_by="Dr. Smith",
            speaker_profiles=[profile1, profile2],
        )

        found = report.get_speaker_by_id("speaker_1")
        assert found is not None
        assert found.speaker_id == "speaker_1"

        not_found = report.get_speaker_by_id("speaker_999")
        assert not_found is None

    def test_get_speaker_by_label(self):
        """Test retrieving speaker by label."""
        profile = AssessmentSpeakerProfile(
            speaker_id="speaker_1",
            speaker_label="Alice",
            num_segments=1,
            total_speaking_time=10.0,
            percentage_of_total=100.0,
            avg_segment_duration=10.0,
        )
        report = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/presentation.mp4",
            assessed_by="Dr. Smith",
            speaker_profiles=[profile],
        )

        found = report.get_speaker_by_label("Alice")
        assert found is not None
        assert found.speaker_label == "Alice"

    def test_has_overlapping_speech(self):
        """Test detection of overlapping speech."""
        # Non-overlapping segments
        seg1 = AssessmentDiarizationSegment(
            speaker_id="speaker_1",
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            confidence=0.95,
        )
        seg2 = AssessmentDiarizationSegment(
            speaker_id="speaker_2",
            start_time=10.0,
            end_time=20.0,
            duration=10.0,
            confidence=0.95,
        )
        report = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/presentation.mp4",
            assessed_by="Dr. Smith",
            diarization_segments=[seg1, seg2],
        )
        assert not report.has_overlapping_speech()

        # Overlapping segments
        seg3 = AssessmentDiarizationSegment(
            speaker_id="speaker_2",
            start_time=5.0,
            end_time=15.0,
            duration=10.0,
            confidence=0.95,
        )
        report.diarization_segments.append(seg3)
        assert report.has_overlapping_speech()

    def test_get_segments_for_speaker(self):
        """Test getting all segments for a speaker."""
        seg1 = AssessmentDiarizationSegment(
            speaker_id="speaker_1",
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            confidence=0.95,
        )
        seg2 = AssessmentDiarizationSegment(
            speaker_id="speaker_1",
            start_time=20.0,
            end_time=30.0,
            duration=10.0,
            confidence=0.95,
        )
        seg3 = AssessmentDiarizationSegment(
            speaker_id="speaker_2",
            start_time=10.0,
            end_time=20.0,
            duration=10.0,
            confidence=0.95,
        )
        report = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/presentation.mp4",
            assessed_by="Dr. Smith",
            diarization_segments=[seg1, seg2, seg3],
        )

        speaker1_segs = report.get_segments_for_speaker("speaker_1")
        assert len(speaker1_segs) == 2

    def test_add_quality_flag(self):
        """Test adding quality flags."""
        report = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/presentation.mp4",
            assessed_by="Dr. Smith",
        )
        assert len(report.quality_flags) == 0

        report.add_quality_flag("poor_audio")
        assert "poor_audio" in report.quality_flags

        # Adding same flag twice should not duplicate
        report.add_quality_flag("poor_audio")
        assert report.quality_flags.count("poor_audio") == 1

    def test_finalize_assessment(self):
        """Test finalizing an assessment."""
        report = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/presentation.mp4",
            assessed_by="Dr. Smith",
        )
        assert report.is_draft
        assert report.version == 1

        report.finalize()
        assert not report.is_draft
        assert report.is_complete
        assert report.version == 2
