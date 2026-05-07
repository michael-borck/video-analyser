"""Tests for assessment integration and grading sheet rendering."""

import tempfile
from pathlib import Path

import pytest

from video_analyser.reports.assessment_integration import AssessmentIntegration
from video_analyser.reports.assessment_report import (
    AssessmentDiarizationSegment,
    AssessmentReport,
    AssessmentSpeakerProfile,
)
from video_analyser.reports.grading_sheet_renderer import GradingSheetRenderer
from video_analyser.reports.report_generator import (
    AnalysisReport,
    AudioMetadata,
    SpeechMetrics,
    TranscriptionSegment,
    VideoMetadata,
)


class TestAssessmentIntegration:
    """Tests for AssessmentIntegration."""

    @pytest.fixture
    def sample_analysis_report(self):
        """Create a sample AnalysisReport."""
        return AnalysisReport(
            video=VideoMetadata(
                file_path="/videos/test.mp4",
                duration=120.0,
                width=1920,
                height=1080,
                fps=30.0,
                format="mp4",
            ),
            audio=AudioMetadata(
                duration=120.0,
                sample_rate=48000,
                channels=2,
            ),
            transcription_segments=[
                TranscriptionSegment(
                    id=1,
                    text="Hello everyone",
                    start_time=0.0,
                    end_time=5.0,
                    confidence=0.95,
                    num_words=2,
                ),
                TranscriptionSegment(
                    id=2,
                    text="Today I will discuss the project",
                    start_time=5.0,
                    end_time=10.0,
                    confidence=0.92,
                    num_words=6,
                ),
                TranscriptionSegment(
                    id=3,
                    text="The results are promising",
                    start_time=10.0,
                    end_time=15.0,
                    confidence=0.90,
                    num_words=4,
                ),
            ],
            full_transcription_text="Hello everyone. Today I will discuss the project. The results are promising.",
            language="en",
            speech_metrics=SpeechMetrics(
                total_words=12,
                total_speech_duration=15.0,
                speaking_rate_wpm=120.0,
                average_confidence=0.92,
            ),
            has_transcription=True,
            has_visual_analysis=True,
        )

    @pytest.fixture
    def sample_assessment(self):
        """Create a sample AssessmentReport."""
        seg1 = AssessmentDiarizationSegment(
            speaker_id="speaker_1",
            speaker_label="Alice",
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            confidence=0.95,
        )
        seg2 = AssessmentDiarizationSegment(
            speaker_id="speaker_2",
            speaker_label="Bob",
            start_time=10.0,
            end_time=15.0,
            duration=5.0,
            confidence=0.92,
        )
        prof1 = AssessmentSpeakerProfile(
            speaker_id="speaker_1",
            speaker_label="Alice",
            num_segments=1,
            total_speaking_time=10.0,
            percentage_of_total=66.7,
            avg_segment_duration=10.0,
        )
        prof2 = AssessmentSpeakerProfile(
            speaker_id="speaker_2",
            speaker_label="Bob",
            num_segments=1,
            total_speaking_time=5.0,
            percentage_of_total=33.3,
            avg_segment_duration=5.0,
        )
        return AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessed_by="Dr. Smith",
            diarization_segments=[seg1, seg2],
            speaker_profiles=[prof1, prof2],
        )

    def test_get_transcription_for_segment(self, sample_analysis_report):
        """Test getting transcription for a time segment."""
        # Get transcription for segment 0-7 (covers first two segments)
        text = AssessmentIntegration.get_transcription_for_segment(
            sample_analysis_report, 0.0, 7.0
        )
        assert "Hello everyone" in text
        assert "Today I will discuss the project" in text

    def test_get_transcription_no_overlap(self, sample_analysis_report):
        """Test getting transcription for non-overlapping segment."""
        # Get transcription for segment outside range
        text = AssessmentIntegration.get_transcription_for_segment(
            sample_analysis_report, 50.0, 60.0
        )
        assert text == ""

    def test_enrich_assessment_with_analysis(
        self, sample_assessment, sample_analysis_report
    ):
        """Test enriching assessment with analysis data."""
        # Before enrichment, segments should have no transcription
        assert not sample_assessment.diarization_segments[0].transcription

        # Enrich
        enriched = AssessmentIntegration.enrich_assessment_with_analysis(
            sample_assessment, sample_analysis_report
        )

        # After enrichment, segments should have transcription
        assert enriched.diarization_segments[0].transcription
        assert "Hello everyone" in enriched.diarization_segments[0].transcription

    def test_create_assessment_from_analysis(self, sample_analysis_report):
        """Test creating assessment session from analysis report."""
        session = AssessmentIntegration.create_assessment_from_analysis(
            sample_analysis_report,
            assessor_name="Dr. Jones",
        )

        assert session.assessment.assessed_by == "Dr. Jones"
        assert session.assessment.video_file_path == "/videos/test.mp4"

    def test_get_speaker_transcription(self, sample_assessment, sample_analysis_report):
        """Test getting transcription for a specific speaker."""
        # First enrich so we have transcriptions
        enriched = AssessmentIntegration.enrich_assessment_with_analysis(
            sample_assessment, sample_analysis_report
        )

        # Get transcription for speaker_1
        text = AssessmentIntegration.get_speaker_transcription(
            enriched, sample_analysis_report, "speaker_1"
        )

        assert text  # Should have some text
        assert "Hello everyone" in text or "Today I will discuss" in text

    def test_get_assessment_context(self, sample_assessment, sample_analysis_report):
        """Test getting assessment context."""
        context = AssessmentIntegration.get_assessment_context(
            sample_assessment, sample_analysis_report
        )

        assert context["video_duration"] == 120.0
        assert context["video_resolution"] == "1920x1080"
        assert context["fps"] == 30.0
        assert context["language"] == "en"
        assert context["num_speakers"] == 2
        assert context["assessor"] == "Dr. Smith"


class TestGradingSheetRenderer:
    """Tests for GradingSheetRenderer."""

    @pytest.fixture
    def renderer_with_analysis(self):
        """Create renderer with analysis report."""
        analysis = AnalysisReport(
            video=VideoMetadata(
                file_path="/videos/test.mp4",
                duration=120.0,
                width=1920,
                height=1080,
                fps=30.0,
            ),
            transcription_segments=[
                TranscriptionSegment(
                    id=1,
                    text="Test transcription",
                    start_time=0.0,
                    end_time=10.0,
                    confidence=0.95,
                    num_words=2,
                ),
            ],
        )
        return GradingSheetRenderer(analysis)

    @pytest.fixture
    def sample_assessment(self):
        """Create sample assessment."""
        return AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessed_by="Dr. Smith",
            assessment_notes="Good presentation overall",
        )

    def test_render_basic(self, renderer_with_analysis, sample_assessment):
        """Test basic HTML rendering."""
        html = renderer_with_analysis.render(sample_assessment)

        assert "<!DOCTYPE html>" in html
        assert "Grading Sheet" in html
        assert "Dr. Smith" in html
        assert sample_assessment.id in html

    def test_render_with_speakers(self, renderer_with_analysis):
        """Test rendering with speaker data."""
        seg = AssessmentDiarizationSegment(
            speaker_id="speaker_1",
            speaker_label="Alice",
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            confidence=0.95,
            transcription="Test speech",
        )
        prof = AssessmentSpeakerProfile(
            speaker_id="speaker_1",
            speaker_label="Alice",
            num_segments=1,
            total_speaking_time=10.0,
            percentage_of_total=100.0,
            first_appearance=0.0,
            last_appearance=10.0,
            avg_segment_duration=10.0,
        )
        assessment = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessed_by="Dr. Smith",
            diarization_segments=[seg],
            speaker_profiles=[prof],
        )

        html = renderer_with_analysis.render(assessment)

        assert "Alice" in html
        assert "100.0%" in html

    def test_render_with_quality_flags(self, renderer_with_analysis):
        """Test rendering with quality flags."""
        assessment = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessed_by="Dr. Smith",
            quality_flags=["poor_audio", "overlapping_speech"],
        )

        html = renderer_with_analysis.render(assessment)

        assert "poor_audio" in html
        assert "overlapping_speech" in html

    def test_render_to_file(self, renderer_with_analysis, sample_assessment):
        """Test rendering to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "grading_sheet.html"
            result_path = renderer_with_analysis.render_to_file(
                sample_assessment, output_path
            )

            assert result_path.exists()
            assert result_path == output_path

            # Read and verify content
            with open(result_path) as f:
                content = f.read()

            assert "<!DOCTYPE html>" in content
            assert "Grading Sheet" in content

    def test_css_included(self, renderer_with_analysis, sample_assessment):
        """Test that CSS is included in output."""
        html = renderer_with_analysis.render(sample_assessment)

        assert "<style>" in html
        assert "body {" in html
        assert ".container {" in html
        assert ".speaker-table {" in html

    def test_render_video_info(self, renderer_with_analysis, sample_assessment):
        """Test video information section rendering."""
        html = renderer_with_analysis.render(sample_assessment)

        assert "Video Information" in html
        assert sample_assessment.video_file_path in html
        assert "1920x1080" in html  # Resolution
        assert "30.0" in html  # FPS

    def test_render_empty_rubric(self, renderer_with_analysis):
        """Test rendering assessment without rubric."""
        assessment = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessed_by="Dr. Smith",
        )

        html = renderer_with_analysis.render(assessment)

        assert "No rubric applied yet" in html

    def test_render_full_assessment(self, renderer_with_analysis, sample_assessment):
        """Test rendering complete assessment."""
        # Add all components
        sample_assessment.assessment_notes = "Overall excellent work"
        sample_assessment.quality_flags.append("excellent_audio")

        html = renderer_with_analysis.render(sample_assessment)

        assert "Overall excellent work" in html
        assert "excellent_audio" in html
        assert "Assessment Notes" in html

    def test_css_media_print(self, renderer_with_analysis, sample_assessment):
        """Test that CSS includes print media query."""
        html = renderer_with_analysis.render(sample_assessment)

        assert "@media print" in html
        assert "box-shadow: none" in html
