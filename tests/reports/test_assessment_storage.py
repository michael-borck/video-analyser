"""Tests for assessment storage module."""

import tempfile

import pytest

from video_analyser.reports.assessment_report import (
    AssessmentReport,
)
from video_analyser.reports.assessment_storage import AssessmentStorage


class TestAssessmentStorage:
    """Tests for AssessmentStorage."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield AssessmentStorage(tmpdir)

    def test_storage_initialization(self, temp_storage):
        """Test storage initialization creates directory."""
        assert temp_storage.storage_dir.exists()

    def test_save_and_load_assessment(self, temp_storage):
        """Test saving and loading an assessment."""
        report = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessed_by="Dr. Smith",
        )
        assessment_id = report.id

        # Save
        path = temp_storage.save(report)
        assert path.exists()

        # Load
        loaded = temp_storage.load(assessment_id)
        assert loaded is not None
        assert loaded.analysis_id == "analysis_123"
        assert loaded.assessed_by == "Dr. Smith"

    def test_load_nonexistent_assessment(self, temp_storage):
        """Test loading non-existent assessment returns None."""
        loaded = temp_storage.load("nonexistent")
        assert loaded is None

    def test_list_assessments(self, temp_storage):
        """Test listing assessments."""
        report1 = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/test1.mp4",
            assessed_by="Dr. Smith",
        )
        report2 = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/test2.mp4",
            assessed_by="Dr. Jones",
        )
        report3 = AssessmentReport(
            analysis_id="analysis_456",
            video_file_path="/videos/test3.mp4",
            assessed_by="Dr. Smith",
        )

        temp_storage.save(report1)
        temp_storage.save(report2)
        temp_storage.save(report3)

        # List all
        all_assessments = temp_storage.list_assessments()
        assert len(all_assessments) == 3

        # Filter by analysis_id
        analysis_123 = temp_storage.list_assessments(analysis_id="analysis_123")
        assert len(analysis_123) == 2

        # Filter by assessor
        smith_assessments = temp_storage.list_assessments(assessed_by="Dr. Smith")
        assert len(smith_assessments) == 2

    def test_list_assessments_excludes_drafts(self, temp_storage):
        """Test that list_assessments excludes drafts by default."""
        draft = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessed_by="Dr. Smith",
            is_draft=True,
        )
        completed = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/test2.mp4",
            assessed_by="Dr. Smith",
            is_draft=False,
        )

        temp_storage.save(draft)
        temp_storage.save(completed)

        # Exclude drafts
        assessments = temp_storage.list_assessments(include_drafts=False)
        assert len(assessments) == 1
        assert assessments[0].id == completed.id

        # Include drafts
        all_assessments = temp_storage.list_assessments(include_drafts=True)
        assert len(all_assessments) == 2

    def test_get_latest_assessment(self, temp_storage):
        """Test getting the latest assessment for an analysis."""
        report1 = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessed_by="Dr. Smith",
            is_draft=False,
        )
        report2 = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessed_by="Dr. Jones",
            is_draft=False,
        )

        temp_storage.save(report1)
        temp_storage.save(report2)

        latest = temp_storage.get_latest_assessment("analysis_123")
        assert latest is not None
        # Latest should be the one saved last
        assert latest.assessed_by == "Dr. Jones"

    def test_get_draft_assessment(self, temp_storage):
        """Test getting draft assessment."""
        draft = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessed_by="Dr. Smith",
            is_draft=True,
        )
        completed = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/test2.mp4",
            assessed_by="Dr. Jones",
            is_draft=False,
        )

        temp_storage.save(draft)
        temp_storage.save(completed)

        draft_assessment = temp_storage.get_draft_assessment("analysis_123")
        assert draft_assessment is not None
        assert draft_assessment.is_draft

    def test_delete_assessment(self, temp_storage):
        """Test deleting an assessment."""
        report = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/test.mp4",
            assessed_by="Dr. Smith",
        )
        assessment_id = report.id
        temp_storage.save(report)

        # Verify it exists
        assert temp_storage.load(assessment_id) is not None

        # Delete
        deleted = temp_storage.delete(assessment_id)
        assert deleted

        # Verify it's gone
        assert temp_storage.load(assessment_id) is None

    def test_delete_nonexistent_assessment(self, temp_storage):
        """Test deleting non-existent assessment."""
        deleted = temp_storage.delete("nonexistent")
        assert not deleted

    def test_search_by_video(self, temp_storage):
        """Test searching assessments by video path."""
        report1 = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/presentation.mp4",
            assessed_by="Dr. Smith",
        )
        report2 = AssessmentReport(
            analysis_id="analysis_456",
            video_file_path="/videos/presentation.mp4",
            assessed_by="Dr. Jones",
        )
        report3 = AssessmentReport(
            analysis_id="analysis_789",
            video_file_path="/videos/other.mp4",
            assessed_by="Dr. Brown",
        )

        temp_storage.save(report1)
        temp_storage.save(report2)
        temp_storage.save(report3)

        results = temp_storage.search_by_video("/videos/presentation.mp4")
        assert len(results) == 2

    def test_get_assessments_by_assessor(self, temp_storage):
        """Test getting all assessments by an assessor."""
        report1 = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/test1.mp4",
            assessed_by="Dr. Smith",
            is_draft=False,
        )
        report2 = AssessmentReport(
            analysis_id="analysis_456",
            video_file_path="/videos/test2.mp4",
            assessed_by="Dr. Smith",
            is_draft=False,
        )
        report3 = AssessmentReport(
            analysis_id="analysis_789",
            video_file_path="/videos/test3.mp4",
            assessed_by="Dr. Jones",
            is_draft=False,
        )

        temp_storage.save(report1)
        temp_storage.save(report2)
        temp_storage.save(report3)

        smith_assessments = temp_storage.get_assessments_by_assessor("Dr. Smith")
        assert len(smith_assessments) == 2

    def test_export_statistics(self, temp_storage):
        """Test exporting statistics."""
        report1 = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/test1.mp4",
            assessed_by="Dr. Smith",
            is_draft=False,
        )
        report2 = AssessmentReport(
            analysis_id="analysis_123",
            video_file_path="/videos/test2.mp4",
            assessed_by="Dr. Jones",
            is_draft=False,
        )

        temp_storage.save(report1)
        temp_storage.save(report2)

        stats = temp_storage.export_statistics("analysis_123")
        assert stats["total_assessments"] == 2
        assert stats["unique_assessors"] == 2

    def test_export_statistics_empty(self, temp_storage):
        """Test exporting statistics with no assessments."""
        stats = temp_storage.export_statistics("nonexistent")
        assert stats["total_assessments"] == 0
        assert stats["average_score"] == 0.0
