"""Comprehensive tests for report generation and export functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from video_analyser.reports.report_generator import (
    APICostSummary,
    ReportCustomization,
    ReportFormat,
    ReportGenerator,
    SpeechMetrics,
    TranscriptionSegment,
)


class TestReportCustomization:
    """Test report customization functionality."""

    def test_default_customization(self):
        """Test default customization settings."""
        custom = ReportCustomization()
        assert custom.include_video_metadata is True
        assert custom.include_frames is True
        assert custom.include_transcription is True
        assert custom.include_api_costs is True
        assert custom.max_frames_in_summary == 50

    def test_custom_customization(self):
        """Test custom customization settings."""
        custom = ReportCustomization(
            include_ocr=False,
            include_object_detection=False,
            max_frames_in_summary=10,
        )
        assert custom.include_ocr is False
        assert custom.include_object_detection is False
        assert custom.max_frames_in_summary == 10

    def test_customization_excludes_multiple_sections(self):
        """Test customization with multiple exclusions."""
        custom = ReportCustomization(
            include_audio_metadata=False,
            include_api_costs=False,
            include_ocr=False,
        )
        assert custom.include_audio_metadata is False
        assert custom.include_api_costs is False
        assert custom.include_ocr is False
        assert custom.include_transcription is True  # Should still be included


class TestReportGenerator:
    """Test ReportGenerator functionality."""

    def test_report_generator_initialization(self):
        """Test ReportGenerator initialization."""
        generator = ReportGenerator()
        assert generator.config is None
        assert isinstance(generator, ReportGenerator)

    def test_report_generator_with_config(self):
        """Test ReportGenerator initialization with config."""
        config = {"key": "value"}
        generator = ReportGenerator(config=config)
        assert generator.config == config

    def test_save_json_creates_directory(self):
        """Test that save_json creates output directory."""
        generator = ReportGenerator()
        test_report = {"test": "data"}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dirs" / "report.json"
            generator.save_json(test_report, output_path)

            assert output_path.exists()
            assert output_path.parent.exists()

    def test_save_json_file_content(self):
        """Test that save_json writes correct content."""
        generator = ReportGenerator()
        test_report = {"video": {"file": "test.mp4"}, "frames": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generator.save_json(test_report, output_path)

            with open(output_path) as f:
                loaded = json.load(f)

            assert loaded["video"]["file"] == "test.mp4"
            assert loaded["frames"] == []


class TestReportExport:
    """Test report export functionality."""

    @pytest.fixture
    def sample_report(self):
        """Create a sample report dictionary."""
        return {
            "video": {
                "file_path": "/path/to/video.mp4",
                "duration": 60.0,
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "format": "mp4",
            },
            "frames": [
                {
                    "frame_number": 1,
                    "timestamp": 10.0,
                    "scene_number": 1,
                    "file_path": "/frames/frame_1.jpg",
                    "width": 1920,
                    "height": 1080,
                    "caption": "A person speaking",
                    "caption_confidence": 0.95,
                    "quality_score": 0.88,
                    "ocr_text": "Slide Title",
                    "ocr_confidence": 0.92,
                    "num_text_regions": 2,
                    "detected_objects": ["person", "podium"],
                    "num_objects_detected": 2,
                },
                {
                    "frame_number": 2,
                    "timestamp": 30.0,
                    "scene_number": 2,
                    "file_path": "/frames/frame_2.jpg",
                    "width": 1920,
                    "height": 1080,
                    "caption": "A presentation slide",
                    "caption_confidence": 0.93,
                    "quality_score": 0.86,
                    "ocr_text": "Data visualization",
                    "ocr_confidence": 0.88,
                    "num_text_regions": 3,
                    "detected_objects": ["chart"],
                    "num_objects_detected": 1,
                },
            ],
            "full_transcription_text": "Welcome to this presentation. Today we will discuss results.",
            "speech_metrics": {
                "total_words": 10,
                "total_speech_duration": 10.0,
                "speaking_rate_wpm": 60.0,
                "average_confidence": 0.97,
            },
            "api_cost_summary": {
                "total_frames_processed": 2,
                "total_tokens_used": 95,
                "total_cost_usd": 0.0019,
                "provider": "claude",
                "model": "vision",
            },
            "has_transcription": True,
            "has_visual_analysis": True,
            "has_captions": True,
            "has_ocr": True,
            "has_object_detection": True,
        }

    def test_export_to_json(self, sample_report):
        """Test exporting report to JSON format."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generator.export_report(sample_report, output_path, ReportFormat.JSON)

            assert output_path.exists()
            with open(output_path) as f:
                loaded = json.load(f)
            assert "video" in loaded
            assert "frames" in loaded
            assert loaded["video"]["file_path"] == "/path/to/video.mp4"

    def test_export_to_csv(self, sample_report):
        """Test exporting report to CSV format."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.csv"
            generator.export_report(sample_report, output_path, ReportFormat.CSV)

            assert output_path.exists()
            with open(output_path) as f:
                content = f.read()
            assert "frame_number" in content
            assert "timestamp" in content
            assert "caption" in content
            assert "1" in content  # Frame 1

    def test_export_to_plain_text(self, sample_report):
        """Test exporting report to plain text format."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.txt"
            generator.export_report(sample_report, output_path, ReportFormat.PLAIN_TEXT)

            assert output_path.exists()
            with open(output_path) as f:
                content = f.read()
            assert "VIDEO ANALYSIS REPORT" in content
            assert "VIDEO INFORMATION" in content
            assert "/path/to/video.mp4" in content

    def test_export_invalid_format(self, sample_report):
        """Test exporting with invalid format raises error."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.xyz"
            with pytest.raises(ValueError, match="Unsupported export format"):
                generator.export_report(
                    sample_report,
                    output_path,
                    "invalid",  # type: ignore
                )


class TestReportFilteringAndCustomization:
    """Test report filtering based on customization."""

    @pytest.fixture
    def sample_report(self):
        """Create a sample report dictionary."""
        return {
            "video": {"file_path": "/path/to/video.mp4"},
            "audio": {"duration": 60.0},
            "frames": [
                {
                    "frame_number": 1,
                    "timestamp": 10.0,
                    "scene_number": 1,
                    "file_path": "/frames/frame_1.jpg",
                    "caption": "Test caption",
                    "ocr_text": "Test OCR",
                    "detected_objects": ["obj1", "obj2"],
                },
                {
                    "frame_number": 2,
                    "timestamp": 20.0,
                    "scene_number": 1,
                    "file_path": "/frames/frame_2.jpg",
                    "caption": "Another caption",
                    "ocr_text": "More OCR",
                    "detected_objects": ["obj3"],
                },
            ],
            "full_transcription_text": "Test transcription",
            "speech_metrics": {"total_words": 10},
            "api_cost_summary": {"total_cost_usd": 0.01},
            "has_transcription": True,
            "has_ocr": True,
            "has_object_detection": True,
        }

    def test_filter_removes_audio_metadata(self, sample_report):
        """Test filtering removes audio metadata."""
        generator = ReportGenerator()
        custom = ReportCustomization(include_audio_metadata=False)
        filtered = generator._filter_report(sample_report, custom)

        assert "audio" not in filtered or filtered["audio"] is None

    def test_filter_removes_transcription(self, sample_report):
        """Test filtering removes transcription."""
        generator = ReportGenerator()
        custom = ReportCustomization(include_transcription=False)
        filtered = generator._filter_report(sample_report, custom)

        assert filtered.get("full_transcription_text") == ""
        assert filtered.get("language") is None

    def test_filter_removes_api_costs(self, sample_report):
        """Test filtering removes API costs."""
        generator = ReportGenerator()
        custom = ReportCustomization(include_api_costs=False)
        filtered = generator._filter_report(sample_report, custom)

        assert (
            "api_cost_summary" not in filtered or filtered["api_cost_summary"] is None
        )

    def test_filter_limits_frames(self, sample_report):
        """Test filtering limits frame count."""
        generator = ReportGenerator()
        custom = ReportCustomization(max_frames_in_summary=1)
        filtered = generator._filter_report(sample_report, custom)

        assert len(filtered["frames"]) == 1
        assert filtered["total_frames"] == 1

    def test_filter_removes_frame_files(self, sample_report):
        """Test filtering removes frame file paths."""
        generator = ReportGenerator()
        custom = ReportCustomization(include_frame_files=False)
        filtered = generator._filter_report(sample_report, custom)

        for frame in filtered["frames"]:
            assert frame.get("file_path") is None or frame.get("file_path") == ""

    def test_filter_removes_detected_objects(self, sample_report):
        """Test filtering removes detected objects."""
        generator = ReportGenerator()
        custom = ReportCustomization(include_detected_objects=False)
        filtered = generator._filter_report(sample_report, custom)

        for frame in filtered["frames"]:
            assert "detected_objects" not in frame or frame["detected_objects"] is None

    def test_export_with_customization(self, sample_report):
        """Test exporting with customization options."""
        generator = ReportGenerator()

        custom = ReportCustomization(
            include_api_costs=False,
            include_ocr=False,
            max_frames_in_summary=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generator.export_report(
                sample_report, output_path, ReportFormat.JSON, custom
            )

            with open(output_path) as f:
                exported = json.load(f)

            # API costs should not be in output
            assert (
                "api_cost_summary" not in exported
                or exported["api_cost_summary"] is None
            )
            # Only 1 frame
            assert len(exported["frames"]) == 1


class TestExportToMultipleFormats:
    """Test exporting to multiple formats."""

    @pytest.fixture
    def sample_report(self):
        """Create a sample report dictionary."""
        return {
            "video": {"file_path": "/test/video.mp4"},
            "frames": [{"frame_number": 1, "timestamp": 10.0, "caption": "Test"}],
            "full_transcription_text": "Test text",
        }

    def test_export_to_multiple_formats(self, sample_report):
        """Test exporting report to multiple formats."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            formats = [ReportFormat.JSON, ReportFormat.CSV, ReportFormat.PLAIN_TEXT]

            output_paths = generator.export_to_formats(
                sample_report, output_dir, formats=formats
            )

            assert len(output_paths) == 3
            assert "json" in output_paths
            assert "csv" in output_paths
            assert "txt" in output_paths

            # Verify all files exist
            for file_path in output_paths.values():
                assert file_path.exists()

    def test_export_default_formats(self, sample_report):
        """Test exporting with default format list."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            output_paths = generator.export_to_formats(sample_report, output_dir)

            # Should use default formats: JSON, CSV, TXT
            assert len(output_paths) >= 3
            assert "json" in output_paths
            assert "csv" in output_paths
            assert "txt" in output_paths


class TestCSVExport:
    """Test CSV export functionality."""

    @pytest.fixture
    def sample_report(self):
        """Create a sample report with multiple frames."""
        return {
            "frames": [
                {
                    "frame_number": 1,
                    "timestamp": 10.0,
                    "scene_number": 1,
                    "caption": "Person speaking",
                    "caption_confidence": 0.95,
                    "quality_score": 0.88,
                    "ocr_text": "Title text",
                    "ocr_confidence": 0.92,
                    "num_text_regions": 2,
                    "num_objects_detected": 1,
                    "detected_objects": ["person"],
                },
                {
                    "frame_number": 2,
                    "timestamp": 20.0,
                    "scene_number": 1,
                    "caption": "Slide view",
                    "caption_confidence": 0.93,
                    "quality_score": 0.86,
                    "ocr_text": "Content",
                    "ocr_confidence": 0.88,
                    "num_text_regions": 1,
                    "num_objects_detected": 0,
                    "detected_objects": [],
                },
            ]
        }

    def test_csv_export_includes_headers(self, sample_report):
        """Test CSV export includes correct headers."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.csv"
            generator.export_report(sample_report, output_path, ReportFormat.CSV)

            with open(output_path) as f:
                first_line = f.readline()

            assert "frame_number" in first_line
            assert "timestamp" in first_line
            assert "scene_number" in first_line

    def test_csv_export_includes_frame_data(self, sample_report):
        """Test CSV export includes frame data."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.csv"
            generator.export_report(sample_report, output_path, ReportFormat.CSV)

            with open(output_path) as f:
                content = f.read()

            # Should contain frame numbers and timestamps
            assert "1" in content
            assert "10" in content

    def test_csv_export_respects_customization(self, sample_report):
        """Test CSV export respects customization settings."""
        generator = ReportGenerator()
        custom = ReportCustomization(include_ocr=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.csv"
            generator.export_report(
                sample_report, output_path, ReportFormat.CSV, custom
            )

            with open(output_path) as f:
                first_line = f.readline()

            # OCR columns should not be present when excluded
            assert "ocr_text" not in first_line


class TestTextExport:
    """Test plain text export functionality."""

    @pytest.fixture
    def sample_report(self):
        """Create a sample report."""
        return {
            "video": {
                "file_path": "/path/to/video.mp4",
                "duration": 60.0,
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
            },
            "frames": [
                {
                    "frame_number": 1,
                    "timestamp": 10.0,
                    "caption": "Test caption",
                    "ocr_text": "Test OCR text",
                }
            ],
            "full_transcription_text": "This is a test transcription.",
            "speech_metrics": {
                "total_words": 5,
                "total_speech_duration": 5.0,
                "speaking_rate_wpm": 60.0,
                "average_confidence": 0.95,
            },
            "api_cost_summary": {
                "provider": "claude",
                "model": "vision",
                "total_tokens_used": 50,
                "total_cost_usd": 0.001,
            },
        }

    def test_text_export_includes_sections(self, sample_report):
        """Test text export includes major sections."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.txt"
            generator.export_report(sample_report, output_path, ReportFormat.PLAIN_TEXT)

            with open(output_path) as f:
                content = f.read()

            assert "VIDEO ANALYSIS REPORT" in content
            assert "VIDEO INFORMATION" in content
            assert "SPEECH METRICS" in content

    def test_text_export_includes_metadata(self, sample_report):
        """Test text export includes video metadata."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.txt"
            generator.export_report(sample_report, output_path, ReportFormat.PLAIN_TEXT)

            with open(output_path) as f:
                content = f.read()

            assert "/path/to/video.mp4" in content
            assert "60.00" in content or "60" in content  # Duration
            assert "1920x1080" in content  # Resolution

    def test_text_export_respects_customization(self, sample_report):
        """Test text export respects customization."""
        generator = ReportGenerator()
        custom = ReportCustomization(include_api_costs=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.txt"
            generator.export_report(
                sample_report, output_path, ReportFormat.PLAIN_TEXT, custom
            )

            with open(output_path) as f:
                content = f.read()

            # API costs section should not be present
            assert "API USAGE COSTS" not in content


class TestReportMetadata:
    """Test report metadata models."""

    def test_api_cost_summary_creation(self):
        """Test APICostSummary creation."""
        cost = APICostSummary(
            total_frames_processed=10,
            total_tokens_used=500,
            total_cost_usd=0.01,
            provider="claude",
            model="vision",
        )
        assert cost.total_frames_processed == 10
        assert cost.total_cost_usd == 0.01
        assert cost.provider == "claude"

    def test_speech_metrics_creation(self):
        """Test SpeechMetrics creation."""
        metrics = SpeechMetrics(
            total_words=100,
            total_speech_duration=60.0,
            speaking_rate_wpm=100.0,
            average_confidence=0.95,
        )
        assert metrics.total_words == 100
        assert metrics.speaking_rate_wpm == 100.0

    def test_transcription_segment_creation(self):
        """Test TranscriptionSegment creation."""
        segment = TranscriptionSegment(
            id=1, text="Hello world", start_time=0.0, end_time=2.0, num_words=2
        )
        assert segment.text == "Hello world"
        assert segment.num_words == 2
