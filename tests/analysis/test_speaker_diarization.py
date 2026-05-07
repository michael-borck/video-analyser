"""Tests for speaker diarization module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from video_analyser.analysis.speaker_diarization import (
    DiarizationResult,
    SpeakerDiarizer,
    SpeakerProfile,
    SpeakerSegment,
)


class TestSpeakerSegment:
    """Tests for SpeakerSegment model."""

    def test_create_speaker_segment(self):
        """Test creating a speaker segment."""
        segment = SpeakerSegment(
            speaker_id="speaker_1",
            start_time=0.0,
            end_time=5.0,
            confidence=0.95,
            duration=5.0,
        )
        assert segment.speaker_id == "speaker_1"
        assert segment.duration == 5.0
        assert segment.confidence == 0.95

    def test_segment_duration_calculation(self):
        """Test that duration is calculated correctly."""
        segment = SpeakerSegment(
            speaker_id="speaker_1",
            start_time=10.0,
            end_time=15.5,
            confidence=0.9,
            duration=5.5,
        )
        assert segment.duration == 5.5

    def test_segment_optional_label(self):
        """Test segment with optional label."""
        segment = SpeakerSegment(
            speaker_id="speaker_1",
            start_time=0.0,
            end_time=5.0,
            confidence=0.9,
            duration=5.0,
            label="Alice",
        )
        assert segment.label == "Alice"

    def test_confidence_bounds(self):
        """Test confidence score bounds."""
        # Valid confidence
        segment = SpeakerSegment(
            speaker_id="speaker_1",
            start_time=0.0,
            end_time=5.0,
            confidence=0.5,
            duration=5.0,
        )
        assert segment.confidence == 0.5

        # Invalid confidence - too high
        with pytest.raises(ValueError):
            SpeakerSegment(
                speaker_id="speaker_1",
                start_time=0.0,
                end_time=5.0,
                confidence=1.5,
                duration=5.0,
            )


class TestSpeakerProfile:
    """Tests for SpeakerProfile model."""

    def test_create_speaker_profile(self):
        """Test creating a speaker profile."""
        profile = SpeakerProfile(
            speaker_id="speaker_1",
            num_segments=5,
            total_speaking_time=120.0,
            percentage_of_total=50.0,
            first_appearance=0.0,
            last_appearance=300.0,
            avg_segment_duration=24.0,
            avg_confidence=0.92,
        )
        assert profile.speaker_id == "speaker_1"
        assert profile.num_segments == 5
        assert profile.total_speaking_time == 120.0
        assert profile.percentage_of_total == 50.0

    def test_profile_with_label(self):
        """Test profile with human label."""
        profile = SpeakerProfile(
            speaker_id="speaker_1",
            label="Alice",
            num_segments=3,
            total_speaking_time=60.0,
            percentage_of_total=25.0,
        )
        assert profile.label == "Alice"

    def test_profile_percentage_bounds(self):
        """Test percentage bounds."""
        # Valid percentage
        profile = SpeakerProfile(
            speaker_id="speaker_1",
            percentage_of_total=75.0,
        )
        assert profile.percentage_of_total == 75.0

        # Invalid percentage - too high
        with pytest.raises(ValueError):
            SpeakerProfile(
                speaker_id="speaker_1",
                percentage_of_total=150.0,
            )


class TestDiarizationResult:
    """Tests for DiarizationResult model."""

    def test_create_diarization_result(self):
        """Test creating a diarization result."""
        segment1 = SpeakerSegment(
            speaker_id="speaker_1",
            start_time=0.0,
            end_time=10.0,
            confidence=0.95,
            duration=10.0,
        )
        profile1 = SpeakerProfile(
            speaker_id="speaker_1",
            num_segments=1,
            total_speaking_time=10.0,
            percentage_of_total=100.0,
        )

        result = DiarizationResult(
            audio_path=Path("test.wav"),
            num_speakers=1,
            segments=[segment1],
            speakers=[profile1],
            total_duration=10.0,
            processing_time=5.0,
        )

        assert result.num_speakers == 1
        assert len(result.segments) == 1
        assert result.total_duration == 10.0

    def test_result_with_overlapping_segments(self):
        """Test result with overlapping speech."""
        result = DiarizationResult(
            audio_path=Path("test.wav"),
            num_speakers=2,
            segments=[],
            speakers=[],
            total_duration=100.0,
            overlapping_speech_segments=[(10.0, 15.0), (50.0, 52.0)],
            processing_time=5.0,
        )

        assert len(result.overlapping_speech_segments) == 2
        assert (10.0, 15.0) in result.overlapping_speech_segments


class TestSpeakerDiarizer:
    """Tests for SpeakerDiarizer class."""

    def test_diarizer_initialization(self):
        """Test diarizer initialization."""
        with patch.dict(
            "sys.modules", {"pyannote": MagicMock(), "pyannote.audio": MagicMock()}
        ):
            diarizer = SpeakerDiarizer(use_gpu=False)
            assert diarizer.use_gpu is False

    def test_missing_pyannote_error(self):
        """Test error when pyannote not installed."""
        with (
            patch.dict("sys.modules", {"pyannote.audio": None}),
            pytest.raises(ImportError),
        ):
            SpeakerDiarizer()

    def test_get_speaker_at_time(self):
        """Test finding speaker at specific time."""
        segment1 = SpeakerSegment(
            speaker_id="speaker_1",
            start_time=0.0,
            end_time=10.0,
            confidence=0.95,
            duration=10.0,
        )
        segment2 = SpeakerSegment(
            speaker_id="speaker_2",
            start_time=10.0,
            end_time=20.0,
            confidence=0.92,
            duration=10.0,
        )

        result = DiarizationResult(
            audio_path=Path("test.wav"),
            num_speakers=2,
            segments=[segment1, segment2],
            speakers=[],
            total_duration=20.0,
            processing_time=5.0,
        )

        with patch.dict(
            "sys.modules", {"pyannote": MagicMock(), "pyannote.audio": MagicMock()}
        ):
            diarizer = SpeakerDiarizer(use_gpu=False)

            # Speaker 1 at 5 seconds
            speaker = diarizer.get_speaker_at_time(result, 5.0)
            assert speaker == "speaker_1"

            # Speaker 2 at 15 seconds
            speaker = diarizer.get_speaker_at_time(result, 15.0)
            assert speaker == "speaker_2"

            # No speaker at 20+ seconds
            speaker = diarizer.get_speaker_at_time(result, 20.0)
            assert speaker is None

    def test_get_speaker_profile(self):
        """Test retrieving speaker profile."""
        profile1 = SpeakerProfile(
            speaker_id="speaker_1",
            num_segments=3,
            total_speaking_time=60.0,
            percentage_of_total=50.0,
        )

        result = DiarizationResult(
            audio_path=Path("test.wav"),
            num_speakers=1,
            segments=[],
            speakers=[profile1],
            total_duration=120.0,
            processing_time=5.0,
        )

        with patch.dict(
            "sys.modules", {"pyannote": MagicMock(), "pyannote.audio": MagicMock()}
        ):
            diarizer = SpeakerDiarizer(use_gpu=False)

            profile = diarizer.get_speaker_profile(result, "speaker_1")
            assert profile is not None
            assert profile.speaker_id == "speaker_1"

            # Non-existent speaker
            profile = diarizer.get_speaker_profile(result, "speaker_999")
            assert profile is None

    def test_relabel_speaker(self):
        """Test relabeling a speaker."""
        profile1 = SpeakerProfile(
            speaker_id="speaker_1",
            num_segments=3,
            total_speaking_time=60.0,
            percentage_of_total=50.0,
        )

        result = DiarizationResult(
            audio_path=Path("test.wav"),
            num_speakers=1,
            segments=[],
            speakers=[profile1],
            total_duration=120.0,
            processing_time=5.0,
        )

        with patch.dict(
            "sys.modules", {"pyannote": MagicMock(), "pyannote.audio": MagicMock()}
        ):
            diarizer = SpeakerDiarizer(use_gpu=False)

            diarizer.relabel_speaker(result, "speaker_1", "Alice")
            assert result.speakers[0].label == "Alice"

    def test_merge_speakers(self):
        """Test merging multiple speakers."""
        profile1 = SpeakerProfile(
            speaker_id="speaker_1",
            num_segments=2,
            total_speaking_time=40.0,
            percentage_of_total=33.0,
            first_appearance=0.0,
            last_appearance=50.0,
        )
        profile2 = SpeakerProfile(
            speaker_id="speaker_2",
            num_segments=2,
            total_speaking_time=40.0,
            percentage_of_total=33.0,
            first_appearance=10.0,
            last_appearance=60.0,
        )

        segment1 = SpeakerSegment(
            speaker_id="speaker_1",
            start_time=0.0,
            end_time=10.0,
            confidence=0.95,
            duration=10.0,
        )
        segment2 = SpeakerSegment(
            speaker_id="speaker_2",
            start_time=10.0,
            end_time=20.0,
            confidence=0.92,
            duration=10.0,
        )

        result = DiarizationResult(
            audio_path=Path("test.wav"),
            num_speakers=2,
            segments=[segment1, segment2],
            speakers=[profile1, profile2],
            total_duration=120.0,
            processing_time=5.0,
        )

        with patch.dict(
            "sys.modules", {"pyannote": MagicMock(), "pyannote.audio": MagicMock()}
        ):
            diarizer = SpeakerDiarizer(use_gpu=False)

            diarizer.merge_speakers(
                result, ["speaker_1", "speaker_2"], "merged_speaker"
            )

            # Check segments were updated
            assert all(s.speaker_id == "merged_speaker" for s in result.segments)

            # Check profiles
            assert len(result.speakers) == 1
            assert result.speakers[0].speaker_id == "merged_speaker"
            assert result.speakers[0].num_segments == 4  # 2 + 2
            assert result.num_speakers == 1
