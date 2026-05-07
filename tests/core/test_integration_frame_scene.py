"""Integration tests for frame extraction with scene detection."""

from unittest.mock import MagicMock, patch

import pytest

from video_analyser.core.scene_detector import Scene, SceneDetector
from video_analyser.core.video_processor import VideoInfo, VideoProcessor


@pytest.fixture
def video_processor():
    """Create VideoProcessor instance for testing."""
    return VideoProcessor()


@pytest.fixture
def scene_detector():
    """Create SceneDetector instance for testing."""
    return SceneDetector()


@pytest.fixture
def mock_video_info(tmp_path):
    """Create mock VideoInfo for testing."""
    test_video = tmp_path / "test_video.mp4"
    test_video.write_text("mock video content")

    return VideoInfo(
        file_path=test_video,
        duration=120.0,
        width=1920,
        height=1080,
        fps=30.0,
        format="mp4",
        size_mb=50.0,
        codec="h264",
    )


@pytest.fixture
def mock_scenes():
    """Create mock scenes for testing."""
    return [
        Scene(
            start_time=0.0,
            end_time=30.0,
            duration=30.0,
            scene_number=1,
            confidence=0.8,
        ),
        Scene(
            start_time=30.0,
            end_time=60.0,
            duration=30.0,
            scene_number=2,
            confidence=0.7,
        ),
        Scene(
            start_time=60.0,
            end_time=120.0,
            duration=60.0,
            scene_number=3,
            confidence=0.9,
        ),
    ]


class TestFrameSceneIntegration:
    """Test integration between frame extraction and scene detection."""

    @patch("ffmpeg.run")
    def test_extract_frames_from_detected_scenes(
        self, mock_run, video_processor, mock_video_info, mock_scenes, tmp_path
    ):
        """Test extracting frames from scenes detected by scene detector."""
        # Mock ffmpeg.run to succeed
        mock_run.return_value = None

        # Create mock output files
        output_dir = tmp_path / "frames"
        output_dir.mkdir(parents=True)

        expected_files = [
            "scene_001_frame_15.00s.jpg",
            "scene_002_frame_45.00s.jpg",
            "scene_003_frame_90.00s.jpg",
        ]

        for filename in expected_files:
            (output_dir / filename).write_bytes(b"mock jpeg content")

        # Convert scenes to the format expected by extract_frames_from_scenes
        scene_tuples = [
            (scene.start_time, scene.end_time, scene.scene_number)
            for scene in mock_scenes
        ]

        # Extract frames
        frame_infos = video_processor.extract_frames_from_scenes(
            mock_video_info, scene_tuples, output_dir
        )

        # Verify results
        assert len(frame_infos) == 3

        # Check frame timestamps (middle of each scene)
        expected_timestamps = [15.0, 45.0, 90.0]
        actual_timestamps = [frame.timestamp for frame in frame_infos]
        assert actual_timestamps == expected_timestamps

        # Check scene numbers
        expected_scene_numbers = [1, 2, 3]
        actual_scene_numbers = [frame.scene_number for frame in frame_infos]
        assert actual_scene_numbers == expected_scene_numbers

    def test_scene_to_frame_extraction_conversion(self, mock_scenes):
        """Test converting Scene objects to frame extraction format."""
        # This demonstrates how to convert Scene objects to the tuple format
        # expected by extract_frames_from_scenes

        scene_tuples = [
            (scene.start_time, scene.end_time, scene.scene_number)
            for scene in mock_scenes
        ]

        expected_tuples = [
            (0.0, 30.0, 1),
            (30.0, 60.0, 2),
            (60.0, 120.0, 3),
        ]

        assert scene_tuples == expected_tuples

    @patch("ffmpeg.run")
    def test_frame_extraction_with_progress_tracking(
        self, mock_run, video_processor, mock_video_info, mock_scenes, tmp_path
    ):
        """Test frame extraction with progress tracking for UI updates."""
        # Mock ffmpeg.run to succeed
        mock_run.return_value = None

        # Create mock output files
        output_dir = tmp_path / "frames"
        output_dir.mkdir(parents=True)

        for i, scene in enumerate(mock_scenes, 1):
            filename = (
                f"scene_{i:03d}_frame_{scene.start_time + scene.duration / 2:.2f}s.jpg"
            )
            (output_dir / filename).write_bytes(b"mock jpeg content")

        # Mock progress callback
        progress_callback = MagicMock()

        # Convert scenes and extract frames
        scene_tuples = [
            (scene.start_time, scene.end_time, scene.scene_number)
            for scene in mock_scenes
        ]

        video_processor.extract_frames_from_scenes(
            mock_video_info, scene_tuples, output_dir, progress_callback
        )

        # Verify progress was tracked
        assert progress_callback.call_count == 3  # One call per scene

        # Verify progress values (should be 1/3, 2/3, 3/3)
        expected_progress = [1 / 3, 2 / 3, 3 / 3]
        actual_progress = [call[0][0] for call in progress_callback.call_args_list]

        for expected, actual in zip(expected_progress, actual_progress, strict=True):
            assert abs(expected - actual) < 0.01  # Allow for floating point precision

    def test_frame_naming_convention(self, mock_scenes):
        """Test that frame naming follows expected convention."""
        # Test the naming convention used in frame extraction

        for scene in mock_scenes:
            timestamp = scene.start_time + (scene.end_time - scene.start_time) / 2
            expected_filename = (
                f"scene_{scene.scene_number:03d}_frame_{timestamp:.2f}s.jpg"
            )

            # Verify naming pattern
            assert expected_filename.startswith(f"scene_{scene.scene_number:03d}_")
            assert expected_filename.endswith(".jpg")
            assert f"{timestamp:.2f}s" in expected_filename

    @patch("ffmpeg.run")
    def test_frame_extraction_error_handling(
        self, mock_run, video_processor, mock_video_info, tmp_path
    ):
        """Test error handling during frame extraction from scenes."""
        # Mock ffmpeg to fail for one scene
        call_count = 0

        def ffmpeg_side_effect(*args, **kwargs):  # noqa: ARG001
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on second scene
                raise RuntimeError("Frame extraction failed")
            return None

        mock_run.side_effect = ffmpeg_side_effect

        # Create mock output files for successful extractions only
        output_dir = tmp_path / "frames"
        output_dir.mkdir(parents=True)

        # Only create files for scenes 1 and 3 (scene 2 will fail)
        (output_dir / "scene_001_frame_15.00s.jpg").write_bytes(b"mock content")
        (output_dir / "scene_003_frame_90.00s.jpg").write_bytes(b"mock content")

        scenes = [
            (0.0, 30.0, 1),
            (30.0, 60.0, 2),  # This will fail
            (60.0, 120.0, 3),
        ]

        # Should continue processing other scenes despite one failure
        frame_infos = video_processor.extract_frames_from_scenes(
            mock_video_info, scenes, output_dir
        )

        # Should get 2 successful extractions (scenes 1 and 3)
        assert len(frame_infos) == 2
        assert frame_infos[0].scene_number == 1
        assert frame_infos[1].scene_number == 3
