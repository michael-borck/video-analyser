"""Tests for visual analyzer with object detection integration."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from video_analyser.analysis.object_detector import (
    DetectedObject,
    ObjectDetectionResult,
    PresentationElement,
)
from video_analyser.analysis.visual_analyzer import ExtractedFrame, FrameExtractor
from video_analyser.core.scene_detector import Scene, SceneDetectionResult
from video_analyser.utils.config import VideoAnalyserConfig, VisualAnalysisConfig


def create_test_quality_metrics():
    """Helper to create test quality metrics with all required fields."""
    from tests.analysis.test_visual_analyzer import (
        create_test_quality_metrics as base_create,
    )

    return base_create()


@pytest.fixture
def mock_config_with_object_detection():
    """Create mock configuration with object detection enabled."""
    config = VideoAnalyserConfig(
        visual_analysis=VisualAnalysisConfig(
            frames_per_scene=2,
            frame_quality=85,
            blur_threshold=100.0,
            contrast_threshold=20.0,
            brightness_min=50.0,
            brightness_max=200.0,
            blur_weight=0.4,
            contrast_weight=0.3,
            brightness_weight=0.3,
            enable_quality_filtering=False,
            save_extracted_frames=False,
            enable_captioning=False,
            enable_ocr=False,
            enable_object_detection=True,  # Enable object detection
            object_detection_confidence=0.5,
        )
    )
    return config


@pytest.fixture
def frame_extractor_with_detection(mock_config_with_object_detection):
    """Create FrameExtractor with object detection enabled."""
    return FrameExtractor(config=mock_config_with_object_detection)


@pytest.fixture
def mock_scene_result():
    """Create mock scene detection result."""
    scenes = [
        Scene(
            scene_number=1,
            start_time=0.0,
            end_time=5.0,
            duration=5.0,
            start_frame=0,
            end_frame=150,
            confidence=0.95,
            transition_type="cut",
        ),
        Scene(
            scene_number=2,
            start_time=5.0,
            end_time=10.0,
            duration=5.0,
            start_frame=150,
            end_frame=300,
            confidence=0.90,
            transition_type="fade",
        ),
    ]

    return SceneDetectionResult(
        scenes=scenes,
        total_scenes=2,
        detection_method="threshold",
        threshold_used=0.3,
        video_duration=10.0,
        average_scene_duration=5.0,
    )


@pytest.fixture
def presentation_video():
    """Create a test video file with presentation-like content."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp.name, fourcc, 30.0, (1280, 720))

        # Generate frames
        for i in range(300):  # 10 seconds at 30fps
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * 240

            # Scene 1: Title slide (0-5 seconds)
            if i < 150:
                cv2.putText(
                    frame,
                    "Presentation Title",
                    (200, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 0, 0),
                    5,
                )
                cv2.putText(
                    frame,
                    "Subtitle Text",
                    (300, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (100, 100, 100),
                    2,
                )
            # Scene 2: Content slide (5-10 seconds)
            else:
                cv2.rectangle(frame, (100, 100), (600, 600), (200, 200, 200), -1)
                cv2.putText(
                    frame,
                    "Chart Title",
                    (150, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                )
                # Draw simple chart
                for j in range(5):
                    height = 100 + j * 50
                    cv2.rectangle(
                        frame,
                        (150 + j * 90, 600 - height),
                        (200 + j * 90, 600),
                        (100, 150, 200),
                        -1,
                    )

            out.write(frame)

        out.release()
        return Path(tmp.name)


class TestObjectDetectionIntegration:
    """Test object detection integration with visual analyzer."""

    def test_extracted_frame_with_object_detection(self):
        """Test ExtractedFrame with object detection results."""
        # Create mock object detection result
        detected_objects = [
            DetectedObject(
                element_type=PresentationElement.SLIDE,
                confidence=0.9,
                bbox=(0, 0, 1280, 720),
                center=(0.5, 0.5),
                area_ratio=1.0,
            ),
            DetectedObject(
                element_type=PresentationElement.TITLE,
                confidence=0.85,
                bbox=(200, 150, 1080, 250),
                center=(0.5, 0.2),
                area_ratio=0.1,
            ),
        ]

        object_detection_result = ObjectDetectionResult(
            detected_objects=detected_objects,
            total_objects=2,
            processing_time=0.1,
            frame_width=1280,
            frame_height=720,
            element_counts={"slide": 1, "title": 1},
            high_confidence_objects=2,
            average_confidence=0.875,
            layout_type="slide",
            has_presenter=False,
            dominant_element="slide",
        )

        # Create ExtractedFrame with object detection
        frame = ExtractedFrame(
            frame_number=150,
            timestamp=5.0,
            scene_number=1,
            width=1280,
            height=720,
            quality_metrics=create_test_quality_metrics(),
            object_detection_result=object_detection_result,
        )

        assert frame.object_detection_result is not None
        assert frame.object_detection_result.total_objects == 2
        assert frame.object_detection_result.layout_type == "slide"

        # Test serialization
        frame_dict = frame.to_dict()
        assert "object_detection_result" in frame_dict
        assert frame_dict["object_detection_result"]["total_objects"] == 2

    def test_frame_extraction_with_object_detection(
        self, frame_extractor_with_detection, presentation_video, mock_scene_result
    ):
        """Test frame extraction with object detection enabled."""
        # Extract frames
        result = frame_extractor_with_detection.extract_frames_from_scenes(
            video_path=presentation_video,
            scene_result=mock_scene_result,
        )

        # Verify extraction
        assert result.total_scenes == 2
        assert result.total_frames_extracted > 0

        # Check that object detection was performed
        for scene_analysis in result.scene_analyses:
            for frame in scene_analysis.frames:
                assert frame.object_detection_result is not None
                assert isinstance(frame.object_detection_result, ObjectDetectionResult)

                # Should detect slide layout
                assert frame.object_detection_result.layout_type in [
                    "slide",
                    "document",
                    "unknown",
                ]

    def test_detect_objects_in_frame_method(self, frame_extractor_with_detection):
        """Test the _detect_objects_in_frame method directly."""
        # Create test frame
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255
        cv2.rectangle(frame, (100, 100), (600, 200), (0, 0, 0), -1)  # Title area
        cv2.rectangle(
            frame, (100, 300), (1180, 600), (200, 200, 200), -1
        )  # Content area

        # Detect objects
        result = frame_extractor_with_detection._detect_objects_in_frame(frame)

        assert result is not None
        assert isinstance(result, ObjectDetectionResult)
        assert result.frame_width == 1280
        assert result.frame_height == 720
        assert result.total_objects >= 0

    def test_object_detection_disabled(self, mock_config_with_object_detection):
        """Test behavior when object detection is disabled."""
        # Disable object detection
        mock_config_with_object_detection.visual_analysis.enable_object_detection = (
            False
        )
        extractor = FrameExtractor(config=mock_config_with_object_detection)

        # Create test frame
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255

        # Try to detect objects
        result = extractor._detect_objects_in_frame(frame)

        assert result is None

    def test_object_detection_error_handling(self, frame_extractor_with_detection):
        """Test error handling in object detection."""
        # Create a mock object detector that raises exception on detect_objects
        mock_detector = MagicMock()
        mock_detector.detect_objects.side_effect = Exception("Detection failed")

        # Replace the object detector
        frame_extractor_with_detection.object_detector = mock_detector

        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255
        result = frame_extractor_with_detection._detect_objects_in_frame(frame)

        # Should return empty result instead of raising
        assert result is not None
        assert result.total_objects == 0
        assert result.layout_type == "unknown"

    def test_cleanup_with_object_detector(self, frame_extractor_with_detection):
        """Test cleanup includes object detector."""
        # Initialize object detector
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 255
        _ = frame_extractor_with_detection._detect_objects_in_frame(frame)

        assert frame_extractor_with_detection.object_detector is not None

        # Cleanup
        frame_extractor_with_detection.cleanup()

        assert frame_extractor_with_detection.object_detector is None

    def test_scene_analysis_with_detected_elements(
        self, frame_extractor_with_detection, presentation_video, mock_scene_result
    ):
        """Test scene analysis includes detected presentation elements."""
        # Extract frames
        result = frame_extractor_with_detection.extract_frames_from_scenes(
            video_path=presentation_video,
            scene_result=mock_scene_result,
        )

        # Analyze first scene
        scene_1 = result.scene_analyses[0]

        # Check if any frames detected presentation elements
        has_presentation_elements = False
        for frame in scene_1.frames:
            if frame.object_detection_result:
                summary = frame.object_detection_result.get_content_summary()
                if summary["total_elements"] > 0:
                    has_presentation_elements = True
                    break

        assert has_presentation_elements, (
            "Should detect at least some presentation elements"
        )

    def test_visual_analysis_result_with_object_detection(
        self, frame_extractor_with_detection, presentation_video, mock_scene_result
    ):
        """Test complete visual analysis with object detection."""
        # Extract frames
        result = frame_extractor_with_detection.extract_frames_from_scenes(
            video_path=presentation_video,
            scene_result=mock_scene_result,
        )

        # Convert to dict for serialization
        result_dict = result.to_dict()

        # Verify object detection results are included
        for scene_dict in result_dict["scene_analyses"]:
            for frame_dict in scene_dict["frames"]:
                assert "object_detection_result" in frame_dict

                if frame_dict["object_detection_result"]:
                    obj_det = frame_dict["object_detection_result"]
                    assert "detected_objects" in obj_det
                    assert "layout_type" in obj_det
                    assert "element_counts" in obj_det

    @pytest.mark.parametrize("enable_all", [True, False])
    def test_combined_analysis_modes(
        self, mock_config_with_object_detection, enable_all
    ):
        """Test combining object detection with other analysis modes."""
        # Configure all analysis modes
        config = mock_config_with_object_detection
        config.visual_analysis.enable_captioning = enable_all
        config.visual_analysis.enable_ocr = enable_all
        config.visual_analysis.enable_object_detection = True

        extractor = FrameExtractor(config=config)

        # Create test frame
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 240
        cv2.putText(
            frame, "Test Text", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3
        )

        # Test object detection
        obj_result = extractor._detect_objects_in_frame(frame)
        assert obj_result is not None

        # Test other modes based on configuration
        if enable_all:
            # With mocked dependencies, these would normally work
            # For now, just verify the methods exist
            assert hasattr(extractor, "_caption_frame")
            assert hasattr(extractor, "_extract_text_from_frame")
