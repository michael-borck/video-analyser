"""Tests for visual analysis functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from video_analyser.analysis.visual_analyzer import (
    ExtractedFrame,
    FrameExtractor,
    FrameQualityMetrics,
    SceneFrameAnalysis,
    VisualAnalysisResult,
    create_frame_extractor,
)
from video_analyser.core.exceptions import ErrorCode, VideoProcessingError
from video_analyser.core.scene_detector import Scene, SceneDetectionResult
from video_analyser.utils.config import VideoAnalyserConfig, VisualAnalysisConfig


def create_test_quality_metrics(
    blur_score=150.5,
    blur_category="good",
    contrast_score=45.2,
    contrast_category="excellent",
    brightness_score=125.0,
    brightness_category="good",
    overall_quality_score=0.75,
    overall_quality_category="good",
):
    """Helper to create FrameQualityMetrics with all required fields."""
    return FrameQualityMetrics(
        blur_score=blur_score,
        blur_category=blur_category,
        contrast_score=contrast_score,
        contrast_category=contrast_category,
        brightness_score=brightness_score,
        brightness_category=brightness_category,
        overall_quality_score=overall_quality_score,
        overall_quality_category=overall_quality_category,
        histogram_metrics={"dynamic_range": 200, "entropy": 7.5},
        sharpness_details={"sobel_mean": 25.0, "edge_density": 0.15},
        color_metrics={
            "saturation": {"mean": 100.0, "std": 30.0, "min": 0.0, "max": 255.0},
            "hue_distribution": {"mean": 90.0, "std": 45.0, "dominant_hue": 120},
            "color_balance": {
                "red_mean": 120.0,
                "green_mean": 130.0,
                "blue_mean": 110.0,
                "color_cast": "neutral",
            },
            "color_diversity": 10.0,
        },
        noise_metrics={
            "noise_std_laplacian": 5.0,
            "noise_estimate_gaussian": 3.0,
            "signal_to_noise_ratio": 50.0,
            "noise_level": "low",
        },
        composition_metrics={
            "rule_of_thirds": {
                "edge_strength_mean": 15.0,
                "edge_strength_std": 5.0,
                "alignment_score": 0.7,
            },
            "symmetry": {"vertical": 0.8, "horizontal": 0.7, "overall": 0.75},
            "focus_point": {"x": 0.5, "y": 0.5, "distance_from_center": 0.0},
            "balance": {
                "horizontal": 0.9,
                "vertical": 0.85,
                "diagonal": 0.8,
                "overall": 0.85,
            },
        },
        quality_report={
            "overall_score": overall_quality_score,
            "overall_category": overall_quality_category,
            "strengths": ["Excellent contrast", "Good sharpness"],
            "issues": [],
            "recommendations": [],
            "technical_details": {
                "blur": {"score": blur_score, "category": blur_category},
                "contrast": {"score": contrast_score, "category": contrast_category},
                "brightness": {
                    "score": brightness_score,
                    "category": brightness_category,
                },
                "dynamic_range": 200,
                "entropy": 7.5,
                "edge_density": 0.15,
                "saturation_mean": 100.0,
                "noise_snr": 50.0,
                "symmetry_score": 0.75,
                "balance_score": 0.85,
            },
        },
    )


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config = VideoAnalyserConfig(
        visual_analysis=VisualAnalysisConfig(
            frames_per_scene=3,
            frame_quality=85,
            blur_threshold=100.0,
            contrast_threshold=20.0,
            brightness_min=50.0,
            brightness_max=200.0,
            blur_weight=0.4,
            contrast_weight=0.3,
            brightness_weight=0.3,
            enable_quality_filtering=True,
            min_quality_score=0.3,
            save_extracted_frames=False,
            enable_captioning=False,  # Disable captioning for tests
            enable_ocr=False,  # Disable OCR for tests
            enable_object_detection=False,  # Disable object detection for tests
        )
    )
    return config


@pytest.fixture
def frame_extractor(mock_config):
    """Create FrameExtractor instance for testing."""
    return FrameExtractor(config=mock_config)


@pytest.fixture
def sample_scene_result():
    """Create sample scene detection result for testing."""
    scenes = [
        Scene(
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            scene_number=1,
            confidence=0.8,
        ),
        Scene(
            start_time=10.0,
            end_time=20.0,
            duration=10.0,
            scene_number=2,
            confidence=0.9,
        ),
    ]

    return SceneDetectionResult(
        scenes=scenes,
        total_scenes=2,
        detection_method="threshold",
        threshold_used=0.4,
        video_duration=20.0,
        average_scene_duration=10.0,
    )


@pytest.fixture
def sample_frame():
    """Create sample frame for testing."""
    # Create a simple test frame (100x100 RGB)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Add some pattern to make it interesting
    frame[25:75, 25:75] = [128, 128, 128]  # Gray square
    frame[40:60, 40:60] = [255, 255, 255]  # White square in center

    return frame


@pytest.fixture
def blur_frame():
    """Create a blurry test frame."""
    frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    # Apply heavy blur to simulate poor quality
    frame = cv2.GaussianBlur(frame, (15, 15), 5)
    return frame


@pytest.fixture
def sharp_frame():
    """Create a sharp test frame."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Create sharp edges
    frame[10:90, 10:20] = [255, 255, 255]
    frame[10:90, 30:40] = [0, 0, 0]
    frame[10:90, 50:60] = [255, 255, 255]
    frame[10:90, 70:80] = [0, 0, 0]

    return frame


class TestFrameQualityMetrics:
    """Test FrameQualityMetrics model."""

    def test_frame_quality_metrics_creation(self):
        """Test creating FrameQualityMetrics object."""
        metrics = create_test_quality_metrics()

        assert metrics.blur_score == 150.5
        assert metrics.blur_category == "good"
        assert metrics.contrast_score == 45.2
        assert metrics.contrast_category == "excellent"
        assert metrics.brightness_score == 125.0
        assert metrics.brightness_category == "good"
        assert metrics.overall_quality_score == 0.75
        assert metrics.overall_quality_category == "good"
        assert metrics.histogram_metrics["dynamic_range"] == 200
        assert metrics.sharpness_details["sobel_mean"] == 25.0


class TestExtractedFrame:
    """Test ExtractedFrame model."""

    def test_extracted_frame_creation(self):
        """Test creating ExtractedFrame object."""
        quality_metrics = create_test_quality_metrics()

        frame = ExtractedFrame(
            frame_number=150,
            timestamp=5.0,
            scene_number=1,
            width=1920,
            height=1080,
            quality_metrics=quality_metrics,
        )

        assert frame.frame_number == 150
        assert frame.timestamp == 5.0
        assert frame.scene_number == 1
        assert frame.width == 1920
        assert frame.height == 1080
        assert frame.quality_metrics.overall_quality_score == 0.75

    def test_extracted_frame_to_dict(self):
        """Test ExtractedFrame to_dict conversion."""
        quality_metrics = create_test_quality_metrics()

        frame = ExtractedFrame(
            frame_number=150,
            timestamp=5.0,
            scene_number=1,
            width=1920,
            height=1080,
            quality_metrics=quality_metrics,
        )

        result = frame.to_dict()

        assert isinstance(result, dict)
        assert result["frame_number"] == 150
        assert result["timestamp"] == 5.0
        assert result["scene_number"] == 1
        assert result["width"] == 1920
        assert result["height"] == 1080
        assert result["quality_metrics"]["overall_quality_score"] == 0.75


class TestSceneFrameAnalysis:
    """Test SceneFrameAnalysis model."""

    def test_scene_frame_analysis_creation(self):
        """Test creating SceneFrameAnalysis object."""
        quality_metrics = create_test_quality_metrics()

        frame = ExtractedFrame(
            frame_number=150,
            timestamp=5.0,
            scene_number=1,
            width=1920,
            height=1080,
            quality_metrics=quality_metrics,
        )

        analysis = SceneFrameAnalysis(
            scene_number=1,
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            frames=[frame],
            total_frames_extracted=1,
            best_frame=frame,
            average_quality_score=0.75,
            quality_distribution={"good": 1, "excellent": 0, "fair": 0, "poor": 0},
            total_frames_processed=1,
            frames_filtered_by_quality=0,
            extraction_success_rate=1.0,
        )

        assert analysis.scene_number == 1
        assert analysis.duration == 10.0
        assert len(analysis.frames) == 1
        assert analysis.best_frame == frame
        assert analysis.average_quality_score == 0.75

    def test_get_frames_by_quality(self):
        """Test getting frames by quality category."""
        good_metrics = create_test_quality_metrics()

        excellent_metrics = create_test_quality_metrics(
            blur_score=250.5,
            blur_category="excellent",
            contrast_score=65.2,
            contrast_category="excellent",
            brightness_score=125.0,
            brightness_category="excellent",
            overall_quality_score=0.95,
            overall_quality_category="excellent",
        )

        good_frame = ExtractedFrame(
            frame_number=150,
            timestamp=5.0,
            scene_number=1,
            width=1920,
            height=1080,
            quality_metrics=good_metrics,
        )

        excellent_frame = ExtractedFrame(
            frame_number=300,
            timestamp=10.0,
            scene_number=1,
            width=1920,
            height=1080,
            quality_metrics=excellent_metrics,
        )

        analysis = SceneFrameAnalysis(
            scene_number=1,
            start_time=0.0,
            end_time=15.0,
            duration=15.0,
            frames=[good_frame, excellent_frame],
            total_frames_extracted=2,
            best_frame=excellent_frame,
            average_quality_score=0.85,
            quality_distribution={"good": 1, "excellent": 1, "fair": 0, "poor": 0},
            total_frames_processed=2,
            frames_filtered_by_quality=0,
            extraction_success_rate=1.0,
        )

        good_frames = analysis.get_frames_by_quality("good")
        excellent_frames = analysis.get_frames_by_quality("excellent")
        poor_frames = analysis.get_frames_by_quality("poor")

        assert len(good_frames) == 1
        assert good_frames[0] == good_frame
        assert len(excellent_frames) == 1
        assert excellent_frames[0] == excellent_frame
        assert len(poor_frames) == 0


class TestVisualAnalysisResult:
    """Test VisualAnalysisResult model."""

    def test_visual_analysis_result_creation(self):
        """Test creating VisualAnalysisResult object."""
        quality_metrics = create_test_quality_metrics()

        frame = ExtractedFrame(
            frame_number=150,
            timestamp=5.0,
            scene_number=1,
            width=1920,
            height=1080,
            quality_metrics=quality_metrics,
        )

        scene_analysis = SceneFrameAnalysis(
            scene_number=1,
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            frames=[frame],
            total_frames_extracted=1,
            best_frame=frame,
            average_quality_score=0.75,
            quality_distribution={"good": 1, "excellent": 0, "fair": 0, "poor": 0},
            total_frames_processed=1,
            frames_filtered_by_quality=0,
            extraction_success_rate=1.0,
        )

        result = VisualAnalysisResult(
            total_scenes=1,
            total_frames_extracted=1,
            total_frames_processed=1,
            overall_success_rate=1.0,
            scene_analyses=[scene_analysis],
            overall_quality_distribution={
                "good": 1,
                "excellent": 0,
                "fair": 0,
                "poor": 0,
            },
            average_quality_score=0.75,
            best_frames_per_scene=[frame],
            video_duration=10.0,
            extraction_method="scene_based",
            processing_time=2.5,
        )

        assert result.total_scenes == 1
        assert result.total_frames_extracted == 1
        assert result.overall_success_rate == 1.0
        assert len(result.scene_analyses) == 1
        assert len(result.best_frames_per_scene) == 1

    def test_get_scene_analysis(self):
        """Test getting analysis for specific scene."""
        quality_metrics = create_test_quality_metrics()

        frame = ExtractedFrame(
            frame_number=150,
            timestamp=5.0,
            scene_number=1,
            width=1920,
            height=1080,
            quality_metrics=quality_metrics,
        )

        scene_analysis_1 = SceneFrameAnalysis(
            scene_number=1,
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            frames=[frame],
            total_frames_extracted=1,
            best_frame=frame,
            average_quality_score=0.75,
            quality_distribution={"good": 1, "excellent": 0, "fair": 0, "poor": 0},
            total_frames_processed=1,
            frames_filtered_by_quality=0,
            extraction_success_rate=1.0,
        )

        scene_analysis_2 = SceneFrameAnalysis(
            scene_number=2,
            start_time=10.0,
            end_time=20.0,
            duration=10.0,
            frames=[],
            total_frames_extracted=0,
            best_frame=None,
            average_quality_score=0.0,
            quality_distribution={"good": 0, "excellent": 0, "fair": 0, "poor": 0},
            total_frames_processed=0,
            frames_filtered_by_quality=0,
            extraction_success_rate=0.0,
        )

        result = VisualAnalysisResult(
            total_scenes=2,
            total_frames_extracted=1,
            total_frames_processed=1,
            overall_success_rate=0.5,
            scene_analyses=[scene_analysis_1, scene_analysis_2],
            overall_quality_distribution={
                "good": 1,
                "excellent": 0,
                "fair": 0,
                "poor": 0,
            },
            average_quality_score=0.375,
            best_frames_per_scene=[frame],
            video_duration=20.0,
            extraction_method="scene_based",
            processing_time=5.0,
        )

        scene_1 = result.get_scene_analysis(1)
        scene_2 = result.get_scene_analysis(2)
        scene_3 = result.get_scene_analysis(3)

        assert scene_1 == scene_analysis_1
        assert scene_2 == scene_analysis_2
        assert scene_3 is None

    def test_get_all_frames(self):
        """Test getting all frames across all scenes."""
        quality_metrics = create_test_quality_metrics()

        frame_1 = ExtractedFrame(
            frame_number=150,
            timestamp=5.0,
            scene_number=1,
            width=1920,
            height=1080,
            quality_metrics=quality_metrics,
        )

        frame_2 = ExtractedFrame(
            frame_number=450,
            timestamp=15.0,
            scene_number=2,
            width=1920,
            height=1080,
            quality_metrics=quality_metrics,
        )

        scene_analysis_1 = SceneFrameAnalysis(
            scene_number=1,
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            frames=[frame_1],
            total_frames_extracted=1,
            best_frame=frame_1,
            average_quality_score=0.75,
            quality_distribution={"good": 1, "excellent": 0, "fair": 0, "poor": 0},
            total_frames_processed=1,
            frames_filtered_by_quality=0,
            extraction_success_rate=1.0,
        )

        scene_analysis_2 = SceneFrameAnalysis(
            scene_number=2,
            start_time=10.0,
            end_time=20.0,
            duration=10.0,
            frames=[frame_2],
            total_frames_extracted=1,
            best_frame=frame_2,
            average_quality_score=0.75,
            quality_distribution={"good": 1, "excellent": 0, "fair": 0, "poor": 0},
            total_frames_processed=1,
            frames_filtered_by_quality=0,
            extraction_success_rate=1.0,
        )

        result = VisualAnalysisResult(
            total_scenes=2,
            total_frames_extracted=2,
            total_frames_processed=2,
            overall_success_rate=1.0,
            scene_analyses=[scene_analysis_1, scene_analysis_2],
            overall_quality_distribution={
                "good": 2,
                "excellent": 0,
                "fair": 0,
                "poor": 0,
            },
            average_quality_score=0.75,
            best_frames_per_scene=[frame_1, frame_2],
            video_duration=20.0,
            extraction_method="scene_based",
            processing_time=5.0,
        )

        all_frames = result.get_all_frames()

        assert len(all_frames) == 2
        assert frame_1 in all_frames
        assert frame_2 in all_frames


class TestFrameExtractor:
    """Test FrameExtractor class."""

    def test_initialization_default_config(self):
        """Test FrameExtractor initialization with default config."""
        with patch(
            "video_analyser.analysis.visual_analyzer.get_config"
        ) as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            extractor = FrameExtractor()

            assert extractor.config == mock_config
            mock_get_config.assert_called_once()

    def test_initialization_custom_config(self, mock_config):
        """Test FrameExtractor initialization with custom config."""
        extractor = FrameExtractor(config=mock_config)

        assert extractor.config == mock_config

    @patch("cv2.VideoCapture")
    def test_extract_frames_file_not_found(
        self, _mock_video_capture, frame_extractor, sample_scene_result
    ):
        """Test frame extraction with non-existent file."""
        video_path = Path("/nonexistent/video.mp4")

        with pytest.raises(VideoProcessingError) as exc_info:
            frame_extractor.extract_frames_from_scenes(video_path, sample_scene_result)

        assert exc_info.value.error_code == ErrorCode.FILE_NOT_FOUND
        assert "Video file not found" in str(exc_info.value)

    @patch("cv2.VideoCapture")
    def test_extract_frames_video_open_failure(
        self, mock_video_capture, frame_extractor, sample_scene_result
    ):
        """Test frame extraction with video open failure."""
        # Create a temporary file to simulate existing but unreadable video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            video_path = Path(temp_file.name)

        try:
            # Mock video capture to fail opening
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_video_capture.return_value = mock_cap

            with pytest.raises(VideoProcessingError) as exc_info:
                frame_extractor.extract_frames_from_scenes(
                    video_path, sample_scene_result
                )

            assert exc_info.value.error_code == ErrorCode.FRAME_EXTRACTION_FAILED
            assert "Failed to open video file" in str(exc_info.value)

        finally:
            # Clean up
            if video_path.exists():
                video_path.unlink()

    @patch("cv2.VideoCapture")
    def test_extract_frames_success(
        self, mock_video_capture, frame_extractor, sample_scene_result, sample_frame
    ):
        """Test successful frame extraction."""
        # Create a temporary file to simulate existing video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            video_path = Path(temp_file.name)

        try:
            # Mock video capture for successful operation
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0  # FPS
            mock_cap.read.return_value = (True, sample_frame)
            mock_video_capture.return_value = mock_cap

            result = frame_extractor.extract_frames_from_scenes(
                video_path, sample_scene_result
            )

            assert isinstance(result, VisualAnalysisResult)
            assert result.total_scenes == 2
            assert result.total_frames_extracted > 0
            assert result.extraction_method == "scene_based"
            assert len(result.scene_analyses) == 2

            # Verify video capture was used correctly
            mock_video_capture.assert_called_once_with(str(video_path))
            mock_cap.isOpened.assert_called()
            mock_cap.release.assert_called()

        finally:
            # Clean up
            if video_path.exists():
                video_path.unlink()

    def test_assess_frame_quality(self, frame_extractor, sample_frame):
        """Test frame quality assessment.

        ``sample_frame`` is a deterministic fixture — a black canvas with a
        centred grey square and a smaller centred white square. With the
        mock_config thresholds (blur=100, contrast=20, brightness range
        [50,200]), this fixture lands in known buckets:

        - brightness ~37 (mostly-black canvas) -> "poor" (outside [50,200])
        - contrast ~68 (sharp black/grey/white transitions) -> "excellent"
          (>= 2 * contrast_threshold = 40)
        - blur ~939 (sharp synthetic edges) -> "excellent"
          (>= 2 * blur_threshold = 200)

        The previous version of this test only asserted "category in enum"
        and "score >= 0", which would still pass even if the categorizer
        wholesale reversed its semantics. These bucket assertions pin the
        deterministic output to the fixture's actual physical properties.
        """
        quality_metrics = frame_extractor._assess_frame_quality(sample_frame)

        assert isinstance(quality_metrics, FrameQualityMetrics)
        assert quality_metrics.blur_score >= 0
        assert quality_metrics.blur_category in ["excellent", "good", "fair", "poor"]
        assert quality_metrics.contrast_score >= 0
        assert quality_metrics.contrast_category in [
            "excellent",
            "good",
            "fair",
            "poor",
        ]
        assert 0 <= quality_metrics.brightness_score <= 255
        assert quality_metrics.brightness_category in [
            "excellent",
            "good",
            "fair",
            "poor",
        ]
        assert 0 <= quality_metrics.overall_quality_score <= 1
        assert quality_metrics.overall_quality_category in [
            "excellent",
            "good",
            "fair",
            "poor",
        ]

        # Deterministic-bucket assertions for this specific fixture.
        # The mostly-black canvas has mean brightness well below the
        # configured min (50), so it must be flagged "poor".
        assert quality_metrics.brightness_category == "poor", (
            f"Mostly-black sample_frame should bucket brightness as 'poor'; "
            f"got '{quality_metrics.brightness_category}' "
            f"(score={quality_metrics.brightness_score:.1f})"
        )
        # Sharp synthetic edges produce a Laplacian variance well above
        # 2 * blur_threshold (200), so blur should be "excellent".
        assert quality_metrics.blur_category == "excellent", (
            f"Sharp-edged sample_frame should bucket blur as 'excellent'; "
            f"got '{quality_metrics.blur_category}' "
            f"(score={quality_metrics.blur_score:.1f})"
        )

    def test_assess_frame_quality_blur_detection(
        self, frame_extractor, sharp_frame, blur_frame
    ):
        """Test blur detection in quality assessment."""
        sharp_metrics = frame_extractor._assess_frame_quality(sharp_frame)
        blur_metrics = frame_extractor._assess_frame_quality(blur_frame)

        # Sharp frame should have higher blur score (less blurry)
        assert sharp_metrics.blur_score > blur_metrics.blur_score

    def test_categorize_blur(self, frame_extractor):
        """Test blur score categorization."""
        # Test with different blur scores
        assert (
            frame_extractor._categorize_blur(250.0) == "excellent"
        )  # High score = sharp
        assert frame_extractor._categorize_blur(150.0) == "good"
        assert frame_extractor._categorize_blur(75.0) == "fair"
        assert frame_extractor._categorize_blur(25.0) == "poor"

    def test_categorize_contrast(self, frame_extractor):
        """Test contrast score categorization."""
        # Test with different contrast scores
        assert frame_extractor._categorize_contrast(50.0) == "excellent"
        assert frame_extractor._categorize_contrast(30.0) == "good"
        assert frame_extractor._categorize_contrast(15.0) == "fair"
        assert frame_extractor._categorize_contrast(5.0) == "poor"

    def test_categorize_brightness(self, frame_extractor):
        """Test brightness score categorization."""
        # Test with different brightness scores
        assert (
            frame_extractor._categorize_brightness(125.0) == "excellent"
        )  # Center of range
        assert frame_extractor._categorize_brightness(100.0) == "good"
        assert frame_extractor._categorize_brightness(75.0) == "fair"
        assert frame_extractor._categorize_brightness(25.0) == "poor"  # Outside range

    def test_analyze_histogram(self, frame_extractor, sample_frame):
        """Test histogram analysis."""
        gray = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
        histogram_metrics = frame_extractor._analyze_histogram(gray)

        assert isinstance(histogram_metrics, dict)
        assert "dynamic_range" in histogram_metrics
        assert "entropy" in histogram_metrics
        assert "num_peaks" in histogram_metrics
        assert "mean_intensity" in histogram_metrics
        assert "std_intensity" in histogram_metrics

        assert histogram_metrics["dynamic_range"] >= 0
        assert histogram_metrics["entropy"] >= 0
        assert histogram_metrics["num_peaks"] >= 0

    def test_analyze_sharpness(self, frame_extractor, sample_frame):
        """Test sharpness analysis."""
        gray = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
        sharpness_details = frame_extractor._analyze_sharpness(gray)

        assert isinstance(sharpness_details, dict)
        assert "sobel_mean" in sharpness_details
        assert "edge_density" in sharpness_details
        assert "gradient_variance" in sharpness_details

        assert sharpness_details["sobel_mean"] >= 0
        assert 0 <= sharpness_details["edge_density"] <= 1
        assert sharpness_details["gradient_variance"] >= 0

    def test_calculate_overall_quality(self, frame_extractor):
        """Test overall quality score calculation."""
        # Test with different component scores
        quality_score_1 = frame_extractor._calculate_overall_quality(200.0, 40.0, 125.0)
        quality_score_2 = frame_extractor._calculate_overall_quality(50.0, 10.0, 50.0)

        assert 0 <= quality_score_1 <= 1
        assert 0 <= quality_score_2 <= 1
        assert (
            quality_score_1 > quality_score_2
        )  # Better components should give higher score

    def test_categorize_overall_quality(self, frame_extractor):
        """Test overall quality categorization."""
        assert frame_extractor._categorize_overall_quality(0.9) == "excellent"
        assert frame_extractor._categorize_overall_quality(0.7) == "good"
        assert frame_extractor._categorize_overall_quality(0.5) == "fair"
        assert frame_extractor._categorize_overall_quality(0.2) == "poor"


class TestFrameExtractorFactory:
    """Test FrameExtractor factory function."""

    def test_create_frame_extractor_no_config(self):
        """Test creating FrameExtractor without config."""
        with patch(
            "video_analyser.analysis.visual_analyzer.get_config"
        ) as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            extractor = create_frame_extractor()

            assert isinstance(extractor, FrameExtractor)
            assert extractor.config == mock_config

    def test_create_frame_extractor_with_config(self, mock_config):
        """Test creating FrameExtractor with config."""
        extractor = create_frame_extractor(config=mock_config)

        assert isinstance(extractor, FrameExtractor)
        assert extractor.config == mock_config


class TestFrameExtractionIntegration:
    """Integration tests for frame extraction workflow."""

    @patch("cv2.VideoCapture")
    def test_full_extraction_workflow(
        self, mock_video_capture, frame_extractor, sample_scene_result, sample_frame
    ):
        """Test complete frame extraction workflow."""
        # Create a temporary file to simulate existing video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            video_path = Path(temp_file.name)

        try:
            # Mock video capture for successful operation
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0  # FPS
            mock_cap.read.return_value = (True, sample_frame)
            mock_video_capture.return_value = mock_cap

            # Run extraction
            result = frame_extractor.extract_frames_from_scenes(
                video_path, sample_scene_result
            )

            # Verify overall results
            assert isinstance(result, VisualAnalysisResult)
            assert result.total_scenes == len(sample_scene_result.scenes)
            assert result.video_duration == sample_scene_result.video_duration
            assert result.processing_time > 0

            # Verify scene-level results
            for i, scene_analysis in enumerate(result.scene_analyses):
                assert (
                    scene_analysis.scene_number
                    == sample_scene_result.scenes[i].scene_number
                )
                assert (
                    scene_analysis.start_time
                    == sample_scene_result.scenes[i].start_time
                )
                assert scene_analysis.end_time == sample_scene_result.scenes[i].end_time

                # Each scene should have extracted frames
                assert len(scene_analysis.frames) > 0
                assert scene_analysis.total_frames_extracted > 0

                # Verify frame details
                for frame in scene_analysis.frames:
                    assert isinstance(frame, ExtractedFrame)
                    assert frame.scene_number == scene_analysis.scene_number
                    assert (
                        scene_analysis.start_time
                        <= frame.timestamp
                        <= scene_analysis.end_time
                    )
                    assert isinstance(frame.quality_metrics, FrameQualityMetrics)

        finally:
            # Clean up
            if video_path.exists():
                video_path.unlink()

    @patch("cv2.VideoCapture")
    def test_quality_filtering_workflow(
        self, mock_video_capture, sample_scene_result, sample_frame, blur_frame
    ):
        """Test frame extraction with quality filtering enabled."""
        # Create config with strict quality filtering
        config = VideoAnalyserConfig(
            visual_analysis=VisualAnalysisConfig(
                frames_per_scene=2,
                enable_quality_filtering=True,
                min_quality_score=0.8,  # High threshold
            )
        )
        frame_extractor = FrameExtractor(config=config)

        # Create a temporary file to simulate existing video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            video_path = Path(temp_file.name)

        try:
            # Mock video capture to return alternating good and bad frames
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0  # FPS

            # First call returns good frame, second returns blurry frame
            mock_cap.read.side_effect = [(True, sample_frame), (True, blur_frame)] * 10
            mock_video_capture.return_value = mock_cap

            result = frame_extractor.extract_frames_from_scenes(
                video_path, sample_scene_result
            )

            # With quality filtering, some frames should be filtered out
            total_processed = sum(
                analysis.total_frames_processed for analysis in result.scene_analyses
            )
            total_extracted = sum(
                analysis.total_frames_extracted for analysis in result.scene_analyses
            )
            total_filtered = sum(
                analysis.frames_filtered_by_quality
                for analysis in result.scene_analyses
            )

            assert total_processed > total_extracted  # Some frames were filtered
            assert total_filtered > 0  # Quality filtering occurred

        finally:
            # Clean up
            if video_path.exists():
                video_path.unlink()

    def test_edge_case_single_frame_scene(self, frame_extractor, sample_frame):
        """Test extraction from very short scene."""
        # Create a very short scene
        short_scene = Scene(
            start_time=0.0,
            end_time=0.5,  # 0.5 seconds
            duration=0.5,
            scene_number=1,
            confidence=0.8,
        )

        scene_result = SceneDetectionResult(
            scenes=[short_scene],
            total_scenes=1,
            detection_method="threshold",
            threshold_used=0.4,
            video_duration=0.5,
            average_scene_duration=0.5,
        )

        # Create a temporary file to simulate existing video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            video_path = Path(temp_file.name)

        try:
            with patch("cv2.VideoCapture") as mock_video_capture:
                mock_cap = MagicMock()
                mock_cap.isOpened.return_value = True
                mock_cap.get.return_value = 30.0  # FPS
                mock_cap.read.return_value = (True, sample_frame)
                mock_video_capture.return_value = mock_cap

                result = frame_extractor.extract_frames_from_scenes(
                    video_path, scene_result
                )

                assert len(result.scene_analyses) == 1
                assert result.scene_analyses[0].duration == 0.5
                # Should still extract at least one frame even from short scene
                assert result.scene_analyses[0].total_frames_extracted > 0

        finally:
            # Clean up
            if video_path.exists():
                video_path.unlink()

    @patch("cv2.VideoCapture")
    def test_extract_frames_with_captioning(
        self, mock_video_capture, frame_extractor, sample_scene_result, sample_frame
    ):
        """Test frame extraction with image captioning enabled."""
        # Enable captioning for this test
        frame_extractor.config.visual_analysis.enable_captioning = True

        # Create a temporary file to simulate existing video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            video_path = Path(temp_file.name)

        try:
            # Mock video capture for successful operation
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0  # FPS
            mock_cap.read.return_value = (True, sample_frame)
            mock_video_capture.return_value = mock_cap

            # Mock the captioner
            with patch.object(frame_extractor, "_caption_frame") as mock_caption_frame:
                # image-analyser owns the Caption schema directly.
                from image_analyser.schemas import Caption

                mock_caption_frame.return_value = Caption(
                    text="A test frame with geometric shapes",
                    backend="local",
                    model="Salesforce/blip2-opt-2.7b",
                    tokens_generated=8,
                    cost_estimate_usd=None,
                )

                result = frame_extractor.extract_frames_from_scenes(
                    video_path, sample_scene_result
                )

            # Verify captioning was performed
            assert len(result.scene_analyses) > 0
            for scene_analysis in result.scene_analyses:
                for frame in scene_analysis.frames:
                    assert frame.caption_result is not None
                    assert (
                        frame.caption_result.text
                        == "A test frame with geometric shapes"
                    )
                    assert frame.caption_result.tokens_generated == 8
                    assert frame.caption_result.backend == "local"

        finally:
            # Clean up
            if video_path.exists():
                video_path.unlink()

    @patch("cv2.VideoCapture")
    def test_extract_frames_with_ocr(
        self, mock_video_capture, frame_extractor, sample_scene_result, sample_frame
    ):
        """Test frame extraction with OCR enabled."""
        # Enable OCR for this test
        frame_extractor.config.visual_analysis.enable_ocr = True

        # Create a temporary file to simulate existing video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            video_path = Path(temp_file.name)

        try:
            # Mock video capture for successful operation
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0  # FPS
            mock_cap.read.return_value = (True, sample_frame)
            mock_video_capture.return_value = mock_cap

            # Mock the OCR detector
            with patch.object(
                frame_extractor, "_extract_text_from_frame"
            ) as mock_ocr_frame:
                # image-analyser owns the Ocr / OcrBlock / BBox schemas directly.
                from image_analyser.schemas import BBox, Ocr, OcrBlock

                mock_ocr_frame.return_value = Ocr(
                    text="Sample presentation title",
                    blocks=[
                        OcrBlock(
                            text="Sample presentation title",
                            bbox=BBox(x=50, y=20, w=200, h=30),
                            confidence=0.85,
                        )
                    ],
                    engine="tesseract",
                    average_confidence=0.85,
                )

                result = frame_extractor.extract_frames_from_scenes(
                    video_path, sample_scene_result
                )

            # Verify OCR was performed
            assert len(result.scene_analyses) > 0
            for scene_analysis in result.scene_analyses:
                for frame in scene_analysis.frames:
                    assert frame.ocr_result is not None
                    assert frame.ocr_result.text == "Sample presentation title"
                    assert len(frame.ocr_result.blocks) == 1
                    assert (
                        frame.ocr_result.blocks[0].text == "Sample presentation title"
                    )

        finally:
            # Clean up
            if video_path.exists():
                video_path.unlink()
