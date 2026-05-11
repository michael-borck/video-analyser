"""Tests for visual quality metrics and reporting functionality."""

import numpy as np
import pytest

from video_analyser.analysis.visual_analyzer import (
    ExtractedFrame,
    FrameExtractor,
    FrameQualityMetrics,
    SceneFrameAnalysis,
    VisualAnalysisResult,
)
from video_analyser.utils.config import VideoAnalyserConfig, VisualAnalysisConfig


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
            enable_captioning=False,
            enable_ocr=False,
        )
    )
    return config


@pytest.fixture
def frame_extractor(mock_config):
    """Create FrameExtractor instance for testing."""
    return FrameExtractor(config=mock_config)


def create_quality_metrics_with_issues():
    """Create FrameQualityMetrics with quality issues."""
    return FrameQualityMetrics(
        blur_score=50.0,  # Poor blur
        blur_category="poor",
        contrast_score=10.0,  # Poor contrast
        contrast_category="poor",
        brightness_score=30.0,  # Too dark
        brightness_category="poor",
        overall_quality_score=0.25,
        overall_quality_category="poor",
        histogram_metrics={
            "dynamic_range": 50,
            "entropy": 3.5,
            "num_peaks": 1,
            "peaks": [10],
            "mean_intensity": 30.0,
            "std_intensity": 10.0,
        },
        sharpness_details={
            "sobel_mean": 5.0,
            "edge_density": 0.05,
            "gradient_variance": 10.0,
        },
        color_metrics={
            "saturation": {"mean": 20.0, "std": 5.0, "min": 0.0, "max": 50.0},
            "hue_distribution": {"mean": 45.0, "std": 10.0, "dominant_hue": 40},
            "color_balance": {
                "red_mean": 150.0,
                "green_mean": 80.0,
                "blue_mean": 70.0,
                "color_cast": "red_cast",
            },
            "color_diversity": 30.0,
        },
        noise_metrics={
            "noise_std_laplacian": 25.0,
            "noise_estimate_gaussian": 20.0,
            "signal_to_noise_ratio": 10.0,
            "noise_level": "high",
        },
        composition_metrics={
            "rule_of_thirds": {
                "edge_strength_mean": 5.0,
                "edge_strength_std": 2.0,
                "alignment_score": 0.3,
            },
            "symmetry": {"vertical": 0.4, "horizontal": 0.3, "overall": 0.35},
            "focus_point": {"x": 0.8, "y": 0.9, "distance_from_center": 0.5},
            "balance": {
                "horizontal": 0.6,
                "vertical": 0.5,
                "diagonal": 0.4,
                "overall": 0.5,
            },
        },
        quality_report={
            "overall_score": 0.25,
            "overall_category": "poor",
            "strengths": [],
            "issues": [
                "Image sharpness is poor (score: 50.0)",
                "Image contrast is poor (score: 10.0)",
                "Image is too dark (brightness: 30.0)",
                "Image has a red cast",
                "Image has high noise levels",
                "Main subject appears off-center",
            ],
            "recommendations": [
                "Ensure camera is stable and properly focused",
                "Improve lighting contrast or adjust camera settings",
                "Increase lighting or adjust exposure settings",
                "Adjust white balance settings",
                "Use better lighting or reduce ISO settings",
                "Consider repositioning subject using rule of thirds",
            ],
            "technical_details": {
                "blur": {"score": 50.0, "category": "poor"},
                "contrast": {"score": 10.0, "category": "poor"},
                "brightness": {"score": 30.0, "category": "poor"},
                "dynamic_range": 50,
                "entropy": 3.5,
                "edge_density": 0.05,
                "saturation_mean": 20.0,
                "noise_snr": 10.0,
                "symmetry_score": 0.35,
                "balance_score": 0.5,
            },
        },
    )


def create_quality_metrics_excellent():
    """Create FrameQualityMetrics with excellent quality."""
    return FrameQualityMetrics(
        blur_score=250.0,  # Excellent blur
        blur_category="excellent",
        contrast_score=50.0,  # Excellent contrast
        contrast_category="excellent",
        brightness_score=125.0,  # Optimal brightness
        brightness_category="excellent",
        overall_quality_score=0.95,
        overall_quality_category="excellent",
        histogram_metrics={
            "dynamic_range": 220,
            "entropy": 7.8,
            "num_peaks": 3,
            "peaks": [50, 125, 200],
            "mean_intensity": 125.0,
            "std_intensity": 40.0,
        },
        sharpness_details={
            "sobel_mean": 35.0,
            "edge_density": 0.25,
            "gradient_variance": 100.0,
        },
        color_metrics={
            "saturation": {"mean": 120.0, "std": 35.0, "min": 20.0, "max": 220.0},
            "hue_distribution": {"mean": 120.0, "std": 60.0, "dominant_hue": 120},
            "color_balance": {
                "red_mean": 120.0,
                "green_mean": 125.0,
                "blue_mean": 130.0,
                "color_cast": "neutral",
            },
            "color_diversity": 5.0,
        },
        noise_metrics={
            "noise_std_laplacian": 3.0,
            "noise_estimate_gaussian": 2.0,
            "signal_to_noise_ratio": 100.0,
            "noise_level": "very_low",
        },
        composition_metrics={
            "rule_of_thirds": {
                "edge_strength_mean": 25.0,
                "edge_strength_std": 8.0,
                "alignment_score": 0.85,
            },
            "symmetry": {"vertical": 0.85, "horizontal": 0.82, "overall": 0.835},
            "focus_point": {"x": 0.5, "y": 0.5, "distance_from_center": 0.0},
            "balance": {
                "horizontal": 0.92,
                "vertical": 0.90,
                "diagonal": 0.88,
                "overall": 0.90,
            },
        },
        quality_report={
            "overall_score": 0.95,
            "overall_category": "excellent",
            "strengths": [
                "Excellent image sharpness",
                "Excellent contrast",
                "Optimal brightness levels",
                "Very low noise levels",
                "Good visual symmetry",
            ],
            "issues": [],
            "recommendations": [],
            "technical_details": {
                "blur": {"score": 250.0, "category": "excellent"},
                "contrast": {"score": 50.0, "category": "excellent"},
                "brightness": {"score": 125.0, "category": "excellent"},
                "dynamic_range": 220,
                "entropy": 7.8,
                "edge_density": 0.25,
                "saturation_mean": 120.0,
                "noise_snr": 100.0,
                "symmetry_score": 0.835,
                "balance_score": 0.90,
            },
        },
    )


class TestQualityReporting:
    """Test quality reporting functionality."""

    def test_frame_quality_summary(self):
        """Test getting quality summary from ExtractedFrame."""
        metrics = create_quality_metrics_with_issues()
        frame = ExtractedFrame(
            frame_number=100,
            timestamp=3.33,
            scene_number=1,
            width=1920,
            height=1080,
            quality_metrics=metrics,
        )

        summary = frame.get_quality_summary()

        assert summary["overall_score"] == 0.25
        assert summary["overall_category"] == "poor"
        assert summary["blur_category"] == "poor"
        assert summary["contrast_category"] == "poor"
        assert summary["brightness_category"] == "poor"
        assert summary["timestamp"] == 3.33
        assert summary["frame_number"] == 100

    def test_scene_quality_report(self):
        """Test generating quality report for a scene."""
        # Create frames with mixed quality
        poor_frame = ExtractedFrame(
            frame_number=100,
            timestamp=3.33,
            scene_number=1,
            width=1920,
            height=1080,
            quality_metrics=create_quality_metrics_with_issues(),
        )

        excellent_frame = ExtractedFrame(
            frame_number=200,
            timestamp=6.66,
            scene_number=1,
            width=1920,
            height=1080,
            quality_metrics=create_quality_metrics_excellent(),
        )

        scene_analysis = SceneFrameAnalysis(
            scene_number=1,
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            frames=[poor_frame, excellent_frame],
            total_frames_extracted=2,
            best_frame=excellent_frame,
            average_quality_score=0.60,
            quality_distribution={"poor": 1, "excellent": 1, "good": 0, "fair": 0},
            total_frames_processed=2,
            frames_filtered_by_quality=0,
            extraction_success_rate=1.0,
        )

        report = scene_analysis.get_quality_report()

        assert report["scene_number"] == 1
        assert report["duration"] == 10.0
        assert report["frames_analyzed"] == 2
        assert report["average_quality"]["score"] == 0.60
        assert report["average_quality"]["category"] == "good"
        assert report["quality_breakdown"]["poor"] == 1
        assert report["quality_breakdown"]["excellent"] == 1
        assert report["best_frame_timestamp"] == 6.66
        # Even though average is good, we still report individual poor frames
        assert len(report["issues"]) == 1
        assert "1 frame(s) with poor quality" in report["issues"][0]
        # Recommendations are only generated when >50% of frames have issues
        # With 1 poor and 1 excellent frame (50/50), no recommendations are generated
        assert len(report["recommendations"]) == 0

    def test_scene_quality_report_with_issues(self):
        """Test quality report for scene with many issues."""
        # Create multiple poor quality frames
        poor_frames = []
        for i in range(3):
            poor_frames.append(
                ExtractedFrame(
                    frame_number=100 * (i + 1),
                    timestamp=3.33 * (i + 1),
                    scene_number=1,
                    width=1920,
                    height=1080,
                    quality_metrics=create_quality_metrics_with_issues(),
                )
            )

        scene_analysis = SceneFrameAnalysis(
            scene_number=1,
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            frames=poor_frames,
            total_frames_extracted=3,
            best_frame=poor_frames[0],
            average_quality_score=0.25,
            quality_distribution={"poor": 3, "excellent": 0, "good": 0, "fair": 0},
            total_frames_processed=4,
            frames_filtered_by_quality=1,
            extraction_success_rate=0.75,
        )

        report = scene_analysis.get_quality_report()

        assert report["average_quality"]["score"] == 0.25
        assert report["average_quality"]["category"] == "poor"
        assert "3 frame(s) with poor quality" in report["issues"]
        assert "Overall scene quality is below acceptable threshold" in report["issues"]
        assert "Low extraction success rate: 75.0%" in report["issues"]
        assert "Consider improving video recording quality" in report["recommendations"]
        assert "Focus on camera stability to reduce blur" in report["recommendations"]
        assert (
            "Improve lighting contrast in recording environment"
            in report["recommendations"]
        )
        assert (
            "Adjust lighting levels for better visibility" in report["recommendations"]
        )

    def test_overall_quality_report(self):
        """Test generating overall quality report for entire video."""
        # Create scene analyses with different quality levels
        poor_frame = ExtractedFrame(
            frame_number=100,
            timestamp=3.33,
            scene_number=1,
            width=1920,
            height=1080,
            quality_metrics=create_quality_metrics_with_issues(),
        )

        excellent_frame = ExtractedFrame(
            frame_number=200,
            timestamp=13.33,
            scene_number=2,
            width=1920,
            height=1080,
            quality_metrics=create_quality_metrics_excellent(),
        )

        scene_1 = SceneFrameAnalysis(
            scene_number=1,
            start_time=0.0,
            end_time=10.0,
            duration=10.0,
            frames=[poor_frame],
            total_frames_extracted=1,
            best_frame=poor_frame,
            average_quality_score=0.25,
            quality_distribution={"poor": 1, "excellent": 0, "good": 0, "fair": 0},
            total_frames_processed=1,
            frames_filtered_by_quality=0,
            extraction_success_rate=1.0,
        )

        scene_2 = SceneFrameAnalysis(
            scene_number=2,
            start_time=10.0,
            end_time=20.0,
            duration=10.0,
            frames=[excellent_frame],
            total_frames_extracted=1,
            best_frame=excellent_frame,
            average_quality_score=0.95,
            quality_distribution={"poor": 0, "excellent": 1, "good": 0, "fair": 0},
            total_frames_processed=1,
            frames_filtered_by_quality=0,
            extraction_success_rate=1.0,
        )

        result = VisualAnalysisResult(
            total_scenes=2,
            total_frames_extracted=2,
            total_frames_processed=2,
            overall_success_rate=1.0,
            scene_analyses=[scene_1, scene_2],
            overall_quality_distribution={
                "poor": 1,
                "excellent": 1,
                "good": 0,
                "fair": 0,
            },
            average_quality_score=0.60,
            best_frames_per_scene=[poor_frame, excellent_frame],
            video_duration=20.0,
            extraction_method="scene_based",
            processing_time=5.0,
        )

        report = result.generate_quality_report()

        # Check summary
        assert report["summary"]["total_scenes"] == 2
        assert report["summary"]["total_frames_analyzed"] == 2
        assert report["summary"]["overall_quality_score"] == 0.60
        assert report["summary"]["overall_quality_category"] == "good"
        assert report["summary"]["processing_success_rate"] == 1.0
        assert report["summary"]["scenes_with_issues"] == 1
        assert report["summary"]["total_quality_issues"] > 0

        # Check quality distribution
        assert report["quality_distribution"]["poor"] == 1
        assert report["quality_distribution"]["excellent"] == 1

        # Check scene reports
        assert len(report["scene_quality_reports"]) == 2
        assert report["scene_quality_reports"][0]["scene_number"] == 1
        assert report["scene_quality_reports"][1]["scene_number"] == 2

        # Check recommendations (should aggregate from poor scene)
        assert len(report["top_recommendations"]) > 0
        assert any("camera" in rec for rec in report["top_recommendations"])

        # Check technical details
        assert report["technical_details"]["video_duration"] == 20.0
        assert report["technical_details"]["extraction_method"] == "scene_based"
        assert report["technical_details"]["processing_time"] == 5.0
        assert report["technical_details"]["frames_per_scene"] == 1.0

        # Check best frames
        assert len(report["best_frames"]) == 2
        assert report["best_frames"][0]["scene"] == 1
        assert report["best_frames"][0]["quality_score"] == 0.25
        assert report["best_frames"][1]["scene"] == 2
        assert report["best_frames"][1]["quality_score"] == 0.95

    def test_quality_metrics_edge_cases(self, frame_extractor):
        """Test quality metrics with edge case values."""
        # Create test frames with extreme values
        black_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        white_frame = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # Test black frame
        black_metrics = frame_extractor._assess_frame_quality(black_frame)
        assert black_metrics.brightness_category == "poor"
        assert black_metrics.brightness_score < 50
        # Find the darkness issue in the list of issues
        darkness_issue_found = any(
            "too dark" in issue.lower()
            for issue in black_metrics.quality_report["issues"]
        )
        assert darkness_issue_found, (
            f"Expected 'too dark' in issues, got: {black_metrics.quality_report['issues']}"
        )

        # Test white frame
        white_metrics = frame_extractor._assess_frame_quality(white_frame)
        assert white_metrics.brightness_category == "poor"
        assert white_metrics.brightness_score > 200
        # Find the overexposure issue in the list of issues
        overexposed_issue_found = any(
            "overexposed" in issue.lower()
            for issue in white_metrics.quality_report["issues"]
        )
        assert overexposed_issue_found, (
            f"Expected 'overexposed' in issues, got: {white_metrics.quality_report['issues']}"
        )

    def test_color_metrics_analysis(self, frame_extractor):
        """Test color metrics analysis."""
        # Create frame with color cast
        red_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        red_frame[:, :, 2] = 200  # High red channel (BGR format)
        red_frame[:, :, 1] = 50  # Low green
        red_frame[:, :, 0] = 50  # Low blue

        metrics = frame_extractor._analyze_color_metrics(red_frame)

        assert metrics["color_balance"]["color_cast"] == "red_cast"
        assert (
            metrics["color_balance"]["red_mean"]
            > metrics["color_balance"]["green_mean"]
        )
        assert (
            metrics["color_balance"]["red_mean"] > metrics["color_balance"]["blue_mean"]
        )
        assert "saturation" in metrics
        assert "hue_distribution" in metrics
        assert "color_diversity" in metrics

    def test_noise_metrics_analysis(self, frame_extractor):
        """Test noise metrics analysis.

        A uniform-random frame is the worst case for the Laplacian high-pass
        noise estimator: every pixel is independent of its neighbours, so the
        noise std is dominated by the pixel-to-pixel variance itself. We pin
        the bucket to ``high`` or ``very_high`` rather than just "in the
        enum" — landing in ``low``/``moderate`` would mean the categorizer
        thresholds drifted out from under us.
        """
        # Use a fixed seed so the test is deterministic across runs.
        rng = np.random.default_rng(seed=42)
        gray_noisy = rng.integers(0, 256, size=(100, 100), dtype=np.uint8)

        metrics = frame_extractor._analyze_noise_metrics(gray_noisy)

        assert "noise_std_laplacian" in metrics
        assert "noise_estimate_gaussian" in metrics
        assert "signal_to_noise_ratio" in metrics
        assert "noise_level" in metrics
        assert metrics["noise_level"] in {"high", "very_high"}, (
            f"Expected uniform-random noise to bucket as high/very_high; "
            f"got {metrics['noise_level']} (std={metrics['noise_std_laplacian']:.2f})"
        )

    def test_composition_metrics_analysis(self, frame_extractor):
        """Test composition metrics analysis.

        A perfectly centered subject with sharp edges should produce a focus
        point essentially *at* the centre — the gradient mass is symmetric
        about the centre lines. We pin distance_from_center < 0.1 (instead of
        the previous >= 0 tautology) so any future regression that moves the
        focus-point detector off-centre by more than ~10% of the frame
        diagonal fails loudly.
        """
        # Create frame with clear subject in center
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        # Add white rectangle in center
        frame[100:200, 150:250] = 255

        metrics = frame_extractor._analyze_composition_metrics(frame)

        assert "rule_of_thirds" in metrics
        assert "symmetry" in metrics
        assert "focus_point" in metrics
        assert "balance" in metrics

        # Check symmetry scores
        assert 0 <= metrics["symmetry"]["vertical"] <= 1
        assert 0 <= metrics["symmetry"]["horizontal"] <= 1
        assert 0 <= metrics["symmetry"]["overall"] <= 1

        # Check focus point
        assert 0 <= metrics["focus_point"]["x"] <= 1
        assert 0 <= metrics["focus_point"]["y"] <= 1
        # Centered subject -> focus point should be at the centre, not just
        # "somewhere on the canvas". Tightened from >= 0 (always true).
        assert metrics["focus_point"]["distance_from_center"] < 0.1, (
            f"Centered white square should produce focus point near centre; "
            f"got distance_from_center={metrics['focus_point']['distance_from_center']:.3f}"
        )

        # Check balance scores
        assert 0 <= metrics["balance"]["overall"] <= 1
