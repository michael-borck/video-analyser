"""Tests for frame analysis pipeline."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from video_analyser.analysis.frame_analyzer import (
    FrameAnalysisPipeline,
    FrameAnalysisStep,
    PipelineMetrics,
    create_frame_analysis_pipeline,
)
from video_analyser.analysis.visual_analyzer import (
    ExtractedFrame,
    SceneFrameAnalysis,
    VisualAnalysisResult,
)
from video_analyser.core.exceptions import VideoProcessingError
from video_analyser.core.scene_detector import Scene, SceneDetectionResult
from video_analyser.utils.config import VideoAnalyserConfig, VisualAnalysisConfig


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
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
            enable_captioning=True,
            enable_ocr=True,
            enable_object_detection=True,
        )
    )
    return config


@pytest.fixture
def mock_config_minimal():
    """Create minimal configuration with only quality assessment."""
    config = VideoAnalyserConfig(
        visual_analysis=VisualAnalysisConfig(
            frames_per_scene=1,
            enable_captioning=False,
            enable_ocr=False,
            enable_object_detection=False,
        )
    )
    return config


@pytest.fixture
def pipeline(mock_config):
    """Create FrameAnalysisPipeline instance for testing."""
    return FrameAnalysisPipeline(config=mock_config)


@pytest.fixture
def pipeline_minimal(mock_config_minimal):
    """Create minimal FrameAnalysisPipeline instance."""
    return FrameAnalysisPipeline(config=mock_config_minimal)


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
def test_video():
    """Create a test video file."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp.name, fourcc, 30.0, (640, 480))

        # Generate frames
        for i in range(300):  # 10 seconds at 30fps
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

            # Add some content
            cv2.putText(
                frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
            )

            # Add different content for different scenes
            if i < 150:
                cv2.rectangle(frame, (100, 100), (300, 300), (0, 0, 255), -1)
            else:
                cv2.circle(frame, (320, 240), 100, (255, 0, 0), -1)

            out.write(frame)

        out.release()
        return Path(tmp.name)


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 240
    cv2.putText(
        frame, "Test Frame", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3
    )
    cv2.rectangle(frame, (50, 150), (590, 450), (100, 100, 100), -1)
    return frame


class TestFrameAnalysisStep:
    """Test FrameAnalysisStep model."""

    def test_frame_analysis_step_creation(self):
        """Test creating FrameAnalysisStep."""
        step = FrameAnalysisStep(
            name="test_step",
            enabled=True,
            weight=0.5,
        )

        assert step.name == "test_step"
        assert step.enabled is True
        assert step.weight == 0.5

    def test_frame_analysis_step_defaults(self):
        """Test default values."""
        step = FrameAnalysisStep(
            name="test_step",
            enabled=False,
        )

        assert step.weight == 1.0


class TestPipelineMetrics:
    """Test PipelineMetrics model."""

    def test_pipeline_metrics_creation(self):
        """Test creating PipelineMetrics."""
        metrics = PipelineMetrics(
            total_frames_processed=10,
            successful_frames=9,
            failed_frames=1,
            total_processing_time=5.0,
            average_frame_time=0.5,
            step_timings={"quality": 0.1, "caption": 0.3},
            step_success_rates={"quality": 1.0, "caption": 0.9},
            average_quality_score=0.75,
            quality_distribution={"excellent": 3, "good": 4, "fair": 2, "poor": 1},
            frames_with_captions=8,
            frames_with_ocr=7,
            frames_with_objects=9,
        )

        assert metrics.total_frames_processed == 10
        assert metrics.successful_frames == 9
        assert metrics.average_frame_time == 0.5

    def test_get_summary(self):
        """Test metrics summary generation."""
        metrics = PipelineMetrics(
            total_frames_processed=10,
            successful_frames=9,
            failed_frames=1,
            total_processing_time=5.0,
            average_frame_time=0.5,
            step_timings={},
            step_success_rates={},
            average_quality_score=0.75,
            quality_distribution={},
            frames_with_captions=8,
            frames_with_ocr=7,
            frames_with_objects=9,
        )

        summary = metrics.get_summary()

        assert summary["success_rate"] == 0.9
        assert summary["average_processing_time"] == 0.5
        assert summary["total_processing_time"] == 5.0
        assert summary["quality_score"] == 0.75
        assert summary["analysis_coverage"]["captions"] == 0.8
        assert summary["analysis_coverage"]["ocr"] == 0.7
        assert summary["analysis_coverage"]["objects"] == 0.9

    def test_get_summary_empty_metrics(self):
        """Test summary with no frames processed."""
        metrics = PipelineMetrics(
            total_frames_processed=0,
            successful_frames=0,
            failed_frames=0,
            total_processing_time=0.0,
            average_frame_time=0.0,
            step_timings={},
            step_success_rates={},
            average_quality_score=0.0,
            quality_distribution={},
            frames_with_captions=0,
            frames_with_ocr=0,
            frames_with_objects=0,
        )

        summary = metrics.get_summary()

        assert summary["success_rate"] == 0
        assert summary["analysis_coverage"]["captions"] == 0


class TestFrameAnalysisPipeline:
    """Test FrameAnalysisPipeline class."""

    def test_initialization_with_all_enabled(self, pipeline):
        """Test pipeline initialization with all analyses enabled."""
        assert pipeline.config is not None
        assert pipeline.frame_extractor is not None
        assert len(pipeline.analysis_steps) == 4

        # Check enabled steps
        enabled_steps = [step.name for step in pipeline.analysis_steps if step.enabled]
        assert "quality_assessment" in enabled_steps
        assert "image_captioning" in enabled_steps
        assert "text_extraction" in enabled_steps
        assert "object_detection" in enabled_steps

        assert pipeline.total_weight > 0

    def test_initialization_minimal(self, pipeline_minimal):
        """Test pipeline initialization with minimal config."""
        enabled_steps = [
            step.name for step in pipeline_minimal.analysis_steps if step.enabled
        ]
        assert "quality_assessment" in enabled_steps
        assert "image_captioning" not in enabled_steps
        assert "text_extraction" not in enabled_steps
        assert "object_detection" not in enabled_steps

    def test_get_enabled_analyses(self, pipeline):
        """Test getting list of enabled analyses."""
        enabled = pipeline.get_enabled_analyses()

        assert "quality_assessment" in enabled
        assert "image_captioning" in enabled
        assert "text_extraction" in enabled
        assert "object_detection" in enabled
        assert len(enabled) == 4

    def test_analyze_single_frame(self, pipeline, sample_frame):
        """Test analyzing a single frame."""
        frame_info = {
            "frame_number": 100,
            "timestamp": 3.33,
            "scene_number": 1,
        }

        # Mock the analysis methods
        with (
            patch.object(
                pipeline.frame_extractor, "_assess_frame_quality"
            ) as mock_quality,
            patch.object(pipeline.frame_extractor, "_caption_frame") as mock_caption,
            patch.object(
                pipeline.frame_extractor, "_extract_text_from_frame"
            ) as mock_ocr,
            patch.object(
                pipeline.frame_extractor, "_detect_objects_in_frame"
            ) as mock_detect,
        ):
            # Set up mocks
            from tests.analysis.test_visual_analyzer import (
                create_test_quality_metrics,
            )

            mock_quality.return_value = create_test_quality_metrics()
            mock_caption.return_value = None
            mock_ocr.return_value = None
            mock_detect.return_value = None

            # Analyze frame
            result = pipeline.analyze_single_frame(sample_frame, frame_info)

            assert isinstance(result, ExtractedFrame)
            assert result.frame_number == 100
            assert result.timestamp == 3.33
            assert result.scene_number == 1
            assert result.width == 640
            assert result.height == 480

            # Verify all methods were called
            mock_quality.assert_called_once()
            mock_caption.assert_called_once()
            mock_ocr.assert_called_once()
            mock_detect.assert_called_once()

    def test_analyze_single_frame_minimal(self, pipeline_minimal, sample_frame):
        """Test analyzing frame with minimal pipeline."""
        with patch.object(
            pipeline_minimal.frame_extractor, "_assess_frame_quality"
        ) as mock_quality:
            from tests.analysis.test_visual_analyzer import create_test_quality_metrics

            mock_quality.return_value = create_test_quality_metrics()

            result = pipeline_minimal.analyze_single_frame(sample_frame)

            assert result.caption_result is None
            assert result.ocr_result is None
            assert result.object_detection_result is None

    def test_analyze_video_frames(self, pipeline, test_video, mock_scene_result):
        """Test analyzing video frames through pipeline."""
        with patch.object(
            pipeline.frame_extractor, "extract_frames_from_scenes"
        ) as mock_extract:
            # Create mock visual result
            from tests.analysis.test_visual_analyzer import create_test_quality_metrics

            mock_frames = []
            for i in range(4):  # 2 scenes x 2 frames per scene
                frame = ExtractedFrame(
                    frame_number=i * 75,
                    timestamp=i * 2.5,
                    scene_number=(i // 2) + 1,
                    width=640,
                    height=480,
                    quality_metrics=create_test_quality_metrics(),
                )
                mock_frames.append(frame)

            mock_scene_analyses = [
                SceneFrameAnalysis(
                    scene_number=1,
                    start_time=0.0,
                    end_time=5.0,
                    duration=5.0,
                    frames=mock_frames[:2],
                    total_frames_extracted=2,
                    best_frame=mock_frames[0],
                    average_quality_score=0.75,
                    quality_distribution={
                        "good": 2,
                        "excellent": 0,
                        "fair": 0,
                        "poor": 0,
                    },
                    total_frames_processed=2,
                    frames_filtered_by_quality=0,
                    extraction_success_rate=1.0,
                ),
                SceneFrameAnalysis(
                    scene_number=2,
                    start_time=5.0,
                    end_time=10.0,
                    duration=5.0,
                    frames=mock_frames[2:],
                    total_frames_extracted=2,
                    best_frame=mock_frames[2],
                    average_quality_score=0.75,
                    quality_distribution={
                        "good": 2,
                        "excellent": 0,
                        "fair": 0,
                        "poor": 0,
                    },
                    total_frames_processed=2,
                    frames_filtered_by_quality=0,
                    extraction_success_rate=1.0,
                ),
            ]

            mock_visual_result = VisualAnalysisResult(
                total_scenes=2,
                total_frames_extracted=4,
                total_frames_processed=4,
                overall_success_rate=1.0,
                scene_analyses=mock_scene_analyses,
                overall_quality_distribution={
                    "good": 4,
                    "excellent": 0,
                    "fair": 0,
                    "poor": 0,
                },
                average_quality_score=0.75,
                best_frames_per_scene=[mock_frames[0], mock_frames[2]],
                video_duration=10.0,
                extraction_method="scene_based",
                processing_time=2.0,
            )

            mock_extract.return_value = mock_visual_result

            # Analyze video
            visual_result, metrics = pipeline.analyze_video_frames(
                video_path=test_video,
                scene_result=mock_scene_result,
            )

            assert visual_result == mock_visual_result
            assert isinstance(metrics, PipelineMetrics)
            assert metrics.total_frames_processed == 4
            assert metrics.successful_frames == 4
            assert metrics.average_quality_score == 0.75

    def test_analyze_video_frames_with_progress(
        self, pipeline, test_video, mock_scene_result
    ):
        """Test video analysis with progress callback."""
        progress_updates = []

        def progress_callback(progress, message):
            progress_updates.append((progress, message))

        with patch.object(
            pipeline.frame_extractor, "extract_frames_from_scenes"
        ) as mock_extract:
            # Create minimal mock result
            mock_visual_result = MagicMock()
            mock_visual_result.total_frames_extracted = 2
            mock_visual_result.total_frames_processed = 2
            mock_visual_result.overall_quality_distribution = {}
            mock_visual_result.average_quality_score = 0.7
            mock_visual_result.scene_analyses = []

            mock_extract.return_value = mock_visual_result

            # Analyze with progress
            visual_result, metrics = pipeline.analyze_video_frames(
                video_path=test_video,
                scene_result=mock_scene_result,
                progress_callback=progress_callback,
            )

            # Check progress updates were made
            assert len(progress_updates) > 0

    def test_analyze_video_frames_error_handling(
        self, pipeline, test_video, mock_scene_result
    ):
        """Test error handling in video analysis."""
        with patch.object(
            pipeline.frame_extractor, "extract_frames_from_scenes"
        ) as mock_extract:
            mock_extract.side_effect = Exception("Extraction failed")

            with pytest.raises(VideoProcessingError) as exc_info:
                pipeline.analyze_video_frames(
                    video_path=test_video,
                    scene_result=mock_scene_result,
                )

            assert "Frame analysis pipeline failed" in str(exc_info.value)

    def test_generate_analysis_summary(self, pipeline):
        """Test generating analysis summary."""
        # Create mock data
        from tests.analysis.test_visual_analyzer import create_test_quality_metrics

        mock_frame = ExtractedFrame(
            frame_number=0,
            timestamp=0.0,
            scene_number=1,
            width=640,
            height=480,
            quality_metrics=create_test_quality_metrics(),
        )

        mock_scene_analysis = SceneFrameAnalysis(
            scene_number=1,
            start_time=0.0,
            end_time=5.0,
            duration=5.0,
            frames=[mock_frame],
            total_frames_extracted=1,
            best_frame=mock_frame,
            average_quality_score=0.75,
            quality_distribution={"good": 1, "excellent": 0, "fair": 0, "poor": 0},
            total_frames_processed=1,
            frames_filtered_by_quality=0,
            extraction_success_rate=1.0,
        )

        visual_result = VisualAnalysisResult(
            total_scenes=1,
            total_frames_extracted=1,
            total_frames_processed=1,
            overall_success_rate=1.0,
            scene_analyses=[mock_scene_analysis],
            overall_quality_distribution={
                "good": 1,
                "excellent": 0,
                "fair": 0,
                "poor": 0,
            },
            average_quality_score=0.75,
            best_frames_per_scene=[mock_frame],
            video_duration=5.0,
            extraction_method="scene_based",
            processing_time=1.0,
        )

        metrics = PipelineMetrics(
            total_frames_processed=1,
            successful_frames=1,
            failed_frames=0,
            total_processing_time=1.0,
            average_frame_time=1.0,
            step_timings={},
            step_success_rates={},
            average_quality_score=0.75,
            quality_distribution={"good": 1},
            frames_with_captions=0,
            frames_with_ocr=0,
            frames_with_objects=0,
        )

        # Generate summary
        summary = pipeline.generate_analysis_summary(visual_result, metrics)

        assert "video_overview" in summary
        assert summary["video_overview"]["duration"] == 5.0
        assert summary["video_overview"]["total_scenes"] == 1

        assert "quality_summary" in summary
        assert summary["quality_summary"]["average_score"] == 0.75

        assert "content_analysis" in summary
        assert "analysis_coverage" in summary
        assert "key_insights" in summary
        assert "performance" in summary

    def test_determine_content_type(self, pipeline):
        """Test content type determination."""
        # Create mock visual result with slide layout
        from video_analyser.analysis.object_detector import ObjectDetectionResult

        mock_obj_result = ObjectDetectionResult(
            detected_objects=[],
            total_objects=0,
            processing_time=0.1,
            frame_width=640,
            frame_height=480,
            element_counts={},
            high_confidence_objects=0,
            average_confidence=0.0,
            layout_type="slide",
            has_presenter=False,
            dominant_element=None,
        )

        from tests.analysis.test_visual_analyzer import create_test_quality_metrics

        mock_frame = ExtractedFrame(
            frame_number=0,
            timestamp=0.0,
            scene_number=1,
            width=640,
            height=480,
            quality_metrics=create_test_quality_metrics(),
            object_detection_result=mock_obj_result,
        )

        mock_scene = SceneFrameAnalysis(
            scene_number=1,
            start_time=0.0,
            end_time=5.0,
            duration=5.0,
            frames=[mock_frame],
            total_frames_extracted=1,
            best_frame=mock_frame,
            average_quality_score=0.75,
            quality_distribution={},
            total_frames_processed=1,
            frames_filtered_by_quality=0,
            extraction_success_rate=1.0,
        )

        visual_result = VisualAnalysisResult(
            total_scenes=1,
            total_frames_extracted=1,
            total_frames_processed=1,
            overall_success_rate=1.0,
            scene_analyses=[mock_scene],
            overall_quality_distribution={},
            average_quality_score=0.75,
            best_frames_per_scene=[mock_frame],
            video_duration=5.0,
            extraction_method="scene_based",
            processing_time=1.0,
        )

        metrics = PipelineMetrics(
            total_frames_processed=1,
            successful_frames=1,
            failed_frames=0,
            total_processing_time=1.0,
            average_frame_time=1.0,
            step_timings={},
            step_success_rates={},
            average_quality_score=0.75,
            quality_distribution={},
            frames_with_captions=0,
            frames_with_ocr=0,
            frames_with_objects=1,
        )

        content_type = pipeline._determine_content_type(visual_result, metrics)
        assert content_type == "presentation"

    def test_cleanup(self, pipeline):
        """Test pipeline cleanup."""
        with patch.object(pipeline.frame_extractor, "cleanup") as mock_cleanup:
            pipeline.cleanup()
            mock_cleanup.assert_called_once()


class TestFrameAnalysisPipelineFactory:
    """Test factory function."""

    def test_create_frame_analysis_pipeline(self):
        """Test creating pipeline with factory."""
        pipeline = create_frame_analysis_pipeline()
        assert isinstance(pipeline, FrameAnalysisPipeline)
        assert pipeline.config is not None

    def test_create_frame_analysis_pipeline_with_config(self, mock_config):
        """Test creating pipeline with custom config."""
        pipeline = create_frame_analysis_pipeline(config=mock_config)
        assert isinstance(pipeline, FrameAnalysisPipeline)
        assert pipeline.config == mock_config


class TestPipelineIntegration:
    """Integration tests for the pipeline."""

    def test_full_pipeline_workflow(self, test_video):
        """Test complete pipeline workflow."""
        # Create pipeline with all analyses enabled
        config = VideoAnalyserConfig(
            visual_analysis=VisualAnalysisConfig(
                frames_per_scene=1,
                enable_captioning=False,  # Disable to avoid model loading
                enable_ocr=False,  # Disable to avoid dependencies
                enable_object_detection=True,  # Keep enabled as it's lightweight
            )
        )
        pipeline = FrameAnalysisPipeline(config=config)

        # Create simple scene result
        scene = Scene(
            scene_number=1,
            start_time=0.0,
            end_time=2.0,
            duration=2.0,
            start_frame=0,
            end_frame=60,
            confidence=0.95,
            transition_type="cut",
        )

        scene_result = SceneDetectionResult(
            scenes=[scene],
            total_scenes=1,
            detection_method="threshold",
            threshold_used=0.3,
            video_duration=2.0,
            average_scene_duration=2.0,
        )

        # Run pipeline
        visual_result, metrics = pipeline.analyze_video_frames(
            video_path=test_video,
            scene_result=scene_result,
        )

        # Verify results
        assert visual_result.total_scenes == 1
        assert visual_result.total_frames_extracted > 0
        assert metrics.total_frames_processed > 0
        assert metrics.average_quality_score >= 0

        # Generate summary
        summary = pipeline.generate_analysis_summary(visual_result, metrics)
        assert summary is not None
        assert "video_overview" in summary
        assert "quality_summary" in summary

        # Cleanup
        pipeline.cleanup()
        test_video.unlink()  # Clean up test file
