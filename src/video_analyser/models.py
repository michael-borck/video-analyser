"""The VideoAnalysis result model — the lens family `produces` output.

Aggregates the per-phase pipeline outputs into one typed result returned by
`PipelineCoordinator.analyse()` and `POST /analyse`. (Before this, the manifest
advertised `produces: "VideoAnalysis"` but no such model existed — the pipeline
only wrote JSON/HTML report files.)
"""
from __future__ import annotations

from pydantic import BaseModel

from video_analyser.analysis.speech_analyzer import SpeechAnalysisResult
from video_analyser.analysis.transcriber import TranscriptionResult
from video_analyser.analysis.visual_analyzer import ExtractedFrame
from video_analyser.core.audio_extractor import AudioInfo
from video_analyser.core.scene_detector import SceneDetectionResult
from video_analyser.core.video_processor import VideoInfo


class VideoAnalysis(BaseModel):
    """Complete video analysis — the typed contract output.

    Fields mirror the pipeline phases: video/audio metadata, scene detection,
    transcription + scene-aware speech metrics (video's own word-level speech
    stack), and per-frame visual analysis (quality/captions/OCR delegated to
    image-analyser).
    """

    input: str
    video_info: VideoInfo
    audio_info: AudioInfo | None = None
    scenes: SceneDetectionResult | None = None
    transcription: TranscriptionResult | None = None
    speech: SpeechAnalysisResult | None = None
    frames: list[ExtractedFrame] = []
    processing_time: float = 0.0
    success: bool = True
    errors: list[str] = []
