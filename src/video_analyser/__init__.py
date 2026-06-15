"""Video Analyser - Video analysis tool for the analyser family."""

from __future__ import annotations

from importlib.metadata import version as _v
from pathlib import Path

from video_analyser.core.exceptions import VideoProcessingError
from video_analyser.manifest import MANIFEST
from video_analyser.models import VideoAnalysis

__version__ = _v("video-analyser")
del _v
__author__ = "Michael Borck"
__description__ = "Video analysis tool — extracts frames, transcripts, and quality metrics for the analyser family"


class VideoAnalyser:
    """Canonical engine facade exposing `.analyse(...)`.

    Thin wrapper over the internal :class:`PipelineCoordinator`. The heavy
    pipeline (and its ffmpeg/model dependencies) is imported lazily inside
    :meth:`analyse`, so importing this package does no heavy work.
    """

    def analyse(
        self, video_path: str | Path, *, fast_mode: bool = False
    ) -> VideoAnalysis:
        """Run the full pipeline and return a typed :class:`VideoAnalysis`."""
        from video_analyser.core.pipeline_coordinator import PipelineCoordinator

        return PipelineCoordinator().analyse(Path(video_path), fast_mode=fast_mode)


def analyse(video_path: str | Path, *, fast_mode: bool = False) -> VideoAnalysis:
    """Module-level convenience wrapper returning a :class:`VideoAnalysis`."""
    return VideoAnalyser().analyse(video_path, fast_mode=fast_mode)


__all__ = [
    "VideoAnalyser",
    "VideoAnalysis",
    "analyse",
    "MANIFEST",
    "VideoProcessingError",
    "__version__",
]
