"""Public API surface tests for the canonical analyser contract.

These assert the canonical names are importable and well-formed. They do NOT
call ``analyse`` (which needs media files, ffmpeg and ML models).
"""

import video_analyser


def test_canonical_names_importable():
    from video_analyser import (  # noqa: F401
        MANIFEST,
        VideoAnalyser,
        VideoAnalysis,
        analyse,
    )


def test_analyse_is_callable():
    assert callable(video_analyser.analyse)


def test_video_analyser_has_analyse():
    assert hasattr(video_analyser.VideoAnalyser, "analyse")


def test_manifest_name():
    assert video_analyser.MANIFEST["name"] == "video-analyser"


def test_version_is_str():
    assert isinstance(video_analyser.__version__, str)


def test_all_lists_canonical_names():
    for name in ("VideoAnalyser", "VideoAnalysis", "analyse", "MANIFEST", "__version__"):
        assert name in video_analyser.__all__
