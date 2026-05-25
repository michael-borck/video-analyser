"""Capability manifest for the lens family (consumed by auto-analyser)."""
from __future__ import annotations

from lens_contract import make_manifest

MANIFEST = make_manifest(
    name="video-analyser",
    accepts=["video"],
    extensions=[".mp4", ".mov", ".avi", ".webm", ".mkv"],
    auto_routable=True,
    produces="VideoAnalysis",
)
