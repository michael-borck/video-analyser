"""Capability manifest for the lens family (consumed by auto-analyser)."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def _version() -> str:
    try:
        return version("video-analyser")
    except PackageNotFoundError:
        return "0.0.0"


MANIFEST: dict = {
    "name": "video-analyser",
    "version": _version(),
    "role": "analyser",
    "accepts": ["video"],
    "extensions": [".mp4", ".mov", ".avi", ".webm", ".mkv"],
    "auto_routable": True,
    "produces": "VideoAnalysis",
}
