"""Presentation-element classification over COCO object labels.

image-analyser returns generic COCO labels (person, tv, book, laptop,
chair, etc.). For presentation video analysis we want a more meaningful
taxonomy: SLIDE, PRESENTER, AUDIENCE, SCREEN, CHART, etc. This module
maps COCO -> PresentationElement and computes layout aggregations.

Kept in video-analyser (not image-analyser) because the mapping is
domain-specific to presentation/lecture analysis.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from image_analyser.schemas import Object as ImageObject


class PresentationElement(str, Enum):
    """Domain-specific element taxonomy for presentation/lecture frames."""

    SLIDE = "slide"
    PRESENTER = "presenter"
    AUDIENCE = "audience"
    SCREEN = "screen"
    CHART = "chart"
    DIAGRAM = "diagram"
    TEXT_BLOCK = "text_block"
    LOGO = "logo"
    OBJECT = "object"  # fallback for unmapped labels


# COCO label -> PresentationElement. Lifted from the deleted object_detector.py
# heuristics. Add/edit as the analyser sees more labels in practice.
_COCO_TO_ELEMENT: dict[str, PresentationElement] = {
    "person": PresentationElement.PRESENTER,
    "tv": PresentationElement.SCREEN,
    "laptop": PresentationElement.SCREEN,
    "monitor": PresentationElement.SCREEN,
    "book": PresentationElement.SLIDE,  # slide deck on a screen
    "cell phone": PresentationElement.OBJECT,
    "remote": PresentationElement.OBJECT,
    "chair": PresentationElement.AUDIENCE,
    "couch": PresentationElement.AUDIENCE,
    "bench": PresentationElement.AUDIENCE,
    # Extend over time as you observe new labels in real video frames.
}


class ClassifiedObject(BaseModel):
    """A single COCO detection mapped into the presentation taxonomy."""

    element_type: PresentationElement
    coco_label: str
    confidence: float  # from image_analyser Object.score
    bbox: tuple[int, int, int, int]  # (x, y, w, h)

    @property
    def description(self) -> str:
        """Human-readable description, used by reports."""
        return f"{self.element_type.value} ({self.coco_label}, {self.confidence:.1%})"


class PresentationLayout(BaseModel):
    """Aggregations over a frame's classified objects."""

    objects: list[ClassifiedObject]
    has_presenter: bool
    has_screen: bool
    presenter_count: int
    dominant_element: PresentationElement | None
    element_counts: dict[PresentationElement, int]
    layout_type: str  # "single-presenter" | "multi-presenter" | "slide-only" | "audience-only" | "mixed" | "empty"

    # ------------------------------------------------------------------
    # Backward-compatible accessors so existing reports / pipeline code
    # that previously read ObjectDetectionResult continue to work without
    # a sweeping refactor.
    # ------------------------------------------------------------------

    @property
    def detected_objects(self) -> list[ClassifiedObject]:
        """Alias for ``objects`` matching the legacy ObjectDetectionResult name."""
        return self.objects

    @property
    def total_objects(self) -> int:
        """Number of detected objects (legacy field name)."""
        return len(self.objects)


def classify(label: str) -> PresentationElement:
    """Map a single COCO label to a PresentationElement."""
    return _COCO_TO_ELEMENT.get(label.lower(), PresentationElement.OBJECT)


def classify_objects(image_objects: list[ImageObject]) -> PresentationLayout:
    """Classify all detected objects and compute layout aggregations."""
    classified = [
        ClassifiedObject(
            element_type=classify(o.label),
            coco_label=o.label,
            confidence=o.score,
            bbox=(o.bbox.x, o.bbox.y, o.bbox.w, o.bbox.h),
        )
        for o in image_objects
    ]
    counts: Counter[PresentationElement] = Counter(c.element_type for c in classified)
    presenter_count = counts.get(PresentationElement.PRESENTER, 0)
    has_screen = (
        counts.get(PresentationElement.SCREEN, 0) > 0
        or counts.get(PresentationElement.SLIDE, 0) > 0
    )
    dominant = max(counts, key=lambda e: counts[e]) if counts else None
    return PresentationLayout(
        objects=classified,
        has_presenter=presenter_count > 0,
        has_screen=has_screen,
        presenter_count=presenter_count,
        dominant_element=dominant,
        element_counts=dict(counts),
        layout_type=_infer_layout(presenter_count, has_screen, counts),
    )


def _infer_layout(
    presenter_count: int,
    has_screen: bool,
    counts: Counter[PresentationElement],
) -> str:
    """Infer a high-level layout label from per-element counts."""
    if not counts:
        return "empty"
    if presenter_count == 1 and has_screen:
        return "single-presenter"
    if presenter_count > 1:
        return "multi-presenter"
    if has_screen and presenter_count == 0:
        return "slide-only"
    audience = counts.get(PresentationElement.AUDIENCE, 0)
    if audience > 0 and presenter_count == 0:
        return "audience-only"
    return "mixed"
