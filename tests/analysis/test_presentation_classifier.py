"""Tests for the presentation_classifier post-processing module."""

from __future__ import annotations

from image_analyser.schemas import BBox
from image_analyser.schemas import Object as ImageObject

from video_analyser.analysis.presentation_classifier import (
    ClassifiedObject,
    PresentationElement,
    PresentationLayout,
    classify,
    classify_objects,
)


def _make_obj(label: str, score: float = 0.9) -> ImageObject:
    """Construct an image_analyser Object with a sensible default bbox."""
    return ImageObject(
        label=label,
        score=score,
        bbox=BBox(x=0, y=0, w=10, h=10),
    )


# ---------------------------------------------------------------------------
# classify()
# ---------------------------------------------------------------------------


def test_classify_person_returns_presenter() -> None:
    assert classify("person") == PresentationElement.PRESENTER


def test_classify_tv_returns_screen() -> None:
    assert classify("tv") == PresentationElement.SCREEN


def test_classify_laptop_returns_screen() -> None:
    assert classify("laptop") == PresentationElement.SCREEN


def test_classify_chair_returns_audience() -> None:
    assert classify("chair") == PresentationElement.AUDIENCE


def test_classify_book_returns_slide() -> None:
    assert classify("book") == PresentationElement.SLIDE


def test_classify_unknown_label_returns_object() -> None:
    assert classify("toaster") == PresentationElement.OBJECT


def test_classify_is_case_insensitive() -> None:
    assert classify("PERSON") == PresentationElement.PRESENTER
    assert classify("TV") == PresentationElement.SCREEN


# ---------------------------------------------------------------------------
# classify_objects()
# ---------------------------------------------------------------------------


def test_layout_empty_when_no_objects() -> None:
    layout = classify_objects([])
    assert isinstance(layout, PresentationLayout)
    assert layout.layout_type == "empty"
    assert layout.dominant_element is None
    assert layout.has_presenter is False
    assert layout.has_screen is False
    assert layout.presenter_count == 0
    assert layout.objects == []
    assert layout.element_counts == {}


def test_layout_single_presenter_with_screen() -> None:
    objs = [_make_obj("person", 0.95), _make_obj("tv", 0.88)]
    layout = classify_objects(objs)
    assert layout.has_presenter is True
    assert layout.has_screen is True
    assert layout.layout_type == "single-presenter"
    assert layout.presenter_count == 1
    # element_counts is keyed by Enum members
    assert layout.element_counts[PresentationElement.PRESENTER] == 1
    assert layout.element_counts[PresentationElement.SCREEN] == 1


def test_layout_multi_presenter() -> None:
    objs = [_make_obj("person", 0.9), _make_obj("person", 0.85)]
    layout = classify_objects(objs)
    assert layout.layout_type == "multi-presenter"
    assert layout.presenter_count == 2
    assert layout.has_presenter is True


def test_layout_slide_only() -> None:
    objs = [_make_obj("tv", 0.9)]
    layout = classify_objects(objs)
    assert layout.layout_type == "slide-only"
    assert layout.has_screen is True
    assert layout.has_presenter is False


def test_layout_slide_only_with_book() -> None:
    """``book`` is mapped to SLIDE (slide deck on a screen) and SLIDE counts
    toward ``has_screen``, so a single book detection should yield the
    ``slide-only`` layout. The existing ``test_layout_slide_only`` only
    exercises the ``tv`` -> SCREEN path; this test pins the parallel
    ``book`` -> SLIDE path so a future change to the COCO mapping that
    accidentally drops SLIDE from has_screen aggregation fails loudly.
    """
    objs = [_make_obj("book", 0.9)]
    layout = classify_objects(objs)
    assert layout.layout_type == "slide-only"
    assert layout.has_screen is True  # SLIDE participates in has_screen
    assert layout.has_presenter is False
    assert layout.element_counts[PresentationElement.SLIDE] == 1


def test_layout_presenter_only_no_screen() -> None:
    """A single ``person`` detection with no screen currently buckets as
    ``mixed`` (not ``single-presenter``, which requires has_screen=True).

    This pins the current behaviour: ``_infer_layout`` only reports
    ``single-presenter`` when both presenter_count == 1 *and* has_screen.
    A presenter standing in front of a non-detected screen — or doing a
    talking-head intro before the slides come up — falls through to
    ``mixed``. If we ever decide that case should be ``single-presenter``
    instead, this test must be updated *intentionally* rather than break
    silently.
    """
    objs = [_make_obj("person", 0.95)]
    layout = classify_objects(objs)
    assert layout.has_presenter is True
    assert layout.has_screen is False
    assert layout.presenter_count == 1
    assert layout.layout_type == "mixed"


def test_layout_audience_only() -> None:
    objs = [_make_obj("chair", 0.9), _make_obj("chair", 0.85)]
    layout = classify_objects(objs)
    assert layout.layout_type == "audience-only"
    assert layout.has_presenter is False
    assert layout.has_screen is False


def test_layout_mixed_when_no_specific_pattern() -> None:
    # Just an unmapped object -> OBJECT bucket -> "mixed" fallback.
    objs = [_make_obj("toaster", 0.9)]
    layout = classify_objects(objs)
    assert layout.layout_type == "mixed"


def test_classified_object_carries_coco_label_and_bbox() -> None:
    objs = [
        ImageObject(label="person", score=0.92, bbox=BBox(x=10, y=20, w=30, h=40)),
    ]
    layout = classify_objects(objs)
    assert len(layout.objects) == 1
    obj: ClassifiedObject = layout.objects[0]
    assert obj.coco_label == "person"
    assert obj.element_type == PresentationElement.PRESENTER
    assert obj.confidence == 0.92
    assert obj.bbox == (10, 20, 30, 40)


def test_classified_object_description_includes_label_and_confidence() -> None:
    obj = ClassifiedObject(
        element_type=PresentationElement.PRESENTER,
        coco_label="person",
        confidence=0.92,
        bbox=(0, 0, 10, 10),
    )
    desc = obj.description
    assert "presenter" in desc
    assert "person" in desc
    assert "92" in desc  # 0.92 -> "92.0%"


def test_layout_dominant_element_picks_most_common() -> None:
    objs = [
        _make_obj("person", 0.9),
        _make_obj("chair", 0.8),
        _make_obj("chair", 0.85),
        _make_obj("chair", 0.82),
    ]
    layout = classify_objects(objs)
    assert layout.dominant_element == PresentationElement.AUDIENCE


def test_legacy_aliases_total_objects_and_detected_objects() -> None:
    """PresentationLayout exposes legacy ObjectDetectionResult field aliases."""
    objs = [_make_obj("person"), _make_obj("tv")]
    layout = classify_objects(objs)
    # Aliases used by report_generator / frame_analyzer
    assert layout.total_objects == 2
    assert layout.detected_objects == layout.objects
