"""Invariant tests — fast, no real ML models, run by default.

These guard against the failure modes that motivated the family-wide
trust-building pass: packaging bugs (records-analyser was un-importable
for weeks because no test ever exercised the bare import), version-string
drift across hardcoded literals, regression of the v0.7.0 image-analyser
delegation back into in-house adapter classes, and silent loud-fallback
holes in the COCO->PresentationElement domain mapping.
"""

from importlib.metadata import version


def test_package_imports_cleanly() -> None:
    """Bare import of the package must succeed.

    A real bug we've already paid for in this family: records-analyser's
    package was un-importable for weeks because no test ever exercised
    the bare import statement. Smoke alarm.
    """
    import video_analyser  # noqa: F401

    assert video_analyser.__version__


def test_version_matches_installed_package() -> None:
    """The package's own __version__ must match the installed metadata.

    Drift trap: code has historically hardcoded version literals while the
    package version moved on, with tests "passing" because both sides were
    wrong in the same way. Pin to importlib.metadata once, here, so any
    future drift between code and packaging shows up loudly.
    """
    import video_analyser

    assert video_analyser.__version__ == version("video-analyser")


def test_unknown_coco_label_falls_back_to_object() -> None:
    """Unknown COCO labels must map to PresentationElement.OBJECT, not raise.

    The v0.7.0 refactor moved domain-mapping (COCO label -> presentation
    element) out of the deleted in-house detector and into a dedicated
    post-processing layer. The contract: every label string maps to
    *some* element; we never crash because image-analyser's underlying
    object detector returned a label our table doesn't know about.
    Equivalent to the family standard "unknown extension routed to
    unrecognised" invariant other analysers carry.
    """
    from video_analyser.analysis.presentation_classifier import (
        PresentationElement,
        classify,
    )

    assert classify("absolutely_not_a_coco_label_xyz") == PresentationElement.OBJECT


def test_visual_analyzer_uses_image_analyser_native_types() -> None:
    """VisualAnalysisResult must hold image-analyser's native schema types.

    Regression guard against re-introducing the CaptionResult/OCRResult
    adapter classes that were deleted in v0.7.0. If a future change adds
    a thin wrapper class with the same shape, this test will fail because
    the field annotation will no longer reference image_analyser.schemas.
    """
    import typing as t

    from image_analyser.schemas import Caption, Ocr

    from video_analyser.analysis.visual_analyzer import ExtractedFrame

    caption_field = ExtractedFrame.model_fields["caption_result"]
    ocr_field = ExtractedFrame.model_fields["ocr_result"]

    # The annotation is `Caption | None` / `Ocr | None`. Walk the union args.
    caption_args = t.get_args(caption_field.annotation)
    ocr_args = t.get_args(ocr_field.annotation)

    assert Caption in caption_args, (
        f"caption_result must be Caption | None; got {caption_field.annotation}"
    )
    assert Ocr in ocr_args, (
        f"ocr_result must be Ocr | None; got {ocr_field.annotation}"
    )
