"""Object detection for identifying presentation elements.

This module provides object detection capabilities specifically tailored
for presentation and video content analysis, identifying elements like
slides, charts, diagrams, text blocks, and other presentation components.
"""

import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel

from video_analyser.analysis.error_handling import (
    ModelInitializationError,
    handle_corrupt_frame,
    validate_image,
    with_retry,
)
from video_analyser.utils.config import get_config

logger = logging.getLogger(__name__)


class PresentationElement(str, Enum):
    """Types of presentation elements that can be detected."""

    SLIDE = "slide"
    CHART = "chart"
    DIAGRAM = "diagram"
    TEXT_BLOCK = "text_block"
    TABLE = "table"
    IMAGE = "image"
    CODE_BLOCK = "code_block"
    BULLET_LIST = "bullet_list"
    TITLE = "title"
    LOGO = "logo"
    GRAPH = "graph"
    SCREENSHOT = "screenshot"
    PERSON = "person"  # For video presentations
    WHITEBOARD = "whiteboard"
    ANNOTATION = "annotation"


class DetectedObject(BaseModel):
    """Information about a detected object."""

    element_type: PresentationElement
    confidence: float  # 0.0 to 1.0
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    center: tuple[float, float]  # Normalized center coordinates (0-1)
    area_ratio: float  # Ratio of object area to frame area

    # Additional metadata
    attributes: dict[str, Any] = {}  # Element-specific attributes

    def get_description(self) -> str:
        """Get human-readable description of the detected object."""
        return f"{self.element_type.value.replace('_', ' ').title()} (confidence: {self.confidence:.1%})"


class ObjectDetectionResult(BaseModel):
    """Results from object detection on a frame."""

    detected_objects: list[DetectedObject]
    total_objects: int
    processing_time: float
    frame_width: int
    frame_height: int

    # Detection statistics
    element_counts: dict[str, int]  # Count by element type
    high_confidence_objects: int  # Objects with confidence > 0.8
    average_confidence: float

    # Layout analysis
    layout_type: str  # e.g., "slide", "video", "mixed"
    has_presenter: bool
    dominant_element: str | None

    def get_presentation_elements(self) -> list[DetectedObject]:
        """Get only presentation-specific elements (exclude people, etc.)."""
        presentation_types = {
            PresentationElement.SLIDE,
            PresentationElement.CHART,
            PresentationElement.DIAGRAM,
            PresentationElement.TEXT_BLOCK,
            PresentationElement.TABLE,
            PresentationElement.CODE_BLOCK,
            PresentationElement.BULLET_LIST,
            PresentationElement.TITLE,
            PresentationElement.GRAPH,
        }
        return [
            obj
            for obj in self.detected_objects
            if obj.element_type in presentation_types
        ]

    def get_content_summary(self) -> dict[str, Any]:
        """Get summary of detected content."""
        presentation_elements = self.get_presentation_elements()

        return {
            "total_elements": len(presentation_elements),
            "has_slides": any(
                obj.element_type == PresentationElement.SLIDE
                for obj in self.detected_objects
            ),
            "has_charts": any(
                obj.element_type
                in [PresentationElement.CHART, PresentationElement.GRAPH]
                for obj in self.detected_objects
            ),
            "has_text": any(
                obj.element_type
                in [PresentationElement.TEXT_BLOCK, PresentationElement.TITLE]
                for obj in self.detected_objects
            ),
            "has_code": any(
                obj.element_type == PresentationElement.CODE_BLOCK
                for obj in self.detected_objects
            ),
            "layout_type": self.layout_type,
            "dominant_element": self.dominant_element,
        }


class ObjectDetector:
    """Object detection for presentation elements."""

    def __init__(self, config: Any = None):
        """Initialize object detector with configuration."""
        self.config = config or get_config()
        self.model = None
        self.device = None
        self._initialized = False

        # Detection settings
        self.confidence_threshold = (
            self.config.visual_analysis.object_detection_confidence
        )
        self.nms_threshold = 0.4  # Non-maximum suppression threshold

        logger.info("ObjectDetector initialized")

    def _ensure_initialized(self) -> None:
        """Ensure the model is loaded."""
        if not self._initialized:
            self._load_model()

    @with_retry(max_attempts=2, delay=1.0)
    def _load_model(self) -> None:
        """Load the object detection model."""
        try:
            import torch

            # Determine device
            if self.config.visual_analysis.object_detection_device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:
                self.device = self.config.visual_analysis.object_detection_device

            logger.info(f"Loading object detection model on device: {self.device}")

            # For now, we'll use a lightweight approach with OpenCV's DNN module
            # This can be replaced with YOLOv5 or other models later
            self._initialized = True

            logger.info("Object detection model loaded successfully")

        except Exception as e:
            error_msg = f"Failed to load object detection model: {str(e)}"
            logger.error(error_msg)
            raise ModelInitializationError(
                message=error_msg,
                model_name="object_detector",
                details={
                    "device": self.device if hasattr(self, "device") else "unknown"
                },
                cause=e,
            ) from e

    def detect_objects(
        self,
        image: Image.Image | NDArray[Any] | None = None,
        image_path: Path | None = None,
    ) -> ObjectDetectionResult:
        """
        Detect presentation elements in an image.

        Args:
            image: PIL Image or numpy array (RGB format)
            image_path: Path to image file (alternative to image)

        Returns:
            ObjectDetectionResult with detected objects

        Raises:
            ProcessingError: If detection fails
        """
        start_time = time.time()

        try:
            # Load and validate image
            image_array: NDArray[Any]
            if image_path:
                image_array = validate_image(image_path, f"image file {image_path}")
            elif image is not None:
                if isinstance(image, Image.Image):
                    image_array = np.array(image)
                else:
                    image_array = image
                image_array = validate_image(image_array, "input image")
            else:
                raise ValueError("Either image or image_path must be provided")

            # Handle corrupt frames
            maybe_corrupt_frame = handle_corrupt_frame(
                image_array, {"source": "object_detection"}
            )
            if maybe_corrupt_frame is None:
                logger.warning("Frame appears to be corrupted, returning empty results")
                return ObjectDetectionResult(
                    detected_objects=[],
                    total_objects=0,
                    processing_time=time.time() - start_time,
                    frame_width=0,
                    frame_height=0,
                    element_counts={},
                    high_confidence_objects=0,
                    average_confidence=0.0,
                    layout_type="unknown",
                    has_presenter=False,
                    dominant_element=None,
                )

            # Re-assign validated frame
            image_array = maybe_corrupt_frame

            height: int = image_array.shape[0]
            width: int = image_array.shape[1]

            # For now, use heuristic-based detection
            # This will be replaced with proper ML model
            detected_objects = self._detect_with_heuristics(image_array)

            # Calculate statistics
            element_counts: dict[str, int] = {}
            high_confidence_count = 0
            total_confidence = 0.0

            for obj in detected_objects:
                element_type = obj.element_type.value
                element_counts[element_type] = element_counts.get(element_type, 0) + 1

                if obj.confidence > 0.8:
                    high_confidence_count += 1

                total_confidence += obj.confidence

            average_confidence = (
                total_confidence / len(detected_objects) if detected_objects else 0.0
            )

            # Analyze layout
            layout_type = self._analyze_layout(detected_objects, width, height)
            has_presenter = any(
                obj.element_type == PresentationElement.PERSON
                for obj in detected_objects
            )

            # Find dominant element
            dominant_element: str | None = None
            if element_counts:
                dominant_element = max(
                    element_counts.items(), key=lambda x: tuple[str, int](x)[1]
                )[0]

            processing_time = time.time() - start_time

            return ObjectDetectionResult(
                detected_objects=detected_objects,
                total_objects=len(detected_objects),
                processing_time=processing_time,
                frame_width=width,
                frame_height=height,
                element_counts=element_counts,
                high_confidence_objects=high_confidence_count,
                average_confidence=average_confidence,
                layout_type=layout_type,
                has_presenter=has_presenter,
                dominant_element=dominant_element,
            )

        except Exception as e:
            error_msg = f"Object detection failed: {str(e)}"
            logger.error(error_msg)

            # Return empty result instead of raising
            return ObjectDetectionResult(
                detected_objects=[],
                total_objects=0,
                processing_time=time.time() - start_time,
                frame_width=0,
                frame_height=0,
                element_counts={},
                high_confidence_objects=0,
                average_confidence=0.0,
                layout_type="unknown",
                has_presenter=False,
                dominant_element=None,
            )

    def _detect_with_heuristics(self, image: NDArray[Any]) -> list[DetectedObject]:
        """
        Detect objects using heuristic methods.

        This is a placeholder implementation that uses traditional CV techniques.
        In production, this would be replaced with a proper ML model.
        """
        detected_objects: list[DetectedObject] = []
        height: int = image.shape[0]
        width: int = image.shape[1]

        # Convert to grayscale for analysis
        gray: NDArray[Any] = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 1. Detect text blocks using morphological operations
        text_blocks = self._detect_text_blocks(gray)
        for bbox in text_blocks:
            x1, y1, x2, y2 = bbox
            center_x: float = (x1 + x2) / 2 / width
            center_y: float = (y1 + y2) / 2 / height
            area_ratio: float = ((x2 - x1) * (y2 - y1)) / (width * height)

            # Determine if it's a title based on position and size
            is_title: bool = y1 < height * 0.2 and area_ratio > 0.1

            detected_objects.append(
                DetectedObject(
                    element_type=PresentationElement.TITLE
                    if is_title
                    else PresentationElement.TEXT_BLOCK,
                    confidence=0.85 if is_title else 0.75,
                    bbox=bbox,
                    center=(center_x, center_y),
                    area_ratio=area_ratio,
                    attributes={"is_title": is_title},
                )
            )

        # 2. Detect charts/graphs using edge detection
        charts = self._detect_charts(image)
        for bbox in charts:
            x1, y1, x2, y2 = bbox
            center_x_chart: float = (x1 + x2) / 2 / width
            center_y_chart: float = (y1 + y2) / 2 / height
            area_ratio_chart: float = ((x2 - x1) * (y2 - y1)) / (width * height)

            detected_objects.append(
                DetectedObject(
                    element_type=PresentationElement.CHART,
                    confidence=0.7,
                    bbox=bbox,
                    center=(center_x_chart, center_y_chart),
                    area_ratio=area_ratio_chart,
                    attributes={"chart_type": "unknown"},
                )
            )

        # 3. Detect slide layout
        if self._is_slide_layout(image):
            # Add a slide detection covering most of the frame
            detected_objects.append(
                DetectedObject(
                    element_type=PresentationElement.SLIDE,
                    confidence=0.9,
                    bbox=(
                        int(width * 0.05),
                        int(height * 0.05),
                        int(width * 0.95),
                        int(height * 0.95),
                    ),
                    center=(0.5, 0.5),
                    area_ratio=0.9,
                    attributes={"slide_type": "standard"},
                )
            )

        # 4. Detect code blocks (monospace text areas)
        code_blocks = self._detect_code_blocks(gray)
        for bbox in code_blocks:
            x1, y1, x2, y2 = bbox
            center_x_code: float = (x1 + x2) / 2 / width
            center_y_code: float = (y1 + y2) / 2 / height
            area_ratio_code: float = ((x2 - x1) * (y2 - y1)) / (width * height)

            detected_objects.append(
                DetectedObject(
                    element_type=PresentationElement.CODE_BLOCK,
                    confidence=0.8,
                    bbox=bbox,
                    center=(center_x_code, center_y_code),
                    area_ratio=area_ratio_code,
                    attributes={"has_syntax_highlighting": False},
                )
            )

        return detected_objects

    def _detect_text_blocks(self, gray: NDArray[Any]) -> list[tuple[int, int, int, int]]:
        """Detect text blocks using morphological operations."""
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Apply morphological operations to merge text regions
        kernel: NDArray[Any] = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilated: NDArray[Any] = cv2.dilate(binary, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        text_blocks: list[tuple[int, int, int, int]] = []
        height: int = gray.shape[0]
        width: int = gray.shape[1]
        min_area: float = width * height * 0.01  # Minimum 1% of frame area

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Filter by area and aspect ratio
            if area > min_area and w > h * 0.5:  # Text is usually wider than tall
                text_blocks.append((x, y, x + w, y + h))

        return text_blocks

    def _detect_charts(self, image: NDArray[Any]) -> list[tuple[int, int, int, int]]:
        """Detect charts and graphs using edge detection and line detection."""
        gray: NDArray[Any] = (
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        )

        # Edge detection
        edges: NDArray[Any] = cv2.Canny(gray, 50, 150)

        # Line detection
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10
        )

        charts: list[tuple[int, int, int, int]] = []

        # Check if we have valid line detection results
        if lines is None or len(lines) == 0:  # type: ignore[reportUnnecessaryComparison]
            return charts

        # Group lines into potential chart regions
        # This is a simplified approach - in practice, would use more sophisticated methods
        height: int = gray.shape[0]
        width: int = gray.shape[1]

        # Look for rectangular regions with many lines
        # (This is a placeholder - real implementation would be more complex)
        if len(lines) > 10:
            # Assume center region might be a chart
            chart_bbox = (
                int(width * 0.2),
                int(height * 0.2),
                int(width * 0.8),
                int(height * 0.8),
            )
            charts.append(chart_bbox)

        return charts

    def _detect_code_blocks(self, gray: NDArray[Any]) -> list[tuple[int, int, int, int]]:
        """Detect code blocks (areas with monospace text patterns)."""
        # Look for regions with consistent horizontal lines (code indentation)
        # This is a simplified heuristic

        # Apply edge detection to find text regions
        edges: NDArray[Any] = cv2.Canny(gray, 50, 150)

        # Look for rectangular regions with regular patterns
        kernel: NDArray[Any] = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 3))
        dilated: NDArray[Any] = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        code_blocks: list[tuple[int, int, int, int]] = []
        height: int = gray.shape[0]
        width: int = gray.shape[1]
        min_area: float = width * height * 0.05  # Minimum 5% of frame area

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Code blocks are usually rectangular with specific aspect ratios
            if area > min_area and h > 50 and w > h * 1.5:
                code_blocks.append((x, y, x + w, y + h))

        return code_blocks

    def _is_slide_layout(self, image: NDArray[Any]) -> bool:
        """Determine if the image has a typical slide layout."""
        height: int = image.shape[0]
        width: int = image.shape[1]

        # Convert to grayscale
        gray: NDArray[Any] = (
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        )

        # Check for consistent background (slides usually have uniform backgrounds)
        # Calculate standard deviation of pixel values
        std_dev: float = float(np.std(gray))

        # Check edges - slides usually have clear borders
        edges: NDArray[Any] = cv2.Canny(gray, 50, 150)
        edge_density: float = float(np.sum(edges > 0)) / float(edges.size)

        # Slides typically have:
        # - Low overall standard deviation (uniform background)
        # - Low edge density (not too busy)
        # - Rectangular aspect ratio close to 16:9 or 4:3
        aspect_ratio: float = width / height
        is_standard_aspect: bool = (
            abs(aspect_ratio - 16 / 9) < 0.2 or abs(aspect_ratio - 4 / 3) < 0.2
        )

        return bool(std_dev < 60 and edge_density < 0.1 and is_standard_aspect)

    def _analyze_layout(
        self, objects: list[DetectedObject], _width: int, _height: int
    ) -> str:
        """Analyze the overall layout type based on detected objects."""
        if not objects:
            return "empty"

        # Check for slide
        has_slide = any(
            obj.element_type == PresentationElement.SLIDE for obj in objects
        )
        if has_slide:
            return "slide"

        # Check for video presentation (person + content)
        has_person = any(
            obj.element_type == PresentationElement.PERSON for obj in objects
        )
        has_content = any(
            obj.element_type
            in [
                PresentationElement.TEXT_BLOCK,
                PresentationElement.CHART,
                PresentationElement.DIAGRAM,
            ]
            for obj in objects
        )

        if has_person and has_content:
            return "video_presentation"
        elif has_person:
            return "video"

        # Check for whiteboard
        has_whiteboard = any(
            obj.element_type == PresentationElement.WHITEBOARD for obj in objects
        )
        if has_whiteboard:
            return "whiteboard"

        # Default to document if we have text/charts
        if has_content:
            return "document"

        return "unknown"

    def cleanup(self):
        """Clean up model resources."""
        self.model = None
        self._initialized = False
        logger.info("ObjectDetector resources cleaned up")


def create_object_detector(config: Any = None) -> ObjectDetector:
    """Create a new ObjectDetector instance.

    Args:
        config: Optional configuration object

    Returns:
        Configured ObjectDetector instance
    """
    return ObjectDetector(config=config)
