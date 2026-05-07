"""Tests for error handling utilities in visual analysis components."""

import logging
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from video_analyser.analysis.error_handling import (
    ErrorRecoveryContext,
    ImageValidationError,
    ModelInitializationError,
    handle_corrupt_frame,
    safe_model_inference,
    validate_image,
    validate_model_inputs,
    with_retry,
)
from video_analyser.core.exceptions import ErrorCode


class TestImageValidation:
    """Test image validation functionality."""

    def test_validate_image_with_valid_numpy_array(self):
        """Test validation with valid numpy array."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = validate_image(image, "test image")
        assert isinstance(result, np.ndarray)
        assert result.shape == (480, 640, 3)
        assert result.dtype == np.uint8

    def test_validate_image_with_pil_image(self):
        """Test validation with PIL Image."""
        pil_image = Image.new("RGB", (640, 480), color=(255, 0, 0))
        result = validate_image(pil_image, "PIL image")
        assert isinstance(result, np.ndarray)
        assert result.shape == (480, 640, 3)
        assert result.dtype == np.uint8

    def test_validate_image_with_file_path(self, tmp_path):
        """Test validation with file path."""
        # Create a test image file
        image_path = tmp_path / "test_image.png"
        test_image = Image.new("RGB", (640, 480), color=(0, 255, 0))
        test_image.save(image_path)

        result = validate_image(image_path, "image file")
        assert isinstance(result, np.ndarray)
        assert result.shape == (480, 640, 3)
        assert result.dtype == np.uint8

    def test_validate_image_with_grayscale(self):
        """Test validation converts grayscale to RGB."""
        gray_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        result = validate_image(gray_image, "grayscale image")
        assert result.shape == (480, 640, 3)

    def test_validate_image_with_rgba(self):
        """Test validation converts RGBA to RGB."""
        rgba_image = np.random.randint(0, 255, (480, 640, 4), dtype=np.uint8)
        result = validate_image(rgba_image, "RGBA image")
        assert result.shape == (480, 640, 3)

    def test_validate_image_with_float_values(self):
        """Test validation converts float images to uint8."""
        float_image = np.random.rand(480, 640, 3).astype(np.float32)
        result = validate_image(float_image, "float image")
        assert result.dtype == np.uint8
        assert result.max() <= 255
        assert result.min() >= 0

    def test_validate_image_with_batch_dimension(self):
        """Test validation handles batch dimension."""
        batch_image = np.random.randint(0, 255, (1, 480, 640, 3), dtype=np.uint8)
        result = validate_image(batch_image, "batch image")
        assert result.shape == (480, 640, 3)

    def test_validate_image_with_invalid_dimensions(self):
        """Test validation fails for invalid dimensions."""
        invalid_image = np.random.randint(0, 255, (10,), dtype=np.uint8)
        with pytest.raises(ImageValidationError) as exc_info:
            validate_image(invalid_image, "1D array")
        assert "Invalid image dimensions" in str(exc_info.value)

    def test_validate_image_with_empty_array(self):
        """Test validation fails for empty arrays."""
        empty_image = np.array([])
        with pytest.raises(ImageValidationError) as exc_info:
            validate_image(empty_image, "empty array")
        assert "Image is empty" in str(exc_info.value)

    def test_validate_image_with_too_small_image(self):
        """Test validation fails for images that are too small."""
        tiny_image = np.random.randint(0, 255, (5, 5, 3), dtype=np.uint8)
        with pytest.raises(ImageValidationError) as exc_info:
            validate_image(tiny_image, "tiny image")
        assert "Image too small" in str(exc_info.value)

    def test_validate_image_with_nonexistent_file(self):
        """Test validation fails for nonexistent files."""
        nonexistent_path = Path("/nonexistent/image.png")
        with pytest.raises(ImageValidationError) as exc_info:
            validate_image(nonexistent_path, "nonexistent file")
        assert "Image file not found" in str(exc_info.value)

    def test_validate_image_with_corrupt_file(self, tmp_path):
        """Test validation fails for corrupt files."""
        # Create a corrupt image file
        corrupt_path = tmp_path / "corrupt.png"
        corrupt_path.write_bytes(b"not an image")

        with pytest.raises(ImageValidationError) as exc_info:
            validate_image(corrupt_path, "corrupt file")
        assert "Failed to load image file" in str(exc_info.value)


class TestRetryDecorator:
    """Test retry decorator functionality."""

    def test_retry_success_on_first_attempt(self):
        """Test function succeeds on first attempt."""
        call_count = 0

        @with_retry(max_attempts=3, delay=0.1)
        def successful_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_function()
        assert result == "success"
        assert call_count == 1

    def test_retry_success_after_failures(self):
        """Test function succeeds after initial failures."""
        call_count = 0

        @with_retry(max_attempts=3, delay=0.1)
        def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = eventually_successful()
        assert result == "success"
        assert call_count == 3

    def test_retry_fails_after_max_attempts(self):
        """Test function fails after max attempts."""
        call_count = 0

        @with_retry(max_attempts=3, delay=0.1)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError) as exc_info:
            always_fails()
        assert "Always fails" in str(exc_info.value)
        assert call_count == 3

    def test_retry_with_specific_exceptions(self):
        """Test retry only catches specified exceptions."""
        call_count = 0

        @with_retry(max_attempts=3, delay=0.1, exceptions=(ValueError,))
        def raises_type_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Wrong type")

        with pytest.raises(TypeError):
            raises_type_error()
        assert call_count == 1  # Should not retry TypeError

    def test_retry_with_exponential_backoff(self):
        """Test retry with exponential backoff."""
        start_time = time.time()

        @with_retry(max_attempts=3, delay=0.1, backoff=2.0)
        def always_fails():
            raise ValueError("Fail")

        with pytest.raises(ValueError):
            always_fails()

        elapsed = time.time() - start_time
        # Should have delays of 0.1 and 0.2 seconds (0.3 total)
        assert elapsed >= 0.25  # Allow some tolerance


class TestSafeModelInference:
    """Test safe model inference wrapper."""

    def test_safe_inference_success(self):
        """Test successful inference."""

        def func(x):
            return x * 2

        wrapped = safe_model_inference(func, fallback_result="fallback")
        result = wrapped(5)
        assert result == 10

    def test_safe_inference_with_exception(self, caplog):
        """Test inference with exception returns fallback."""

        def failing_inference(_x):
            raise RuntimeError("Model failed")

        wrapped = safe_model_inference(
            failing_inference,
            fallback_result="fallback",
            operation_name="test inference",
        )

        result = wrapped(5)
        assert result == "fallback"
        assert "test inference failed" in caplog.text
        assert "Model failed" in caplog.text


class TestValidateModelInputs:
    """Test model input validation decorator."""

    def test_validate_inputs_success(self):
        """Test successful input validation."""

        @validate_model_inputs(
            image=lambda x: np.array(x) if isinstance(x, list) else x
        )
        def process_image(image):
            return image.shape

        result = process_image(image=[[1, 2], [3, 4]])
        assert result == (2, 2)

    def test_validate_inputs_failure(self):
        """Test input validation failure."""

        @validate_model_inputs(value=lambda x: float(x))
        def process_value(value):
            return value * 2

        with pytest.raises(ValueError) as exc_info:
            process_value(value="not a number")
        assert "Validation failed for parameter 'value'" in str(exc_info.value)


class TestErrorRecoveryContext:
    """Test error recovery context manager."""

    def test_recovery_context_success(self, caplog):
        """Test context manager with successful operation."""
        with (
            caplog.at_level(logging.INFO),
            ErrorRecoveryContext("test operation") as ctx,
        ):
            pass

        assert ctx.error is None
        assert "Starting test operation" in caplog.text
        assert "Completed test operation" in caplog.text

    def test_recovery_context_with_error(self, caplog):
        """Test context manager with error."""
        with pytest.raises(ValueError), ErrorRecoveryContext("test operation"):
            raise ValueError("Test error")

        assert "test operation failed" in caplog.text
        assert "Test error" in caplog.text

    def test_recovery_context_with_fallback(self, caplog):
        """Test context manager with fallback action."""
        fallback_called = False

        def fallback():
            nonlocal fallback_called
            fallback_called = True

        with (
            caplog.at_level(logging.INFO),
            pytest.raises(ValueError),
            ErrorRecoveryContext("test operation", fallback_action=fallback),
        ):
            raise ValueError("Test error")

        assert fallback_called
        assert "Executing fallback for test operation" in caplog.text

    def test_recovery_context_suppress_errors(self):
        """Test context manager suppressing errors."""
        with ErrorRecoveryContext("test operation", suppress_errors=True) as ctx:
            raise ValueError("Test error")

        assert ctx.error is not None
        assert isinstance(ctx.error, ValueError)

    def test_recovery_context_fallback_error(self, caplog):
        """Test context manager when fallback also fails."""

        def failing_fallback():
            raise RuntimeError("Fallback failed")

        with (
            pytest.raises(ValueError),
            ErrorRecoveryContext("test operation", fallback_action=failing_fallback),
        ):
            raise ValueError("Original error")

        assert "Fallback action failed" in caplog.text


class TestHandleCorruptFrame:
    """Test corrupt frame handling."""

    def test_handle_valid_frame(self):
        """Test handling of valid frame."""
        valid_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = handle_corrupt_frame(valid_frame)
        assert result is not None
        # Result may be the same frame or slightly processed
        assert result is not None
        assert result.shape == valid_frame.shape

    def test_handle_black_frame(self, caplog):
        """Test handling of completely black frame."""
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = handle_corrupt_frame(black_frame)
        assert result is None
        assert "Frame is completely black" in caplog.text

    def test_handle_white_frame(self, caplog):
        """Test handling of completely white frame."""
        white_frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        result = handle_corrupt_frame(white_frame)
        assert result is None
        assert "Frame is completely white" in caplog.text

    def test_handle_noisy_frame(self, caplog):
        """Test handling of noisy frame."""
        # Create a very noisy frame
        noisy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add extreme noise
        for _ in range(5):
            noise = np.random.normal(0, 100, noisy_frame.shape)
            noisy_frame = np.clip(noisy_frame + noise, 0, 255).astype(np.uint8)

        handle_corrupt_frame(noisy_frame)
        # Result may be None or denoised, depending on noise level
        # Either noise detected or denoising applied
        assert (
            "Frame appears to be corrupted" in caplog.text
            or "Applied denoising" in caplog.text
        )

    def test_handle_invalid_frame(self, caplog):
        """Test handling of invalid frame data."""
        invalid_frame = "not an array"
        result = handle_corrupt_frame(invalid_frame)
        assert result is None
        assert "Frame validation failed" in caplog.text


class TestModelInitializationError:
    """Test ModelInitializationError exception."""

    def test_model_initialization_error_creation(self):
        """Test creating ModelInitializationError."""
        cause = RuntimeError("Original error")
        error = ModelInitializationError(
            message="Failed to load model",
            model_name="test_model",
            details={"device": "cuda"},
            cause=cause,
        )

        assert "Failed to load model" in str(error)
        assert error.error_code == ErrorCode.MODEL_LOADING_FAILED
        assert error.details["model_name"] == "test_model"
        assert error.details["device"] == "cuda"
