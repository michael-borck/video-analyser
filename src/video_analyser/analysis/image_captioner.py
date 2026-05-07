"""Image captioning for frame analysis using BLIP-2 and similar models or APIs.

This module provides image captioning capabilities for extracted frames
to generate descriptive text about visual content. Supports both local
models (BLIP-2) and API-based captioning (Claude, GPT-4V, Gemini).
"""

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel
from transformers import (
    AutoProcessor,  # type: ignore
    Blip2ForConditionalGeneration,  # type: ignore
    Blip2Processor,  # type: ignore
    BlipForConditionalGeneration,  # type: ignore
    BlipProcessor,  # type: ignore
)

from video_analyser.analysis.error_handling import (
    ModelInitializationError,
    validate_image,
    with_retry,
)
from video_analyser.core.exceptions import ErrorCode, VideoProcessingError
from video_analyser.utils.config import get_config

logger = logging.getLogger(__name__)

# Suppress some transformers warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


class CaptionResult(BaseModel):
    """Result of image captioning operation."""

    caption: str
    confidence: float  # Model confidence (0-1)
    processing_time: float  # Time taken for captioning (seconds)
    model_used: str
    tokens_generated: int
    cost_estimate: float | None = None  # Cost in USD (for API calls)
    alternative_captions: list[str] = []  # Multiple caption candidates


class ImageCaptioner:
    """Image captioning using local models or API services."""

    def __init__(self, config: Any = None):
        """Initialize image captioner with configuration."""
        self.config = config or get_config()
        self.backend = self.config.visual_analysis.captioning_backend

        # API captioner (lazy initialization)
        self.api_captioner = None

        # Local model components
        self.model = None
        self.processor = None
        self.device = self._determine_device()

        logger.info(
            f"ImageCaptioner initialized: backend={self.backend}, device={self.device}"
        )

    def _determine_device(self) -> str:
        """Determine the best device for inference."""
        config_device = self.config.visual_analysis.captioning_device.lower()

        if config_device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("CUDA available, using GPU for image captioning")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon
                logger.info(
                    "MPS available, using Apple Silicon GPU for image captioning"
                )
            else:
                device = "cpu"
                logger.info("Using CPU for image captioning")
        else:
            device = config_device
            logger.info(f"Using configured device: {device}")

        return device

    @with_retry(max_attempts=3, delay=2.0)
    def _load_model(self) -> tuple[Any, Any]:  # type: ignore[misc]
        """Load captioning model and processor if not already loaded."""
        if self.model is None or self.processor is None:
            model_name = self.config.visual_analysis.captioning_model
            logger.info(f"Loading image captioning model: {model_name}")

            try:
                # Determine model type and load appropriate classes
                if "blip2" in model_name.lower():
                    # BLIP-2 models
                    self.processor = Blip2Processor.from_pretrained(model_name)
                    self.model = Blip2ForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16
                        if self.device == "cuda"
                        else torch.float32,
                    )
                elif "blip" in model_name.lower():
                    # Original BLIP models
                    self.processor = BlipProcessor.from_pretrained(model_name)
                    self.model = BlipForConditionalGeneration.from_pretrained(
                        model_name
                    )
                else:
                    # Try generic auto loading
                    self.processor = AutoProcessor.from_pretrained(model_name)
                    self.model = BlipForConditionalGeneration.from_pretrained(
                        model_name
                    )

                # Move model to device
                self.model = self.model.to(self.device)
                self.model.eval()  # Set to evaluation mode

                logger.info(f"Successfully loaded {model_name} on {self.device}")

            except Exception as e:
                error_msg = (
                    f"Failed to load image captioning model {model_name}: {str(e)}"
                )
                logger.error(error_msg)
                raise ModelInitializationError(
                    message=error_msg,
                    model_name=model_name,
                    details={"device": self.device},
                    cause=e,
                ) from e

        # Type ignore because transformers models are untyped
        return self.model, self.processor  # type: ignore[return-value]

    def caption_image(
        self,
        image_path: Path | None = None,
        image_array: Any = None,
        pil_image: Image.Image | None = None,
        prompt: str | None = None,
        max_length: int | None = None,
        temperature: float | None = None,
        num_beams: int = 4,
        num_return_sequences: int = 1,
    ) -> CaptionResult:
        """
        Generate caption for an image.

        Args:
            image_path: Path to image file
            image_array: Numpy array or tensor representation of image
            pil_image: PIL Image object
            prompt: Optional text prompt to guide captioning
            max_length: Maximum caption length in tokens
            temperature: Sampling temperature for generation
            num_beams: Number of beams for beam search
            num_return_sequences: Number of alternative captions to generate

        Returns:
            CaptionResult with generated caption and metadata

        Raises:
            VideoProcessingError: If captioning fails
            ValueError: If no image input is provided
        """
        import time

        start_time = time.time()

        # Validate inputs
        if not any([image_path, image_array is not None, pil_image]):
            raise ValueError(
                "Must provide one of: image_path, image_array, or pil_image"
            )

        # Route to API if configured
        if self.backend == "api":
            return self._caption_with_api(
                image_path=image_path,
                image_array=image_array,
                pil_image=pil_image,
            )

        # Otherwise use local model

        if not self.config.visual_analysis.enable_captioning:
            # Return placeholder if captioning is disabled
            return CaptionResult(
                caption="Image captioning disabled",
                confidence=0.0,
                processing_time=0.0,
                model_used="none",
                tokens_generated=0,
                alternative_captions=[],
            )

        # Use config defaults if parameters not provided
        if max_length is None:
            max_length = self.config.visual_analysis.max_caption_length
        if temperature is None:
            temperature = self.config.visual_analysis.caption_temperature

        logger.debug(
            f"Generating caption with max_length={max_length}, temperature={temperature}"
        )

        try:
            # Load model and processor
            model, processor = self._load_model()

            # Prepare image
            if pil_image is not None:
                # Validate PIL image
                image_array = np.array(pil_image)
                validated_array = validate_image(image_array, "PIL image")
                image = Image.fromarray(validated_array)
            elif image_path is not None:
                if not image_path.exists():
                    raise VideoProcessingError(
                        message=f"Image file not found: {image_path}",
                        error_code=ErrorCode.FILE_NOT_FOUND,
                        file_path=image_path,
                    )
                # Use validate_image to handle image loading and validation
                validated_array = validate_image(image_path, f"image file {image_path}")
                image = Image.fromarray(validated_array)
            else:
                # Validate and convert numpy array to PIL Image
                validated_array = validate_image(image_array, "numpy array")
                image = Image.fromarray(validated_array)

            # Prepare inputs
            if prompt:
                # Conditional captioning with prompt
                inputs = processor(image, prompt, return_tensors="pt").to(self.device)
            else:
                # Unconditional captioning
                inputs = processor(image, return_tensors="pt").to(self.device)

            # Generate captions
            with torch.no_grad():
                # Adjust generation parameters based on model type
                generation_kwargs: dict[str, Any] = {
                    "max_length": max_length,
                    "num_beams": num_beams,
                    "num_return_sequences": min(num_return_sequences, num_beams),
                    "do_sample": temperature > 1.0
                    if temperature is not None
                    else False,
                    "early_stopping": True,
                }

                # Add temperature if applicable
                if temperature is not None and temperature > 1.0:
                    generation_kwargs["temperature"] = temperature

                # Add pad_token_id if available
                if hasattr(processor, "tokenizer") and hasattr(
                    processor.tokenizer, "pad_token_id"
                ):
                    pad_token_id = processor.tokenizer.pad_token_id
                    if pad_token_id is not None:
                        generation_kwargs["pad_token_id"] = pad_token_id

                outputs = model.generate(**inputs, **generation_kwargs)

            # Decode captions
            generated_texts = processor.batch_decode(outputs, skip_special_tokens=True)

            # Extract main caption and alternatives
            if generated_texts:
                main_caption = generated_texts[0].strip()
                alternative_captions = (
                    [text.strip() for text in generated_texts[1:]]
                    if len(generated_texts) > 1
                    else []
                )
            else:
                main_caption = "Unable to generate caption"
                alternative_captions = []

            # Calculate confidence (simplified heuristic based on beam scores)
            # For now, use a simple heuristic based on caption length and content
            confidence = self._estimate_caption_confidence(main_caption)

            processing_time = time.time() - start_time
            tokens_generated = len(outputs[0]) if len(outputs) > 0 else 0

            result = CaptionResult(
                caption=main_caption,
                confidence=confidence,
                processing_time=processing_time,
                model_used=self.config.visual_analysis.captioning_model,
                tokens_generated=tokens_generated,
                alternative_captions=alternative_captions,
            )

            logger.debug(
                f"Caption generated: '{main_caption}' "
                f"(confidence: {confidence:.3f}, processing time: {processing_time:.1f}s)"
            )

            return result

        except VideoProcessingError:
            # Re-raise VideoProcessingError as-is to preserve original error code
            raise
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = (
                f"Image captioning failed after {processing_time:.1f}s: {str(e)}"
            )
            logger.error(error_msg)

            if "CUDA out of memory" in str(e):
                error_code = ErrorCode.INSUFFICIENT_MEMORY
                details = {
                    "device": self.device,
                    "model": self.config.visual_analysis.captioning_model,
                }
            elif "No module named" in str(e) or "cannot import" in str(e):
                error_code = ErrorCode.MISSING_DEPENDENCY
                details = {"missing_module": str(e)}
            else:
                error_code = (
                    ErrorCode.FRAME_EXTRACTION_FAILED
                )  # Generic visual processing error
                details = {"processing_time": processing_time}

            raise VideoProcessingError(
                message=error_msg,
                error_code=error_code,
                details=details,
                cause=e,
            ) from e

    def _estimate_caption_confidence(self, caption: str) -> float:
        """Estimate caption confidence based on heuristics."""
        if not caption or caption.lower() in [
            "unable to generate caption",
            "image captioning disabled",
        ]:
            return 0.0

        # Simple heuristics for confidence estimation
        confidence = 0.5  # Base confidence

        # Length-based adjustments
        if 10 <= len(caption) <= 50:
            confidence += 0.2  # Good length
        elif len(caption) < 5:
            confidence -= 0.3  # Too short
        elif len(caption) > 100:
            confidence -= 0.1  # Potentially verbose

        # Content-based adjustments
        meaningful_words = [
            "person",
            "people",
            "man",
            "woman",
            "child",
            "building",
            "room",
            "table",
            "chair",
            "screen",
            "computer",
            "presentation",
            "slide",
            "text",
            "image",
            "standing",
            "sitting",
            "holding",
            "wearing",
            "looking",
            "speaking",
        ]

        found_meaningful = sum(
            1 for word in meaningful_words if word in caption.lower()
        )
        confidence += min(found_meaningful * 0.05, 0.2)

        # Avoid generic/vague captions
        generic_phrases = ["a picture of", "an image of", "a photo of", "this is"]
        if any(phrase in caption.lower() for phrase in generic_phrases):
            confidence -= 0.1

        return max(0.0, min(1.0, confidence))

    def caption_batch(
        self,
        images: list[Path | Image.Image],
        prompts: list[str] | None = None,
        **kwargs: Any,
    ) -> list[CaptionResult]:
        """
        Generate captions for a batch of images.

        Args:
            images: List of image paths or PIL Images
            prompts: Optional list of prompts (one per image)
            **kwargs: Additional arguments passed to caption_image

        Returns:
            List of CaptionResult objects
        """
        if prompts and len(prompts) != len(images):
            raise ValueError("Number of prompts must match number of images")

        results = []
        batch_size = self.config.visual_analysis.caption_batch_size

        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            batch_prompts = (
                prompts[i : i + batch_size] if prompts else [None] * len(batch_images)
            )

            for image, prompt in zip(batch_images, batch_prompts, strict=True):
                try:
                    if isinstance(image, Path):
                        result = self.caption_image(
                            image_path=image, prompt=prompt, **kwargs
                        )
                    else:
                        result = self.caption_image(
                            pil_image=image, prompt=prompt, **kwargs
                        )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to caption image {image}: {e}")
                    # Add failed result
                    results.append(
                        CaptionResult(
                            caption=f"Caption failed: {str(e)}",
                            confidence=0.0,
                            processing_time=0.0,
                            model_used=self.config.visual_analysis.captioning_model,
                            tokens_generated=0,
                            alternative_captions=[],
                        )
                    )

        return results  # type: ignore[return-value]

    def get_supported_models(self) -> list[str]:
        """Get list of supported captioning models."""
        return [
            "Salesforce/blip2-opt-2.7b",
            "Salesforce/blip2-opt-6.7b",
            "Salesforce/blip2-flan-t5-xl",
            "Salesforce/blip-image-captioning-base",
            "Salesforce/blip-image-captioning-large",
        ]

    def cleanup(self):
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

            # Clear CUDA cache if using GPU
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Image captioning model resources cleaned up")

    def _caption_with_api(
        self,
        image_path: Path | None = None,
        image_array: Any = None,
        pil_image: Image.Image | None = None,
    ) -> CaptionResult:
        """
        Caption image using API service.

        Args:
            image_path: Path to image file
            image_array: Numpy array representation of image
            pil_image: PIL Image object

        Returns:
            CaptionResult with generated caption
        """
        # Lazy initialization of API captioner
        if self.api_captioner is None:
            try:
                from video_analyser.analysis.api_image_captioner import APIImageCaptioner

                self.api_captioner = APIImageCaptioner(config=self.config)
                logger.info(
                    f"Initialized API captioner: {self.config.visual_analysis.api_provider}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize API captioner: {e}")
                raise VideoProcessingError(
                    message=f"API captioner initialization failed: {e}",
                    error_code=ErrorCode.FRAME_EXTRACTION_FAILED,
                ) from e

        # Prepare image for API
        if image_path:
            image_input = image_path
        elif pil_image:
            image_input = pil_image
        elif image_array is not None:
            # Convert numpy array to PIL Image
            import numpy as np

            if isinstance(image_array, np.ndarray):
                # Convert BGR to RGB if needed
                # Type: ignore for numpy array shape which is dynamic
                arr_shape: tuple[int, ...] = image_array.shape  # type: ignore[assignment]
                # Type: ignore for len() on potentially unknown type
                if len(arr_shape) == 3 and arr_shape[2] == 3:  # type: ignore[arg-type]
                    # Type annotation to help with numpy array indexing
                    # type: ignore for numpy array slicing which has dynamic typing
                    image_array_rgb = image_array[  # type: ignore[var-annotated]
                        :, :, ::-1
                    ]
                    image_input = Image.fromarray(image_array_rgb)  # type: ignore[arg-type]
                else:
                    # Type: ignore because numpy arrays are partially typed
                    image_input = Image.fromarray(image_array)  # type: ignore[arg-type]
            else:
                image_input = image_array
        else:
            raise ValueError("No valid image input provided")

        # Call API captioner
        try:
            api_result = self.api_captioner.caption_image(image_input)

            # Convert to CaptionResult format
            return CaptionResult(
                caption=api_result.caption,
                confidence=api_result.confidence,
                processing_time=api_result.processing_time,
                model_used=f"{api_result.provider}:{api_result.model}",
                tokens_generated=api_result.tokens_used or 0,
                cost_estimate=api_result.cost_estimate,
                alternative_captions=[],
            )

        except Exception as e:
            logger.error(f"API captioning failed: {e}")
            raise VideoProcessingError(
                message=f"API captioning failed: {e}",
                error_code=ErrorCode.FRAME_EXTRACTION_FAILED,
            ) from e


def create_image_captioner(config: Any = None) -> ImageCaptioner:
    """Create a new ImageCaptioner instance.

    Args:
        config: Optional configuration object

    Returns:
        Configured ImageCaptioner instance
    """
    return ImageCaptioner(config=config)
