"""Speech transcription using Faster-Whisper with word-level timestamps.

This module provides speech-to-text functionality with detailed timing information
for video analysis applications. Uses Faster-Whisper for improved performance on
Apple Silicon and other platforms.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, TypedDict

import torch
from faster_whisper import WhisperModel
from pydantic import BaseModel

from video_analyser.core.audio_extractor import AudioInfo
from video_analyser.core.exceptions import (
    AudioProcessingError,
    ErrorCode,
    VideoProcessingError,
)
from video_analyser.utils.config import get_config

logger = logging.getLogger(__name__)

# Suppress some warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Type aliases for Whisper results
class WhisperWord(TypedDict):
    word: str
    start: float
    end: float


class WhisperSegment(TypedDict):
    id: int
    text: str
    start: float
    end: float
    words: list[WhisperWord]


class WhisperResult(TypedDict):
    text: str
    segments: list[WhisperSegment]
    language: str


class WordTimestamp(BaseModel):
    """Word-level timestamp information."""

    word: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    confidence: float  # Confidence score (0.0 to 1.0)


class Segment(BaseModel):
    """Transcription segment with timing and word-level details."""

    id: int
    text: str
    start: float
    end: float
    avg_logprob: float
    no_speech_prob: float
    words: list[WordTimestamp] = []
    language: str | None = None


class LanguageDetectionResult(BaseModel):
    """Language detection result with confidence scores."""

    detected_language: str
    confidence: float
    all_probabilities: dict[str, float] = {}
    detection_method: str  # "whisper", "manual", "config"


class TranscriptionResult(BaseModel):
    """Complete transcription result with metadata."""

    text: str
    segments: list[Segment]
    language: str
    language_probability: float
    duration: float
    model_used: str
    word_count: int
    processing_time: float
    language_detection: LanguageDetectionResult | None = None

    def get_words_at_time(
        self, timestamp: float, tolerance: float = 0.5
    ) -> list[WordTimestamp]:
        """Get words spoken around a specific timestamp."""
        words: list[WordTimestamp] = []
        for segment in self.segments:
            for word in segment.words:
                # Check if timestamp is within or near the word's time range
                if (
                    word.start <= timestamp <= word.end
                    or abs(word.start - timestamp) <= tolerance
                    or abs(word.end - timestamp) <= tolerance
                ):
                    words.append(
                        WordTimestamp(
                            word=word.word,
                            start=word.start,
                            end=word.end,
                            confidence=1.0,  # Default confidence
                        )
                    )
        return words

    def get_text_between_times(self, start_time: float, end_time: float) -> str:
        """Get transcribed text between two timestamps."""
        text_parts: list[str] = []
        for segment in self.segments:
            # Check if segment overlaps with time range
            if segment.end >= start_time and segment.start <= end_time:
                if segment.words:
                    # Use word-level timing for precise extraction
                    segment_words: list[str] = []
                    for word in segment.words:
                        if word.end >= start_time and word.start <= end_time:
                            segment_words.append(word.word)
                    if segment_words:
                        text_parts.append(" ".join(segment_words))
                else:
                    # Fallback to segment-level text
                    text_parts.append(segment.text.strip())

        return " ".join(text_parts).strip()

    def get_speaking_rate(
        self, start_time: float = 0.0, end_time: float | None = None
    ) -> float:
        """Calculate speaking rate (words per minute) for a time range."""
        if end_time is None:
            end_time = self.duration

        # Count words in the time range
        word_count = 0
        for segment in self.segments:
            if segment.words:
                for word in segment.words:
                    if word.start >= start_time and word.end <= end_time:
                        word_count += 1
            else:
                # Fallback: estimate based on segment overlap
                if segment.end >= start_time and segment.start <= end_time:
                    # Rough estimate: segment word count proportional to overlap
                    overlap_start = max(segment.start, start_time)
                    overlap_end = min(segment.end, end_time)
                    overlap_duration = overlap_end - overlap_start
                    segment_duration = segment.end - segment.start
                    if segment_duration > 0:
                        word_count += len(segment.text.split()) * (
                            overlap_duration / segment_duration
                        )

        # Calculate WPM
        duration_minutes = (end_time - start_time) / 60.0
        if duration_minutes <= 0:
            return 0.0

        return word_count / duration_minutes

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            "text": self.text,
            "segments": [segment.model_dump() for segment in self.segments],
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "model_used": self.model_used,
            "word_count": self.word_count,
            "processing_time": self.processing_time,
        }


class FasterWhisperTranscriber:
    """Faster-Whisper-based speech transcription with word-level timestamps.

    Uses faster-whisper for better performance on Apple Silicon and other platforms.
    """

    def __init__(self, config: Any = None):
        """Initialize transcriber with configuration."""
        self.config = config or get_config()
        self.model = None
        self.device = self._determine_device()
        self.compute_type = self._determine_compute_type()

        logger.info(
            f"FasterWhisperTranscriber initialized with device: {self.device}, "
            f"compute_type: {self.compute_type}"
        )

    def _determine_device(self) -> str:
        """Determine the best device for inference."""
        config_device = self.config.transcription.device.lower()

        if config_device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("CUDA available, using GPU for transcription")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "cpu"  # Use CPU for Apple Silicon with faster-whisper
                logger.info(
                    "Apple Silicon detected, using CPU with faster-whisper "
                    "(better compatibility than MPS)"
                )
            else:
                device = "cpu"
                logger.info("Using CPU for transcription")
        else:
            device = config_device
            logger.info(f"Using configured device: {device}")

        return device

    def _determine_compute_type(self) -> str:
        """Determine the best compute type for faster-whisper."""
        if self.device == "cuda":
            return "float16"  # Use half precision for CUDA GPUs
        else:
            return "int8"  # Use int8 quantization for CPU (faster and less memory)

    def _load_model(self) -> WhisperModel:
        """Load Faster-Whisper model if not already loaded."""
        if self.model is None:
            model_name = self.config.transcription.model
            logger.info(f"Loading Faster-Whisper model: {model_name}")

            try:
                # Remove 'whisper-' prefix if present
                model_size = model_name.replace("whisper-", "")
                self.model = WhisperModel(
                    model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=None,
                    local_files_only=False,
                )
                logger.info(
                    f"Successfully loaded {model_name} on {self.device} "
                    f"with compute_type={self.compute_type}"
                )
            except Exception as e:
                error_msg = (
                    f"Failed to load Faster-Whisper model {model_name}: {str(e)}"
                )
                logger.error(error_msg)
                raise VideoProcessingError(
                    message=error_msg,
                    error_code=ErrorCode.MISSING_DEPENDENCY,
                    details={
                        "model": model_name,
                        "device": self.device,
                        "compute_type": self.compute_type,
                    },
                    cause=e,
                ) from e

        return self.model

    def transcribe_audio(
        self,
        audio_info: AudioInfo,
        language: str | None = None,
        temperature: float | None = None,
        word_timestamps: bool | None = None,
        auto_detect_language: bool | None = None,
        min_detection_confidence: float = 0.5,
    ) -> TranscriptionResult:
        """
        Transcribe audio file using Whisper with enhanced language detection.

        Args:
            audio_info: AudioInfo object with audio file details
            language: Language code (e.g., 'en', 'es') or language name ('english', 'spanish')
                     for manual override. None for automatic detection.
            temperature: Sampling temperature (0.0 to 1.0)
            word_timestamps: Whether to include word-level timestamps
            auto_detect_language: Force automatic detection even if language is configured.
                                 If None, uses config setting.
            min_detection_confidence: Minimum confidence for automatic language detection

        Returns:
            TranscriptionResult with detailed transcription, timing info, and language detection results

        Raises:
            AudioProcessingError: If transcription fails
            VideoProcessingError: If model loading fails
            ValueError: If manually specified language is invalid
        """
        import time

        start_time = time.time()

        # Validate audio file
        if not audio_info.file_path.exists():
            raise AudioProcessingError(
                message=f"Audio file not found: {audio_info.file_path}",
                error_code=ErrorCode.FILE_NOT_FOUND,
                file_path=audio_info.file_path,
            )

        # Use config defaults if parameters not provided
        if temperature is None:
            temperature = self.config.transcription.temperature
        if word_timestamps is None:
            word_timestamps = self.config.transcription.word_timestamps
        if auto_detect_language is None:
            auto_detect_language = self.config.transcription.language == "auto"

        # Handle language detection and validation
        language_detection_result = None
        final_language = None

        # If language is manually specified, validate it
        if language is not None and language.lower() != "auto":
            try:
                final_language = self.validate_language_override(language)
                language_detection_result = LanguageDetectionResult(
                    detected_language=final_language,
                    confidence=1.0,  # Manual override has full confidence
                    all_probabilities={final_language: 1.0},
                    detection_method="manual",
                )
                logger.info(
                    f"Using manual language override: {final_language} ({self.get_language_name(final_language)})"
                )
            except ValueError as e:
                raise AudioProcessingError(
                    message=f"Invalid language specification: {str(e)}",
                    error_code=ErrorCode.INVALID_INPUT,
                    file_path=audio_info.file_path,
                    details={"provided_language": language},
                ) from e

        # If auto-detection is enabled or no language specified
        elif auto_detect_language or language is None or language.lower() == "auto":
            try:
                language_detection_result = self.detect_language(audio_info)

                # Check if detection confidence is sufficient
                if language_detection_result.confidence >= min_detection_confidence:
                    final_language = language_detection_result.detected_language
                    logger.info(
                        f"Auto-detected language: {final_language} "
                        f"({self.get_language_name(final_language)}) "
                        f"with confidence {language_detection_result.confidence:.3f}"
                    )
                else:
                    logger.warning(
                        f"Low confidence language detection: {language_detection_result.detected_language} "
                        f"(confidence: {language_detection_result.confidence:.3f}, "
                        f"threshold: {min_detection_confidence}). Proceeding with auto-detection."
                    )
                    final_language = (
                        None  # Let Whisper auto-detect during transcription
                    )
            except AudioProcessingError:
                logger.warning(
                    "Language detection failed, falling back to automatic detection during transcription"
                )
                final_language = None

        # Use config language as fallback
        else:
            config_language = self.config.transcription.language
            if config_language != "auto":
                final_language = config_language
                language_detection_result = LanguageDetectionResult(
                    detected_language=final_language,
                    confidence=1.0,
                    all_probabilities={final_language: 1.0},
                    detection_method="config",
                )
                logger.info(f"Using configured language: {final_language}")

        logger.info(f"Transcribing audio: {audio_info.file_path.name}")
        logger.info(
            f"Parameters: language={final_language}, temperature={temperature}, word_timestamps={word_timestamps}"
        )

        try:
            # Load model
            model = self._load_model()

            # Transcribe with Faster-Whisper
            # Note: faster_whisper is not well-typed, so we use type: ignore
            result = model.transcribe(  # type: ignore[no-untyped-call]
                str(audio_info.file_path),
                language=final_language,
                temperature=temperature or 0.0,  # Convert None to 0.0
                word_timestamps=word_timestamps
                or False,  # Enable word-level timestamps
            )

            # Process result - result is a tuple (generator, info)
            segments_result, info = result  # type: ignore[misc]
            segments: list[Segment] = []
            for i, segment_data in enumerate(segments_result):  # type: ignore[var-annotated]
                # Process word-level timestamps if available
                words: list[WordTimestamp] = []
                if (
                    word_timestamps
                    and hasattr(segment_data, "words")  # type: ignore[arg-type]
                    and segment_data.words  # type: ignore[attr-defined]
                ):
                    for word_data in segment_data.words:  # type: ignore[attr-defined]
                        words.append(
                            WordTimestamp(
                                word=str(word_data.word).strip(),  # type: ignore[attr-defined]
                                start=float(word_data.start),  # type: ignore[attr-defined]
                                end=float(word_data.end),  # type: ignore[attr-defined]
                                confidence=1.0,  # Faster-whisper doesn't provide per-word confidence
                            )
                        )

                segment = Segment(
                    id=i,
                    text=str(segment_data.text).strip(),  # type: ignore[attr-defined]
                    start=float(segment_data.start),  # type: ignore[attr-defined]
                    end=float(segment_data.end),  # type: ignore[attr-defined]
                    avg_logprob=float(
                        getattr(segment_data, "avg_logprob", 0.0)  # type: ignore[arg-type]
                    ),  # Faster-whisper may not provide this
                    no_speech_prob=float(
                        getattr(segment_data, "no_speech_prob", 0.0)  # type: ignore[arg-type]
                    ),  # Faster-whisper may not provide this
                    words=words,
                    language=info.language,  # type: ignore[attr-defined]
                )
                segments.append(segment)

            # Calculate word count
            total_words = 0
            for segment in segments:
                if segment.words:
                    total_words += len(segment.words)
                else:
                    # Fallback: count words in text
                    total_words += len(segment.text.split())

            processing_time = time.time() - start_time

            # Get language detection result from faster-whisper
            transcription_language: str = info.language or "unknown"  # type: ignore[attr-defined]
            transcription_prob = float(
                getattr(info, "language_probability", 0.0)  # type: ignore[arg-type]
            )

            # If we didn't have pre-detection, create one with Faster-Whisper's detection
            if language_detection_result is None:
                language_detection_result = LanguageDetectionResult(
                    detected_language=transcription_language,  # type: ignore[arg-type]
                    confidence=transcription_prob,
                    all_probabilities={transcription_language: transcription_prob},
                    detection_method="faster_whisper",
                )

            # If Faster-Whisper detected a different language than our pre-detection
            # and it wasn't a manual override, update with Faster-Whisper's result
            elif (
                language_detection_result.detected_language != transcription_language
                and language_detection_result.detection_method != "manual"
            ):
                language_detection_result = LanguageDetectionResult(
                    detected_language=transcription_language,  # type: ignore[arg-type]
                    confidence=transcription_prob,
                    all_probabilities={transcription_language: transcription_prob},
                    detection_method="faster_whisper_transcription",
                )

            # Build full transcription text from segments
            full_text = " ".join(segment.text for segment in segments).strip()

            transcription_result = TranscriptionResult(
                text=full_text,
                segments=segments,
                language=transcription_language,  # type: ignore[arg-type]
                language_probability=transcription_prob,
                duration=audio_info.duration,
                model_used=self.config.transcription.model,
                word_count=total_words,
                processing_time=processing_time,
                language_detection=language_detection_result,
            )

            logger.info(
                f"Transcription complete: {len(segments)} segments, "
                f"{total_words} words, {processing_time:.1f}s processing time"
            )

            return transcription_result

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Transcription failed after {processing_time:.1f}s: {str(e)}"
            logger.error(error_msg)

            if "CUDA out of memory" in str(e):
                error_code = ErrorCode.INSUFFICIENT_MEMORY
                details = {
                    "device": self.device,
                    "model": self.config.transcription.model,
                }
            elif "No module named" in str(e) or "cannot import" in str(e):
                error_code = ErrorCode.MISSING_DEPENDENCY
                details = {"missing_module": str(e)}
            else:
                error_code = ErrorCode.AUDIO_EXTRACTION_FAILED
                details = {"processing_time": processing_time}

            raise AudioProcessingError(
                message=error_msg,
                error_code=error_code,
                file_path=audio_info.file_path,
                details=details,
                cause=e,
            ) from e

    def transcribe_from_path(
        self, audio_path: Path | str, language: str | None = None, **kwargs: Any
    ) -> TranscriptionResult:
        """
        Convenience method to transcribe audio from a file path.

        This method probes the audio file to get metadata, then calls transcribe_audio.

        Args:
            audio_path: Path to audio file
            language: Language code or None for auto-detection
            **kwargs: Additional arguments passed to transcribe_audio

        Returns:
            TranscriptionResult with transcription and metadata
        """
        import ffmpeg

        audio_path = Path(audio_path)
        logger.info(f"Transcribing audio from path: {audio_path}")

        # Probe audio file to get metadata
        try:
            probe = ffmpeg.probe(str(audio_path))
            audio_stream = next(
                s for s in probe["streams"] if s["codec_type"] == "audio"
            )

            duration = float(probe["format"]["duration"])
            sample_rate = int(audio_stream["sample_rate"])
            channels = int(audio_stream["channels"])
            size_mb = int(probe["format"]["size"]) / (1024 * 1024)

            # Create AudioInfo object
            audio_info = AudioInfo(
                file_path=audio_path,
                duration=duration,
                sample_rate=sample_rate,
                channels=channels,
                size_mb=size_mb,
                format=audio_stream["codec_name"],
            )

            # Call the main transcription method
            return self.transcribe_audio(audio_info, language=language, **kwargs)

        except Exception as e:
            logger.error(f"Failed to probe or transcribe audio file: {e}")
            raise AudioProcessingError(
                message=f"Failed to transcribe audio from path: {e}",
                error_code=ErrorCode.AUDIO_EXTRACTION_FAILED,
                file_path=audio_path,
                cause=e,
            ) from e

    def transcribe_audio_segment(
        self, audio_info: AudioInfo, start_time: float, end_time: float, **kwargs: Any
    ) -> TranscriptionResult:
        """
        Transcribe a specific segment of audio.

        Args:
            audio_info: AudioInfo object with audio file details
            start_time: Start time in seconds
            end_time: End time in seconds
            **kwargs: Additional parameters for transcribe_audio

        Returns:
            TranscriptionResult for the audio segment
        """
        # Note: This is a simplified implementation
        # In practice, you might want to extract the audio segment first
        # For now, we'll transcribe the full audio and filter results

        full_result = self.transcribe_audio(audio_info, **kwargs)

        # Filter segments to the requested time range
        filtered_segments: list[Segment] = []
        segment_id = 0

        for segment in full_result.segments:
            # Check if segment overlaps with requested time range
            if segment.end >= start_time and segment.start <= end_time:
                # Create filtered segment
                filtered_words = []
                if segment.words:
                    filtered_words = [
                        word
                        for word in segment.words
                        if word.end >= start_time and word.start <= end_time
                    ]

                # Adjust segment timing relative to the start_time
                adjusted_segment = Segment(
                    id=segment_id,
                    text=segment.text,
                    start=max(segment.start - start_time, 0.0),
                    end=min(segment.end - start_time, end_time - start_time),
                    avg_logprob=segment.avg_logprob,
                    no_speech_prob=segment.no_speech_prob,
                    words=[
                        WordTimestamp(
                            word=word.word,
                            start=word.start - start_time,
                            end=word.end - start_time,
                            confidence=word.confidence,
                        )
                        for word in filtered_words
                    ],
                    language=segment.language,
                )
                filtered_segments.append(adjusted_segment)
                segment_id += 1

        # Create filtered result
        filtered_text = " ".join(str(segment.text) for segment in filtered_segments)
        word_count = sum(len(segment.words) for segment in filtered_segments)

        return TranscriptionResult(
            text=filtered_text,
            segments=filtered_segments,
            language=full_result.language,
            language_probability=full_result.language_probability,
            duration=end_time - start_time,
            model_used=full_result.model_used,
            word_count=word_count,
            processing_time=full_result.processing_time,
        )

    def detect_language(
        self,
        audio_info: AudioInfo,
        sample_duration: float = 30.0,
    ) -> LanguageDetectionResult:
        """
        Detect the language of the audio using Faster-Whisper's built-in detection.

        Args:
            audio_info: AudioInfo object with audio file details
            sample_duration: Duration in seconds to sample for detection (max 30s)

        Returns:
            LanguageDetectionResult with detected language and confidence

        Raises:
            AudioProcessingError: If language detection fails
        """
        logger.info(f"Detecting language for audio: {audio_info.file_path.name}")

        if not audio_info.file_path.exists():
            raise AudioProcessingError(
                message=f"Audio file not found: {audio_info.file_path}",
                error_code=ErrorCode.FILE_NOT_FOUND,
                file_path=audio_info.file_path,
            )

        try:
            model = self._load_model()

            # Faster-whisper's detect_language requires transcription
            # We'll use a short sample to detect language efficiently
            sample_duration = min(sample_duration, 30.0)  # Limit to 30 seconds

            result = model.transcribe(  # type: ignore[no-untyped-call]
                str(audio_info.file_path),
                language=None,  # Auto-detect
            )

            # Process result tuple
            segments_result, info = result  # type: ignore[misc]

            # Consume the generator to get language info
            # We don't need the actual segments for language detection
            for _ in segments_result:  # type: ignore[var-annotated]
                pass

            detected_language: str = info.language or "unknown"  # type: ignore[attr-defined]
            confidence = float(
                getattr(info, "language_probability", 0.9)  # type: ignore[arg-type]
            )

            logger.info(
                f"Language detection complete: {detected_language} "
                f"(confidence: {confidence:.3f})"
            )

            return LanguageDetectionResult(
                detected_language=detected_language,
                confidence=confidence,
                all_probabilities={detected_language: confidence},
                detection_method="faster_whisper",
            )

        except Exception as e:
            error_msg = f"Language detection failed: {str(e)}"
            logger.error(error_msg)

            raise AudioProcessingError(
                message=error_msg,
                error_code=ErrorCode.AUDIO_EXTRACTION_FAILED,
                file_path=audio_info.file_path,
                details={"sample_duration": sample_duration},
                cause=e,
            ) from e

    def validate_language_override(self, language: str) -> str:
        """
        Validate and normalize a manually specified language code.

        Args:
            language: Language code to validate (e.g., 'en', 'english', 'English')

        Returns:
            Normalized language code

        Raises:
            ValueError: If language is not supported
        """
        if not language:
            raise ValueError("Language code cannot be empty")

        # Normalize input
        language = language.lower().strip()

        # Check if it's already a valid code
        supported_codes = self.get_supported_languages()
        if language in supported_codes:
            return language

        # Try to match full language names to codes
        language_mapping = {
            "english": "en",
            "spanish": "es",
            "french": "fr",
            "german": "de",
            "italian": "it",
            "portuguese": "pt",
            "russian": "ru",
            "chinese": "zh",
            "japanese": "ja",
            "korean": "ko",
            "arabic": "ar",
            "hindi": "hi",
            "dutch": "nl",
            "swedish": "sv",
            "danish": "da",
            "norwegian": "no",
            "finnish": "fi",
            "polish": "pl",
            "czech": "cs",
            "hungarian": "hu",
            "turkish": "tr",
            "hebrew": "he",
            "thai": "th",
            "vietnamese": "vi",
            "indonesian": "id",
            "malay": "ms",
            "ukrainian": "uk",
            "greek": "el",
            "bulgarian": "bg",
            "romanian": "ro",
            "serbian": "sr",
            "croatian": "hr",
            "slovenian": "sl",
            "slovak": "sk",
            "lithuanian": "lt",
            "latvian": "lv",
            "estonian": "et",
            "macedonian": "mk",
            "albanian": "sq",
            "welsh": "cy",
            "irish": "ga",
            "scots": "gd",
            "breton": "br",
            "basque": "eu",
            "catalan": "ca",
            "galician": "gl",
            "maltese": "mt",
            "icelandic": "is",
            "faroese": "fo",
            "armenian": "hy",
            "georgian": "ka",
            "azerbaijani": "az",
            "kazakh": "kk",
            "kyrgyz": "ky",
            "tajik": "tg",
            "turkmen": "tk",
            "uzbek": "uz",
            "mongolian": "mn",
            "tibetan": "bo",
            "burmese": "my",
            "khmer": "km",
            "lao": "lo",
            "bengali": "bn",
            "punjabi": "pa",
            "gujarati": "gu",
            "oriya": "or",
            "tamil": "ta",
            "telugu": "te",
            "kannada": "kn",
            "malayalam": "ml",
            "sinhalese": "si",
            "nepali": "ne",
            "marathi": "mr",
            "urdu": "ur",
            "persian": "fa",
            "pashto": "ps",
            "dari": "prs",
            "kurdish": "ku",
            "amharic": "am",
            "tigrinya": "ti",
            "somali": "so",
            "swahili": "sw",
            "yoruba": "yo",
            "igbo": "ig",
            "hausa": "ha",
            "zulu": "zu",
            "xhosa": "xh",
            "afrikaans": "af",
            "hawaiian": "haw",
            "maori": "mi",
            "tagalog": "tl",
            "cebuano": "ceb",
            "javanese": "jv",
            "sundanese": "su",
            "balinese": "ban",
            "latin": "la",
            "esperanto": "eo",
            "interlingua": "ia",
            "volapük": "vo",
        }

        if language in language_mapping:
            return language_mapping[language]

        # If still not found, suggest closest matches
        close_matches = [
            lang for lang in language_mapping if language in lang or lang in language
        ]
        if close_matches:
            suggestion = close_matches[0]
            suggested_code = language_mapping[suggestion]
            raise ValueError(
                f"Language '{language}' not supported. Did you mean '{suggestion}' ({suggested_code})? "
                f"Supported languages: {', '.join(supported_codes[:20])}..."
            )

        raise ValueError(
            f"Language '{language}' not supported. "
            f"Supported languages: {', '.join(supported_codes[:20])}..."
        )

    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes."""
        # Whisper supported languages
        return [
            "en",
            "zh",
            "de",
            "es",
            "ru",
            "ko",
            "fr",
            "ja",
            "pt",
            "tr",
            "pl",
            "ca",
            "nl",
            "ar",
            "sv",
            "it",
            "id",
            "hi",
            "fi",
            "vi",
            "he",
            "uk",
            "el",
            "ms",
            "cs",
            "ro",
            "da",
            "hu",
            "ta",
            "no",
            "th",
            "ur",
            "hr",
            "bg",
            "lt",
            "la",
            "mi",
            "ml",
            "cy",
            "sk",
            "te",
            "fa",
            "lv",
            "bn",
            "sr",
            "az",
            "sl",
            "kn",
            "et",
            "mk",
            "br",
            "eu",
            "is",
            "hy",
            "ne",
            "mn",
            "bs",
            "kk",
            "sq",
            "sw",
            "gl",
            "mr",
            "pa",
            "si",
            "km",
            "sn",
            "yo",
            "so",
            "af",
            "oc",
            "ka",
            "be",
            "tg",
            "sd",
            "gu",
            "am",
            "yi",
            "lo",
            "uz",
            "fo",
            "ht",
            "ps",
            "tk",
            "nn",
            "mt",
            "sa",
            "lb",
            "my",
            "bo",
            "tl",
            "mg",
            "as",
            "tt",
            "haw",
            "ln",
            "ha",
            "ba",
            "jw",
            "su",
        ]

    def get_language_name(self, language_code: str) -> str:
        """
        Get the full language name from a language code.

        Args:
            language_code: Two-letter language code

        Returns:
            Full language name
        """
        language_names = {
            "en": "English",
            "zh": "Chinese",
            "de": "German",
            "es": "Spanish",
            "ru": "Russian",
            "ko": "Korean",
            "fr": "French",
            "ja": "Japanese",
            "pt": "Portuguese",
            "tr": "Turkish",
            "pl": "Polish",
            "ca": "Catalan",
            "nl": "Dutch",
            "ar": "Arabic",
            "sv": "Swedish",
            "it": "Italian",
            "id": "Indonesian",
            "hi": "Hindi",
            "fi": "Finnish",
            "vi": "Vietnamese",
            "he": "Hebrew",
            "uk": "Ukrainian",
            "el": "Greek",
            "ms": "Malay",
            "cs": "Czech",
            "ro": "Romanian",
            "da": "Danish",
            "hu": "Hungarian",
            "ta": "Tamil",
            "no": "Norwegian",
            "th": "Thai",
            "ur": "Urdu",
            "hr": "Croatian",
            "bg": "Bulgarian",
            "lt": "Lithuanian",
            "la": "Latin",
            "mi": "Maori",
            "ml": "Malayalam",
            "cy": "Welsh",
            "sk": "Slovak",
            "te": "Telugu",
            "fa": "Persian",
            "lv": "Latvian",
            "bn": "Bengali",
            "sr": "Serbian",
            "az": "Azerbaijani",
            "sl": "Slovenian",
            "kn": "Kannada",
            "et": "Estonian",
            "mk": "Macedonian",
            "br": "Breton",
            "eu": "Basque",
            "is": "Icelandic",
            "hy": "Armenian",
            "ne": "Nepali",
            "mn": "Mongolian",
            "bs": "Bosnian",
            "kk": "Kazakh",
            "sq": "Albanian",
            "sw": "Swahili",
            "gl": "Galician",
            "mr": "Marathi",
            "pa": "Punjabi",
            "si": "Sinhala",
            "km": "Khmer",
            "sn": "Shona",
            "yo": "Yoruba",
            "so": "Somali",
            "af": "Afrikaans",
            "oc": "Occitan",
            "ka": "Georgian",
            "be": "Belarusian",
            "tg": "Tajik",
            "sd": "Sindhi",
            "gu": "Gujarati",
            "am": "Amharic",
            "yi": "Yiddish",
            "lo": "Lao",
            "uz": "Uzbek",
            "fo": "Faroese",
            "ht": "Haitian Creole",
            "ps": "Pashto",
            "tk": "Turkmen",
            "nn": "Norwegian Nynorsk",
            "mt": "Maltese",
            "sa": "Sanskrit",
            "lb": "Luxembourgish",
            "my": "Myanmar",
            "bo": "Tibetan",
            "tl": "Tagalog",
            "mg": "Malagasy",
            "as": "Assamese",
            "tt": "Tatar",
            "haw": "Hawaiian",
            "ln": "Lingala",
            "ha": "Hausa",
            "ba": "Bashkir",
            "jw": "Javanese",
            "su": "Sundanese",
        }

        return language_names.get(language_code, language_code.upper())

    def cleanup(self):
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None

            # Clear CUDA cache if using GPU
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Faster-Whisper model resources cleaned up")


# Backward compatibility alias
WhisperTranscriber = FasterWhisperTranscriber


def create_transcriber(config: Any = None) -> FasterWhisperTranscriber:
    """Create a new FasterWhisperTranscriber instance.

    Uses Faster-Whisper for improved performance on Apple Silicon and other platforms.

    Args:
        config: Optional configuration object

    Returns:
        Configured FasterWhisperTranscriber instance
    """
    return FasterWhisperTranscriber(config=config)
