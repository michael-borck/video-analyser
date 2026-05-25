"""Speech transcription for video-analyser — delegated to speech-analyser.

video-analyser used to run its own faster-whisper engine here, duplicating
speech-analyser's transcription. The engine now lives in speech-analyser (the
family's single transcription engine, deepened to word-level in 0.5.0); this module
just adapts its word-level output into the `TranscriptionResult` shape that video's
scene-aware `SpeechAnalyzer` consumes. The data models + their helper methods are
unchanged, so the downstream scene-correlation logic needs no changes.
"""

import logging
import time
import warnings
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from video_analyser.core.exceptions import AudioProcessingError, ErrorCode
from video_analyser.utils.config import get_config

logger = logging.getLogger(__name__)

# Suppress some warnings
warnings.filterwarnings("ignore", category=UserWarning)


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
    detection_method: str  # "whisper", "manual", "config", "speech-analyser"


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
                            confidence=word.confidence,
                        )
                    )
        return words

    def get_text_between_times(self, start_time: float, end_time: float) -> str:
        """Get transcribed text between two timestamps."""
        text_parts: list[str] = []
        for segment in self.segments:
            if segment.end >= start_time and segment.start <= end_time:
                if segment.words:
                    segment_words: list[str] = []
                    for word in segment.words:
                        if word.end >= start_time and word.start <= end_time:
                            segment_words.append(word.word)
                    if segment_words:
                        text_parts.append(" ".join(segment_words))
                else:
                    text_parts.append(segment.text.strip())

        return " ".join(text_parts).strip()

    def get_speaking_rate(
        self, start_time: float = 0.0, end_time: float | None = None
    ) -> float:
        """Calculate speaking rate (words per minute) for a time range."""
        if end_time is None:
            end_time = self.duration

        word_count = 0.0
        for segment in self.segments:
            if segment.words:
                for word in segment.words:
                    if word.start >= start_time and word.end <= end_time:
                        word_count += 1
            else:
                if segment.end >= start_time and segment.start <= end_time:
                    overlap_start = max(segment.start, start_time)
                    overlap_end = min(segment.end, end_time)
                    overlap_duration = overlap_end - overlap_start
                    segment_duration = segment.end - segment.start
                    if segment_duration > 0:
                        word_count += len(segment.text.split()) * (
                            overlap_duration / segment_duration
                        )

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


class SpeechTranscriber:
    """Transcribes audio via speech-analyser, adapting it to TranscriptionResult.

    The whisper engine lives in speech-analyser (the family's single transcription
    engine); this only maps its word-level output into the shape video's scene-aware
    SpeechAnalyzer expects.
    """

    def __init__(self, config: Any = None):
        self.config = config or get_config()

    def transcribe_from_path(
        self, audio_path: Path | str, language: str | None = None, **kwargs: Any
    ) -> TranscriptionResult:
        """Transcribe an audio file via speech-analyser.

        `language` is accepted for backward compatibility but speech-analyser
        auto-detects (it doesn't take a forced language); video's config default is
        "auto" anyway.
        """
        from speech_analyser import SpeechAnalyser, SpeechAnalyserError

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise AudioProcessingError(
                message=f"Audio file not found: {audio_path}",
                error_code=ErrorCode.FILE_NOT_FOUND,
                file_path=audio_path,
            )

        start = time.time()
        model_size = str(self.config.transcription.model).replace("whisper-", "").strip() or "base"
        logger.info(f"Transcribing via speech-analyser (model={model_size}): {audio_path.name}")

        try:
            audio = SpeechAnalyser(model_size=model_size).analyse(audio_path)
        except SpeechAnalyserError as e:
            raise AudioProcessingError(
                message=f"Transcription failed: {e}",
                error_code=ErrorCode.AUDIO_EXTRACTION_FAILED,
                file_path=audio_path,
                cause=e,
            ) from e

        lang = str(audio.get("language") or "unknown")
        segments: list[Segment] = []
        for i, seg in enumerate(audio.get("segments", [])):
            words = [
                WordTimestamp(
                    word=str(w["word"]).strip(),
                    start=float(w["start"]),
                    end=float(w["end"]),
                    confidence=float(w.get("probability", 1.0)),
                )
                for w in seg.get("words", [])
            ]
            segments.append(
                Segment(
                    id=i,
                    text=str(seg["text"]).strip(),
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    avg_logprob=0.0,
                    no_speech_prob=0.0,
                    words=words,
                    language=lang,
                )
            )

        total_words = sum(
            len(s.words) if s.words else len(s.text.split()) for s in segments
        )
        processing_time = time.time() - start
        logger.info(
            f"Transcription complete: {len(segments)} segments, {total_words} words, "
            f"{processing_time:.1f}s"
        )

        return TranscriptionResult(
            text=str(audio.get("transcript", "")),
            segments=segments,
            language=lang,
            language_probability=1.0,
            duration=float(audio.get("duration", 0.0)),
            model_used=f"speech-analyser:{model_size}",
            word_count=total_words,
            processing_time=processing_time,
            language_detection=LanguageDetectionResult(
                detected_language=lang,
                confidence=1.0,
                all_probabilities={lang: 1.0},
                detection_method="speech-analyser",
            ),
        )


# Backward-compatible aliases — pipeline_coordinator imports WhisperTranscriber.
WhisperTranscriber = SpeechTranscriber
FasterWhisperTranscriber = SpeechTranscriber


def create_transcriber(config: Any = None) -> SpeechTranscriber:
    """Create a SpeechTranscriber (transcription delegated to speech-analyser)."""
    return SpeechTranscriber(config=config)
