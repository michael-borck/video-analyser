"""Speech analysis for calculating speaking rate, pauses, and other speech metrics.

This module provides comprehensive speech analysis capabilities for video content,
focusing on speaking rate calculations per scene and overall speech quality metrics.
"""

import logging
from typing import Any

from pydantic import BaseModel

from video_analyser.analysis.transcriber import Segment, TranscriptionResult
from video_analyser.core.scene_detector import SceneDetectionResult
from video_analyser.utils.config import get_config

logger = logging.getLogger(__name__)


class SceneSpeechMetrics(BaseModel):
    """Speech metrics for a single scene."""

    scene_number: int
    start_time: float
    end_time: float
    duration: float

    # Word and speaking metrics
    word_count: int
    speaking_rate_wpm: float  # Words per minute

    # Content analysis
    text_content: str
    character_count: int
    sentence_count: int

    # Timing analysis
    actual_speaking_time: float  # Time with speech (excluding pauses)
    pause_time: float  # Time without speech
    speech_ratio: float  # Ratio of speaking time to total time

    # Quality metrics
    avg_confidence: float  # Average transcription confidence
    low_confidence_words: int  # Number of words below threshold

    # Filler word analysis
    filler_word_count: int  # Total filler words detected
    filler_word_rate: float  # Filler words per minute
    filler_words_detected: list[
        dict[str, Any]
    ]  # List of detected filler words with timestamps

    # Silence analysis
    silence_segments: list[dict[str, Any]]  # List of silence periods with timing
    total_silence_time: float  # Total time of silence gaps
    silence_ratio: float  # Ratio of silence to total scene duration
    average_silence_duration: float  # Average length of silence gaps
    longest_silence_duration: float  # Longest continuous silence period
    silence_count: int  # Number of distinct silence periods

    # Sentiment analysis
    sentiment_score: float  # Overall sentiment score (-1 to 1, negative to positive)
    sentiment_label: str  # Categorical sentiment (negative, neutral, positive)
    sentiment_confidence: float  # Confidence in sentiment prediction (0 to 1)
    emotional_keywords: list[
        dict[str, Any]
    ]  # List of detected emotional words with scores

    # Quality assessment and confidence scoring
    quality_score: float  # Overall transcription quality score (0 to 1)
    quality_category: str  # Quality category (excellent, good, fair, poor)
    confidence_issues: list[dict[str, Any]]  # List of identified confidence issues
    segment_confidence_scores: list[dict[str, Any]]  # Per-segment confidence analysis
    avg_segment_logprob: float  # Average log probability of segments
    no_speech_ratio: float  # Ratio of time classified as no-speech
    problematic_segments: list[dict[str, Any]]  # Segments with quality issues


class OverallSpeechMetrics(BaseModel):
    """Overall speech metrics across all scenes."""

    total_duration: float
    total_scenes: int

    # Aggregate word metrics
    total_word_count: int
    average_wpm: float
    median_wpm: float
    min_wpm: float
    max_wpm: float

    # Speech quality
    overall_speech_ratio: float
    total_speaking_time: float
    total_pause_time: float

    # Consistency metrics
    wpm_variance: float
    wpm_standard_deviation: float

    # Content metrics
    total_characters: int
    total_sentences: int
    average_words_per_sentence: float

    # Quality assessment
    overall_confidence: float
    low_confidence_percentage: float

    # Filler word analysis
    total_filler_words: int
    filler_word_rate: float  # Filler words per minute across all scenes
    most_common_filler_words: list[dict[str, Any]]  # Top filler words with counts

    # Silence analysis
    total_silence_time: float  # Total silence time across all scenes
    overall_silence_ratio: float  # Overall ratio of silence to total duration
    average_silence_per_scene: float  # Average silence time per scene
    longest_silence_overall: float  # Longest silence period across all scenes
    total_silence_segments: int  # Total number of silence periods

    # Sentiment analysis
    overall_sentiment_score: float  # Weighted average sentiment score across all scenes
    sentiment_distribution: dict[
        str, int
    ]  # Count of scenes by sentiment (negative, neutral, positive)
    most_positive_scene: int | None  # Scene number with highest positive sentiment
    most_negative_scene: int | None  # Scene number with highest negative sentiment
    emotional_intensity: (
        float  # Average absolute sentiment score (emotional engagement)
    )
    total_emotional_keywords: int  # Total count of emotional keywords detected

    # Quality assessment and confidence scoring
    overall_quality_score: float  # Overall transcription quality across all scenes
    quality_distribution: dict[str, int]  # Count of scenes by quality category
    average_confidence: float  # Average word-level confidence across all scenes
    low_quality_scenes: list[int]  # Scene numbers with quality issues
    critical_issues: list[dict[str, Any]]  # Critical quality problems identified
    transcription_reliability: float  # Overall reliability assessment (0 to 1)
    recommended_actions: list[str]  # Suggested improvements for quality issues


class SpeechAnalysisResult(BaseModel):
    """Complete speech analysis result."""

    scene_metrics: list[SceneSpeechMetrics]
    overall_metrics: OverallSpeechMetrics
    processing_time: float
    language: str
    model_used: str

    # Analysis metadata
    confidence_threshold: float
    target_wpm_range: list[int]

    def get_scene_metrics(self, scene_number: int) -> SceneSpeechMetrics | None:
        """Get metrics for a specific scene."""
        for metrics in self.scene_metrics:
            if metrics.scene_number == scene_number:
                return metrics
        return None

    def get_scenes_by_wpm_range(
        self, min_wpm: float, max_wpm: float
    ) -> list[SceneSpeechMetrics]:
        """Get scenes within a specific WPM range."""
        return [
            scene
            for scene in self.scene_metrics
            if min_wpm <= scene.speaking_rate_wpm <= max_wpm
        ]

    def get_slow_scenes(self) -> list[SceneSpeechMetrics]:
        """Get scenes below target WPM range."""
        min_target = self.target_wpm_range[0]
        return [
            scene
            for scene in self.scene_metrics
            if scene.speaking_rate_wpm < min_target
        ]

    def get_fast_scenes(self) -> list[SceneSpeechMetrics]:
        """Get scenes above target WPM range."""
        max_target = self.target_wpm_range[1]
        return [
            scene
            for scene in self.scene_metrics
            if scene.speaking_rate_wpm > max_target
        ]

    def get_optimal_scenes(self) -> list[SceneSpeechMetrics]:
        """Get scenes within target WPM range."""
        min_target, max_target = self.target_wpm_range
        return self.get_scenes_by_wpm_range(min_target, max_target)

    def get_scenes_with_high_filler_words(
        self, threshold_rate: float = 5.0
    ) -> list[SceneSpeechMetrics]:
        """Get scenes with high filler word rate (per minute)."""
        return [
            scene
            for scene in self.scene_metrics
            if scene.filler_word_rate > threshold_rate
        ]

    def get_scenes_with_low_filler_words(
        self, threshold_rate: float = 2.0
    ) -> list[SceneSpeechMetrics]:
        """Get scenes with low filler word rate (per minute)."""
        return [
            scene
            for scene in self.scene_metrics
            if scene.filler_word_rate <= threshold_rate
        ]

    def get_total_filler_words_by_type(self) -> dict[str, int]:
        """Get total count of each filler word type across all scenes."""
        filler_counts: dict[str, int] = {}
        for scene in self.scene_metrics:
            for filler in scene.filler_words_detected:
                normalized_word = filler["normalized"]
                filler_counts[normalized_word] = (
                    filler_counts.get(normalized_word, 0) + 1
                )
        return filler_counts

    def get_scenes_with_excessive_silence(
        self, threshold_ratio: float = 0.3
    ) -> list[SceneSpeechMetrics]:
        """Get scenes with high silence ratio."""
        return [
            scene
            for scene in self.scene_metrics
            if scene.silence_ratio > threshold_ratio
        ]

    def get_scenes_with_minimal_silence(
        self, threshold_ratio: float = 0.1
    ) -> list[SceneSpeechMetrics]:
        """Get scenes with low silence ratio."""
        return [
            scene
            for scene in self.scene_metrics
            if scene.silence_ratio <= threshold_ratio
        ]

    def get_all_silence_segments(self) -> list[dict[str, Any]]:
        """Get all silence segments across all scenes with scene information."""
        all_silences: list[dict[str, Any]] = []
        for scene in self.scene_metrics:
            for silence in scene.silence_segments:
                silence_with_scene = silence.copy()
                silence_with_scene["scene_number"] = scene.scene_number
                all_silences.append(silence_with_scene)
        return sorted(all_silences, key=lambda x: float(x["start"]))

    def get_longest_silences(self, count: int = 5) -> list[dict[str, Any]]:
        """Get the longest silence periods across all scenes."""
        all_silences = self.get_all_silence_segments()
        return sorted(all_silences, key=lambda x: x["duration"], reverse=True)[:count]

    def get_scenes_by_sentiment(self, sentiment_label: str) -> list[SceneSpeechMetrics]:
        """Get scenes with specific sentiment (positive, negative, neutral)."""
        return [
            scene
            for scene in self.scene_metrics
            if scene.sentiment_label.lower() == sentiment_label.lower()
        ]

    def get_most_positive_scenes(self, count: int = 3) -> list[SceneSpeechMetrics]:
        """Get scenes with highest positive sentiment scores."""
        return sorted(
            self.scene_metrics, key=lambda x: x.sentiment_score, reverse=True
        )[:count]

    def get_most_negative_scenes(self, count: int = 3) -> list[SceneSpeechMetrics]:
        """Get scenes with highest negative sentiment scores."""
        return sorted(self.scene_metrics, key=lambda x: x.sentiment_score)[:count]

    def get_highly_emotional_scenes(
        self, intensity_threshold: float = 0.5
    ) -> list[SceneSpeechMetrics]:
        """Get scenes with high emotional intensity (absolute sentiment score)."""
        return [
            scene
            for scene in self.scene_metrics
            if abs(scene.sentiment_score) > intensity_threshold
        ]

    def get_neutral_scenes(
        self, neutrality_threshold: float = 0.2
    ) -> list[SceneSpeechMetrics]:
        """Get scenes with neutral sentiment (low absolute sentiment score)."""
        return [
            scene
            for scene in self.scene_metrics
            if abs(scene.sentiment_score) <= neutrality_threshold
        ]

    def get_scenes_by_quality(self, quality_category: str) -> list[SceneSpeechMetrics]:
        """Get scenes with specific quality category (excellent, good, fair, poor)."""
        return [
            scene
            for scene in self.scene_metrics
            if scene.quality_category.lower() == quality_category.lower()
        ]

    def get_high_quality_scenes(
        self, min_quality_score: float = 0.8
    ) -> list[SceneSpeechMetrics]:
        """Get scenes with high quality scores."""
        return [
            scene
            for scene in self.scene_metrics
            if scene.quality_score >= min_quality_score
        ]

    def get_low_quality_scenes(
        self, max_quality_score: float = 0.6
    ) -> list[SceneSpeechMetrics]:
        """Get scenes with low quality scores."""
        return [
            scene
            for scene in self.scene_metrics
            if scene.quality_score <= max_quality_score
        ]

    def get_scenes_with_confidence_issues(self) -> list[SceneSpeechMetrics]:
        """Get scenes that have confidence issues identified."""
        return [
            scene for scene in self.scene_metrics if len(scene.confidence_issues) > 0
        ]

    def get_scenes_with_high_no_speech_ratio(
        self, threshold: float = 0.3
    ) -> list[SceneSpeechMetrics]:
        """Get scenes with high no-speech probability ratios."""
        return [
            scene for scene in self.scene_metrics if scene.no_speech_ratio > threshold
        ]

    def get_scenes_with_low_avg_logprob(
        self, threshold: float = -1.0
    ) -> list[SceneSpeechMetrics]:
        """Get scenes with low average log probabilities."""
        return [
            scene
            for scene in self.scene_metrics
            if scene.avg_segment_logprob < threshold
        ]

    def get_problematic_segments_summary(self) -> dict[str, Any]:
        """Get summary of all problematic segments across scenes."""
        all_problematic: list[dict[str, Any]] = []
        issue_counts: dict[str, int] = {}

        for scene in self.scene_metrics:
            for segment in scene.problematic_segments:
                segment_with_scene = segment.copy()
                segment_with_scene["scene_number"] = scene.scene_number
                all_problematic.append(segment_with_scene)

                issue_type = segment.get("issue_type", "unknown")
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        return {
            "total_problematic_segments": len(all_problematic),
            "issue_type_distribution": issue_counts,
            "segments": sorted(
                all_problematic, key=lambda x: int(x.get("severity", 0)), reverse=True
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            "scene_metrics": [scene.model_dump() for scene in self.scene_metrics],
            "overall_metrics": self.overall_metrics.model_dump(),
            "processing_time": self.processing_time,
            "language": self.language,
            "model_used": self.model_used,
            "confidence_threshold": self.confidence_threshold,
            "target_wpm_range": self.target_wpm_range,
            "analysis_summary": {
                "total_scenes": len(self.scene_metrics),
                "slow_scenes": len(self.get_slow_scenes()),
                "optimal_scenes": len(self.get_optimal_scenes()),
                "fast_scenes": len(self.get_fast_scenes()),
                "high_filler_scenes": len(self.get_scenes_with_high_filler_words()),
                "low_filler_scenes": len(self.get_scenes_with_low_filler_words()),
                "total_filler_words": self.overall_metrics.total_filler_words,
                "filler_word_rate": self.overall_metrics.filler_word_rate,
                "filler_words_by_type": self.get_total_filler_words_by_type(),
                "excessive_silence_scenes": len(
                    self.get_scenes_with_excessive_silence()
                ),
                "minimal_silence_scenes": len(self.get_scenes_with_minimal_silence()),
                "total_silence_time": self.overall_metrics.total_silence_time,
                "overall_silence_ratio": self.overall_metrics.overall_silence_ratio,
                "longest_silence": self.overall_metrics.longest_silence_overall,
                "total_silence_segments": self.overall_metrics.total_silence_segments,
                "overall_sentiment_score": self.overall_metrics.overall_sentiment_score,
                "sentiment_distribution": self.overall_metrics.sentiment_distribution,
                "positive_scenes": len(self.get_scenes_by_sentiment("positive")),
                "negative_scenes": len(self.get_scenes_by_sentiment("negative")),
                "neutral_scenes": len(self.get_scenes_by_sentiment("neutral")),
                "highly_emotional_scenes": len(self.get_highly_emotional_scenes()),
                "emotional_intensity": self.overall_metrics.emotional_intensity,
                "total_emotional_keywords": self.overall_metrics.total_emotional_keywords,
                "overall_quality_score": self.overall_metrics.overall_quality_score,
                "quality_distribution": self.overall_metrics.quality_distribution,
                "high_quality_scenes": len(self.get_high_quality_scenes()),
                "low_quality_scenes": len(self.get_low_quality_scenes()),
                "scenes_with_confidence_issues": len(
                    self.get_scenes_with_confidence_issues()
                ),
                "scenes_with_high_no_speech": len(
                    self.get_scenes_with_high_no_speech_ratio()
                ),
                "scenes_with_low_logprob": len(self.get_scenes_with_low_avg_logprob()),
                "transcription_reliability": self.overall_metrics.transcription_reliability,
                "critical_issues_count": len(self.overall_metrics.critical_issues),
                "problematic_segments_summary": self.get_problematic_segments_summary(),
            },
        }


class SpeechAnalyzer:
    """Analyzer for calculating speaking rate and speech quality metrics per scene."""

    def __init__(self, config: Any = None):
        """Initialize speech analyzer with configuration."""
        self.config = config or get_config()
        logger.info("SpeechAnalyzer initialized")

    def analyze_speech(
        self,
        transcription_result: TranscriptionResult,
        scene_result: SceneDetectionResult,
        confidence_threshold: float | None = None,
    ) -> SpeechAnalysisResult:
        """
        Analyze speech metrics for each scene in the video.

        Args:
            transcription_result: Complete transcription with timing information
            scene_result: Scene detection results with timestamps
            confidence_threshold: Minimum confidence for word inclusion in metrics

        Returns:
            SpeechAnalysisResult with detailed per-scene and overall metrics
        """
        import time

        start_time = time.time()

        # Ensure confidence_threshold is not None
        threshold: float = (
            confidence_threshold
            if confidence_threshold is not None
            else self.config.analysis.confidence_threshold
        )

        logger.info(f"Analyzing speech across {scene_result.total_scenes} scenes")
        logger.info(f"Using confidence threshold: {threshold}")

        # Calculate metrics for each scene
        scene_metrics: list[SceneSpeechMetrics] = []
        for i, scene in enumerate(scene_result.scenes):
            scene_metrics.append(
                self._analyze_scene_speech(
                    scene_number=i + 1,
                    start_time=scene.start_time,
                    end_time=scene.end_time,
                    transcription_result=transcription_result,
                    confidence_threshold=threshold,
                )
            )

        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(
            scene_metrics, transcription_result.duration
        )

        processing_time = time.time() - start_time

        result = SpeechAnalysisResult(
            scene_metrics=scene_metrics,
            overall_metrics=overall_metrics,
            processing_time=processing_time,
            language=transcription_result.language,
            model_used=transcription_result.model_used,
            confidence_threshold=threshold,
            target_wpm_range=self.config.analysis.target_wpm_range,
        )

        logger.info(
            f"Speech analysis complete: {len(scene_metrics)} scenes analyzed "
            f"in {processing_time:.2f}s"
        )
        logger.info(f"Overall speaking rate: {overall_metrics.average_wpm:.1f} WPM")

        return result

    def _analyze_scene_speech(
        self,
        scene_number: int,
        start_time: float,
        end_time: float,
        transcription_result: TranscriptionResult,
        confidence_threshold: float,
    ) -> SceneSpeechMetrics:
        """Analyze speech metrics for a single scene."""
        duration = end_time - start_time

        # Get text and words for this scene
        scene_text = transcription_result.get_text_between_times(start_time, end_time)
        scene_words = self._get_scene_words(
            transcription_result, start_time, end_time, confidence_threshold
        )

        # Calculate word-based metrics
        word_count = len(scene_words)
        speaking_rate_wpm = transcription_result.get_speaking_rate(start_time, end_time)

        # Calculate timing metrics
        actual_speaking_time = self._calculate_actual_speaking_time(
            scene_words, start_time, end_time
        )
        pause_time = duration - actual_speaking_time
        speech_ratio = actual_speaking_time / duration if duration > 0 else 0.0

        # Calculate content metrics
        character_count = len(scene_text)
        sentence_count = self._count_sentences(scene_text)

        # Calculate quality metrics
        avg_confidence, low_confidence_words = self._calculate_confidence_metrics(
            scene_words, confidence_threshold
        )

        # Detect filler words
        filler_count, filler_rate, detected_fillers = self._detect_filler_words(
            scene_words, duration
        )

        # Detect silence segments
        (
            silence_segments,
            total_silence_time,
            silence_ratio,
            average_silence_duration,
            longest_silence_duration,
            silence_count,
        ) = self._detect_silence_segments(scene_words, start_time, end_time)

        # Analyze sentiment
        (
            sentiment_score,
            sentiment_label,
            sentiment_confidence,
            emotional_keywords,
        ) = self._analyze_sentiment(scene_text)

        # Assess transcription quality
        (
            quality_score,
            quality_category,
            confidence_issues,
            segment_confidence_scores,
            avg_segment_logprob,
            no_speech_ratio,
            problematic_segments,
        ) = self._assess_transcription_quality(
            transcription_result, start_time, end_time, scene_words
        )

        return SceneSpeechMetrics(
            scene_number=scene_number,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            word_count=word_count,
            speaking_rate_wpm=speaking_rate_wpm,
            text_content=scene_text,
            character_count=character_count,
            sentence_count=sentence_count,
            actual_speaking_time=actual_speaking_time,
            pause_time=pause_time,
            speech_ratio=speech_ratio,
            avg_confidence=avg_confidence,
            low_confidence_words=low_confidence_words,
            filler_word_count=filler_count,
            filler_word_rate=filler_rate,
            filler_words_detected=detected_fillers,
            silence_segments=silence_segments,
            total_silence_time=total_silence_time,
            silence_ratio=silence_ratio,
            average_silence_duration=average_silence_duration,
            longest_silence_duration=longest_silence_duration,
            silence_count=silence_count,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            sentiment_confidence=sentiment_confidence,
            emotional_keywords=emotional_keywords,
            quality_score=quality_score,
            quality_category=quality_category,
            confidence_issues=confidence_issues,
            segment_confidence_scores=segment_confidence_scores,
            avg_segment_logprob=avg_segment_logprob,
            no_speech_ratio=no_speech_ratio,
            problematic_segments=problematic_segments,
        )

    def _get_scene_words(
        self,
        transcription_result: TranscriptionResult,
        start_time: float,
        end_time: float,
        confidence_threshold: float,
    ) -> list[dict[str, Any]]:
        """Extract words within scene timeframe with confidence filtering."""
        scene_words: list[dict[str, Any]] = []

        for segment in transcription_result.segments:
            # Check if segment overlaps with scene
            if segment.end >= start_time and segment.start <= end_time:
                for word in segment.words:
                    # Check if word is within scene timeframe
                    if (
                        word.start >= start_time
                        and word.end <= end_time
                        and word.confidence >= confidence_threshold
                    ):
                        scene_words.append(
                            {
                                "word": word.word,
                                "start": word.start,
                                "end": word.end,
                                "confidence": word.confidence,
                                "duration": word.end - word.start,
                            }
                        )

        return scene_words

    def _calculate_actual_speaking_time(
        self,
        scene_words: list[dict[str, Any]],
        start_time: float,
        end_time: float,
    ) -> float:
        """Calculate actual time spent speaking (excluding pauses between words)."""
        if not scene_words:
            return 0.0

        # Sort words by start time
        sorted_words = sorted(scene_words, key=lambda x: x["start"])

        total_speaking_time = 0.0

        for word in sorted_words:
            # Ensure word times are within scene bounds
            word_start = max(word["start"], start_time)
            word_end = min(word["end"], end_time)

            if word_end > word_start:
                total_speaking_time += word_end - word_start

        return total_speaking_time

    def _count_sentences(self, text: str) -> int:
        """Count sentences in text using simple punctuation detection."""
        if not text.strip():
            return 0

        # Count sentence-ending punctuation
        sentence_endings = text.count(".") + text.count("!") + text.count("?")

        # If no punctuation found, treat as one sentence if there's content
        return max(sentence_endings, 1) if text.strip() else 0

    def _calculate_confidence_metrics(
        self,
        scene_words: list[dict[str, Any]],
        confidence_threshold: float,
    ) -> tuple[float, int]:
        """Calculate average confidence and count of low-confidence words."""
        if not scene_words:
            return 0.0, 0

        confidences = [word["confidence"] for word in scene_words]
        avg_confidence = sum(confidences) / len(confidences)

        low_confidence_words = sum(
            1 for conf in confidences if conf < confidence_threshold
        )

        return avg_confidence, low_confidence_words

    def _detect_filler_words(
        self,
        scene_words: list[dict[str, Any]],
        duration: float,
    ) -> tuple[int, float, list[dict[str, Any]]]:
        """
        Detect filler words in scene text with configurable word lists.

        Args:
            scene_words: List of words with timing and confidence information
            duration: Scene duration in minutes for rate calculation

        Returns:
            Tuple of (count, rate per minute, detected filler words with details)
        """
        if not scene_words:
            return 0, 0.0, []

        filler_words_config = self.config.analysis.filler_words
        # Normalize filler words for case-insensitive matching
        normalized_fillers = [word.lower().strip() for word in filler_words_config]

        detected_fillers: list[dict[str, Any]] = []
        filler_count = 0

        for word_info in scene_words:
            word_text = str(word_info["word"]).lower().strip()

            # Remove punctuation for better matching
            clean_word = word_text.strip(".,!?;:")

            # Check if word is a filler word
            if clean_word in normalized_fillers:
                filler_count += 1
                detected_fillers.append(
                    {
                        "word": word_info["word"],
                        "normalized": clean_word,
                        "start": word_info["start"],
                        "end": word_info["end"],
                        "confidence": word_info["confidence"],
                    }
                )

        # Calculate rate per minute
        duration_minutes = duration / 60.0 if duration > 0 else 0.0
        filler_rate = filler_count / duration_minutes if duration_minutes > 0 else 0.0

        return filler_count, filler_rate, detected_fillers

    def _detect_silence_segments(
        self,
        scene_words: list[dict[str, Any]],
        start_time: float,
        end_time: float,
        min_silence_duration: float = 0.5,
    ) -> tuple[list[dict[str, Any]], float, float, float, float, int]:
        """
        Detect silence segments between words in a scene.

        Args:
            scene_words: List of words with timing information
            start_time: Scene start time
            end_time: Scene end time
            min_silence_duration: Minimum duration to consider as silence

        Returns:
            Tuple of (silence_segments, total_silence_time, silence_ratio,
                     average_silence_duration, longest_silence_duration, silence_count)
        """
        if not scene_words:
            # Entire scene is silence
            total_duration = end_time - start_time
            silence_segment = {
                "start": start_time,
                "end": end_time,
                "duration": total_duration,
                "type": "scene_silence",
            }
            return (
                [silence_segment],
                total_duration,
                1.0,
                total_duration,
                total_duration,
                1,
            )

        # Sort words by start time
        sorted_words = sorted(scene_words, key=lambda x: float(x["start"]))
        silence_segments: list[dict[str, Any]] = []
        total_silence_time = 0.0

        # Check for silence at the beginning of scene
        if sorted_words[0]["start"] > start_time + min_silence_duration:
            silence_duration = sorted_words[0]["start"] - start_time
            silence_segments.append(
                {
                    "start": start_time,
                    "end": sorted_words[0]["start"],
                    "duration": silence_duration,
                    "type": "pre_speech",
                }
            )
            total_silence_time += silence_duration

        # Check for silence between words
        for i in range(len(sorted_words) - 1):
            current_word_end = sorted_words[i]["end"]
            next_word_start = sorted_words[i + 1]["start"]

            gap_duration = next_word_start - current_word_end
            if gap_duration >= min_silence_duration:
                silence_segments.append(
                    {
                        "start": current_word_end,
                        "end": next_word_start,
                        "duration": gap_duration,
                        "type": "inter_word",
                    }
                )
                total_silence_time += gap_duration

        # Check for silence at the end of scene
        if sorted_words[-1]["end"] < end_time - min_silence_duration:
            silence_duration = end_time - sorted_words[-1]["end"]
            silence_segments.append(
                {
                    "start": sorted_words[-1]["end"],
                    "end": end_time,
                    "duration": silence_duration,
                    "type": "post_speech",
                }
            )
            total_silence_time += silence_duration

        # Calculate metrics
        scene_duration = end_time - start_time
        silence_ratio = (
            total_silence_time / scene_duration if scene_duration > 0 else 0.0
        )
        silence_count = len(silence_segments)

        if silence_segments:
            average_silence_duration = total_silence_time / silence_count
            longest_silence_duration = max(
                float(seg["duration"]) for seg in silence_segments
            )
        else:
            average_silence_duration = 0.0
            longest_silence_duration = 0.0

        return (
            silence_segments,
            total_silence_time,
            silence_ratio,
            average_silence_duration,
            longest_silence_duration,
            silence_count,
        )

    def _analyze_sentiment(
        self, text: str
    ) -> tuple[float, str, float, list[dict[str, Any]]]:
        """
        Analyze sentiment of text using a lightweight rule-based approach.

        Args:
            text: Text content to analyze

        Returns:
            Tuple of (sentiment_score, sentiment_label, confidence, emotional_keywords)
        """
        if not self.config.analysis.sentiment_analysis or not text.strip():
            return 0.0, "neutral", 0.0, []

        # Simple sentiment lexicon - can be enhanced with spaCy later
        positive_words = {
            "excellent",
            "great",
            "good",
            "amazing",
            "wonderful",
            "fantastic",
            "brilliant",
            "outstanding",
            "impressive",
            "successful",
            "effective",
            "helpful",
            "useful",
            "valuable",
            "important",
            "significant",
            "positive",
            "optimistic",
            "confident",
            "pleased",
            "happy",
            "excited",
            "passionate",
            "enthusiastic",
            "motivated",
            "inspired",
            "proud",
            "clear",
            "simple",
            "easy",
            "smooth",
            "perfect",
            "ideal",
        }

        negative_words = {
            "bad",
            "terrible",
            "awful",
            "horrible",
            "disappointing",
            "failed",
            "failure",
            "problem",
            "issue",
            "difficulty",
            "challenge",
            "struggle",
            "complicated",
            "complex",
            "confusing",
            "unclear",
            "uncertain",
            "worried",
            "concerned",
            "frustrated",
            "disappointed",
            "sad",
            "angry",
            "difficult",
            "hard",
            "impossible",
            "wrong",
            "mistake",
            "error",
            "poor",
            "weak",
            "limited",
            "lacking",
            "insufficient",
            "inadequate",
        }

        # Normalize text for analysis
        words = text.lower().split()
        emotional_keywords: list[dict[str, Any]] = []
        positive_score = 0
        negative_score = 0

        for word in words:
            # Remove punctuation for matching
            clean_word = word.strip(".,!?;:")

            if clean_word in positive_words:
                positive_score += 1
                emotional_keywords.append(
                    {"word": word, "sentiment": "positive", "score": 1.0}
                )
            elif clean_word in negative_words:
                negative_score += 1
                emotional_keywords.append(
                    {"word": word, "sentiment": "negative", "score": -1.0}
                )

        # Calculate sentiment score (-1 to 1)
        total_emotional_words = positive_score + negative_score
        if total_emotional_words == 0:
            sentiment_score = 0.0
            sentiment_label = "neutral"
            confidence = 0.5
        else:
            sentiment_score = (positive_score - negative_score) / len(words)
            # Clamp to [-1, 1] range
            sentiment_score = max(-1.0, min(1.0, sentiment_score))

            # Determine label based on score
            if sentiment_score > 0.1:
                sentiment_label = "positive"
            elif sentiment_score < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"

            # Calculate confidence based on emotional word density
            confidence = min(0.9, total_emotional_words / max(len(words), 1) * 5)
            confidence = max(0.1, confidence)

        return sentiment_score, sentiment_label, confidence, emotional_keywords

    def _assess_transcription_quality(
        self,
        transcription_result: TranscriptionResult,
        start_time: float,
        end_time: float,
        scene_words: list[dict[str, Any]],
    ) -> tuple[
        float,  # quality_score
        str,  # quality_category
        list[dict[str, Any]],  # confidence_issues
        list[dict[str, Any]],  # segment_confidence_scores
        float,  # avg_segment_logprob
        float,  # no_speech_ratio
        list[dict[str, Any]],  # problematic_segments
    ]:
        """
        Assess transcription quality for a scene using multiple indicators.

        Args:
            transcription_result: Complete transcription result
            start_time: Scene start time
            end_time: Scene end time
            scene_words: Words in this scene with confidence scores

        Returns:
            Tuple of quality assessment results
        """
        if not scene_words:
            return (
                0.0,
                "poor",
                [{"issue": "no_words", "severity": "critical"}],
                [],
                0.0,
                1.0,
                [],
            )

        # Get segments overlapping with this scene
        scene_segments: list[Segment] = []
        for segment in transcription_result.segments:
            if segment.end >= start_time and segment.start <= end_time:
                scene_segments.append(segment)

        if not scene_segments:
            return (
                0.0,
                "poor",
                [{"issue": "no_segments", "severity": "critical"}],
                [],
                0.0,
                1.0,
                [],
            )

        # Calculate word-level confidence metrics
        confidences = [word["confidence"] for word in scene_words]
        avg_word_confidence = sum(confidences) / len(confidences)
        low_confidence_count = sum(1 for conf in confidences if conf < 0.7)
        low_confidence_ratio = low_confidence_count / len(scene_words)

        # Calculate segment-level metrics
        segment_scores: list[dict[str, Any]] = []
        logprobs: list[float] = []
        no_speech_probs: list[float] = []
        problematic_segments: list[dict[str, Any]] = []

        for segment in scene_segments:
            segment_score: dict[str, Any] = {
                "segment_id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "avg_logprob": segment.avg_logprob,
                "no_speech_prob": segment.no_speech_prob,
                "word_count": len(segment.words),
                "confidence_score": avg_word_confidence,
            }
            segment_scores.append(segment_score)
            logprobs.append(segment.avg_logprob)
            no_speech_probs.append(segment.no_speech_prob)

            # Identify problematic segments
            issues = []
            severity_score = 0

            if segment.avg_logprob < -1.0:
                issues.append("low_logprob")
                severity_score += 2
            if segment.no_speech_prob > 0.5:
                issues.append("high_no_speech")
                severity_score += 3
            if len(segment.words) == 0:
                issues.append("no_words")
                severity_score += 4

            if issues:
                problematic_segments.append(
                    {
                        "segment_id": segment.id,
                        "start": segment.start,
                        "end": segment.end,
                        "issues": issues,
                        "severity": severity_score,
                        "issue_type": "transcription_quality",
                    }
                )

        # Calculate aggregate metrics
        avg_segment_logprob = sum(logprobs) / len(logprobs) if logprobs else 0.0
        avg_no_speech_prob = (
            sum(no_speech_probs) / len(no_speech_probs) if no_speech_probs else 0.0
        )

        # Identify confidence issues
        confidence_issues: list[dict[str, Any]] = []
        if low_confidence_ratio > 0.3:
            confidence_issues.append(
                {
                    "issue": "high_low_confidence_ratio",
                    "value": low_confidence_ratio,
                    "threshold": 0.3,
                    "severity": "medium",
                }
            )
        if avg_word_confidence < 0.7:
            confidence_issues.append(
                {
                    "issue": "low_average_confidence",
                    "value": avg_word_confidence,
                    "threshold": 0.7,
                    "severity": "medium",
                }
            )
        if avg_segment_logprob < -0.8:
            confidence_issues.append(
                {
                    "issue": "low_segment_logprob",
                    "value": avg_segment_logprob,
                    "threshold": -0.8,
                    "severity": "medium",
                }
            )
        if avg_no_speech_prob > 0.3:
            confidence_issues.append(
                {
                    "issue": "high_no_speech_probability",
                    "value": avg_no_speech_prob,
                    "threshold": 0.3,
                    "severity": "high",
                }
            )

        # Calculate overall quality score (0 to 1)
        # Factors: word confidence, segment logprob, no-speech ratio, problematic segments
        confidence_factor = avg_word_confidence  # 0 to 1
        logprob_factor = max(0, 1 + avg_segment_logprob)  # Convert logprob to 0-1 scale
        no_speech_factor = 1 - avg_no_speech_prob  # Invert so higher is better
        problems_factor = max(
            0, 1 - len(problematic_segments) / max(len(scene_segments), 1)
        )

        # Weighted combination
        quality_score = (
            confidence_factor * 0.3
            + logprob_factor * 0.3
            + no_speech_factor * 0.2
            + problems_factor * 0.2
        )
        quality_score = max(0.0, min(1.0, quality_score))

        # Determine quality category
        if quality_score >= 0.85:
            quality_category = "excellent"
        elif quality_score >= 0.7:
            quality_category = "good"
        elif quality_score >= 0.5:
            quality_category = "fair"
        else:
            quality_category = "poor"

        return (
            quality_score,
            quality_category,
            confidence_issues,
            segment_scores,
            avg_segment_logprob,
            avg_no_speech_prob,
            problematic_segments,
        )

    def _calculate_overall_metrics(
        self,
        scene_metrics: list[SceneSpeechMetrics],
        total_duration: float,
    ) -> OverallSpeechMetrics:
        """Calculate overall metrics across all scenes."""
        if not scene_metrics:
            return OverallSpeechMetrics(
                total_duration=total_duration,
                total_scenes=0,
                total_word_count=0,
                average_wpm=0.0,
                median_wpm=0.0,
                min_wpm=0.0,
                max_wpm=0.0,
                overall_speech_ratio=0.0,
                total_speaking_time=0.0,
                total_pause_time=total_duration,
                wpm_variance=0.0,
                wpm_standard_deviation=0.0,
                total_characters=0,
                total_sentences=0,
                average_words_per_sentence=0.0,
                overall_confidence=0.0,
                low_confidence_percentage=0.0,
                total_filler_words=0,
                filler_word_rate=0.0,
                most_common_filler_words=[],
                total_silence_time=total_duration,
                overall_silence_ratio=1.0,
                average_silence_per_scene=total_duration,
                longest_silence_overall=total_duration,
                total_silence_segments=1,
                overall_sentiment_score=0.0,
                sentiment_distribution={"positive": 0, "negative": 0, "neutral": 0},
                most_positive_scene=None,
                most_negative_scene=None,
                emotional_intensity=0.0,
                total_emotional_keywords=0,
                overall_quality_score=0.0,
                quality_distribution={"excellent": 0, "good": 0, "fair": 0, "poor": 0},
                average_confidence=0.0,
                low_quality_scenes=[],
                critical_issues=[],
                transcription_reliability=0.0,
                recommended_actions=["No transcription data available"],
            )

        # Aggregate basic metrics
        total_word_count = sum(scene.word_count for scene in scene_metrics)
        total_speaking_time = sum(scene.actual_speaking_time for scene in scene_metrics)
        total_pause_time = sum(scene.pause_time for scene in scene_metrics)
        total_characters = sum(scene.character_count for scene in scene_metrics)
        total_sentences = sum(scene.sentence_count for scene in scene_metrics)

        # Calculate WPM statistics
        wpm_values = [scene.speaking_rate_wpm for scene in scene_metrics]
        average_wpm = sum(wpm_values) / len(wpm_values)

        # Calculate median WPM
        sorted_wpm = sorted(wpm_values)
        n = len(sorted_wpm)
        if n % 2 == 0:
            median_wpm = (sorted_wpm[n // 2 - 1] + sorted_wpm[n // 2]) / 2
        else:
            median_wpm = sorted_wpm[n // 2]

        min_wpm = min(wpm_values)
        max_wpm = max(wpm_values)

        # Calculate variance and standard deviation
        wpm_variance = sum((wpm - average_wpm) ** 2 for wpm in wpm_values) / len(
            wpm_values
        )
        wpm_standard_deviation = wpm_variance**0.5

        # Calculate ratios and averages
        overall_speech_ratio = (
            total_speaking_time / total_duration if total_duration > 0 else 0.0
        )
        average_words_per_sentence = (
            total_word_count / total_sentences if total_sentences > 0 else 0.0
        )

        # Calculate confidence metrics
        total_words_with_confidence = sum(scene.word_count for scene in scene_metrics)
        total_low_confidence = sum(
            scene.low_confidence_words for scene in scene_metrics
        )

        # Calculate filler word metrics
        total_filler_words = sum(scene.filler_word_count for scene in scene_metrics)
        overall_filler_rate = (
            total_filler_words / (total_duration / 60.0) if total_duration > 0 else 0.0
        )

        # Aggregate filler words by type
        filler_word_counts: dict[str, int] = {}
        for scene in scene_metrics:
            for filler in scene.filler_words_detected:
                normalized_word = filler["normalized"]
                filler_word_counts[normalized_word] = (
                    filler_word_counts.get(normalized_word, 0) + 1
                )

        # Get most common filler words
        most_common_fillers = [
            {"word": word, "count": count}
            for word, count in sorted(
                filler_word_counts.items(), key=lambda x: x[1], reverse=True
            )[:5]  # Top 5 most common
        ]

        # Calculate silence metrics
        total_silence_time = sum(scene.total_silence_time for scene in scene_metrics)
        overall_silence_ratio = (
            total_silence_time / total_duration if total_duration > 0 else 0.0
        )
        average_silence_per_scene = (
            total_silence_time / len(scene_metrics) if scene_metrics else 0.0
        )
        longest_silence_overall = (
            max(scene.longest_silence_duration for scene in scene_metrics)
            if scene_metrics
            else 0.0
        )
        total_silence_segments = sum(scene.silence_count for scene in scene_metrics)

        # Calculate sentiment metrics
        sentiment_scores = [scene.sentiment_score for scene in scene_metrics]
        word_counts = [scene.word_count for scene in scene_metrics]

        # Calculate weighted average sentiment score
        if sum(word_counts) > 0:
            overall_sentiment_score = sum(
                score * count
                for score, count in zip(sentiment_scores, word_counts, strict=True)
            ) / sum(word_counts)
        else:
            overall_sentiment_score = 0.0

        # Count scenes by sentiment label
        sentiment_distribution = {"positive": 0, "negative": 0, "neutral": 0}
        for scene in scene_metrics:
            sentiment_distribution[scene.sentiment_label] += 1

        # Find most positive and negative scenes
        most_positive_scene = None
        most_negative_scene = None
        if scene_metrics:
            most_positive = max(scene_metrics, key=lambda x: x.sentiment_score)
            most_negative = min(scene_metrics, key=lambda x: x.sentiment_score)

            # Only assign if sentiment is significantly positive/negative
            if most_positive.sentiment_score > 0.1:
                most_positive_scene = most_positive.scene_number
            if most_negative.sentiment_score < -0.1:
                most_negative_scene = most_negative.scene_number

        # Calculate emotional intensity (average absolute sentiment)
        emotional_intensity = (
            sum(abs(score) for score in sentiment_scores) / len(sentiment_scores)
            if sentiment_scores
            else 0.0
        )

        # Count total emotional keywords
        total_emotional_keywords = sum(
            len(scene.emotional_keywords) for scene in scene_metrics
        )

        # Calculate quality assessment metrics
        quality_scores = [scene.quality_score for scene in scene_metrics]
        overall_quality_score = (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        )

        # Count scenes by quality category
        quality_distribution = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        for scene in scene_metrics:
            quality_distribution[scene.quality_category] += 1

        # Calculate average confidence across all scenes
        confidence_scores = [scene.avg_confidence for scene in scene_metrics]
        average_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0.0
        )

        # Identify low quality scenes (quality score < 0.6)
        low_quality_scenes = [
            scene.scene_number for scene in scene_metrics if scene.quality_score < 0.6
        ]

        # Collect critical issues from all scenes
        critical_issues: list[dict[str, Any]] = []
        for scene in scene_metrics:
            for issue in scene.confidence_issues:
                if issue.get("severity") == "high":
                    issue_copy = issue.copy()
                    issue_copy["scene_number"] = scene.scene_number
                    critical_issues.append(issue_copy)

        # Calculate transcription reliability (weighted by scene quality)
        if scene_metrics:
            scene_durations = [scene.duration for scene in scene_metrics]
            total_scene_duration = sum(scene_durations)

            if total_scene_duration > 0:
                transcription_reliability = (
                    sum(scene.quality_score * scene.duration for scene in scene_metrics)
                    / total_scene_duration
                )
            else:
                transcription_reliability = overall_quality_score
        else:
            transcription_reliability = 0.0

        # Generate recommended actions based on quality issues
        recommended_actions: list[str] = []
        if overall_quality_score < 0.7:
            recommended_actions.append(
                "Consider re-recording audio with better microphone quality"
            )
        if len(low_quality_scenes) > 0:
            recommended_actions.append(
                f"Review {len(low_quality_scenes)} scenes with low quality transcription"
            )
        if len(critical_issues) > 0:
            recommended_actions.append(
                "Address critical transcription confidence issues"
            )
        if average_confidence < 0.7:
            recommended_actions.append(
                "Consider manual review of low-confidence transcriptions"
            )
        if not recommended_actions:
            recommended_actions.append("Transcription quality is good")

        if total_words_with_confidence > 0:
            # Weight confidence by word count per scene
            weighted_confidence = (
                sum(scene.avg_confidence * scene.word_count for scene in scene_metrics)
                / total_words_with_confidence
            )

            low_confidence_percentage = (
                total_low_confidence / total_words_with_confidence
            ) * 100
        else:
            weighted_confidence = 0.0
            low_confidence_percentage = 0.0

        return OverallSpeechMetrics(
            total_duration=total_duration,
            total_scenes=len(scene_metrics),
            total_word_count=total_word_count,
            average_wpm=average_wpm,
            median_wpm=median_wpm,
            min_wpm=min_wpm,
            max_wpm=max_wpm,
            overall_speech_ratio=overall_speech_ratio,
            total_speaking_time=total_speaking_time,
            total_pause_time=total_pause_time,
            wpm_variance=wpm_variance,
            wpm_standard_deviation=wpm_standard_deviation,
            total_characters=total_characters,
            total_sentences=total_sentences,
            average_words_per_sentence=average_words_per_sentence,
            overall_confidence=weighted_confidence,
            low_confidence_percentage=low_confidence_percentage,
            total_filler_words=total_filler_words,
            filler_word_rate=overall_filler_rate,
            most_common_filler_words=most_common_fillers,
            total_silence_time=total_silence_time,
            overall_silence_ratio=overall_silence_ratio,
            average_silence_per_scene=average_silence_per_scene,
            longest_silence_overall=longest_silence_overall,
            total_silence_segments=total_silence_segments,
            overall_sentiment_score=overall_sentiment_score,
            sentiment_distribution=sentiment_distribution,
            most_positive_scene=most_positive_scene,
            most_negative_scene=most_negative_scene,
            emotional_intensity=emotional_intensity,
            total_emotional_keywords=total_emotional_keywords,
            overall_quality_score=overall_quality_score,
            quality_distribution=quality_distribution,
            average_confidence=average_confidence,
            low_quality_scenes=low_quality_scenes,
            critical_issues=critical_issues,
            transcription_reliability=transcription_reliability,
            recommended_actions=recommended_actions,
        )


def create_speech_analyzer(config: Any = None) -> SpeechAnalyzer:
    """Create a new SpeechAnalyzer instance.

    Args:
        config: Optional configuration object

    Returns:
        Configured SpeechAnalyzer instance
    """
    return SpeechAnalyzer(config=config)
