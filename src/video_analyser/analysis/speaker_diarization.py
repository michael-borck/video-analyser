"""Speaker diarization and identification using pyannote.audio."""

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SpeakerSegment(BaseModel):
    """A segment where a specific speaker is speaking."""

    speaker_id: str = Field(
        ..., description="Unique speaker identifier (e.g., 'speaker_1')"
    )
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    duration: float = Field(..., description="Duration in seconds")
    label: str | None = Field(None, description="Optional manual label for speaker")

    def __post_init__(self) -> None:
        """Calculate duration if not provided."""
        if not hasattr(self, "duration"):
            object.__setattr__(self, "duration", self.end_time - self.start_time)


class SpeakerProfile(BaseModel):
    """Profile information for a detected speaker."""

    speaker_id: str = Field(..., description="Unique speaker identifier")
    label: str | None = Field(
        None, description="User-assigned speaker label (e.g., 'Alice', 'Bob')"
    )
    num_segments: int = Field(default=0, description="Number of speaking segments")
    total_speaking_time: float = Field(
        default=0.0, description="Total speaking time in seconds"
    )
    percentage_of_total: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Percentage of total speaking time"
    )
    first_appearance: float | None = Field(
        None, description="Time of first appearance in seconds"
    )
    last_appearance: float | None = Field(
        None, description="Time of last appearance in seconds"
    )
    avg_segment_duration: float = Field(
        default=0.0, description="Average duration per speaking segment"
    )
    avg_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Average confidence score"
    )


class DiarizationResult(BaseModel):
    """Complete diarization result for a video."""

    audio_path: Path = Field(..., description="Path to analyzed audio file")
    num_speakers: int = Field(..., ge=1, description="Number of speakers detected")
    segments: list[SpeakerSegment] = Field(
        default_factory=list, description="All speaker segments"
    )
    speakers: list[SpeakerProfile] = Field(
        default_factory=list, description="Profile for each speaker"
    )
    total_duration: float = Field(..., description="Total audio duration in seconds")
    overlapping_speech_segments: list[tuple[float, float]] = Field(
        default_factory=list, description="Segments where multiple speakers overlap"
    )
    processing_time: float = Field(
        ..., description="Time taken for diarization in seconds"
    )
    model_used: str = Field(
        default="pyannote/speaker-diarization-3.1",
        description="Diarization model used",
    )


class SpeakerDiarizer:
    """
    Speaker diarization using pyannote.audio.

    Identifies and separates different speakers in audio/video content.
    Requires the pyannote.audio library and a HuggingFace access token.
    """

    def __init__(self, config: Any = None, use_gpu: bool = True):
        """
        Initialize SpeakerDiarizer.

        Args:
            config: Configuration object (optional)
            use_gpu: Whether to use GPU for processing (default: True)
        """
        self.config = config
        self.use_gpu = use_gpu

        try:
            from pyannote.audio import Pipeline  # type: ignore[import-untyped]

            self.Pipeline = Pipeline
            logger.info("pyannote.audio loaded successfully")
        except ImportError as e:
            raise ImportError(
                "pyannote.audio is required for speaker diarization. "
                "Install with: pip install pyannote.audio"
            ) from e

        self._pipeline: Any = None
        logger.info(f"SpeakerDiarizer initialized (GPU: {use_gpu})")

    def _get_pipeline(self) -> Any:
        """Get or create the diarization pipeline."""
        if self._pipeline is None:
            try:
                import torch  # type: ignore[import-untyped]

                device = (
                    "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
                )
                logger.info(f"Loading diarization pipeline on device: {device}")

                # Load the pipeline
                self._pipeline = self.Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=True,
                )
                self._pipeline.to(torch.device(device))
            except Exception as e:
                logger.error(f"Failed to load diarization pipeline: {e}")
                raise

        return self._pipeline  # type: ignore[return-value]

    async def diarize(
        self, audio_path: Path, progress_callback: Any = None
    ) -> DiarizationResult:
        """
        Perform speaker diarization on audio file.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            progress_callback: Optional callback for progress updates

        Returns:
            DiarizationResult with speaker segments and profiles

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If diarization fails
        """
        import asyncio
        import time
        from pathlib import Path as PathlibPath

        if not PathlibPath(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        start_time = time.time()

        try:
            if progress_callback:
                progress_callback(0.1, desc="Loading diarization model...")

            # Run diarization in thread to avoid blocking
            pipeline = self._get_pipeline()

            if progress_callback:
                progress_callback(0.3, desc="Processing audio...")

            # Run diarization (blocking operation, run in executor)
            loop = asyncio.get_event_loop()
            diarization = await loop.run_in_executor(None, pipeline, str(audio_path))

            if progress_callback:
                progress_callback(0.7, desc="Processing results...")

            # Process diarization results
            result = self._process_diarization(diarization, audio_path, start_time)

            if progress_callback:
                progress_callback(1.0, desc="Diarization complete")

            return result

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise RuntimeError(f"Speaker diarization failed: {e}") from e

    def _process_diarization(
        self, diarization: Any, audio_path: Path, start_time: float
    ) -> DiarizationResult:
        """
        Process raw diarization output into structured result.

        Args:
            diarization: Raw diarization object from pyannote
            audio_path: Path to audio file
            start_time: When diarization started (for timing)

        Returns:
            Structured DiarizationResult
        """
        import time

        segments: list[SpeakerSegment] = []
        speaker_times: dict[str, list[tuple[float, float]]] = {}
        overlapping_segments: list[tuple[float, float]] = []

        # Extract segments
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment = SpeakerSegment(
                speaker_id=speaker,
                start_time=float(turn.start),
                end_time=float(turn.end),
                confidence=1.0,  # pyannote doesn't provide per-segment confidence
                duration=float(turn.end - turn.start),
                label=None,
            )
            segments.append(segment)

            if speaker not in speaker_times:
                speaker_times[speaker] = []
            speaker_times[speaker].append((float(turn.start), float(turn.end)))

        # Detect overlapping segments
        for i, seg1 in enumerate(segments):
            for seg2 in segments[i + 1 :]:
                if seg1.start_time < seg2.end_time and seg2.start_time < seg1.end_time:
                    overlap_start = max(seg1.start_time, seg2.start_time)
                    overlap_end = min(seg1.end_time, seg2.end_time)
                    if (overlap_start, overlap_end) not in overlapping_segments:
                        overlapping_segments.append((overlap_start, overlap_end))

        # Create speaker profiles
        speakers: list[SpeakerProfile] = []
        total_duration = max((s.end_time for s in segments), default=0.0)

        for speaker_id in sorted(speaker_times.keys()):
            times = speaker_times[speaker_id]
            total_time = sum(end - start for start, end in times)
            percentage = (
                (total_time / total_duration * 100) if total_duration > 0 else 0
            )

            profile = SpeakerProfile(
                speaker_id=speaker_id,
                label=None,
                num_segments=len(times),
                total_speaking_time=total_time,
                percentage_of_total=percentage,
                first_appearance=min(t[0] for t in times) if times else None,
                last_appearance=max(t[1] for t in times) if times else None,
                avg_segment_duration=total_time / len(times) if times else 0,
                avg_confidence=1.0,  # Default since pyannote doesn't provide this
            )
            speakers.append(profile)

        processing_time = time.time() - start_time

        return DiarizationResult(
            audio_path=audio_path,
            num_speakers=len(speakers),
            segments=segments,
            speakers=speakers,
            total_duration=total_duration,
            overlapping_speech_segments=overlapping_segments,
            processing_time=processing_time,
        )

    def get_speaker_at_time(
        self, result: DiarizationResult, time_seconds: float
    ) -> str | None:
        """
        Get the speaker ID at a specific time in the audio.

        Args:
            result: DiarizationResult from diarize()
            time_seconds: Time in seconds

        Returns:
            Speaker ID or None if no speaker at that time
        """
        for segment in result.segments:
            if segment.start_time <= time_seconds < segment.end_time:
                return segment.speaker_id
        return None

    def get_speaker_profile(
        self, result: DiarizationResult, speaker_id: str
    ) -> SpeakerProfile | None:
        """
        Get the profile for a specific speaker.

        Args:
            result: DiarizationResult
            speaker_id: Speaker identifier

        Returns:
            SpeakerProfile or None if not found
        """
        for speaker in result.speakers:
            if speaker.speaker_id == speaker_id:
                return speaker
        return None

    def relabel_speaker(
        self, result: DiarizationResult, speaker_id: str, new_label: str
    ) -> None:
        """
        Manually relabel a speaker.

        Args:
            result: DiarizationResult
            speaker_id: Speaker to relabel
            new_label: New label (e.g., "Alice", "Bob")
        """
        for speaker in result.speakers:
            if speaker.speaker_id == speaker_id:
                speaker.label = new_label
                logger.info(f"Relabeled {speaker_id} to '{new_label}'")
                return

        logger.warning(f"Speaker {speaker_id} not found in result")

    def merge_speakers(
        self, result: DiarizationResult, speaker_ids: list[str], merged_id: str
    ) -> None:
        """
        Merge multiple speakers into one (for correction).

        Args:
            result: DiarizationResult
            speaker_ids: List of speaker IDs to merge
            merged_id: New speaker ID
        """
        # Update segments
        for segment in result.segments:
            if segment.speaker_id in speaker_ids:
                segment.speaker_id = merged_id

        # Remove old profiles and create new one
        merged_profiles = [s for s in result.speakers if s.speaker_id in speaker_ids]
        result.speakers = [
            s for s in result.speakers if s.speaker_id not in speaker_ids
        ]

        if merged_profiles:
            # Aggregate stats
            total_time = sum(s.total_speaking_time for s in merged_profiles)
            new_profile = SpeakerProfile(
                speaker_id=merged_id,
                label=None,
                num_segments=sum(s.num_segments for s in merged_profiles),
                total_speaking_time=total_time,
                percentage_of_total=(
                    (total_time / result.total_duration * 100)
                    if result.total_duration > 0
                    else 0
                ),
                first_appearance=min(
                    (s.first_appearance for s in merged_profiles if s.first_appearance),
                    default=None,
                ),
                last_appearance=max(
                    (s.last_appearance for s in merged_profiles if s.last_appearance),
                    default=None,
                ),
            )
            result.speakers.append(new_profile)
            result.num_speakers = len(result.speakers)

        logger.info(f"Merged {speaker_ids} into {merged_id}")
