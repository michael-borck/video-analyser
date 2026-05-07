# Speaker Diarization & Rubric System - Quick Integration Reference

## Quick Status Check

### Speaker Diarization (speaker_diarization.py)
- **Current Status:** Standalone, not integrated
- **Production Ready:** 60%
- **Main Issue:** Missing pipeline integration & configuration
- **Key Missing:** Error handling, token validation, timeout handling

### Rubric System (rubric_system.py)
- **Current Status:** Standalone, fully functional
- **Production Ready:** 75%
- **Main Issue:** Missing report integration
- **Key Missing:** Logging in scorer, transaction safety, database persistence

---

## How to Use Each Module Independently

### Speaker Diarization

```python
from pathlib import Path
from video_analyser.analysis.speaker_diarization import SpeakerDiarizer
import asyncio

# Initialize (requires pyannote.audio installed)
diarizer = SpeakerDiarizer(use_gpu=True)

# Run diarization on audio file
async def diarize_audio():
    result = await diarizer.diarize(
        audio_path=Path("audio.wav"),
        progress_callback=lambda progress, desc: print(f"{desc}: {progress}")
    )
    return result

# Get results
result = asyncio.run(diarize_audio())

# Query results
speaker_id = diarizer.get_speaker_at_time(result, time_seconds=10.5)
profile = diarizer.get_speaker_profile(result, speaker_id="speaker_1")

# Modify results
diarizer.relabel_speaker(result, "speaker_1", "Alice")
diarizer.merge_speakers(result, ["speaker_2", "speaker_3"], "merged_speaker")

# Access data
print(f"Speakers detected: {result.num_speakers}")
print(f"Segments: {len(result.segments)}")
print(f"Processing time: {result.processing_time}s")
```

### Rubric System

```python
from pathlib import Path
from video_analyser.analysis.rubric_system import (
    RubricBuilder, RubricRepository, RubricScorer
)

# Create a rubric using builder pattern
builder = RubricBuilder("Presentation Rubric")
content_cat = builder.add_category("Content", weight=2.0)
content_cat.add_criterion("Clear Thesis", weight=2.0)
content_cat.add_criterion("Supporting Evidence", weight=1.5)

delivery_cat = builder.add_category("Delivery", weight=1.0)
delivery_cat.add_criterion("Eye Contact")
delivery_cat.add_criterion("Pace")

rubric = builder.build(tags=["presentation", "academic"])

# Save rubric to disk
repo = RubricRepository(Path("data/rubrics"))
repo.save(rubric)

# Load rubric
loaded = repo.load(rubric.id)

# Score a presentation
scorer = RubricScorer(rubric)
scores = {
    rubric.categories[0].id: {
        rubric.categories[0].criteria[0].id: 4,  # Clear Thesis: 4/5
        rubric.categories[0].criteria[1].id: 3,  # Evidence: 3/5
    },
    rubric.categories[1].id: {
        rubric.categories[1].criteria[0].id: 5,  # Eye Contact: 5/5
        rubric.categories[1].criteria[1].id: 4,  # Pace: 4/5
    },
}

assessment = scorer.score_all_categories(scores)

# Access assessment
print(f"Overall Score: {assessment.overall_score:.2f}/5")
print(f"Overall Percentage: {assessment.overall_percentage:.1f}%")
for cat_score in assessment.category_scores:
    print(f"  {cat_score.category_name}: {cat_score.category_percentage:.1f}%")

# Search rubrics
results = repo.search("presentation")
```

---

## Installation Requirements

### For Speaker Diarization
```bash
# Currently NOT in pyproject.toml - manual install needed
pip install pyannote.audio
huggingface-cli login  # Obtain token first from huggingface.co
```

### For Rubric System
```bash
# Already included in base dependencies
# Just ensure you have video-analyser installed
pip install -e .
```

---

## Integration Roadmap

### Phase 1: Configuration (Required)
Add to `src/video_analyser/utils/config.py`:
```python
class DiarizationConfig(BaseModel):
    enabled: bool = Field(default=False)
    model: str = Field(default="pyannote/speaker-diarization-3.1")
    use_gpu: bool = Field(default=True)
    num_speakers: int | None = Field(None)

class RubricConfig(BaseModel):
    enabled: bool = Field(default=False)
    storage_dir: Path = Field(default=Path("data/rubrics"))
    auto_save: bool = Field(default=True)
```

### Phase 2: Pipeline Integration
Modify `src/video_analyser/core/pipeline_coordinator.py`:
```python
async def analyze_speech(self, audio_path, ...):
    # Add diarization step
    from video_analyser.analysis.speaker_diarization import SpeakerDiarizer
    diarizer = SpeakerDiarizer(config=self.config)
    diar_result = await diarizer.diarize(audio_path)
    
    # Integrate with transcription
    for segment in transcription.segments:
        speaker = diarizer.get_speaker_at_time(diar_result, segment.start)
        segment.speaker_id = speaker
```

### Phase 3: CLI Integration
Modify `src/video_analyser/cli.py`:
Add diarization to operations list (line 178-186)

### Phase 4: Report Integration
Modify `src/video_analyser/reports/report_generator.py`:
- Add speaker diarization results to JSON output
- Add speaker labels to HTML output
- Add rubric assessment to report (if provided)

---

## Testing

### Speaker Diarization Tests
Location: `tests/analysis/test_speaker_diarization.py`

Run specific tests:
```bash
pytest tests/analysis/test_speaker_diarization.py -v

# Test speaker operations
pytest tests/analysis/test_speaker_diarization.py::TestSpeakerDiarizer -v
```

### Rubric System Tests
Location: `tests/analysis/test_rubric_system.py`

Run specific tests:
```bash
pytest tests/analysis/test_rubric_system.py -v

# Test scoring
pytest tests/analysis/test_rubric_system.py::TestRubricScorer -v
```

---

## Known Limitations

### Speaker Diarization
- Confidence scores hardcoded to 1.0 (pyannote limitation)
- Overlap detection uses O(n²) algorithm (slow for >100 segments)
- No caching of results (expensive operation)
- No speaker identification database (only IDs, not names)
- Requires HuggingFace token authentication

### Rubric System
- No database persistence (files only)
- No versioning system (timestamp exists but not used)
- No concurrent access handling
- No assessment immutability
- No audit trail
- No export to Excel/CSV

---

## Error Handling Guide

### Speaker Diarization Errors

```python
try:
    result = await diarizer.diarize(audio_path)
except ImportError:
    # pyannote.audio not installed
    print("Install with: pip install pyannote.audio")
except FileNotFoundError:
    # Audio file doesn't exist
    print(f"File not found: {audio_path}")
except RuntimeError as e:
    # Diarization failed (no token, network error, etc.)
    print(f"Diarization error: {e}")
```

### Rubric System Errors

```python
try:
    rubric = repo.load(rubric_id)
    if not rubric:
        print(f"Rubric not found: {rubric_id}")
except json.JSONDecodeError:
    # Corrupted JSON file
    print(f"Corrupted rubric file")
except ValueError as e:
    # Validation error
    print(f"Invalid rubric: {e}")
```

---

## Data Models Quick Reference

### Speaker Diarization Output
```
DiarizationResult
├─ num_speakers: int (e.g., 2)
├─ segments: [SpeakerSegment]
│  ├─ speaker_id: str (e.g., "speaker_1")
│  ├─ start_time: float (seconds)
│  ├─ end_time: float (seconds)
│  ├─ confidence: float (0-1)
│  └─ duration: float (seconds)
├─ speakers: [SpeakerProfile]
│  ├─ speaker_id: str
│  ├─ total_speaking_time: float (seconds)
│  ├─ percentage_of_total: float (%)
│  ├─ first_appearance: float (seconds)
│  └─ avg_confidence: float (0-1)
└─ total_duration: float (seconds)
```

### Rubric Assessment Output
```
RubricAssessment
├─ overall_score: float (e.g., 4.2 out of 5)
├─ overall_percentage: float (e.g., 80.0%)
├─ category_scores: [RubricCategoryScore]
│  ├─ category_name: str
│  ├─ category_total: float
│  ├─ category_percentage: float
│  └─ criterion_scores: [RubricScore]
│     ├─ criterion_id: str
│     ├─ score: int
│     └─ notes: str | None
└─ feedback: str | None (overall feedback)
```

---

## Performance Notes

### Speaker Diarization
- Model loading: ~5-10 seconds (first use)
- Diarization processing: ~10-30 seconds per minute of audio
- Memory: ~2-4 GB for GPU, variable for CPU
- Can run async to avoid blocking

### Rubric System
- Rubric creation: <1ms
- Scoring: <1ms
- Search: Linear O(n) - optimize if >1000 rubrics
- File I/O: Depends on disk speed (typically <100ms)

---

## Next Steps

1. **To Use Diarization:**
   - Install: `pip install pyannote.audio`
   - Get HuggingFace token from huggingface.co
   - Add `DiarizationConfig` to `config.py`
   - Integrate into pipeline

2. **To Use Rubric System:**
   - No installation needed
   - Create rubrics using RubricBuilder
   - Save to RubricRepository
   - Score with RubricScorer
   - Integrate results into reports

