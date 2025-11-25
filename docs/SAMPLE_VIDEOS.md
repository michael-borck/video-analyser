# Sample Videos for DeepBrief

This document explains the sample videos generated for testing and development.

## Current Status

The generated sample videos contain **tone audio** (sine waves) instead of actual speech. This is because text-to-speech (TTS) tools are not installed on the system.

## Impact on Development

- **Speech-to-text testing**: Limited - Whisper will transcribe the tones as silence or noise
- **Filler word detection**: Not possible with tone audio
- **Speaking rate analysis**: Not possible with tone audio
- **Transcript analysis**: Not possible with tone audio

## Solutions

### Option 1: Install TTS Tools (Recommended)

Install espeak for synthetic speech generation:
```bash
sudo apt install espeak
```

Then regenerate videos:
```bash
python3 scripts/generate_test_videos.py
```

### Option 2: Use Real Video Samples

For production-quality testing, use real presentation videos:

1. Record a short presentation (2-3 minutes)
2. Place in `samples/` directory
3. Use for development and testing

### Option 3: Online TTS Services

Modify the script to use online TTS services:
- Google Text-to-Speech API
- Amazon Polly
- Azure Speech Services

## Current Test Coverage

**Working Tests:**
- Video file validation
- Format support (MP4, WebM, AVI)
- Frame extraction
- Scene detection (visual)
- File processing pipeline

**Limited Tests:**
- Audio extraction (extracts tone audio)
- Speech transcription (will return empty/noise)
- Speech analysis features

## Recommended Next Steps

1. Install espeak: `sudo apt install espeak`
2. Regenerate samples with speech
3. Or use real video samples for development
4. Consider adding online TTS integration for CI/CD

## Sample Video Specifications

### Test Fixtures (`tests/fixtures/`)
- `minimal_test.mp4` - 2s, 160x120, for basic testing
- `no_audio_test.mp4` - 3s, 160x120, silent video
- `short_presentation.mp4` - 10s, 640x480, presentation style
- `different_format_test.*` - 5s, 320x240, multiple formats

### Development Samples (`samples/`)
- `dev_sample_good.mp4` - 30s, 1280x720, good presentation
- `dev_sample_needs_work.mp4` - 25s, 1280x720, needs improvement
- `presentation_with_pauses.mp4` - 45s, 1280x720, with pauses
- `fast_speaker.mp4` - 30s, 1280x720, fast-paced
- `with_filler_words.mp4` - 35s, 1280x720, contains filler words

All samples currently have tone audio (440Hz sine wave) instead of speech.