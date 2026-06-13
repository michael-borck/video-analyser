# Basic usage

`video-analyser` extracts frames, transcripts, and quality metrics from a video for presentation feedback, for the analyser family.

## Install

```bash
pip install video-analyser
```

## CLI

```bash
video-analyser path/to/presentation.mp4 --json
```

Use `--fast` for transcript-only (skips visual analysis).

## HTTP

```bash
video-analyser serve
curl -F file=@path/to/presentation.mp4 http://localhost:8002/analyse
```
