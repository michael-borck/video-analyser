# DeepBrief Development Guide

## Quick Start

This guide helps you get started with DeepBrief development quickly.

### Prerequisites

- Python 3.11+ 
- ffmpeg (for video processing)
- uv (fast Python package manager)

### One-Command Setup

```bash
# Clone and set up everything automatically
git clone https://github.com/michael-borck/deep-brief.git
cd deep-brief
python scripts/setup_dev.py
```

### Manual Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Verify installation
deep-brief --help
pytest -v
```

## Development Workflow

### Code Quality (Run before committing)

```bash
# Format, lint, type check, and test
ruff format . && ruff check . && basedpyright && pytest -v
```

### Running Tests

```bash
# All tests
pytest -v

# By category
pytest -m "not slow"           # Skip slow tests
pytest -m video               # Video processing tests
pytest -m "ai or audio"       # AI/ML and audio tests

# Coverage report
pytest --cov-report=html
```

### CLI Usage

```bash
# Show help
deep-brief --help

# Analyze a video
deep-brief analyze video.mp4

# Use API for image captioning
deep-brief analyze video.mp4 --use-api --api-provider anthropic

# Show configuration
deep-brief config --all
```

## Project Structure

```
src/deep_brief/          # Main package
├── core/                # Video processing pipeline
├── analysis/            # Speech and visual analysis  
├── reports/             # Report generation
├── interface/           # Gradio web interface (planned)
└── utils/               # Configuration and utilities

tests/                   # Test suite (mirrors src structure)
config/                  # Configuration files
scripts/                 # Development and utility scripts
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Config Files

Default configuration in `config/config.yaml`. Override with:
- Environment variables (prefix: `DEEP_BRIEF_`)
- Custom config file: `deep-brief --config my-config.yaml`

## API Keys (Optional)

For image captioning with AI models:

```bash
# Anthropic Claude (recommended)
ANTHROPIC_API_KEY=sk-ant-your-key

# OpenAI GPT-4o
OPENAI_API_KEY=sk-proj-your-key

# Google Gemini
GOOGLE_API_KEY=your-key
```

Get keys from:
- Anthropic: https://console.anthropic.com/
- OpenAI: https://platform.openai.com/api-keys
- Google: https://makersuite.google.com/app/apikey

## Common Issues

### Package Import Errors

```bash
# Reinstall in development mode
uv pip install -e ".[dev]"
```

### Missing ffmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Model Download Issues

Models are downloaded automatically on first use. Ensure:
- Internet connection
- Sufficient disk space (~2GB for all models)
- Write permissions in cache directory

## Development Tips

### Running Single Tests

```bash
pytest tests/test_config.py::test_load_config -v
```

### Debug Mode

```bash
deep-brief analyze video.mp4 --verbose
# or set DEEP_BRIEF_DEBUG=true
```

### Test Video Generation

```bash
# Generate sample videos for testing
python3 scripts/generate_test_videos.py

# Install espeak for realistic speech
sudo apt install espeak  # Ubuntu/Debian
brew install espeak      # macOS
```

## Code Style

- Use ruff for formatting and linting
- Strict type checking with basedpyright
- Follow existing patterns and conventions
- Add tests for new functionality

## Task Tracking

Current development tasks are tracked in `tasks/tasks-prd-phase1-mvp.md`.

The project follows a strict one-task-at-a-time approach. See `CLAUDE.md` for details.