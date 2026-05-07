# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Development Commands

### Setup and Environment
```bash
# Clone and setup development environment
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Verify installation
video-analyser --help
```

### Code Quality (Run ALL before committing)
```bash
# Format, lint, type check, and test in one command
ruff format . && ruff check . && basedpyright && pytest -v

# Individual commands
ruff format .           # Format code
ruff check .           # Lint with auto-fix: ruff check . --fix
basedpyright           # Strict type checking
pytest -v              # Run tests with verbose output
```

### Testing
```bash
# Run all tests
pytest -v

# Run tests by category
pytest -m "not slow"           # Skip slow tests
pytest -m video               # Video processing tests only
pytest -m "ai or audio"       # AI/ML and audio tests

# Run single test
pytest tests/test_config.py::test_load_config -v

# Coverage report
pytest --cov-report=html      # Generate HTML coverage report
```

### Sample Video Generation
```bash
# Generate test videos (requires espeak for speech synthesis)
python3 scripts/generate_test_videos.py

# Install espeak for realistic speech samples
sudo apt install espeak      # Ubuntu/Debian
brew install espeak          # macOS
```

### Application Commands
```bash
video-analyser --help            # Show all commands
video-analyser version           # Show version
video-analyser config --all      # Show full configuration
video-analyser analyze           # Launch web interface (when implemented)
```

## Architecture Overview

### Core Architecture Pattern
This is a **video analysis pipeline** application with a modular, layered architecture:

**Entry Points:**
- CLI (`src/video_analyser/cli.py`) - Typer-based command interface
- Web UI (`src/video_analyser/interface/`) - Gradio-based web interface (planned)

**Processing Pipeline:**
1. **Core** (`src/video_analyser/core/`) - Video processing, audio extraction, scene detection
2. **Analysis** (`src/video_analyser/analysis/`) - Speech transcription, visual analysis, AI feedback
3. **Reports** (`src/video_analyser/reports/`) - JSON/HTML report generation

**Configuration System:**
- Pydantic-based hierarchical configuration (`src/video_analyser/utils/config.py`)
- YAML files with environment variable overrides
- Nested settings: `VIDEO_ANALYSER_TRANSCRIPTION__MODEL=whisper-large`

### Key Design Principles

**Modern Python Toolchain:**
- Uses `uv` instead of pip for fast package management
- `pyproject.toml` only (no setup.py or requirements.txt)
- Strict type checking with basedpyright
- CPU-only dependencies by default, GPU optional with `[gpu]` extra

**Configuration Management:**
- All settings centralized in Pydantic models with validation
- Environment variables override file settings using `__` delimiter
- Multiple config file locations checked automatically

**Testing Strategy:**
- Tests mirror `src/` structure exactly
- Pytest markers for test categorization (video, audio, ai, slow)
- Custom fixtures for video processing tests
- Skip tests when dependencies (ffmpeg, espeak) unavailable

### Development State

**✅ Complete:**
- Project infrastructure and packaging
- Configuration system with full validation
- CLI framework with rich output
- Test infrastructure with video fixtures
- PyPI publication pipeline

**🚧 In Progress (Phase 1 MVP):**
- Core video processing pipeline (`src/core/`)
- Speech transcription with Whisper (`src/analysis/`)
- Report generation system (`src/reports/`)

**📋 Planned:**
- Gradio web interface
- Advanced AI analysis features
- Production deployment configurations

### Critical Files

**Configuration:** `src/video_analyser/utils/config.py` - Complete Pydantic-based config system with validation
**CLI Entry:** `src/video_analyser/cli.py` - Main application interface
**Task Tracking:** `tasks/tasks-prd-phase1-mvp.md` - Detailed implementation roadmap
**Dependencies:** `pyproject.toml` - All project configuration and dependencies

### Video Sample Management

- **Test fixtures** (`tests/fixtures/`): Small videos for automated testing (committed to git)
- **Development samples** (`samples/`): Large realistic videos (gitignored)
- Generate with: `python3 scripts/generate_test_videos.py`
- Requires espeak for speech synthesis, falls back to tone audio otherwise

### Common Gotchas

**Package Installation:** Always use `uv pip install -e ".[dev]"` for development
**Type Checking:** Uses basedpyright (not mypy) in strict mode
**Config Loading:** Environment variables use double underscore: `VIDEO_ANALYSER_PROCESSING__MAX_VIDEO_SIZE_MB`
**Test Running:** Use pytest markers to control which tests run during development
**Video Processing:** All samples currently use synthetic speech via espeak

### GPU Migration (Future)
```bash
# Install GPU-accelerated dependencies
uv pip install -e ".[gpu]" --extra-index-url https://download.pytorch.org/whl/cu121
```
All APIs remain identical - PyTorch automatically detects and uses GPU when available.

## Task Implementation Process

**CRITICAL:** This project follows a strict task-by-task development process:

### Task Processing Protocol
1. **One sub-task at a time** - Never start the next sub-task until user gives explicit permission
2. **Stop and ask** - After completing each sub-task, stop and wait for user approval before continuing
3. **Update task list immediately** - Mark completed sub-tasks `[x]` in `tasks/tasks-prd-phase1-mvp.md`
4. **Parent task completion** - Only mark parent tasks `[x]` when ALL sub-tasks underneath are complete

### Task List Management
- **Primary task list:** `tasks/tasks-prd-phase1-mvp.md` - Contains all Phase 1 MVP implementation tasks
- **Task processing guidelines:** `docs/ai-dev-tasks/3-process-task-list.mdc` - Detailed protocol rules
- **Relevant Files section** - Keep updated with every file created or modified

### Implementation Workflow
1. Check task list to identify next uncompleted sub-task
2. Ask user permission before starting implementation
3. Implement the single sub-task completely
4. Update task list marking sub-task as `[x]` completed
5. Update "Relevant Files" section if new files created/modified
6. Stop and ask user for permission to continue to next sub-task

**Never implement multiple sub-tasks without explicit user approval for each one.**