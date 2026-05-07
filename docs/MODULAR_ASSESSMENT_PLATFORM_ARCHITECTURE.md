# Modular Assessment Platform Architecture

## Vision

Build a **modular assessment platform** where students submit diverse deliverables (videos, documents, code, websites, etc.), the system intelligently extracts standardized content representations, and applies consistent analysis/grading workflows.

Core principle: **Extract once, analyze many ways**
- Each submission format has a dedicated **ingestion lens** that extracts content
- Extracted content flows to appropriate **content analyzers** for insights
- All analyses feed into **grading-lens** for rubric-based scoring

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Assessment Orchestration Layer (feedforward/assessment-bench)
│  - Route submissions by file type
│  - Aggregate results
│  - Manage workflows
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Format Ingestion Layer (LENSES)
│  - Extract structured content from raw files/artifacts
│  - Output standardized data structures
│  - May compose/delegate to other lenses
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Content Analysis Layer (ANALYZERS)
│  - Analyze extracted content
│  - Generate metrics, insights, flags
│  - Reusable across multiple lenses
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  Grading & Feedback Layer
│  - grading-lens: Apply rubric to all extracted content
│  - Generate score + personalized feedback
│  - accessibility-lens: Cross-cutting accessibility analysis
└─────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Format Ingestion Lenses

### Simple Format Lenses (Single Artifact)

#### video-analyser
**Current:** `video-analyser` (being renamed)

- **Input:** Video files (mp4, mov, mkv, webm)
- **Extraction:**
  - Frame sampling
  - Audio track extraction
  - Speech-to-text transcription
  - Scene/shot detection
  - Visual quality metrics
- **Output:**
  ```json
  {
    "frames": [{"timestamp": 0.5, "image": "base64"}],
    "audio_track": "audio_data",
    "transcript": {
      "text": "...",
      "segments": [{"start": 0, "end": 5, "speaker": "...", "text": "..."}],
      "language": "en"
    },
    "visual_metrics": {
      "quality_score": 0.85,
      "scene_changes": [0.5, 2.3, 5.1],
      "detected_objects": [...]
    },
    "metadata": {"duration": 120, "resolution": "1920x1080"}
  }
  ```
- **Delegate to:**
  - `text-analyzer` on transcript
  - `image-analyzer` on frames
  - `audio-analyzer` on audio track (if audio-lens not used)

---

#### audio-lens
- **Input:** Audio files (mp3, wav, flac, m4a, podcast feeds)
- **Extraction:**
  - Speech-to-text transcription
  - Speaker diarization
  - Audio quality analysis
  - Noise/silence detection
  - Metadata extraction
- **Output:**
  ```json
  {
    "transcript": {
      "text": "...",
      "segments": [{"start": 0, "end": 5, "speaker": "Speaker 1", "text": "..."}],
      "language": "en",
      "confidence": 0.92
    },
    "audio_metrics": {
      "quality_score": 0.88,
      "noise_level": 0.15,
      "silence_periods": [{"start": 10, "end": 12}],
      "dynamic_range": 60
    },
    "speakers": ["Speaker 1", "Speaker 2"],
    "metadata": {"duration": 180, "bitrate": 128}
  }
  ```
- **Delegate to:**
  - `text-analyzer` on transcript

---

#### image-lens
- **Input:** Image files (jpg, png, gif, svg, webp, screenshots)
- **Extraction:**
  - OCR (optical character recognition) for text
  - Object detection
  - Visual quality metrics
  - Composition analysis
  - Accessibility metadata (alt text)
- **Output:**
  ```json
  {
    "extracted_text": "Text found in image via OCR",
    "visual_analysis": {
      "quality_score": 0.82,
      "composition": {
        "rule_of_thirds": 0.7,
        "balance": 0.8,
        "focus_areas": [[100, 200], [400, 350]]
      },
      "detected_objects": [
        {"label": "person", "confidence": 0.95, "bbox": [10, 20, 100, 150]},
        {"label": "document", "confidence": 0.87, "bbox": [150, 50, 300, 400]}
      ],
      "colors": {"dominant": "#FF5733", "palette": [...]},
      "blur_score": 0.05,
      "brightness": 0.65
    },
    "accessibility": {
      "suggested_alt_text": "A person holding a document",
      "text_density": 0.15
    }
  }
  ```
- **Delegate to:**
  - `text-analyzer` on extracted_text (if code detected, also code-analyzer)
  - `image-analyzer` on visual metrics

---

#### document-lens
Modules: `pdf`, `docx`, `txt`, `md`, `rtf`, `odt`

- **Input:** Document files (pdf, docx, txt, markdown, etc.)
- **Extraction (per format module):**
  - Text extraction
  - Structural metadata (headings, sections, lists)
  - Embedded images
  - Tables and formatted data
  - Links and references
- **Output:**
  ```json
  {
    "text": "Full extracted text content",
    "structure": {
      "headings": [{"level": 1, "text": "Introduction", "position": 0}],
      "sections": [{"title": "Methods", "content": "..."}],
      "tables": [{"headers": [...], "rows": [...]}],
      "lists": [{"type": "ordered", "items": [...]}]
    },
    "embedded_assets": {
      "images": [{"id": "img_1", "base64": "...", "position": 450}],
      "embedded_files": [...]
    },
    "metadata": {
      "author": "...",
      "created": "2024-01-15",
      "word_count": 2500,
      "pages": 10
    }
  }
  ```
- **Delegate to:**
  - `text-analyzer` on full text + structure
  - `image-analyzer` on embedded images

---

#### slide-lens
Modules: `pptx`, `odp`, `pdf` (slide decks)

- **Input:** Slide presentation files (pptx, odp, keynote, pdf with slides)
- **Extraction:**
  - Per-slide breakdown
  - Text from slides (titles, bullets, notes)
  - Embedded images and media
  - Slide layout/design metadata
  - Speaker notes
- **Output:**
  ```json
  {
    "slides": [
      {
        "number": 1,
        "title": "Introduction",
        "text": "Full text content on slide",
        "structure": {
          "headings": [...],
          "bullets": [...]
        },
        "images": [{"id": "img_1", "base64": "...", "position": "center"}],
        "layout": "title_and_content",
        "speaker_notes": "Additional speaker notes"
      }
    ],
    "metadata": {
      "title": "Presentation Title",
      "author": "...",
      "slide_count": 20,
      "created": "2024-01-15"
    }
  }
  ```
- **Delegate to:**
  - `text-analyzer` on all extracted text
  - `image-analyzer` on slide images
  - Process slides both individually and as a sequence

---

#### code-lens
Modules: `python`, `javascript`, `php`, `java`, etc.

- **Input:** Single code files (.py, .js, .java, .rb, etc.) or Jupyter notebooks (.ipynb)
- **Extraction:**
  - Source code as structured AST (abstract syntax tree)
  - Comments and docstrings
  - Function/class definitions
  - Imports and dependencies
  - Code metrics (cyclomatic complexity, lines of code, etc.)
  - For notebooks: cells, outputs, visualizations
- **Output:**
  ```json
  {
    "code": "Full source code",
    "language": "python",
    "structure": {
      "imports": [{"module": "numpy", "alias": "np"}],
      "functions": [
        {
          "name": "analyze_data",
          "line": 45,
          "docstring": "Analyzes input data",
          "parameters": ["data", "threshold"],
          "returns": "dict"
        }
      ],
      "classes": [...],
      "comments": [{"line": 10, "text": "Initialize variables"}]
    },
    "metrics": {
      "lines_of_code": 245,
      "cyclomatic_complexity": 8,
      "function_count": 5,
      "class_count": 2
    },
    "for_notebooks": {
      "cells": [
        {
          "index": 0,
          "type": "code",
          "source": "import numpy as np",
          "output": "..."
        }
      ],
      "visualizations": [{"type": "plot", "location": "cell_5"}]
    }
  }
  ```
- **Delegate to:**
  - `code-analyzer` on structure and metrics
  - `text-analyzer` on docstrings and comments

---

#### web-lens
Modules: `react`, `wordpress`, `vanilla_html`, `url`

**Input:**
- Local source files (React components, WordPress theme files, HTML/CSS/JS)
- Live URLs (render and analyze)

**Extraction:**
- Source code structure (component hierarchy for React, theme structure for WordPress)
- Rendered visual output (screenshots)
- DOM/accessibility structure
- Static assets (images, stylesheets)
- Configuration and dependencies
- Performance metrics

**Output:**
```json
{
  "source_analysis": {
    "code_files": [...],  // Routes to code-lens
    "images": [...],      // Routes to image-lens
    "config": {...}       // Routes to document-lens
  },
  "rendered_output": {
    "screenshots": [...],  // Routes to image-lens
    "viewport_sizes": ["mobile", "tablet", "desktop"]
  },
  "accessibility": {
    "wcag_issues": [...],
    "alt_text_coverage": 0.85,
    "color_contrast_issues": [...]
  },
  "performance": {
    "lighthouse_score": 87,
    "page_load_time": 1.2,
    "bundle_size": 245000
  },
  "structure": {
    "for_react": {
      "components": [...],
      "component_tree": "..."
    },
    "for_wordpress": {
      "theme": "twentytwentyfour",
      "plugins": [...],
      "pages": [...]
    }
  }
}
```

**Delegate to:**
- `code-lens` on source files
- `image-analyzer` on screenshots
- `text-analyzer` on extracted content

---

### Composite/Orchestration Lenses (Multi-Artifact)

#### notebook-lens
**Input:** Jupyter notebooks (.ipynb)

Note: Could be handled by `code-lens` as a special case, but deserves own lens for rich cell-by-cell analysis.

- **Extraction:**
  - Code cells (delegate to code-lens features)
  - Markdown cells (narrative)
  - Output cells (visualizations, tables, text)
  - Execution metadata (kernel, execution time)
- **Output:** Combines code and document structure
- **Delegate to:**
  - `code-lens` for code cells
  - `document-lens` for markdown cells
  - `image-analyzer` for visualizations/plots

---

#### repo-lens
**Input:** GitHub repository URLs or local repo directories

**Process:**
1. Clone/download repository
2. Scan file types
3. Route files to appropriate lenses:
   - `.py` files → `code-lens`
   - `.ipynb` files → `notebook-lens`
   - `.js`, `.jsx`, `.tsx` → `code-lens`
   - `README.md`, documentation → `document-lens`
   - Images in `/assets` → `image-lens`
   - If React project → also `web-lens`
   - If Django/Flask → also `web-lens`

**Output:**
```json
{
  "repository_metadata": {
    "url": "https://github.com/...",
    "language": "python",
    "primary_languages": ["python", "javascript"],
    "stars": 150,
    "last_updated": "2024-01-15"
  },
  "structure": {
    "files_analyzed": 45,
    "directories": {...}
  },
  "analysis_by_lens": {
    "code_lens_results": [...],
    "notebook_lens_results": [...],
    "document_lens_results": [...],
    "image_lens_results": [...]
  },
  "aggregated_metrics": {
    "total_lines_of_code": 5000,
    "test_coverage": 0.75,
    "dependencies": 25
  }
}
```

**Delegate to:** Multiple lenses in sequence

---

## Layer 2: Content Analyzers

These are **reusable tools** that analyze extracted content. Called by multiple lenses as appropriate.

### text-analyzer
**Input:** Text content (transcripts, extracted text, docstrings, comments, etc.)

**Analysis:**
- Readability metrics (Flesch-Kincaid, Gunning Fog, SMOG)
- Grammar and spelling
- Sentiment analysis
- Key term extraction
- Topic modeling
- Argument structure analysis
- Citation/reference analysis
- Vocabulary diversity

**Output:**
```json
{
  "readability": {
    "grade_level": 10.5,
    "flesch_kincaid": "10th grade",
    "reading_time_minutes": 5
  },
  "quality": {
    "grammar_score": 0.92,
    "spelling_errors": 2,
    "clarity_score": 0.85
  },
  "content": {
    "key_terms": ["machine learning", "neural network", ...],
    "topics": [{"name": "AI", "coverage": 0.35}, ...],
    "argument_structure": {...},
    "citations": 12
  },
  "vocabulary": {
    "unique_terms": 450,
    "vocabulary_richness": 0.78,
    "technical_term_ratio": 0.15
  }
}
```

---

### image-analyzer
**Input:** Visual content (images, screenshots, extracted from video frames, slide images, etc.)

**Analysis:**
- Composition (rule of thirds, balance, focal points)
- Quality metrics (sharpness, noise, blur)
- Accessibility (alt text quality, contrast ratios)
- Content descriptiveness
- Visual hierarchy
- Color analysis
- Object recognition insights

**Output:**
```json
{
  "composition": {
    "rule_of_thirds_adherence": 0.7,
    "balance_score": 0.8,
    "focal_points": [[100, 200], [400, 350]],
    "visual_hierarchy": "clear"
  },
  "quality": {
    "blur_score": 0.05,
    "noise_level": 0.08,
    "brightness": 0.65,
    "contrast": 0.75,
    "overall_quality": 0.82
  },
  "accessibility": {
    "suggested_alt_text": "A person presenting at a conference",
    "color_contrast_adequate": true,
    "text_in_image_readable": true
  },
  "content": {
    "description_quality": 0.85,
    "information_density": "moderate",
    "objects_identified": [...],
    "visual_appeal": "high"
  }
}
```

---

### code-analyzer
**Input:** Code structure and metrics from code-lens

**Analysis:**
- Code quality metrics (complexity, duplication, maintainability)
- Best practices adherence (naming, structure, patterns)
- Style consistency
- Documentation quality (docstrings, comments)
- Test coverage (if available)
- Performance anti-patterns
- Security issues (basic static analysis)
- Functionality assessment

**Output:**
```json
{
  "quality_metrics": {
    "maintainability_index": 75,
    "cyclomatic_complexity": 8,
    "code_duplication_ratio": 0.08,
    "test_coverage": 0.72
  },
  "best_practices": {
    "naming_conventions": 0.90,
    "function_length": 0.85,
    "class_cohesion": 0.88,
    "documentation_completeness": 0.75
  },
  "style": {
    "consistency_score": 0.92,
    "violations": [{"line": 45, "rule": "line_too_long"}]
  },
  "issues": {
    "potential_bugs": [...],
    "security_concerns": [...],
    "performance_issues": [...]
  }
}
```

---

### accessibility-analyzer
**Cross-cutting analyzer** - can run on outputs from multiple lenses

**Input:**
- Document structure (headings, lists, links)
- Images with alt text
- Color choices
- Interactive elements
- Video captions
- Code comments

**Analysis (WCAG 2.1 compliance):**
- Color contrast ratios
- Alt text presence and quality
- Heading hierarchy
- Link descriptiveness
- Caption availability (video)
- Keyboard navigation support
- Screen reader compatibility

**Output:**
```json
{
  "wcag_level": "AA",
  "score": 0.85,
  "issues": [
    {
      "level": "error",
      "criterion": "1.4.3 Contrast (Minimum)",
      "location": "slide_3",
      "suggestion": "Increase contrast ratio from 3:1 to 4.5:1"
    }
  ],
  "strengths": [
    "Good heading hierarchy",
    "Descriptive link text"
  ]
}
```

---

## Layer 3: Grading & Feedback Lenses

### grading-lens
**Input:**
- Extracted content from all applicable lenses
- All analyzer outputs
- Rubric (structured criteria + point values)

**Process:**
1. Map extracted content/analyses to rubric dimensions
2. Score each dimension
3. Calculate final grade
4. Generate targeted feedback
5. Optionally route to accessibility-analyzer for additional notes

**Output:**
```json
{
  "submission_id": "...",
  "rubric_type": "general",
  "total_score": 87,
  "scores_by_criteria": [
    {
      "criterion": "Content Clarity",
      "weight": 0.3,
      "score": 90,
      "feedback": "Excellent explanation of key concepts..."
    },
    {
      "criterion": "Visual Quality",
      "weight": 0.2,
      "score": 85,
      "feedback": "Good use of visuals, slight audio issues..."
    }
  ],
  "feedback": {
    "summary": "Overall strong submission with minor areas for improvement",
    "strengths": ["Clear communication", "Well-structured"],
    "improvements": ["Add citations", "Check audio levels"],
    "detail_level": "summary",  // "short", "summary", or "long"
    "audience": "student"        // "student" or "teacher"
  },
  "accessibility_notes": "3 images missing alt text",
  "suggested_resources": [...]
}
```

---

### accessibility-lens
**Cross-cutting lens** for accessibility compliance assessment

Can be run:
- During grading (as part of grading-lens)
- Standalone (audit mode)
- Per-submission-type

**Output:** See accessibility-analyzer above

---

## Workflow Patterns

### Pattern 1: Simple Document Submission
```
Student submits: essay.docx

document-lens extracts:
  ├─ text
  ├─ structure
  └─ images

text-analyzer processes text
image-analyzer processes images
accessibility-analyzer checks WCAG compliance

grading-lens applies rubric:
  ├─ Content quality (text-analyzer + manual)
  ├─ Organization (document structure)
  ├─ Visual presentation (image-analyzer)
  └─ Accessibility (accessibility-analyzer)

Output: Score + feedback
```

---

### Pattern 2: Video Presentation
```
Student submits: presentation.mp4

video-analyser extracts:
  ├─ frames → image-analyzer
  ├─ audio track
  ├─ transcript → text-analyzer
  └─ visual metrics

audio-lens (optional, if separate audio analysis needed):
  └─ additional audio analysis

image-analyzer processes:
  ├─ frame composition
  ├─ visual quality
  └─ on-screen text (OCR)

text-analyzer processes:
  ├─ spoken content (from transcript)
  ├─ on-screen text
  └─ readability of visual text

accessibility-analyzer:
  ├─ Caption availability
  ├─ Color contrast
  └─ Text size (on screen)

grading-lens applies rubric:
  ├─ Content delivery (text-analyzer on transcript)
  ├─ Visual quality (image-analyzer on frames)
  ├─ Pacing/structure (video-analyser metrics)
  ├─ Audio quality (audio-analyzer)
  └─ Accessibility (accessibility-analyzer)

Output: Score + feedback
```

---

### Pattern 3: Code Project (Repository)
```
Student submits: GitHub repo URL

repo-lens discovers and routes files:
  ├─ .py files → code-lens → code-analyzer
  ├─ README.md → document-lens → text-analyzer
  ├─ Jupyter notebooks → notebook-lens → (code + text + image analyzers)
  └─ /docs folder → document-lens

Aggregates all results:
  ├─ Code quality metrics
  ├─ Documentation completeness
  ├─ Functionality (if tests included)
  └─ Accessibility (if web project)

grading-lens applies rubric:
  ├─ Code quality (code-analyzer)
  ├─ Documentation (text-analyzer + structure)
  ├─ Testing (if test files present)
  └─ Functionality (manual + test results)

Output: Score + feedback
```

---

### Pattern 4: Web Project Submission
```
Student submits: React app (GitHub repo or uploaded files)

web-lens processes:
  ├─ Source code → code-lens → code-analyzer
  ├─ Renders pages → screenshots → image-analyzer
  ├─ Extracts DOM structure
  └─ Runs accessibility checks

image-analyzer processes:
  ├─ UI composition
  ├─ Visual design quality
  └─ Screenshot clarity

code-analyzer processes:
  ├─ Component architecture
  ├─ Code quality
  └─ Best practices

accessibility-analyzer processes:
  ├─ WCAG compliance
  ├─ Color contrast
  ├─ Alt text (if images)
  └─ Keyboard navigation

grading-lens applies rubric:
  ├─ Functionality (manual + code analysis)
  ├─ Code quality (code-analyzer)
  ├─ UI/UX design (image-analyzer + accessibility)
  ├─ Accessibility (accessibility-analyzer)
  └─ Deployment/documentation (document-lens)

Output: Score + feedback
```

---

### Pattern 5: Multi-Artifact Portfolio
```
Student submits: portfolio/ with:
  ├─ presentation.mp4
  ├─ analysis.pdf
  ├─ code.py
  └─ screenshots/

assessment-bench:
1. Detects file types
2. Routes each to appropriate lens:
   ├─ presentation.mp4 → video-analyser
   ├─ analysis.pdf → document-lens
   ├─ code.py → code-lens
   └─ screenshots/ → image-lens
3. Aggregates all results

Each lens extracts content
Each analyzer processes its content type
accessibility-analyzer runs across all artifacts

grading-lens processes aggregated results:
  ├─ Scores each component
  ├─ Looks for coherence across artifacts
  ├─ Weights by rubric

Output: Overall score + per-component feedback
```

---

## Implementation Phases

### Phase 1: Core Foundation (Months 1-2)
**Goal:** Establish architecture and implement core lenses for primary use case

**Implement:**
- ✅ `video-analyser` (already complete as video-analyser → video-analyser)
- ✅ `code-lens` (python module, basic)
- `text-analyzer`
- `grading-lens` (basic rubric support)
- `accessibility-analyzer` (text-based checks)

**Not yet:**
- Other lenses, composite orchestration

**Test with:** Student video presentation submissions

**Deliverable:**
- Clear, working flow: video submission → video-analyser → text-analyzer → grading-lens → score + feedback

---

### Phase 2: Expand Ingestion (Months 3-4)
**Goal:** Add support for document and image submissions

**Add Lenses:**
- `document-lens` (pdf, docx, txt modules)
- `image-lens`
- `slide-lens` (pptx)

**Enhance:**
- `image-analyzer` (composition, quality metrics)
- `text-analyzer` (readability, sentiment)
- `grading-lens` (weighted rubrics)

**Test with:** Multi-format submissions (video + document, slides + code, etc.)

**Deliverable:**
- Students can submit videos, documents, images, slides
- Each gets appropriate analysis
- Unified grading workflow

---

### Phase 3: Code & Web Projects (Months 5-6)
**Goal:** Support code and web development submissions

**Add Lenses:**
- `code-lens` (expand: javascript, java modules)
- `notebook-lens`
- `web-lens` (react, vanilla_html modules)
- `repo-lens` (GitHub integration)

**Enhance:**
- `code-analyzer` (metrics, quality, testing)
- `web-lens` (accessibility, performance metrics)
- `grading-lens` (code rubrics)

**Test with:** CS course: coding assignments, web dev projects, repos

**Deliverable:**
- Code and web project submissions fully supported
- Automated code quality and accessibility feedback

---

### Phase 4: Advanced Features & Composition (Months 7-8)
**Goal:** Full orchestration, advanced analysis, workflow optimization

**Add:**
- `audio-lens` (full implementation)
- `accessibility-lens` (advanced WCAG checks)
- Lens composition (repo-lens calling code-lens, etc.)
- Multi-artifact portfolio handling
- Performance optimization (parallel processing)

**Enhance:**
- All analyzers (additional metrics, insights)
- Feedback generation (audience-specific, detail levels)
- Rubric management (complex hierarchical rubrics)

**Test with:** Real course deployments across formats

**Deliverable:**
- Fully modular assessment platform
- Handles any submission format through composition
- Advanced AI-assisted feedback

---

### Phase 5+: Specialized Lenses & Domain-Specific Analysis
**Later phases:**
- `presentation-analysis` (specialized for speeches/talks)
- `podcast-lens` (feed parsing, guest detection)
- `dataset-lens` (CSV, JSON, database analysis)
- Custom domain lenses (domain-specific rubrics)
- ML-based assessment (essay scoring, plagiarism detection)

---

## Key Design Decisions

### 1. Lens Granularity
- **One lens per format** (video-analyser, audio-lens, document-lens, etc.)
- **Not** one lens per file type (pptx handled by slide-lens, not separate pptx-lens)
- **Composite lenses** (repo-lens) orchestrate and delegate

### 2. Analyzer Reusability
- Analyzers are **standalone, format-agnostic**
- `text-analyzer` works on any extracted text (from video transcript, document, code comment, etc.)
- No analyzer knows which lens produced its input
- Enables composition and flexibility

### 3. Format Standardization
- Each lens outputs **structured, standardized JSON**
- Downstream processing doesn't need to know lens origin
- Enables swapping implementations without breaking analyzers/grading

### 4. Delegation vs. Composition
- **Simple delegation:** video-analyser calls text-analyzer on transcript
- **Orchestration:** repo-lens discovers files, routes each to appropriate lens
- Clear pattern for adding new complexity

### 5. Pragmatic Scope
- **In scope:** Extracting and analyzing content for assessment
- **Out of scope:** Building submission UI, managing courses, etc.
- These are handled by `assessment-bench` and `feedforward`

---

## Technology Stack (Suggested)

### Lens Implementations
- **video-analyser:** FFmpeg, Whisper (transcription), OpenCV (frame analysis)
- **audio-lens:** Whisper, librosa, pyannote (speaker diarization)
- **image-lens:** PIL, OpenCV, Tesseract (OCR), YOLO (object detection)
- **document-lens:** PyPDF, python-docx, markdown, pypandoc
- **slide-lens:** python-pptx, pdf2image
- **code-lens:** AST parsing, tree-sitter, radon (complexity)
- **web-lens:** Selenium/Playwright (rendering), BeautifulSoup (parsing)
- **repo-lens:** GitPython, GitHub API

### Analyzers
- **text-analyzer:** spaCy, NLTK, transformers (sentiment), textstat (readability)
- **image-analyzer:** OpenCV, PIL, scikit-image, YOLO
- **code-analyzer:** radon, pylint, flake8, vulture
- **accessibility-analyzer:** axe-core, webaim

### Orchestration
- **assessment-bench:** Orchestrate lens routing
- **feedforward:** Manage assessments, collect feedback
- **grading-lens:** Apply rubrics, generate feedback (Claude API for advanced feedback)

---

## Future Considerations

### Scalability
- Parallel lens processing for multi-artifact submissions
- Caching of extracted content
- Distributed processing for heavy computations (video processing, large repos)

### Extensibility
- Plugin architecture for custom lenses
- Domain-specific rubric plugins
- Custom analyzer plugins

### Integration
- LMS integration (Canvas, Blackboard, Moodle)
- GitHub Classroom integration
- Slack notifications for feedback
- Email delivery of reports

### ML Enhancements
- Fine-tuned models for educational content
- Plagiarism detection
- Learning outcome mapping
- Predictive feedback (suggest improvements before submission)

---

## Glossary

- **Lens:** Ingestion component that extracts standardized content from a specific format
- **Analyzer:** Reusable analysis tool that processes extracted content
- **Rubric:** Scoring criteria with point values and dimension weights
- **Grading-lens:** Applies rubric to extracted content and analyzers to generate score/feedback
- **assessment-bench:** Orchestration system that routes submissions to appropriate lenses
- **feedforward:** Overall assessment management and feedback platform
