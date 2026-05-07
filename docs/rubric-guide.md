# Rubric Management Guide

This guide explains how to use, create, and manage rubrics in Deep-Brief for assessment and grading presentations.

## Table of Contents

- [Overview](#overview)
- [Default Rubrics](#default-rubrics)
- [CLI Commands](#cli-commands)
- [Creating Custom Rubrics](#creating-custom-rubrics)
- [Python API](#python-api)
- [JSON Format](#json-format)
- [Examples](#examples)

## Overview

Rubrics are used to evaluate presentations against specific criteria. Deep-Brief includes:

- **4 built-in default rubrics** for common presentation types
- **Flexible rubric system** for creating custom rubrics
- **File-based storage** for easy sharing and version control
- **Scoring scales** from 1-5, percentages, or custom ranges

## Default Rubrics

Deep-Brief includes 4 pre-built default rubrics for common scenarios:

### 1. Academic Presentation
For research presentations, conference talks, and academic seminars.

**Categories:**
- Content Quality (research depth, thesis clarity, supporting evidence)
- Delivery & Presentation (clarity, engagement, visual aids)
- Organization & Structure (logical flow, time management)

**Usage:**
```bash
video-analyser rubric create --type academic
```

### 2. Business Pitch
For startup pitches, investor presentations, and business proposals.

**Categories:**
- Business Model & Value Prop (problem statement, value proposition, market understanding)
- Financial Viability (revenue model, financial projections)
- Presentation Quality (persuasiveness, confidence, visual design)

**Usage:**
```bash
video-analyser rubric create --type business
```

### 3. Teaching Demonstration
For teaching job interviews, lesson observations, and pedagogical presentations.

**Categories:**
- Pedagogical Approach (learning objectives, instructional design, active learning)
- Content & Communication (accuracy, clarity, use of examples)
- Student Engagement & Classroom Management

**Usage:**
```bash
video-analyser rubric create --type teaching
```

### 4. General Presentation
A flexible rubric for any type of presentation.

**Categories:**
- Content (relevance, completeness, accuracy)
- Delivery (clarity, confidence, engagement)
- Organization (logical flow, transitions)
- Visuals (visual quality and design)

**Usage:**
```bash
video-analyser rubric create --type general
```

## CLI Commands

### List All Rubrics

Show all available default and custom rubrics:

```bash
video-analyser rubric list
```

**Output example:**
```
Available Default Rubrics
(Can be created with: video-analyser rubric create --type <type>)

  ACADEMIC
    Evaluate academic presentations at conferences, seminars, or classroom settings
    Categories: 3
    Criteria: 8

  BUSINESS
    Evaluate business pitches, startup presentations, and investor pitches
    Categories: 3
    Criteria: 7

...

Custom Rubrics

  My Custom Rubric [Custom]
    ID: 550e8400-e29b-41d4-a716-446655440000
    Categories: 4
    Created: 2025-11-12
```

### Show Rubric Details

Display detailed information about a specific rubric:

```bash
video-analyser rubric show --id <rubric_id>
```

**Example:**
```bash
video-analyser rubric show --id 550e8400-e29b-41d4-a716-446655440000
```

**Output:**
```
Academic Presentation

Evaluate academic presentations at conferences, seminars, or classroom settings

ID: 550e8400-e29b-41d4-a716-446655440000
Score Range: 1-5
Tags: academic, conference, research

Categories:

  Content Quality (weight: 2.0)
  Evaluation of presentation content
  Criteria:
    • Research Depth (weight: 1.5)
      Demonstrates thorough understanding of research methodology and findings
    • Clear Thesis/Purpose (weight: 1.5)
      Main research question or contribution is clearly stated
    • Supporting Evidence (weight: 1.0)
      Uses relevant data, citations, and examples to support claims

...
```

### Create Rubric from Template

Create a new rubric based on a default template:

```bash
video-analyser rubric create --type <type>
```

**Available types:** `academic`, `business`, `teaching`, `general`

**Example:**
```bash
video-analyser rubric create --type business
```

**Output:**
```
✓ Created rubric: Business Pitch
  ID: 550e8400-e29b-41d4-a716-446655440000
  Categories: 3
  Criteria: 7

Rubric saved to: rubrics/550e8400-e29b-41d4-a716-446655440000.json
```

### Export Rubric to JSON

Export a rubric to a JSON file for sharing or version control:

```bash
video-analyser rubric export --id <rubric_id> --output <file_path>
```

**Example:**
```bash
video-analyser rubric export \
  --id 550e8400-e29b-41d4-a716-446655440000 \
  --output ./rubrics/business-pitch-rubric.json
```

### Delete Rubric

Remove a custom rubric:

```bash
video-analyser rubric delete --id <rubric_id>
```

**Example:**
```bash
video-analyser rubric delete --id 550e8400-e29b-41d4-a716-446655440000
```

(You'll be prompted to confirm deletion)

## Creating Custom Rubrics

### Using Python API

Create a custom rubric programmatically:

```python
from video_analyser.analysis.rubric_system import RubricBuilder, RubricRepository

# Create rubric
builder = RubricBuilder(
    name="My Custom Rubric",
    description="Custom evaluation rubric for my presentations"
)

# Add Content category
content = builder.add_category(
    name="Content Quality",
    description="Evaluation of presentation content",
    weight=2.0
)
content.add_criterion(
    name="Clarity of Main Points",
    description="Main ideas are clearly articulated",
    weight=1.5,
    scoring_guide="1=Unclear, 3=Adequate, 5=Very clear"
)
content.add_criterion(
    name="Use of Examples",
    description="Relevant examples support main points",
    weight=1.0,
    scoring_guide="1=No examples, 3=Some examples, 5=Excellent examples"
)

# Add Delivery category
delivery = builder.add_category(
    name="Delivery",
    weight=1.5
)
delivery.add_criterion(name="Clarity", weight=1.0)
delivery.add_criterion(name="Engagement", weight=1.0)

# Set custom scoring scale
builder.set_scoring_scale(
    min_score=1,
    max_score=5,
    labels={
        1: "Poor",
        2: "Below Average",
        3: "Average",
        4: "Good",
        5: "Excellent"
    }
)

# Build and save
rubric = builder.build(is_template=False, tags=["custom"])
repo = RubricRepository("rubrics/")
repo.save(rubric)

print(f"Created rubric with ID: {rubric.id}")
```

### Using JSON File

Create a rubric by writing a JSON file:

```json
{
  "id": "my-custom-rubric-123",
  "name": "My Custom Rubric",
  "description": "Custom evaluation rubric for my presentations",
  "categories": [
    {
      "id": "cat-1",
      "name": "Content Quality",
      "description": "Evaluation of presentation content",
      "weight": 2.0,
      "criteria": [
        {
          "id": "crit-1",
          "name": "Clarity of Main Points",
          "description": "Main ideas are clearly articulated",
          "weight": 1.5,
          "scoring_guide": "1=Unclear, 3=Adequate, 5=Very clear"
        },
        {
          "id": "crit-2",
          "name": "Use of Examples",
          "description": "Relevant examples support main points",
          "weight": 1.0,
          "scoring_guide": "1=No examples, 3=Some examples, 5=Excellent examples"
        }
      ]
    },
    {
      "id": "cat-2",
      "name": "Delivery",
      "weight": 1.5,
      "criteria": [
        {
          "id": "crit-3",
          "name": "Clarity",
          "weight": 1.0
        },
        {
          "id": "crit-4",
          "name": "Engagement",
          "weight": 1.0
        }
      ]
    }
  ],
  "scoring_scale": {
    "id": "default",
    "name": "1-5 Scale",
    "min_score": 1,
    "max_score": 5,
    "labels": {
      "1": "Poor",
      "2": "Below Average",
      "3": "Average",
      "4": "Good",
      "5": "Excellent"
    }
  },
  "is_template": false,
  "tags": ["custom"]
}
```

Save this as `rubrics/<rubric_id>.json` and Deep-Brief will automatically load it.

## Python API

### RubricBuilder

Fluent API for creating rubrics:

```python
from video_analyser.analysis.rubric_system import RubricBuilder

builder = RubricBuilder(name="Rubric Name", description="Optional description")

# Add categories
category = builder.add_category(name="Category Name", weight=1.5)

# Add criteria to category
category.add_criterion(
    name="Criterion Name",
    description="Detailed description",
    weight=1.0,
    scoring_guide="Guidance for scorers"
)

# Set custom scoring scale
builder.set_scoring_scale(
    min_score=1,
    max_score=5,
    labels={1: "Poor", 5: "Excellent"}
)

# Build final rubric
rubric = builder.build(is_template=False, tags=["custom"])
```

### RubricRepository

File-based storage for rubrics:

```python
from video_analyser.analysis.rubric_system import RubricRepository

# Initialize repository
repo = RubricRepository("rubrics/")

# Save a rubric
repo.save(rubric)

# Load a rubric
loaded = repo.load("rubric-id")

# List all rubrics
all_rubrics = repo.list_rubrics()

# List only templates
templates = repo.list_rubrics(template_only=True)

# Search rubrics
results = repo.search("academic")

# Delete a rubric
repo.delete("rubric-id")
```

### RubricScorer

Score presentations against a rubric:

```python
from video_analyser.analysis.rubric_system import RubricScorer

scorer = RubricScorer(rubric)

# Score all categories
scores = {
    "cat-1": {"crit-1": 4, "crit-2": 5},
    "cat-2": {"crit-3": 3, "crit-4": 4}
}

result = scorer.score_all_categories(scores)

print(f"Overall Score: {result.overall_percentage}%")
for cat_score in result.category_scores:
    print(f"  {cat_score.category_name}: {cat_score.category_percentage}%")
```

## JSON Format

### Rubric Structure

```json
{
  "id": "unique-rubric-id",
  "name": "Rubric Name",
  "description": "Optional description",
  "version": 1,
  "categories": [
    {
      "id": "cat-1",
      "name": "Category Name",
      "description": "Optional category description",
      "weight": 1.5,
      "criteria": [
        {
          "id": "crit-1",
          "name": "Criterion Name",
          "description": "Detailed description",
          "weight": 1.0,
          "scoring_guide": "Guidance for scorers",
          "level_descriptions": {
            "1": "Poor performance",
            "5": "Excellent performance"
          }
        }
      ]
    }
  ],
  "scoring_scale": {
    "id": "default",
    "name": "1-5 Scale",
    "min_score": 1,
    "max_score": 5,
    "labels": {
      "1": "Poor",
      "2": "Below Average",
      "3": "Average",
      "4": "Good",
      "5": "Excellent"
    }
  },
  "is_template": false,
  "tags": ["tag1", "tag2"],
  "created_by": "Optional creator name",
  "created_at": "2025-11-12T10:30:00+00:00",
  "modified_at": "2025-11-12T10:30:00+00:00"
}
```

## Examples

### Example 1: Create and Use a Teaching Rubric

```bash
# Create a teaching rubric
video-analyser rubric create --type teaching

# List to see the ID
video-analyser rubric list

# Show details
video-analyser rubric show --id <rubric-id>

# Export for sharing
video-analyser rubric export --id <rubric-id> --output teaching-rubric.json
```

### Example 2: Create Custom Research Rubric

```python
from video_analyser.analysis.rubric_system import RubricBuilder, RubricRepository

builder = RubricBuilder(
    name="Research Presentation Rubric",
    description="For evaluating research poster presentations"
)

# Research Content
research = builder.add_category("Research Content", weight=3.0)
research.add_criterion("Research Question", weight=1.5)
research.add_criterion("Methodology", weight=1.5)
research.add_criterion("Results & Analysis", weight=2.0)
research.add_criterion("Conclusions", weight=1.0)

# Visual Design
design = builder.add_category("Visual Design", weight=1.5)
design.add_criterion("Layout & Organization", weight=1.0)
design.add_criterion("Typography & Readability", weight=0.75)
design.add_criterion("Visual Elements", weight=0.75)

# Presentation Skills
skills = builder.add_category("Presentation Skills", weight=1.5)
skills.add_criterion("Verbal Communication", weight=1.0)
skills.add_criterion("Engagement with Audience", weight=0.75)
skills.add_criterion("Knowledge & Confidence", weight=0.75)

rubric = builder.build(tags=["research", "poster"])
repo = RubricRepository("rubrics/")
repo.save(rubric)
```

### Example 3: Use Rubric in Assessment

```python
from video_analyser.reports.assessment_session import AssessmentSession
from video_analyser.analysis.rubric_system import RubricRepository

# Load assessment session
session = AssessmentSession(
    analysis_id="video_123",
    video_file_path="/path/to/video.mp4",
    assessor_name="Dr. Smith"
)

# Load and apply rubric
repo = RubricRepository("rubrics/")
rubric = repo.load("research-rubric-id")

# Score the presentation
scores = {
    "research-cat": {
        "research-question": 5,
        "methodology": 4,
        "results": 5,
        "conclusions": 4
    },
    "design-cat": {
        "layout": 5,
        "typography": 4,
        "visuals": 4
    },
    "skills-cat": {
        "communication": 4,
        "engagement": 5,
        "knowledge": 5
    }
}

session.apply_rubric(rubric, scores)

# Add detailed feedback
session.add_criterion_feedback(
    criterion_id="research-question",
    score=5,
    feedback="Excellent, clear research question with strong motivation",
    evidence_timestamps=[(0.5, 30.0)]
)

# Finalize
session.finalize()
```

## Best Practices

1. **Start with defaults**: Begin with a default rubric and modify if needed
2. **Clear criteria**: Write specific, measurable criteria
3. **Consistent weights**: Use weights to reflect importance
4. **Include guides**: Provide scoring guides to help evaluators
5. **Version control**: Save rubrics to version control
6. **Share templates**: Create templates for commonly used rubrics
7. **Document tags**: Use tags to organize and find rubrics easily

## Tips

- **Exporting**: Export rubrics to JSON for easy sharing and version control
- **Searching**: Use `repo.search()` to find rubrics by name or tags
- **Templates**: Mark reusable rubrics as templates with `is_template=True`
- **Custom scales**: Define custom scoring scales for different evaluation systems
- **Backup**: Regularly backup your `rubrics/` directory

## Troubleshooting

**Rubric not found?**
- Check the rubric ID: `video-analyser rubric list`
- Ensure rubric is in the `rubrics/` directory

**Can't create rubric from type?**
- Check available types: `video-analyser rubric create --type invalid` will list options
- Try: `video-analyser rubric create --type general`

**Want to modify an existing rubric?**
- Export it: `video-analyser rubric export --id <id> --output rubric.json`
- Edit the JSON file
- Delete the old: `video-analyser rubric delete --id <id>`
- Re-import by placing the JSON in `rubrics/` directory

## Related Documentation

- [Assessment Guide](./assessment-guide.md) - How to use rubrics in assessments
- [CLI Reference](./cli-reference.md) - Complete CLI command reference
- [Grading System](./grading-system.md) - How grades are calculated
