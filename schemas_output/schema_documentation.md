# Video Analyser Video Analysis JSON Schemas

This document describes the JSON schemas used for video analysis results in Video Analyser.

## Overview

The Video Analyser video analysis system produces structured results in JSON format.
These schemas ensure consistent data structure for:

- API responses
- Report generation
- Data export/import
- Third-party integrations

## Video Analysis Result

**Schema ID:** `https://video-analyser.ai/schemas/video_analysis_result.json`

**Description:** JSON Schema for video_analysis_result in Video Analyser video analysis

### Properties

- **analysis_id** (`string`) (required): Unique identifier for this analysis
- **analysis_version** (`string`) (required): Version of the analysis schema
- **video_metadata** (`object`) (required): Source video metadata
- **processing_metadata** (`object`) (required): Processing pipeline metadata
- **scene_detection** (`object`) (required): Scene detection results
- **transcription** (`object`) (optional): Speech transcription results
- **overall_speech_metrics** (`object`) (optional): Overall speech analysis
- **visual_analysis** (`object`) (optional): Visual analysis results
- **scene_analyses** (`array`) (optional): Detailed analysis results for each scene
- **quality_assessment** (`object`) (required): Overall quality assessment
- **content_summary** (`object`) (required): High-level content summary
- **analysis_insights** (`object`) (required): Insights and recommendations
- **export_timestamp** (`string`) (optional): When this result was exported
- **export_format_version** (`string`) (optional): Version of the export format

### Example Usage

```python
from video_analyser.reports.analysis_schema import VideoAnalysisResult

# Validate data against schema
result = VideoAnalysisResult.model_validate(data)

# Convert to JSON
json_data = result.model_dump()
```

## Analysis Report

**Schema ID:** `https://video-analyser.ai/schemas/analysis_report.json`

**Description:** JSON Schema for analysis_report in Video Analyser video analysis

### Properties

- **analysis_result** (`object`) (required): Complete analysis results
- **report_id** (`string`) (required): Unique identifier for this report
- **report_title** (`string`) (required): Human-readable report title
- **report_type** (`string`) (optional): Type of report generated
- **executive_summary** (`string`) (required): Executive summary of findings
- **detailed_findings** (`string`) (required): Detailed analysis findings
- **recommendations** (`string`) (required): Formatted recommendations
- **sections** (`object`) (optional): Structured report sections with content
- **charts_data** (`object`) (optional): Data for charts and visualizations
- **available_formats** (`array`) (optional): Available export formats for this report
- **generated_at** (`string`) (optional): When this report was generated
- **template_version** (`string`) (optional): Version of the report template used

### Example Usage

```python
from video_analyser.reports.analysis_schema import AnalysisReport

# Validate data against schema
result = AnalysisReport.model_validate(data)

# Convert to JSON
json_data = result.model_dump()
```
