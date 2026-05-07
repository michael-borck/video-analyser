"""HTML grading sheet renderer for assessment reports."""

import logging
from datetime import datetime
from pathlib import Path

from .assessment_integration import AssessmentIntegration
from .assessment_report import AssessmentReport
from .report_generator import AnalysisReport

logger = logging.getLogger(__name__)


class GradingSheetRenderer:
    """Render assessment reports as HTML grading sheets."""

    def __init__(self, analysis_report: AnalysisReport | None = None):
        """
        Initialize renderer.

        Args:
            analysis_report: Optional AnalysisReport for context
        """
        self.analysis_report = analysis_report

    def render(self, assessment: AssessmentReport) -> str:
        """
        Render assessment as HTML grading sheet.

        Args:
            assessment: AssessmentReport to render

        Returns:
            HTML string
        """
        html_parts = [
            self._render_header(assessment),
            self._render_video_info(assessment),
            self._render_diarization_section(assessment),
            self._render_rubric_section(assessment),
            self._render_footer(assessment),
        ]

        return "\n".join(html_parts)

    def render_to_file(
        self, assessment: AssessmentReport, output_path: str | Path
    ) -> Path:
        """
        Render assessment to HTML file.

        Args:
            assessment: AssessmentReport to render
            output_path: Path to write HTML file

        Returns:
            Path to created file
        """
        html = self.render(assessment)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(html)

        logger.info(f"Grading sheet rendered to {output_path}")
        return output_path

    def _render_header(self, assessment: AssessmentReport) -> str:
        """Render HTML header and assessment title."""
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Grading Sheet - {assessment.id}</title>
            <style>
                {self._get_css()}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📋 Grading Sheet</h1>
                    <div class="header-info">
                        <p><strong>Assessment ID:</strong> {assessment.id}</p>
                        <p><strong>Assessor:</strong> {assessment.assessed_by}</p>
                        <p><strong>Assessed:</strong> {assessment.assessed_at.strftime("%Y-%m-%d %H:%M:%S")}</p>
                        <p><strong>Status:</strong> <span class="status-{"complete" if assessment.is_complete else "draft"}">
                            {"COMPLETE" if assessment.is_complete else "DRAFT"}</span></p>
                    </div>
                </div>
        """

    def _render_video_info(self, assessment: AssessmentReport) -> str:
        """Render video information section."""
        context = (
            AssessmentIntegration.get_assessment_context(
                assessment, self.analysis_report
            )
            if self.analysis_report
            else {}
        )

        lines = [
            '<div class="section"><h2>📹 Video Information</h2><div class="info-grid">'
        ]

        lines.append(
            f'<div class="info-item"><strong>File:</strong> {assessment.video_file_path}</div>'
        )

        if context:
            lines.append(
                f'<div class="info-item"><strong>Duration:</strong> {context.get("video_duration", "N/A")}s</div>'
            )
            lines.append(
                f'<div class="info-item"><strong>Resolution:</strong> {context.get("video_resolution", "N/A")}</div>'
            )
            lines.append(
                f'<div class="info-item"><strong>FPS:</strong> {context.get("fps", "N/A")}</div>'
            )
            lines.append(
                f'<div class="info-item"><strong>Language:</strong> {context.get("language", "Unknown")}</div>'
            )

        lines.append("</div></div>")
        return "\n".join(lines)

    def _render_diarization_section(self, assessment: AssessmentReport) -> str:
        """Render speaker diarization section."""
        lines = ['<div class="section"><h2>🎤 Speaker Diarization</h2>']

        if not assessment.speaker_profiles and not assessment.diarization_segments:
            lines.append("<p>No speakers detected.</p>")
        else:
            # Speaker table
            if assessment.speaker_profiles:
                lines.append('<table class="speaker-table">')
                lines.append(
                    "<thead><tr><th>Speaker</th><th>Segments</th>"
                    "<th>Duration</th><th>% of Total</th><th>First</th><th>Last</th></tr></thead>"
                )
                lines.append("<tbody>")

                for profile in assessment.speaker_profiles:
                    label = profile.speaker_label or profile.speaker_id
                    first_app = profile.first_appearance or 0.0
                    last_app = profile.last_appearance or 0.0
                    lines.append(
                        f"<tr>"
                        f"<td><strong>{label}</strong></td>"
                        f"<td>{profile.num_segments}</td>"
                        f"<td>{profile.total_speaking_time:.1f}s</td>"
                        f"<td>{profile.percentage_of_total:.1f}%</td>"
                        f"<td>{first_app:.1f}s</td>"
                        f"<td>{last_app:.1f}s</td>"
                        f"</tr>"
                    )

                lines.append("</tbody></table>")

        # Quality flags (shown regardless of speakers)
        if assessment.quality_flags:
            lines.append('<div class="quality-flags">')
            lines.append("<h3>⚠️ Quality Flags</h3>")
            for flag in assessment.quality_flags:
                lines.append(f'<span class="flag">{flag}</span>')
            lines.append("</div>")

        lines.append("</div>")
        return "\n".join(lines)

    def _render_rubric_section(self, assessment: AssessmentReport) -> str:
        """Render rubric scoring section."""
        lines = ['<div class="section"><h2>📊 Rubric Scoring</h2>']

        if not assessment.rubric_application:
            lines.append("<p>No rubric applied yet.</p>")
            lines.append("</div>")
            return "\n".join(lines)

        rubric = assessment.rubric_application

        # Overall score
        lines.append(
            f'<div class="overall-score">'
            f'<h3>Overall Score: <span class="score-value">{rubric.overall_percentage:.1f}%</span></h3>'
            f'<div class="score-bar">'
            f'<div class="score-fill" style="width: {rubric.overall_percentage}%"></div>'
            f"</div>"
            f"</div>"
        )

        # Category breakdown
        lines.append('<div class="categories">')

        for category in rubric.category_assessments:
            lines.append('<div class="category">')
            lines.append(
                f"<h4>{category.category_name} "
                f'<span class="cat-score">{category.category_percentage:.1f}%</span></h4>'
            )

            # Criteria within category
            lines.append('<div class="criteria">')
            for feedback in category.criterion_feedbacks:
                lines.append('<div class="criterion">')
                lines.append(
                    f"<p><strong>{feedback.criterion_name}:</strong> {feedback.score}</p>"
                )

                if feedback.feedback:
                    lines.append(f'<p class="feedback-text">{feedback.feedback}</p>')

                if feedback.evidence_timestamps:
                    lines.append('<p class="evidence"><strong>Evidence:</strong>')
                    timestamps = [
                        f"{start:.1f}s-{end:.1f}s"
                        for start, end in feedback.evidence_timestamps
                    ]
                    lines.append(", ".join(timestamps))
                    lines.append("</p>")

                lines.append("</div>")

            lines.append("</div>")
            lines.append("</div>")

        lines.append("</div>")

        # General feedback
        if rubric.general_feedback:
            lines.append('<div class="general-feedback">')
            lines.append("<h3>Overall Feedback</h3>")
            lines.append(f"<p>{rubric.general_feedback}</p>")
            lines.append("</div>")

        lines.append("</div>")
        return "\n".join(lines)

    def _render_footer(self, assessment: AssessmentReport) -> str:
        """Render HTML footer."""
        return f"""
            {self._render_assessment_notes(assessment)}
            </div>
            <footer>
                <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Assessment ID: {assessment.id}</p>
            </footer>
        </body>
        </html>
        """

    def _render_assessment_notes(self, assessment: AssessmentReport) -> str:
        """Render assessment notes section."""
        if not assessment.assessment_notes:
            return ""

        return f"""
        <div class="section">
            <h2>📝 Assessment Notes</h2>
            <p>{assessment.assessment_notes}</p>
        </div>
        """

    @staticmethod
    def _get_css() -> str:
        """Get CSS styles for grading sheet."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        .header {
            border-bottom: 3px solid #007bff;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }

        .header h1 {
            color: #007bff;
            margin-bottom: 15px;
            font-size: 2em;
        }

        .header-info {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            font-size: 0.95em;
        }

        .status-complete {
            background-color: #28a745;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: bold;
        }

        .status-draft {
            background-color: #ffc107;
            color: #333;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: bold;
        }

        .section {
            margin-bottom: 30px;
            padding: 20px;
            background-color: #f9f9f9;
            border-left: 4px solid #007bff;
            border-radius: 4px;
        }

        .section h2 {
            color: #007bff;
            margin-bottom: 15px;
            font-size: 1.5em;
        }

        .section h3 {
            color: #333;
            margin-top: 15px;
            margin-bottom: 10px;
            font-size: 1.2em;
        }

        .section h4 {
            color: #555;
            margin-bottom: 10px;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }

        .info-item {
            padding: 10px;
            background-color: white;
            border-radius: 4px;
            border-left: 3px solid #007bff;
        }

        .speaker-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }

        .speaker-table th {
            background-color: #007bff;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }

        .speaker-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }

        .speaker-table tbody tr:hover {
            background-color: #f0f8ff;
        }

        .quality-flags {
            margin-top: 15px;
            padding: 15px;
            background-color: #fff3cd;
            border-radius: 4px;
            border: 1px solid #ffc107;
        }

        .flag {
            display: inline-block;
            background-color: #ffc107;
            color: #333;
            padding: 5px 10px;
            margin: 5px 5px 5px 0;
            border-radius: 4px;
            font-size: 0.9em;
            font-weight: bold;
        }

        .overall-score {
            background-color: white;
            padding: 20px;
            border-radius: 6px;
            margin-bottom: 20px;
            border: 2px solid #007bff;
        }

        .score-value {
            font-size: 1.8em;
            color: #007bff;
            font-weight: bold;
        }

        .score-bar {
            height: 40px;
            background-color: #e9ecef;
            border-radius: 6px;
            overflow: hidden;
            margin-top: 10px;
        }

        .score-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745 0%, #007bff 50%, #ffc107 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 0.3s ease;
        }

        .categories {
            margin-top: 20px;
        }

        .category {
            background-color: white;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 6px;
            border-left: 4px solid #17a2b8;
        }

        .cat-score {
            background-color: #e7f3ff;
            color: #007bff;
            padding: 4px 10px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 0.9em;
        }

        .criteria {
            margin-top: 10px;
        }

        .criterion {
            padding: 10px;
            margin-bottom: 10px;
            background-color: #f9f9f9;
            border-radius: 4px;
            border-left: 3px solid #17a2b8;
        }

        .criterion p {
            margin: 5px 0;
        }

        .feedback-text {
            font-style: italic;
            color: #555;
            margin-left: 15px;
            padding: 10px;
            background-color: #fff;
            border-left: 3px solid #ffc107;
        }

        .evidence {
            color: #666;
            font-size: 0.9em;
            margin-left: 15px;
            padding: 8px;
            background-color: #f0f0f0;
            border-radius: 4px;
        }

        .general-feedback {
            background-color: #e7f3ff;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #007bff;
            margin-top: 20px;
        }

        .general-feedback p {
            color: #333;
            line-height: 1.8;
        }

        footer {
            border-top: 1px solid #ddd;
            padding-top: 15px;
            margin-top: 30px;
            text-align: center;
            font-size: 0.9em;
            color: #666;
        }

        @media print {
            body {
                background-color: white;
                padding: 0;
            }
            .container {
                box-shadow: none;
                padding: 0;
            }
        }
        """
