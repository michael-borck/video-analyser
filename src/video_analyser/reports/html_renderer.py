"""HTML report renderer for Video Analyser analysis results."""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HTMLRenderer:
    """Render analysis reports as HTML."""

    def __init__(self):
        """Initialize HTML renderer."""
        logger.info("HTMLRenderer initialized")

    def render_report(self, report: dict[str, Any]) -> str:
        """Render report as HTML."""
        logger.info("Rendering HTML report")

        video = report.get("video", {})
        audio = report.get("audio", {})
        scenes = report.get("scenes", [])
        frames = report.get("frames", [])
        segments = report.get("transcription_segments", [])
        speech_metrics = report.get("speech_metrics", {})

        # Build frames HTML
        frames_html = ""
        for frame in frames:
            caption = frame.get("caption", "")
            ocr = frame.get("ocr_text", "")
            objects = frame.get("detected_objects", [])

            frame_path = frame.get("file_path", "")
            img_tag = (
                f'<img src="{frame_path}" style="max-width: 600px; border: 1px solid #ddd; margin: 10px 0;">'
                if frame_path
                else ""
            )

            frames_html += f"""
            <div class="frame" style="margin: 20px 0; padding: 15px; background: #f9f9f9; border-left: 4px solid #007bff;">
                <h4>Frame {frame.get("frame_number", 0)} @ {frame.get("timestamp", 0):.2f}s (Scene {frame.get("scene_number", 0)})</h4>
                {img_tag}
                <div><strong>Resolution:</strong> {frame.get("width", 0)}x{frame.get("height", 0)}</div>
                {"<div><strong>Caption:</strong> " + caption + f" <em>({frame.get('caption_model', 'unknown')})</em></div>" if caption else ""}
                {"<div><strong>Text (OCR):</strong> " + ocr + "</div>" if ocr else ""}
                {"<div><strong>Objects detected:</strong> " + ", ".join(objects) + "</div>" if objects else ""}
                {"<div><strong>Quality score:</strong> " + f"{frame.get('quality_score', 0):.2f}" + "</div>" if frame.get("quality_score") else ""}
            </div>
            """

        # Build transcription HTML
        transcription_html = ""
        if segments:
            transcription_html = "<h2>Transcription</h2>"
            transcription_html += f'<div style="margin-bottom: 20px;"><strong>Language:</strong> {report.get("language", "Unknown")}</div>'
            if speech_metrics:
                transcription_html += f"""
                <div style="background: #f0f0f0; padding: 15px; margin-bottom: 20px;">
                    <h3>Speech Metrics</h3>
                    <div><strong>Total Words:</strong> {speech_metrics.get("total_words", 0)}</div>
                    <div><strong>Speech Duration:</strong> {speech_metrics.get("total_speech_duration", 0):.1f}s</div>
                    <div><strong>Speaking Rate:</strong> {speech_metrics.get("speaking_rate_wpm", 0):.1f} WPM</div>
                    <div><strong>Average Confidence:</strong> {speech_metrics.get("average_confidence", 0):.2f}</div>
                </div>
                """
            for seg in segments:
                transcription_html += f"""
                <div style="margin: 10px 0; padding: 10px; background: #fafafa; border-left: 3px solid #28a745;">
                    <div><strong>[{seg.get("start_time", 0):.2f}s - {seg.get("end_time", 0):.2f}s]</strong></div>
                    <div style="margin-top: 5px;">{seg.get("text", "")}</div>
                </div>
                """

        # Audio information section
        audio_html = ""
        if audio:
            duration_val = audio.get("duration", 0)
            sample_rate_val = audio.get("sample_rate", 0)
            channels_val = audio.get("channels", 0)

            audio_html = f"""
        <h2>Audio Information</h2>
        <div style="background: #f0f0f0; padding: 15px; margin-bottom: 20px;">
            <div><strong>Duration:</strong> {duration_val:.1f}s</div>
            <div><strong>Sample Rate:</strong> {sample_rate_val} Hz</div>
            <div><strong>Channels:</strong> {channels_val}</div>
        </div>
            """

        # Scenes information section
        scenes_html = ""
        if scenes:
            scenes_html = "<h2>Scene Breakdown</h2>"
            for i, scene in enumerate(scenes[:5]):  # Show first 5 scenes
                scene_num = scene.get("scene_number", i + 1)
                start_val = scene.get("start_time", 0)
                end_val = scene.get("end_time", 0)
                dur_val = scene.get("duration", 0)

                scenes_html += f"""
                <div style="margin: 10px 0; padding: 10px; background: #fafafa; border-left: 3px solid #007bff;">
                    <div><strong>Scene {scene_num}</strong></div>
                    <div><strong>Time:</strong> {start_val:.2f}s - {end_val:.2f}s</div>
                    <div><strong>Duration:</strong> {dur_val:.2f}s</div>
                </div>
                """
            if len(scenes) > 5:
                scenes_html += f"<p><em>... and {len(scenes) - 5} more scenes</em></p>"

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Video Analysis Report</title>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e9ecef;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }}
        .check {{ color: #28a745; }}
        .cross {{ color: #dc3545; }}
        h1 {{ color: #007bff; }}
        h2 {{ color: #495057; border-bottom: 2px solid #e9ecef; padding-bottom: 10px; }}
        h3 {{ color: #6c757d; }}
        .frame {{ border-radius: 5px; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎥 Video Analysis Report</h1>
            <p style="color: #6c757d; font-size: 1.1em;">
                {video.get("file_path", "Unknown")}
            </p>
        </div>

        <div class="metrics">
            <div class="metric">
                <strong>Duration</strong>
                {video.get("duration", 0):.1f} seconds
            </div>
            <div class="metric">
                <strong>Resolution</strong>
                {video.get("width", 0)}x{video.get("height", 0)} @ {video.get("fps", 0):.1f} fps
            </div>
            <div class="metric">
                <strong>Scenes Detected</strong>
                {report.get("total_scenes", 0)}
            </div>
            <div class="metric">
                <strong>Frames Analyzed</strong>
                {report.get("total_frames", 0)}
            </div>
            <div class="metric">
                <strong>Analysis Features</strong>
                <div>Transcription: <span class="{"check" if report.get("has_transcription") else "cross"}">{"✓" if report.get("has_transcription") else "✗"}</span></div>
                <div>Captions: <span class="{"check" if report.get("has_captions") else "cross"}">{"✓" if report.get("has_captions") else "✗"}</span></div>
                <div>OCR: <span class="{"check" if report.get("has_ocr") else "cross"}">{"✓" if report.get("has_ocr") else "✗"}</span></div>
            </div>
        </div>

        {transcription_html}
        {audio_html}
        {scenes_html}

        <h2>Frame Analysis</h2>
        <div>{frames_html if frames_html else "<p>No frames analyzed</p>"}</div>

    </div>
</body>
</html>"""

        return html

    def save_html(self, html_content: str, output_path: Path) -> None:
        """Save HTML content to file."""
        logger.info(f"Saving HTML report to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"HTML report saved: {output_path}")
