"""Gradio web interface for Video Analyser video analysis."""

import logging
from pathlib import Path
from typing import Any

import gradio as gr  # type: ignore[import-untyped]

from video_analyser.core.pipeline_coordinator import PipelineCoordinator
from video_analyser.utils.config import get_config
from video_analyser.utils.progress_display import CLIProgressTracker

logger = logging.getLogger(__name__)


class GradioInterface:
    """Main Gradio web interface for video analysis."""

    def __init__(self):
        """Initialize the Gradio interface."""
        self.config = get_config()
        self.pipeline = PipelineCoordinator(config=self.config)
        self.progress_tracker = CLIProgressTracker()

    def analyze_video_file(
        self,
        video_file: Any,
        api_provider: str = "anthropic",
        use_api: bool = False,
        progress: Any = gr.Progress(),
    ) -> tuple[str, str, str]:
        """
        Analyze uploaded video file.

        Args:
            video_file: Uploaded video file
            api_provider: API provider for image captioning
            use_api: Whether to use API for captioning
            progress: Gradio progress tracker

        Returns:
            Tuple of (status_message, html_report_path, json_report_path)
        """
        if video_file is None:
            return "❌ Please upload a video file", "", ""

        try:
            # Convert to Path
            video_path = Path(video_file.name)

            # Update config for API settings
            if use_api:
                self.config.visual_analysis.captioning_backend = "api"
                self.config.visual_analysis.api_provider = api_provider

            # Create output directory
            output_dir = Path("outputs") / video_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)

            # Run analysis
            progress(0.1, desc="Starting video analysis...")

            result = self.pipeline.analyze_video(
                video_path=video_path,
                extract_audio=True,
                detect_scenes=True,
                extract_frames=True,
                output_dir=output_dir,
            )

            if not result.success:
                return f"❌ Analysis failed: {result.error_message}", "", ""

            progress(0.5, desc="Analyzing speech...")

            # Speech analysis
            speech_analysis = None
            if result.audio_info:
                speech_analysis = self.pipeline.analyze_speech(
                    audio_path=result.audio_info.file_path,
                    scene_result=result.scene_result,
                )

            progress(0.8, desc="Analyzing frames...")

            # Visual analysis
            visual_analysis: dict[str, Any] | None = None
            if result.frame_infos:
                frame_paths = [frame.frame_path for frame in result.frame_infos]
                visual_analysis = self.pipeline.analyze_frames(
                    frame_paths=frame_paths,
                )

            progress(0.9, desc="Generating reports...")

            # Generate reports
            report_paths = self.pipeline.generate_reports(
                video_info=result.video_info,
                audio_info=result.audio_info,
                scene_result=result.scene_result,
                speech_analysis=speech_analysis,
                visual_analysis=visual_analysis,
                output_dir=output_dir,
            )

            progress(1.0, desc="Analysis complete!")

            # Prepare result message
            status_msg = "✅ Analysis completed successfully!"
            if report_paths.get("html"):
                status_msg += f"\n📄 HTML Report: {report_paths['html']}"
            if report_paths.get("json"):
                status_msg += f"\n📊 JSON Report: {report_paths['json']}"

            html_path = str(report_paths.get("html", ""))
            json_path = str(report_paths.get("json", ""))

            return status_msg, html_path, json_path

        except Exception as e:
            logger.error(f"Video analysis failed: {e}", exc_info=True)
            return f"❌ Analysis failed: {str(e)}", "", ""

    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(
            title="Video Analyser Video Analysis",
            theme=gr.themes.Soft(),  # type: ignore[attr-defined]
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .main-header {
                text-align: center;
                margin-bottom: 2rem;
            }
            .upload-area {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 2rem;
                text-align: center;
                margin: 1rem 0;
            }
            """,
        ) as interface:
            # Header
            gr.HTML("""
            <div class="main-header">
                <h1>🎥 Video Analyser</h1>
                <p>Video Analysis for Presentation Feedback</p>
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    # File upload
                    video_input = gr.File(
                        label="Upload Video",
                        file_types=[".mp4", ".mov", ".avi", ".webm"],
                        type="filepath",
                    )

                    # Settings
                    with gr.Accordion("Analysis Settings", open=False):
                        use_api_checkbox = gr.Checkbox(
                            label="Use AI API for image captioning",
                            value=False,
                            info="Faster but requires API key",
                        )
                        api_provider_dropdown = gr.Dropdown(
                            choices=["anthropic", "openai", "google"],
                            value="anthropic",
                            label="API Provider",
                            info="Choose your preferred AI provider",
                        )

                    # Analyze button
                    analyze_btn = gr.Button(
                        "🚀 Analyze Video",
                        variant="primary",
                        size="lg",
                    )

                with gr.Column(scale=1):
                    # Info panel
                    gr.Markdown("""
                    ### 📋 Supported Features
                    - **Speech Analysis**: Transcription with speaking rate and filler word detection
                    - **Visual Analysis**: Frame captioning and quality assessment
                    - **Scene Detection**: Automatic scene segmentation
                    - **Professional Reports**: HTML and JSON output formats

                    ### 📝 Tips
                    - MP4, MOV, AVI, WebM formats supported
                    - Max file size: 500MB
                    - Processing time varies by video length
                    - API captioning is faster but requires keys
                    """)

            # Results section
            with gr.Row():
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3,
                )

            with gr.Row():
                html_report = gr.File(
                    label="HTML Report",
                    visible=True,
                )
                json_report = gr.File(
                    label="JSON Report",
                    visible=True,
                )

            # Event handlers
            analyze_btn.click(
                fn=self.analyze_video_file,
                inputs=[video_input, api_provider_dropdown, use_api_checkbox],
                outputs=[status_output, html_report, json_report],
            )

            # Footer
            from .. import __version__ as _v
            gr.HTML(f"""
            <div style="text-align: center; margin-top: 2rem; color: #666;">
                <p>Video Analyser v{_v} - Video Analysis Tool for the Analyser Family</p>
                <p><a href="https://github.com/michaelborck-education/video-analyser" target="_blank">GitHub</a></p>
            </div>
            """)

        return interface

    def launch(self, **kwargs: Any) -> None:
        """Launch the Gradio interface."""
        interface = self.create_interface()

        # Default launch settings with explicit types
        server_name: str = kwargs.get("server_name", "0.0.0.0")
        server_port: int = kwargs.get("server_port", 7860)
        share: bool = kwargs.get("share", False)
        show_error: bool = kwargs.get("show_error", True)
        quiet: bool = kwargs.get("quiet", False)

        logger.info(f"Launching Gradio interface on port {server_port}")
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            show_error=show_error,
            quiet=quiet,
        )


def create_app() -> gr.Blocks:
    """Create and return the Gradio application."""
    app = GradioInterface()
    return app.create_interface()


def launch_app(**kwargs: Any) -> None:
    """Launch the Gradio application."""
    app = GradioInterface()
    app.launch(**kwargs)


if __name__ == "__main__":
    launch_app()
