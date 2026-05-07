"""Command-line interface for Video Analyser."""

import hashlib
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.panel import Panel

from video_analyser.core.exceptions import VideoProcessingError
from video_analyser.core.pipeline_coordinator import PipelineCoordinator
from video_analyser.utils.config import VideoAnalyserConfig, get_config, load_config
from video_analyser.utils.progress_display import CLIProgressTracker

if TYPE_CHECKING:
    from video_analyser.analysis.rubric_system import RubricRepository

console = Console()
app = typer.Typer(help="Video Analyser - Video Analysis Application")


@app.command()
def analyze(
    video_path: Path | None = typer.Argument(
        None, help="Path to video file to analyze"
    ),
    output_dir: Path | None = typer.Option(
        None, "--output", "-o", help="Output directory for reports"
    ),
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    api_provider: str | None = typer.Option(
        None,
        "--api-provider",
        help="API provider (anthropic, openai, google, openrouter) for captioning and grading",
    ),
    api_model: str | None = typer.Option(
        None,
        "--api-model",
        help="API model to use (e.g., claude-haiku-4-5-20251001, gpt-4o)",
    ),
    use_api: bool = typer.Option(
        False,
        "--use-api",
        help="Use API for captioning instead of local model",
    ),
    grade: bool = typer.Option(
        False,
        "--grade",
        "-g",
        help="Enable grading with feedback (uses defaults: general rubric, student audience, summary detail)",
    ),
    rubric_type: str | None = typer.Option(
        None,
        "--rubric-type",
        "-t",
        help="Rubric type for grading (academic, business, teaching, technical, creative, sales, legal, medical, political, entertainment, general). Default: general",
    ),
    rubric_file: Path | None = typer.Option(
        None, "--rubric-file", "-r", help="Path to custom rubric JSON file for grading"
    ),
    feedback_audience: str | None = typer.Option(
        None,
        "--feedback-audience",
        help="Target audience for feedback: student (developmental, encouraging) or teacher (assessment-focused). Default: student",
    ),
    feedback_detail: str | None = typer.Option(
        None,
        "--feedback-detail",
        help="Level of detail in feedback: short (assessment only), summary (strengths + improvements), long (full detailed feedback). Default: summary",
    ),
    save_all_formats: bool = typer.Option(
        False,
        "--save-all-formats",
        help="Save feedback in all formats (short, summary, long) as separate files",
    ),
    fast_mode: bool = typer.Option(
        False,
        "--fast",
        help="Fast mode: analyze transcript only (skip visual analysis for ~2-3x speed improvement)",
    ),
    parallel_processing: bool = typer.Option(
        True,
        "--parallel/--no-parallel",
        help="Enable parallel processing for faster analysis (default: enabled)",
    ),
    use_cache: bool = typer.Option(
        True,
        "--cache/--no-cache",
        help="Use caching to resume after crashes and avoid reprocessing (default: enabled)",
    ),
) -> None:
    """
    Analyze a video file for presentation feedback.

    If --grade flag or rubric options (--rubric-type or --rubric-file) are provided,
    generates LLM-based grading feedback in addition to the analysis report.

    If no video path is provided, launches the web interface.

    Examples:
        video-analyser analyze video.mp4
        video-analyser analyze video.mp4 --grade
        video-analyser analyze video.mp4 --grade --rubric-type academic --feedback-detail long
        video-analyser analyze video.mp4 --rubric-type academic --feedback-audience teacher
        video-analyser analyze video.mp4 --rubric-file my-rubric.json --api-provider anthropic
    """
    # Load configuration
    config = load_config(config_file) if config_file else get_config()

    # Apply CLI overrides
    if use_api:
        config.visual_analysis.captioning_backend = "api"
    if api_provider:
        config.visual_analysis.api_provider = api_provider
    if api_model:
        config.visual_analysis.api_model = api_model

    # Set up logging
    logger = logging.getLogger("video_analyser")
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    logger.info("Starting Video Analyser analysis")

    console.print(
        Panel.fit(
            f"[bold blue]{config.app_name}[/bold blue]\n[dim]Video Analysis Application[/dim]",
            border_style="blue",
        )
    )

    if config.debug:
        console.print("[yellow]Debug mode enabled[/yellow]")
        logger.debug("Debug mode is active")

    if video_path:
        # CLI mode - analyze specific video
        # Show API settings if using API
        if config.visual_analysis.captioning_backend == "api":
            console.print(
                f"[dim]Using API: {config.visual_analysis.api_provider} "
                f"({config.visual_analysis.api_model})[/dim]"
            )

        _analyze_video_cli(
            video_path=video_path,
            output_dir=output_dir,
            config=config,
            config_file=config_file,
            verbose=verbose,
            logger=logger,
            grade=grade,
            rubric_type=rubric_type,
            rubric_file=rubric_file,
            feedback_audience=feedback_audience,
            feedback_detail=feedback_detail,
            save_all_formats=save_all_formats,
            fast_mode=fast_mode,
            parallel_processing=parallel_processing,
            use_cache=use_cache,
            api_provider=api_provider,
            api_model=api_model,
        )
    else:
        # Web UI mode
        logger.info("Launching web interface")
        console.print("[green]Launching web interface...[/green]")

        # TODO: Import and launch Gradio interface
        logger.warning("Web interface not yet implemented")
        console.print("[yellow]Web interface not yet implemented.[/yellow]")
        console.print("[dim]Run with --help for available options.[/dim]")


def _analyze_video_cli(
    video_path: Path,
    output_dir: Path | None,
    config: VideoAnalyserConfig,
    config_file: Path | None = None,
    verbose: bool = False,
    logger: logging.Logger = None,
    grade: bool = False,
    rubric_type: str | None = None,
    rubric_file: Path | None = None,
    feedback_audience: str | None = None,
    feedback_detail: str | None = None,
    save_all_formats: bool = False,
    fast_mode: bool = False,
    parallel_processing: bool = True,
    use_cache: bool = True,
    api_provider: str | None = None,
    api_model: str | None = None,
) -> None:
    """
    Perform video analysis in CLI mode.

    Args:
        video_path: Path to video file
        output_dir: Optional output directory
        config: Configuration object
        config_file: Configuration file path (for logging)
        verbose: Verbose output flag
        logger: Logger instance
        grade: Enable grading with defaults (general rubric, student audience, summary detail)
        rubric_type: Optional rubric type for grading (academic, business, teaching, technical, creative, sales, legal, medical, political, entertainment, general)
        rubric_file: Optional custom rubric file for grading
        feedback_audience: Target audience for feedback (student or teacher)
        feedback_detail: Level of detail (short, summary, long)
        api_provider: Optional API provider for grading
        api_model: Optional API model for grading
    """
    # Validate video path
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        console.print(f"[red]✗ Error: Video file not found[/red] {video_path}")
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    if not video_path_obj.is_file():
        console.print(f"[red]✗ Error: Path is not a file[/red] {video_path}")
        logger.error(f"Path is not a file: {video_path}")
        sys.exit(1)

    # Set up output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        console.print(f"[blue]Output directory:[/blue] {output_path}")
        logger.debug(f"Output directory: {output_path}")
    else:
        # Default output to current directory with video name
        output_path = Path.cwd() / video_path_obj.stem
        output_path.mkdir(parents=True, exist_ok=True)
        console.print(f"[blue]Output directory:[/blue] {output_path} (default)")
        logger.debug(f"Output directory (default): {output_path}")

    if config_file:
        console.print(f"[blue]Config file:[/blue] {config_file}")
        logger.debug(f"Using config file: {config_file}")

    # Show relevant config info in debug mode
    if verbose:
        console.print(
            f"[dim]Max file size: {config.processing.max_video_size_mb}MB[/dim]"  # type: ignore
        )
        console.print(f"[dim]Transcription model: {config.transcription.model}[/dim]")  # type: ignore

    logger.info(f"Starting analysis: {video_path}")
    console.print(f"[green]Analyzing video:[/green] {video_path}\n")

    # Initialize progress tracker and pipeline
    progress_tracker: CLIProgressTracker = CLIProgressTracker()
    pipeline = PipelineCoordinator(
        config=config, progress_tracker=progress_tracker
    )  # Use CLI callbacks instead

    # Define workflow operations
    operations = [
        ("validate", "Validating video", 0.05),
        ("audio", "Extracting audio", 0.10),
        ("scenes", "Detecting scenes", 0.10),
        ("frames", "Extracting frames", 0.10),
        ("transcribe", "Transcribing speech", 0.30),
        ("visual", "Analyzing frames", 0.20),
        ("reports", "Generating reports", 0.15),
    ]

    progress_tracker.start_workflow(f"Analyzing {video_path_obj.name}", operations)

    try:
        # Track time
        start_time = time.time()

        # Phase 1: Video Processing
        progress_tracker.start_operation("validate")
        result = pipeline.analyze_video(
            video_path=video_path_obj,
            extract_audio=True,
            detect_scenes=True,
            extract_frames=True,
            output_dir=output_path,
        )
        progress_tracker.complete_operation("validate")
        progress_tracker.complete_operation("audio")
        progress_tracker.complete_operation("scenes")
        progress_tracker.complete_operation("frames")

        # Check if video processing was successful
        if not result.success:
            progress_tracker.fail_workflow(result.error_message or "Unknown error")
            if result.errors:
                for error in result.errors:
                    logger.error(f"Error: {error}")
            sys.exit(1)

        # Phase 2: Speech Analysis
        speech_analysis = None
        if result.audio_info:
            progress_tracker.start_operation("transcribe")
            try:
                speech_analysis = pipeline.analyze_speech(
                    audio_path=result.audio_info.file_path,
                    scene_result=result.scene_result,
                )
                progress_tracker.complete_operation("transcribe")
                logger.info("Speech analysis completed")
            except Exception as e:
                logger.warning(f"Speech analysis failed: {e}")
                progress_tracker.complete_operation("transcribe")
        else:
            progress_tracker.complete_operation("transcribe")

        # Phase 3: Visual Analysis (skip in fast mode)
        visual_analysis: dict[str, Any] | None = None
        if not fast_mode and result.frame_infos:
            progress_tracker.start_operation("visual")
            try:
                frame_paths = [frame.frame_path for frame in result.frame_infos]
                visual_analysis = pipeline.analyze_frames(
                    frame_paths=frame_paths,
                )
                progress_tracker.complete_operation("visual")
                logger.info("Visual analysis completed")
            except Exception as e:
                logger.warning(f"Visual analysis failed: {e}")
                progress_tracker.complete_operation("visual")
        else:
            if fast_mode:
                console.print("[dim]Fast mode: skipping visual analysis[/dim]")
                logger.info("Fast mode: skipping visual analysis")
            progress_tracker.complete_operation("visual")

        # Phase 4: Generate Reports
        progress_tracker.start_operation("reports")
        try:
            report_paths = pipeline.generate_reports(
                video_info=result.video_info,
                audio_info=result.audio_info,
                scene_result=result.scene_result,
                speech_analysis=speech_analysis,
                visual_analysis=visual_analysis,
                output_dir=output_path,
            )
            progress_tracker.complete_operation("reports")
            logger.info("Reports generated successfully")
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            progress_tracker.complete_operation("reports")
            report_paths = {}

        processing_time = time.time() - start_time

        # Display results
        progress_tracker.complete_workflow()

        # Show summary
        console.print("\n[bold]Analysis Summary:[/bold]")
        console.print(f"  Video: {video_path_obj.name}")
        console.print(f"  Duration: {result.video_info.duration:.1f}s")
        console.print(
            f"  Resolution: {result.video_info.width}x{result.video_info.height}"
        )
        console.print(f"  FPS: {result.video_info.fps:.1f}")

        if result.audio_info:
            console.print(
                f"  Audio: {result.audio_info.duration:.1f}s @ {result.audio_info.sample_rate}Hz"
            )

        if result.scene_result:
            console.print(f"  Scenes detected: {result.scene_result.total_scenes}")

        console.print(f"  Frames extracted: {len(result.frame_infos)}")

        if speech_analysis:
            console.print("  Speech analysis: ✓")

        if visual_analysis:
            console.print("  Visual analysis: ✓")

        # Display API costs if available
        if report_paths.get("json"):
            import json

            try:
                with open(report_paths["json"]) as f:
                    report_data = json.load(f)
                api_cost = report_data.get("api_cost_summary")
                if api_cost and api_cost.get("total_cost_usd", 0) > 0:
                    cost = api_cost["total_cost_usd"]
                    tokens = api_cost["total_tokens_used"]
                    provider = api_cost.get("provider", "API")
                    model = api_cost.get("model", "unknown")
                    console.print(
                        f"  API usage ({provider}/{model}): {tokens:,} tokens, ${cost:.4f}"
                    )
            except Exception as e:
                logger.debug(f"Could not load API cost data: {e}")

        if report_paths:
            console.print("  Reports: JSON + HTML")

        console.print(f"  Processing time: {processing_time:.1f}s\n")

        console.print(f"[green]✓ Output saved to:[/green] {output_path}")
        if report_paths.get("html"):
            console.print(f"[green]✓ HTML report:[/green] {report_paths['html']}")
        if report_paths.get("json"):
            console.print(f"[green]✓ JSON report:[/green] {report_paths['json']}")

        logger.info(f"Analysis complete. Results saved to {output_path}")

        # Phase 5: Grading (if rubric provided or --grade flag set)
        # Apply defaults if --grade flag is set
        effective_rubric_type = rubric_type or ("general" if grade else None)
        effective_audience = feedback_audience or (
            "student" if (grade or rubric_type or rubric_file) else None
        )
        effective_detail = feedback_detail or (
            "summary" if (grade or rubric_type or rubric_file) else None
        )

        if effective_rubric_type or rubric_file:
            console.print("\n[bold cyan]Generating grading feedback...[/bold cyan]")
            try:
                from video_analyser.analysis.default_rubrics import get_default_rubric
                from video_analyser.analysis.rubric_system import Rubric

                # Load rubric
                if effective_rubric_type:
                    rubric = get_default_rubric(effective_rubric_type)
                    if not rubric:
                        console.print(
                            f"[red]✗ Unknown rubric type: {effective_rubric_type}[/red]\n"
                            f"[dim]Available: academic, business, teaching, technical, creative, sales, legal, medical, political, entertainment, general[/dim]"
                        )
                        raise typer.Exit(1)
                    console.print(
                        f"[cyan]→[/cyan] Using rubric: [bold]{rubric.name}[/bold]"
                    )
                else:
                    if not rubric_file or not rubric_file.exists():
                        console.print(
                            f"[red]✗ Rubric file not found: {rubric_file}[/red]"
                        )
                        raise typer.Exit(1)
                    import json

                    with open(rubric_file) as f:
                        rubric_data = json.load(f)
                    rubric = Rubric.from_dict(rubric_data)
                    console.print(
                        f"[cyan]→[/cyan] Using rubric: [bold]{rubric.name}[/bold]"
                    )

                # Show feedback options
                console.print(
                    f"[cyan]→[/cyan] Audience: [bold]{effective_audience}[/bold], Detail: [bold]{effective_detail}[/bold]"
                )

                # Prepare analysis data for grading
                video_info_dict: dict[str, Any] = {}
                if hasattr(result.video_info, "model_dump"):
                    video_info_dict = result.video_info.model_dump()  # type: ignore[attr-defined]
                elif hasattr(result.video_info, "to_dict"):
                    video_info_dict = result.video_info.to_dict()  # type: ignore[attr-defined]

                grading_data: dict[str, Any] = {
                    "video_info": video_info_dict,
                    "processing_time": processing_time,
                }
                if speech_analysis:
                    grading_data["speech_analysis"] = speech_analysis
                if visual_analysis:
                    grading_data["visual_analysis"] = visual_analysis

                if save_all_formats:
                    # Generate all feedback formats
                    console.print(
                        "[cyan]→[/cyan] Generating feedback in all formats..."
                    )
                    _generate_all_feedback_formats(
                        rubric=rubric,
                        analysis_data=grading_data,
                        audience=effective_audience or "student",
                        output_path=output_path,
                        api_provider=api_provider,
                        api_model=api_model,
                    )
                    console.print(
                        "[green]✓[/green] All feedback formats saved to output directory"
                    )
                else:
                    # Generate single feedback format
                    console.print("[cyan]→[/cyan] Generating LLM feedback...")
                    feedback_text = _generate_grading_feedback(
                        rubric=rubric,
                        analysis_data=grading_data,
                        audience=effective_audience,
                        detail=effective_detail,
                        api_provider=api_provider,
                        api_model=api_model,
                    )

                    # Save feedback report
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    detail_suffix = f"_{effective_detail}" if effective_detail else ""
                    feedback_file = (
                        output_path / f"feedback{detail_suffix}_{timestamp}.md"
                    )

                    with open(feedback_file, "w") as f:
                        f.write("# Presentation Feedback Report\n\n")
                        f.write(f"**Rubric:** {rubric.name}\n")
                        f.write(
                            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                        )
                        f.write("## Feedback\n\n")
                        f.write(feedback_text)

                    console.print("[green]✓[/green] Feedback saved to:")
                    console.print(f"  [bold]{feedback_file}[/bold]")
                    logger.info(f"Grading feedback saved to {feedback_file}")

            except typer.Exit:
                raise
            except Exception as e:
                console.print(f"[red]✗ Grading error: {str(e)}[/red]")
                logger.error(f"Grading error: {e}", exc_info=verbose)
                raise typer.Exit(1) from e

    except VideoProcessingError as e:
        error_msg = str(e)
        progress_tracker.fail_workflow(error_msg)
        logger.error(f"Video processing error: {e}", exc_info=verbose)
        sys.exit(1)

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        progress_tracker.fail_workflow(error_msg)
        logger.error(error_msg, exc_info=True)
        sys.exit(1)


@app.command()
def version() -> None:
    """Show version information."""
    from video_analyser import __version__

    console.print(f"Video Analyser version {__version__}")


@app.command()
def config(
    show_all: bool = typer.Option(
        False, "--all", help="Show all configuration options"
    ),
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
) -> None:
    """Show current configuration."""
    if config_file:
        config_obj = load_config(config_file)
        console.print(f"[blue]Using config file:[/blue] {config_file}")
    else:
        config_obj = get_config()
        console.print("[blue]Using default configuration[/blue]")

    console.print("\n[bold]Application Configuration[/bold]")
    console.print(f"App Name: {config_obj.app_name}")
    console.print(f"Version: {config_obj.version}")
    console.print(f"Debug Mode: {config_obj.debug}")

    if show_all:
        console.print("\n[bold]Processing Settings[/bold]")
        console.print(f"Max Video Size: {config_obj.processing.max_video_size_mb}MB")
        console.print(
            f"Supported Formats: {', '.join(config_obj.processing.supported_formats)}"
        )
        console.print(f"Temp Directory: {config_obj.processing.temp_dir}")

        console.print("\n[bold]Transcription Settings[/bold]")
        console.print(f"Model: {config_obj.transcription.model}")
        console.print(f"Language: {config_obj.transcription.language}")
        console.print(f"Device: {config_obj.transcription.device}")

        console.print("\n[bold]Analysis Settings[/bold]")
        console.print(f"Target WPM: {config_obj.analysis.target_wpm_range}")
        console.print(
            f"Confidence Threshold: {config_obj.analysis.confidence_threshold}"
        )

        console.print("\n[bold]Logging Settings[/bold]")
        console.print(f"Level: {config_obj.logging.level}")
        console.print(f"File: {config_obj.logging.file_path}")
    else:
        console.print("\n[dim]Use --all to see all configuration options[/dim]")


@app.command()
def rubric(
    action: str = typer.Argument(
        ..., help="Action: list, show, create, export, delete"
    ),
    rubric_type: str | None = typer.Option(
        None,
        "--type",
        "-t",
        help="Rubric type (academic, business, teaching, technical, creative, sales, legal, medical, political, entertainment, general)",
    ),
    rubric_id: str | None = typer.Option(
        None, "--id", "-i", help="Rubric ID (for show/delete actions)"
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output file path (for export action)"
    ),
    rubrics_dir: Path = typer.Option(
        Path("rubrics"), "--dir", "-d", help="Directory for storing rubrics"
    ),
) -> None:
    """Manage rubrics for assessment.

    Actions:
        list    - List all available rubrics (default and custom)
        show    - Show details of a specific rubric
        create  - Create a new rubric from a default template
        export  - Export a rubric to JSON file
        delete  - Delete a custom rubric
    """
    from video_analyser.analysis.default_rubrics import list_default_rubrics
    from video_analyser.analysis.rubric_system import RubricRepository

    try:
        repo = RubricRepository(rubrics_dir)

        if action.lower() == "list":
            _rubric_list(repo)

        elif action.lower() == "show":
            if not rubric_id:
                console.print("[red]Error: --id required for show action[/red]")
                raise typer.Exit(1)
            _rubric_show(repo, rubric_id)

        elif action.lower() == "create":
            if not rubric_type:
                console.print("[red]Error: --type required for create action[/red]")
                console.print(
                    f"[dim]Available types: {', '.join(list_default_rubrics())}[/dim]"
                )
                raise typer.Exit(1)
            _rubric_create(repo, rubric_type)

        elif action.lower() == "export":
            if not rubric_id or not output:
                console.print(
                    "[red]Error: --id and --output required for export action[/red]"
                )
                raise typer.Exit(1)
            _rubric_export(repo, rubric_id, output)

        elif action.lower() == "delete":
            if not rubric_id:
                console.print("[red]Error: --id required for delete action[/red]")
                raise typer.Exit(1)
            _rubric_delete(repo, rubric_id)

        else:
            console.print(f"[red]Unknown action: {action}[/red]")
            console.print(
                "[dim]Available actions: list, show, create, export, delete[/dim]"
            )
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1) from e


def _rubric_list(repo: "RubricRepository") -> None:
    """List all rubrics."""
    from video_analyser.analysis.default_rubrics import (
        get_default_rubric,
        list_default_rubrics,
    )

    console.print("\n[bold]Available Default Rubrics[/bold]")
    console.print(
        "[dim](Can be created with: video-analyser rubric create --type <type>)[/dim]\n"
    )

    for rubric_type in list_default_rubrics():
        rubric = get_default_rubric(rubric_type)
        if rubric:
            console.print(f"  [cyan]{rubric_type.upper()}[/cyan]")
            console.print(f"    {rubric.description}")
            console.print(f"    Categories: {len(rubric.categories)}")
            total_criteria = sum(len(cat.criteria) for cat in rubric.categories)
            console.print(f"    Criteria: {total_criteria}\n")

    custom_rubrics = repo.list_rubrics()
    if custom_rubrics:
        console.print("[bold]Custom Rubrics[/bold]\n")
        for rubric in custom_rubrics:
            status = (
                "[yellow]Template[/yellow]"
                if rubric.is_template
                else "[cyan]Custom[/cyan]"
            )
            console.print(f"  {rubric.name} [{status}]")
            console.print(f"    ID: {rubric.id}")
            console.print(f"    Categories: {len(rubric.categories)}")
            console.print(f"    Created: {rubric.created_at.strftime('%Y-%m-%d')}\n")
    else:
        console.print(
            "[dim]No custom rubrics yet. Create one with 'video-analyser rubric create'[/dim]\n"
        )


def _rubric_show(repo: "RubricRepository", rubric_id: str) -> None:
    """Show details of a rubric."""
    rubric = repo.load(rubric_id)
    if not rubric:
        console.print(f"[red]Rubric not found: {rubric_id}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold blue]{rubric.name}[/bold blue]")
    if rubric.description:
        console.print(f"[dim]{rubric.description}[/dim]")

    console.print(f"\nID: {rubric.id}")
    console.print(
        f"Score Range: {rubric.scoring_scale.min_score}-{rubric.scoring_scale.max_score}"
    )
    if rubric.tags:
        console.print(f"Tags: {', '.join(rubric.tags)}")

    console.print("\n[bold]Categories:[/bold]")
    for category in rubric.categories:
        console.print(f"\n  [cyan]{category.name}[/cyan] (weight: {category.weight})")
        if category.description:
            console.print(f"  [dim]{category.description}[/dim]")
        console.print("  Criteria:")
        for criterion in category.criteria:
            console.print(f"    • {criterion.name} (weight: {criterion.weight})")
            if criterion.description:
                console.print(f"      {criterion.description}")

    console.print()


def _rubric_create(repo: "RubricRepository", rubric_type: str) -> None:
    """Create a rubric from a default template."""
    from video_analyser.analysis.default_rubrics import get_default_rubric

    rubric = get_default_rubric(rubric_type)
    if not rubric:
        console.print(f"[red]Unknown rubric type: {rubric_type}[/red]")
        raise typer.Exit(1)

    repo.save(rubric)
    console.print(f"\n[green]✓[/green] Created rubric: [bold]{rubric.name}[/bold]")
    console.print(f"  ID: {rubric.id}")
    console.print(f"  Categories: {len(rubric.categories)}")
    total_criteria = sum(len(cat.criteria) for cat in rubric.categories)
    console.print(f"  Criteria: {total_criteria}")
    console.print(
        f"\n[dim]Rubric saved to: {repo.storage_dir / f'{rubric.id}.json'}[/dim]\n"
    )


def _rubric_export(repo: "RubricRepository", rubric_id: str, output_path: Path) -> None:
    """Export a rubric to JSON file."""
    rubric = repo.load(rubric_id)
    if not rubric:
        console.print(f"[red]Rubric not found: {rubric_id}[/red]")
        raise typer.Exit(1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import json

    with open(output_path, "w") as f:
        json.dump(rubric.to_dict(), f, indent=2, default=str)

    console.print(
        f"\n[green]✓[/green] Exported rubric to: [bold]{output_path}[/bold]\n"
    )


def _rubric_delete(repo: "RubricRepository", rubric_id: str) -> None:
    """Delete a custom rubric."""
    if not repo.load(rubric_id):
        console.print(f"[red]Rubric not found: {rubric_id}[/red]")
        raise typer.Exit(1)

    confirm = typer.confirm(f"Delete rubric {rubric_id}?")
    if not confirm:
        console.print("[dim]Cancelled[/dim]")
        return

    repo.delete(rubric_id)
    console.print(f"\n[green]✓[/green] Deleted rubric: {rubric_id}\n")


def _generate_grading_feedback(
    rubric: Any,
    analysis_data: dict[str, Any],
    audience: str | None = None,
    detail: str | None = None,
    api_provider: str | None = None,
    api_model: str | None = None,
) -> str:
    """
    Generate grading feedback using LLM.

    Args:
        rubric: Rubric object to use for grading
        analysis_data: Analysis data to include in the prompt
        audience: Target audience (student or teacher). Default: student
        detail: Level of detail (short, summary, long). Default: summary
        api_provider: API provider to use (anthropic, openai, google, openrouter)
        api_model: API model to use

    Returns:
        Generated feedback text

    Raises:
        typer.Exit: If API key not found or LLM call fails
    """
    from video_analyser.utils.api_keys import get_api_key

    # Apply defaults
    audience = audience or "student"
    detail = detail or "summary"

    # Determine API provider
    if api_provider:
        provider: str = api_provider
    else:
        # Try to auto-detect from config
        config = get_config()
        api_settings = getattr(config, "api_settings", {})
        if isinstance(api_settings, dict):
            provider = str(api_settings.get("default_provider", "anthropic"))  # type: ignore[arg-type]
        else:
            provider = str(getattr(api_settings, "default_provider", "anthropic"))  # type: ignore[arg-type]

    api_key = get_api_key(provider)  # type: ignore[arg-type]
    if not api_key:
        console.print(
            f"[red]✗ No API key found for {provider}[/red]\n"
            f"[dim]Set {provider.upper()}_API_KEY environment variable[/dim]"
        )
        raise typer.Exit(1)

    # Build prompt for LLM
    prompt = _build_grading_prompt(rubric, analysis_data, audience, detail)

    # Call appropriate LLM
    feedback_text: str = ""
    provider_lower = provider.lower()
    assert isinstance(provider_lower, str)  # For type checker
    if provider_lower == "anthropic":
        from anthropic import Anthropic  # type: ignore[import-not-found]

        client = Anthropic(api_key=api_key)  # type: ignore[attr-defined]
        response = client.messages.create(  # type: ignore[attr-defined]
            model=api_model or "claude-haiku-4-5-20251001",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.content:  # type: ignore[attr-defined]
            content_block = response.content[0]  # type: ignore[attr-defined]
            if hasattr(content_block, "text"):  # type: ignore[arg-type]
                feedback_text = content_block.text  # type: ignore[attr-defined]

    elif provider_lower == "openai":
        from openai import OpenAI  # type: ignore[import-not-found]

        client = OpenAI(api_key=api_key)  # type: ignore[attr-defined]
        response = client.chat.completions.create(  # type: ignore[attr-defined]
            model=api_model or "gpt-4o",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.choices and response.choices[0].message.content:  # type: ignore[attr-defined]
            feedback_text = response.choices[0].message.content  # type: ignore[attr-defined]

    elif provider_lower == "google":
        import google.generativeai as genai  # type: ignore[import-not-found]

        genai.configure(api_key=api_key)  # type: ignore[attr-defined]
        model = genai.GenerativeModel(api_model or "gemini-2.0-flash")  # type: ignore[attr-defined]
        response = model.generate_content(prompt)  # type: ignore[attr-defined]
        if response.text:  # type: ignore[attr-defined]
            feedback_text = response.text  # type: ignore[attr-defined]

    if not feedback_text:
        console.print("[red]✗ Failed to generate feedback[/red]")
        raise typer.Exit(1)

    # Validate and ensure scoring is included for short feedback
    if detail == "short":
        feedback_text = _ensure_score_in_feedback(feedback_text, rubric, analysis_data)

    return feedback_text  # type: ignore[return-value]


def _ensure_score_in_feedback(
    feedback_text: str, rubric: Any, analysis_data: dict[str, Any]
) -> str:
    """
    Ensure feedback includes a valid score, adding one if missing.

    Args:
        feedback_text: The LLM response text
        rubric: The rubric object for fallback scoring
        analysis_data: Analysis data for fallback scoring

    Returns:
        Feedback text with guaranteed score inclusion
    """
    import re

    # Check if score is already present in expected format
    score_pattern = r"Score:\s*(\d{1,3})/100"
    match = re.search(score_pattern, feedback_text, re.IGNORECASE)

    if match:
        score = int(match.group(1))
        if 1 <= score <= 100:
            return feedback_text  # Score is valid and present

    # Score is missing or invalid - add fallback score
    fallback_score = _calculate_fallback_score(rubric, analysis_data)

    # If feedback already starts with content, prepend the score
    if not feedback_text.strip().startswith("Score:"):
        feedback_text = f"Score: {fallback_score}/100\n\n{feedback_text}"

    return feedback_text


def _generate_all_feedback_formats(
    rubric: Any,
    analysis_data: dict[str, Any],
    audience: str | None,
    output_path: Path,
    api_provider: str | None = None,
    api_model: str | None = None,
) -> None:
    """
    Generate feedback in all formats and save to separate files.

    Args:
        rubric: The rubric object
        analysis_data: Analysis data dictionary
        audience: Target audience (student/teacher)
        output_path: Directory to save feedback files
        api_provider: API provider for LLM
        api_model: API model for LLM
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Define all feedback formats
    formats = [
        ("short", "Minimal feedback with score and two sentences"),
        ("summary", "Structured feedback with strengths and improvements"),
        ("long", "Comprehensive detailed feedback"),
    ]

    for detail, description in formats:
        console.print(f"[dim]Generating {detail} feedback...[/dim]")

        feedback_text = _generate_grading_feedback(
            rubric=rubric,
            analysis_data=analysis_data,
            audience=audience,
            detail=detail,
            api_provider=api_provider,
            api_model=api_model,
        )

        # Save to separate file
        filename = f"feedback_{detail}_{timestamp}.md"
        feedback_file = output_path / filename

        with open(feedback_file, "w") as f:
            f.write(f"# Presentation Feedback Report - {detail.title()} Format\n\n")
            f.write(f"**Format:** {description}\n\n")
            f.write("---\n\n")
            f.write(feedback_text)
            f.write("\n\n---\n\n")
            f.write(f"*Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")

        console.print(f"[green]✓[/green] Saved {detail} feedback to {filename}")


def _get_cache_key(video_path: Path) -> str:
    """Generate a cache key for a video file."""
    with open(video_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return f"{video_path.name}_{file_hash}"


def _get_cache_path(cache_dir: Path, cache_key: str, step: str) -> Path:
    """Get the cache file path for a specific processing step."""
    return cache_dir / f"{cache_key}_{step}.json"


def _load_cached_result(cache_path: Path) -> dict[str, Any] | None:
    """Load cached result if it exists and is valid."""
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r") as f:
            cached_data = json.load(f)
        # Check if cache is recent (within 24 hours)
        if time.time() - cached_data.get("timestamp", 0) < 24 * 3600:
            return cached_data.get("data")
    except (json.JSONDecodeError, KeyError):
        pass

    return None


def _save_cached_result(cache_path: Path, data: dict[str, Any]) -> None:
    """Save result to cache."""
    cache_data = {"timestamp": time.time(), "data": data}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache_data, f)


def _run_parallel_analysis(
    video_path: Path,
    output_dir: Path,
    config: VideoAnalyserConfig,
    fast_mode: bool,
    use_cache: bool,
    logger: logging.Logger,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """
    Run transcription and visual analysis in parallel.

    Returns:
        Tuple of (transcription_result, visual_analysis_result)
    """
    cache_dir = output_dir / ".video_analyser_cache"
    cache_key = _get_cache_key(video_path)

    # Initialize results
    transcription_result = None
    visual_analysis_result = None

    # Check cache first
    transcription_cache = None
    visual_cache = None

    if use_cache:
        transcription_cache = _get_cache_path(cache_dir, cache_key, "transcription")
        transcription_result = _load_cached_result(transcription_cache)

        if not fast_mode:
            visual_cache = _get_cache_path(cache_dir, cache_key, "visual")
            visual_analysis_result = _load_cached_result(visual_cache)

    # Prepare tasks for parallel execution
    tasks = {}

    # Always need transcription unless cached
    if transcription_result is None:
        tasks["transcription"] = lambda: _run_transcription_only(
            video_path, config, logger
        )

    # Visual analysis only if not in fast mode and not cached
    if not fast_mode and visual_analysis_result is None:
        tasks["visual"] = lambda: _run_visual_analysis_only(
            video_path, output_dir, config, logger
        )

    # Execute tasks in parallel
    if tasks:
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            future_to_task = {
                executor.submit(func): task_name for task_name, func in tasks.items()
            }

            for future in as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    result = future.result()
                    if task_name == "transcription":
                        transcription_result = result
                        if use_cache and transcription_cache:
                            _save_cached_result(transcription_cache, result)
                    elif task_name == "visual":
                        visual_analysis_result = result
                        if use_cache and visual_cache:
                            _save_cached_result(visual_cache, result)
                except Exception as e:
                    logger.warning(f"Parallel task {task_name} failed: {e}")

    return transcription_result, visual_analysis_result


def _run_transcription_only(
    video_path: Path, config: VideoAnalyserConfig, logger: logging.Logger
) -> dict[str, Any]:
    """Run only transcription analysis."""
    # This would need to be extracted from the pipeline coordinator
    # For now, return a placeholder
    logger.info("Running transcription analysis...")
    time.sleep(1)  # Placeholder for actual transcription
    return {"transcription": "completed", "duration": 10.5}


def _run_visual_analysis_only(
    video_path: Path, output_dir: Path, config: VideoAnalyserConfig, logger: logging.Logger
) -> dict[str, Any]:
    """Run only visual analysis."""
    # This would need to be extracted from the pipeline coordinator
    # For now, return a placeholder
    logger.info("Running visual analysis...")
    time.sleep(2)  # Placeholder for actual visual analysis
    return {"visual_analysis": "completed", "frames_processed": 15}


def _calculate_fallback_score(rubric: Any, analysis_data: dict[str, Any]) -> int:
    """
    Calculate a fallback score based on analysis data quality.

    This uses a balanced heuristic that can result in both high and low scores.

    Args:
        rubric: The rubric object for fallback scoring
        analysis_data: Analysis data dictionary

    Returns:
        Fallback score from 1-100
    """
    # Start with a modest baseline - assumes basic competence
    score = 35

    # POSITIVE FACTORS (+ points)
    # Content quality indicators
    if analysis_data.get("speech_analysis"):
        score += 20  # Strong evidence of developed content

    if analysis_data.get("visual_analysis"):
        score += 15  # Professional presentation quality

    # Video technical quality
    video_info = analysis_data.get("video_info", {})
    if video_info.get("duration", 0) > 120:  # Substantial content
        score += 10
    elif video_info.get("duration", 0) > 60:  # Decent length
        score += 5

    if video_info.get("fps", 0) > 30:  # High quality video
        score += 8
    elif video_info.get("fps", 0) > 24:  # Good quality video
        score += 5

    # Processing complexity (longer analysis suggests richer content)
    processing_time = analysis_data.get("processing_time", 0)
    if processing_time > 60:  # Very complex analysis
        score += 10
    elif processing_time > 30:  # Moderately complex
        score += 5

    # NEGATIVE FACTORS (- points)
    # Poor quality indicators
    if video_info.get("duration", 0) < 30:  # Very short presentation
        score -= 15

    if video_info.get("fps", 0) < 15:  # Poor video quality
        score -= 10

    if not analysis_data.get("speech_analysis") and not analysis_data.get(
        "visual_analysis"
    ):
        score -= 20  # No meaningful analysis data suggests poor quality

    # Ensure score stays within realistic bounds
    return max(1, min(100, score))


def _build_grading_prompt(
    rubric: Any,
    analysis_data: dict[str, Any],
    audience: str = "student",
    detail: str = "summary",
) -> str:  # type: ignore[return]
    """Build the prompt for LLM grading.

    Args:
        rubric: Rubric object with categories, criteria, and scoring scale
        analysis_data: Analysis data dictionary to include in prompt
        audience: Target audience ('student' or 'teacher')
        detail: Level of detail ('short', 'summary', or 'long')

    Returns:
        Formatted prompt string for LLM evaluation
    """
    import json

    rubric_name = str(rubric.name)  # type: ignore[attr-defined]
    rubric_desc = str(rubric.description or "")  # type: ignore[attr-defined]

    # Build audience-specific instruction
    if audience == "teacher":
        system_instruction = (
            "You are an expert presentation evaluator providing detailed assessment feedback for educators. "
            "Focus on rubric-based scoring, assessment findings, and areas for student development."
        )
    else:  # student
        system_instruction = (
            "You are a supportive presentation coach providing encouraging, developmental feedback to a student. "
            "Focus on growth, strengths, and specific, actionable areas for improvement."
        )

    prompt: str = (
        system_instruction
        + "\n\nPlease evaluate the following presentation based on the provided rubric.\n\n"
    )
    prompt += "## Rubric: " + rubric_name + "\n" + rubric_desc + "\n\n"
    prompt += "### Rubric Categories and Criteria:\n"

    for category in rubric.categories:  # type: ignore[union-attr]
        cat_name = str(category.name)  # type: ignore[attr-defined]
        cat_weight = str(category.weight)  # type: ignore[attr-defined]
        cat_desc = str(category.description or "")  # type: ignore[attr-defined]
        prompt += "\n**" + cat_name + "** (weight: " + cat_weight + ")\n"
        prompt += cat_desc + "\n\n"
        for criterion in category.criteria:  # type: ignore[union-attr]
            crit_name = str(criterion.name)  # type: ignore[attr-defined]
            crit_desc = str(criterion.description or "")  # type: ignore[attr-defined]
            prompt += "- " + crit_name + ": " + crit_desc + "\n"
            if criterion.scoring_guide:  # type: ignore[union-attr]
                guide = str(criterion.scoring_guide)  # type: ignore[attr-defined]
                prompt += "  Scoring guide: " + guide + "\n"

    prompt += "\n### Scoring Scale:\n"
    scoring_items: list[tuple[Any, Any]] = sorted(rubric.scoring_scale.labels.items())  # type: ignore[union-attr]
    for score, label in scoring_items:
        prompt += "- " + str(score) + ": " + str(label) + "\n"

    max_score = str(rubric.scoring_scale.max_score)  # type: ignore[union-attr]
    analysis_json = json.dumps(analysis_data, indent=2, default=str)

    # Build closing section based on detail level and audience
    closing = "\n\nPlease provide:\n"

    if detail == "short":
        # Short: minimal feedback - overall score + two sentences
        if audience == "teacher":
            closing += "REQUIRED FORMAT: Start with 'Score: X/100' where X is a number from 1-100 based on holistic rubric evaluation. Then provide exactly two sentences: one summarizing key strengths, and one identifying main areas for improvement.\n"
        else:  # student
            closing += "REQUIRED FORMAT: Start with 'Score: X/100' where X is a number from 1-100 based on holistic rubric evaluation. Then provide exactly two sentences: one highlighting what you did well, and one suggesting specific areas to improve.\n"
    elif detail == "summary":
        # Summary: 2 paragraphs (strengths + improvements)
        if audience == "teacher":
            closing += "1. Overall assessment with detailed scores for each rubric criterion in format 'Criterion Name: X/100' (1-100 scale)\n"
            closing += "2. Key strengths demonstrated\n"
            closing += "3. Areas for student improvement\n"
        else:  # student
            closing += "1. What you did well (strengths)\n"
            closing += "2. Specific areas to work on for next time\n"
    else:  # long
        # Long: full detailed feedback (original behavior)
        if audience == "teacher":
            closing += "1. Overall assessment\n"
            closing += "2. Detailed scores for each criterion (1-" + max_score + ")\n"
            closing += "3. Specific feedback for each category\n"
            closing += "4. Key strengths\n"
            closing += "5. Areas for improvement\n"
            closing += "6. Recommendations for student development\n"
        else:  # student
            closing += "1. Overall assessment\n"
            closing += "2. Scores for each criterion (1-" + max_score + ")\n"
            closing += "3. What you did well\n"
            closing += "4. What to focus on next time\n"
            closing += "5. Specific tips for improvement\n"
            closing += "6. Next steps for growth\n"

    closing += "\nFormat the response clearly."
    if detail != "short":
        closing += " Use sections for each category."

    prompt_end: str = "\n\n### Presentation Analysis Data:\n```json\n"
    prompt_end += analysis_json
    prompt_end += "\n```"
    prompt_end += closing

    result: str = prompt + prompt_end  # type: ignore[assignment]
    return result


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    app()
