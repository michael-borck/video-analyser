"""Command-line interface for Video Analyser."""

import hashlib
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from video_analyser.core.exceptions import VideoProcessingError
from video_analyser.core.pipeline_coordinator import PipelineCoordinator
from video_analyser.utils.config import VideoAnalyserConfig, get_config, load_config
from video_analyser.utils.progress_display import CLIProgressTracker

console = Console()


def _cmd_analyse(args) -> None:
    """Analyse a video file (the family's default / bare-positional command)."""
    video_path = args.video_path
    output_dir = args.output
    config_file = args.config
    verbose = args.verbose
    api_provider = args.api_provider
    api_model = args.api_model
    use_api = args.use_api
    fast_mode = args.fast
    parallel_processing = args.parallel
    use_cache = args.cache

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
        fast_mode=fast_mode,
        parallel_processing=parallel_processing,
        use_cache=use_cache,
    )


def _analyze_video_cli(
    video_path: Path,
    output_dir: Path | None,
    config: VideoAnalyserConfig,
    config_file: Path | None = None,
    verbose: bool = False,
    logger: logging.Logger = None,
    fast_mode: bool = False,
    parallel_processing: bool = True,
    use_cache: bool = True,
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


def _cmd_config(args) -> None:
    """Show current configuration."""
    show_all = args.all
    config_file = args.config
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
        with open(cache_path) as f:
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
    video_path: Path,
    output_dir: Path,
    config: VideoAnalyserConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Run only visual analysis."""
    # This would need to be extracted from the pipeline coordinator
    # For now, return a placeholder
    logger.info("Running visual analysis...")
    time.sleep(2)  # Placeholder for actual visual analysis
    return {"visual_analysis": "completed", "frames_processed": 15}


def main() -> None:
    """Entry point for the CLI (argparse + lens-contract, the family pattern)."""
    import argparse
    from importlib.metadata import version as _pkg_version

    from lens_contract import run_contract_subcommands

    from video_analyser.manifest import MANIFEST

    # `serve` and `manifest` are the family's shared subcommands (lens-contract).
    if run_contract_subcommands(
        MANIFEST,
        app_path="video_analyser.api:app",
        default_port=8002,
        env_prefix="VIDEO_ANALYSER",
    ):
        return

    argv = sys.argv[1:]

    # Video-specific subcommands beyond the contract.
    if argv and argv[0] == "config":
        p = argparse.ArgumentParser(prog="video-analyser config")
        p.add_argument(
            "--all", action="store_true", help="Show all configuration options"
        )
        p.add_argument(
            "--config", "-c", type=Path, default=None, help="Configuration file path"
        )
        _cmd_config(p.parse_args(argv[1:]))
        return

    # Default (bare positional) command: analyse a video.
    p = argparse.ArgumentParser(
        prog="video-analyser",
        description="Analyse a video file for presentation feedback",
        epilog="subcommands: `serve`, `manifest`, `config` (run each with -h)",
    )
    p.add_argument(
        "--version", action="version", version=_pkg_version("video-analyser")
    )
    p.add_argument("video_path", type=Path, help="Path to the video file to analyse")
    p.add_argument(
        "--output", "-o", type=Path, default=None, help="Output directory for reports"
    )
    p.add_argument(
        "--config", "-c", type=Path, default=None, help="Configuration file path"
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    p.add_argument(
        "--api-provider",
        dest="api_provider",
        default=None,
        help="API provider for captioning",
    )
    p.add_argument("--api-model", dest="api_model", default=None, help="API model")
    p.add_argument(
        "--use-api", dest="use_api", action="store_true", help="Use API for captioning"
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: transcript only (skip visual analysis)",
    )
    p.add_argument(
        "--no-parallel",
        dest="parallel",
        action="store_false",
        help="Disable parallel processing",
    )
    p.add_argument(
        "--no-cache", dest="cache", action="store_false", help="Disable caching"
    )
    _cmd_analyse(p.parse_args(argv))


if __name__ == "__main__":
    main()
