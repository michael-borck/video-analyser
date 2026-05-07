#!/usr/bin/env python3
"""Generate test video fixtures for Video Analyser development and testing."""

import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import rich.progress as progress
from rich.console import Console

console = Console()


class VideoSpec(NamedTuple):
    """Specification for generating a test video."""

    name: str
    duration: int  # seconds
    width: int
    height: int
    fps: int
    audio_freq: int | None  # Hz, None for no audio
    text_overlay: str | None
    description: str


# Test video specifications
TEST_VIDEOS = [
    # Minimal test videos for automated testing
    VideoSpec(
        name="minimal_test",
        duration=2,
        width=160,
        height=120,
        fps=1,
        audio_freq=440,  # A4 note
        text_overlay="Test Video",
        description="Minimal test video for CI/CD",
    ),
    VideoSpec(
        name="no_audio_test",
        duration=3,
        width=160,
        height=120,
        fps=1,
        audio_freq=None,
        text_overlay="Silent Test",
        description="Test video without audio track",
    ),
    VideoSpec(
        name="short_presentation",
        duration=10,
        width=640,
        height=480,
        fps=2,
        audio_freq=800,
        text_overlay="Sample Presentation",
        description="Short presentation-style video",
    ),
    VideoSpec(
        name="different_format_test",
        duration=5,
        width=320,
        height=240,
        fps=1,
        audio_freq=600,
        text_overlay="Format Test",
        description="Test video for format conversion testing",
    ),
    # Larger development samples
    VideoSpec(
        name="dev_sample_good",
        duration=30,
        width=1280,
        height=720,
        fps=5,
        audio_freq=1000,
        text_overlay="Good Presentation Sample",
        description="Development sample - good presentation style",
    ),
    VideoSpec(
        name="dev_sample_needs_work",
        duration=25,
        width=1280,
        height=720,
        fps=5,
        audio_freq=1200,
        text_overlay="Needs Improvement Sample",
        description="Development sample - presentation needing improvement",
    ),
]


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_espeak() -> bool:
    """Check if espeak is available for text-to-speech."""
    try:
        result = subprocess.run(
            ["espeak", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def create_speech_audio(
    text: str, output_path: Path, duration: int | None = None
) -> bool:
    """
    Create speech audio from text using espeak if available.

    Args:
        text: Text to convert to speech
        output_path: Output audio file path
        duration: Target duration in seconds (optional, not used with espeak)

    Returns:
        True if successful, False otherwise
    """
    # Note: duration parameter is not used with espeak but kept for API compatibility
    _ = duration  # Mark as intentionally unused
    if not check_espeak():
        return False

    try:
        # Create speech with espeak
        cmd = [
            "espeak",
            "-s",
            "150",  # words per minute
            "-v",
            "en",  # voice
            "-w",
            str(output_path),  # write to file
            text,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.returncode == 0 and output_path.exists()

    except Exception:
        return False


def generate_video(spec: VideoSpec, output_path: Path, format_ext: str = "mp4") -> bool:
    """
    Generate a test video based on specification.

    Args:
        spec: Video specification
        output_path: Output file path
        format_ext: Output format extension

    Returns:
        True if successful, False otherwise
    """
    output_file = output_path / f"{spec.name}.{format_ext}"

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y"]  # -y to overwrite existing files

    # Video input: colored bars with text overlay
    video_filter = f"testsrc2=duration={spec.duration}:size={spec.width}x{spec.height}:rate={spec.fps}"

    if spec.text_overlay:
        # Add text overlay
        video_filter += f",drawtext=text='{spec.text_overlay}':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-text_w)/2:y=(h-text_h)/2"

    cmd.extend(["-f", "lavfi", "-i", video_filter])

    # Audio input (if specified)
    if spec.audio_freq:
        audio_input = f"sine=frequency={spec.audio_freq}:duration={spec.duration}"
        cmd.extend(["-f", "lavfi", "-i", audio_input])
        cmd.extend(["-c:a", "aac", "-b:a", "32k"])
    else:
        cmd.extend(["-an"])  # No audio

    # Video encoding
    cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "28"])

    # Format-specific options
    if format_ext == "webm":
        cmd.extend(["-c:v", "libvpx-vp9", "-c:a", "libvorbis"])
    elif format_ext == "avi":
        cmd.extend(["-c:v", "libx264", "-c:a", "mp3"])

    cmd.append(str(output_file))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            console.print(f"[red]Error generating {output_file}:[/red]")
            console.print(result.stderr)
            return False

        file_size = output_file.stat().st_size
        console.print(
            f"[green]✓[/green] Generated {output_file.name} ({file_size:,} bytes)"
        )
        return True

    except Exception as e:
        console.print(f"[red]Error generating {output_file}: {e}[/red]")
        return False


def create_realistic_samples(output_path: Path) -> bool:
    """Create more realistic sample videos with speech-like audio patterns."""

    samples = [
        {
            "name": "presentation_with_pauses",
            "description": "Realistic presentation with pauses and varied audio",
            "duration": 45,
            "script": "Welcome everyone to today's presentation. As you can see on this slide, our quarterly results show significant growth. Let me walk you through the key metrics.",
        },
        {
            "name": "fast_speaker",
            "description": "Fast-paced presentation style",
            "duration": 30,
            "script": "Good morning team, today we'll quickly review our performance metrics, analyze market trends, and discuss implementation strategies for next quarter.",
        },
        {
            "name": "with_filler_words",
            "description": "Presentation with common filler words",
            "duration": 35,
            "script": "So, um, today we're going to, like, discuss our quarterly results. You know, the numbers are, uh, really exciting for the team.",
        },
    ]

    success_count = 0

    # Check if we can create speech audio
    speech_available = check_espeak()
    if not speech_available:
        console.print(
            "[yellow]Warning: espeak not available. Videos will have tone audio instead of speech.[/yellow]"
        )
        console.print("Install espeak for realistic speech: sudo apt install espeak")

    for sample in samples:
        output_file = output_path / f"{sample['name']}.mp4"
        audio_file = None

        try:
            if speech_available:
                # Try to create speech audio
                audio_file = output_path / f"{sample['name']}_speech.wav"
                if create_speech_audio(
                    sample["script"], audio_file, sample["duration"]
                ):
                    # Use speech audio
                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        f"testsrc2=duration={sample['duration']}:size=1280x720:rate=2",
                        "-i",
                        str(audio_file),
                        "-filter_complex",
                        f"[0:v]drawtext=text='{sample['name']}':fontcolor=white:fontsize=36:box=1:boxcolor=black@0.7:boxborderw=5:x=(w-text_w)/2:y=(h-text_h)/2[v]",
                        "-map",
                        "[v]",
                        "-map",
                        "1:a",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "23",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "64k",
                        "-t",
                        str(sample["duration"]),
                        str(output_file),
                    ]
                else:
                    speech_available = False  # Fall back to tone audio

            if not speech_available:
                # Fall back to tone audio
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    f"testsrc2=duration={sample['duration']}:size=1280x720:rate=2",
                    "-f",
                    "lavfi",
                    "-i",
                    f"sine=frequency=440:duration={sample['duration']}",
                    "-filter_complex",
                    f"[0:v]drawtext=text='{sample['name']}':fontcolor=white:fontsize=36:box=1:boxcolor=black@0.7:boxborderw=5:x=(w-text_w)/2:y=(h-text_h)/2[v]",
                    "-map",
                    "[v]",
                    "-map",
                    "1:a",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "64k",
                    "-t",
                    str(sample["duration"]),
                    str(output_file),
                ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                file_size = output_file.stat().st_size
                audio_type = "speech" if speech_available else "tone"
                console.print(
                    f"[green]✓[/green] Generated {sample['name']}.mp4 ({file_size:,} bytes) [{audio_type}]"
                )
                console.print(f"  [dim]{sample['description']}[/dim]")
                success_count += 1
            else:
                console.print(f"[red]✗[/red] Failed to generate {sample['name']}.mp4")
                console.print(f"  [dim]{result.stderr}[/dim]")

        except Exception as e:
            console.print(f"[red]✗[/red] Error generating {sample['name']}.mp4: {e}")
        finally:
            # Clean up temporary audio file
            if audio_file and audio_file.exists():
                audio_file.unlink()

    return success_count == len(samples)


def main() -> int:
    """Main function to generate all test videos."""
    console.print("[bold blue]Video Analyser Test Video Generator[/bold blue]\n")

    # Check prerequisites
    if not check_ffmpeg():
        console.print(
            "[red]Error: ffmpeg not found. Please install ffmpeg first.[/red]"
        )
        console.print("Install with:")
        console.print("  macOS: brew install ffmpeg")
        console.print("  Ubuntu: sudo apt install ffmpeg")
        console.print("  Windows: Download from https://ffmpeg.org/")
        return 1

    # Create output directories
    base_path = Path(__file__).parent.parent
    fixtures_path = base_path / "tests" / "fixtures"
    samples_path = base_path / "samples"

    fixtures_path.mkdir(parents=True, exist_ok=True)
    samples_path.mkdir(parents=True, exist_ok=True)

    console.print("Generating test videos in:")
    console.print(f"  Test fixtures: {fixtures_path}")
    console.print(f"  Development samples: {samples_path}\n")

    total_videos = 0
    successful_videos = 0

    # Generate basic test videos
    console.print("[bold]Generating test fixture videos...[/bold]")
    with progress.Progress() as prog:
        task = prog.add_task("Generating videos...", total=len(TEST_VIDEOS))

        for spec in TEST_VIDEOS:
            total_videos += 1

            # Determine output path based on video purpose
            output_path = samples_path if "dev_sample" in spec.name else fixtures_path

            # Generate MP4 version
            if generate_video(spec, output_path, "mp4"):
                successful_videos += 1

            # Generate additional formats for format testing
            if spec.name == "different_format_test":
                for fmt in ["webm", "avi"]:
                    total_videos += 1
                    if generate_video(spec, fixtures_path, fmt):
                        successful_videos += 1

            prog.advance(task)

    # Generate realistic samples
    console.print("\n[bold]Generating realistic sample videos...[/bold]")
    if create_realistic_samples(samples_path):
        console.print("[green]✓[/green] All realistic samples generated successfully")
    else:
        console.print("[yellow]⚠[/yellow] Some realistic samples failed to generate")

    # Summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"Generated {successful_videos}/{total_videos} videos successfully")

    if successful_videos == total_videos:
        console.print("[green]All videos generated successfully! 🎉[/green]")
        return 0
    else:
        console.print(
            f"[yellow]{total_videos - successful_videos} videos failed to generate[/yellow]"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
