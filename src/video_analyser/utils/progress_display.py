"""CLI progress display utilities for real-time workflow visualization."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from video_analyser.core.progress_tracker import ProgressTracker

console = Console()


@dataclass
class OperationProgress:
    """Track progress for a single operation."""

    name: str
    total_steps: float = 1.0
    current_step: float = 0.0

    def update(self, progress: float) -> None:
        """Update progress (0.0 to 1.0)."""
        self.current_step = min(progress, self.total_steps)

    def get_percentage(self) -> int:
        """Get progress as percentage."""
        return int((self.current_step / self.total_steps) * 100)


class CLIProgressTracker(ProgressTracker):
    """Tracks and displays progress for video analysis workflow."""

    def __init__(self):
        """Initialize progress tracker."""
        super().__init__()
        self.cli_operations: dict[str, OperationProgress] = {}
        self.current_operation: str | None = None
        self.progress: Progress | None = None
        self.tasks: dict[str, int] = {}

    def start_workflow(
        self, workflow_name: str, operations: list[tuple[str, str, float]]
    ) -> None:
        """
        Start a new workflow with named operations.

        Args:
            workflow_name: Name of the workflow
            operations: List of (op_id, op_name, weight) tuples
        """
        self.cli_operations = {}
        self.tasks = {}

        # Display workflow header
        console.print(f"\n[bold blue]▶ {workflow_name}[/bold blue]\n")

        # Create progress bar
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        )

        # Start the progress bar
        self.progress.start()

        # Register operations
        for op_id, op_name, weight in operations:
            self.cli_operations[op_id] = OperationProgress(op_name, total_steps=weight)

    def start_operation(
        self,
        operation_id: str,
        operation_name: str | None = None,  # noqa: ARG002
        total_steps: int | None = None,  # noqa: ARG002
        details: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> None:
        """Mark an operation as started."""
        # Parameters operation_name, total_steps, details are part of the interface
        # but not currently used in the CLI implementation
        if not self.progress:
            return

        self.current_operation = operation_id
        if operation_id in self.cli_operations:
            op = self.cli_operations[operation_id]
            if operation_id not in self.tasks:
                task_id = self.progress.add_task(op.name, total=100)
                self.tasks[operation_id] = task_id

    def update_progress(
        self,
        operation_id: str,
        progress: float,
        current_step: str | None = None,
        current_step_number: int | None = None,  # noqa: ARG002
        details: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> None:
        """
        Update current operation progress.

        Args:
            operation_id: Operation ID (uses current if None)
            progress: Progress value (0.0 to 1.0)
            current_step: Optional description of current step
            current_step_number: Optional step number
            details: Optional additional details
        """
        # Parameters current_step_number and details are part of the interface
        # but not currently used in the CLI implementation
        if not self.progress:
            return

        # Use provided operation_id or current operation
        op_id = operation_id or self.current_operation
        if not op_id:
            return

        if op_id in self.cli_operations:
            self.cli_operations[op_id].update(progress)
            task_id = self.tasks.get(op_id)
            if task_id is not None:
                percentage = int((progress or 0) * 100)
                desc = self.cli_operations[op_id].name
                if current_step:
                    desc = f"{desc} • {current_step}"
                from rich.progress import TaskID

                self.progress.update(
                    TaskID(task_id), completed=percentage, description=desc
                )

    def complete_operation(
        self,
        operation_id: str,
        details: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> None:
        """Mark an operation as complete."""
        # Parameter details is part of the interface but not currently used
        # in the CLI implementation
        if not self.progress:
            return

        if operation_id in self.tasks:
            from rich.progress import TaskID

            task_id = self.tasks[operation_id]
            self.progress.update(TaskID(task_id), completed=100)

    def fail_operation(
        self,
        operation_id: str,
        error: str,  # noqa: ARG002
        details: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> None:
        """Mark an operation as failed."""
        # Parameters error and details are part of the interface but not currently
        # used in the CLI implementation
        if not self.progress:
            return

        if operation_id in self.tasks:
            from rich.progress import TaskID

            task_id = self.tasks[operation_id]
            op_name = self.cli_operations.get(operation_id)
            desc = f"❌ {op_name.name if op_name else 'Operation'} failed"
            self.progress.update(TaskID(task_id), description=desc)

    def create_sub_progress_callback(
        self, operation_id: str, step_weight: float = 1.0, step_name: str | None = None
    ) -> Callable[[float], None]:
        """
        Create a sub-progress callback for a weighted operation step.

        Args:
            operation_id: ID of the parent operation
            step_weight: Weight of this step (0.0-1.0)
            step_name: Name of the step

        Returns:
            Callback function that accepts progress (0.0-1.0)
        """

        def callback(progress: float) -> None:
            # Update with step name and weighted progress
            self.update_progress(
                operation_id=operation_id,
                progress=progress * step_weight,
                current_step=step_name,
            )

        return callback

    def complete_workflow(self) -> None:
        """Complete the workflow and display completion message."""
        if self.progress:
            self.progress.stop()
        console.print("[green]✓ Analysis complete![/green]\n")

    def fail_workflow(self, error_message: str) -> None:
        """Fail the workflow with an error message."""
        if self.progress:
            self.progress.stop()
        console.print(f"[red]✗ Analysis failed: {error_message}[/red]\n")


def create_progress_callback(
    tracker: CLIProgressTracker, operation_id: str
) -> Callable[[float], None]:
    """
    Create a progress callback function for the PipelineCoordinator.

    Args:
        tracker: CLIProgressTracker instance
        operation_id: ID of the operation to update progress for

    Returns:
        Callback function accepting progress (0.0-1.0)
    """

    def callback(progress: float) -> None:
        tracker.update_progress(operation_id=operation_id, progress=progress)

    return callback
