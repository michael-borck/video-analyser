"""Comprehensive progress tracking system for video processing operations."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class OperationStatus(Enum):
    """Status of a processing operation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressUpdate:
    """Progress update information."""

    operation_id: str
    operation_name: str
    status: OperationStatus
    progress: float  # 0.0 to 1.0
    current_step: str
    total_steps: int = 1
    current_step_number: int = 1
    estimated_seconds_remaining: float | None = None
    elapsed_seconds: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


class ProgressTracker:
    """Central progress tracking system for video processing operations."""

    def __init__(self):
        """Initialize progress tracker."""
        self.operations: dict[str, ProgressUpdate] = {}
        self.callbacks: list[Callable[[ProgressUpdate], None]] = []
        self.start_times: dict[str, float] = {}

    def add_callback(self, callback: Callable[[ProgressUpdate], None]) -> None:
        """
        Add a progress update callback function.

        Args:
            callback: Function to call with progress updates
        """
        self.callbacks.append(callback)
        logger.debug(f"Added progress callback: {callback}")

    def remove_callback(self, callback: Callable[[ProgressUpdate], None]) -> None:
        """
        Remove a progress update callback function.

        Args:
            callback: Function to remove from callbacks
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.debug(f"Removed progress callback: {callback}")

    def start_operation(
        self,
        operation_id: str,
        operation_name: str,
        total_steps: int = 1,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Start tracking a new operation.

        Args:
            operation_id: Unique identifier for the operation
            operation_name: Human-readable name for the operation
            total_steps: Total number of steps in the operation
            details: Additional details about the operation
        """
        self.start_times[operation_id] = time.time()

        progress_update = ProgressUpdate(
            operation_id=operation_id,
            operation_name=operation_name,
            status=OperationStatus.RUNNING,
            progress=0.0,
            current_step="Starting...",
            total_steps=total_steps,
            current_step_number=0,
            elapsed_seconds=0.0,
            details=details or {},
        )

        self.operations[operation_id] = progress_update
        self._notify_callbacks(progress_update)

        logger.info(f"Started operation: {operation_name} ({operation_id})")

    def update_progress(
        self,
        operation_id: str,
        progress: float,
        current_step: str | None = None,
        current_step_number: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Update progress for an operation.

        Args:
            operation_id: Unique identifier for the operation
            progress: Progress value (0.0 to 1.0)
            current_step: Description of current step
            current_step_number: Current step number (1-based)
            details: Additional details to update
        """
        if operation_id not in self.operations:
            logger.warning(f"Unknown operation ID: {operation_id}")
            return

        operation = self.operations[operation_id]

        # Update progress
        operation.progress = max(0.0, min(1.0, progress))

        if current_step:
            operation.current_step = current_step

        if current_step_number is not None:
            operation.current_step_number = current_step_number

        if details:
            operation.details.update(details)

        # Calculate elapsed time and estimate remaining time
        if operation_id in self.start_times:
            operation.elapsed_seconds = time.time() - self.start_times[operation_id]

            if operation.progress > 0.0:
                total_estimated = operation.elapsed_seconds / operation.progress
                operation.estimated_seconds_remaining = max(
                    0.0, total_estimated - operation.elapsed_seconds
                )

        self._notify_callbacks(operation)

        logger.debug(
            f"Progress update: {operation.operation_name} - "
            f"{operation.progress:.1%} - {operation.current_step}"
        )

    def complete_operation(
        self, operation_id: str, details: dict[str, Any] | None = None
    ) -> None:
        """
        Mark an operation as completed.

        Args:
            operation_id: Unique identifier for the operation
            details: Additional completion details
        """
        if operation_id not in self.operations:
            logger.warning(f"Unknown operation ID: {operation_id}")
            return

        operation = self.operations[operation_id]
        operation.status = OperationStatus.COMPLETED
        operation.progress = 1.0
        operation.current_step = "Completed"
        operation.estimated_seconds_remaining = 0.0

        if operation_id in self.start_times:
            operation.elapsed_seconds = time.time() - self.start_times[operation_id]
            del self.start_times[operation_id]

        if details:
            operation.details.update(details)

        self._notify_callbacks(operation)

        logger.info(
            f"Completed operation: {operation.operation_name} "
            f"in {operation.elapsed_seconds:.1f}s"
        )

    def fail_operation(
        self, operation_id: str, error: str, details: dict[str, Any] | None = None
    ) -> None:
        """
        Mark an operation as failed.

        Args:
            operation_id: Unique identifier for the operation
            error: Error message describing the failure
            details: Additional failure details
        """
        if operation_id not in self.operations:
            logger.warning(f"Unknown operation ID: {operation_id}")
            return

        operation = self.operations[operation_id]
        operation.status = OperationStatus.FAILED
        operation.current_step = f"Failed: {error}"
        operation.estimated_seconds_remaining = None

        if operation_id in self.start_times:
            operation.elapsed_seconds = time.time() - self.start_times[operation_id]
            del self.start_times[operation_id]

        failure_details = {"error": error}
        if details:
            failure_details.update(details)
        operation.details.update(failure_details)

        self._notify_callbacks(operation)

        logger.error(f"Failed operation: {operation.operation_name} - {error}")

    def cancel_operation(self, operation_id: str) -> None:
        """
        Cancel an operation.

        Args:
            operation_id: Unique identifier for the operation
        """
        if operation_id not in self.operations:
            logger.warning(f"Unknown operation ID: {operation_id}")
            return

        operation = self.operations[operation_id]
        operation.status = OperationStatus.CANCELLED
        operation.current_step = "Cancelled"
        operation.estimated_seconds_remaining = None

        if operation_id in self.start_times:
            operation.elapsed_seconds = time.time() - self.start_times[operation_id]
            del self.start_times[operation_id]

        self._notify_callbacks(operation)

        logger.info(f"Cancelled operation: {operation.operation_name}")

    def get_operation_status(self, operation_id: str) -> ProgressUpdate | None:
        """
        Get current status of an operation.

        Args:
            operation_id: Unique identifier for the operation

        Returns:
            Current progress update or None if operation not found
        """
        return self.operations.get(operation_id)

    def get_all_operations(self) -> dict[str, ProgressUpdate]:
        """
        Get status of all operations.

        Returns:
            Dictionary mapping operation IDs to progress updates
        """
        return self.operations.copy()

    def clear_completed_operations(self) -> None:
        """Remove completed and failed operations from tracking."""
        to_remove = [
            op_id
            for op_id, operation in self.operations.items()
            if operation.status
            in (
                OperationStatus.COMPLETED,
                OperationStatus.FAILED,
                OperationStatus.CANCELLED,
            )
        ]

        for op_id in to_remove:
            del self.operations[op_id]
            if op_id in self.start_times:
                del self.start_times[op_id]

        logger.debug(f"Cleared {len(to_remove)} completed operations")

    def create_sub_progress_callback(
        self, operation_id: str, step_weight: float = 1.0, step_name: str | None = None
    ) -> Callable[[float], None]:
        """
        Create a progress callback for a sub-operation.

        Args:
            operation_id: Parent operation ID
            step_weight: Weight of this step in overall progress (0.0 to 1.0)
            step_name: Name of the step

        Returns:
            Progress callback function for the sub-operation
        """

        def sub_progress_callback(sub_progress: float) -> None:
            if operation_id not in self.operations:
                return

            operation = self.operations[operation_id]

            # Calculate overall progress based on current step and sub-progress
            step_number = operation.current_step_number
            total_steps = operation.total_steps

            if total_steps > 0:
                # Progress from completed steps
                completed_progress = (step_number - 1) / total_steps
                # Progress from current step
                current_step_progress = (sub_progress * step_weight) / total_steps
                # Total progress
                total_progress = completed_progress + current_step_progress
            else:
                total_progress = sub_progress * step_weight

            current_step_desc = step_name or operation.current_step

            self.update_progress(
                operation_id=operation_id,
                progress=total_progress,
                current_step=current_step_desc,
                details={"sub_progress": sub_progress},
            )

        return sub_progress_callback

    def _notify_callbacks(self, progress_update: ProgressUpdate) -> None:
        """
        Notify all registered callbacks of a progress update.

        Args:
            progress_update: Progress update to send to callbacks
        """
        for callback in self.callbacks:
            try:
                callback(progress_update)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")


class CompositeProgressTracker:
    """Progress tracker for complex workflows with multiple operations."""

    def __init__(self, tracker: ProgressTracker):
        """
        Initialize composite progress tracker.

        Args:
            tracker: Base progress tracker to use
        """
        self.tracker = tracker
        self.workflow_id: str | None = None
        self.operations: list[tuple[str, str, float]] = []  # (id, name, weight)
        self.current_operation_index = 0

    def start_workflow(
        self,
        workflow_id: str,
        workflow_name: str,
        operations: list[tuple[str, str, float]],  # (id, name, weight)
    ) -> None:
        """
        Start a complex workflow with multiple operations.

        Args:
            workflow_id: Unique identifier for the workflow
            workflow_name: Human-readable name for the workflow
            operations: List of (operation_id, operation_name, weight) tuples
        """
        self.workflow_id = workflow_id
        self.operations = operations
        self.current_operation_index = 0

        total_weight = sum(weight for _, _, weight in operations)
        if abs(total_weight - 1.0) > 0.01:  # Allow for small floating point errors
            logger.warning(f"Operation weights sum to {total_weight}, not 1.0")

        self.tracker.start_operation(
            operation_id=workflow_id,
            operation_name=workflow_name,
            total_steps=len(operations),
            details={
                "operations": [
                    {"id": op_id, "name": op_name, "weight": weight}
                    for op_id, op_name, weight in operations
                ]
            },
        )

    def start_next_operation(self) -> Callable[[float], None] | None:
        """
        Start the next operation in the workflow.

        Returns:
            Progress callback for the operation, or None if workflow is complete
        """
        if not self.workflow_id or self.current_operation_index >= len(self.operations):
            return None

        _op_id, op_name, weight = self.operations[self.current_operation_index]

        # Update workflow progress to show we're starting this operation
        self.tracker.update_progress(
            operation_id=self.workflow_id,
            progress=self.current_operation_index / len(self.operations),
            current_step=f"Starting {op_name}...",
            current_step_number=self.current_operation_index + 1,
        )

        # Create sub-progress callback for this operation
        return self.tracker.create_sub_progress_callback(
            operation_id=self.workflow_id, step_weight=weight, step_name=op_name
        )

    def complete_current_operation(self) -> bool:
        """
        Mark current operation as complete and move to next.

        Returns:
            True if there are more operations, False if workflow is complete
        """
        if not self.workflow_id:
            return False

        self.current_operation_index += 1

        if self.current_operation_index >= len(self.operations):
            # Workflow complete
            self.tracker.complete_operation(self.workflow_id)
            return False

        return True

    def fail_workflow(self, error: str) -> None:
        """
        Mark the entire workflow as failed.

        Args:
            error: Error message describing the failure
        """
        if self.workflow_id:
            self.tracker.fail_operation(self.workflow_id, error)


# Global progress tracker instance
_global_tracker: ProgressTracker | None = None


def get_progress_tracker() -> ProgressTracker:
    """Get the global progress tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ProgressTracker()
    return _global_tracker


def create_composite_tracker() -> CompositeProgressTracker:
    """Create a new composite progress tracker."""
    return CompositeProgressTracker(get_progress_tracker())
