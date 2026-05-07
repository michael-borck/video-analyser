"""Tests for progress tracking system."""

import time
from unittest.mock import MagicMock

from video_analyser.core.progress_tracker import (
    CompositeProgressTracker,
    OperationStatus,
    ProgressTracker,
    ProgressUpdate,
    create_composite_tracker,
    get_progress_tracker,
)


class TestProgressTracker:
    """Test basic progress tracking functionality."""

    def test_progress_tracker_initialization(self):
        """Test progress tracker initialization."""
        tracker = ProgressTracker()

        assert tracker.operations == {}
        assert tracker.callbacks == []
        assert tracker.start_times == {}

    def test_start_operation(self):
        """Test starting a new operation."""
        tracker = ProgressTracker()

        tracker.start_operation(
            operation_id="test_op",
            operation_name="Test Operation",
            total_steps=5,
            details={"type": "test"},
        )

        assert "test_op" in tracker.operations
        assert "test_op" in tracker.start_times

        operation = tracker.operations["test_op"]
        assert operation.operation_id == "test_op"
        assert operation.operation_name == "Test Operation"
        assert operation.status == OperationStatus.RUNNING
        assert operation.progress == 0.0
        assert operation.total_steps == 5
        assert operation.details["type"] == "test"
        assert operation.current_step == "Starting..."

    def test_update_progress(self):
        """Test updating operation progress."""
        tracker = ProgressTracker()

        tracker.start_operation("test_op", "Test Operation")

        # Small delay to ensure elapsed time calculation
        time.sleep(0.01)

        tracker.update_progress(
            operation_id="test_op",
            progress=0.5,
            current_step="Halfway done",
            current_step_number=3,
            details={"processed": 10},
        )

        operation = tracker.operations["test_op"]
        assert operation.progress == 0.5
        assert operation.current_step == "Halfway done"
        assert operation.current_step_number == 3
        assert operation.details["processed"] == 10
        assert operation.elapsed_seconds > 0
        assert operation.estimated_seconds_remaining is not None

    def test_complete_operation(self):
        """Test completing an operation."""
        tracker = ProgressTracker()

        tracker.start_operation("test_op", "Test Operation")
        time.sleep(0.01)

        tracker.complete_operation(
            operation_id="test_op", details={"result": "success"}
        )

        operation = tracker.operations["test_op"]
        assert operation.status == OperationStatus.COMPLETED
        assert operation.progress == 1.0
        assert operation.current_step == "Completed"
        assert operation.estimated_seconds_remaining == 0.0
        assert operation.details["result"] == "success"
        assert "test_op" not in tracker.start_times

    def test_fail_operation(self):
        """Test failing an operation."""
        tracker = ProgressTracker()

        tracker.start_operation("test_op", "Test Operation")

        tracker.fail_operation(
            operation_id="test_op",
            error="Something went wrong",
            details={"error_code": 500},
        )

        operation = tracker.operations["test_op"]
        assert operation.status == OperationStatus.FAILED
        assert operation.current_step == "Failed: Something went wrong"
        assert operation.estimated_seconds_remaining is None
        assert operation.details["error"] == "Something went wrong"
        assert operation.details["error_code"] == 500

    def test_cancel_operation(self):
        """Test cancelling an operation."""
        tracker = ProgressTracker()

        tracker.start_operation("test_op", "Test Operation")

        tracker.cancel_operation("test_op")

        operation = tracker.operations["test_op"]
        assert operation.status == OperationStatus.CANCELLED
        assert operation.current_step == "Cancelled"
        assert operation.estimated_seconds_remaining is None

    def test_unknown_operation_handling(self):
        """Test handling of unknown operation IDs."""
        tracker = ProgressTracker()

        # These should not raise exceptions
        tracker.update_progress("unknown", 0.5)
        tracker.complete_operation("unknown")
        tracker.fail_operation("unknown", "error")
        tracker.cancel_operation("unknown")

        # Should remain empty
        assert tracker.operations == {}

    def test_progress_bounds(self):
        """Test progress value bounds enforcement."""
        tracker = ProgressTracker()

        tracker.start_operation("test_op", "Test Operation")

        # Test negative progress
        tracker.update_progress("test_op", -0.5)
        assert tracker.operations["test_op"].progress == 0.0

        # Test progress > 1.0
        tracker.update_progress("test_op", 1.5)
        assert tracker.operations["test_op"].progress == 1.0

    def test_callback_management(self):
        """Test adding and removing callbacks."""
        tracker = ProgressTracker()
        callback1 = MagicMock()
        callback2 = MagicMock()

        # Add callbacks
        tracker.add_callback(callback1)
        tracker.add_callback(callback2)
        assert len(tracker.callbacks) == 2

        # Remove callback
        tracker.remove_callback(callback1)
        assert len(tracker.callbacks) == 1
        assert callback2 in tracker.callbacks

    def test_callback_notifications(self):
        """Test that callbacks are called on progress updates."""
        tracker = ProgressTracker()
        callback = MagicMock()

        tracker.add_callback(callback)

        # Start operation should trigger callback
        tracker.start_operation("test_op", "Test Operation")
        assert callback.call_count == 1

        # Update progress should trigger callback
        tracker.update_progress("test_op", 0.5)
        assert callback.call_count == 2

        # Complete operation should trigger callback
        tracker.complete_operation("test_op")
        assert callback.call_count == 3

        # Verify callback was called with ProgressUpdate
        for call in callback.call_args_list:
            assert isinstance(call[0][0], ProgressUpdate)

    def test_callback_error_handling(self):
        """Test error handling in callbacks."""
        tracker = ProgressTracker()

        # Add a callback that raises an exception
        def error_callback(progress_update):  # noqa: ARG001
            raise ValueError("Callback error")

        tracker.add_callback(error_callback)

        # Should not raise exception
        tracker.start_operation("test_op", "Test Operation")

        # Operation should still be tracked
        assert "test_op" in tracker.operations

    def test_get_operation_status(self):
        """Test getting operation status."""
        tracker = ProgressTracker()

        # Non-existent operation
        assert tracker.get_operation_status("unknown") is None

        # Existing operation
        tracker.start_operation("test_op", "Test Operation")
        status = tracker.get_operation_status("test_op")
        assert isinstance(status, ProgressUpdate)
        assert status.operation_id == "test_op"

    def test_get_all_operations(self):
        """Test getting all operations."""
        tracker = ProgressTracker()

        # Empty tracker
        assert tracker.get_all_operations() == {}

        # With operations
        tracker.start_operation("op1", "Operation 1")
        tracker.start_operation("op2", "Operation 2")

        all_ops = tracker.get_all_operations()
        assert len(all_ops) == 2
        assert "op1" in all_ops
        assert "op2" in all_ops

        # Should be a copy, not reference
        all_ops["op3"] = "test"
        assert "op3" not in tracker.operations

    def test_clear_completed_operations(self):
        """Test clearing completed operations."""
        tracker = ProgressTracker()

        # Create various operations
        tracker.start_operation("running", "Running Op")
        tracker.start_operation("completed", "Completed Op")
        tracker.start_operation("failed", "Failed Op")
        tracker.start_operation("cancelled", "Cancelled Op")

        # Change statuses
        tracker.complete_operation("completed")
        tracker.fail_operation("failed", "error")
        tracker.cancel_operation("cancelled")

        # Clear completed
        tracker.clear_completed_operations()

        # Only running operation should remain
        assert len(tracker.operations) == 1
        assert "running" in tracker.operations
        assert tracker.operations["running"].status == OperationStatus.RUNNING

    def test_create_sub_progress_callback(self):
        """Test creating sub-progress callbacks."""
        tracker = ProgressTracker()

        tracker.start_operation("parent_op", "Parent Operation", total_steps=2)
        tracker.update_progress("parent_op", 0.0, current_step_number=1)

        # Create sub-progress callback
        sub_callback = tracker.create_sub_progress_callback(
            operation_id="parent_op", step_weight=0.5, step_name="Sub-step 1"
        )

        # Use sub-callback
        sub_callback(0.5)  # 50% of step 1

        # Should update parent progress
        operation = tracker.operations["parent_op"]
        expected_progress = 0.5 * 0.5 / 2  # 50% of 50% weight divided by 2 steps
        assert abs(operation.progress - expected_progress) < 0.01
        assert operation.current_step == "Sub-step 1"

    def test_sub_progress_callback_unknown_operation(self):
        """Test sub-progress callback with unknown operation."""
        tracker = ProgressTracker()

        # Create callback for non-existent operation
        sub_callback = tracker.create_sub_progress_callback("unknown", 0.5)

        # Should not raise exception
        sub_callback(0.5)


class TestCompositeProgressTracker:
    """Test composite progress tracking for workflows."""

    def test_composite_tracker_initialization(self):
        """Test composite tracker initialization."""
        base_tracker = ProgressTracker()
        composite = CompositeProgressTracker(base_tracker)

        assert composite.tracker is base_tracker
        assert composite.workflow_id is None
        assert composite.operations == []
        assert composite.current_operation_index == 0

    def test_start_workflow(self):
        """Test starting a workflow."""
        base_tracker = ProgressTracker()
        composite = CompositeProgressTracker(base_tracker)

        operations = [
            ("validate", "Validate input", 0.2),
            ("process", "Process data", 0.6),
            ("output", "Generate output", 0.2),
        ]

        composite.start_workflow("workflow_1", "Test Workflow", operations)

        assert composite.workflow_id == "workflow_1"
        assert composite.operations == operations
        assert composite.current_operation_index == 0

        # Should have started operation in base tracker
        assert "workflow_1" in base_tracker.operations
        workflow_op = base_tracker.operations["workflow_1"]
        assert workflow_op.operation_name == "Test Workflow"
        assert workflow_op.total_steps == 3

    def test_start_next_operation(self):
        """Test starting next operation in workflow."""
        base_tracker = ProgressTracker()
        composite = CompositeProgressTracker(base_tracker)
        callback = MagicMock()
        base_tracker.add_callback(callback)

        operations = [
            ("validate", "Validate input", 0.3),
            ("process", "Process data", 0.7),
        ]

        composite.start_workflow("workflow_1", "Test Workflow", operations)

        # Start first operation
        progress_callback = composite.start_next_operation()
        assert progress_callback is not None
        assert callable(progress_callback)

        # Should update workflow progress
        progress_callback(0.5)  # 50% of validate step

        workflow_op = base_tracker.operations["workflow_1"]
        assert "Validate input" in workflow_op.current_step

    def test_complete_current_operation(self):
        """Test completing current operation."""
        base_tracker = ProgressTracker()
        composite = CompositeProgressTracker(base_tracker)

        operations = [
            ("validate", "Validate input", 0.5),
            ("process", "Process data", 0.5),
        ]

        composite.start_workflow("workflow_1", "Test Workflow", operations)

        # Complete first operation
        has_more = composite.complete_current_operation()
        assert has_more is True
        assert composite.current_operation_index == 1

        # Complete second operation
        has_more = composite.complete_current_operation()
        assert has_more is False

        # Workflow should be complete
        workflow_op = base_tracker.operations["workflow_1"]
        assert workflow_op.status == OperationStatus.COMPLETED

    def test_fail_workflow(self):
        """Test failing a workflow."""
        base_tracker = ProgressTracker()
        composite = CompositeProgressTracker(base_tracker)

        operations = [("validate", "Validate input", 1.0)]

        composite.start_workflow("workflow_1", "Test Workflow", operations)

        composite.fail_workflow("Validation failed")

        workflow_op = base_tracker.operations["workflow_1"]
        assert workflow_op.status == OperationStatus.FAILED
        assert "Validation failed" in workflow_op.current_step

    def test_workflow_weight_validation(self):
        """Test workflow weight validation."""
        base_tracker = ProgressTracker()
        composite = CompositeProgressTracker(base_tracker)

        # Weights that don't sum to 1.0
        operations = [
            ("validate", "Validate input", 0.3),
            ("process", "Process data", 0.5),  # Sum = 0.8
        ]

        # Should log warning but not fail
        composite.start_workflow("workflow_1", "Test Workflow", operations)

        assert composite.workflow_id == "workflow_1"

    def test_start_next_operation_no_workflow(self):
        """Test starting next operation without workflow."""
        base_tracker = ProgressTracker()
        composite = CompositeProgressTracker(base_tracker)

        # No workflow started
        callback = composite.start_next_operation()
        assert callback is None

    def test_start_next_operation_workflow_complete(self):
        """Test starting next operation when workflow is complete."""
        base_tracker = ProgressTracker()
        composite = CompositeProgressTracker(base_tracker)

        operations = [("validate", "Validate input", 1.0)]

        composite.start_workflow("workflow_1", "Test Workflow", operations)

        # Complete the only operation
        composite.complete_current_operation()

        # Should return None when workflow is complete
        callback = composite.start_next_operation()
        assert callback is None


class TestProgressUpdate:
    """Test ProgressUpdate dataclass."""

    def test_progress_update_creation(self):
        """Test creating a ProgressUpdate."""
        update = ProgressUpdate(
            operation_id="test_op",
            operation_name="Test Operation",
            status=OperationStatus.RUNNING,
            progress=0.5,
            current_step="Processing",
            total_steps=10,
            current_step_number=5,
            estimated_seconds_remaining=30.0,
            elapsed_seconds=15.0,
            details={"processed": 50},
        )

        assert update.operation_id == "test_op"
        assert update.operation_name == "Test Operation"
        assert update.status == OperationStatus.RUNNING
        assert update.progress == 0.5
        assert update.current_step == "Processing"
        assert update.total_steps == 10
        assert update.current_step_number == 5
        assert update.estimated_seconds_remaining == 30.0
        assert update.elapsed_seconds == 15.0
        assert update.details["processed"] == 50

    def test_progress_update_defaults(self):
        """Test ProgressUpdate with default values."""
        update = ProgressUpdate(
            operation_id="test_op",
            operation_name="Test Operation",
            status=OperationStatus.PENDING,
            progress=0.0,
            current_step="Starting",
        )

        assert update.total_steps == 1
        assert update.current_step_number == 1
        assert update.estimated_seconds_remaining is None
        assert update.elapsed_seconds == 0.0
        assert update.details == {}


class TestOperationStatus:
    """Test OperationStatus enum."""

    def test_operation_status_values(self):
        """Test all operation status values."""
        assert OperationStatus.PENDING.value == "pending"
        assert OperationStatus.RUNNING.value == "running"
        assert OperationStatus.COMPLETED.value == "completed"
        assert OperationStatus.FAILED.value == "failed"
        assert OperationStatus.CANCELLED.value == "cancelled"


class TestGlobalProgressTracker:
    """Test global progress tracker functionality."""

    def test_get_progress_tracker_singleton(self):
        """Test global progress tracker singleton behavior."""
        tracker1 = get_progress_tracker()
        tracker2 = get_progress_tracker()

        assert tracker1 is tracker2
        assert isinstance(tracker1, ProgressTracker)

    def test_create_composite_tracker(self):
        """Test creating composite tracker."""
        composite = create_composite_tracker()

        assert isinstance(composite, CompositeProgressTracker)
        assert isinstance(composite.tracker, ProgressTracker)


class TestProgressTrackerIntegration:
    """Test integration scenarios with progress tracking."""

    def test_multi_operation_workflow(self):
        """Test a complete multi-operation workflow."""
        tracker = ProgressTracker()
        callback = MagicMock()
        tracker.add_callback(callback)

        composite = CompositeProgressTracker(tracker)

        operations = [
            ("validate", "Validate input", 0.1),
            ("extract", "Extract data", 0.3),
            ("process", "Process data", 0.4),
            ("output", "Generate output", 0.2),
        ]

        # Start workflow
        composite.start_workflow("multi_op", "Multi-Operation Workflow", operations)

        # Process each operation
        for _i in range(len(operations)):
            progress_callback = composite.start_next_operation()
            assert progress_callback is not None

            # Simulate progress
            progress_callback(0.5)  # 50% of current operation
            progress_callback(1.0)  # Complete current operation

            composite.complete_current_operation()

        # Verify workflow completion
        workflow_op = tracker.operations["multi_op"]
        assert workflow_op.status == OperationStatus.COMPLETED
        assert workflow_op.progress == 1.0

        # Verify callbacks were called
        assert callback.call_count > 0

    def test_workflow_with_failure(self):
        """Test workflow handling with operation failure."""
        tracker = ProgressTracker()
        composite = CompositeProgressTracker(tracker)

        operations = [
            ("validate", "Validate input", 0.5),
            ("process", "Process data", 0.5),
        ]

        composite.start_workflow("failing_workflow", "Failing Workflow", operations)

        # Start first operation
        progress_callback = composite.start_next_operation()
        progress_callback(0.5)

        # Fail the workflow
        composite.fail_workflow("Process failed")

        workflow_op = tracker.operations["failing_workflow"]
        assert workflow_op.status == OperationStatus.FAILED
        assert "Process failed" in workflow_op.current_step

    def test_nested_progress_tracking(self):
        """Test nested progress tracking scenarios."""
        tracker = ProgressTracker()

        # Start parent operation
        tracker.start_operation("parent", "Parent Operation", total_steps=3)

        # Create sub-progress callback for step 1
        sub_callback1 = tracker.create_sub_progress_callback(
            "parent", step_weight=0.33, step_name="Step 1"
        )

        # Update parent to step 1
        tracker.update_progress("parent", progress=0.0, current_step_number=1)

        # Use sub-callback
        sub_callback1(0.5)  # 50% of step 1

        # Move to step 2
        tracker.update_progress("parent", progress=0.33, current_step_number=2)
        sub_callback2 = tracker.create_sub_progress_callback(
            "parent", step_weight=0.33, step_name="Step 2"
        )
        sub_callback2(1.0)  # Complete step 2

        # Move to step 3
        tracker.update_progress("parent", progress=0.66, current_step_number=3)
        sub_callback3 = tracker.create_sub_progress_callback(
            "parent", step_weight=0.34, step_name="Step 3"
        )
        sub_callback3(1.0)  # Complete step 3

        # Complete parent
        tracker.complete_operation("parent")

        parent_op = tracker.operations["parent"]
        assert parent_op.status == OperationStatus.COMPLETED
        assert parent_op.progress == 1.0
