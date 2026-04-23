#!/usr/bin/env python3
"""
CrewAI Integration for DriftDetectorAgent (Apr 13, 2026)
Event-based drift monitoring for CrewAI tasks + agents

Solves CrewAI Issues:
  #5155 - Session-Boundary Drift Detection
  #4682 - Agent Loop Detection Middleware
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass

from drift_detector.core.drift_detector_agent import DriftDetectorAgent, AgentConfig


@dataclass
class TaskSnapshot:
    """Snapshot of a CrewAI task execution"""
    task_id: str
    agent_id: str
    status: str  # started, completed, failed
    output: str
    tools_used: list


class DriftDetectorEventListener:
    """
    Listens to CrewAI task events and measures drift.

    Integration point: crew.add_event_listener(DriftDetectorEventListener())

    Tracks:
      - TaskStartedEvent → takes snapshot (before)
      - TaskCompletedEvent → takes snapshot (after) → measures drift
      - Tool calls → records for loop detection
    """

    def __init__(self, detector: Optional[DriftDetectorAgent] = None, task_type: str = "general", **kwargs):
        """
        Initialize event listener with drift detector.

        Args:
            detector: Optional DriftDetectorAgent instance
            task_type: Task type for adaptive thresholds ("coding", "research", "trading", "general")
        """
        # If detector not provided, create one with appropriate config
        if detector is None:
            config = AgentConfig(
                agent_id="crewai_event_listener",
                role="monitor",
                task_type=task_type,
                drift_threshold=0.6,  # Use same thresholds as main detector
                signal_threshold=0.7
            )
            self.detector = DriftDetectorAgent(config)
        else:
            self.detector = detector
        self.task_snapshots: Dict[str, Any] = {}
        self.active_tasks: Dict[str, Dict] = {}

    def on_event(self, event: Any) -> None:
        """
        Handle CrewAI events.

        Expected event types:
          - TaskStartedEvent
          - TaskCompletedEvent
          - TaskFailedEvent
          - ToolCallEvent (optional)
        """
        event_type = type(event).__name__

        if event_type == "TaskStartedEvent":
            self._handle_task_started(event)
        elif event_type == "TaskCompletedEvent":
            self._handle_task_completed(event)
        elif event_type == "TaskFailedEvent":
            self._handle_task_failed(event)
        elif event_type == "ToolCallEvent":
            self._handle_tool_call(event)

    def _handle_task_started(self, event: Any) -> None:
        """Task started - take snapshot (before state)"""
        try:
            task_id = getattr(event, 'task_id', 'unknown')
            agent_id = getattr(event, 'agent_id', 'unknown')

            # Store task info
            self.active_tasks[task_id] = {
                'agent_id': agent_id,
                'tools_used': []
            }

            # Take "before" snapshot
            self.task_snapshots[f"{task_id}_before"] = self.detector.snapshot(
                agent_id=f"task_{task_id}_before",
                response_text="[Task starting]",
                tool_calls=[]
            )
        except Exception as e:
            print(f"Error in task_started: {e}")

    def _handle_task_completed(self, event: Any) -> None:
        """Task completed - take snapshot (after state) and measure drift"""
        try:
            task_id = getattr(event, 'task_id', 'unknown')
            output = getattr(event, 'output', '')

            if task_id not in self.active_tasks:
                return  # Task not tracked (started event missed)

            agent_id = self.active_tasks[task_id].get('agent_id', 'unknown')
            tools_used = self.active_tasks[task_id].get('tools_used', [])

            # Take "after" snapshot
            snap_after = self.detector.snapshot(
                agent_id=f"task_{task_id}_after",
                response_text=output,
                tool_calls=tools_used
            )

            # Get "before" snapshot
            snap_before_key = f"{task_id}_before"
            if snap_before_key in self.task_snapshots:
                snap_before = self.task_snapshots[snap_before_key]

                # Measure drift
                report = self.detector.measure_drift(snap_before, snap_after)

                # Log result
                status = "✓" if not report.is_drifting else "⚠️"
                print(f"{status} Task {task_id}: drift={report.combined_drift_score:.3f}")

            # Cleanup
            del self.active_tasks[task_id]
            if snap_before_key in self.task_snapshots:
                del self.task_snapshots[snap_before_key]

        except Exception as e:
            print(f"Error in task_completed: {e}")

    def _handle_task_failed(self, event: Any) -> None:
        """Task failed - log error snapshot"""
        try:
            task_id = getattr(event, 'task_id', 'unknown')
            error = getattr(event, 'error', 'Unknown error')

            # Clean up active task
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]

            print(f"✗ Task {task_id} failed: {error}")
        except Exception as e:
            print(f"Error in task_failed: {e}")

    def _handle_tool_call(self, event: Any) -> None:
        """Tool call - track for loop detection"""
        try:
            task_id = getattr(event, 'task_id', 'unknown')
            tool_name = getattr(event, 'tool_name', 'unknown')

            if task_id in self.active_tasks:
                self.active_tasks[task_id]['tools_used'].append(tool_name)
        except Exception as e:
            print(f"Error in tool_call: {e}")

    def get_drift_report(self) -> Dict:
        """Get comprehensive drift report"""
        return {
            "total_drifts": sum(1 for r in self.detector.drift_history if r.is_drifting),
            "history": [
                {
                    "timestamp": r.timestamp,
                    "combined_drift": r.combined_drift_score,
                    "ghost_loss": r.ghost_loss,
                    "behavior_shift": r.behavior_shift,
                    "agreement": r.agreement_score,
                    "stagnation": r.stagnation_score,
                    "is_drifting": r.is_drifting,
                    "loop_detected": r.loop_report.is_looping if r.loop_report else False,
                }
                for r in self.detector.drift_history
            ]
        }


class DriftMonitoringCrewMixin:
    """
    Mixin for CrewAI Crew to add drift monitoring.

    Usage:
        class MyCrewWithDrift(Crew, DriftMonitoringCrewMixin):
            pass

        crew = MyCrewWithDrift()
        crew.drift_listener = DriftDetectorEventListener()
        crew.add_event_listener(crew.drift_listener)

        result = crew.kickoff()
        report = crew.drift_listener.get_drift_report()

    Alternative (simpler):
        crew = MyCrewWithDrift()
        listener = DriftDetectorEventListener()
        crew.add_event_listener(listener)
        result = crew.kickoff()
        print(listener.get_drift_report())
    """

    def __init__(self, *args, **kwargs):
        """Initialize crew with drift monitoring ready"""
        super().__init__(*args, **kwargs)
        self.drift_listener: Optional[DriftDetectorEventListener] = None

    def enable_drift_monitoring(self, task_type: str = "general") -> DriftDetectorEventListener:
        """
        Enable drift monitoring and return listener.

        Args:
            task_type: Task type for adaptive thresholds ("coding", "research", "trading", "general")
        """
        self.drift_listener = DriftDetectorEventListener(task_type=task_type)
        self.add_event_listener(self.drift_listener)
        return self.drift_listener

    def get_drift_report(self) -> Optional[Dict]:
        """Get drift report if monitoring enabled"""
        if self.drift_listener:
            return self.drift_listener.get_drift_report()
        return None


__all__ = ["DriftDetectorEventListener", "DriftMonitoringCrewMixin", "TaskSnapshot"]
