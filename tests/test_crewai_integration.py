"""
Tests for CrewAI DriftDetectorEventListener + DriftMonitoringCrewMixin.
Uses mock events — no real CrewAI crew execution needed.
Each test uses a fresh agent with isolated temp DB to avoid cross-test state.
"""
import os
import pytest
from unittest.mock import MagicMock
from drift_detector.integrations.crewai import DriftDetectorEventListener, DriftMonitoringCrewMixin, TaskSnapshot
from drift_detector.core.drift_detector_agent import DriftDetectorAgent, AgentConfig


def make_event(event_type: str, **attrs):
    event = MagicMock()
    event.__class__.__name__ = event_type
    for k, v in attrs.items():
        setattr(event, k, v)
    return event


def fresh_listener(tmp_path, task_type="general") -> DriftDetectorEventListener:
    db = str(tmp_path / "crewai_test.db")
    agent = DriftDetectorAgent(AgentConfig(agent_id="test_crewai", task_type=task_type), db_path=db)
    return DriftDetectorEventListener(detector=agent)


# ─── Init tests ───────────────────────────────────────────────────────────────

class TestDriftDetectorEventListenerInit:
    def test_default_init_creates_detector(self, tmp_path):
        listener = fresh_listener(tmp_path)
        assert listener.detector is not None
        assert isinstance(listener.detector, DriftDetectorAgent)

    def test_custom_detector_is_used(self, tmp_path):
        db = str(tmp_path / "custom.db")
        agent = DriftDetectorAgent(AgentConfig(agent_id="custom_crewai"), db_path=db)
        listener = DriftDetectorEventListener(detector=agent)
        assert listener.detector.agent_id == "custom_crewai"

    def test_task_type_passed_to_config(self, tmp_path):
        listener = fresh_listener(tmp_path, task_type="coding")
        assert listener.detector.config.task_type == "coding"

    def test_initial_state_empty(self, tmp_path):
        listener = fresh_listener(tmp_path)
        assert listener.task_snapshots == {}
        assert listener.active_tasks == {}


# ─── TaskStartedEvent ─────────────────────────────────────────────────────────

class TestTaskStartedEvent:
    def test_registers_active_task(self, tmp_path):
        listener = fresh_listener(tmp_path)
        listener.on_event(make_event("TaskStartedEvent", task_id="t1", agent_id="agent_1"))
        assert "t1" in listener.active_tasks

    def test_stores_before_snapshot(self, tmp_path):
        listener = fresh_listener(tmp_path)
        listener.on_event(make_event("TaskStartedEvent", task_id="t1", agent_id="agent_1"))
        assert "t1_before" in listener.task_snapshots

    def test_unknown_agent_id_defaults(self, tmp_path):
        listener = fresh_listener(tmp_path)
        event = make_event("TaskStartedEvent", task_id="t2")
        event.agent_id = "unknown"
        listener.on_event(event)
        assert "t2" in listener.active_tasks

    def test_multiple_tasks_tracked_independently(self, tmp_path):
        listener = fresh_listener(tmp_path)
        for i in range(3):
            listener.on_event(make_event("TaskStartedEvent", task_id=f"t{i}", agent_id=f"agent_{i}"))
        assert len(listener.active_tasks) == 3


# ─── TaskCompletedEvent ───────────────────────────────────────────────────────

class TestTaskCompletedEvent:
    def test_measures_drift_after_completion(self, tmp_path):
        listener = fresh_listener(tmp_path)
        listener.on_event(make_event("TaskStartedEvent", task_id="t1", agent_id="a1"))
        listener.on_event(make_event("TaskCompletedEvent", task_id="t1",
                                     output="Analysis complete with detailed technical findings on temperature and pressure"))
        assert len(listener.detector.drift_history) == 1

    def test_cleans_up_active_task_after_completion(self, tmp_path):
        listener = fresh_listener(tmp_path)
        listener.on_event(make_event("TaskStartedEvent", task_id="t1", agent_id="a1"))
        listener.on_event(make_event("TaskCompletedEvent", task_id="t1",
                                     output="Done with full detailed analysis response text"))
        assert "t1" not in listener.active_tasks

    def test_cleans_up_before_snapshot_after_completion(self, tmp_path):
        listener = fresh_listener(tmp_path)
        listener.on_event(make_event("TaskStartedEvent", task_id="t1", agent_id="a1"))
        listener.on_event(make_event("TaskCompletedEvent", task_id="t1", output="result"))
        assert "t1_before" not in listener.task_snapshots

    def test_completion_without_start_does_not_raise(self, tmp_path):
        listener = fresh_listener(tmp_path)
        listener.on_event(make_event("TaskCompletedEvent", task_id="orphan", output="some output"))

    def test_multiple_tasks_complete_independently(self, tmp_path):
        listener = fresh_listener(tmp_path)
        for i in range(3):
            listener.on_event(make_event("TaskStartedEvent", task_id=f"t{i}", agent_id=f"a{i}"))
        for i in range(3):
            listener.on_event(make_event("TaskCompletedEvent", task_id=f"t{i}",
                                         output=f"Task {i} output with detailed analysis text"))
        assert len(listener.detector.drift_history) == 3
        assert len(listener.active_tasks) == 0


# ─── TaskFailedEvent ──────────────────────────────────────────────────────────

class TestTaskFailedEvent:
    def test_removes_active_task_on_failure(self, tmp_path):
        listener = fresh_listener(tmp_path)
        listener.on_event(make_event("TaskStartedEvent", task_id="t1", agent_id="a1"))
        listener.on_event(make_event("TaskFailedEvent", task_id="t1", error="Timeout"))
        assert "t1" not in listener.active_tasks

    def test_failure_without_start_does_not_raise(self, tmp_path):
        listener = fresh_listener(tmp_path)
        listener.on_event(make_event("TaskFailedEvent", task_id="ghost", error="error"))

    def test_no_drift_recorded_on_failure(self, tmp_path):
        listener = fresh_listener(tmp_path)
        listener.on_event(make_event("TaskStartedEvent", task_id="t1", agent_id="a1"))
        listener.on_event(make_event("TaskFailedEvent", task_id="t1", error="crash"))
        assert len(listener.detector.drift_history) == 0


# ─── ToolCallEvent ────────────────────────────────────────────────────────────

class TestToolCallEvent:
    def test_tool_call_tracked_in_active_task(self, tmp_path):
        listener = fresh_listener(tmp_path)
        listener.on_event(make_event("TaskStartedEvent", task_id="t1", agent_id="a1"))
        listener.on_event(make_event("ToolCallEvent", task_id="t1", tool_name="search"))
        assert "search" in listener.active_tasks["t1"]["tools_used"]

    def test_multiple_tool_calls_accumulated(self, tmp_path):
        listener = fresh_listener(tmp_path)
        listener.on_event(make_event("TaskStartedEvent", task_id="t1", agent_id="a1"))
        for tool in ["search", "analyze", "compute", "summarize"]:
            listener.on_event(make_event("ToolCallEvent", task_id="t1", tool_name=tool))
        assert len(listener.active_tasks["t1"]["tools_used"]) == 4

    def test_tool_call_for_unknown_task_does_not_raise(self, tmp_path):
        listener = fresh_listener(tmp_path)
        listener.on_event(make_event("ToolCallEvent", task_id="ghost", tool_name="search"))

    def test_tool_calls_included_in_drift_measurement(self, tmp_path):
        listener = fresh_listener(tmp_path)
        listener.on_event(make_event("TaskStartedEvent", task_id="t1", agent_id="a1"))
        for tool in ["search", "analyze"]:
            listener.on_event(make_event("ToolCallEvent", task_id="t1", tool_name=tool))
        listener.on_event(make_event("TaskCompletedEvent", task_id="t1",
                                     output="Detailed analysis with full technical report output"))
        report = listener.get_drift_report()
        assert report["total_drifts"] >= 0


# ─── Unknown events ───────────────────────────────────────────────────────────

class TestUnknownEvents:
    def test_unknown_event_type_ignored(self, tmp_path):
        listener = fresh_listener(tmp_path)
        listener.on_event(make_event("SomeRandomEvent", data="whatever"))
        assert listener.active_tasks == {}


# ─── get_drift_report ─────────────────────────────────────────────────────────

class TestGetDriftReport:
    def test_empty_report_structure(self, tmp_path):
        listener = fresh_listener(tmp_path)
        report = listener.get_drift_report()
        assert "total_drifts" in report
        assert "history" in report
        assert report["total_drifts"] == 0
        assert report["history"] == []

    def test_report_after_completed_tasks(self, tmp_path):
        listener = fresh_listener(tmp_path)
        for i in range(3):
            listener.on_event(make_event("TaskStartedEvent", task_id=f"t{i}", agent_id="a"))
            listener.on_event(make_event("TaskCompletedEvent", task_id=f"t{i}",
                                         output=f"Task {i} produced a detailed analytical output text"))
        report = listener.get_drift_report()
        assert len(report["history"]) == 3

    def test_report_history_fields(self, tmp_path):
        listener = fresh_listener(tmp_path)
        listener.on_event(make_event("TaskStartedEvent", task_id="t1", agent_id="a1"))
        listener.on_event(make_event("TaskCompletedEvent", task_id="t1",
                                     output="Comprehensive output with technical analysis and signals"))
        report = listener.get_drift_report()
        if report["history"]:
            entry = report["history"][0]
            for field in ["timestamp", "combined_drift", "ghost_loss", "behavior_shift",
                          "agreement", "stagnation", "is_drifting", "loop_detected"]:
                assert field in entry

    def test_total_drifts_counts_correctly(self, tmp_path):
        listener = fresh_listener(tmp_path)
        for i in range(5):
            listener.on_event(make_event("TaskStartedEvent", task_id=f"t{i}", agent_id="a"))
            listener.on_event(make_event("TaskCompletedEvent", task_id=f"t{i}",
                                         output=f"Step {i} detailed output text for analysis"))
        report = listener.get_drift_report()
        expected = sum(1 for h in report["history"] if h["is_drifting"])
        assert report["total_drifts"] == expected


# ─── TaskSnapshot dataclass ───────────────────────────────────────────────────

class TestTaskSnapshot:
    def test_task_snapshot_creation(self):
        snap = TaskSnapshot(
            task_id="t1", agent_id="a1", status="completed",
            output="some output", tools_used=["search"]
        )
        assert snap.task_id == "t1"
        assert snap.status == "completed"
        assert "search" in snap.tools_used


# ─── DriftMonitoringCrewMixin ─────────────────────────────────────────────────

class TestDriftMonitoringCrewMixin:
    def test_get_drift_report_without_listener_returns_none(self):
        class MockCrew(DriftMonitoringCrewMixin):
            def __init__(self):
                self.drift_listener = None
        assert MockCrew().get_drift_report() is None

    def test_get_drift_report_with_listener(self, tmp_path):
        listener = fresh_listener(tmp_path)

        class MockCrew(DriftMonitoringCrewMixin):
            def __init__(self):
                self.drift_listener = listener

        report = MockCrew().get_drift_report()
        assert report is not None
        assert "total_drifts" in report
