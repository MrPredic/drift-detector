"""
Tests for SessionStorage — SQLite-based drift session persistence.
Covers: create_session, add_report, get_reports, update_stats, close_session.
"""
import pytest
import tempfile
import os
from datetime import date, datetime
from drift_detector.core.session_storage import SessionStorage


@pytest.fixture
def storage(tmp_path):
    """Fresh in-memory-style storage per test (tmp file)."""
    db = str(tmp_path / "test_drift.db")
    return SessionStorage(db_path=db)


class TestCreateSession:
    def test_creates_session_with_today(self, storage):
        sid = storage.create_session()
        assert sid.startswith("session_")
        assert date.today().isoformat() in sid

    def test_creates_session_with_specific_date(self, storage):
        d = date(2026, 1, 15)
        sid = storage.create_session(session_date=d)
        assert sid == "session_2026-01-15"

    def test_duplicate_session_is_idempotent(self, storage):
        sid1 = storage.create_session(session_date=date(2026, 3, 1))
        sid2 = storage.create_session(session_date=date(2026, 3, 1))
        assert sid1 == sid2
        sessions = storage.get_sessions()
        assert len([s for s in sessions if s["session_id"] == sid1]) == 1

    def test_multiple_sessions_different_dates(self, storage):
        storage.create_session(session_date=date(2026, 1, 1))
        storage.create_session(session_date=date(2026, 1, 2))
        sessions = storage.get_sessions()
        assert len(sessions) == 2


class TestAddReport:
    def test_add_single_report(self, storage):
        sid = storage.create_session()
        storage.add_report(sid, {
            "step_number": 1,
            "timestamp": datetime.now().isoformat(),
            "combined_drift_score": 0.45,
            "ghost_loss": 0.2,
            "behavior_shift": 0.1,
            "agreement_score": 0.8,
            "stagnation_score": 0.05,
            "is_drifting": False,
        })
        reports = storage.get_session_reports(sid)
        assert len(reports) == 1
        assert reports[0]["combined_drift_score"] == pytest.approx(0.45)

    def test_add_multiple_reports(self, storage):
        sid = storage.create_session()
        for i in range(5):
            storage.add_report(sid, {
                "step_number": i + 1,
                "timestamp": datetime.now().isoformat(),
                "combined_drift_score": 0.1 * i,
                "ghost_loss": 0.0,
                "behavior_shift": 0.0,
                "agreement_score": 1.0,
                "stagnation_score": 0.0,
                "is_drifting": False,
            })
        reports = storage.get_session_reports(sid)
        assert len(reports) == 5

    def test_report_is_drifting_true(self, storage):
        sid = storage.create_session()
        storage.add_report(sid, {
            "step_number": 1,
            "timestamp": datetime.now().isoformat(),
            "combined_drift_score": 0.85,
            "ghost_loss": 0.9,
            "behavior_shift": 0.7,
            "agreement_score": 0.2,
            "stagnation_score": 0.8,
            "is_drifting": True,
        })
        reports = storage.get_session_reports(sid)
        assert reports[0]["is_drifting"] == 1  # SQLite stores bool as int

    def test_report_with_metadata(self, storage):
        sid = storage.create_session()
        storage.add_report(sid, {
            "step_number": 1,
            "timestamp": datetime.now().isoformat(),
            "combined_drift_score": 0.3,
            "is_drifting": False,
            "metadata": {"agent": "test_agent", "model": "groq"},
        })
        reports = storage.get_session_reports(sid)
        assert len(reports) == 1

    def test_reports_isolated_between_sessions(self, storage):
        sid1 = storage.create_session(session_date=date(2026, 1, 1))
        sid2 = storage.create_session(session_date=date(2026, 1, 2))
        storage.add_report(sid1, {"step_number": 1, "timestamp": datetime.now().isoformat(),
                                  "combined_drift_score": 0.3, "is_drifting": False})
        storage.add_report(sid1, {"step_number": 2, "timestamp": datetime.now().isoformat(),
                                  "combined_drift_score": 0.4, "is_drifting": False})
        storage.add_report(sid2, {"step_number": 1, "timestamp": datetime.now().isoformat(),
                                  "combined_drift_score": 0.1, "is_drifting": False})
        assert len(storage.get_session_reports(sid1)) == 2
        assert len(storage.get_session_reports(sid2)) == 1


class TestGetSessions:
    def test_empty_db_returns_empty_list(self, storage):
        assert storage.get_sessions() == []

    def test_sessions_ordered_by_date_desc(self, storage):
        storage.create_session(session_date=date(2026, 1, 1))
        storage.create_session(session_date=date(2026, 3, 1))
        storage.create_session(session_date=date(2026, 2, 1))
        sessions = storage.get_sessions()
        dates = [s["session_date"] for s in sessions]
        assert dates == sorted(dates, reverse=True)

    def test_session_has_required_fields(self, storage):
        storage.create_session()
        sessions = storage.get_sessions()
        required = {"session_id", "session_date", "start_time", "total_reports"}
        assert required.issubset(set(sessions[0].keys()))


class TestUpdateSessionStats:
    def test_stats_update_after_reports(self, storage):
        sid = storage.create_session()
        scores = [0.2, 0.5, 0.8]
        for i, score in enumerate(scores):
            storage.add_report(sid, {
                "step_number": i + 1,
                "timestamp": datetime.now().isoformat(),
                "combined_drift_score": score,
                "is_drifting": score > 0.6,
            })
        storage.update_session_stats(sid)
        sessions = storage.get_sessions()
        s = next(s for s in sessions if s["session_id"] == sid)
        assert s["total_reports"] == 3
        assert s["max_drift"] == pytest.approx(0.8)
        assert s["min_drift"] == pytest.approx(0.2)
        assert s["avg_drift"] == pytest.approx(0.5)

    def test_stats_empty_session(self, storage):
        sid = storage.create_session()
        storage.update_session_stats(sid)
        sessions = storage.get_sessions()
        s = next(s for s in sessions if s["session_id"] == sid)
        assert s["total_reports"] == 0


class TestCloseSession:
    def test_close_sets_end_time(self, storage):
        sid = storage.create_session()
        storage.close_session(sid)
        sessions = storage.get_sessions()
        s = next(s for s in sessions if s["session_id"] == sid)
        assert s["end_time"] is not None

    def test_close_unknown_session_does_not_raise(self, storage):
        # Should not raise
        storage.close_session("session_nonexistent")


class TestGetSessionReports:
    def test_empty_session_returns_empty_list(self, storage):
        sid = storage.create_session()
        assert storage.get_session_reports(sid) == []

    def test_unknown_session_returns_empty_list(self, storage):
        assert storage.get_session_reports("session_unknown") == []

    def test_reports_contain_all_fields(self, storage):
        sid = storage.create_session()
        storage.add_report(sid, {
            "step_number": 1,
            "timestamp": datetime.now().isoformat(),
            "combined_drift_score": 0.42,
            "ghost_loss": 0.1,
            "behavior_shift": 0.2,
            "agreement_score": 0.9,
            "stagnation_score": 0.0,
            "is_drifting": False,
        })
        r = storage.get_session_reports(sid)[0]
        for field in ["session_id", "step_number", "timestamp", "combined_drift_score",
                      "ghost_loss", "behavior_shift", "agreement_score", "stagnation_score", "is_drifting"]:
            assert field in r
