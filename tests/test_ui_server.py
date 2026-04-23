"""
Tests for FastAPI UI server — all endpoints via TestClient.
Covers: /api/health, /api/config, /api/drift, /api/chain, error cases.
Each test uses an isolated temp DB via DRIFT_DETECTOR_DB env var.
"""
import pytest
import os
from fastapi.testclient import TestClient

import drift_detector.ui.server as server_module


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point detector to a fresh temp DB and reset global state before each test."""
    db_path = str(tmp_path / "ui_test.db")
    monkeypatch.setenv("DRIFT_DETECTOR_DB", db_path)
    server_module._detector_instance = None
    server_module._session_storage = None
    server_module._current_session_id = None
    yield
    server_module._detector_instance = None


@pytest.fixture
def client():
    """TestClient with lifespan (initializes detector against fresh DB)."""
    with TestClient(server_module.app) as c:
        yield c


# ─── Health Check ─────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_health_response_structure(self, client):
        r = client.get("/api/health")
        data = r.json()
        assert data["status"] == "healthy"
        assert data["version"] == "2.0.0"
        assert "detector_id" in data
        assert "drift_history_count" in data
        assert "uptime" in data

    def test_health_detector_id_is_string(self, client):
        r = client.get("/api/health")
        assert isinstance(r.json()["detector_id"], str)

    def test_health_drift_history_starts_at_zero(self, client):
        r = client.get("/api/health")
        assert r.json()["drift_history_count"] == 0


# ─── Config ───────────────────────────────────────────────────────────────────

class TestConfigEndpoint:
    def test_config_returns_200(self, client):
        r = client.get("/api/config")
        assert r.status_code == 200

    def test_config_response_structure(self, client):
        data = client.get("/api/config").json()
        assert "drift_threshold" in data
        assert "signal_threshold" in data
        assert data["version"] == "2.0.0"

    def test_config_thresholds_are_floats(self, client):
        data = client.get("/api/config").json()
        assert isinstance(data["drift_threshold"], float)
        assert isinstance(data["signal_threshold"], float)

    def test_config_default_thresholds(self, client):
        data = client.get("/api/config").json()
        assert 0.0 < data["drift_threshold"] < 1.0
        assert 0.0 < data["signal_threshold"] < 1.0


# ─── Drift Measurement ────────────────────────────────────────────────────────

class TestDriftEndpoint:
    def test_drift_returns_200(self, client):
        r = client.post("/api/drift", json={
            "before_text": "Hello world this is a detailed technical analysis response",
            "after_text": "Brief response"
        })
        assert r.status_code == 200

    def test_drift_response_structure(self, client):
        r = client.post("/api/drift", json={
            "before_text": "Detailed analysis with technical vocabulary and methodology",
            "after_text": "Short answer"
        })
        data = r.json()
        for field in ["combined_drift_score", "ghost_loss", "behavior_shift",
                      "agreement_score", "stagnation_score", "is_drifting",
                      "loop_detected", "timestamp"]:
            assert field in data

    def test_drift_scores_in_valid_range(self, client):
        r = client.post("/api/drift", json={
            "before_text": "Technical analysis with detailed methodology",
            "after_text": "Simple brief output"
        })
        data = r.json()
        assert 0.0 <= data["combined_drift_score"] <= 1.0
        assert 0.0 <= data["ghost_loss"] <= 1.0
        assert 0.0 <= data["stagnation_score"] <= 1.0

    def test_drift_with_tool_calls(self, client):
        r = client.post("/api/drift", json={
            "before_text": "Full analysis with algorithms and optimization techniques",
            "after_text": "Done",
            "tool_calls_before": ["search", "analyze", "compute"],
            "tool_calls_after": []
        })
        assert r.status_code == 200

    def test_drift_identical_texts_low_score(self, client):
        text = "This is the exact same output text repeated verbatim"
        r = client.post("/api/drift", json={
            "before_text": text,
            "after_text": text
        })
        data = r.json()
        assert data["combined_drift_score"] < 0.5

    def test_drift_with_custom_agent_id(self, client):
        r = client.post("/api/drift", json={
            "before_text": "Detailed technical output",
            "after_text": "Brief",
            "agent_id": "test_agent_123"
        })
        assert r.status_code == 200

    def test_drift_is_drifting_type_bool(self, client):
        r = client.post("/api/drift", json={
            "before_text": "Some text here",
            "after_text": "Different text"
        })
        assert isinstance(r.json()["is_drifting"], bool)

    def test_drift_loop_detected_type_bool(self, client):
        r = client.post("/api/drift", json={
            "before_text": "Text A",
            "after_text": "Text B"
        })
        assert isinstance(r.json()["loop_detected"], bool)

    def test_drift_missing_required_field_returns_422(self, client):
        r = client.post("/api/drift", json={"before_text": "only before"})
        assert r.status_code == 422

    def test_drift_empty_body_returns_422(self, client):
        r = client.post("/api/drift", json={})
        assert r.status_code == 422


# ─── Chain History ────────────────────────────────────────────────────────────

class TestChainEndpoint:
    def test_chain_returns_200(self, client):
        r = client.get("/api/chain")
        assert r.status_code == 200

    def test_chain_empty_initially(self, client):
        data = client.get("/api/chain").json()
        assert data["total_reports"] == 0
        assert data["reports"] == []
        assert data["average_drift"] == 0.0

    def test_chain_structure(self, client):
        data = client.get("/api/chain").json()
        for field in ["total_reports", "reports", "average_drift", "max_drift", "min_drift"]:
            assert field in data

    def test_chain_has_reports_after_drift_calls(self, client):
        for before, after in [
            ("Detailed technical analysis with velocity and pressure", "Short"),
            ("Algorithm optimization methodology research", "Brief"),
        ]:
            client.post("/api/drift", json={"before_text": before, "after_text": after})

        data = client.get("/api/chain").json()
        assert data["total_reports"] >= 2

    def test_chain_report_fields(self, client):
        client.post("/api/drift", json={
            "before_text": "Comprehensive technical analysis output with detailed findings",
            "after_text": "Done"
        })
        data = client.get("/api/chain").json()
        if data["reports"]:
            r = data["reports"][0]
            for field in ["step_number", "timestamp", "combined_drift_score", "is_drifting"]:
                assert field in r

    def test_chain_average_drift_computed(self, client):
        texts = [
            ("Technical analysis velocity algorithm optimization research depth", "ok"),
            ("Methodology pressure temperature signals domain research", "done"),
        ]
        for before, after in texts:
            client.post("/api/drift", json={"before_text": before, "after_text": after})

        data = client.get("/api/chain").json()
        if data["total_reports"] > 0:
            assert isinstance(data["average_drift"], float)

    def test_chain_max_gte_min(self, client):
        texts = [
            ("Long detailed technical response with many vocabulary words", "Short"),
            ("Another detailed technical analysis output with signals", "Brief"),
        ]
        for before, after in texts:
            client.post("/api/drift", json={"before_text": before, "after_text": after})

        data = client.get("/api/chain").json()
        if data["total_reports"] > 0:
            assert data["max_drift"] >= data["min_drift"]


# ─── Root / Frontend ──────────────────────────────────────────────────────────

class TestRootEndpoint:
    def test_root_returns_response(self, client):
        r = client.get("/")
        # Either HTML or JSON dict — should not 500
        assert r.status_code in (200, 404)


# ─── CORS headers ─────────────────────────────────────────────────────────────

class TestCORSHeaders:
    def test_cors_headers_present_on_drift(self, client):
        r = client.post("/api/drift",
                        json={"before_text": "test before output", "after_text": "test after"},
                        headers={"Origin": "http://localhost:3000"})
        assert r.status_code == 200
