"""
Tests for LangChain DriftDetectionCallback.
Unit tests use isolated temp DB. Integration tests hit real Groq API.
"""
import os
import pytest
from dotenv import load_dotenv

load_dotenv()

from drift_detector.integrations.langchain import DriftDetectionCallback
from drift_detector.core.drift_detector_agent import DriftDetectorAgent, AgentConfig


def fresh_callback(tmp_path, verbose=False) -> DriftDetectionCallback:
    """Callback backed by isolated temp DB — no shared history."""
    db = str(tmp_path / "lc_test.db")
    agent = DriftDetectorAgent(AgentConfig(agent_id="lc_test"), db_path=db)
    return DriftDetectionCallback(agent=agent, verbose=verbose)


# ─── Init ─────────────────────────────────────────────────────────────────────

class TestDriftDetectionCallbackInit:
    def test_init_without_agent_creates_default(self):
        cb = DriftDetectionCallback()
        assert isinstance(cb.agent, DriftDetectorAgent)

    def test_init_with_custom_agent(self, tmp_path):
        db = str(tmp_path / "t.db")
        agent = DriftDetectorAgent(AgentConfig(agent_id="test_cb"), db_path=db)
        cb = DriftDetectionCallback(agent=agent)
        assert cb.agent.agent_id == "test_cb"

    def test_verbose_default_true(self):
        cb = DriftDetectionCallback()
        assert cb.verbose is True

    def test_verbose_can_be_disabled(self):
        cb = DriftDetectionCallback(verbose=False)
        assert cb.verbose is False

    def test_prev_snapshot_initially_none(self):
        cb = DriftDetectionCallback()
        assert cb.prev_snapshot is None


# ─── on_chain_end ─────────────────────────────────────────────────────────────

class TestOnChainEnd:
    def test_first_call_stores_snapshot(self, tmp_path):
        cb = fresh_callback(tmp_path)
        cb.on_chain_end({"output": "Hello world this is a detailed response with many tokens"})
        assert cb.prev_snapshot is not None

    def test_second_call_measures_drift(self, tmp_path):
        cb = fresh_callback(tmp_path)
        cb.on_chain_end({"output": "Detailed analysis with technical vocabulary and many domain signals"})
        cb.on_chain_end({"output": "Hi"})
        assert len(cb.agent.drift_history) == 1

    def test_empty_output_handled_gracefully(self, tmp_path):
        cb = fresh_callback(tmp_path)
        cb.on_chain_end({"output": ""})

    def test_missing_output_key_handled(self, tmp_path):
        cb = fresh_callback(tmp_path)
        cb.on_chain_end({})

    def test_tool_calls_captured(self, tmp_path):
        cb = fresh_callback(tmp_path)
        cb.on_chain_end({
            "output": "Analysis complete with detailed findings",
            "tool_calls": ["search", "analyze", "summarize"]
        })
        assert cb.prev_snapshot is not None

    def test_drift_detected_on_large_change(self, tmp_path):
        cb = fresh_callback(tmp_path)
        long_text = " ".join(["technical", "analysis", "deep", "research", "velocity",
                               "temperature", "pressure", "algorithm", "optimization"] * 5)
        cb.on_chain_end({"output": long_text, "tool_calls": ["search", "analyze", "compute"]})
        cb.on_chain_end({"output": "ok", "tool_calls": []})
        assert len(cb.agent.drift_history) > 0

    def test_snapshot_updates_after_each_call(self, tmp_path):
        cb = fresh_callback(tmp_path)
        cb.on_chain_end({"output": "First detailed response with technical terms"})
        snap1 = cb.prev_snapshot
        cb.on_chain_end({"output": "Second different response"})
        snap2 = cb.prev_snapshot
        assert snap1 is not snap2

    def test_on_drift_called_when_drift_detected(self, tmp_path):
        cb = fresh_callback(tmp_path)
        drift_alerts = []
        cb.on_drift = lambda alert, report: drift_alerts.append(alert)
        long_text = " ".join(["analysis", "research", "velocity", "temperature",
                               "pressure", "algorithm"] * 10)
        cb.on_chain_end({"output": long_text, "tool_calls": ["search", "compute", "analyze"]})
        cb.on_chain_end({"output": "ok", "tool_calls": []})
        history = cb.get_history()
        if any(h["is_drifting"] for h in history):
            assert len(drift_alerts) > 0

    def test_multiple_outputs_accumulate_history(self, tmp_path):
        cb = fresh_callback(tmp_path)
        outputs = [
            "First detailed technical response with many vocabulary tokens",
            "Second analysis with different vocabulary and domain terms",
            "Third output has completely different language patterns",
            "Short",
            "Brief again",
        ]
        for out in outputs:
            cb.on_chain_end({"output": out})
        assert len(cb.agent.drift_history) == len(outputs) - 1


# ─── get_stats ────────────────────────────────────────────────────────────────

class TestGetStats:
    def test_get_stats_returns_dict(self, tmp_path):
        cb = fresh_callback(tmp_path)
        assert isinstance(cb.get_stats(), dict)

    def test_get_stats_after_calls(self, tmp_path):
        cb = fresh_callback(tmp_path)
        cb.on_chain_end({"output": "A detailed technical response with many words"})
        cb.on_chain_end({"output": "Brief"})
        assert isinstance(cb.get_stats(), dict)


# ─── get_history ──────────────────────────────────────────────────────────────

class TestGetHistory:
    def test_empty_history(self, tmp_path):
        cb = fresh_callback(tmp_path)
        assert cb.get_history() == []

    def test_history_after_two_calls(self, tmp_path):
        cb = fresh_callback(tmp_path)
        cb.on_chain_end({"output": "First detailed analytical response with many signals"})
        cb.on_chain_end({"output": "Second brief"})
        history = cb.get_history()
        assert len(history) == 1
        assert "combined_drift" in history[0]
        assert "ghost_loss" in history[0]
        assert "is_drifting" in history[0]

    def test_history_fields_are_valid_types(self, tmp_path):
        cb = fresh_callback(tmp_path)
        cb.on_chain_end({"output": "Detailed analysis with technical vocabulary and methodology"})
        cb.on_chain_end({"output": "Short"})
        for entry in cb.get_history():
            assert isinstance(entry["combined_drift"], float)
            assert isinstance(entry["is_drifting"], bool)

    def test_history_grows_with_calls(self, tmp_path):
        cb = fresh_callback(tmp_path)
        texts = [
            "First detailed output with technical content and domain vocabulary",
            "Second output shifts to different vocabulary and patterns",
            "Third brief",
            "Fourth very different analysis approach methodology",
        ]
        for t in texts:
            cb.on_chain_end({"output": t})
        assert len(cb.get_history()) == len(texts) - 1


# ─── get_drift_report ─────────────────────────────────────────────────────────

class TestGetDriftReport:
    def test_report_structure(self, tmp_path):
        cb = fresh_callback(tmp_path)
        report = cb.get_drift_report()
        assert "total_drifts" in report
        assert "history" in report
        assert report["total_drifts"] == 0

    def test_report_counts_drifts_correctly(self, tmp_path):
        cb = fresh_callback(tmp_path)
        long_text = " ".join(["technical", "analysis", "velocity", "algorithm",
                               "temperature", "pressure"] * 10)
        for _ in range(3):
            cb.on_chain_end({"output": long_text, "tool_calls": ["search", "compute"]})
            cb.on_chain_end({"output": "ok", "tool_calls": []})
        report = cb.get_drift_report()
        assert report["total_drifts"] == sum(1 for h in report["history"] if h["is_drifting"])

    def test_report_total_drifts_is_int(self, tmp_path):
        cb = fresh_callback(tmp_path)
        assert isinstance(cb.get_drift_report()["total_drifts"], int)


# ─── Integration Tests (real Groq API) ───────────────────────────────────────

GROQ_KEY = os.getenv("GROQ_API_KEY")
skip_if_no_groq = pytest.mark.skipif(not GROQ_KEY, reason="GROQ_API_KEY not set")


@skip_if_no_groq
class TestWithRealGroqChain:
    def test_direct_on_chain_end_with_real_groq_output(self, tmp_path):
        """Real Groq output → manually passed to on_chain_end."""
        from langchain_groq import ChatGroq

        llm = ChatGroq(api_key=GROQ_KEY, model="llama-3.1-8b-instant", temperature=0)
        prompt = "In two sentences, explain gradient descent optimization."
        response = llm.invoke(prompt)
        output_text = response.content if hasattr(response, "content") else str(response)

        cb = fresh_callback(tmp_path)
        cb.on_chain_end({"output": output_text})
        assert cb.prev_snapshot is not None

    def test_two_real_groq_outputs_measure_drift(self, tmp_path):
        """Two real Groq outputs → drift is measured."""
        from langchain_groq import ChatGroq

        llm = ChatGroq(api_key=GROQ_KEY, model="llama-3.1-8b-instant", temperature=0)
        r1 = llm.invoke("Explain gradient descent with regularization in detail.")
        r2 = llm.invoke("Say: ok")

        cb = fresh_callback(tmp_path)
        cb.on_chain_end({"output": r1.content})
        cb.on_chain_end({"output": r2.content})

        assert len(cb.agent.drift_history) == 1
        report = cb.get_drift_report()
        assert "total_drifts" in report

    def test_multiple_real_groq_outputs_build_history(self, tmp_path):
        """Multiple Groq outputs accumulate drift history."""
        from langchain_groq import ChatGroq

        llm = ChatGroq(api_key=GROQ_KEY, model="llama-3.1-8b-instant", temperature=0)
        prompts = [
            "Explain machine learning in 3 sentences.",
            "Describe neural network architectures briefly.",
            "What is overfitting?",
            "ok",
        ]
        cb = fresh_callback(tmp_path)
        for p in prompts:
            r = llm.invoke(p)
            cb.on_chain_end({"output": r.content})

        assert len(cb.get_history()) == len(prompts) - 1

    def test_groq_detects_domain_shift(self, tmp_path):
        """Drift should be higher when domain shifts radically."""
        from langchain_groq import ChatGroq

        llm = ChatGroq(api_key=GROQ_KEY, model="llama-3.1-8b-instant", temperature=0)
        technical = llm.invoke("Explain gradient descent, backpropagation, regularization, and neural network optimization in detail.").content
        brief = llm.invoke("Say the word: ok").content

        cb = fresh_callback(tmp_path)
        cb.on_chain_end({"output": technical})
        cb.on_chain_end({"output": brief})

        history = cb.get_history()
        assert len(history) == 1
        # Technical → brief should show non-trivial drift
        assert history[0]["combined_drift"] > 0.1
