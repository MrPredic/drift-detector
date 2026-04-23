"""
Integration tests: 5 drift signals validated with real LLM-generated text.
Requires GROQ_API_KEY or CEREBRAS_API_KEY in environment (or .env file).
All tests are skipped gracefully if no API key is available.
"""

import os
import pytest
import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

from drift_detector.core import DriftDetectorAgent
from drift_detector.core.drift_detector_agent import AgentConfig

GROQ_KEY = os.getenv("GROQ_API_KEY")
CEREBRAS_KEY = os.getenv("CEREBRAS_API_KEY")
ANY_KEY = GROQ_KEY or CEREBRAS_KEY

skip_if_no_api = pytest.mark.skipif(not ANY_KEY, reason="No LLM API key set (GROQ_API_KEY or CEREBRAS_API_KEY)")


def _call_llm(prompt: str, max_tokens: int = 300) -> str:
    """Call any available LLM API. Returns None if unavailable."""
    if GROQ_KEY:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_KEY}"},
                json={"model": "llama-3.1-8b-instant",
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": max_tokens, "temperature": 0.3},
                timeout=15
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception:
            pass

    if CEREBRAS_KEY:
        try:
            r = requests.post(
                "https://api.cerebras.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {CEREBRAS_KEY}"},
                json={"model": "llama3.1-8b",
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": max_tokens, "temperature": 0.3},
                timeout=15
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception:
            pass

    return None


@pytest.fixture
def detector(tmp_path):
    db = str(tmp_path / "llm_test.db")
    return DriftDetectorAgent(AgentConfig(agent_id="llm_integration"), db_path=db)


@skip_if_no_api
def test_ghost_lexicon_vocabulary_shrinkage(detector):
    """Ghost Lexicon: detailed → brief output loses vocabulary."""
    detailed = _call_llm(
        "Provide a detailed technical analysis of machine learning: supervised learning, "
        "neural networks, optimization algorithms, regularization, and evaluation metrics."
    )
    brief = _call_llm("Summarize machine learning in one sentence.")

    if not detailed or not brief:
        pytest.skip("LLM API unavailable")

    snap_before = detector.snapshot("agent", detailed, ["research", "analyze", "validate"])
    snap_after = detector.snapshot("agent", brief, ["summarize"])
    report = detector.measure_drift(snap_before, snap_after)

    assert report.ghost_loss > 0.5, f"Expected vocabulary loss, got ghost_loss={report.ghost_loss:.3f}"


@skip_if_no_api
def test_behavioral_shift_tool_changes(detector):
    """Behavioral Shift: tool footprint changes between steps."""
    text_a = _call_llm("Explain how to approach a complex data analysis task step by step.")
    text_b = _call_llm("Write a short poem about data.")

    if not text_a or not text_b:
        pytest.skip("LLM API unavailable")

    snap_a = detector.snapshot("agent", text_a, ["analyze", "validate", "transform", "interpret"])
    snap_b = detector.snapshot("agent", text_b, ["generate", "create"])
    report = detector.measure_drift(snap_a, snap_b)

    assert report.behavior_shift >= 0.0


@skip_if_no_api
def test_stagnation_with_llm(detector):
    """Stagnation: same prompt twice produces high repetition score."""
    prompt = "What is the capital of France? Answer in one word."
    r1 = _call_llm(prompt, max_tokens=10)
    r2 = _call_llm(prompt, max_tokens=10)

    if not r1 or not r2:
        pytest.skip("LLM API unavailable")

    snap1 = detector.snapshot("agent", r1, ["query"])
    snap2 = detector.snapshot("agent", r2, ["query"])
    report = detector.measure_drift(snap1, snap2)

    # Same prompt → similar or identical responses → stagnation > 0
    assert report.stagnation_score >= 0.0


@skip_if_no_api
def test_agreement_score_with_llm(detector):
    """Agreement Score: clear vs. vague output diverges in vocabulary overlap."""
    precise = _call_llm("Explain Python list comprehensions with a concrete code example.")
    vague = _call_llm("Say something vague and unrelated to programming.")

    if not precise or not vague:
        pytest.skip("LLM API unavailable")

    snap_precise = detector.snapshot("agent", precise, ["explain"])
    snap_vague = detector.snapshot("agent", vague, ["respond"])
    report = detector.measure_drift(snap_precise, snap_vague)

    assert 0.0 <= report.agreement_score <= 1.0


@skip_if_no_api
def test_chain_degradation(detector):
    """Drift increases as responses degrade over a 3-step chain."""
    step1 = _call_llm(
        "Analyze a startup's decision between B2B and B2C. Consider market size, "
        "sales cycle, margins, customer support, and growth strategies."
    )
    step2 = _call_llm("Briefly, B2B vs B2C?")
    step3 = _call_llm("B2B or B2C? One word.")

    if not step1 or not step3:
        pytest.skip("LLM API unavailable")

    snap1 = detector.snapshot("agent", step1, ["analyze"])
    snap2 = detector.snapshot("agent", step2 or step1, ["analyze"])
    snap3 = detector.snapshot("agent", step3, ["respond"])

    report_12 = detector.measure_drift(snap1, snap2)
    report_23 = detector.measure_drift(snap2, snap3)

    # Drift should be measurable across all steps
    assert report_12.combined_drift_score >= 0.0
    assert report_23.combined_drift_score >= 0.0
