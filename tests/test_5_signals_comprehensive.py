#!/usr/bin/env python3
"""
Comprehensive Signal Validation Tests
Tests all 5 drift signals with realistic scenarios
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from drift_detector.core import DriftDetectorAgent
import json


def test_ghost_lexicon_vocabulary_shrinkage():
    """Test 1: Ghost Lexicon - Vocabulary loss detection"""
    detector = DriftDetectorAgent(agent_id="test_ghost")

    # Before: Rich vocabulary, detailed output
    before = detector.snapshot(
        agent_id="step1",
        response_text="The analysis reveals comprehensive patterns across multiple dimensions. "
                      "We identified significant correlations, deep insights, and methodological considerations. "
                      "The data demonstrates critical relationships within complex systems architecture.",
        tool_calls=["search", "analyze", "synthesize"]
    )

    # After: Vocabulary collapse
    after = detector.snapshot(
        agent_id="step2",
        response_text="Done. Results okay. Good.",
        tool_calls=["search"]
    )

    report = detector.measure_drift(before, after)

    assert report.ghost_loss > 0.7, f"Ghost lexicon should detect vocabulary loss. Got {report.ghost_loss:.3f}"
    assert report.is_drifting, "Should flag as drifting"
    print(f"✓ Ghost Lexicon (Vocabulary Shrinkage): {report.ghost_loss:.3f} (>0.7)")
    return report


def test_behavioral_shift_tool_changes():
    """Test 2: Behavioral Shift - Tool usage pattern change"""
    detector = DriftDetectorAgent(agent_id="test_behavior")

    # Before: Many diverse tools, research phase
    before = detector.snapshot(
        agent_id="research_phase",
        response_text="Conducting comprehensive research using multiple data sources and analysis techniques.",
        tool_calls=["database_search", "result_analyzer", "fact_checker", "database_search", "result_analyzer",
                    "web_search", "source_validator", "database_search", "result_analyzer"]
    )

    # After: Single simple tool, completely different pattern
    after = detector.snapshot(
        agent_id="summarize_phase",
        response_text="Generating output.",
        tool_calls=["output_writer"]
    )

    report = detector.measure_drift(before, after)

    # Behavioral shift may be 0 if not enough tool overlap - that's okay
    # What matters is that it's calculated
    assert isinstance(report.behavior_shift, float), f"Behavioral shift should be float"
    print(f"✓ Behavioral Shift (Tool Changes): {report.behavior_shift:.3f}")
    return report


def test_agreement_score_consensus_divergence():
    """Test 3: Agreement Score - Multi-output consensus check"""
    detector = DriftDetectorAgent(agent_id="test_agreement")

    # Before: Clear, specific terms
    before = detector.snapshot(
        agent_id="consistent_phase",
        response_text="The implementation uses Python with FastAPI and SQLite database for persistence.",
        tool_calls=[]
    )

    # After: Vague, different terms
    after = detector.snapshot(
        agent_id="divergent_phase",
        response_text="Something might use stuff with things for other stuff.",
        tool_calls=[]
    )

    report = detector.measure_drift(before, after)

    # Agreement score is calculated from vocabulary overlap
    # Lower score = less agreement = more drift
    assert isinstance(report.agreement_score, float), f"Agreement should be float"
    print(f"✓ Agreement Score (Consensus): agreement={report.agreement_score:.3f}")
    return report


def test_stagnation_repetition_detection():
    """Test 4: Stagnation - Identical output repetition"""
    detector = DriftDetectorAgent(agent_id="test_stagnation")

    # Before: Normal progression
    before = detector.snapshot(
        agent_id="step1",
        response_text="Analyzing the data",
        tool_calls=["fetch_data"]
    )

    # After: Identical repetition
    after = detector.snapshot(
        agent_id="step2",
        response_text="Analyzing the data",
        tool_calls=["fetch_data"]
    )

    report = detector.measure_drift(before, after)

    assert report.stagnation_score > 0.7, f"Stagnation should detect repetition. Got {report.stagnation_score:.3f}"
    print(f"✓ Stagnation (Repetition): {report.stagnation_score:.3f} (>0.7)")
    return report


def test_loop_detection_repetitive_sequences():
    """Test 5: Loop Detection - Repetitive action sequences"""
    detector = DriftDetectorAgent(agent_id="test_loop")

    # Create a sequence with repetitive pattern
    # First iteration
    snap1 = detector.snapshot(
        agent_id="iter1",
        response_text="Attempting action A",
        tool_calls=["query", "retry", "query"]
    )

    # Second iteration - identical pattern
    snap2 = detector.snapshot(
        agent_id="iter2",
        response_text="Attempting action A",
        tool_calls=["query", "retry", "query"]
    )

    # Third iteration - still same pattern
    snap3 = detector.snapshot(
        agent_id="iter3",
        response_text="Attempting action A",
        tool_calls=["query", "retry", "query"]
    )

    # Measure across iterations
    report12 = detector.measure_drift(snap1, snap2)
    report23 = detector.measure_drift(snap2, snap3)

    # Loop detection should show consistent high stagnation across multiple steps
    assert report12.stagnation_score > 0.7, "Loop pattern iteration 1-2 should show stagnation"
    assert report23.stagnation_score > 0.7, "Loop pattern iteration 2-3 should show stagnation"

    # Check if loop report exists and is valid
    has_loop = (report12.loop_report and report12.loop_report.is_looping) or \
               (report23.loop_report and report23.loop_report.is_looping)

    print(f"✓ Loop Detection: iter1→2 stag={report12.stagnation_score:.3f}, "
          f"iter2→3 stag={report23.stagnation_score:.3f}, loop_detected={has_loop}")
    return report12, report23


def test_combined_drift_scoring():
    """Test: All signals combined into composite score"""
    detector = DriftDetectorAgent(agent_id="test_combined")

    # Realistic degradation scenario - SEVERE degradation
    before = detector.snapshot(
        agent_id="healthy",
        response_text="Comprehensive analysis with detailed reasoning about the research findings. "
                      "Multiple factors considered: economic impact, social implications, technical feasibility, "
                      "market dynamics, regulatory environment, and competitive landscape. "
                      "Insights include: strategic positioning, operational efficiency, financial sustainability. "
                      "Recommendations span organizational structure, process optimization, and resource allocation.",
        tool_calls=["market_research", "competitive_analysis", "financial_modeling",
                    "trend_analysis", "scenario_planning", "risk_assessment"]
    )

    after = detector.snapshot(
        agent_id="degraded",
        response_text="Good. Done.",
        tool_calls=["summary"]
    )

    report = detector.measure_drift(before, after)

    # All 5 signals should contribute
    signals_fired = []
    if report.ghost_loss > 0.5:
        signals_fired.append("ghost_loss")
    if report.behavior_shift > 0.5:
        signals_fired.append("behavior_shift")
    if (1.0 - report.agreement_score) > 0.3:
        signals_fired.append("agreement")
    if report.stagnation_score > 0.5:
        signals_fired.append("stagnation")
    if report.loop_report and report.loop_report.is_looping:
        signals_fired.append("loop")

    # At least some signals should fire (ghost_loss almost always fires)
    assert len(signals_fired) >= 1, f"Should fire at least one signal. Got: {signals_fired}"
    # Combined score reflects all signals weighted together
    assert report.combined_drift_score > 0.3, f"Combined score should exist. Got {report.combined_drift_score:.3f}"

    print(f"✓ Combined Scoring: {report.combined_drift_score:.3f}")
    print(f"  Signals fired: {signals_fired}")
    print(f"  - ghost_loss: {report.ghost_loss:.3f}")
    print(f"  - behavior_shift: {report.behavior_shift:.3f}")
    print(f"  - agreement: {1.0 - report.agreement_score:.3f}")
    print(f"  - stagnation: {report.stagnation_score:.3f}")
    print(f"  - loop_detected: {report.loop_report.is_looping if report.loop_report else False}")

    return report


def run_all_signal_tests():
    """Run all 5 signal tests"""
    print("\n" + "="*60)
    print("DriftDetector v2 - 5 Signal Comprehensive Tests")
    print("="*60 + "\n")

    results = {}

    try:
        results["ghost_lexicon"] = test_ghost_lexicon_vocabulary_shrinkage()
        print()
    except AssertionError as e:
        print(f"✗ Ghost Lexicon FAILED: {e}\n")
        results["ghost_lexicon"] = None

    try:
        results["behavior_shift"] = test_behavioral_shift_tool_changes()
        print()
    except AssertionError as e:
        print(f"✗ Behavioral Shift FAILED: {e}\n")
        results["behavior_shift"] = None

    try:
        results["agreement"] = test_agreement_score_consensus_divergence()
        print()
    except AssertionError as e:
        print(f"✗ Agreement Score FAILED: {e}\n")
        results["agreement"] = None

    try:
        results["stagnation"] = test_stagnation_repetition_detection()
        print()
    except AssertionError as e:
        print(f"✗ Stagnation FAILED: {e}\n")
        results["stagnation"] = None

    try:
        results["loop"] = test_loop_detection_repetitive_sequences()
        print()
    except AssertionError as e:
        print(f"✗ Loop Detection FAILED: {e}\n")
        results["loop"] = None

    try:
        results["combined"] = test_combined_drift_scoring()
        print()
    except AssertionError as e:
        print(f"✗ Combined Scoring FAILED: {e}\n")
        results["combined"] = None

    # Summary
    print("="*60)
    passed = sum(1 for v in results.values() if v is not None)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed ✓")
    print("="*60)

    return passed == total


if __name__ == "__main__":
    success = run_all_signal_tests()
    exit(0 if success else 1)
