#!/usr/bin/env python3
"""
Example: Degradation Mode Monitoring

Shows how to use is_degradation=True in behavioral_footprint_shift
to detect when an agent is degrading (losing tools, losing output quality)
rather than evolving.

Scenario: Quality monitoring for an LLM service that should detect
when model performance is degrading (tool removal penalty 0.8x vs 0.5x).
"""

from drift_detector.core.drift_detector_agent import (
    DriftDetectorAgent,
    AgentConfig,
    behavioral_footprint_shift,
)


def example_degradation_vs_evolution():
    """
    Compare degradation mode vs normal mode.
    """

    print("=" * 80)
    print("DEGRADATION MONITORING EXAMPLE")
    print("=" * 80)
    print()

    # Scenario 1: Tool removal (potential degradation)
    before = type("Snapshot", (), {
        "tool_calls": ["search", "analyze", "validate", "report"],
        "response_length": 500,
    })()

    after = type("Snapshot", (), {
        "tool_calls": ["search", "report"],  # Lost analyze, validate
        "response_length": 250,
    })()

    # Normal mode (balanced penalties)
    shift_normal = behavioral_footprint_shift(before, after, is_degradation=False)

    # Degradation mode (penalizes removal more heavily)
    shift_degradation = behavioral_footprint_shift(before, after, is_degradation=True)

    print("Scenario: Agent loses 2 tools (analyze, validate)")
    print()
    print(f"Normal mode (is_degradation=False):")
    print(f"  Shift score: {shift_normal:.3f}")
    print(f"  Removal penalty: 0.5x, Addition penalty: 0.1x")
    print()
    print(f"Degradation mode (is_degradation=True):")
    print(f"  Shift score: {shift_degradation:.3f}")
    print(f"  Removal penalty: 0.8x, Addition penalty: 0.2x")
    print(f"  → Penalizes tool loss more heavily")
    print()
    print(f"Difference: {abs(shift_degradation - shift_normal):.3f}")
    if shift_degradation > shift_normal:
        print("✓ Degradation mode correctly penalizes tool removal more")
    print()
    print("=" * 80)


def example_degradation_monitoring_full():
    """
    Full example: Quality monitoring service with degradation detection.
    """

    print("\nFull Degradation Monitoring Example")
    print("-" * 80)

    # Create detector in degradation mode
    config = AgentConfig(
        agent_id="quality_monitor_1",
        is_degradation=True,  # Monitor for degradation
    )
    detector = DriftDetectorAgent(config)

    # Day 1: Full capability
    day1_snapshot = detector.snapshot(
        agent_id="quality_monitor_1",
        response_text=(
            "Searched 5 sources. "
            "Analyzed results with NLP. "
            "Validated facts. "
            "Generated comprehensive report."
        ),
        tool_calls=["search", "nlp_analyze", "fact_check", "report_gen"],
    )

    # Day 2: Degradation (lost 2 tools)
    day2_snapshot = detector.snapshot(
        agent_id="quality_monitor_1",
        response_text=(
            "Searched sources. "
            "Generated report."
        ),
        tool_calls=["search", "report_gen"],  # Lost nlp_analyze, fact_check
    )

    # Measure drift with degradation mode
    report = detector.measure_drift(day1_snapshot, day2_snapshot)

    print()
    print(f"Day 1→Day 2 Drift Analysis:")
    print(f"  Ghost Loss: {report.ghost_loss:.3f}")
    print(f"  Behavior Shift: {report.behavior_shift:.3f}")
    print(f"    → Tool loss penalized heavily (0.8x)")
    print(f"    → Response shrinkage penalized (lost 50% of length)")
    print()
    print(f"  Combined Drift: {report.combined_drift_score:.3f}")
    print(f"  Status: {'DEGRADATION ALERT ⚠️' if report.is_drifting else 'NORMAL ✓'}")
    print()
    print("Use case: Alert DevOps when model quality degrades")
    print("-" * 80)


def example_length_tolerance():
    """
    Show how 30% length tolerance works in degradation mode.
    """

    print("\nLength Tolerance Example (30% threshold)")
    print("-" * 80)

    # Small growth (within tolerance)
    before_small = type("Snapshot", (), {
        "tool_calls": ["search"],
        "response_length": 100,
    })()

    after_small = type("Snapshot", (), {
        "tool_calls": ["search"],
        "response_length": 115,  # +15% (within 30%)
    })()

    shift_small = behavioral_footprint_shift(before_small, after_small)

    # Large growth (exceeds tolerance)
    after_large = type("Snapshot", (), {
        "tool_calls": ["search"],
        "response_length": 160,  # +60% (exceeds 30%)
    })()

    shift_large = behavioral_footprint_shift(before_small, after_large)

    print(f"Response length: 100 → 115 (+15%, within 30% tolerance)")
    print(f"  Shift score: {shift_small:.3f}")
    print()
    print(f"Response length: 100 → 160 (+60%, exceeds 30% tolerance)")
    print(f"  Shift score: {shift_large:.3f}")
    print()
    if shift_large > shift_small:
        print("✓ Large growth correctly penalized beyond 30% threshold")
    print("-" * 80)


if __name__ == "__main__":
    example_degradation_vs_evolution()
    example_degradation_monitoring_full()
    example_length_tolerance()
