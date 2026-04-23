#!/usr/bin/env python3
"""
Example: Coding Agent with Task-Type Adaptive Loop Detection

Shows how to use task_type="coding" in detect_loops_ensemble to allow
more repetition tolerance (min_unique_actions=4 vs general=3).

Scenario: A coding assistant that recursively calls same functions
(normal for coding) shouldn't be flagged as looping.
"""

from drift_detector.core.drift_detector_agent import (
    DriftDetectorAgent,
    AgentConfig,
    detect_loops_ensemble,
)


def example_coding_with_recursion():
    """
    Example: Coding task with nested/recursive patterns.
    """

    print("=" * 80)
    print("CODING AGENT LOOP DETECTION EXAMPLE")
    print("=" * 80)
    print()

    # Action sequence: parse → analyze → compile → parse → analyze → compile
    # This is normal for coding (3-step cycle repeated)
    coding_actions = [
        "parse_input",
        "analyze_syntax",
        "compile",
        "parse_input",
        "analyze_syntax",
        "compile",
        "parse_input",
        "analyze_syntax",
        "compile",
    ]

    # Test with different task types
    print("Scenario: Coding assistant parsing → analyzing → compiling (3-step cycle)")
    print(f"Actions: {coding_actions}")
    print()

    # Coding task (more tolerant)
    report_coding = detect_loops_ensemble(coding_actions, task_type="coding")
    print("With task_type='coding' (min_unique=4):")
    print(f"  Diversity: {report_coding.diversity_score:.3f}")
    print(f"  Is Looping: {report_coding.is_looping}")
    print(f"  → Tolerates 3 unique actions (normal coding pattern)")
    print()

    # General task (less tolerant)
    report_general = detect_loops_ensemble(coding_actions, task_type="general")
    print("With task_type='general' (min_unique=3):")
    print(f"  Diversity: {report_general.diversity_score:.3f}")
    print(f"  Is Looping: {report_general.is_looping}")
    print(f"  → Also ok with 3 unique actions (borderline)")
    print()

    # Research task (very strict)
    report_research = detect_loops_ensemble(coding_actions, task_type="research")
    print("With task_type='research' (min_unique=2):")
    print(f"  Diversity: {report_research.diversity_score:.3f}")
    print(f"  Is Looping: {report_research.is_looping}")
    print(f"  → More tolerant (expects exploration with only 2+ unique)")
    print()

    print("=" * 80)


def example_coding_agent_full():
    """
    Example: Full DriftDetectorAgent with task_type="coding".
    """

    # Create coding agent detector
    config = AgentConfig(
        agent_id="code_assistant_1",
        task_type="coding",  # Allows nested loops, recursion
    )
    detector = DriftDetectorAgent(config)

    # Before: Complex code analysis
    before = detector.snapshot(
        agent_id="code_assistant_1",
        response_text=(
            "Parsed function with recursive calls. "
            "Detected loop structure. "
            "Generated optimized bytecode."
        ),
        tool_calls=["tokenizer", "parser", "optimizer", "codegen"],
    )

    # After: Simplified output (some tools removed)
    after = detector.snapshot(
        agent_id="code_assistant_1",
        response_text=(
            "Parsed the code. "
            "Generated output."
        ),
        tool_calls=["tokenizer", "parser"],  # Lost optimizer, codegen
    )

    report = detector.measure_drift(
        before, after,
        action_history=[
            "tokenize", "parse", "optimize", "tokenize", "parse", "optimize"
        ],
    )

    print("\nFull Drift Report:")
    print(f"  Ghost Loss: {report.ghost_loss:.3f}")
    print(f"  Behavior Shift: {report.behavior_shift:.3f}")
    print(f"  Combined Drift: {report.combined_drift_score:.3f}")
    print(f"  Is Drifting: {report.is_drifting}")


if __name__ == "__main__":
    example_coding_with_recursion()
    print()
    example_coding_agent_full()
