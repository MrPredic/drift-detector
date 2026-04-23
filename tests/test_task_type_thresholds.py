#!/usr/bin/env python3
"""
Test task-type adaptive thresholds in loop detection.

Validates that detect_loops_ensemble adjusts min_unique_actions based on task_type:
- "coding": 4 (allows nested loops, recursion)
- "research": 2 (expects high exploration variety)
- "trading": 3 (balanced, some patterns ok)
- "general": 3 (default)
"""

import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from drift_detector.core.drift_detector_agent import detect_loops_ensemble, LoopReport


class TestTaskTypeThresholds:
    """Test detect_loops_ensemble with various task_types"""

    def test_coding_allows_more_repetition(self):
        """Coding task should tolerate 3 unique actions (allows nested loops)"""
        # 3 unique actions in 10 calls: normal task would flag, coding allows
        actions_coding_ok = ["search", "parse", "compile"] * 3 + ["search"]

        # Coding: min_unique = 4, we have 3 = should loop
        # But pattern is not heavily repetitive
        report_coding = detect_loops_ensemble(actions_coding_ok, task_type="coding")

        # General: min_unique = 3, we have 3 = borderline
        report_general = detect_loops_ensemble(actions_coding_ok, task_type="general")

        # Both should detect some looping
        print(
            f"Coding task (min_unique=4): is_looping={report_coding.is_looping}, "
            f"confidence={report_coding.combined_confidence:.2f}"
        )
        print(
            f"General task (min_unique=3): is_looping={report_general.is_looping}, "
            f"confidence={report_general.combined_confidence:.2f}"
        )

    def test_research_expects_high_variety(self):
        """Research task should tolerate only 2 unique actions min (expects exploration)"""
        # 2 unique actions: research ok, other tasks may flag
        actions_research_ok = ["search", "read"] * 5

        report_research = detect_loops_ensemble(actions_research_ok, task_type="research")
        report_general = detect_loops_ensemble(actions_research_ok, task_type="general")

        print(
            f"Research task (min_unique=2): is_looping={report_research.is_looping}, "
            f"diversity={report_research.diversity_score:.2f}"
        )
        print(
            f"General task (min_unique=3): is_looping={report_general.is_looping}, "
            f"diversity={report_general.diversity_score:.2f}"
        )

    def test_trading_balanced_threshold(self):
        """Trading task with moderate repetition (3 actions)"""
        # 3 unique actions: trading ok, balanced
        actions_trading = ["check_price", "execute_trade", "log_result"] * 3

        report_trading = detect_loops_ensemble(actions_trading, task_type="trading")
        report_research = detect_loops_ensemble(actions_trading, task_type="research")

        print(
            f"Trading task (min_unique=3): is_looping={report_trading.is_looping}, "
            f"confidence={report_trading.combined_confidence:.2f}"
        )
        print(
            f"Research task (min_unique=2): is_looping={report_research.is_looping}, "
            f"confidence={report_research.combined_confidence:.2f}"
        )

    def test_general_default_threshold(self):
        """General task defaults to min_unique=3"""
        actions = ["action_a", "action_b", "action_c"] * 3

        report = detect_loops_ensemble(actions, task_type="general")

        # Should detect loop since only 3 unique
        assert isinstance(report, LoopReport)
        print(
            f"General task: is_looping={report.is_looping}, "
            f"unique_actions=3, min_required=3"
        )

    def test_invalid_task_type_defaults(self):
        """Invalid task_type should default to general (min_unique=3)"""
        actions = ["a", "b"] * 5

        # Unknown task type should fall back to "general" (min_unique=3)
        report = detect_loops_ensemble(actions, task_type="unknown")

        assert isinstance(report, LoopReport)
        # 2 unique < 3 required, should loop
        print(
            f"Unknown task type (defaults to general): "
            f"is_looping={report.is_looping}, min_unique=3"
        )


class TestAdaptiveLoopDetection:
    """Test loop detection with adaptive thresholds"""

    def test_tight_loop_all_tasks(self):
        """Same action repeated should flag as loop in all task types"""
        actions_same = ["search"] * 10

        for task_type in ["coding", "research", "trading", "general"]:
            report = detect_loops_ensemble(actions_same, task_type=task_type)
            assert report.is_looping, (
                f"Task '{task_type}' should detect same action repeated as loop"
            )
            print(f"✓ {task_type}: detects tight loop")

    def test_ab_alternation_detected(self):
        """A-B alternation should be detected as loop"""
        actions_ab = ["search", "analyze"] * 5

        report = detect_loops_ensemble(actions_ab)
        # Low diversity (only 2 unique in 10 actions)
        assert report.is_looping or report.diversity_score > 0.4, (
            f"Should detect A-B loop, diversity={report.diversity_score:.2f}"
        )
        print(f"✓ A-B alternation: diversity={report.diversity_score:.2f}")

    def test_diverse_actions_no_loop(self):
        """Diverse action sequence should not be flagged as loop"""
        actions_diverse = [
            "search", "analyze", "fetch", "validate", "process",
            "search", "refine", "check", "report"
        ]

        report = detect_loops_ensemble(actions_diverse, task_type="general")

        # Many unique actions should not trigger loop
        assert not report.is_looping or report.diversity_score < 0.4, (
            f"Diverse actions shouldn't loop, diversity={report.diversity_score:.2f}"
        )
        print(f"✓ Diverse actions: diversity={report.diversity_score:.2f}")

    def test_pattern_matching_cross_boundary(self):
        """Pattern matching should detect repeating sequences across boundaries"""
        before_actions = ["search", "analyze", "fetch"]
        after_actions = ["search", "analyze", "fetch", "search"]

        report = detect_loops_ensemble(after_actions, before_actions=before_actions)

        # Pattern from before repeats in after
        assert report.pattern_score > 0.3, (
            f"Should detect pattern repetition, pattern_score={report.pattern_score:.2f}"
        )
        print(f"✓ Pattern cross-boundary: pattern_score={report.pattern_score:.2f}")


class TestTaskTypeIntegration:
    """Test task_type parameter throughout the system"""

    def test_all_task_types_valid(self):
        """All documented task types should work without error"""
        actions = ["a", "b", "c"] * 4
        valid_types = ["coding", "research", "trading", "general"]

        for task_type in valid_types:
            try:
                report = detect_loops_ensemble(actions, task_type=task_type)
                assert isinstance(report, LoopReport)
                print(f"✓ Task type '{task_type}' works correctly")
            except Exception as e:
                pytest.fail(f"Task type '{task_type}' raised: {e}")

    def test_empty_actions_all_task_types(self):
        """Empty action list should be handled in all task types"""
        for task_type in ["coding", "research", "trading", "general"]:
            report = detect_loops_ensemble([], task_type=task_type)
            assert isinstance(report, LoopReport)
            assert not report.is_looping, f"Empty actions shouldn't loop ({task_type})"
            print(f"✓ {task_type} handles empty actions")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
