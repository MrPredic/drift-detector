#!/usr/bin/env python3
"""
Test stagnation detection with adaptive window detection.

Validates DriftDetectorAgent.detect_stagnation with:
- adaptive_window=True: Auto-detect repeating pattern length
- adaptive_window=False: Fixed window (backward compatible)
"""

import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from drift_detector.core.drift_detector_agent import (
    SessionSnapshot,
    DriftDetectorAgent,
    AgentConfig,
)


class TestStagnationAdaptiveWindow:
    """Test detect_stagnation with adaptive_window parameter"""

    def test_stagnation_identical_outputs(self):
        """Identical outputs should have high stagnation score"""
        config = AgentConfig(agent_id="test_stagnation")
        detector = DriftDetectorAgent(config)

        before = SessionSnapshot(
            "test", "2026-04-11T00:00:00",
            "The same result", 15,
            ["tool_a"], 1, "hash1"
        )
        after = SessionSnapshot(
            "test", "2026-04-11T00:01:00",
            "The same result", 15,
            ["tool_a"], 1, "hash2"
        )

        score_adaptive = detector.detect_stagnation(before, after, adaptive_window=True)
        score_fixed = detector.detect_stagnation(before, after, adaptive_window=False)

        # Both should detect high stagnation
        assert score_adaptive > 0.8, f"Adaptive should detect stagnation, got {score_adaptive:.2f}"
        assert score_fixed > 0.8, f"Fixed should detect stagnation, got {score_fixed:.2f}"
        print(
            f"✓ Identical outputs: adaptive={score_adaptive:.2f}, "
            f"fixed={score_fixed:.2f}"
        )

    def test_stagnation_different_outputs(self):
        """Different outputs should have low stagnation"""
        config = AgentConfig(agent_id="test_stagnation")
        detector = DriftDetectorAgent(config)

        before = SessionSnapshot(
            "test", "2026-04-11T00:00:00",
            "First result with unique content", 32,
            ["tool_a"], 1, "hash1"
        )
        after = SessionSnapshot(
            "test", "2026-04-11T00:01:00",
            "Completely different output", 27,
            ["tool_b"], 1, "hash2"
        )

        score = detector.detect_stagnation(before, after, adaptive_window=True)

        # Should have low stagnation
        assert score < 0.5, f"Different outputs should have low stagnation, got {score:.2f}"
        print(f"✓ Different outputs: stagnation={score:.2f}")

    def test_stagnation_partial_overlap(self):
        """Partial overlap should have moderate stagnation"""
        config = AgentConfig(agent_id="test_stagnation")
        detector = DriftDetectorAgent(config)

        before = SessionSnapshot(
            "test", "2026-04-11T00:00:00",
            "The patient has fever and chills", 32,
            ["tool_a"], 1, "hash1"
        )
        after = SessionSnapshot(
            "test", "2026-04-11T00:01:00",
            "The patient has headache", 24,
            ["tool_a"], 1, "hash2"
        )

        score = detector.detect_stagnation(before, after, adaptive_window=True)

        # Should be moderate (some shared terms: "The patient has")
        assert 0.3 < score < 0.8, (
            f"Partial overlap should be moderate, got {score:.2f}"
        )
        print(f"✓ Partial overlap: stagnation={score:.2f}")

    def test_empty_output_handling(self):
        """Empty outputs should be handled gracefully"""
        config = AgentConfig(agent_id="test_stagnation")
        detector = DriftDetectorAgent(config)

        before = SessionSnapshot(
            "test", "2026-04-11T00:00:00", "", 0, [], 0, "hash1"
        )
        after = SessionSnapshot(
            "test", "2026-04-11T00:01:00", "", 0, [], 0, "hash2"
        )

        score = detector.detect_stagnation(before, after, adaptive_window=True)

        # Both empty = perfectly similar
        assert score == 1.0, f"Empty outputs should return 1.0, got {score:.2f}"
        print(f"✓ Empty outputs: stagnation={score:.2f}")


class TestAdaptiveWindowDetection:
    """Test adaptive window pattern detection in detect_stagnation"""

    def test_adaptive_window_boosts_repeated_stagnation(self):
        """Adaptive window should boost confidence if stagnation repeats"""
        config = AgentConfig(agent_id="test_adaptive")
        detector = DriftDetectorAgent(config)

        # Simulate 3 consecutive stagnation detections
        snapshot_a = SessionSnapshot(
            "test", "2026-04-11T00:00:00", "Same result", 11, ["tool"], 1, "h1"
        )
        snapshot_b = SessionSnapshot(
            "test", "2026-04-11T00:01:00", "Same result", 11, ["tool"], 1, "h2"
        )
        snapshot_c = SessionSnapshot(
            "test", "2026-04-11T00:02:00", "Same result", 11, ["tool"], 1, "h3"
        )
        snapshot_d = SessionSnapshot(
            "test", "2026-04-11T00:03:00", "Same result", 11, ["tool"], 1, "h4"
        )

        # Measure drift 3 times to build history
        detector.measure_drift(snapshot_a, snapshot_b)
        detector.measure_drift(snapshot_b, snapshot_c)
        detector.measure_drift(snapshot_c, snapshot_d)

        # Now test adaptive window detection
        # With history showing repeated high stagnation, score should be boosted
        score_adaptive = detector.detect_stagnation(
            snapshot_d, snapshot_d, adaptive_window=True
        )

        # Score might be boosted by 0.1 due to pattern
        assert score_adaptive >= 0.9, (
            f"Repeated stagnation should boost confidence, got {score_adaptive:.2f}"
        )
        print(f"✓ Adaptive window boosts repeated pattern: {score_adaptive:.2f}")

    def test_adaptive_vs_fixed_window(self):
        """Compare adaptive vs fixed window behavior"""
        config = AgentConfig(agent_id="test_compare")
        detector = DriftDetectorAgent(config)

        before = SessionSnapshot(
            "test", "2026-04-11T00:00:00",
            "Result with some variation", 26,
            ["tool_a"], 1, "h1"
        )
        after = SessionSnapshot(
            "test", "2026-04-11T00:01:00",
            "Result with similar content", 27,
            ["tool_a"], 1, "h2"
        )

        score_adaptive = detector.detect_stagnation(before, after, adaptive_window=True)
        score_fixed = detector.detect_stagnation(before, after, adaptive_window=False)

        # Scores should be close (both use token similarity)
        assert abs(score_adaptive - score_fixed) < 0.2, (
            f"Scores should be similar: adaptive={score_adaptive:.2f}, "
            f"fixed={score_fixed:.2f}"
        )
        print(
            f"✓ Adaptive vs fixed: "
            f"adaptive={score_adaptive:.2f}, fixed={score_fixed:.2f}"
        )


class TestStagnationEdgeCases:
    """Test edge cases in stagnation detection"""

    def test_stagnation_single_token(self):
        """Single token repeated should have high stagnation"""
        config = AgentConfig(agent_id="test_edge")
        detector = DriftDetectorAgent(config)

        before = SessionSnapshot("test", "2026-04-11T00:00:00", "result", 6, [], 0, "h1")
        after = SessionSnapshot("test", "2026-04-11T00:01:00", "result", 6, [], 0, "h2")

        score = detector.detect_stagnation(before, after, adaptive_window=True)

        assert score == 1.0, f"Single token repeated should be 1.0, got {score:.2f}"
        print(f"✓ Single token: {score:.2f}")

    def test_stagnation_case_insensitive(self):
        """Stagnation should be case-insensitive"""
        config = AgentConfig(agent_id="test_edge")
        detector = DriftDetectorAgent(config)

        before = SessionSnapshot(
            "test", "2026-04-11T00:00:00", "Result TEXT", 11, [], 0, "h1"
        )
        after = SessionSnapshot(
            "test", "2026-04-11T00:01:00", "result text", 11, [], 0, "h2"
        )

        score = detector.detect_stagnation(before, after, adaptive_window=True)

        # Case differences shouldn't matter
        assert score > 0.8, f"Case-insensitive match should be high, got {score:.2f}"
        print(f"✓ Case-insensitive: {score:.2f}")

    def test_stagnation_whitespace_handling(self):
        """Whitespace differences should be normalized"""
        config = AgentConfig(agent_id="test_edge")
        detector = DriftDetectorAgent(config)

        before = SessionSnapshot(
            "test", "2026-04-11T00:00:00", "The   quick   brown", 19, [], 0, "h1"
        )
        after = SessionSnapshot(
            "test", "2026-04-11T00:01:00", "The quick brown", 15, [], 0, "h2"
        )

        score = detector.detect_stagnation(before, after, adaptive_window=True)

        # Same words despite whitespace differences
        assert score > 0.7, f"Should handle whitespace, got {score:.2f}"
        print(f"✓ Whitespace normalization: {score:.2f}")


class TestBackwardCompatibility:
    """Ensure backward compatibility of detect_stagnation"""

    def test_stagnation_default_adaptive_true(self):
        """Default parameter value should work"""
        config = AgentConfig(agent_id="test_default")
        detector = DriftDetectorAgent(config)

        before = SessionSnapshot(
            "test", "2026-04-11T00:00:00", "Same output", 11, [], 0, "h1"
        )
        after = SessionSnapshot(
            "test", "2026-04-11T00:01:00", "Same output", 11, [], 0, "h2"
        )

        # Call without adaptive_window parameter
        score = detector.detect_stagnation(before, after)

        # Should work (defaults to adaptive_window=True)
        assert score > 0.8, f"Default should detect stagnation, got {score:.2f}"
        print(f"✓ Default adaptive_window=True: {score:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
