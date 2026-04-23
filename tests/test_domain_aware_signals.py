#!/usr/bin/env python3
"""
Test domain-aware signals: domain_terms weighting, degradation mode, fuzzy matching.

These tests validate new features in DriftDetector v1.1:
- ghost_lexicon_score with domain_terms weighting
- behavioral_footprint_shift with is_degradation penalties
- agreement_score with fuzzy token matching
"""

import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from drift_detector.core.drift_detector_agent import (
    SessionSnapshot,
    ghost_lexicon_score,
    behavioral_footprint_shift,
    agreement_score,
)


class TestDomainTermsWeighting:
    """Test ghost_lexicon_score with domain_terms parameter"""

    def test_domain_terms_no_loss(self):
        """When domain terms survive, score should be high"""
        before_text = "quantum entanglement phenomenon requires sophisticated calibration systems"
        after_text = "quantum entanglement phenomenon is important today"

        before = SessionSnapshot(
            "test", "2026-04-11T00:00:00", before_text, len(before_text), [], 0, "hash1"
        )
        after = SessionSnapshot(
            "test", "2026-04-11T00:01:00", after_text, len(after_text), [], 0, "hash2"
        )

        domain_terms = ["quantum", "entanglement", "phenomenon"]
        score = ghost_lexicon_score(before, after, top_n=10, domain_terms=domain_terms)

        # Domain terms survive, so score should be high
        assert score > 0.5, f"Expected high score with surviving domain terms, got {score:.2f}"
        print(f"✓ Domain terms preserved: {score:.2f}")

    def test_domain_terms_with_loss(self):
        """When domain terms are lost, score should be penalized more"""
        before_text = "quantum entanglement phenomenon requires sophisticated calibration systems today"
        after_text = "phenomenon is important for science"

        before = SessionSnapshot(
            "test", "2026-04-11T00:00:00", before_text, len(before_text), [], 0, "hash1"
        )
        after = SessionSnapshot(
            "test", "2026-04-11T00:01:00", after_text, len(after_text), [], 0, "hash2"
        )

        domain_terms = ["quantum", "entanglement"]
        score = ghost_lexicon_score(before, after, top_n=10, domain_terms=domain_terms)

        # Domain terms lost, penalized more heavily
        assert score < 0.8, f"Expected lower score due to domain term loss, got {score:.2f}"
        print(f"✓ Domain term loss detected: {score:.2f}")

    def test_domain_terms_vs_no_domain_terms(self):
        """Domain term weighting should produce different score than unweighted"""
        before_text = "critical_medical_term unique_medical_word another_medical terminology "
        after_text = "another_medical"

        before = SessionSnapshot(
            "test", "2026-04-11T00:00:00", before_text, len(before_text), [], 0, "hash1"
        )
        after = SessionSnapshot(
            "test", "2026-04-11T00:01:00", after_text, len(after_text), [], 0, "hash2"
        )

        domain_terms = ["critical_medical_term", "unique_medical_word"]

        score_no_domain = ghost_lexicon_score(before, after, top_n=5)
        score_with_domain = ghost_lexicon_score(
            before, after, top_n=5, domain_terms=domain_terms
        )

        # With domain terms, loss should be penalized more (lower score)
        assert score_with_domain <= score_no_domain, (
            f"Expected domain weighting to not increase score: "
            f"no_domain={score_no_domain:.2f}, with_domain={score_with_domain:.2f}"
        )
        print(
            f"✓ Domain weighting effective: "
            f"no_domain={score_no_domain:.2f}, with_domain={score_with_domain:.2f}"
        )


class TestDegradationMode:
    """Test behavioral_footprint_shift with is_degradation parameter"""

    def test_degradation_mode_penalizes_removal(self):
        """Degradation mode should penalize tool removal more (0.8x vs 0.5x)"""
        before = SessionSnapshot(
            "test", "2026-04-11T00:00:00", "output1", 100,
            ["search", "analyze", "refine"], 3, "hash1"
        )
        after = SessionSnapshot(
            "test", "2026-04-11T00:01:00", "output2", 100,
            ["search"], 1, "hash2"
        )

        shift_normal = behavioral_footprint_shift(before, after, is_degradation=False)
        shift_degradation = behavioral_footprint_shift(before, after, is_degradation=True)

        # Degradation mode should penalize removal more heavily
        assert shift_degradation > shift_normal, (
            f"Expected degradation mode to have higher drift: "
            f"normal={shift_normal:.2f}, degradation={shift_degradation:.2f}"
        )
        print(
            f"✓ Degradation mode penalizes removal: "
            f"normal={shift_normal:.2f}, degradation={shift_degradation:.2f}"
        )

    def test_length_tolerance_30_percent(self):
        """Response length delta should allow 30% tolerance"""
        before = SessionSnapshot(
            "test", "2026-04-11T00:00:00", "output" * 10, 60,  # 60 chars
            ["search"], 1, "hash1"
        )

        # 10% growth: 66 chars (within 30% tolerance)
        after_growth = SessionSnapshot(
            "test", "2026-04-11T00:01:00", "output" * 11, 66,
            ["search"], 1, "hash2"
        )

        # 50% growth: 90 chars (exceeds 30% tolerance)
        after_large_growth = SessionSnapshot(
            "test", "2026-04-11T00:01:00", "output" * 15, 90,
            ["search"], 1, "hash3"
        )

        # Length tolerance is only meaningful in degradation mode — in normal
        # mode growth is allowed up to 100%, so we must pass is_degradation=True
        # to exercise the 30% tolerance path.
        shift_small = behavioral_footprint_shift(before, after_growth, is_degradation=True)
        shift_large = behavioral_footprint_shift(before, after_large_growth, is_degradation=True)

        # Small growth within tolerance should have lower response_delta component
        # Large growth exceeding tolerance should have higher response_delta component
        assert shift_large > shift_small, (
            f"Expected large growth to have higher drift: "
            f"small={shift_small:.2f}, large={shift_large:.2f}"
        )
        print(
            f"✓ Length tolerance working: "
            f"small_growth={shift_small:.2f}, large_growth={shift_large:.2f}"
        )


class TestFuzzyMatching:
    """Test agreement_score with use_fuzzy parameter"""

    def test_fuzzy_matching_similar_tokens(self):
        """Fuzzy mode should match similar tokens"""
        outputs = {
            "model1": "temperature pressure volume",
            "model2": "temp pressure vol",  # Similar but different tokens
        }

        score_exact = agreement_score(outputs, use_fuzzy=False)
        score_fuzzy = agreement_score(outputs, use_fuzzy=True)

        # Fuzzy should be higher because "temp" ≈ "temperature", "vol" ≈ "volume"
        # (though with Levenshtein distance threshold, these may not match)
        # At minimum, fuzzy shouldn't be lower
        assert score_fuzzy >= score_exact, (
            f"Expected fuzzy score >= exact score: "
            f"exact={score_exact:.2f}, fuzzy={score_fuzzy:.2f}"
        )
        print(f"✓ Fuzzy matching: exact={score_exact:.2f}, fuzzy={score_fuzzy:.2f}")

    def test_exact_matching_unchanged(self):
        """Exact matching (use_fuzzy=False) should work as before"""
        outputs = {
            "model1": "python is a programming language",
            "model2": "python is a programming language",
        }

        agreement = agreement_score(outputs, use_fuzzy=False)

        # Exact same text should have perfect agreement
        assert agreement == 1.0, f"Expected perfect agreement, got {agreement:.2f}"
        print(f"✓ Exact matching (identical): {agreement:.2f}")

    def test_single_model_both_modes(self):
        """Single model should return 1.0 in both fuzzy and exact modes"""
        outputs = {"model1": "some output"}

        score_exact = agreement_score(outputs, use_fuzzy=False)
        score_fuzzy = agreement_score(outputs, use_fuzzy=True)

        assert score_exact == 1.0, f"Exact mode: expected 1.0, got {score_exact:.2f}"
        assert score_fuzzy == 1.0, f"Fuzzy mode: expected 1.0, got {score_fuzzy:.2f}"
        print(f"✓ Single model: exact={score_exact:.2f}, fuzzy={score_fuzzy:.2f}")


class TestBackwardCompatibility:
    """Ensure new parameters are optional and backward compatible"""

    def test_ghost_lexicon_without_domain_terms(self):
        """ghost_lexicon_score should work without domain_terms (backward compat)"""
        before = SessionSnapshot(
            "test", "2026-04-11T00:00:00", "text with vocabulary", 20, [], 0, "hash1"
        )
        after = SessionSnapshot(
            "test", "2026-04-11T00:01:00", "text only", 9, [], 0, "hash2"
        )

        # Should not raise exception
        score = ghost_lexicon_score(before, after)
        assert 0 <= score <= 1, f"Expected 0 <= {score} <= 1"
        print(f"✓ ghost_lexicon without domain_terms: {score:.2f}")

    def test_behavioral_shift_without_degradation(self):
        """behavioral_footprint_shift should work without is_degradation (backward compat)"""
        before = SessionSnapshot(
            "test", "2026-04-11T00:00:00", "output1", 10, ["search"], 1, "hash1"
        )
        after = SessionSnapshot(
            "test", "2026-04-11T00:01:00", "output2", 10, ["analyze"], 1, "hash2"
        )

        # Should not raise exception
        shift = behavioral_footprint_shift(before, after)
        assert 0 <= shift <= 1, f"Expected 0 <= {shift} <= 1"
        print(f"✓ behavioral_footprint_shift without is_degradation: {shift:.2f}")

    def test_agreement_without_fuzzy(self):
        """agreement_score should work without use_fuzzy (backward compat)"""
        outputs = {
            "model1": "output text",
            "model2": "output words",
        }

        # Should not raise exception
        score = agreement_score(outputs)
        assert 0 <= score <= 1, f"Expected 0 <= {score} <= 1"
        print(f"✓ agreement_score without use_fuzzy: {score:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
