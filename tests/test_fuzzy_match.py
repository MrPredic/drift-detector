"""
Tests for fuzzy_match utilities — levenshtein_distance + fuzzy_token_similarity.
"""
import pytest
from drift_detector.utils.fuzzy_match import levenshtein_distance, fuzzy_token_similarity


class TestLevenshteinDistance:
    def test_identical_strings(self):
        assert levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self):
        assert levenshtein_distance("", "") == 0

    def test_one_empty(self):
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("", "abc") == 3

    def test_single_insertion(self):
        assert levenshtein_distance("cat", "cats") == 1

    def test_single_deletion(self):
        assert levenshtein_distance("cats", "cat") == 1

    def test_single_substitution(self):
        assert levenshtein_distance("cat", "bat") == 1

    def test_completely_different(self):
        assert levenshtein_distance("abc", "xyz") == 3

    def test_symmetry(self):
        assert levenshtein_distance("kitten", "sitting") == levenshtein_distance("sitting", "kitten")

    def test_classic_kitten_sitting(self):
        assert levenshtein_distance("kitten", "sitting") == 3

    def test_longer_strings(self):
        assert levenshtein_distance("temperature", "temperature") == 0
        assert levenshtein_distance("temperature", "temp") == 7

    def test_single_chars(self):
        assert levenshtein_distance("a", "b") == 1
        assert levenshtein_distance("a", "a") == 0

    def test_partial_overlap(self):
        # "algorithm" → "algorithmic" = 2 insertions
        assert levenshtein_distance("algorithm", "algorithmic") == 2


class TestFuzzyTokenSimilarity:
    def test_both_empty(self):
        assert fuzzy_token_similarity(set(), set()) == 1.0

    def test_one_empty(self):
        assert fuzzy_token_similarity({"a", "b"}, set()) == 0.0
        assert fuzzy_token_similarity(set(), {"a", "b"}) == 0.0

    def test_identical_sets(self):
        tokens = {"temperature", "pressure", "velocity"}
        assert fuzzy_token_similarity(tokens, tokens) == pytest.approx(1.0)

    def test_completely_different_sets(self):
        t1 = {"alpha", "beta"}
        t2 = {"xyz", "qrs"}
        result = fuzzy_token_similarity(t1, t2, threshold=0)
        assert result == 0.0

    def test_partial_match_exact(self):
        t1 = {"cat", "dog", "fish"}
        t2 = {"cat", "dog", "bird"}
        # 2 exact matches out of union of 4 unique tokens → 2/4 = 0.5
        result = fuzzy_token_similarity(t1, t2, threshold=0)
        assert result == pytest.approx(0.5)

    def test_fuzzy_match_within_threshold(self):
        # "cats" and "cat" differ by 1 edit → within threshold=2
        t1 = {"cats"}
        t2 = {"cat"}
        result = fuzzy_token_similarity(t1, t2, threshold=2)
        assert result > 0.0

    def test_fuzzy_no_match_above_threshold(self):
        # "temperature" and "temp" differ by 7 → outside threshold=2
        t1 = {"temperature"}
        t2 = {"temp"}
        result = fuzzy_token_similarity(t1, t2, threshold=2)
        assert result == 0.0

    def test_threshold_zero_exact_only(self):
        t1 = {"hello", "world"}
        t2 = {"hello", "word"}  # "word" vs "world" = 1 edit
        result = fuzzy_token_similarity(t1, t2, threshold=0)
        # Only "hello" matches exactly → 1 match / 3 in union = 1/3
        assert result == pytest.approx(1 / 3)

    def test_full_match_fuzzy(self):
        # Both tokens close variants — all should match with threshold=2
        # union = {"cat","bat","cats","bats"} = 4, matches = 2 → 2/4 = 0.5
        t1 = {"cat", "bat"}
        t2 = {"cats", "bats"}  # each 1 edit away
        result = fuzzy_token_similarity(t1, t2, threshold=2)
        assert result >= 0.5

    def test_single_token_match(self):
        result = fuzzy_token_similarity({"hello"}, {"hello"}, threshold=0)
        assert result == pytest.approx(1.0)

    def test_return_type_is_float(self):
        result = fuzzy_token_similarity({"a"}, {"b"})
        assert isinstance(result, float)

    def test_result_within_bounds(self):
        import random
        random.seed(42)
        tokens = [set(["word1", "word2", "word3"][:i]) for i in range(1, 4)]
        for t1 in tokens:
            for t2 in tokens:
                r = fuzzy_token_similarity(t1, t2)
                assert 0.0 <= r <= 1.0
