#!/usr/bin/env python3
"""
Fuzzy matching utilities for DriftDetectorAgent.
Pure Python implementation (no external dependencies).

Used in agreement_score() with use_fuzzy=True to enable semantic similarity matching.
"""


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein (edit) distance between two strings.
    Measures minimum number of single-character edits needed to transform s1 → s2.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Integer distance (0 = identical, higher = more different)

    Example:
        levenshtein_distance("temperature", "temp") = 7
        levenshtein_distance("algorithm", "algorithmic") = 2
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def fuzzy_token_similarity(tokens1: set, tokens2: set, threshold: int = 2) -> float:
    """
    Fuzzy matching between token sets using Levenshtein distance.

    Tokens within 'threshold' edit distance are considered matching.
    Useful for matching similar terms (e.g., "temp" ≈ "temperature").

    Args:
        tokens1: Set of tokens from first output
        tokens2: Set of tokens from second output
        threshold: Max Levenshtein distance to consider a match (default: 2)

    Returns:
        Float 0.0-1.0, ratio of matched tokens to union size

    Example:
        tokens1 = {"temperature", "pressure"}
        tokens2 = {"temp", "pressure"}
        # "temperature" ≈ "temp" (distance 11 > threshold 2) = no match
        # So this would return 1/3 ≈ 0.33 (only "pressure" matches)
    """
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0

    matches = 0
    for t1 in tokens1:
        for t2 in tokens2:
            if levenshtein_distance(t1, t2) <= threshold:
                matches += 1
                break  # Count each t1 at most once

    union_size = len(tokens1 | tokens2)
    return matches / union_size if union_size > 0 else 0.0
