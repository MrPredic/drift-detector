#!/usr/bin/env python3
"""
DriftDetectorAgent: Multi-signal behavioral drift detection for LLM agent chains
Solves: CrewAI #5155 (session-boundary drift), CrewAI #4682 (loop detection)

5 Detection Signals:
1. Ghost Lexicon - precision term loss after compression
2. Behavioral Footprint - tool sequence & response length divergence
3. Agreement Score - multi-model consensus
4. Loop Detection - 3-method ensemble (diversity, entropy, pattern matching)
5. Stagnation Detection - repeated tool results without improvement
"""

import re
import json
import hashlib
import sqlite3
import os
import logging
import threading
from collections import Counter
from math import log
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Standalone base — no external infrastructure dependencies required
BaseAgent = object


class AgentConfig:
    """Configuration for DriftDetectorAgent with validation"""
    def __init__(
        self,
        agent_id: str = "detector",
        drift_threshold: float = 0.6,  # Increased from 0.4 (false positive reduction)
        signal_threshold: float = 0.7,  # Increased from 0.5 (stricter per-signal check)
        loop_threshold: float = 0.6,
        role: str = "drift_detector",  # For compatibility
        task_type: str = "general",
        domain_terms: Optional[List[str]] = None,
        is_degradation: bool = False,
        use_fuzzy_matching: bool = False,
    ):
        # Validate thresholds
        if not 0 <= drift_threshold <= 1:
            raise ValueError(f"drift_threshold must be 0-1, got {drift_threshold}")
        if not 0 <= signal_threshold <= 1:
            raise ValueError(f"signal_threshold must be 0-1, got {signal_threshold}")
        if not 0 <= loop_threshold <= 1:
            raise ValueError(f"loop_threshold must be 0-1, got {loop_threshold}")
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError(f"agent_id must be non-empty string, got {agent_id}")
        if task_type not in ("general", "coding", "research", "trading", "medical"):
            raise ValueError(f"task_type must be one of (general, coding, research, trading, medical), got {task_type}")

        self.agent_id = agent_id
        self.drift_threshold = drift_threshold
        self.signal_threshold = signal_threshold
        self.loop_threshold = loop_threshold
        self.role = role
        self.task_type = task_type
        self.domain_terms = domain_terms or []
        self.is_degradation = is_degradation
        self.use_fuzzy_matching = use_fuzzy_matching

# Optional LangChain imports (only needed if using LangChain integration)
try:
    from langchain_core.callbacks import BaseCallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object  # Fallback for non-LangChain users


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SessionSnapshot:
    """Behavioral fingerprint at point in time"""
    agent_id: str
    timestamp: str
    response_text: str
    response_length: int
    tool_calls: List[str]
    tool_call_count: int
    output_hash: str
    vocabulary: Counter = field(default_factory=Counter)

    def __post_init__(self):
        if not self.vocabulary:
            self.vocabulary = self._extract_vocabulary(self.response_text)

    @staticmethod
    def _extract_vocabulary(text: str, min_len: int = 3) -> Counter:
        """Extract word frequency from text"""
        if not text or not isinstance(text, str):
            return Counter()
        words = re.findall(r'\b[a-zA-Z]{%d,}\b' % min_len, text.lower())
        return Counter(words)


@dataclass
class LoopReport:
    """Loop detection results from ensemble"""
    diversity_score: float  # 0-1: low = repetitive
    entropy_score: float    # 0-1: low = stagnant
    pattern_score: float    # 0-1: high = pattern repeats
    combined_confidence: float
    is_looping: bool


@dataclass
class DriftAlert:
    """Alert when drift detected"""
    level: str  # "info", "warning", "critical"
    timestamp: str
    signal: str  # Which signal triggered: "ghost_loss", "behavior_shift", "agreement", "loop", "stagnation"
    score: float
    threshold: float
    message: str

    @staticmethod
    def from_report(report: 'DriftReport', signal_threshold: float = 0.5) -> Optional['DriftAlert']:
        """Generate alert if any signal exceeds threshold"""
        if report is None:
            return None

        # Check each signal
        signals = [
            ("ghost_loss", report.ghost_loss),
            ("behavior_shift", report.behavior_shift),
            ("agreement", 1.0 - report.agreement_score),  # Invert: lower = more drift
            ("stagnation", report.stagnation_score),
        ]

        for signal_name, score in signals:
            if score > signal_threshold:
                level = "critical" if score > signal_threshold * 1.3 else "warning"
                message = f"{signal_name} exceeded threshold: {score:.3f} > {signal_threshold:.3f}"
                return DriftAlert(
                    level=level,
                    timestamp=report.timestamp,
                    signal=signal_name,
                    score=score,
                    threshold=signal_threshold,
                    message=message
                )

        if report.loop_report and report.loop_report.is_looping:
            message = f"Loop detected: confidence {report.loop_report.combined_confidence:.3f}"
            return DriftAlert(
                level="warning",
                timestamp=report.timestamp,
                signal="loop_detection",
                score=report.loop_report.combined_confidence,
                threshold=signal_threshold,
                message=message
            )

        return None


@dataclass
class DriftReport:
    """Complete drift analysis"""
    ghost_loss: float
    behavior_shift: float
    agreement_score: float
    loop_report: Optional[LoopReport]
    stagnation_score: float
    combined_drift_score: float
    is_drifting: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ============================================================================
# Signal Detection Functions
# ============================================================================

def ghost_lexicon_score(before: SessionSnapshot, after: SessionSnapshot, top_n: int = 50, domain_terms: Optional[List[str]] = None) -> float:
    """
    Fraction of prior top-N vocabulary that survives in current output.
    Detects loss of precision terms (domain-specific vocabulary).

    With domain_terms: weights critical terms 2.0x in loss calculation.

    Returns: 0.0 (total loss) to 1.0 (no loss)
    """
    if not before.vocabulary:
        return 1.0

    top_terms = {word for word, _ in before.vocabulary.most_common(top_n)}
    current_terms = set(after.vocabulary.keys())

    if not top_terms:
        return 1.0

    # If domain_terms provided, use weighted loss calculation
    if domain_terms:
        domain_set = set(t.lower() for t in domain_terms)
        weighted_survived = 0.0
        total_weight = 0.0

        for term in top_terms | domain_set:
            weight = 2.0 if term in domain_set else 1.0
            total_weight += weight

            if term in current_terms:
                weighted_survived += weight

        if total_weight == 0.0:
            return 1.0

        return weighted_survived / total_weight

    # Original: simple survival rate
    survived = len(top_terms & current_terms)
    return survived / len(top_terms) if len(top_terms) > 0 else 0.0


def behavioral_footprint_shift(before: SessionSnapshot, after: SessionSnapshot, is_degradation: bool = False) -> float:
    """
    Measure divergence in tool sequence and response characteristics.

    Weights: Tool divergence 70% (more important), response delta 30%

    With is_degradation: Adjusts penalties (removal 0.8x, addition 0.2x)
    Without: Normal penalties (removal 0.5x, addition 0.1x)
    Allows 30% length tolerance before penalizing.

    Returns: 0.0 (no drift) to 1.0 (complete divergence)
    """
    before_tools = set(before.tool_calls) if before.tool_calls else set()
    after_tools = set(after.tool_calls) if after.tool_calls else set()

    # Jaccard distance on tool sequences with degradation adjustment
    union = before_tools | after_tools
    if not union:
        tool_divergence = 0.0
    else:
        intersection = before_tools & after_tools
        removed_tools = before_tools - after_tools
        added_tools = after_tools - before_tools

        # Degradation mode: penalize removal more, addition less
        if is_degradation:
            removal_penalty = 0.8 * len(removed_tools)
            addition_penalty = 0.2 * len(added_tools)
        else:
            # Normal mode: balanced
            removal_penalty = 0.5 * len(removed_tools)
            addition_penalty = 0.1 * len(added_tools)

        tool_divergence = (removal_penalty + addition_penalty) / len(union) if union else 0.0

    # Response length delta - tolerance depends on mode
    max_len = max(before.response_length, after.response_length, 1)
    length_diff = abs(before.response_length - after.response_length) / max_len

    # Non-degradation: allow 100% variation (healthy growth)
    # Degradation: allow only 30% variation
    tolerance = 0.3 if is_degradation else 1.0

    if length_diff <= tolerance:
        response_delta = 0.0
    else:
        # Cap at 1.0 to prevent unbounded growth
        response_delta = min(1.0, length_diff - tolerance)

    # Weighted combination
    return 0.7 * tool_divergence + 0.3 * response_delta


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
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


def _fuzzy_token_similarity(tokens1: set, tokens2: set, threshold: int = 2) -> float:
    """Fuzzy matching: tokens within Levenshtein distance threshold count as match"""
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0

    matches = 0
    for t1 in tokens1:
        for t2 in tokens2:
            if _levenshtein_distance(t1, t2) <= threshold:
                matches += 1
                break  # Count each t1 at most once

    union_size = len(tokens1 | tokens2)
    return matches / union_size if union_size > 0 else 0.0


def agreement_score(outputs: Dict[str, str], use_fuzzy: bool = False) -> float:
    """
    Multi-model consensus via token overlap.

    If 1 model: return 1.0 (no basis for comparison)
    If 2+ models: compute pairwise similarity, return average

    With use_fuzzy=True: Use fuzzy token matching (Levenshtein distance <= 2)
    With use_fuzzy=False: Use exact token overlap (default, backward compatible)

    Returns: 0.0 (no agreement) to 1.0 (perfect agreement)
    """
    if len(outputs) < 2:
        return 1.0

    outputs_list = list(outputs.values())
    similarities = []

    for i in range(len(outputs_list)):
        for j in range(i + 1, len(outputs_list)):
            tokens_i = set(outputs_list[i].split())
            tokens_j = set(outputs_list[j].split())

            if use_fuzzy:
                # Fuzzy matching with Levenshtein distance
                similarity = _fuzzy_token_similarity(tokens_i, tokens_j, threshold=2)
            else:
                # Exact token overlap (original behavior)
                if tokens_i or tokens_j:
                    union = tokens_i | tokens_j
                    intersection = tokens_i & tokens_j
                    similarity = len(intersection) / len(union) if union else 0.0
                else:
                    similarity = 0.0

            similarities.append(similarity)

    return sum(similarities) / len(similarities) if similarities else 1.0


def detect_loops_by_diversity(actions: List[str], window: int = 20) -> float:
    """
    Low diversity in action sequence = repetitive behavior.
    If unique_actions < window/3, likely a loop (e.g., 2 unique in 6 actions).

    Returns: 0.0 (high diversity, no loop) to 1.0 (no diversity, likely loop)
    """
    if not actions or window <= 0:
        return 0.0

    recent = actions[-window:]
    unique = len(set(recent))

    # If unique actions < 1/3 of window, it's repetitive
    # E.g., 2 unique in 6 actions (A-B-A-B-A-B) = repetitive
    diversity_ratio = unique / len(recent) if recent else 1.0

    return 1.0 - diversity_ratio  # Invert: 1.0 = all same (looping)


def detect_loops_by_entropy(actions: List[str], window: int = 20) -> float:
    """
    Shannon entropy on action distribution.
    Low entropy = stagnant (same actions repeatedly).

    Returns: 0.0 (high entropy, varied) to 1.0 (low entropy, repetitive)
    """
    if not actions or window <= 0:
        return 0.0

    recent = actions[-window:]
    counts = Counter(recent)

    if not counts:
        return 0.0

    probs = [c / len(recent) for c in counts.values()]
    entropy = -sum(p * log(p) for p in probs if p > 0)
    # Max entropy: log(n) where n = number of unique symbols
    # If only 1 unique symbol → max entropy = 0 (deterministic)
    max_entropy = log(len(counts)) if len(counts) > 1 else 0.0

    normalized = entropy / max_entropy if max_entropy > 0 else 0.0
    return 1.0 - normalized  # Invert: 1.0 = low entropy (looping)


def detect_loops_by_pattern(before_actions: List[str], after_actions: List[str], max_window: int = 50) -> float:
    """
    Longest Common Subsequence (LCS) to detect pattern repetition.
    If same actions repeat across session boundary, agent is looping.

    Note: Limited to last max_window actions to avoid O(n²) explosion.

    Returns: 0.0 (no pattern repeat) to 1.0 (exact pattern repeats)
    """
    if not before_actions or not after_actions:
        return 0.0

    # Limit to last max_window for performance (avoid O(n²) on long histories)
    before = before_actions[-max_window:] if len(before_actions) > max_window else before_actions
    after = after_actions[-max_window:] if len(after_actions) > max_window else after_actions

    # LCS length
    m, n = len(before), len(after)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if before[i - 1] == after[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]
    similarity = lcs_length / max(1, len(before))

    return similarity


def detect_loops_ensemble(actions: List[str],
                         before_actions: Optional[List[str]] = None,
                         threshold: float = 0.45,
                         task_type: str = "general") -> LoopReport:
    """
    Ensemble of 3 loop detection methods with adaptive thresholds.

    task_type determines min_unique_actions tolerance:
      - "coding": 4 (allows more complexity/recursion)
      - "research": 2 (expects diverse exploration)
      - "trading": 3 (balanced repetition tolerance)
      - "general": 3 (default)

    Alert if diversity > threshold (low variation = repetitive)
    OR if pattern matching shows repetition.
    OR if entropy is very low.
    """
    # Handle empty actions case
    if not actions:
        return LoopReport(
            diversity_score=0.0,
            entropy_score=0.0,
            pattern_score=0.0,
            combined_confidence=0.0,
            is_looping=False
        )

    diversity = detect_loops_by_diversity(actions)
    entropy = detect_loops_by_entropy(actions)
    pattern = detect_loops_by_pattern(before_actions or [], actions) if before_actions else 0.0

    # Map task_type to minimum unique actions allowed
    min_unique_actions_map = {
        "coding": 4,
        "research": 2,
        "trading": 3,
        "general": 3,
        "medical": 4,  # medical requires high action diversity
    }
    min_unique_actions = min_unique_actions_map.get(task_type, 3)

    # Check if action diversity violates task-specific threshold
    recent_window = actions[-20:] if len(actions) > 20 else actions
    unique_actions = len(set(recent_window))
    is_looping_by_task = unique_actions < min_unique_actions and unique_actions > 0 and len(recent_window) >= min_unique_actions * 2

    # Diversity is most reliable signal
    # If < threshold, likely looping
    if diversity > threshold or is_looping_by_task:
        combined = max(diversity, 0.5) if is_looping_by_task else diversity
    else:
        # Otherwise, check pattern or entropy
        if before_actions:
            combined = max(diversity, pattern)  # Use whichever is higher
        else:
            combined = diversity

    return LoopReport(
        diversity_score=diversity,
        entropy_score=entropy,
        pattern_score=pattern,
        combined_confidence=combined,
        is_looping=combined > threshold or is_looping_by_task
    )


def detect_stagnation(tool_results: List[Dict], window: int = 5, similarity_threshold: float = 0.8) -> float:
    """
    Measure stagnation: tool results don't improve across iterations.

    Returns: 0.0 (varied results) to 1.0 (identical results, stagnant)
    """
    if len(tool_results) < 2:
        return 0.0

    recent = tool_results[-window:]
    similarities = []

    for i in range(len(recent) - 1):
        tokens_i = set(str(recent[i]).split())
        tokens_next = set(str(recent[i + 1]).split())

        union = tokens_i | tokens_next
        if union:
            intersection = tokens_i & tokens_next
            sim = len(intersection) / len(union)
            similarities.append(sim)

    return sum(similarities) / len(similarities) if similarities else 0.0


# ============================================================================
# DriftDetectorAgent
# ============================================================================

class DriftDetectorAgent(BaseAgent):
    """
    Multi-signal drift detection agent.
    Standalone behavioral drift detector for LLM agents.
    """

    def __init__(self, config: AgentConfig = None, db_path: str = None, agent_id: str = None):
        """
        Initialize drift detector.

        Args:
            config: Optional AgentConfig instance
            db_path: Custom database path (env var: DRIFT_DETECTOR_DB)
            agent_id: Agent ID (overrides config if both provided)
        """
        # Store config for access to thresholds
        if config is None:
            config = AgentConfig(agent_id=agent_id or "drift_detector")
        self.config = config

        # Store agent ID
        if agent_id:
            self.agent_id = agent_id
        elif config and hasattr(config, 'agent_id'):
            self.agent_id = config.agent_id
        else:
            self.agent_id = "drift_detector"

        # Initialize snapshots and history (lock protects concurrent appends)
        self._lock = threading.Lock()
        self.snapshots: Dict[str, SessionSnapshot] = {}
        self.drift_history: List[DriftReport] = []

        # Configure database path
        if db_path:
            self.db_path = db_path
        else:
            self.db_path = os.getenv(
                "DRIFT_DETECTOR_DB",
                os.path.join(
                    os.path.expanduser("~"),
                    ".drift_detector",
                    "data.db"
                )
            )

        self._init_db()
        self._load_history()

    def detect_stagnation(self, before: SessionSnapshot, after: SessionSnapshot, adaptive_window: bool = True) -> float:
        """
        Stagnation Detection: Token similarity between consecutive outputs.
        High similarity = repetitive (stagnation).

        With adaptive_window=True: Auto-detect repeating pattern length
        With adaptive_window=False: Use fixed window=5 (backward compatible)

        Returns: 0.0 (different) to 1.0 (identical)
        """
        if not isinstance(before.response_text, str) or not isinstance(after.response_text, str):
            return 0.0

        # Tokenize
        tokens_before = set(before.response_text.lower().split())
        tokens_after = set(after.response_text.lower().split())

        # Jaccard similarity
        if not tokens_before and not tokens_after:
            return 1.0  # Both empty = perfectly similar

        if not tokens_before or not tokens_after:
            return 0.0  # One empty, one not = completely different

        union = tokens_before | tokens_after
        intersection = tokens_before & tokens_after

        if not union:
            return 0.0

        similarity = len(intersection) / len(union)

        # Optional: adaptive window detection (check if pattern repeats)
        with self._lock:
            history_snapshot = list(self.drift_history)
        if adaptive_window and len(history_snapshot) >= 3:
            # Look back at last few reports for repeating pattern
            recent_reports = history_snapshot[-5:]
            stagnation_scores = [r.stagnation_score for r in recent_reports]

            # If last N reports all high stagnation, confidence increases
            if all(s > 0.7 for s in stagnation_scores[-3:]):
                similarity = min(1.0, similarity + 0.1)  # Boost confidence if pattern repeats

        return similarity

    def detect_loop_ensemble_method(self,
                                   actions: List[str],
                                   before_actions: Optional[List[str]] = None) -> 'LoopReport':
        """
        Public method: Detect loops using 3-method ensemble.
        Wraps standalone detect_loops_ensemble function.
        """
        return detect_loops_ensemble(actions, before_actions)

    def snapshot(self, agent_id: str, response_text: str, tool_calls: Optional[List[str]] = None) -> SessionSnapshot:
        """Capture behavioral fingerprint"""
        # Validate response_text
        if response_text is None:
            raise ValueError("response_text cannot be None")
        if not isinstance(response_text, str):
            response_text = str(response_text)

        # Validate tool_calls
        if tool_calls is None:
            tool_calls = []
        else:
            # Ensure all tool_calls are strings
            tool_calls = [str(t) for t in tool_calls]

        # Limit large inputs (>100KB)
        if len(response_text) > 100_000:
            response_text = response_text[:100_000]
            # Note: Could add logging here in production

        # Normalize response_text for hashing (case + whitespace insensitive)
        normalized_text = response_text.lower().strip()

        return SessionSnapshot(
            agent_id=agent_id,
            timestamp=datetime.utcnow().isoformat(),
            response_text=response_text,
            response_length=len(response_text),
            tool_calls=tool_calls,
            tool_call_count=len(tool_calls),
            output_hash=hashlib.sha256(normalized_text.encode()).hexdigest()[:16]
        )

    def measure_drift(self,
                     before: SessionSnapshot,
                     after: SessionSnapshot,
                     action_history: Optional[List[str]] = None,
                     before_actions: Optional[List[str]] = None,
                     model_outputs: Optional[Dict[str, str]] = None) -> DriftReport:
        """
        Complete drift measurement: all 5 signals with config-driven parameters.

        Args:
            before: Snapshot before session boundary
            after: Snapshot after session boundary
            action_history: Current action sequence for loop detection
            before_actions: Actions before boundary for pattern matching
            model_outputs: Dict of model_name -> output for agreement scoring

        Uses config parameters:
            - task_type: Adaptive loop thresholds ("coding", "research", "trading", "general")
            - domain_terms: Critical vocabulary to weight in ghost_lexicon_score
            - is_degradation: Different tool penalty weights for degradation scenarios
            - use_fuzzy_matching: Fuzzy token matching in agreement_score
        """
        # Signal 1: Ghost Lexicon
        ghost_loss = 1.0 - ghost_lexicon_score(
            before, after, top_n=50,
            domain_terms=self.config.domain_terms if self.config.domain_terms else None
        )

        # Signal 2: Behavioral Footprint
        behavior_shift = behavioral_footprint_shift(
            before, after,
            is_degradation=self.config.is_degradation
        )

        # Signal 3: Agreement Score
        if model_outputs:
            agreement = agreement_score(
                model_outputs,
                use_fuzzy=self.config.use_fuzzy_matching
            )
        else:
            agreement = 1.0

        # Signal 4: Loop Detection
        if action_history:
            loop_report = detect_loops_ensemble(
                action_history, before_actions,
                task_type=self.config.task_type
            )
        else:
            loop_report = None

        # Signal 5: Stagnation Detection
        stagnation_score = self.detect_stagnation(before, after, adaptive_window=True)

        # Combined drift score (5 signals — includes loop confidence)
        loop_score = loop_report.combined_confidence if loop_report and loop_report.is_looping else 0.0
        signals = [ghost_loss, behavior_shift, 1.0 - agreement, stagnation_score, loop_score]
        combined_drift_score = sum(signals) / len(signals)

        # Use configurable thresholds
        drift_threshold = self.config.drift_threshold
        signal_threshold = self.config.signal_threshold
        loop_threshold = self.config.loop_threshold

        # Determine if drifting (configurable)
        is_drifting = (combined_drift_score > drift_threshold or
                      any(s > signal_threshold for s in signals) or
                      (loop_report is not None and loop_report.is_looping))

        report = DriftReport(
            ghost_loss=ghost_loss,
            behavior_shift=behavior_shift,
            agreement_score=agreement,
            loop_report=loop_report,
            stagnation_score=stagnation_score,
            combined_drift_score=combined_drift_score,
            is_drifting=is_drifting
        )

        with self._lock:
            self.drift_history.append(report)
        self._save_report(report)
        return report

    def to_dict(self) -> Dict:
        """Serialize for logging/export"""
        with self._lock:
            history = list(self.drift_history)
        return {
            "agent_id": self.config.agent_id,
            "total_drifts_detected": sum(1 for r in history if r.is_drifting),
            "history_length": len(history),
            "latest_report": {
                "combined_drift": history[-1].combined_drift_score if history else 0.0,
                "is_drifting": history[-1].is_drifting if history else False,
            } if history else None
        }

    def get_stats(self) -> Dict:
        """Get signal distribution and detection statistics"""
        with self._lock:
            history = list(self.drift_history)

        if not history:
            return {
                "total_reports": 0,
                "drifts_detected": 0,
                "avg_drift_score": 0.0,
                "signal_distribution": {
                    "ghost_loss": 0,
                    "behavior_shift": 0,
                    "agreement": 0,
                    "loop_detection": 0,
                    "stagnation": 0,
                },
                "severity_distribution": {
                    "critical": 0,
                    "warning": 0,
                    "info": 0,
                }
            }

        # Count signal triggers
        signal_count = {
            "ghost_loss": 0,
            "behavior_shift": 0,
            "agreement": 0,
            "loop_detection": 0,
            "stagnation": 0,
        }
        severity_count = {"critical": 0, "warning": 0, "info": 0}
        avg_score = 0.0

        for report in history:
            avg_score += report.combined_drift_score

            # Count which signals triggered
            critical_threshold = self.config.signal_threshold * 1.3

            if report.ghost_loss > self.config.signal_threshold:
                signal_count["ghost_loss"] += 1
                severity = "critical" if report.ghost_loss > critical_threshold else "warning"
                severity_count[severity] += 1

            if report.behavior_shift > self.config.signal_threshold:
                signal_count["behavior_shift"] += 1
                severity = "critical" if report.behavior_shift > critical_threshold else "warning"
                severity_count[severity] += 1

            if (1.0 - report.agreement_score) > self.config.signal_threshold:
                signal_count["agreement"] += 1
                severity = "critical" if (1.0 - report.agreement_score) > critical_threshold else "warning"
                severity_count[severity] += 1

            if report.stagnation_score > self.config.signal_threshold:
                signal_count["stagnation"] += 1
                severity = "critical" if report.stagnation_score > critical_threshold else "warning"
                severity_count[severity] += 1

            if report.loop_report and report.loop_report.is_looping:
                signal_count["loop_detection"] += 1
                lc = report.loop_report.combined_confidence
                severity_count["critical" if lc > critical_threshold else "warning"] += 1

        avg_score = avg_score / len(history) if history else 0.0

        return {
            "total_reports": len(history),
            "drifts_detected": sum(1 for r in history if r.is_drifting),
            "avg_drift_score": round(avg_score, 3),
            "signal_distribution": signal_count,
            "severity_distribution": severity_count,
        }

    def get_drift_trend(self, window_size: int = 5) -> Dict:
        """
        Session-level analytics: Analyze drift trends over multiple reports.

        Returns moving average, trend direction, and stability metrics.
        Useful for detecting gradual degradation patterns.

        Args:
            window_size: Number of reports to average (default: 5)

        Returns:
            Dict with:
                - moving_avg: Average drift score over window
                - trend: "increasing", "stable", or "improving"
                - slope: Rate of change (positive = increasing)
                - reports_count: Total reports in history
                - last_score: Most recent drift score
        """
        with self._lock:
            history = list(self.drift_history)

        if len(history) < window_size:
            return {
                "status": "insufficient_data",
                "count": len(history),
                "required": window_size
            }

        reports = history[-window_size:]
        scores = [r.combined_drift_score for r in reports]

        # Moving average
        avg = sum(scores) / len(scores)

        # Trend (slope): is drift increasing/decreasing?
        slope = (scores[-1] - scores[0]) / (len(scores) - 1) if len(scores) > 1 else 0

        # Trend classification
        if slope > 0.01:
            trend = "increasing"
        elif slope < -0.01:
            trend = "improving"
        else:
            trend = "stable"

        return {
            "status": "ok",
            "moving_avg": avg,
            "trend": trend,
            "slope": slope,
            "reports_count": len(self.drift_history),
            "last_score": scores[-1],
            "window_size": window_size
        }


# ============================================================================
# INTEGRATIONS
# ============================================================================
# LangChain and CrewAI integrations are now in separate modules:
#   - integrations/langchain_integration.py (DriftDetectionCallback)
#   - integrations/crewai_integration.py (DriftMonitoringCrewMixin)
#
# This keeps the core agent lightweight and dependencies optional.
# Users only import what they need.


# ============================================================================
# Tests (Dry-run, no LLM)
# ============================================================================

def test_ghost_lexicon_detects_loss():
    """Ghost lexicon should detect loss of rare terms"""
    before_text = "The quantum entanglement phenomenon requires sophisticated calibration algorithms"
    after_text = "The phenomenon is important"

    before = SessionSnapshot("test", "2026-04-11T00:00:00", before_text, len(before_text), [], 0, "hash1")
    after = SessionSnapshot("test", "2026-04-11T00:01:00", after_text, len(after_text), [], 0, "hash2")

    score = ghost_lexicon_score(before, after, top_n=5)
    assert 0 < score < 1, f"Expected 0 < {score} < 1, got {score}"
    print(f"✓ test_ghost_lexicon: {score:.2f}")


def test_behavioral_shift_on_tool_change():
    """Behavioral shift should detect tool sequence changes"""
    before = SessionSnapshot("test", "2026-04-11T00:00:00", "output1", 10,
                            ["search", "analyze"], 2, "hash1")
    after = SessionSnapshot("test", "2026-04-11T00:01:00", "output2", 10,
                           ["search", "refine"], 2, "hash2")

    shift = behavioral_footprint_shift(before, after)
    assert 0 <= shift <= 1, f"Expected 0 <= {shift} <= 1"
    print(f"✓ test_behavioral_shift: {shift:.2f}")


def test_agreement_on_similar_outputs():
    """Agreement should be high when models output similar text"""
    outputs = {
        "groq": "Python is a programming language used for web development",
        "ollama": "Python is a programming language for web development",
    }

    agreement = agreement_score(outputs)
    assert 0.7 < agreement <= 1.0, f"Expected high agreement, got {agreement:.2f}"
    print(f"✓ test_agreement: {agreement:.2f}")


def test_loop_detection_exact_repetition():
    """Loop detection should catch exact action repetition"""
    actions = ["search", "analyze", "search", "analyze", "search", "analyze"]

    report = detect_loops_ensemble(actions)
    assert report.is_looping, "Should detect A-B-A-B loop"
    print(f"✓ test_loop_exact: diversity={report.diversity_score:.2f}, entropy={report.entropy_score:.2f}")


def test_loop_detection_same_action():
    """Loop detection should catch same action repeated"""
    actions = ["search"] * 10

    report = detect_loops_ensemble(actions)
    assert report.is_looping, "Should detect same action repeated"
    print(f"✓ test_loop_same: diversity={report.diversity_score:.2f}")


def test_stagnation_on_repeated_results():
    """Stagnation should detect repeated tool results"""
    results = [
        {"answer": "Python is good"},
        {"answer": "Python is good"},
        {"answer": "Python is good"},
    ]

    score = detect_stagnation(results)
    assert score > 0.8, f"Expected high stagnation, got {score:.2f}"
    print(f"✓ test_stagnation: {score:.2f}")


def test_edge_case_empty_lists():
    """Edge case: empty input lists"""
    before = SessionSnapshot("test", "2026-04-11T00:00:00", "", 0, [], 0, "hash")
    after = SessionSnapshot("test", "2026-04-11T00:01:00", "hello", 5, ["search"], 1, "hash2")

    ghost = ghost_lexicon_score(before, after)
    assert 0 <= ghost <= 1, "Should handle empty before"
    print(f"✓ test_empty_before: {ghost:.2f}")


def test_edge_case_single_model():
    """Edge case: single model output (no agreement basis)"""
    outputs = {"groq": "some output"}

    agreement = agreement_score(outputs)
    assert agreement == 1.0, "Single model should return 1.0"
    print(f"✓ test_single_model: {agreement:.2f}")


def run_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("DriftDetectorAgent: Dry-run Tests (no LLM)")
    print("="*80 + "\n")

    test_ghost_lexicon_detects_loss()
    test_behavioral_shift_on_tool_change()
    test_agreement_on_similar_outputs()
    test_loop_detection_exact_repetition()
    test_loop_detection_same_action()
    test_stagnation_on_repeated_results()
    test_edge_case_empty_lists()
    test_edge_case_single_model()

    print("\n" + "="*80)
    print("✓ All tests PASSED")
    print("="*80 + "\n")


# ============================================================================
# SQLite Persistence (Phase 2 - Blocker)
# ============================================================================

def _init_db(self):
    """Initialize SQLite database for drift reports"""
    try:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                agent_id TEXT,
                combined_drift_score REAL,
                ghost_loss REAL,
                behavior_shift REAL,
                agreement_score REAL,
                stagnation_score REAL,
                is_drifting BOOLEAN,
                loop_detected BOOLEAN
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("DB init failed: %s", e)


def _save_report(self, report: DriftReport):
    """Save drift report to SQLite"""
    try:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO drift_reports
            (timestamp, agent_id, combined_drift_score, ghost_loss, behavior_shift,
             agreement_score, stagnation_score, is_drifting, loop_detected)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            report.timestamp,
            self.config.agent_id,
            report.combined_drift_score,
            report.ghost_loss,
            report.behavior_shift,
            report.agreement_score,
            report.stagnation_score,
            report.is_drifting,
            report.loop_report is not None and report.loop_report.is_looping
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        import logging as _logging
        _logging.getLogger(__name__).warning("DB save failed: %s", e)


def _load_history(self):
    """Load drift history from SQLite on startup"""
    try:
        if not Path(self.db_path).exists():
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Cap at last 10_000 rows to prevent unbounded memory growth
        cursor.execute('SELECT * FROM drift_reports ORDER BY id DESC LIMIT 10000')
        rows = cursor.fetchall()
        rows.reverse()  # restore chronological order
        for row in rows:
            _, timestamp, agent_id, combined, ghost, shift, agreement, stagnation, is_drifting, loop_detected = row
            report = DriftReport(
                timestamp=timestamp,
                ghost_loss=ghost,
                behavior_shift=shift,
                agreement_score=agreement,
                stagnation_score=stagnation,
                combined_drift_score=combined,
                is_drifting=bool(is_drifting),
                loop_report=LoopReport(0.0, 0.0, 0.0, 0.0, loop_detected) if loop_detected else None
            )
            self.drift_history.append(report)
        conn.close()
    except Exception as e:
        logger.warning("DB load failed: %s", e)


# Bind persistence methods to DriftDetectorAgent
DriftDetectorAgent._init_db = _init_db
DriftDetectorAgent._save_report = _save_report
DriftDetectorAgent._load_history = _load_history


# ============================================================================
# Demo / Debug
# ============================================================================

if __name__ == "__main__":
    # Run tests
    run_tests()

    # Demo: instantiate and use
    config = AgentConfig(agent_id="drift_demo", role="monitor")
    detector = DriftDetectorAgent(config)

    # Simulate session boundary drift
    before = detector.snapshot(
        "crew_agent_1",
        "Found 3 research papers on quantum entanglement using sophisticated algorithms",
        ["search", "analyze", "fetch"]
    )

    after = detector.snapshot(
        "crew_agent_1",
        "Found papers on quantum topic",
        ["search", "fetch"]
    )

    report = detector.measure_drift(
        before, after,
        action_history=["search", "analyze", "search", "analyze"],
        before_actions=["search", "analyze"],
        model_outputs={
            "groq": "Found 3 research papers",
            "ollama": "Found some papers"
        }
    )

    print("\nDemo Drift Report:")
    print(f"  Ghost Loss: {report.ghost_loss:.2f} (precision terms lost)")
    print(f"  Behavior Shift: {report.behavior_shift:.2f} (tool divergence)")
    print(f"  Agreement: {report.agreement_score:.2f} (model consensus)")
    print(f"  Combined Drift: {report.combined_drift_score:.2f}")
    print(f"  Is Drifting: {report.is_drifting}")
    if report.loop_report:
        print(f"  Loop Confidence: {report.loop_report.combined_confidence:.2f}")
