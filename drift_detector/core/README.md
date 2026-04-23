# DriftDetector Core

**Main detection engine** - Behavioral drift monitoring for LLM agents

- **Files:** `drift_detector_agent.py`, `session_storage.py`
- **Status:** ✅ Production Ready
- **Tests:** see `tests/` (run `pytest tests/ -v`)

---

## What It Does

Monitors LLM agent behavior and detects when outputs/actions drift from expected patterns.

### 5 Drift Signals

1. **Ghost Lexicon** (Vocabulary Loss)
   - Detects when agent uses simpler/shorter vocabulary
   - Indicates lost precision or degraded understanding
   - Range: 0.0-1.0 (0=no loss, 1=complete loss)

2. **Behavioral Shift** (Tool Changes)
   - Tracks changes in tool usage patterns
   - Detects when agent stops using certain tools
   - Range: 0.0-1.0 (0=same tools, 1=completely different)

3. **Agreement Score** (Multi-Model Consensus)
   - Measures how similar outputs are across different LLMs
   - Single model: 1.0 (perfect agreement with itself)
   - Multi-model: 0.0-1.0 (how much they agree)

4. **Loop Detection** (Repetitive Actions)
   - Identifies when agent repeats same actions infinitely
   - Uses pattern ensemble (diversity + entropy analysis)
   - Triggers on low diversity + low entropy

5. **Stagnation** (Output Repetition)
   - Detects identical or near-identical outputs
   - Measures token overlap (Jaccard similarity)
   - Range: 0.0-1.0 (0=different, 1=identical)

### Combined Drift Score

```
combined_drift = (ghost_loss + behavior_shift + stagnation) / 3
                 (normalized)

is_drifting = combined_drift > drift_threshold (default 0.4)
```

---

## Core Classes

### DriftDetectorAgent

```python
from drift_detector.core import DriftDetectorAgent
from agents.base_agent import AgentConfig

# Create detector
config = AgentConfig(
    agent_id="my_agent",
    role="monitor",
    drift_threshold=0.4,      # Alert when drift > this
    signal_threshold=0.5       # Each signal must exceed this
)
detector = DriftDetectorAgent(config)

# Take snapshot (vocabulary + tools)
snap = detector.snapshot(
    agent_id="task_1",
    response_text="The analysis reveals...",
    tool_calls=["search", "analyze", "report"]
)

# Measure drift between two snapshots
report = detector.measure_drift(snap_before, snap_after)

# Access signals
print(f"Ghost Loss: {report.ghost_loss:.3f}")
print(f"Behavior Shift: {report.behavior_shift:.3f}")
print(f"Agreement: {report.agreement_score:.3f}")
print(f"Loop Detected: {report.loop_report.is_looping}")
print(f"Stagnation: {report.stagnation_score:.3f}")
```

### SessionSnapshot

```python
@dataclass
class SessionSnapshot:
    agent_id: str
    timestamp: str
    response_text: str
    tool_calls: List[str]
    vocabulary: Set[str]
    response_tokens: int
```

### DriftReport

```python
@dataclass
class DriftReport:
    timestamp: str
    combined_drift_score: float
    ghost_loss: float
    behavior_shift: float
    agreement_score: float
    stagnation_score: float
    is_drifting: bool
    loop_report: Optional[LoopReport]
```

---

## Usage Examples

### Example 1: Simple Task Monitoring

```python
detector = DriftDetectorAgent()

# Task step 1: Research
snap1 = detector.snapshot(
    "research",
    "Found 5 academic papers on neural networks...",
    ["search_papers"]
)

# Task step 2: Summarize
snap2 = detector.snapshot(
    "summarize",
    "Summary here",
    ["report"]
)

report = detector.measure_drift(snap1, snap2)
if report.is_drifting:
    print("⚠️ Agent drifting!")
```

### Example 2: Multi-Step Chain

```python
detector = DriftDetectorAgent()
snapshots = []

# Monitor 5 steps
steps = [
    ("detailed analysis with metrics", ["research", "analyze"]),
    ("comparison of approaches", ["research", "compare"]),
    ("brief summary", ["summarize"]),
    ("final report", ["report"]),
    ("executive summary", ["summarize"])
]

for i, (text, tools) in enumerate(steps):
    snap = detector.snapshot(f"step_{i}", text, tools)
    snapshots.append(snap)
    
    if i > 0:
        # Check drift from previous step
        report = detector.measure_drift(snapshots[i-1], snap)
        if report.stagnation_score > 0.8:
            print(f"Step {i}: Output repetition detected!")
```

### Example 3: Loop Detection

```python
detector = DriftDetectorAgent()

# Agent repeating same actions
tools_history = [
    ["search", "analyze"],
    ["search", "analyze"],    # Same
    ["search", "analyze"],    # Same again
    ["search", "analyze"],    # LOOP!
]

for i, tools in enumerate(tools_history):
    snap = detector.snapshot(f"attempt_{i}", f"Output {i}", tools)
    if i > 0:
        report = detector.measure_drift(snapshots[i-1], snap)
        if report.loop_report and report.loop_report.is_looping:
            print(f"Infinite loop detected at step {i}!")
```

---

## Configuration

### Default Thresholds

```python
# config/base_agent.py
drift_threshold = 0.4       # Drift alert threshold
signal_threshold = 0.5       # Individual signal threshold
loop_threshold = 0.6         # Loop detection threshold
```

### Custom Configuration

```python
config = AgentConfig(
    agent_id="custom_agent",
    drift_threshold=0.3,      # More sensitive
    signal_threshold=0.6,     # Stricter signals
    loop_threshold=0.5        # Catch loops sooner
)

detector = DriftDetectorAgent(config)
```

---

## Persistence (SQLite)

```python
# Automatic persistence
detector = DriftDetectorAgent()

snap1 = detector.snapshot(...)
snap2 = detector.snapshot(...)
report = detector.measure_drift(snap1, snap2)
# ✅ Automatically saved to drift_detector.db

# Access history
history = detector.drift_history
print(f"Total reports: {len(history)}")

# Get statistics
stats = detector.get_stats()
print(stats)
```

---

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| snapshot() | <5ms | Create pre/post snapshot |
| measure_drift() | <10ms | Compare two snapshots |
| _save_report() | <20ms | DB insert |
| get_stats() | <50ms | Calculate aggregates |

---

## Testing

```bash
# Run all tests
python3 -m pytest tests/ -v

# Core signal tests
python3 -m pytest tests/test_5_signals_comprehensive.py -v
python3 -m pytest tests/test_domain_aware_signals.py -v

# With coverage (requires pytest-cov)
python3 -m pytest tests/ --cov=drift_detector.core
```

---

## Architecture

```
DriftDetectorAgent (main class)
├── snapshot() → SessionSnapshot
├── measure_drift() → DriftReport
├── _calculate_ghost_lexicon() → float
├── _calculate_behavior_shift() → float
├── _calculate_agreement_score() → float
├── _detect_loops() → LoopReport
├── _calculate_stagnation() → float
├── _save_report() → None (SQLite)
├── _load_history() → None (SQLite)
└── get_stats() → dict
```

---

## Known Limitations (v2.0)

- ⚠️ Not fully thread-safe (single event loop recommended)
- ⚠️ SQLite only (no distributed persistence)
- ⚠️ Single detector per API instance
- ✅ Scales to 10+ concurrent tasks

---

## Next Steps

- See [integrations/README.md](../integrations/README.md) for LangChain/CrewAI
- See [examples/README.md](../../examples/README.md) for real usage
- See [docs/README.md](../../docs/README.md) for detection theory

---

**Ready to integrate!** Use with LangChain or CrewAI via integrations module.
