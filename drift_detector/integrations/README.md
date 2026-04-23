# DriftDetector Integrations

**Adapters for LangChain & CrewAI** - Connect your agent framework to drift detection

| Framework | File | Status | Pattern |
|-----------|------|--------|---------|
| **LangChain** | `langchain.py` | ✅ Working | BaseCallbackHandler |
| **CrewAI** | `crewai.py` | ✅ Working | EventListener |

---

## LangChain Integration

### DriftDetectionCallback

```python
from drift_detector.integrations import DriftDetectionCallback
from langchain.agents import AgentExecutor
from langchain.tools import tool

# Define tools
@tool
def search_docs(query: str) -> str:
    """Search documentation"""
    return "Results..."

@tool  
def analyze(data: str) -> str:
    """Analyze data"""
    return "Analysis..."

tools = [search_docs, analyze]

# Create agent
agent = ...
executor = AgentExecutor(agent=agent, tools=tools)

# Attach drift detection
callback = DriftDetectionCallback()
result = executor.run(
    "Your task here",
    callbacks=[callback]
)

# Get drift report
report = callback.get_drift_report()
print(f"Total drift events: {report['total_drifts']}")
for h in report['history']:
    print(f"  Drift: {h['combined_drift']:.3f}, Loop: {h['loop_detected']}")
```

### How It Works

1. **Callback hooks into agent:**
   - `on_agent_action()` - captures tool calls
   - `on_agent_finish()` - captures final output

2. **Takes snapshots:**
   - Before: agent starts thinking
   - After: agent produces output

3. **Measures drift:**
   - Compares vocabulary, tools, output patterns
   - Generates DriftReport with 5 signals

4. **Stores in database:**
   - Auto-saves to SQLite
   - Accessible via dashboard

---

## CrewAI Integration

### DriftDetectorEventListener

```python
from drift_detector.integrations import DriftDetectorEventListener
from crewai import Crew, Agent, Task

# Create crew (your agents + tasks)
crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2]
)

# Attach drift listener
listener = DriftDetectorEventListener()
crew.add_event_listener(listener)

# Run crew
result = crew.kickoff()

# Get drift report
report = listener.get_drift_report()
print(f"Tasks with drift: {report['total_drifts']}")
for h in report['history']:
    print(f"  Task: {h['timestamp']}, Drift: {h['combined_drift']:.3f}")
```

### CrewAI Event Lifecycle

```
Task Starts
  ↓
TaskStartedEvent
  → DriftDetectorEventListener._handle_task_started()
  → Take "before" snapshot

Task Executes
  ↓
ToolCallEvent (multiple)
  → DriftDetectorEventListener._handle_tool_call()
  → Track tool usage for loop detection

Task Completes
  ↓
TaskCompletedEvent
  → DriftDetectorEventListener._handle_task_completed()
  → Take "after" snapshot
  → Measure drift between snapshots

(or)

TaskFailedEvent
  → DriftDetectorEventListener._handle_task_failed()
  → Clean up tracking
```

### DriftMonitoringCrewMixin

Simpler alternative - subclass your Crew:

```python
from drift_detector.integrations import DriftMonitoringCrewMixin
from crewai import Crew

class MyCrewWithDrift(Crew, DriftMonitoringCrewMixin):
    pass

crew = MyCrewWithDrift()
listener = crew.enable_drift_monitoring()

result = crew.kickoff()
report = crew.get_drift_report()
```

---

## API Compatibility

### LangChain

```
✅ 0.3.x
✅ 0.2.x
✅ 0.1.x (older)

Min version: 0.1.0
Max version: 0.3.x
```

### CrewAI

```
✅ 0.5.x (Apr 2026)
✅ 0.4.x
⚠️ 0.3.x (older, verify)

Min version: 0.3.0
Recommended: 0.5.x
```

---

## Usage Examples

### Example 1: Simple LangChain Chain

```python
from drift_detector.integrations import DriftDetectionCallback
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

llm = ChatOpenAI(model="gpt-4")

# Create agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# Run with drift detection
callback = DriftDetectionCallback()
result = agent.run(
    "Research AI trends and summarize",
    callbacks=[callback]
)

# Check results
report = callback.get_drift_report()
if report['total_drifts'] > 5:
    print("⚠️ High drift - agent behavior unstable")
```

### Example 2: CrewAI with Multiple Agents

```python
from crewai import Crew, Agent, Task
from drift_detector.integrations import DriftDetectorEventListener

# Define agents
researcher = Agent(
    role="researcher",
    goal="Find relevant information",
    tools=[search_tool, analyze_tool]
)

writer = Agent(
    role="writer",
    goal="Summarize findings",
    tools=[write_tool]
)

# Define tasks
research_task = Task(
    description="Research AI safety",
    agent=researcher
)

write_task = Task(
    description="Write summary",
    agent=writer
)

# Create crew with drift monitoring
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task]
)

listener = DriftDetectorEventListener()
crew.add_event_listener(listener)

# Run
result = crew.kickoff()

# Analyze drift per task
for entry in listener.get_drift_report()['history']:
    if entry['is_drifting']:
        print(f"⚠️ Drift at {entry['timestamp']}: {entry['combined_drift']:.3f}")
```

### Example 3: Detecting Specific Signals

```python
callback = DriftDetectionCallback()

# Run agent
executor.run("task", callbacks=[callback])

report = callback.get_drift_report()
for h in report['history']:
    # Check for vocabulary loss
    if h['ghost_loss'] > 0.5:
        print(f"⚠️ Agent output becoming less detailed")
    
    # Check for tool changes
    if h['behavior_shift'] > 0.6:
        print(f"⚠️ Agent changing tool usage patterns")
    
    # Check for repetition
    if h['stagnation'] > 0.8:
        print(f"⚠️ Agent repeating same output")
    
    # Check for loops
    if h['loop_detected']:
        print(f"🔴 INFINITE LOOP DETECTED!")
```

---

## Signal Interpretation

### Ghost Lexicon (Vocabulary Loss)

```
0.0-0.2   ✅ Normal (vocabulary stable)
0.2-0.5   ⚠️ Slight loss (less detailed)
0.5-0.8   🔴 Significant loss (degraded precision)
0.8-1.0   🔴🔴 Critical (minimal vocabulary)
```

### Behavior Shift (Tool Changes)

```
0.0-0.2   ✅ Same tools being used
0.2-0.5   ⚠️ Some tool changes
0.5-0.8   🔴 Major tool pattern shift
0.8-1.0   🔴🔴 Completely different tools
```

### Stagnation (Output Repetition)

```
0.0-0.3   ✅ Outputs are different
0.3-0.6   ⚠️ Some similarity
0.6-0.9   🔴 High repetition
0.9-1.0   🔴🔴 Identical outputs
```

---

## Troubleshooting

### "Drift Detection Not Triggering"

```python
# Check 1: Is callback registered?
executor.run("task", callbacks=[callback])  # ✅

# Check 2: Are snapshots being captured?
print(len(callback.detector.drift_history))  # Should > 0

# Check 3: Are thresholds too high?
callback.detector.config.drift_threshold = 0.2  # Lower it
```

### "Too Much Noise in Reports"

```python
# Increase thresholds
config = AgentConfig(
    drift_threshold=0.6,      # Only alert on high drift
    signal_threshold=0.7      # Strict signal detection
)
```

### "Missing Some Signals"

```python
# Check report fields
report = callback.get_drift_report()
for h in report['history']:
    print(h.keys())  # Should have: ghost_loss, behavior_shift, agreement, stagnation, loop_detected
```

---

## Performance Impact

```
LangChain Callback:
  - Overhead: <10ms per agent step
  - Memory: ~1KB per snapshot
  - DB: <20ms per save

CrewAI Listener:
  - Overhead: <5ms per event
  - Memory: ~1KB per task snapshot
  - DB: <20ms per save
```

---

## Testing

```bash
# Run the full test suite
python3 -m pytest tests/ -v

# Run examples (agnostic — no cloud keys required)
python3 examples/test_langchain_chain_with_drift.py
python3 examples/test_crewai_with_drift.py

# Run examples that hit real LLM providers (requires keys)
python3 examples/test_langchain_chain_real_groq.py
```

---

## Next Steps

- See [examples/README.md](../../examples/README.md) for working examples
- See [core/README.md](../core/README.md) for API details
- See [docs/README.md](../../docs/README.md) for detection theory & signal math

---

**Ready to integrate!** Pick LangChain or CrewAI and start detecting drift.
