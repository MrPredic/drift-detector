# DriftDetector

**Behavioral Drift Detection for LLM Agents — LangChain · CrewAI · Any Python Agent**

> *"My LangChain agent is stuck in a loop."*
> *"My CrewAI crew keeps repeating the same output."*
> *"My AI agent started using weird vocabulary after step 3."*
> *"How do I detect when my LLM agent changes behavior mid-session?"*

**DriftDetector solves exactly this.** It monitors your LLM agents for behavioral drift — vocabulary shrinkage, tool pattern changes, repetitive loops, and output stagnation — and alerts you before they break production.

5 detection signals · <10ms overhead · 3-line integration · MIT License

![Status](https://img.shields.io/badge/status-production--ready-brightgreen)
![Tests](https://img.shields.io/badge/tests-159%2F159%20pass-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Security](https://img.shields.io/badge/security-audited-blue)

---

## Install

```bash
pip install drift-detector-agent
```

---

## Quick Start (3 lines)

```python
from drift_detector.core import DriftDetectorAgent

detector = DriftDetectorAgent()
report = detector.measure_drift(snapshot_before, snapshot_after)
print(f"Is drifting? {report.is_drifting} (score: {report.combined_drift_score:.3f})")
```

---

## Setup (Interactive)

**Guided Installation with Setup Wizard:**

```bash
python setup_wizard.py
```

Wizard guides you through:
1. **Choose your setup:** Core, Core+UI, Core+LangChain, Core+CrewAI, All, or Custom
2. **Generate .env:** Auto-creates config with all API providers listed
3. **Health check:** Verifies installation

Or **skip wizard:**

```bash
pip install drift-detector-agent[ui]          # Core + Web Dashboard
pip install drift-detector-agent[langchain]   # Core + LangChain
pip install drift-detector-agent[all]         # Everything
```

---

## Web Dashboard (UI)

**Monitor drift in real-time with web interface:**

```bash
python -m drift_detector.ui.server
# Visit: http://127.0.0.1:8000
```

Features:
- ✅ Measure drift interactively
- ✅ View real-time trends (moving average, slope)
- ✅ Historical reports chart
- ✅ Session-level analytics
- ✅ REST API: /api/health, /api/config, /api/drift, /api/chain

---

## What It Detects

### 5 Drift Signals

1. **Ghost Lexicon** - Vocabulary shrinkage (lost precision)
2. **Behavioral Shift** - Tool usage changes
3. **Agreement Score** - Multi-model consensus divergence
4. **Loop Detection** - Repetitive action sequences
5. **Stagnation** - Identical output repetition

---

## LangChain Integration

```python
from drift_detector.integrations import DriftDetectionCallback
from langchain.agents import AgentExecutor

callback = DriftDetectionCallback()
result = executor.run(
    "Your task here",
    callbacks=[callback]
)

report = callback.get_drift_report()
if report['total_drifts'] > 5:
    print("⚠️ Agent behavior unstable!")
```

---

## CrewAI Integration

```python
from drift_detector.integrations import DriftDetectorEventListener
from crewai import Crew

listener = DriftDetectorEventListener()
crew.add_event_listener(listener)

result = crew.kickoff()
report = listener.get_drift_report()

for entry in report['history']:
    if entry['is_drifting']:
        print(f"Drift detected at {entry['timestamp']}")
```

---

## Multi-LLM Support

Works with **5 LLM providers** (all tested & verified Apr 2026):

| Provider | Model | Status | Rate Limit |
|----------|-------|--------|-----------|
| **Groq** | llama-3.3-70b-versatile | ✅ Working | 30 RPM, 12K tok/min |
| **Cerebras** | llama3.1-8b | ✅ Working | 30 RPM, 60K tok/min |
| **Gemini** | gemini-2.5-flash | ✅ Working | 5-15 RPM, 250K tok/min |
| **OpenRouter** | llama-3.3-70b-instruct | ✅ Working | 20 RPM free |
| **Ollama** | llama2 (local) | ✅ Optional | Unlimited |

**Just add your API keys to `.env` - done!**

---

## Setup

### Requirements
- Python 3.8 or higher
- pip (package installer)
- At least one LLM API key (Groq free tier recommended)

### 1. Install Package

```bash
pip install drift-detector-agent
```

Verify installation:
```bash
python3 -c "from drift_detector.core import DriftDetectorAgent; print('✅ Installed')"
```

### 2. Get API Key

**Groq (Recommended - Free)**
1. Go to https://console.groq.com/keys
2. Create API key
3. Copy the key (starts with `gsk_`)

### 3. Configure API Key

**Option A: Environment variable (recommended)**
```bash
export GROQ_API_KEY="your_key_here"
python3 your_script.py
```

**Option B: .env file**
```bash
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=your_key_here
python3 your_script.py
```

### 4. Run Your First Detection

```python
from drift_detector.core import DriftDetectorAgent

# Initialize detector
detector = DriftDetectorAgent(agent_id="my_agent")

# Create snapshots
snap1 = detector.snapshot(
    agent_id="step1",
    response_text="This is detailed output with many tokens",
    tool_calls=["search", "analyze"]
)

snap2 = detector.snapshot(
    agent_id="step2",
    response_text="Brief output",
    tool_calls=["summarize"]
)

# Measure drift
report = detector.measure_drift(snap1, snap2)
print(f"Drift: {report.combined_drift_score:.3f}")
print(f"Is drifting: {report.is_drifting}")
```

### 5. Use with LangChain or CrewAI

See [examples/](examples/) for working code with real LLMs.

---

## Documentation

- **[API & Signals Reference](docs/README.md)** - Detection theory, signal math, API surface
- **[Integration Guides](drift_detector/integrations/README.md)** - LangChain & CrewAI patterns
- **[Examples](examples/README.md)** - Working code samples
- **[Security Policy](SECURITY.md)** - Deployment hardening & vulnerability reporting
- **[Changelog](CHANGELOG.md)** - Release history

---

## Performance

| Operation | Time |
|-----------|------|
| Create snapshot | <5ms |
| Measure drift | <10ms |
| Save to database | <20ms |

**Overhead:** <15ms per detection cycle. Safe for production.

---

## Real Example

**Task:** Multi-step research pipeline (4 steps)

```
Step 1: Research detailed analysis
  └─ Vocabulary: 1,500+ tokens, detailed
  
Step 2: Analyze findings
  └─ Vocabulary: 800+ tokens, synthesis
  └─ Drift detected: 0.304 (Ghost Lexicon 0.180)

Step 3: Summarize
  └─ Vocabulary: 260 tokens, brief
  └─ Drift detected: 0.433 (Ghost Lexicon 0.780) ⚠️ STRONG

Step 4: Final report
  └─ Vocabulary: 300 tokens, recovery
  └─ Drift detected: 0.267 (normal)
```

✅ **System correctly identified vocabulary collapse at step 3**

---

## Features

✅ **5 Drift Signals** - Detects vocabulary loss, tool changes, loops, stagnation, consensus divergence

✅ **LangChain Integration** - One callback for auto-detection

✅ **CrewAI Integration** - Event listener for multi-agent tasks

✅ **Multi-LLM Support** - Groq, Cerebras, Gemini, OpenRouter, Ollama

✅ **SQLite Persistence** - Auto-saves all reports

✅ **Real-Time Monitoring** - FastAPI dashboard included

✅ **Production Ready** - 159/159 tests pass, security audited

✅ **<10ms Overhead** - Safe for real-time systems

---

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=drift_detector
```

**Status:** 159/159 tests PASS ✅

---

## Common Issues

### "No API key found"
```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### "Model not available"
- Use the optional `LLMRouter` (auto-selects the fastest provider that has an API key set)
- Requires the `langchain` extra: `pip install drift-detector-agent[langchain]`

### "Drift not detected"
- Lower `drift_threshold` in config (default: 0.4)
- See [docs/README.md](docs/README.md) for signal interpretation

---

## Status & Roadmap

**v2.0.0 (Current)** - Open-source release
- ✅ 5 drift signals
- ✅ LangChain + CrewAI integrations
- ✅ Multi-LLM support (Groq, Cerebras, Gemini, OpenRouter, Ollama)
- ✅ SQLite persistence
- ✅ FastAPI dashboard with pinned SRI-integrity Chart.js
- ✅ 159/159 tests passing
- ✅ Security-audited (see `SECURITY.md` and `SECURITY_AUDIT_REPORT.md`)

**v2.1 (Planned)**
- Distributed persistence (PostgreSQL)
- Custom signal weights
- Advanced loop detection
- WebSocket alerts
- Multi-detector orchestration

---

## License

MIT License - See [LICENSE](LICENSE)

---

## Support & Issues

**Found a bug?** → [GitHub Issues](https://github.com/MrPredic/drift-detector/issues)

**Question?** → [GitHub Discussions](https://github.com/MrPredic/drift-detector/discussions)

**Need support?** → Open an issue or discussion on GitHub

---

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests
4. Submit a pull request

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'drift_detector'"
```bash
# Install the package
pip install drift-detector-agent

# Or verify installation
pip list | grep drift-detector-agent
```

### "ImportError: No module named 'langchain'"
LangChain is optional. Install if using LangChain:
```bash
pip install langchain langchain-openai
```

### "API key not found"
Make sure your API key is set:
```bash
# Check if GROQ_API_KEY is set
echo $GROQ_API_KEY

# If empty, set it
export GROQ_API_KEY="your_key_here"
```

### "No module named 'groq'"
This is handled gracefully - DriftDetector works without Groq. But for examples, install:
```bash
pip install langchain-groq
```

### ".env file contains my API key - is it safe?"
⚠️ **IMPORTANT**: Never commit `.env` to version control!

The `.gitignore` already excludes `.env` files. Verify:
```bash
git status .env
# Should show: .env is ignored
```

---

**Ready to detect agent drift?**

```bash
pip install drift-detector-agent
```

[Quick Start Guide](README.md#setup) • [View Examples](examples/) • [Read Docs](docs/) • [GitHub](https://github.com/MrPredic/drift-detector)
