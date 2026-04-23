# DriftDetector Examples

Runnable demonstrations for LangChain, CrewAI, and framework-agnostic use.

## Files

| File | Framework | What it shows |
|------|-----------|---------------|
| `example_coding_agent.py` | Agnostic | Tracking drift across iterations of a coding agent |
| `example_degradation_monitoring.py` | Agnostic | Using `is_degradation=True` to detect capability regressions |
| `example_medical_drift_detection.py` | Agnostic | Domain-specific drift (medical terminology / ghost lexicon) |
| `stress_test_drift_detector.py` | Agnostic | High-volume load & stability test |
| `test_langchain_chain_with_drift.py` | LangChain | Attach `DriftDetectionCallback` to a chain (no API key needed) |
| `test_langchain_chain_real_groq.py` | LangChain | Real multi-step Groq calls (requires `GROQ_API_KEY`) |
| `test_crewai_with_drift.py` | CrewAI | `DriftDetectorEventListener` on a Crew run |

## Quick start

### 1. Framework-agnostic (no dependencies beyond the core package)

```bash
python3 examples/example_coding_agent.py
python3 examples/example_degradation_monitoring.py
python3 examples/example_medical_drift_detection.py
```

These are the safest starting point — they need only the core package
and do not hit any external API.

### 2. LangChain integration

```bash
pip install drift-detector-agent[langchain]
python3 examples/test_langchain_chain_with_drift.py
```

Shows:
- How to attach `DriftDetectionCallback` to a LangChain chain
- Per-step drift reports
- Signal interpretation (Ghost Lexicon, Behavioral Shift)

### 3. LangChain with real LLM calls

```bash
pip install drift-detector-agent[langchain]
export GROQ_API_KEY="gsk_..."   # Groq keys start with gsk_
python3 examples/test_langchain_chain_real_groq.py
```

Runs a multi-step `research → analyze → summarize → report` task with
real Groq calls and reports drift between steps.

### 4. CrewAI integration

```bash
pip install drift-detector-agent[crewai]
python3 examples/test_crewai_with_drift.py
```

Shows:
- CrewAI task-event lifecycle
- `DriftDetectorEventListener` tracking
- Multi-task drift report

### 5. Stress test

```bash
python3 examples/stress_test_drift_detector.py
```

Pushes the detector with many sessions to validate performance and
stability. Use before release.

## Expected output (excerpt)

```
[STEP 1] Research Phase
  Output length: 5276 chars

[STEP 2] Analysis Phase
  Output length: 605 chars
  → Drift (1→2): 0.438
  - Ghost Lexicon: 0.560
  - Behavioral Shift: 0.483

[STEP 3] Summary Phase
  Output length: 260 chars
  → Drift (2→3): 0.319
  - Ghost Lexicon: 0.760  ← strong signal

[SIGNALS] Analysis
✓ Multi-step task executed
✓ Drift detected between steps
✓ Ghost Lexicon triggered (vocabulary shrinkage)
✓ Groq API working
✓ DriftDetector tracking
```

## API key setup

Only needed for the examples that hit real cloud LLMs (the `test_*_real_*`
files).

```bash
# Option A: environment variables
export GROQ_API_KEY="gsk_..."
export CEREBRAS_API_KEY="..."
export GOOGLE_API_KEY="..."

# Option B: .env file
cp .env.example .env
# then edit .env
```

The core drift-detection engine works offline and needs no keys at all.

## Dashboard

After running any example you can inspect results in the dashboard:

```bash
pip install drift-detector-agent[ui]
python3 -m uvicorn drift_detector.ui.server:app --host 127.0.0.1 --port 8000
# then open http://127.0.0.1:8000
```

The dashboard shows all drift reports, per-signal visualizations, loop
alerts, and updates in real time.

---

See `../drift_detector/integrations/README.md` for additional
integration patterns and API references.
