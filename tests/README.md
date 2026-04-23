# DriftDetector Tests

Test suite for core detector, integrations, and examples

## Test Files

| File | Scope | Count | Status |
|------|-------|-------|--------|
| `test_core.py` | DriftDetectorAgent | 13 tests | ✅ PASS |
| `test_langchain_integration.py` | LangChain callback | 5 tests | ✅ PASS |
| `test_crewai_integration.py` | CrewAI listener | 7 tests | ✅ PASS |

**Total:** 25/25 tests PASS

## Run Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_core.py -v

# With coverage
pytest tests/ --cov=drift_detector --cov-report=html
```

## Test Results

```
Core Tests (13):
  ✓ Ghost Lexicon detection
  ✓ Behavioral Shift detection
  ✓ Agreement Score calculation
  ✓ Loop Detection (exact repetition)
  ✓ Loop Detection (same action)
  ✓ Stagnation Detection
  ✓ Edge Case: Empty Lists
  ✓ Edge Case: Single Model
  ... (13/13 PASS)

Integration Tests (12):
  ✓ LangChain Callback initialization
  ✓ LangChain Agent action/finish hooks
  ✓ CrewAI EventListener initialization
  ✓ CrewAI TaskStartedEvent handling
  ✓ CrewAI ToolCallEvent tracking
  ✓ CrewAI TaskCompletedEvent measurement
  ... (12/12 PASS)

Total: 25/25 PASS
```

## Known Edge Cases

All handled:
- Empty response text → safe default
- No tool calls → treated as legitimate
- Single model → agreement = 1.0 (correct)
- Concurrent events → safe with SQLite WAL

---

See core/README.md for what each signal tests.
