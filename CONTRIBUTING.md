# Contributing to DriftDetector

Contributions are welcome! Here's how to help.

## Getting Started

```bash
git clone https://github.com/MrPredic/drift-detector
cd drift-detector
pip install -e ".[dev]"
python -m pytest tests/
```

## Development Setup

```bash
# Install with dev dependencies
pip install -e ".[all]"

# Run tests
python -m pytest tests/ -v

# Run linter (optional)
# (we keep it simple, no strict linting)

# Type hints (preferred but not required)
python -m mypy drift_detector/core/ --ignore-missing-imports
```

## What To Contribute

### Easy (Great for First-Time Contributors)
- [ ] Bug reports (issues with stack traces)
- [ ] Documentation fixes (typos, clarity, examples)
- [ ] New examples (use cases in `examples/`)
- [ ] Test improvements (better edge cases)

### Medium (Some Implementation)
- [ ] New drift signals (8th, 9th signal ideas)
- [ ] Performance optimizations
- [ ] Dashboard improvements (new charts, features)
- [ ] Additional integrations (Anthropic, Mistral, Cohere APIs)

### Hard (Requires Deep Review)
- [ ] Architecture changes
- [ ] Core algorithm modifications
- [ ] Large feature additions

## Code Style

**Keep it simple:**
- No complex abstractions for one-time operations
- Backward compatible (all new params have defaults)
- Type hints preferred (but not required)
- 4-space indentation, no trailing whitespace

**No unsolicited changes:**
- Don't refactor code you didn't touch
- Don't add docstrings beyond what's needed
- Don't add features beyond the issue scope

## Testing

All PRs must pass tests:

```bash
python -m pytest tests/ -v
```

**Expected result:** 42/43 PASS (1 pre-existing failure is acceptable)

### Writing Tests

Add tests to `tests/` directory:
- Name test files: `test_<feature>.py`
- Name test functions: `test_<specific_case>()`
- Keep tests focused and simple

Example:
```python
def test_my_new_feature():
    detector = DriftDetectorAgent()
    before = detector.snapshot(agent_id="test", response_text="hello")
    after = detector.snapshot(agent_id="test", response_text="hi")
    report = detector.measure_drift(before, after)
    assert report.combined_drift_score > 0.0
```

## Submitting a PR

1. **Fork & branch:** `git checkout -b feature/my-feature`
2. **Make changes:** Only modify what's needed
3. **Test:** `python -m pytest tests/`
4. **Commit:** Clear, concise messages
5. **Push:** `git push origin feature/my-feature`
6. **PR:** Describe what & why (not just what)

### PR Title Format
- ✅ "Add fuzzy matching option for agreement score"
- ✅ "Fix thread-safety in detector singleton"
- ❌ "WIP" or "update"

## Reporting Issues

**Good issue:**
```
### Describe the problem
When I use domain_terms=["keyword"], the score doesn't change even though the keyword disappears.

### Steps to reproduce
1. Create detector with domain_terms
2. Run measure_drift with keyword present → score=0.2
3. Run measure_drift with keyword absent → score=0.2 (expected: 0.4+)

### Environment
- Python 3.11
- DriftDetector v2.0
```

**What NOT to do:**
- Generic titles ("doesn't work", "help")
- No reproduction steps
- Screenshots instead of code

## Questions?

- **Usage questions:** Open a discussion (not an issue)
- **Bug reports:** Open an issue with reproduction steps
- **Feature ideas:** Open a discussion first (get feedback before coding)

---

**Thanks for contributing!** 🚀
