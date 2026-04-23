# Changelog

## [2.0.1] - 2026-04-23

### 🐛 Bug Fixes
- **LangChain callback:** `on_chain_end` now coerces output across all common key names (`output`, `text`, `result`, `answer`, `response`) — previously no-oped silently on most chains
- **combined_drift_score:** Loop detection signal (5th) now included in mean — was missing, contradicting README
- **DB timestamp collision:** Removed `UNIQUE` constraint on timestamp column — silent data loss under concurrent load is fixed
- **Threading:** `drift_history.append` now protected by `threading.Lock` — concurrent FastAPI requests no longer race on list mutation
- **CORS:** `allow_credentials=True` + `allow_origins=["*"]` was an invalid combo; credentials now disabled when origins is wildcard
- **Severity thresholds:** `critical`/`warning` levels now scale with `signal_threshold` config instead of hardcoded `0.7`
- **Vocabulary regex:** `min_len` default lowered from 5 → 3, capturing short but critical terms like `AI`, `LLM`, `API`, `SQL`
- **Loop map:** `task_type="medical"` now has an explicit entry (`min_unique_actions=4`) instead of silently falling back to default
- **Memory:** `_load_history` now caps at last 10,000 rows (`ORDER BY id DESC LIMIT 10000`) — prevents unbounded memory growth
- **Logging:** Core DB errors now use `logger.warning()` instead of `print()` for production observability

## [2.0.0] - 2026-04-20

### ✨ New Features
- **Web Dashboard v4:** Real-time drift visualization with zone-based coloring
- **Session Management:** Optional SQLite-backed session persistence
- **5 Drift Signals:** Ghost Lexicon, Behavioral Shift, Agreement Score, Loop Detection, Stagnation
- **Domain-Aware Detection:** Custom domain term weighting
- **Task-Type Adaptation:** Thresholds optimized per task type (coding, research, trading, medical)
- **Fuzzy Matching:** Levenshtein distance-based token comparison
- **Degradation Mode:** Higher penalties for tool/vocabulary loss

### 🔒 Security & Quality
- **Thread-Safe Architecture:** Lifespan-based lifecycle management with dependency injection
- **Subresource Integrity:** Chart.js pinned version (v4.5.1) with SRI hash
- **CORS Configuration:** Restrictable origins (default: localhost only in dev)
- **Config Validation:** AgentConfig validates all thresholds & parameters at startup
- **Error Handling:** Graceful failures with non-leaking error messages

### 📊 UI Improvements
- **Zone-Based Visualization:** Green (safe), Yellow (warning), Red (danger) zones
- **Dark Mode Labels:** High-contrast Y-axis labels (#d1d5db, 13px)
- **Session Sidebar:** Browse historical sessions, click to switch
- **Responsive Design:** Works on mobile, tablet, desktop

### 📚 Documentation
- **SECURITY.md:** Deployment modes (local, internal, production) with best practices
- **Updated README:** Setup wizard, quick start, API reference
- **Examples:** Working code for medical, coding, and degradation scenarios

### 🐛 Bug Fixes
- Fixed global detector singleton (now thread-safe via FastAPI lifespan)
- Fixed CORS allowing `["*"]` (now configurable via CORS_ORIGINS env)
- Fixed Chart.js CDN without integrity verification (now pinned with SRI)

### 📈 Performance
- No external dependencies added (pure Python, minimal LangChain)
- Negligible overhead for new features (O(n) only when enabled)
- In-memory default, optional SQLite for persistence

### 🧪 Testing
- 159/159 tests PASS
- Comprehensive signal tests, integration tests, domain-aware tests
- Task-type threshold validation tests

---

## Installation & Upgrade

### New Install
```bash
pip install drift-detector-agent
python setup_wizard.py
```

### From v1.x
- Backward compatible: v1.x code works without changes
- Optional features: domain_terms, is_degradation, use_fuzzy_matching (all default off)
- Session storage: Disabled by default (enable via `PERSIST_SESSIONS=true`)

---

## Migration Guide (If Upgrading)

**No breaking changes.** All v1.x code continues to work:

```python
# v1.x code (still works)
detector = DriftDetectorAgent()
report = detector.measure_drift(before, after)

# v2.0 optional enhancements
detector = DriftDetectorAgent(
    domain_terms=["critical", "important"],  # NEW: domain weighting
    task_type="medical"  # NEW: task-specific thresholds
)
report = detector.measure_drift(before, after)
```

---

## Known Limitations

- **No built-in auth:** Add at reverse proxy layer (nginx/AWS ALB)
- **No rate limiting:** Configure at proxy layer
- **In-memory default:** Session storage is optional (PERSIST_SESSIONS=false by default)
- All 43 tests pass as of v2.0.0

---

## Next Steps

- [ ] More examples (finance, healthcare, customer support)
- [ ] Dashboard enhancements (multi-session comparison, alerts)
- [ ] Performance optimizations for high-volume scenarios
- [ ] Community contributions welcome!

---

See [SECURITY.md](SECURITY.md) for deployment best practices.
