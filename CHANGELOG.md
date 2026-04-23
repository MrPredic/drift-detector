# Changelog

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
