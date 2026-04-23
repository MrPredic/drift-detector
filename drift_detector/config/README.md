# DriftDetector Config

Multi-provider LLM routing and secret management

## Files

- `llm_config.py` - LLM provider routing
- `secrets.py` - API key management

## LLM Routing

```python
from drift_detector.config import LLMRouter, LLMProvider

router = LLMRouter()

# Get primary LLM (Groq)
llm = router.get_llm()

# Get specific provider
llm = router.get_llm(LLMProvider.CEREBRAS)

# Check available providers
print(router.status())
# {'groq': True, 'cerebras': True, 'gemini': True, ...}
```

## Fallback Order

1. **Groq** (fastest, 30 RPM)
2. **Cerebras** (60K tokens/min)
3. **OpenRouter** (diverse models)
4. **Gemini** (long context)
5. **Ollama** (local, offline)

First available is used automatically.

## API Keys

Set in `.env`:

```
GROQ_API_KEY=gsk_...
CEREBRAS_API_KEY=csk_...
OPENROUTER_API_KEY=sk-or-v1-...
GOOGLE_API_KEY=AIzaSy...
OLLAMA_HOST=http://localhost:11434
```

Or environment variables:

```bash
export GROQ_API_KEY="..."
```

## Models (Apr 13, 2026)

| Provider | Model | Status |
|----------|-------|--------|
| Groq | llama-3.3-70b-versatile | ✅ |
| Cerebras | llama3.1-8b | ✅ |
| OpenRouter | meta-llama/llama-3.3-70b-instruct | ✅ |
| Gemini | gemini-2.5-flash | ✅ |
| Ollama | llama2 | ✅ |

---

See ACTIVE_MODELS.md in parent directory for full details.
