"""
LLM Provider Configuration & Routing (OPTIONAL).

This module is only needed if you use the optional LangChain-based LLM router
shipped with DriftDetector for the example scripts. The drift-detection core
itself does NOT require any LLM library.

Importing this module has no side effects — you must instantiate `LLMRouter`
explicitly to initialize providers.
"""
from enum import Enum
from typing import Optional

from drift_detector.config.secrets import secrets


class LLMProvider(Enum):
    GROQ = "groq"
    CEREBRAS = "cerebras"
    OPENROUTER = "openrouter"
    GEMINI = "gemini"
    OLLAMA = "ollama"


class LLMRouter:
    """Route tasks to the best LLM based on provider availability.

    LangChain is imported lazily so that `drift_detector.config` stays
    importable without LangChain installed.
    """

    def __init__(self):
        try:
            from langchain_openai import ChatOpenAI  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "LLMRouter requires LangChain. "
                "Install with: pip install drift-detector[langchain]"
            ) from exc

        self.providers = self._init_providers()

    def _init_providers(self) -> dict:
        from langchain_openai import ChatOpenAI

        providers: dict = {}

        # Groq (fast, free tier)
        if secrets.groq_key():
            providers[LLMProvider.GROQ] = ChatOpenAI(
                api_key=secrets.groq_key(),
                base_url="https://api.groq.com/openai/v1",
                model_name="llama-3.3-70b-versatile",
                temperature=0.7,
            )

        # Cerebras
        if secrets.cerebras_key():
            providers[LLMProvider.CEREBRAS] = ChatOpenAI(
                api_key=secrets.cerebras_key(),
                base_url="https://api.cerebras.ai/v1",
                model_name="llama3.1-8b",
                temperature=0.7,
            )

        # OpenRouter
        if secrets.openrouter_key():
            providers[LLMProvider.OPENROUTER] = ChatOpenAI(
                api_key=secrets.openrouter_key(),
                base_url="https://openrouter.ai/api/v1",
                model_name="meta-llama/llama-3.3-70b-instruct",
                temperature=0.7,
            )

        # Google Gemini (requires separate package)
        if secrets.google_api_key():
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                providers[LLMProvider.GEMINI] = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    api_key=secrets.google_api_key(),
                    temperature=0.7,
                )
            except ImportError:
                pass  # Gemini optional

        # Ollama (local)
        providers[LLMProvider.OLLAMA] = ChatOpenAI(
            base_url=secrets.ollama_host(),
            model_name="llama2",
            temperature=0.7,
            api_key="ollama",  # dummy for local
        )

        return providers

    def get_llm(self, provider: Optional[LLMProvider] = None):
        """Return an LLM instance (fastest available if `provider` is None)."""
        if provider and provider in self.providers:
            return self.providers[provider]

        for p in (
            LLMProvider.GROQ,
            LLMProvider.CEREBRAS,
            LLMProvider.OPENROUTER,
            LLMProvider.GEMINI,
            LLMProvider.OLLAMA,
        ):
            if p in self.providers:
                return self.providers[p]

        raise RuntimeError("No LLM providers available")

    def status(self) -> dict:
        """Report which providers have been initialized."""
        return {p.value: p in self.providers for p in LLMProvider}
