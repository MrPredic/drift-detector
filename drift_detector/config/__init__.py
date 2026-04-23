"""Configuration - LLM routing and secrets management (optional)"""

try:
    from drift_detector.config.llm_config import LLMRouter, LLMProvider
except ImportError:
    LLMRouter = None
    LLMProvider = None

__all__ = ["LLMRouter", "LLMProvider"]
