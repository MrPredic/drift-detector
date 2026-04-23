"""Secure API Key Management.

Reads secrets from a local `.env` file (if present) and/or `os.environ`.

Design:
- `SecretsManager()` NEVER raises at construction time, even if `.env` is
  missing. A missing `.env` just means "fall back to os.environ".
- The module exposes a lazy, cached `secrets` proxy so that
  `import drift_detector.config.secrets` has no side effects. The underlying
  `SecretsManager` instance is created the first time an attribute is
  actually used.
"""
import os
from pathlib import Path
from typing import Optional


class SecretsManager:
    """Load and manage API keys safely.

    If an `env_path` is provided, that file is loaded. Otherwise, a local
    `.env` next to the package is loaded *if it exists*. Missing `.env`
    is not an error — the class simply relies on values already in
    `os.environ` (this is the normal case in CI/CD, Docker, etc.).
    """

    def __init__(self, env_path: Optional[str] = None):
        if env_path:
            self._load_file(env_path)
            return

        # Best-effort: load a local .env if one is present. Silent no-op otherwise.
        local_env = Path(__file__).parent.parent / ".env"
        if local_env.exists():
            self._load_file(str(local_env))

    def _load_file(self, path: str) -> None:
        """Load a `.env` file into `os.environ` without overwriting existing vars."""
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                # Do not clobber values already set in the real environment.
                os.environ.setdefault(key, val)

    @staticmethod
    def get(key: str, default: str = "") -> str:
        return os.environ.get(key, default)

    @staticmethod
    def groq_key() -> str:
        return os.environ.get("GROQ_API_KEY", "")

    @staticmethod
    def cerebras_key() -> str:
        return os.environ.get("CEREBRAS_API_KEY", "")

    @staticmethod
    def openrouter_key() -> str:
        return os.environ.get("OPENROUTER_API_KEY", "")

    @staticmethod
    def google_api_key() -> str:
        """Google Gemini API Key."""
        return os.environ.get("GOOGLE_API_KEY", "")

    @staticmethod
    def ollama_host() -> str:
        return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class _LazySecrets:
    """Lazy, thread-safe-ish proxy around `SecretsManager`.

    The real instance is created the first time an attribute is accessed,
    so simply importing this module has NO side effects.
    """

    __slots__ = ("_impl",)

    def __init__(self) -> None:
        self._impl: Optional[SecretsManager] = None

    def _get(self) -> SecretsManager:
        if self._impl is None:
            self._impl = SecretsManager()
        return self._impl

    def __getattr__(self, name: str):
        return getattr(self._get(), name)


# Public module-level proxy. Safe to import; does not touch the filesystem
# until a caller actually asks for a secret.
secrets = _LazySecrets()
