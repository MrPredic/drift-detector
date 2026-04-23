"""DriftDetector - Behavioral drift detection for LLM agents"""

__version__ = "2.0.2"

from drift_detector.core.drift_detector_agent import DriftDetectorAgent, DriftReport, SessionSnapshot

# Optional: Config and integrations (loaded only if needed)
try:
    from drift_detector.config.llm_config import LLMRouter, LLMProvider
except ImportError:
    pass

try:
    from drift_detector.integrations.langchain import DriftDetectionCallback
    from drift_detector.integrations.crewai import DriftDetectorEventListener, DriftMonitoringCrewMixin
except ImportError:
    pass

__all__ = [
    "DriftDetectorAgent",
    "DriftReport",
    "SessionSnapshot",
    "LLMRouter",
    "LLMProvider",
    "DriftDetectionCallback",
    "DriftDetectorEventListener",
    "DriftMonitoringCrewMixin",
]
