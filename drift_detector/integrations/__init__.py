"""Integrations - LangChain and CrewAI adapters (optional)"""

try:
    from drift_detector.integrations.langchain import DriftDetectionCallback
except ImportError:
    DriftDetectionCallback = None

try:
    from drift_detector.integrations.crewai import DriftDetectorEventListener, DriftMonitoringCrewMixin
except ImportError:
    DriftDetectorEventListener = None
    DriftMonitoringCrewMixin = None

__all__ = ["DriftDetectionCallback", "DriftDetectorEventListener", "DriftMonitoringCrewMixin"]
