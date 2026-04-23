#!/usr/bin/env python3
"""
LangChain Integration for DriftDetectorAgent
Provides callback handler for monitoring drift during chain execution
"""

from typing import Dict, Optional

try:
    from langchain_core.callbacks import BaseCallbackHandler
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    BaseCallbackHandler = object  # Fallback so class definition doesn't fail

from drift_detector.core.drift_detector_agent import DriftDetectorAgent, SessionSnapshot, DriftAlert, DriftReport


class DriftDetectionCallback(BaseCallbackHandler):
    """
    LangChain callback handler for drift detection.
    Hooks into chain execution to auto-snapshot agent outputs and detect drift.

    Detects:
    - Session boundary drift (output changes significantly)
    - Loop detection (agent repeating same actions)
    - Stagnation (repeated identical outputs)
    - Agreement degradation (multi-model inconsistency)

    Usage Option 1 (Standalone):
        # Initialize detector
        detector = DriftDetectorAgent(AgentConfig(
            agent_id="my_chain",
            role="monitor"
        ))

        # Create callback
        callback = DriftDetectionCallback(detector)

        # Add to chain
        result = chain.invoke({"input": "..."}, callbacks=[callback])

    Usage Option 2 (With Dashboard):
        # Use global detector from API (shared with dashboard)
        from api.server import get_detector

        callback = DriftDetectionCallback(get_detector())
        result = chain.invoke({"input": "..."}, callbacks=[callback])

        # Dashboard at localhost:8000 shows real-time data!

    Alerts are printed automatically. For custom handling, subclass and override on_drift().
    """

    name = "drift_detection"

    def __init__(self, agent: Optional[DriftDetectorAgent] = None, verbose: bool = True):
        """
        Initialize drift detection callback.

        Args:
            agent: DriftDetectorAgent instance. If None, creates a new instance.
            verbose: Print alerts to stdout (default True)

        Raises:
            ImportError: If langchain-core is not installed.
                         Install with: pip install drift-detector-agent[langchain]
        """
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for DriftDetectionCallback. "
                "Install with: pip install drift-detector-agent[langchain]"
            )
        super().__init__()

        # Use provided agent or create new one
        if agent is None:
            self.agent = DriftDetectorAgent(agent_id="langchain_callback")
        else:
            self.agent = agent

        self.prev_snapshot: Optional[SessionSnapshot] = None
        self.verbose = verbose

    def on_chain_end(self, outputs: Dict, **kwargs):
        """Called when chain execution completes"""
        try:
            output_text = outputs.get('output', '')
            tool_calls = outputs.get('tool_calls', [])

            # Create snapshot
            try:
                current_snapshot = self.agent.snapshot(
                    agent_id=self.agent.config.agent_id,
                    response_text=output_text,
                    tool_calls=tool_calls
                )
            except ValueError as e:
                if self.verbose:
                    print(f"⚠️  Snapshot validation error: {str(e)}")
                return

            # Measure drift if we have previous snapshot
            if self.prev_snapshot:
                try:
                    report = self.agent.measure_drift(
                        self.prev_snapshot,
                        current_snapshot,
                        model_outputs={"chain": output_text}
                    )
                    # Note: task_type automatically used from self.agent.config.task_type

                    # Generate alert if drift detected
                    alert = DriftAlert.from_report(report, self.agent.config.signal_threshold)

                    if alert:
                        self.on_drift(alert, report)

                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Drift measurement error: {str(e)}")
                    return

            # Store for next iteration
            self.prev_snapshot = current_snapshot

        except Exception as e:
            # Graceful degradation - never crash the chain
            if self.verbose:
                print(f"✗ Drift callback error (non-blocking): {str(e)}")

    def on_drift(self, alert: DriftAlert, report):
        """
        Called when drift is detected.
        Override this method for custom drift handling.

        Args:
            alert: DriftAlert with severity level
            report: DriftReport with detailed metrics
        """
        if self.verbose:
            print(f"⚠️  DRIFT [{alert.level.upper()}]: {alert.message}")
            print(f"     Signal: {alert.signal}")
            print(f"     Score: {alert.score:.3f} > {alert.threshold:.3f}")

    def get_stats(self) -> Dict:
        """Get drift detection statistics"""
        return self.agent.get_stats()

    def get_history(self) -> list:
        """Get drift history"""
        return [
            {
                "timestamp": r.timestamp,
                "combined_drift": r.combined_drift_score,
                "ghost_loss": r.ghost_loss,
                "behavior_shift": r.behavior_shift,
                "agreement": r.agreement_score,
                "stagnation": r.stagnation_score,
                "is_drifting": r.is_drifting,
            }
            for r in self.agent.drift_history
        ]

    def get_drift_report(self) -> Dict:
        """Get drift report (total_drifts + history)"""
        history = self.get_history()
        return {
            "total_drifts": len([h for h in history if h.get("is_drifting", False)]),
            "history": history
        }


__all__ = ["DriftDetectionCallback"]
