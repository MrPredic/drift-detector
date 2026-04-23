#!/usr/bin/env python3
"""
Example: Medical AI with Domain-Aware Drift Detection

Shows how to use domain_terms in DriftDetectorAgent to weight
critical medical terminology more heavily in loss calculations.

Scenario: Medical diagnosis assistant that loses critical terms like
"intubation", "SpO2", "ventilator" - these should be penalized more
heavily than generic vocabulary loss.
"""

from drift_detector.core.drift_detector_agent import (
    DriftDetectorAgent,
    AgentConfig,
)


def example_medical_drift_detection():
    """
    Example: Medical domain with critical vocabulary weighting.
    """

    # Define critical medical terms that must survive
    medical_domain_terms = [
        "intubation",
        "ventilator",
        "oxygen_saturation",
        "SpO2",
        "respiratory",
        "hemoglobin",
        "hemodynamics",
        "cardiac",
        "arrhythmia",
        "sepsis",
    ]

    # Create detector with domain terms
    config = AgentConfig(
        agent_id="medical_ai_1",
        task_type="general",
        domain_terms=medical_domain_terms,  # Critical medical vocab
    )
    detector = DriftDetectorAgent(config)

    # Simulate medical diagnosis before session boundary
    before_snapshot = detector.snapshot(
        agent_id="medical_ai_1",
        response_text=(
            "Patient requires intubation due to respiratory failure. "
            "SpO2 at 82% despite supplemental oxygen. "
            "Ventilator settings: FiO2=80%, PEEP=8. "
            "Hemoglobin critically low at 6.2 g/dL. "
            "Cardiac arrhythmia detected. Signs of sepsis."
        ),
        tool_calls=["vitals", "bloodwork", "cardiac_monitor", "imaging"],
    )

    # Simulate session boundary (model resets, context compresses)
    after_snapshot = detector.snapshot(
        agent_id="medical_ai_1",
        response_text=(
            "Patient has respiratory issues. "
            "Oxygen levels low at 82%. "
            "Blood readings show anemia. "
            "Heart problems detected."
        ),
        tool_calls=["vitals", "bloodwork"],  # Lost cardiac_monitor, imaging
    )

    # Measure drift
    report = detector.measure_drift(before_snapshot, after_snapshot)

    # Report
    print("=" * 80)
    print("MEDICAL DRIFT DETECTION EXAMPLE")
    print("=" * 80)
    print()
    print("Scenario: Medical diagnosis assistant losing critical terminology")
    print()
    print("Domain terms (weighted 2.0x):", ", ".join(medical_domain_terms[:5]))
    print()
    print("Results:")
    print(f"  Ghost Loss (vocabulary): {report.ghost_loss:.3f}")
    print(f"    → Lost critical terms: intubation, SpO2, ventilator, hemoglobin")
    print(f"    → Weighted heavily (2.0x) due to domain_terms parameter")
    print()
    print(f"  Behavior Shift (tools): {report.behavior_shift:.3f}")
    print(f"    → Lost tools: cardiac_monitor, imaging")
    print()
    print(f"  Combined Drift: {report.combined_drift_score:.3f}")
    print(f"  Status: {'DRIFTING ⚠️' if report.is_drifting else 'STABLE ✓'}")
    print()
    print("Use case: Alert clinical teams when medical AI loses precision terms")
    print("=" * 80)


if __name__ == "__main__":
    example_medical_drift_detection()
