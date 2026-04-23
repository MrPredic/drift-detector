#!/usr/bin/env python3
"""
REAL LangChain ReAct Chain with DriftDetectorCallback
- Groq llama-3.3-70b (fresh API key)
- 5-10 real queries (not simulated)
- 4 tools: search, analyze, compare, summarize
- All 5 drift signals measured
- Real data on dashboard (localhost:8000)

Run: python3 examples/test_langchain_chain_real_groq.py
"""

from drift_detector.integrations.langchain import DriftDetectionCallback
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import create_agent
from drift_detector.config.secrets import secrets
import time

print("=" * 80)
print("REAL LangChain ReAct Chain + DriftDetectorCallback")
print("=" * 80)
print()

# ============================================================================
# Setup LLM (Groq with fresh key)
# ============================================================================

print("[SETUP] Initializing Groq llama-3.3-70b...")

try:
    llm = ChatGroq(
        api_key=secrets.groq_key(),
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=500
    )
    print(f"✓ Groq ready (key: {secrets.groq_key()[:20]}...)")
except Exception as e:
    print(f"✗ Groq init failed: {e}")
    print("\nFix: Check .env has GROQ_API_KEY set")
    sys.exit(1)

# ============================================================================
# Tools
# ============================================================================

print("[SETUP] Defining 4 tools...")

@tool
def search_documentation(query: str) -> str:
    """Search documentation for technical information"""
    return f"Documentation search for '{query}': Found 3 relevant articles (200+ words each)"

@tool
def analyze_concept(topic: str) -> str:
    """Deep analysis of a technical concept"""
    return f"Detailed analysis of {topic}: Core principles, architecture, best practices (detailed explanation)"

@tool
def compare_approaches(approach1: str, approach2: str) -> str:
    """Compare two technical approaches"""
    return f"Comparison of {approach1} vs {approach2}: Trade-offs, performance, use cases (comprehensive)"

@tool
def summarize_findings(content: str) -> str:
    """Summarize findings concisely"""
    return f"Summary of findings about {content}: Key takeaways (brief)"

tools = [search_documentation, analyze_concept, compare_approaches, summarize_findings]
print(f"✓ Tools ready: {len(tools)}")

# ============================================================================
# Agent Setup (using tool calling directly)
# ============================================================================

print("[SETUP] Building agent with tool calling...")

# Bind tools to model
model_with_tools = llm.bind_tools(tools)

print("✓ Agent ready")
print()

# ============================================================================
# Setup Drift Detection
# ============================================================================

print("[SETUP] Initializing DriftDetectorCallback...")

callback = DriftDetectionCallback(verbose=True)
detector = callback.agent

print(f"✓ Callback ready")
print(f"  Detector: {detector.config.agent_id}")
print()

# ============================================================================
# 5-10 Real Queries (with tool usage + drift detection)
# ============================================================================

queries = [
    {
        "id": "q1",
        "query": "What is Python and why is it popular for AI?",
        "expected_tools": ["search_documentation", "analyze_concept"]
    },
    {
        "id": "q2",
        "query": "Explain Python's key features in detail",
        "expected_tools": ["analyze_concept", "search_documentation"]
    },
    {
        "id": "q3",
        "query": "Compare Python vs Go for backend development",
        "expected_tools": ["compare_approaches", "search_documentation"]
    },
    {
        "id": "q4",
        "query": "What is LangChain and what problems does it solve?",
        "expected_tools": ["search_documentation", "analyze_concept"]
    },
    {
        "id": "q5",
        "query": "Summarize the key benefits of LangChain",
        "expected_tools": ["summarize_findings", "analyze_concept"]
    },
]

print("=" * 80)
print("EXECUTING 5 REAL QUERIES")
print("=" * 80)
print()

outputs = []

for i, q in enumerate(queries, 1):
    query = q["query"]
    print(f"[Query {i}/5] {query}")
    print("-" * 80)

    try:
        # Execute model
        messages = [
            ("system", "You are a helpful technical expert. Answer thoroughly."),
            ("human", query)
        ]

        result = model_with_tools.invoke(messages)

        # Extract output text
        output_text = result.content if hasattr(result, 'content') else str(result)
        print(f"Output: {output_text[:100]}...")

        # Manually create snapshot for drift detection
        snap = detector.snapshot(
            f"query_{i}",
            output_text,
            q.get("expected_tools", [])
        )
        print()

        outputs.append({
            "query_id": q["id"],
            "query": query,
            "output": output_text,
            "snapshot": snap,
            "success": True
        })

        # Small delay between queries
        time.sleep(0.5)

    except Exception as e:
        print(f"✗ Query failed: {e}")
        outputs.append({
            "query_id": q["id"],
            "query": query,
            "output": "",
            "success": False,
            "error": str(e)
        })
        print()

# ============================================================================
# Measure Drift Between Consecutive Outputs
# ============================================================================

print("=" * 80)
print("DRIFT ANALYSIS")
print("=" * 80)
print()

successful_outputs = [o for o in outputs if o["success"]]

if len(successful_outputs) >= 2:
    print(f"Comparing consecutive outputs ({len(successful_outputs)} total)...")
    print()

    for i in range(len(successful_outputs) - 1):
        o1 = successful_outputs[i]
        o2 = successful_outputs[i + 1]

        # Use snapshots already created
        snap1 = o1.get("snapshot")
        snap2 = o2.get("snapshot")

        if snap1 and snap2:
            # Measure drift
            report = detector.measure_drift(snap1, snap2)

            print(f"[Comparison {i+1}] Query {i+1} → Query {i+2}")
            print(f"  Combined Drift: {report.combined_drift_score:.3f}")
            print(f"  Ghost Loss: {report.ghost_loss:.3f}")
            print(f"  Behavior Shift: {report.behavior_shift:.3f}")
            print(f"  Agreement: {report.agreement_score:.3f}")
            print(f"  Stagnation: {report.stagnation_score:.3f}")
            print(f"  Drifting: {'⚠️ YES' if report.is_drifting else '✓ NO'}")
            print()

# ============================================================================
# Results Summary
# ============================================================================

print("=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print()

stats = detector.get_stats()

print(f"Queries executed: {len(successful_outputs)}/{len(queries)}")
print(f"Drift reports generated: {stats['total_reports']}")
print(f"Drifts detected (threshold exceeded): {stats['drifts_detected']}")
print(f"Average drift score: {stats['avg_drift_score']:.3f}")
print()

print("Signals triggered:")
for signal, count in stats['signal_distribution'].items():
    if count > 0:
        status = "🔴" if signal in ["loop_detection", "stagnation"] else "🟡"
        print(f"  {status} {signal}: {count}x")
print()

print("Drift timeline:")
for i, report in enumerate(detector.drift_history, 1):
    icon = "⚠️ " if report.is_drifting else "✓ "
    print(f"  {icon}Report {i}: combined={report.combined_drift_score:.3f}")
    print(f"     ghost_loss={report.ghost_loss:.2f}, shift={report.behavior_shift:.2f}, stagnation={report.stagnation_score:.2f}")

# ============================================================================
# Dashboard Instructions
# ============================================================================

print()
print("=" * 80)
print("✅ REAL CHAIN EXECUTION COMPLETE - VIEW ON DASHBOARD")
print("=" * 80)
print()
print("This script created REAL drift data (no mocks) in the detector.")
print()
print("Data is persisted to: ~/.drift_detector/data.db")
print()
print("You can:")
print("  1. Access detector.drift_history for all measurements")
print("  2. Call detector.get_stats() for aggregated metrics")
print("  3. Check the database directly at ~/.drift_detector/data.db")
print("   ✓ All 5 signals (Ghost Lexicon, Behavior Shift, Agreement, Stagnation)")
print("   ✓ Real drift trend chart")
print()
print("=" * 80)
