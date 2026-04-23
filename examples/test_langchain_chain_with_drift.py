#!/usr/bin/env python3
"""
Real LangChain Chain with DriftDetectorCallback
Uses Groq/Cerebras (not Claude) with ReAct agent pattern
Shows real-time drift monitoring on dashboard

Run this, then open http://localhost:8000 in another terminal
"""

from drift_detector.integrations.langchain import DriftDetectionCallback
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

print("=" * 80)
print("LangChain Chain + DriftDetectorCallback Test")
print("=" * 80)
print()

# ============================================================================
# Step 1: Setup LLM (Groq/Cerebras)
# ============================================================================

print("[1] Initializing LLM...")

llm = None
try:
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0.7,
        max_tokens=500
    )
    print("✓ Using Groq (llama-3.1-70b-versatile)")
except Exception as e:
    print(f"  Groq failed: {e}")
    print("  Trying Cerebras...")
    try:
        from langchain_community.chat_models import ChatCerebras
        llm = ChatCerebras(model="llama-3.1-70b")
        print("✓ Using Cerebras (llama-3.1-70b)")
    except Exception as e2:
        print(f"✗ Both failed: {e2}")
        print("\nSetup Instructions:")
        print("  pip install langchain langchain-groq langchain-community")
        print("  export GROQ_API_KEY=your_key")
        print("  export CEREBRAS_API_KEY=your_key")
        sys.exit(1)

# ============================================================================
# Step 2: Define Tools
# ============================================================================

print("\n[2] Defining tools...")

@tool
def search_documentation(query: str) -> str:
    """Search documentation for information about a topic"""
    docs = {
        "python": {
            "description": "Python is a high-level programming language",
            "features": ["Dynamic typing", "Easy to learn", "Great for AI/ML"],
            "use_cases": ["Web development", "Data science", "Automation"]
        },
        "langchain": {
            "description": "Framework for building applications with LLMs",
            "features": ["Chains", "Agents", "Memory", "Tools"],
            "use_cases": ["Chatbots", "Q&A systems", "Code generation"]
        },
        "drift detection": {
            "description": "Monitoring system for detecting behavioral changes",
            "features": ["Ghost Lexicon", "Behavioral Shift", "Loop Detection"],
            "use_cases": ["LLM monitoring", "Agent health", "Quality assurance"]
        },
    }

    query_lower = query.lower()
    for key, value in docs.items():
        if key in query_lower:
            return f"Found docs for '{key}': {value['description']}. Features: {', '.join(value['features'])}"

    return f"No docs found for '{query}'. Available topics: {', '.join(docs.keys())}"


@tool
def analyze_concept(concept: str) -> str:
    """Analyze a concept and provide detailed explanation"""
    analyses = {
        "python": "Python combines simplicity with power. It has dynamic typing, automatic memory management, and a huge ecosystem.",
        "langchain": "LangChain abstracts LLM complexity. It provides Chain composition, Agent orchestration, and Memory management.",
        "drift detection": "Drift detection monitors if agent behavior is changing. Detects 5 signals: ghost lexicon loss, behavioral shift, agreement degradation, loops, and stagnation.",
        "testing": "Testing ensures code quality. Unit tests check individual functions. Integration tests check component interactions.",
    }

    concept_lower = concept.lower()
    for key, analysis in analyses.items():
        if key in concept_lower:
            return f"Analysis of '{key}': {analysis}"

    return f"Analysis: '{concept}' is an important topic in software development."


@tool
def compare_approaches(topic: str) -> str:
    """Compare different approaches to a topic"""
    comparisons = {
        "langchain vs custom": "LangChain: easier to start, opinionated. Custom: more flexible, harder to build.",
        "groq vs cerebras": "Groq: faster inference, more stable. Cerebras: cheaper, experimental features.",
        "streaming vs batch": "Streaming: real-time updates, complex. Batch: simpler, higher latency.",
    }

    topic_lower = topic.lower()
    for key, comparison in comparisons.items():
        if any(part in topic_lower for part in key.split()):
            return f"Comparison ({key}): {comparison}"

    return f"Comparison: Different approaches have trade-offs in speed, cost, and complexity."


@tool
def summarize_findings(text: str) -> str:
    """Summarize findings into key points"""
    # Extract first 100 chars as key point
    summary = text[:100] + "..." if len(text) > 100 else text
    key_points = summary.split(".")[:2]
    return f"Summary: {'. '.join(key_points)}"


tools = [search_documentation, analyze_concept, compare_approaches, summarize_findings]
print(f"✓ Defined 4 tools: {[t.name for t in tools]}")

# ============================================================================
# Step 3: Create Chain
# ============================================================================

print("\n[3] Creating ReAct chain...")

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful research assistant.

Your task is to research topics thoroughly using the available tools.
For each query:
1. Search documentation for context
2. Analyze the main concept
3. Compare relevant approaches
4. Summarize findings

Be thorough but concise. Use tools to gather information."""),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

try:
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=5)
    print("✓ ReAct chain created (max 5 steps per query)")
except Exception as e:
    print(f"✗ Failed to create chain: {e}")
    sys.exit(1)

# ============================================================================
# Step 4: Setup Drift Detection
# ============================================================================

print("\n[4] Setting up drift detection...")

try:
    callback = DriftDetectionCallback(verbose=True)
    print(f"✓ Callback initialized")
    print(f"  Detector: {callback.agent.config.agent_id}")
    print(f"  Thresholds: drift={callback.agent.config.drift_threshold}, signal={callback.agent.config.signal_threshold}")
except Exception as e:
    print(f"✗ Failed: {e}")
    sys.exit(1)

# ============================================================================
# Step 5: Execute Chain with Drift Monitoring
# ============================================================================

print("\n" + "=" * 80)
print("CHAIN EXECUTION WITH DRIFT MONITORING")
print("=" * 80)
print()

queries = [
    "Search for information about Python. Analyze it. What are the key features?",
    "Compare LangChain vs custom implementation. Which is better and why?",
    "Explain drift detection. What signals does it monitor?",
    "Compare Groq vs Cerebras. Which would you choose for production?",
]

results = []
try:
    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}/{len(queries)}] {query[:60]}...")
        print("-" * 80)

        # Run chain with drift callback!
        result = executor.invoke(
            {"input": query},
            callbacks=[callback]
        )

        output = result.get("output", "")
        print(f"Output: {output[:150]}...")
        results.append(result)
        print()

except Exception as e:
    print(f"\n✗ Chain execution failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Step 6: Show Results
# ============================================================================

print("\n" + "=" * 80)
print("DRIFT DETECTION RESULTS")
print("=" * 80)
print()

stats = callback.agent.get_stats()
print(f"Queries processed: {len(results)}")
print(f"Drift reports: {stats['total_reports']}")
print(f"Drifts detected (> threshold): {stats['drifts_detected']}")
print(f"Average drift score: {stats['avg_drift_score']:.3f}")
print()

if stats['signal_distribution']:
    print("Signals triggered:")
    for signal, count in stats['signal_distribution'].items():
        if count > 0:
            print(f"  • {signal}: {count}x")
print()

if callback.agent.drift_history:
    print("Drift timeline:")
    for i, report in enumerate(callback.agent.drift_history[-3:], 1):
        status = "⚠️ DRIFT" if report.is_drifting else "✓ OK"
        print(f"  Report {i}: {status} (combined={report.combined_drift_score:.3f})")
        print(f"    └─ Ghost Loss: {report.ghost_loss:.2f}, Shift: {report.behavior_shift:.2f}, Agreement: {report.agreement_score:.2f}")

# ============================================================================
# Step 7: Dashboard Instructions
# ============================================================================

print()
print("=" * 80)
print("VIEW REAL-TIME DATA ON DASHBOARD")
print("=" * 80)
print()
print("To see live drift monitoring:")
print()
print("1. In another terminal, start the API:")
print("   python3 -m uvicorn drift_detector.ui.server:app --host 127.0.0.1 --port 8000 --reload")
print()
print("2. Open browser:")
print("   http://localhost:8000")
print()
print("3. You'll see:")
print("   ✓ Real drift data from THIS script")
print("   ✓ All 5 signals monitored")
print("   ✓ Chain step-by-step execution")
print("   ✓ Live drift trends")
print()
print("=" * 80)
