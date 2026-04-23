#!/usr/bin/env python3
"""
Stress Test: DriftDetectorAgent Performance & Reliability
Tests: Performance, Memory, Edge Cases, Load, Accuracy
Uses: Groq only (0 Claude tokens), Chain does the work
"""

import os
import time
import json
import tracemalloc
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from drift_detector.core.drift_detector_agent import DriftDetectorAgent, AgentConfig

print("="*80)
print("STRESS TEST: DriftDetectorAgent")
print("="*80 + "\n")

groq_model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

drift_detector = DriftDetectorAgent(AgentConfig(
    agent_id="stress_test",
    role="monitor",
    memory_enabled=False
))

results = {
    "performance": {},
    "edge_cases": {},
    "load_test": {},
    "memory": {},
    "failures": []
}

# ============================================================================
# TEST 1: PERFORMANCE (Time per operation)
# ============================================================================

print("[1/5] PERFORMANCE TEST")
print("-" * 80)

template = ChatPromptTemplate.from_template("Explain {topic} in 2 sentences.")

topics = ["ML", "AI", "data"]
times = defaultdict(list)

for topic in topics:
    # Get model output
    t0 = time.time()
    resp = template.pipe(groq_model).invoke({"topic": topic})
    text = resp.content if hasattr(resp, 'content') else str(resp)
    model_time = time.time() - t0
    times["model_invoke"].append(model_time)

    # Snapshot
    t0 = time.time()
    snap = drift_detector.snapshot(f"test_{topic}", text, ["test"])
    snap_time = time.time() - t0
    times["snapshot"].append(snap_time)

    # Measure drift (self vs self)
    t0 = time.time()
    report = drift_detector.measure_drift(
        snap, snap,
        model_outputs={"a": text[:100], "b": text[:100]}
    )
    drift_time = time.time() - t0
    times["measure_drift"].append(drift_time)

print(f"model.invoke():     avg {sum(times['model_invoke'])/len(times['model_invoke'])*1000:.1f}ms")
print(f"snapshot():         avg {sum(times['snapshot'])/len(times['snapshot'])*1000:.1f}ms")
print(f"measure_drift():    avg {sum(times['measure_drift'])/len(times['measure_drift'])*1000:.1f}ms")
print()

results["performance"]["model_invoke_ms"] = sum(times['model_invoke'])/len(times['model_invoke'])*1000
results["performance"]["snapshot_ms"] = sum(times['snapshot'])/len(times['snapshot'])*1000
results["performance"]["measure_drift_ms"] = sum(times['measure_drift'])/len(times['measure_drift'])*1000

# ============================================================================
# TEST 2: EDGE CASES
# ============================================================================

print("[2/5] EDGE CASES")
print("-" * 80)

edge_cases = {
    "tiny_output": "OK",
    "long_output": "a" * 10000,
    "special_chars": "测试中文😀\n\t!@#$%^&*()",
    "repeated_words": "test " * 100,
    "empty": "",
}

for name, text in edge_cases.items():
    try:
        snap = drift_detector.snapshot(name, text, ["test"])
        snap2 = drift_detector.snapshot(name + "_2", text, ["test"])
        report = drift_detector.measure_drift(snap, snap2, model_outputs={})
        status = "✓ PASS"
        results["edge_cases"][name] = {"status": "pass", "drift": report.combined_drift_score}
    except Exception as e:
        status = f"✗ FAIL: {str(e)[:50]}"
        results["edge_cases"][name] = {"status": "fail", "error": str(e)[:50]}
        results["failures"].append({"test": f"edge_case_{name}", "error": str(e)[:100]})

    print(f"  {name:20s} {status}")

print()

# ============================================================================
# TEST 3: LOAD TEST (Rapid calls)
# ============================================================================

print("[3/5] LOAD TEST (100 rapid snapshots)")
print("-" * 80)

t0 = time.time()
load_snaps = []
for i in range(100):
    snap = drift_detector.snapshot(f"load_{i}", f"output {i}" * 10, ["test"])
    load_snaps.append(snap)

load_time = time.time() - t0
print(f"  100 snapshots in {load_time:.2f}sec")
print(f"  Rate: {100/load_time:.0f} snapshots/sec")

# Measure drifts between random pairs
drift_samples = []
for i in range(10):
    try:
        report = drift_detector.measure_drift(
            load_snaps[i], load_snaps[i+10],
            model_outputs={}
        )
        drift_samples.append(report.combined_drift_score)
    except Exception as e:
        results["failures"].append({"test": "load_test_drift", "error": str(e)[:100]})

if drift_samples:
    print(f"  Sample drifts (10 pairs): avg={sum(drift_samples)/len(drift_samples):.2f}")

results["load_test"]["snapshots_per_sec"] = 100/load_time
results["load_test"]["total_time_sec"] = load_time

print()

# ============================================================================
# TEST 4: MEMORY USAGE
# ============================================================================

print("[4/5] MEMORY TEST")
print("-" * 80)

tracemalloc.start()

# Create many snapshots
mem_snaps = []
for i in range(100):
    text = "x" * 5000  # 5KB per snapshot
    snap = drift_detector.snapshot(f"mem_{i}", text, ["test"])
    mem_snaps.append(snap)

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"  100 snapshots (5KB each): {peak / 1024 / 1024:.1f}MB peak")
print(f"  Avg per snapshot: {(peak / 100 / 1024):.1f}KB")

results["memory"]["peak_mb"] = peak / 1024 / 1024
results["memory"]["per_snapshot_kb"] = peak / 100 / 1024

print()

# ============================================================================
# TEST 5: ACCURACY UNDER STRESS (Same tests, many times)
# ============================================================================

print("[5/5] ACCURACY TEST (1000 drift measurements)")
print("-" * 80)

acc_pass = 0
acc_fail = 0

template = ChatPromptTemplate.from_template("Topic: {t}")

for i in range(100):
    try:
        # Get 10 different outputs
        outputs = []
        for j in range(10):
            resp = template.pipe(groq_model).invoke({"t": ["ML", "AI", "data", "code"][j % 4]})
            text = resp.content if hasattr(resp, 'content') else str(resp)
            outputs.append(text)

        # Measure 10 drifts
        for j in range(10):
            snap_a = drift_detector.snapshot(f"acc_{i}_{j}", outputs[j % len(outputs)], ["test"])
            snap_b = drift_detector.snapshot(f"acc_{i}_{j+1}", outputs[(j+1) % len(outputs)], ["test"])
            report = drift_detector.measure_drift(snap_a, snap_b, model_outputs={})

            # Check: drift should be 0 ≤ drift < 10
            if 0 <= report.combined_drift_score < 10:
                acc_pass += 1
            else:
                acc_fail += 1

    except Exception as e:
        acc_fail += 10
        results["failures"].append({"test": f"accuracy_iter_{i}", "error": str(e)[:100]})

print(f"  Passed: {acc_pass}/1000")
print(f"  Failed: {acc_fail}/1000")
print(f"  Accuracy: {acc_pass/(acc_pass+acc_fail)*100:.1f}%")

results["accuracy"] = {
    "pass": acc_pass,
    "fail": acc_fail,
    "accuracy_pct": acc_pass/(acc_pass+acc_fail)*100 if (acc_pass+acc_fail) > 0 else 0
}

print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("STRESS TEST RESULTS")
print("="*80)
print()

print("✓ PERFORMANCE:")
print(f"  measure_drift():  {results['performance']['measure_drift_ms']:.1f}ms (target <100ms) {'✓' if results['performance']['measure_drift_ms'] < 100 else '✗'}")
print(f"  snapshot():       {results['performance']['snapshot_ms']:.1f}ms (target <10ms) {'✓' if results['performance']['snapshot_ms'] < 10 else '✗'}")
print()

print("✓ EDGE CASES:")
for name, result in results["edge_cases"].items():
    status = "✓" if result["status"] == "pass" else "✗"
    print(f"  {name:20s} {status}")
print()

print("✓ LOAD:")
print(f"  Rate: {results['load_test']['snapshots_per_sec']:.0f} snapshots/sec")
print()

print("✓ MEMORY:")
print(f"  Peak: {results['memory']['peak_mb']:.1f}MB (100 snapshots)")
print()

print("✓ ACCURACY:")
print(f"  {results['accuracy']['accuracy_pct']:.1f}% ({results['accuracy']['pass']}/1000)")
print()

if results["failures"]:
    print("⚠️ FAILURES:")
    for failure in results["failures"][:5]:  # Show first 5
        print(f"  {failure['test']}: {failure['error']}")
    if len(results["failures"]) > 5:
        print(f"  ... and {len(results['failures'])-5} more")
else:
    print("✓ NO FAILURES")

print()

# Save results
with open("/tmp/stress_test_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"✓ Full results: /tmp/stress_test_results.json")
print()

# Final verdict
print("="*80)
print("VERDICT")
print("="*80)

verdict = "✅ PRODUCTION READY" if (
    results['performance']['measure_drift_ms'] < 100 and
    results['accuracy']['accuracy_pct'] > 99 and
    len(results['failures']) == 0
) else "⚠️ NEEDS FIXES"

print(f"{verdict}")
print()
