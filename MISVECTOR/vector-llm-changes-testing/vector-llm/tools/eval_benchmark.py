"""
tools/eval_benchmark.py
=======================
Evaluation & Benchmarking suite for Vector LLM Microservice.
Evaluates query intent classification, Plotly chart generation, decision recommendations,
and response latency.

USAGE:
  python tools/eval_benchmark.py --host http://localhost:8001 --api_key test_key
"""

import argparse
import json
import time
import urllib.request
import urllib.error

TEST_QUERIES = [
    {
        "name": "Dashboard - Bar Chart",
        "endpoint": "/query",
        "payload": {"query": "Show monthly sales as a bar chart", "stream": False},
        "expected_intent": "dashboard",
        "validate_fn": lambda res: "data" in res and "chart_type" in res
    },
    {
        "name": "Decision - Strategy Recommendation",
        "endpoint": "/query",
        "payload": {"query": "Should we expand our sales team in Q4?", "stream": False},
        "expected_intent": "decision",
        "validate_fn": lambda res: "recommendation" in res and "confidence" in res
    },
    {
        "name": "General Question - Fact Retrieval",
        "endpoint": "/query",
        "payload": {"query": "What is our company policy on remote work?", "stream": False},
        "expected_intent": "question",
        "validate_fn": lambda res: len(res.get("answer", "")) > 0
    },
    {
        "name": "Direct Dashboard Spec Endpoint",
        "endpoint": "/dashboard",
        "payload": {"query": "Line chart of website traffic 2024", "raw_data": "Month,Visits\nJan,10000\nFeb,15000\nMar,22000"},
        "expected_intent": None,
        "validate_fn": lambda res: res.get("chart_type") == "line" and "confidence" in res
    },
    {
        "name": "Direct Decision Recommendation Endpoint",
        "endpoint": "/decide",
        "payload": {"query": "Which region has better ROI: East or West?", "csv_data": "Region,Revenue,Cost\nEast,100000,50000\nWest,80000,30000"},
        "expected_intent": None,
        "validate_fn": lambda res: "recommendation" in res and "key_metrics" in res
    }
]


def run_benchmark(host: str, api_key: str):
    print(f"\n=======================================================")
    print(f"   VECTOR LLM MICROSERVICE EVALUATION & BENCHMARK")
    print(f"   Target Host: {host}")
    print(f"=======================================================\n")

    # 1. Health Check
    health_url = f"{host.rstrip('/')}/health"
    try:
        req = urllib.request.Request(health_url)
        with urllib.request.urlopen(req) as resp:
            health_data = json.loads(resp.read().decode())
            print(f"[HEALTH CHECK] Status: {health_data.get('status')} | Active Model: {health_data.get('model')}")
            print(f"               Ollama: {health_data.get('ollama', {}).get('ollama_running')} | ChromaDB: {health_data.get('chromadb', {}).get('chroma_running')}\n")
    except Exception as e:
        print(f"[ERROR] Could not connect to microservice at {health_url}: {e}")
        print("Ensure the microservice is running via: cd vector-llm; uvicorn main:app --port 8001")
        return

    results = []
    total_latency = 0

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key
    }

    for test in TEST_QUERIES:
        url = f"{host.rstrip('/')}{test['endpoint']}"
        payload_bytes = json.dumps(test["payload"]).encode("utf-8")
        req = urllib.request.Request(url, data=payload_bytes, headers=headers, method="POST")

        start_time = time.time()
        try:
            with urllib.request.urlopen(req) as resp:
                latency_ms = round((time.time() - start_time) * 1000, 2)
                total_latency += latency_ms
                res_data = json.loads(resp.read().decode())

                # If wrapped in /query response, parse the inner answer string if JSON
                parsed_ans = res_data
                if test["endpoint"] == "/query" and "answer" in res_data:
                    try:
                        parsed_ans = json.loads(res_data["answer"])
                    except Exception:
                        parsed_ans = res_data

                valid = test["validate_fn"](parsed_ans)
                intent_match = True
                if test["expected_intent"]:
                    intent_match = res_data.get("intent") == test["expected_intent"]

                passed = valid and intent_match
                results.append({
                    "name": test["name"],
                    "passed": passed,
                    "latency_ms": latency_ms,
                    "intent": res_data.get("intent", "N/A"),
                    "confidence": res_data.get("confidence", parsed_ans.get("confidence", 1.0))
                })

                status_str = "PASS" if passed else "FAIL"
                print(f"[{status_str}] {test['name']:<40} | Latency: {latency_ms:>7.1f} ms | Confidence: {res_data.get('confidence', 1.0)}")

        except urllib.error.HTTPError as e:
            latency_ms = round((time.time() - start_time) * 1000, 2)
            print(f"[FAIL] {test['name']:<40} | HTTP {e.code}: {e.reason}")
            results.append({"name": test["name"], "passed": False, "latency_ms": latency_ms, "intent": "ERROR", "confidence": 0.0})
        except Exception as e:
            latency_ms = round((time.time() - start_time) * 1000, 2)
            print(f"[FAIL] {test['name']:<40} | Error: {e}")
            results.append({"name": test["name"], "passed": False, "latency_ms": latency_ms, "intent": "ERROR", "confidence": 0.0})

    passed_count = sum(1 for r in results if r["passed"])
    avg_latency = round(total_latency / len(results), 2) if results else 0.0

    print(f"\n-------------------------------------------------------")
    print(f" BENCHMARK SUMMARY")
    print(f" Tests Passed: {passed_count}/{len(results)} ({passed_count/len(results)*100:.1f}%)")
    print(f" Average Latency: {avg_latency} ms")
    print(f"-------------------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Vector LLM Microservice")
    parser.add_argument("--host", default="http://localhost:8001", help="Microservice host URL")
    parser.add_argument("--api_key", default="dev-key-do-not-use-in-prod", help="API key for authentication")
    args = parser.parse_args()
    run_benchmark(args.host, args.api_key)
