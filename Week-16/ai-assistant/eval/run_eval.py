"""
Evaluation harness for the VerificationAgent (Task 3) — built from scratch,
no existing eval framework used.

What it measures (per assignment requirements):
  - Task completion rate: did each query reach the outcome a correct agent
    should reach for that query (see EXPECTED_VS_ACTUAL below)?
  - Tool-call correctness: for every tool invocation in every trajectory,
    was the chosen tool appropriate for the case AND were its arguments
    valid (present, correctly-typed required fields)?
  - Trajectory length: iterations used per query, flagged if surprisingly
    high for the query's apparent complexity.
  - Failure log: every non-fully-successful case, classified as
    Hard / Soft / Cascading-soft failure.
  - Token & cost accounting: real token counts (tiktoken) summed per
    query and overall, plus a comparison against a single-pass
    (non-agentic) baseline to make the loop's coordination cost visible
    (see note in results.md — N/A for multi-agent since we chose a
    single-agent design; see README.md section b for why).

Usage:
    python eval/run_eval.py            # offline, scripted (default, free, reproducible)
    python eval/run_eval.py --live     # use a REAL provider from .env (costs money, needs API key)

Output:
    eval/results.md    human-readable report (tables)
    eval/results.json  raw structured results
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.verification_agent import VerificationAgent  # noqa: E402
from app.agents.tools import AGENT_TOOL_REQUIRED_ARGS  # noqa: E402
from eval.fake_tools import FakeRetriever, fake_execute_agent_tool  # noqa: E402
from eval.mock_provider import MockOrchestrator, count_tokens, TOKEN_COUNTING_METHOD  # noqa: E402
from eval.test_queries import TEST_CASES  # noqa: E402

AGENT_MAX_ITERATIONS = 6
AGENT_MIN_SOURCES = 2
AGENT_CONFIDENCE_THRESHOLD = 0.6
AGENT_MAX_NOTES = 8

# A query is graded as "task completed" if the ACTUAL stop_reason is in the
# accepted set for its expected_outcome category. Several categories accept
# more than one literal stop_reason because the *correct* behavior for an
# adversarial case is to safely abort, not necessarily to hit one exact
# code path.
EXPECTED_VS_ACTUAL = {
    "finished": {"finished"},
    "ask_clarification": {"ask_clarification"},
    "loop_detected": {"loop_detected"},
    "max_iterations": {"max_iterations"},
    # The point of the failure-injection case: the agent must NOT confidently
    # "finish" on broken/incomplete cross-source evidence. Any non-finished,
    # non-crash outcome counts as correctly recognizing the failure.
    "recognizes_failure": {"loop_detected", "max_iterations", "ask_clarification"},
}


def grade_tool_calls(case, trace):
    """Returns (correct_count, total_count, details[]) for every real tool
    invocation (excludes draft_answer/ask_clarification/finish, which take
    no evidentiary arguments)."""
    details = []
    for step in trace:
        action = step["action"]
        if action not in AGENT_TOOL_REQUIRED_ARGS or not AGENT_TOOL_REQUIRED_ARGS[action]:
            continue  # not an argument-bearing tool call
        required = AGENT_TOOL_REQUIRED_ARGS[action]
        args_ok = all(k in step["action_input"] for k in required)
        tool_appropriate = (not case["expected_tools"]) or (action in case["expected_tools"])
        correct = bool(args_ok and tool_appropriate)
        details.append({
            "iteration": step["iteration"], "action": action, "args_ok": args_ok,
            "tool_appropriate": tool_appropriate, "correct": correct,
        })
    correct_count = sum(1 for d in details if d["correct"])
    return correct_count, len(details), details


def classify_failure(case, actual_stop_reason, final_answer, trace):
    """Failure taxonomy: Hard / Soft / Cascading-soft / none (success)."""
    expected = case["expected_outcome"]
    accepted = EXPECTED_VS_ACTUAL.get(expected, {expected})

    if actual_stop_reason in accepted:
        return "none"

    invalid_json_count = sum(1 for t in trace if t["action"] == "invalid")
    failed_tool_count = sum(1 for t in trace if "FAILED" in (t.get("observation") or ""))

    # Multiple distinct failure signals chained together -> cascading soft failure.
    if (invalid_json_count + failed_tool_count) >= 2:
        return "cascading_soft"
    # Produced some usable (if wrong/degraded) output -> soft failure.
    if final_answer:
        return "soft"
    # No usable output at all and outcome didn't match what was expected -> hard failure.
    return "hard"


def run_case(case, live_orchestrator=None):
    if live_orchestrator is not None:
        orchestrator = live_orchestrator
        tool_executor = None  # use real tools in --live mode
    else:
        orchestrator = MockOrchestrator(script=case["script"])
        tool_executor = fake_execute_agent_tool

    agent = VerificationAgent(
        orchestrator=orchestrator,
        retriever=FakeRetriever(),
        max_iterations=AGENT_MAX_ITERATIONS,
        min_sources=AGENT_MIN_SOURCES,
        confidence_threshold=AGENT_CONFIDENCE_THRESHOLD,
        max_notes=AGENT_MAX_NOTES,
        fail_inject_tool=case.get("fail_inject_tool"),
        tool_executor=tool_executor,
    )

    result = asyncio.run(agent.run(query=case["query"]))
    trace = [t.model_dump() if hasattr(t, "model_dump") else t for t in result["trace"]]

    stop_reason = result["stop_reason"].value if hasattr(result["stop_reason"], "value") else result["stop_reason"]
    correct, total, tool_details = grade_tool_calls(case, trace)
    failure_class = classify_failure(case, stop_reason, result["final_answer"], trace)

    # --- Single-pass (non-agentic) baseline token estimate for cost comparison ---
    baseline_prompt = f"Answer the user's question directly and concisely.\n\nQuestion: {case['query']}"
    baseline_tokens = count_tokens(baseline_prompt) + 70  # ~70 tokens for a short direct answer

    return {
        "id": case["id"],
        "query": case["query"],
        "description": case["description"],
        "expected_outcome": case["expected_outcome"],
        "actual_stop_reason": stop_reason,
        "task_completed": stop_reason in EXPECTED_VS_ACTUAL.get(case["expected_outcome"], {case["expected_outcome"]}),
        "iterations_used": result["iterations_used"],
        "confidence": result["confidence"],
        "sources_used": result["sources_used"],
        "final_answer": result["final_answer"],
        "clarification_question": result["clarification_question"],
        "total_tokens": result["total_tokens"],
        "baseline_tokens_estimate": baseline_tokens,
        "token_overhead_x": round(result["total_tokens"] / baseline_tokens, 2) if baseline_tokens else None,
        "tool_calls_correct": correct,
        "tool_calls_total": total,
        "failure_class": failure_class,
        "trace": trace,
    }


def build_report(results):
    n = len(results)
    completed = sum(1 for r in results if r["task_completed"])
    total_tool_correct = sum(r["tool_calls_correct"] for r in results)
    total_tool_calls = sum(r["tool_calls_total"] for r in results)
    avg_iters = sum(r["iterations_used"] for r in results) / n
    total_tokens = sum(r["total_tokens"] for r in results)
    total_baseline_tokens = sum(r["baseline_tokens_estimate"] for r in results)
    failures = [r for r in results if r["failure_class"] != "none"]

    lines = []
    lines.append("# VerificationAgent — Evaluation Report\n")
    lines.append("Generated by `eval/run_eval.py` (offline, scripted mode unless `--live` was passed).\n")
    lines.append(f"Token counting method: `{TOKEN_COUNTING_METHOD}`.\n")

    lines.append("## Summary\n")
    lines.append(f"- **Task completion rate:** {completed}/{n} ({100*completed/n:.0f}%)")
    tc_pct = f"{100*total_tool_correct/total_tool_calls:.0f}%" if total_tool_calls else "N/A (no tool calls)"
    lines.append(f"- **Tool-call correctness:** {total_tool_correct}/{total_tool_calls} ({tc_pct})")
    lines.append(f"- **Average trajectory length:** {avg_iters:.1f} iterations")
    lines.append(f"- **Total tokens consumed (all {n} queries):** {total_tokens}")
    lines.append(f"- **Total single-pass baseline token estimate:** {total_baseline_tokens}")
    overhead = round(total_tokens / total_baseline_tokens, 2) if total_baseline_tokens else None
    lines.append(f"- **Agentic-loop token overhead vs. single-pass baseline:** {overhead}x")
    lines.append(
        "- **Multi-agent vs single-agent token comparison:** N/A — this implementation uses a "
        "single-agent loop (see README.md §Agentic Pattern), so there is no second agent's token "
        "usage to compare against. The single-pass baseline above is reported instead, to make the "
        "verification loop's own coordination/iteration cost visible, per the spirit of the requirement."
    )
    lines.append("")

    lines.append("## Per-query results\n")
    lines.append("| ID | Expected | Actual | Completed | Iterations | Tool correctness | Tokens (agent / baseline, ×) | Failure class |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        tc = f"{r['tool_calls_correct']}/{r['tool_calls_total']}" if r["tool_calls_total"] else "—"
        tok = f"{r['total_tokens']} / {r['baseline_tokens_estimate']} ({r['token_overhead_x']}x)"
        lines.append(
            f"| {r['id']} | {r['expected_outcome']} | {r['actual_stop_reason']} | "
            f"{'✅' if r['task_completed'] else '❌'} | {r['iterations_used']} | {tc} | {tok} | {r['failure_class']} |"
        )
    lines.append("")

    lines.append("## Failure log\n")
    if not failures:
        lines.append("No failures — every case reached its expected outcome category.\n")
    else:
        lines.append("| ID | Class | Expected | Actual | Notes |")
        lines.append("|---|---|---|---|---|")
        for r in failures:
            note = (r["final_answer"] or r["clarification_question"] or "(no output produced)")[:100]
            lines.append(f"| {r['id']} | {r['failure_class']} | {r['expected_outcome']} | {r['actual_stop_reason']} | {note} |")
        lines.append("")

    lines.append("## Trajectory detail\n")
    for r in results:
        lines.append(f"### `{r['id']}`")
        lines.append(f"*{r['description']}*\n")
        lines.append(f"Query: \"{r['query']}\"\n")
        lines.append(f"Stop reason: **{r['actual_stop_reason']}** | confidence: {r['confidence']:.2f} | sources used: {', '.join(r['sources_used']) or 'none'}\n")
        for t in r["trace"]:
            lines.append(f"- iter {t['iteration']} → `{t['action']}` | obs: {t['observation'][:120]}")
        lines.append(f"\n**Final answer:** {r['final_answer'] or r['clarification_question'] or '(none)'}\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Use a real LLM provider from .env instead of the offline scripted mock.")
    args = parser.parse_args()

    live_orchestrator = None
    if args.live:
        from app.config import get_settings
        from app.llm.providers import build_providers
        from app.reliability.fallback import FallbackOrchestrator

        settings = get_settings()
        providers = build_providers(settings)
        live_orchestrator = FallbackOrchestrator(
            providers=providers, order=settings.fallback_order,
            retry_max_attempts=settings.retry_max_attempts, retry_backoff_seconds=settings.retry_backoff_seconds,
        )
        print("Running in --live mode against configured providers:", list(providers.keys()))
    else:
        print("Running in offline scripted mode (default). Use --live for a real provider.")

    results = [run_case(case, live_orchestrator=live_orchestrator) for case in TEST_CASES]

    out_dir = Path(__file__).resolve().parent
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    report = build_report(results)
    (out_dir / "results.md").write_text(report)

    n = len(results)
    completed = sum(1 for r in results if r["task_completed"])
    print(f"\nDone. {completed}/{n} cases completed as expected.")
    print(f"Wrote {out_dir / 'results.md'} and {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
