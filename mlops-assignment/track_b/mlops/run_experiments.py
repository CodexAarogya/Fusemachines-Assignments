"""Track B: run each prompt version through the regression set, trace every step, judge with Evidently,
log everything to MLflow, and apply the promotion gate. Usage: uv run python -m mlops.run_experiments"""
import asyncio, json, sys
from pathlib import Path
import mlflow, pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import CategoryCount
from evidently.tests import gte

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.agents.verification_agent import VerificationAgent
from eval.fake_tools import FakeRetriever, fake_execute_agent_tool
from eval.run_eval import EXPECTED_VS_ACTUAL, grade_tool_calls
from mlops.judge import judge_correctness, judge_disclosure
from mlops.policy import PromptPolicyOrchestrator

mlflow.set_tracking_uri(f"sqlite:///{ROOT/'mlflow.db'}")
CFG = dict(model="policy-sim (offline surrogate)", temperature=0.2, max_iterations=6, min_sources=2,
           confidence_threshold=0.6, max_notes=8, retrieval_top_k=4, chunk_size=800)
PASS_GATE = 80.0  # promote a version only if >= 80% of regression checks pass
CASES = json.loads((ROOT / "mlops" / "regression_set.json").read_text())
out = ROOT / "reports"; out.mkdir(exist_ok=True)


def step_records(trace):  # {step, tool, args, result, reasoning} per loop iteration
    return [dict(step=t["iteration"], tool=t["action"], args=t["action_input"], result=t.get("raw_result"),
                 observation=t["observation"], reasoning=t["thought"], tokens=t["tokens_used"]) for t in trace]


def run_version(version, min_sources):
    prompt = (ROOT / "prompts" / f"prompt_{version}.txt").read_text()
    rows, traces = [], []
    for c in CASES:
        agent = VerificationAgent(orchestrator=PromptPolicyOrchestrator(), retriever=FakeRetriever(), tool_executor=fake_execute_agent_tool,
                                  max_iterations=CFG["max_iterations"], min_sources=min_sources, confidence_threshold=CFG["confidence_threshold"],
                                  max_notes=CFG["max_notes"], fail_inject_tool=c["fail_inject_tool"], system_prompt=prompt)
        r = asyncio.run(agent.run(c["query"]))
        trace = [t.model_dump() for t in r["trace"]]
        stop = r["stop_reason"].value
        resp = r["final_answer"] or r["clarification_question"]
        ok, tot, _ = grade_tool_calls(c, trace)
        v1, w1 = judge_correctness(resp, c, stop); v2, w2 = judge_disclosure(resp, c, r["sources_used"])
        rows.append(dict(id=c["id"], query=c["query"], response=resp or "", reference=c["reference"], stop_reason=stop,
                         completed=stop in EXPECTED_VS_ACTUAL[c["expected_outcome"]], iterations=r["iterations_used"], tokens=r["total_tokens"],
                         latency_ms=r["latency_ms"], tool_ok=ok, tool_total=tot, correctness=v1, correctness_reason=w1, disclosure=v2, disclosure_reason=w2))
        traces.append(dict(case=c["id"], stop_reason=stop, iterations=r["iterations_used"], final=resp, steps=step_records(trace)))
    return prompt, pd.DataFrame(rows), traces


def evidently_suite(df, version):
    dd = DataDefinition(categorical_columns=["correctness", "disclosure"], text_columns=["query", "response", "reference"])
    ds = Dataset.from_pandas(df[["query", "response", "reference", "correctness", "disclosure"]], data_definition=dd)
    rep = Report([CategoryCount(column="correctness", category="correct", share_tests=[gte(PASS_GATE / 100)]),
                  CategoryCount(column="disclosure", category="correct", share_tests=[gte(PASS_GATE / 100)])], include_tests=True)
    snap = rep.run(ds, None)
    path = out / f"regression_{version}.html"; snap.save_html(str(path))
    tests = json.loads(snap.json()).get("tests", [])
    return path, tests


def main():
    mlflow.set_experiment("assistant-prompt-versions")
    summary = []
    # min_sources=1 isolates the PROMPT effect (no code-level guard); min_sources=2 is the production W16 gate.
    for v, gate in [("v1", 1), ("v2", 1), ("v3", 1), ("v1", 2), ("v3", 2)]:
        tag = f"{v}_gate{gate}"
        prompt, df, traces = run_version(v, gate)
        html, tests = evidently_suite(df, tag)
        checks = len(df) * 2
        passed = int((df.correctness == "correct").sum() + (df.disclosure == "correct").sum())
        m = dict(task_completion_rate=df.completed.mean(), tool_call_correctness=df.tool_ok.sum() / max(1, df.tool_total.sum()),
                 avg_iterations=df.iterations.mean(), total_tokens=int(df.tokens.sum()), avg_latency_ms=df.latency_ms.mean(),
                 pct_tests_passed=100 * passed / checks, correctness_pass_rate=(df.correctness == "correct").mean(),
                 disclosure_pass_rate=(df.disclosure == "correct").mean(), evidently_tests_passed=sum(1 for t in tests if t.get("status") == "SUCCESS"))
        promoted = m["pct_tests_passed"] >= PASS_GATE
        with mlflow.start_run(run_name=f"prompt_{tag}"):
            mlflow.log_params({"prompt_version": v, "prompt_chars": len(prompt), **{**CFG, "min_sources": gate}})
            mlflow.set_tags({"promotion": "approved" if promoted else "BLOCKED (regression)", "judge": "heuristic-offline"})
            mlflow.log_metrics(m)
            mlflow.log_text(prompt, f"prompt_{v}.txt")
            mlflow.log_dict(traces, "traces/all_traces.json")
            good = next((t for t in traces if t["stop_reason"] == "finished"), traces[0]); bad = [t for t, r in zip(traces, df.to_dict("records")) if r["correctness"] == "incorrect"]
            mlflow.log_dict(good, "traces/representative_success.json")
            for t in bad[:2]: mlflow.log_dict(t, f"traces/failure_{t['case']}.json")
            df.to_csv(out / f"regression_results_{tag}.csv", index=False); mlflow.log_artifact(str(out / f"regression_results_{tag}.csv"))
            mlflow.log_artifact(str(html))
        summary.append(dict(version=tag, **{k: round(x, 3) for k, x in m.items()}, gate="PROMOTE" if promoted else "BLOCK"))
        json.dump(traces, open(out / f"traces_{tag}.json", "w"), indent=1, default=str)
    s = pd.DataFrame(summary); s.to_csv(out / "prompt_version_comparison.csv", index=False)
    (out / "prompt_version_comparison.md").write_text(s.to_markdown(index=False)); print(s.to_string(index=False))


if __name__ == "__main__":
    main()
