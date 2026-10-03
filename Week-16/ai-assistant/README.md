# AI Assistant — RAG + Tool Calling + Production Deployment + Agentic Verification Loop

An end-to-end AI assistant covering three assignments:

- **Task 1 (Applied AI):** LLM integration, prompt engineering, structured JSON output, tool/function calling, a full RAG pipeline, local open-source model serving via vLLM, Docker packaging.
- **Task 2 (Engineering AI Systems):** a Streamlit web UI, async/concurrent request handling, caching, retries, rate limiting, multi-provider fallback, graceful degradation, Docker Compose deployment.
- **Task 3 (Agentify the Assistant):** a single-agent, cross-source-verification agentic loop with structured-notes context engineering, a from-scratch evaluation harness, and a failure-injection test.

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for diagrams of all three.

## Project Layout

```
ai-assistant/
├── app/
│   ├── main.py                      # FastAPI: /chat, /chat/batch, /ingest, /health, /agent/verify
│   ├── config.py                     # env-driven settings
│   ├── schemas.py                     # Pydantic schemas (chat + agent)
│   ├── llm/
│   │   ├── providers.py              # OpenAI / Anthropic / local-vLLM clients + token usage
│   │   └── tools.py                   # W15 tool-calling tools (calculator, weather, time, rag_search)
│   ├── rag/                           # ingest, embeddings, vector_store, retriever
│   ├── reliability/                   # retry, rate_limiter, cache, fallback
│   └── agents/                        # Task 3
│       ├── tools.py                   # agent tools incl. wikipedia_search + failure-injection hook
│       └── verification_agent.py      # the agentic loop itself
├── ui/streamlit_app.py                # web UI: Chat tab (W15) + Verification Agent tab (W16)
├── eval/                              # Task 3 evaluation harness (built from scratch)
│   ├── test_queries.py                # 9 scripted test cases
│   ├── mock_provider.py               # deterministic offline LLM stand-in + token counting
│   ├── fake_tools.py                  # deterministic offline tools/retriever
│   ├── run_eval.py                    # harness runner -> results.md / results.json
│   ├── results.md                     # generated report (already run, see below)
│   └── results.json
├── scripts/export_embedding_onnx.py
├── data/sample_docs/sample.txt
├── tests/test_api.py, test_agent.py
├── Dockerfile, docker/Dockerfile.ui, docker/Dockerfile.vllm, docker-compose.yml
├── ARCHITECTURE.md
└── .env.example
```

## Quickstart (local, no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# add at least one of OPENAI_API_KEY / ANTHROPIC_API_KEY, or run a local vLLM server

uvicorn app.main:app --reload --port 8000
# in a second terminal:
streamlit run ui/streamlit_app.py
```

Open http://localhost:8501. Use the **Chat** tab for W15 RAG+tools, or the
**Verification Agent** tab for the Task 3 agentic loop — try: *"What
technique does vLLM use to improve throughput, and is that confirmed by an
external source?"*

## Quickstart (Docker Compose)

```bash
cp .env.example .env   # fill in API keys
docker compose up --build
```
API: http://localhost:8000/docs · UI: http://localhost:8501. GPU owners can add `--profile local-llm` to also run vLLM locally; otherwise set `PROVIDER_FALLBACK_ORDER=openai,anthropic`.

## API Reference (new in this update)

### `POST /agent/verify`
```json
{"query": "What technique does vLLM use, confirmed by an external source?", "session_id": "s1"}
```
Returns `final_answer` (or `clarification_question` if the agent needs more
info from you — resume by POSTing again with `clarification_answer` set and
the same `session_id`), `stop_reason`, `confidence`, `sources_used`,
`notes` (the structured-notes artifact), the full `trace`, and
`total_tokens`. See `/docs` for the full schema. Existing `/chat`,
`/chat/batch`, `/ingest`, `/health` endpoints are unchanged from W15.

---

# Task 3 Documentation

## a. Context Engineering Technique

**Technique used: Structured external notes.**

**Where applied:** `app/agents/verification_agent.py`, `_extract_note()` +
`_cap_notes()`. After every tool call, the raw tool payload (a full
Wikipedia extract, a full RAG chunk, a full error object) is immediately
converted into one short `EvidenceNote` (source + 1–2 sentence content)
and the raw payload is discarded. Only these compact notes — capped at
`AGENT_MAX_NOTES` (default 8, keeping the most recent) — are serialized
into the next iteration's prompt, alongside the original query and a
list of action names already tried.

**Problem it solves:** the W15 tool-calling loop (`app/main.py`,
`_run_tool_loop`) appends the full raw tool result into the message list
on every turn — fine for a loop bounded at 4 iterations, but a
cross-source verification agent may run several rounds of retrieval
across two different sources, each returning hundreds of characters of
raw text. Carrying all of that forward verbatim causes **context
saturation**: the prompt grows every iteration, token cost grows with
it, and eventually the original question and the stopping rules get
diluted among stale raw payloads the model has already acted on. Compact
notes keep the prompt size roughly constant regardless of how many
iterations a verification run takes, which matters specifically because
this feature's whole point is to allow a variable, discovery-driven
number of iterations.

## b. Agentic Pattern

**Single-agent loop**, not a multi-agent system.

**Why:** the task is bounded — five tools, one domain, one coherent
objective (verify-then-answer) — so the benefits that justify
multi-agent coordination don't apply here. Using the five structural
failures as a checklist: **context saturation** is already addressed
directly by structured notes, not by farming verbose exploration out to
a sub-agent; **sequential bottleneck** is acceptable because evidence
gathering here is inherently sequential anyway (you usually want the
internal-KB result before deciding whether an external check is even
needed); **skill dilution** isn't a risk with only five simple,
unambiguous tools in one system prompt; **self-verification paradox**
(the model grading its own "I'm done") is handled in code, not by a
second LLM: the orchestrator's self-check gate (`action == "finish"`
branch) *programmatically* enforces `>= min_sources` independent
sources and a confidence floor before trusting a "finish" decision,
rather than asking a second agent to approve the first agent's answer;
**single point of failure** is a real limitation of any single LLM
call, but it's mitigated the same way Task 2 mitigates it everywhere
else in this app — the multi-provider `FallbackOrchestrator` sits under
every `orchestrator.chat()` call the agent makes. Splitting this into
a "gatherer" agent and a "verifier" agent would roughly double the
token cost per query (see the harness's token-overhead numbers) for a
task that doesn't need context isolation (one coherent context is
fine here) or parallelization (sources are cheap and fast to check
sequentially) to work well. A single-agent design was the better fit.

## c. Evaluation Harness

Built from scratch in `eval/` (no existing eval framework). It runs the
**real** `VerificationAgent` against 9 scripted offline trajectories
(`eval/test_queries.py`) via a deterministic `MockOrchestrator`
(`eval/mock_provider.py`) and deterministic fake tools
(`eval/fake_tools.py`), so results are free, reproducible, and need no
network or API key. Run it with `python eval/run_eval.py` (add `--live`
to instead run against a real configured provider). Full generated
output is already committed at `eval/results.md` / `eval/results.json`
— **current run: 9/9 cases completed as expected, 19/20 (95%) tool
calls correct, 3.3 average iterations.** Summary of what's measured:

| Metric | How it's computed |
|---|---|
| Task completion rate | actual `stop_reason` compared against each case's expected outcome category (`eval/run_eval.py::EXPECTED_VS_ACTUAL`) |
| Tool-call correctness | per tool-call step: required arguments present AND tool was an appropriate choice for that case (`grade_tool_calls`) |
| Trajectory length | `iterations_used` per case, reported per-query and averaged |
| Failure log | every non-matching case classified Hard / Soft / Cascading-soft (`classify_failure`) — current run has none, see `eval/results.md` §Failure log |
| Token/cost accounting | real token counts (tiktoken, falling back to a documented char/4 estimate when offline) summed per query and overall, compared against a single-pass baseline estimate |

## Additional Requirements

**1. Skill vs. Agent.** `wikipedia_search` and `calculator` are simple,
deterministic, single-step lookups with no decision-making of their own
— they *could* have been written as a Skill (a documented procedure
the model follows) rather than a tool, but we kept them as tools because
they still need to be dynamically selected and sequenced by the agent
mid-loop based on what it has already found, which is exactly what tool
calls (not static Skill instructions) are for. No new sub-agent was
added for this feature at all: a second agent would only have made
sense for context isolation/parallelization/specialization (§b), none
of which this bounded, single-domain task needs.

**2. Token and Cost Accounting.** Recorded per query and overall in
`eval/results.md`. Current run: **16,118 total tokens across 9 queries**
(single-agent loop) vs. **952 tokens** for a single-pass baseline
estimate of the same 9 queries — roughly **17x overhead**, which is the
visible cost of iterative, multi-source verification versus answering
once. A multi-agent-vs-single-agent comparison is **N/A**, since §b
justifies a single-agent design — the baseline comparison above is
reported instead, in the same spirit, to make the loop's own
coordination/iteration cost visible.

**3. Failure Injection Test.** `AGENT_FAIL_INJECT_TOOL` (env var) /
`fail_inject_tool` param forces a named tool to fail. Test case
`failure_injection_wikipedia_down` forces `wikipedia_search` to fail
after `rag_search` already succeeded, leaving only one of the two
required independent sources available. **Observed behavior:** the
agent does *not* confidently answer "confirmed externally" — the
self-check gate (§b) rejects its "finish" attempt (only 1/2 sources),
forcing another round; because the only independent second source is
down, it cannot actually gather a second source, and the run correctly
terminates via `loop_detected` rather than fabricating cross-source
confirmation it doesn't have. See `eval/results.md` for the full trace.

**4. Tool vs. Agent Boundary.** `wikipedia_search` calls a stateless,
single-request external REST API (one HTTP call in, one JSON summary
out) — there is no multi-step conversation or persistent state on the
other end, so it is modeled as a **bounded tool call**, not an
agent-to-agent interaction. If a future version needed to drive a
genuinely multi-step, stateful external service (e.g. a separate
research agent that itself plans and executes several searches before
replying), that would call for an agent-to-agent pattern instead, since
treating a multi-step process as a single tool call would hide
intermediate decisions this agent's self-check gate needs to see.

---

## How Each Task 1/2 Requirement Is Met

(unchanged from the W15 submission — summarized here for convenience)

| Requirement | Where |
|---|---|
| LLM integration | `app/llm/providers.py` |
| Structured JSON output | `QueryAnalysis` schema + `response_format=json_schema` + Pydantic validation w/ repair retry |
| Tool/function calling | `app/llm/tools.py` + tool loop in `main.py` |
| RAG ingestion/chunking/embeddings/vector DB | `app/rag/*` |
| Local model via vLLM | `docker/Dockerfile.vllm` + `local` provider |
| Web UI | `ui/streamlit_app.py` |
| Concurrency/async/batch | async FastAPI, `ConcurrencyLimiter`, `/chat/batch` |
| Caching / Retries / Rate limiting / Fallback / Graceful degradation | `app/reliability/*` |
| Containerization | `Dockerfile`, `docker-compose.yml` |

## Testing

```bash
pytest tests/ -v            # 16 tests: W15 pieces + Task 3 agent control-flow
python eval/run_eval.py     # Task 3 evaluation harness (offline, free, reproducible)
```

## Known Limitations / Next Steps

- Rate limiter, cache, and agent-session store are in-process; back them
  with Redis for multi-instance deployments.
- vLLM requires a GPU; CPU-only environments should rely on the cloud
  fallback chain.
- `wikipedia_search` hits the live Wikipedia API in production; the eval
  harness intentionally uses a fixture-based fake instead so it stays
  offline and reproducible (see `eval/fake_tools.py`).
- Streaming responses aren't wired into the UI yet.
