# Architecture

> Diagrams are in [Mermaid](https://mermaid.js.org/) — they render natively on
> GitHub/GitLab and in most Markdown viewers (VS Code, Obsidian, etc.).

## Task 1 — AI Assistant Core (RAG + Tools + LLM)

```mermaid
flowchart TB
    subgraph Client
        U[User via Streamlit UI / API client]
    end

    subgraph API["FastAPI Backend (app/main.py)"]
        EP["/chat endpoint"]
        RL[Rate Limiter]
        CL[Concurrency Limiter - asyncio.Semaphore]
        CACHE[(Response Cache - TTL)]
        LOOP[Tool-Calling Loop]
    end

    subgraph RAG["RAG Pipeline"]
        ING[Ingestion & Chunking\napp/rag/ingest.py]
        EMB[Embedding Model\nsentence-transformers]
        VS[(ChromaDB\nVector Store)]
        RET[Retriever\napp/rag/retriever.py]
    end

    subgraph Tools["Tool / Function Calling"]
        T1[calculator]
        T2[get_current_time]
        T3[get_weather]
        T4[rag_search]
    end

    subgraph LLM["LLM Provider Layer"]
        ORCH[Fallback Orchestrator]
        P1[OpenAI Cloud API]
        P2[Local vLLM Server\nOpenAI-compatible\nLlama 3 / Mistral]
        P3[Anthropic Claude API]
    end

    U -->|POST /chat| EP
    EP --> RL --> CACHE
    CACHE -->|miss| RET
    RET --> EMB
    RET --> VS
    EP --> LOOP
    LOOP --> CL --> ORCH
    ORCH -->|1st try| P1
    ORCH -->|fallback| P2
    ORCH -->|fallback| P3
    LOOP -->|model requests tool| Tools
    T4 --> RET
    Tools -->|tool result| LOOP
    LOOP -->|final answer| CACHE
    CACHE -->|response| U

    DOC[Raw Documents\n.txt / .pdf] -->|POST /ingest| ING --> EMB --> VS
```

---

## Task 2 — Production Deployment

```mermaid
flowchart TB
    subgraph Client
        Browser[Browser]
    end

    subgraph Compose["docker-compose.yml"]
        UI[ui container\nStreamlit :8501]
        API[api container\nFastAPI + Uvicorn :8000]
        VLLM["vllm container (optional, GPU profile)\nLlama 3 / Mistral :8001"]
        VOL[(chroma_data volume)]
    end

    subgraph Cloud["Cloud LLM Providers"]
        OA[OpenAI]
        AN[Anthropic]
    end

    Browser -->|:8501| UI
    UI -->|BACKEND_URL http://api:8000| API
    API <-->|persist| VOL
    API -->|fallback chain| OA
    API -->|fallback chain| VLLM
    API -->|fallback chain| AN

    subgraph Reliability["Reliability layer inside API"]
        direction LR
        R1[Rate Limiter] --> R2[Concurrency Semaphore] --> R3[Retry w/ backoff] --> R4[Provider Fallback] --> R5[Response Cache]
    end
```

| Concern | Implementation |
|---|---|
| Web UI | `ui/streamlit_app.py` — chat tab + agent tab |
| Concurrency | `asyncio` throughout; `ConcurrencyLimiter` bounds in-flight LLM calls; `/chat/batch` via `asyncio.gather` |
| Caching | `app/reliability/cache.py` — TTL + max-size cache |
| Retries | `app/reliability/retry.py` — tenacity, exponential backoff + jitter |
| Rate limiting | `app/reliability/rate_limiter.py` — sliding window per client |
| Fallback provider | `app/reliability/fallback.py` — ordered chain: cloud → local vLLM → secondary cloud |
| Graceful degradation | All-providers-failed path returns `200` with an explanatory message, never a crash |
| Containerization | `Dockerfile`, `docker/Dockerfile.ui`, `docker/Dockerfile.vllm`, `docker-compose.yml` |

### Why vLLM instead of ONNX for the generative model
ONNX Runtime is built for static-shape, single-pass inference (e.g. BERT/ResNet). Autoregressive LLM decoding has a dynamic, growing KV-cache that ONNX's static graph export handles poorly. vLLM's PagedAttention gives materially better real-world throughput/latency for concurrent LLM serving, so it is used for the locally-served model instead. ONNX conversion is still demonstrated for the embedding model (`scripts/export_embedding_onnx.py`), where it's a good fit.

---

## Task 3 — Agentic Loop (VerificationAgent)

```mermaid
flowchart TB
    U[User query] --> INIT[Build prompt:\nquery + compact structured notes\n+ actions already taken]
    INIT --> LLM[LLM call\nforced AgentStep JSON schema\nthought / action / action_input / confidence]
    LLM --> DECIDE{action?}

    DECIDE -->|rag_search| T1[Execute tool]
    DECIDE -->|wikipedia_search| T1
    DECIDE -->|calculator| T1
    DECIDE -->|get_weather| T1
    DECIDE -->|draft_answer| REC[Record draft, continue loop]
    DECIDE -->|ask_clarification| STOPQ[STOP:\nreturn question to user]
    DECIDE -->|finish| GATE{Self-check gate:\n>=2 independent sources\nAND confidence >= threshold?}

    T1 --> NOTE[Extract COMPACT structured note\nfrom raw tool result\nraw payload discarded]
    NOTE --> CAP[Cap notes list\nkeep most recent N]
    CAP --> LOOPCHK{Same action+input\nas previous step?}
    LOOPCHK -->|yes| STOPL[STOP: loop_detected]
    LOOPCHK -->|no| ITERCHK{iteration >= max_iterations?}

    GATE -->|yes| STOPF[STOP: finished\nreturn final_answer]
    GATE -->|no| REJECT[Reject finish,\nappend self-check note,\nforce another round]
    REJECT --> ITERCHK
    REC --> ITERCHK

    ITERCHK -->|yes| STOPM[STOP: max_iterations\nreturn best-available draft]
    ITERCHK -->|no| INIT

    style STOPF fill:#c8e6c9
    style STOPQ fill:#fff9c4
    style STOPL fill:#ffccbc
    style STOPM fill:#ffccbc
```

**Pattern:** single-agent loop (not multi-agent) — see README.md §Agentic Pattern for the justification. There is no coordination/hand-off structure to diagram because everything above runs inside one agent and one FastAPI request (`POST /agent/verify`); the diagram's branches are the agent's own decision points at each iteration, not separate agents.

**Context engineering, visualized above:** the `NOTE` → `CAP` step is where structured external notes are produced — this is what keeps the loop's prompt size roughly constant across iterations instead of growing with every raw tool payload (contrast with Task 1's `_run_tool_loop`, which does append full raw tool output to the message list each turn).

**Stopping conditions (loop cannot run indefinitely):** `finished` (self-check gate passed), `ask_clarification` (explicit user hand-off), `loop_detected` (identical action+input repeated), `max_iterations` (hard ceiling, `AGENT_MAX_ITERATIONS` env var, default 6), and `hard_failure` (the LLM layer itself — all providers — failed; see fallback chain in Task 2 diagram, which the agent also relies on for every LLM call).
