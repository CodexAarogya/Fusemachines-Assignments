# AI Assistant — RAG + Tool Calling + Production Deployment

An end-to-end AI assistant covering both assignment tasks:

- **Task 1 (Applied AI):** LLM integration, prompt engineering, structured
  JSON output, tool/function calling, a full RAG pipeline (chunking →
  embeddings → vector DB), local open-source model serving via vLLM, and
  Docker packaging.
- **Task 2 (Engineering AI Systems):** a Streamlit web UI on top of the same
  backend, async/concurrent request handling, caching, retries, rate
  limiting, multi-provider fallback, graceful degradation, and Docker Compose
  deployment.

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for diagrams of both tasks.

## Project Layout

```
ai-assistant/
├── app/
│   ├── main.py                 # FastAPI app: /chat, /chat/batch, /ingest, /health
│   ├── config.py                # env-driven settings
│   ├── schemas.py                # Pydantic request/response + structured-output schema
│   ├── llm/
│   │   ├── providers.py         # OpenAI / Anthropic / local-vLLM clients (unified interface)
│   │   └── tools.py              # function-calling tool definitions + executor
│   ├── rag/
│   │   ├── ingest.py             # loading + recursive chunking
│   │   ├── embeddings.py         # sentence-transformers wrapper
│   │   ├── vector_store.py       # ChromaDB wrapper
│   │   └── retriever.py          # ties embeddings + vector store together
│   ├── reliability/
│   │   ├── retry.py              # tenacity exponential backoff
│   │   ├── rate_limiter.py       # sliding-window limiter + concurrency semaphore
│   │   ├── cache.py              # TTL response cache
│   │   └── fallback.py           # multi-provider fallback orchestrator
│   └── utils/logging_config.py
├── ui/streamlit_app.py           # web UI (Task 2)
├── scripts/export_embedding_onnx.py
├── data/sample_docs/sample.txt   # demo document for RAG
├── tests/test_api.py
├── Dockerfile                     # API image
├── docker/Dockerfile.ui           # UI image
├── docker/Dockerfile.vllm         # local model server image (GPU)
├── docker-compose.yml
├── ARCHITECTURE.md
└── .env.example
```

## Quickstart (local, no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add at least one of: OPENAI_API_KEY or ANTHROPIC_API_KEY
# (or leave both blank and rely on a running vLLM instance for LOCAL_LLM_BASE_URL)

uvicorn app.main:app --reload --port 8000
```

In a second terminal:
```bash
streamlit run ui/streamlit_app.py
```

Open http://localhost:8501, upload `data/sample_docs/sample.txt` from the
sidebar, then ask e.g. *"What does vLLM use to improve throughput?"* and
watch it cite the ingested document.

## Quickstart (Docker Compose)

```bash
cp .env.example .env   # fill in API keys
docker compose up --build
```

- API: http://localhost:8000 (docs at `/docs`)
- UI: http://localhost:8501

To also run a local open-source model via vLLM (requires an NVIDIA GPU +
`nvidia-container-toolkit`):
```bash
docker compose --profile local-llm up --build
```
This starts a third container serving `meta-llama/Meta-Llama-3-8B-Instruct`
on an OpenAI-compatible endpoint at `:8001`, wired in as the `local` entry
in `PROVIDER_FALLBACK_ORDER`.

No GPU available? Leave the `local-llm` profile off and set
`PROVIDER_FALLBACK_ORDER=openai,anthropic` in `.env` — the app degrades
cleanly to cloud-only providers.

## API Reference

### `POST /chat`
```json
{
  "message": "What is RAG?",
  "use_rag": true,
  "structured": false,
  "temperature": 0.3,
  "top_p": 0.9
}
```
Response:
```json
{
  "answer": "...",
  "structured_output": null,
  "citations": [{"source": "sample.txt", "chunk_id": "sample-0-abc123", "score": 0.83}],
  "tool_calls": [],
  "provider_used": "openai",
  "cached": false,
  "latency_ms": 842.1
}
```

Set `"structured": true` to force the model to return validated JSON
matching the `QueryAnalysis` schema in `app/schemas.py` (intent, sentiment,
entities, requires_followup, summary) — demonstrates guaranteed-valid
structured output with automatic one-shot repair on validation failure.

### `POST /chat/batch`
Accepts a JSON array of the same request body and processes them
concurrently (bounded by `MAX_CONCURRENT_REQUESTS`), returning an array of
responses — used for batch/concurrent load.

### `POST /ingest`
Multipart file upload (`.txt`, `.md`, `.pdf`). Chunks and embeds the
document into the persistent Chroma collection.

### `GET /health`
Reports reachability of every configured LLM provider.

## How Each Requirement Is Met

### Task 1
| Requirement | Where |
|---|---|
| LLM integration (major provider) | `app/llm/providers.py` — OpenAI + Anthropic clients |
| Prompt engineering / temperature / top_p | `SYSTEM_PROMPT` in `main.py`; `temperature`/`top_p` are per-request tunable and env-configurable |
| Structured JSON output | `QueryAnalysis` schema + `response_format=json_schema` (OpenAI) / schema-in-prompt (Anthropic) + Pydantic validation with repair retry |
| Tool/function calling | `app/llm/tools.py` (`calculator`, `get_current_time`, `get_weather`, `rag_search`) + tool-calling loop in `main.py` |
| RAG: ingestion & chunking | `app/rag/ingest.py` — recursive character chunking with overlap |
| RAG: embeddings + vector DB | `app/rag/embeddings.py` (sentence-transformers) + `app/rag/vector_store.py` (ChromaDB, persisted) |
| Local open-source model via vLLM | `docker/Dockerfile.vllm` + `local` provider entry (OpenAI-compatible client pointed at vLLM) |
| Containerization | `Dockerfile`, `docker-compose.yml` |

### Task 2
| Requirement | Where |
|---|---|
| Web UI connected to backend | `ui/streamlit_app.py` |
| ONNX / justification | `scripts/export_embedding_onnx.py` exports the embedding model to ONNX; generative LLM uses vLLM instead — justification in `ARCHITECTURE.md` |
| Concurrent/async/batch requests | Fully `async def` FastAPI handlers; `ConcurrencyLimiter` (semaphore); `/chat/batch` via `asyncio.gather` |
| Latency/throughput optimization | Response caching, bounded concurrency, vLLM PagedAttention for local serving |
| Prompt/response caching | `app/reliability/cache.py` |
| Retry mechanism | `app/reliability/retry.py` (tenacity, exponential backoff + jitter) |
| Rate limiting | `app/reliability/rate_limiter.py` (sliding window per client) |
| Fallback model/provider | `app/reliability/fallback.py` (ordered chain, e.g. `openai → local vLLM → anthropic`) |
| Error handling & graceful degradation | All-providers-failed path returns a normal `200` with an explanatory message rather than crashing; validation-failure repair retry for structured output |
| Dockerized full app | `docker-compose.yml` (api + ui + optional vllm, with a persisted volume) |
| Deployment instructions | This README + "Deploying to the Cloud" below |

## Deploying to the Cloud (bonus)

**Azure Container Apps** (no GPU needed if using cloud LLM providers only):
```bash
az group create -n ai-assistant-rg -l eastus
az acr create -n aiassistantacr -g ai-assistant-rg --sku Basic
az acr build -r aiassistantacr -t ai-assistant-api:latest -f Dockerfile .
az acr build -r aiassistantacr -t ai-assistant-ui:latest -f docker/Dockerfile.ui .

az containerapp env create -n ai-assistant-env -g ai-assistant-rg -l eastus

az containerapp create -n ai-assistant-api -g ai-assistant-rg \
  --environment ai-assistant-env \
  --image aiassistantacr.azurecr.io/ai-assistant-api:latest \
  --target-port 8000 --ingress external \
  --env-vars OPENAI_API_KEY=secretref:openai-key PROVIDER_FALLBACK_ORDER=openai,anthropic

az containerapp create -n ai-assistant-ui -g ai-assistant-rg \
  --environment ai-assistant-env \
  --image aiassistantacr.azurecr.io/ai-assistant-ui:latest \
  --target-port 8501 --ingress external \
  --env-vars BACKEND_URL=https://ai-assistant-api.<env-domain>
```

**AWS**: push both images to ECR, deploy with ECS Fargate (task definitions
mirroring the two services in `docker-compose.yml`; use an EFS volume for
`chroma_data`); put the API behind an ALB.

**GCP**: `gcloud builds submit` to Artifact Registry, then `gcloud run deploy`
for both the API and UI as separate Cloud Run services (Cloud Run doesn't
support GPUs on the standard tier, so run vLLM on GCE with a GPU, or skip it
and rely on cloud providers only).

## Configuration Reference

All settings are environment variables (see `.env.example`), including
provider API keys, `PROVIDER_FALLBACK_ORDER`, generation params
(`TEMPERATURE`, `TOP_P`, `MAX_TOKENS`), RAG params (`CHUNK_SIZE`,
`CHUNK_OVERLAP`, `RETRIEVAL_TOP_K`), and reliability knobs
(`RATE_LIMIT_PER_MINUTE`, `MAX_CONCURRENT_REQUESTS`, `CACHE_TTL_SECONDS`,
`RETRY_MAX_ATTEMPTS`).

## Testing

```bash
pytest tests/ -v
```
Covers chunking, the calculator tool's safe-eval sandboxing, the response
cache, and the rate limiter without requiring live API keys. One
end-to-end test (`test_chat_endpoint_live`) is skipped unless
`OPENAI_API_KEY` is set and a server is running, for a full-stack smoke test.

## Known Limitations / Next Steps

- The rate limiter and cache are in-process; for multi-instance
  horizontal scaling, back them with Redis (noted inline in the code).
- vLLM requires a GPU; CPU-only environments should rely on the cloud
  provider fallback chain (`openai,anthropic`) and treat the `local-llm`
  Compose profile as optional.
- Streaming responses (token-by-token) aren't wired into the UI yet —
  the OpenAI/Anthropic clients support `stream=True` and would be a
  natural follow-up for perceived-latency improvements.
