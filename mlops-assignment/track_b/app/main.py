"""
FastAPI backend for the AI Assistant.

W15 endpoints: /health, /ingest, /chat, /chat/batch
W16 endpoint:  /agent/verify  — the Task 3 agentic loop (VerificationAgent)

/chat (W15) request flow:
  1. Rate limit + cache check
  2. (optional) RAG retrieval -> context injected into system prompt
  3. LLM call (via FallbackOrchestrator) with tool schemas attached
  4. If the model asks for tool calls -> execute them -> call LLM again
     (loop, bounded) until it returns a final answer
  5. If `structured=True` -> validate against QueryAnalysis, one retry
     on validation failure
  6. Cache + return

/agent/verify (W16) request flow: see app/agents/verification_agent.py.
Sessions that stop with `ask_clarification` are kept in an in-memory
store so a follow-up call with the same session_id can resume the loop
(simple multi-turn support).
"""
import logging
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from app.agents.verification_agent import VerificationAgent
from app.config import get_settings
from app.llm.providers import build_providers
from app.llm.tools import TOOL_SCHEMAS, execute_tool
from app.rag.embeddings import get_embedding_model
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.reliability.cache import ResponseCache
from app.reliability.fallback import AllProvidersFailedError, FallbackOrchestrator
from app.reliability.rate_limiter import ConcurrencyLimiter, RateLimitExceeded, RateLimiter
from app.schemas import (
    AgentRequest,
    AgentResponse,
    ChatRequest,
    ChatResponse,
    Citation,
    HealthResponse,
    IngestResponse,
    QUERY_ANALYSIS_JSON_SCHEMA,
    QueryAnalysis,
    StopReason,
    ToolCallRecord,
)
from app.utils.logging_config import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger("ai_assistant.main")

SYSTEM_PROMPT = """You are a helpful, precise AI assistant built for a fellowship applied-AI assignment.
- Use the provided CONTEXT (retrieved documents) when it is relevant; do not invent facts not supported by it or your own knowledge.
- Use tools when they would produce a more accurate/current answer than reasoning alone.
- Be concise and cite which source a fact came from when context was used.
- If you don't know, say so plainly rather than guessing."""

MAX_TOOL_ITERATIONS = 4


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Assistant service...")
    app.state.providers = build_providers(settings)
    app.state.orchestrator = FallbackOrchestrator(
        providers=app.state.providers,
        order=settings.fallback_order,
        retry_max_attempts=settings.retry_max_attempts,
        retry_backoff_seconds=settings.retry_backoff_seconds,
    )
    embedding_model = get_embedding_model(settings.embedding_model)
    vector_store = VectorStore(persist_dir=settings.chroma_persist_dir)
    app.state.retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    app.state.cache = ResponseCache(maxsize=settings.cache_max_size, ttl_seconds=settings.cache_ttl_seconds)
    app.state.rate_limiter = RateLimiter(requests_per_minute=settings.rate_limit_per_minute)
    app.state.concurrency_limiter = ConcurrencyLimiter(max_concurrent=settings.max_concurrent_requests)
    # In-memory store for agent sessions awaiting user clarification (see /agent/verify).
    # Like the cache/rate-limiter, this is process-local; a multi-instance deployment
    # would back this with Redis instead.
    app.state.agent_sessions = {}
    logger.info("Providers available: %s", list(app.state.providers.keys()))
    yield
    logger.info("Shutting down AI Assistant service...")


app = FastAPI(title="AI Assistant", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def _client_key(request: Request) -> str:
    return request.headers.get("x-api-key") or (request.client.host if request.client else "anonymous")


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    results = {}
    for name, provider in request.app.state.providers.items():
        results[name] = await provider.health()
    status = "ok" if any(results.values()) else "degraded"
    return HealthResponse(status=status, providers=results)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request, file: UploadFile = File(...)):
    tmp_dir = Path("/tmp/uploads")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{uuid.uuid4().hex}_{file.filename}"
    with tmp_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks_created = request.app.state.retriever.ingest_path(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(documents_ingested=1, chunks_created=chunks_created)


# ---------------------------------------------------------------------------
# W15: single-pass chat with RAG + tool-calling loop
# ---------------------------------------------------------------------------
async def _run_tool_loop(app_state, messages: List[dict], temperature: float, top_p: float):
    tool_records: List[ToolCallRecord] = []
    provider_used = "unknown"
    total_tokens = 0

    for _ in range(MAX_TOOL_ITERATIONS):
        async with app_state.concurrency_limiter:
            try:
                result = await app_state.orchestrator.chat(
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=settings.max_tokens,
                )
            except AllProvidersFailedError as e:
                degraded = FallbackOrchestrator.graceful_degradation_response(e.errors)
                return degraded["content"], tool_records, degraded["provider_used"], total_tokens

        provider_used = result["provider_used"]
        total_tokens += result.get("usage", {}).get("total_tokens", 0)

        if not result["tool_calls"]:
            return result["content"], tool_records, provider_used, total_tokens

        messages.append({"role": "assistant", "content": result["content"] or ""})
        for tc in result["tool_calls"]:
            tool_output = execute_tool(tc["name"], tc["arguments"], retriever=app_state.retriever)
            tool_records.append(ToolCallRecord(name=tc["name"], arguments=tc["arguments"], result=tool_output))
            messages.append({"role": "user", "content": f"[Tool result for {tc['name']}]: {tool_output}"})

    return "I reached the maximum number of tool-call iterations without a final answer.", tool_records, provider_used, total_tokens


@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest, request: Request):
    start = time.perf_counter()
    app_state = request.app.state

    try:
        await app_state.rate_limiter.check(_client_key(request))
    except RateLimitExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))

    temperature = chat_request.temperature if chat_request.temperature is not None else settings.temperature
    top_p = chat_request.top_p if chat_request.top_p is not None else settings.top_p

    cache_key = app_state.cache.make_key(
        message=chat_request.message, use_rag=chat_request.use_rag,
        structured=chat_request.structured, temperature=temperature, top_p=top_p,
    )
    cached = app_state.cache.get(cache_key)
    if cached is not None:
        cached = dict(cached)
        cached["cached"] = True
        cached["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return ChatResponse(**cached)

    citations: List[Citation] = []
    context_block = ""
    if chat_request.use_rag:
        results = app_state.retriever.query(chat_request.message, top_k=settings.retrieval_top_k)
        context_block = app_state.retriever.build_context_block(results)
        citations = [Citation(source=r["metadata"].get("source", "unknown"), chunk_id=r["chunk_id"], score=r["score"]) for r in results]

    system_content = SYSTEM_PROMPT
    if context_block:
        system_content += f"\n\nCONTEXT:\n{context_block}"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": chat_request.message},
    ]

    structured_output = None
    total_tokens = 0
    if chat_request.structured:
        async with app_state.concurrency_limiter:
            try:
                result = await app_state.orchestrator.chat(
                    messages=messages, tools=None, temperature=temperature, top_p=top_p,
                    max_tokens=settings.max_tokens, json_schema=QUERY_ANALYSIS_JSON_SCHEMA,
                )
            except AllProvidersFailedError as e:
                result = FallbackOrchestrator.graceful_degradation_response(e.errors)

        provider_used = result["provider_used"]
        total_tokens += result.get("usage", {}).get("total_tokens", 0)
        raw_content = result["content"] or "{}"
        try:
            structured_output = QueryAnalysis.model_validate_json(raw_content).model_dump()
            answer = structured_output["summary"]
        except (ValidationError, ValueError) as e:
            logger.warning("Structured output validation failed, retrying once: %s", e)
            messages.append({"role": "assistant", "content": raw_content})
            messages.append({"role": "user", "content": f"That was not valid JSON matching the schema ({e}). Return ONLY corrected JSON."})
            async with app_state.concurrency_limiter:
                retry_result = await app_state.orchestrator.chat(
                    messages=messages, tools=None, temperature=0.0, top_p=top_p,
                    max_tokens=settings.max_tokens, json_schema=QUERY_ANALYSIS_JSON_SCHEMA,
                )
            total_tokens += retry_result.get("usage", {}).get("total_tokens", 0)
            try:
                structured_output = QueryAnalysis.model_validate_json(retry_result["content"] or "{}").model_dump()
                answer = structured_output["summary"]
            except (ValidationError, ValueError) as e2:
                answer = f"Failed to produce valid structured output after retry: {e2}"
        tool_records: List[ToolCallRecord] = []
    else:
        answer, tool_records, provider_used, tt = await _run_tool_loop(app_state, messages, temperature, top_p)
        total_tokens += tt

    response_payload = {
        "answer": answer or "",
        "structured_output": structured_output,
        "citations": [c.model_dump() for c in citations],
        "tool_calls": [t.model_dump() for t in tool_records],
        "provider_used": provider_used,
        "cached": False,
        "total_tokens": total_tokens,
    }
    app_state.cache.set(cache_key, response_payload)

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return ChatResponse(**response_payload, latency_ms=latency_ms)


@app.post("/chat/batch", response_model=List[ChatResponse])
async def chat_batch(chat_requests: List[ChatRequest], request: Request):
    import asyncio

    async def _one(cr: ChatRequest):
        return await chat(cr, request)

    return await asyncio.gather(*(_one(cr) for cr in chat_requests))


# ---------------------------------------------------------------------------
# W16 (Task 3): agentic verification loop
# ---------------------------------------------------------------------------
@app.post("/agent/verify", response_model=AgentResponse)
async def agent_verify(agent_request: AgentRequest, request: Request):
    app_state = request.app.state

    try:
        await app_state.rate_limiter.check(_client_key(request))
    except RateLimitExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))

    session = app_state.agent_sessions.get(agent_request.session_id)
    query = agent_request.query
    if session and session.get("awaiting_clarification") and agent_request.clarification_answer:
        query = session["original_query"]

    agent = VerificationAgent(
        orchestrator=app_state.orchestrator,
        retriever=app_state.retriever,
        max_iterations=settings.agent_max_iterations,
        min_sources=settings.agent_min_sources,
        confidence_threshold=settings.agent_confidence_threshold,
        max_notes=settings.agent_max_notes,
        fail_inject_tool=settings.agent_fail_inject_tool or None,
    )

    async with app_state.concurrency_limiter:
        result = await agent.run(query=query, clarification_answer=agent_request.clarification_answer)

    if result["stop_reason"] == StopReason.ask_clarification:
        app_state.agent_sessions[agent_request.session_id] = {
            "awaiting_clarification": True,
            "original_query": query,
        }
    else:
        app_state.agent_sessions.pop(agent_request.session_id, None)

    return AgentResponse(
        session_id=agent_request.session_id,
        final_answer=result["final_answer"],
        clarification_question=result["clarification_question"],
        stop_reason=result["stop_reason"],
        confidence=result["confidence"],
        sources_used=result["sources_used"],
        notes=result["notes"],
        trace=result["trace"],
        iterations_used=result["iterations_used"],
        total_tokens=result["total_tokens"],
        latency_ms=result["latency_ms"],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=False)
