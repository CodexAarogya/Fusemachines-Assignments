"""
Tests for the W15 pieces. Run with: pytest tests/ -v
These avoid live API keys / GPUs; see tests/test_agent.py for Task 3 tests.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.llm.tools import calculator, execute_tool
from app.rag.ingest import recursive_chunk
from app.reliability.cache import ResponseCache
from app.reliability.rate_limiter import RateLimiter, RateLimitExceeded


def test_chunking_respects_size_limit():
    text = "Paragraph one.\n\n" + ("word " * 500) + "\n\nParagraph two."
    chunks = recursive_chunk(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 200 + 20 + 5


def test_chunking_empty_text():
    assert recursive_chunk("", chunk_size=200) == []


def test_calculator_basic():
    result = calculator("2 + 2 * 3")
    assert result["result"] == 8


def test_calculator_rejects_unsafe_input():
    result = calculator("__import__('os').system('echo hi')")
    assert "error" in result


def test_execute_tool_dispatch():
    result = execute_tool("calculator", {"expression": "10 / 2"})
    assert result["result"] == 5.0


def test_execute_tool_unknown():
    result = execute_tool("does_not_exist", {})
    assert "error" in result


def test_response_cache_roundtrip():
    cache = ResponseCache(maxsize=10, ttl_seconds=60)
    key = cache.make_key(message="hello")
    assert cache.get(key) is None
    cache.set(key, {"answer": "hi"})
    assert cache.get(key) == {"answer": "hi"}


@pytest.mark.asyncio
async def test_rate_limiter_blocks_after_limit():
    limiter = RateLimiter(requests_per_minute=2)
    await limiter.check("client-a")
    await limiter.check("client-a")
    with pytest.raises(RateLimitExceeded):
        await limiter.check("client-a")


@pytest.mark.asyncio
async def test_rate_limiter_separates_clients():
    limiter = RateLimiter(requests_per_minute=1)
    await limiter.check("client-a")
    await limiter.check("client-b")


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="requires live API key + running server")
def test_chat_endpoint_live():
    import requests

    resp = requests.post("http://localhost:8000/chat", json={"message": "What is RAG?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
