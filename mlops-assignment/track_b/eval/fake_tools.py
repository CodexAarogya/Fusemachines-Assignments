"""
Deterministic, offline stand-ins for the agent's tools and retriever, used
only by the evaluation harness. The real tools (app/agents/tools.py) hit
live network services (Wikipedia, wttr.in) which would make the harness
slow, flaky, and non-reproducible in a grading environment without
network access — so the harness uses these instead, while exercising the
exact same VerificationAgent loop, stopping conditions, note-extraction,
and self-check logic as production.
"""
from typing import Any, Dict, List, Optional


class FakeRetriever:
    """Simulates the internal RAG knowledge base with a small fixed corpus."""

    _CORPUS = [
        {"chunk_id": "sample-6-aaa111", "text": "vLLM improves throughput using PagedAttention, which manages the KV-cache efficiently and enables much higher concurrent request throughput than naive serving.", "score": 0.91, "metadata": {"source": "sample.txt"}},
        {"chunk_id": "sample-1-bbb222", "text": "RAG combines a retrieval system (a vector database) with a generative model, reducing hallucination by grounding answers in retrieved documents.", "score": 0.88, "metadata": {"source": "sample.txt"}},
        {"chunk_id": "sample-7-ccc333", "text": "Naively appending every raw tool result to conversation history causes context saturation; structured external notes keep the prompt bounded.", "score": 0.86, "metadata": {"source": "sample.txt"}},
    ]

    def query(self, text: str, top_k: int = 4) -> List[Dict[str, Any]]:
        # Simple keyword overlap scoring so different queries surface different chunks,
        # without needing a real embedding model in the harness.
        words = set(text.lower().split())
        scored = []
        for doc in self._CORPUS:
            overlap = len(words & set(doc["text"].lower().split()))
            scored.append((overlap, doc))
        scored.sort(key=lambda x: -x[0])
        return [doc for _, doc in scored[:top_k]] if any(s for s, _ in scored) else self._CORPUS[:top_k]


_WIKI_FIXTURES = {
    "vllm": {"title": "vLLM", "extract": "vLLM is a fast inference and serving engine for large language models, known for PagedAttention which improves memory efficiency and throughput.", "url": "https://en.wikipedia.org/wiki/VLLM"},
    "retrieval-augmented generation": {"title": "Retrieval-augmented generation", "extract": "Retrieval-augmented generation (RAG) is a technique that combines information retrieval with text generation to improve factual accuracy.", "url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"},
    "mistral ai": {"title": "Mistral AI", "extract": "Mistral AI is a French company that develops open-weight large language models.", "url": "https://en.wikipedia.org/wiki/Mistral_AI"},
    "llama (language model)": {"title": "Llama (language model)", "extract": "Llama is a family of open-weight large language models released by Meta.", "url": "https://en.wikipedia.org/wiki/Llama_(language_model)"},
    "context engineering": {"title": "Context engineering", "extract": "Context engineering refers to techniques for managing what information is kept in a language model's context window, such as summarization, note-taking, and compaction.", "url": "https://en.wikipedia.org/wiki/Context_engineering"},
}


def fake_wikipedia_search(query: str) -> Dict[str, Any]:
    key = query.strip().lower()
    if key in _WIKI_FIXTURES:
        return dict(_WIKI_FIXTURES[key])
    return {"query": query, "error": "no fixture for this query (simulated 'no article found')"}


def fake_calculator(expression: str) -> Dict[str, Any]:
    try:
        # Reuse the real safe evaluator so arithmetic correctness is still real.
        from app.llm.tools import calculator as real_calculator
        return real_calculator(expression)
    except Exception as e:  # noqa: BLE001
        return {"expression": expression, "error": str(e)}


def fake_get_weather(city: str) -> Dict[str, Any]:
    fixtures = {"paris": {"temperature_c": "18", "description": "Partly cloudy"}, "tokyo": {"temperature_c": "24", "description": "Clear"}}
    data = fixtures.get(city.strip().lower())
    if not data:
        return {"city": city, "error": "no fixture for this city (simulated lookup miss)"}
    return {"city": city, **data}


def fake_execute_agent_tool(
    name: str,
    arguments: Dict[str, Any],
    retriever=None,
    fail_inject_tool: Optional[str] = None,
) -> Dict[str, Any]:
    """Signature-compatible with app.agents.tools.execute_agent_tool."""
    inject = bool(fail_inject_tool) and fail_inject_tool == name
    try:
        if name == "rag_search":
            if inject:
                return {"error": "retriever returned malformed response (injected failure)"}
            if retriever is None:
                return {"error": "retriever not available"}
            if "query" not in arguments:
                return {"error": "missing required argument 'query'"}
            return {"results": retriever.query(arguments.get("query", ""), top_k=arguments.get("top_k", 4))}

        if name == "wikipedia_search":
            if inject:
                return {"error": "wikipedia_search is unavailable (injected failure)"}
            if "query" not in arguments:
                return {"error": "missing required argument 'query'"}
            return fake_wikipedia_search(arguments.get("query", ""))

        if name == "calculator":
            if inject:
                return {"error": "calculator tool unavailable (injected failure)"}
            if "expression" not in arguments:
                return {"error": "missing required argument 'expression'"}
            return fake_calculator(arguments.get("expression", ""))

        if name == "get_weather":
            if inject:
                return {"error": "weather tool unavailable (injected failure)"}
            if "city" not in arguments:
                return {"error": "missing required argument 'city'"}
            return fake_get_weather(arguments.get("city", ""))

        return {"error": f"unknown tool '{name}'"}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
