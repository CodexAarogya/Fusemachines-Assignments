"""
Tools available to the Task 3 VerificationAgent.

Reuses the W15 calculator/get_weather tools and adds `wikipedia_search`,
a second, independent evidence source distinct from the internal RAG
knowledge base. Having two independent sources (internal docs vs.
external encyclopedia) is what makes "cross-source verification"
possible — a single-source pipeline has nothing to cross-check against.

`AGENT_FAIL_INJECT_TOOL` is a test-only hook (set via settings) used by
the failure-injection test required by the assignment: when it matches a
tool name, that tool raises instead of returning a result, so we can
observe whether the agent notices and adapts rather than confidently
answering off bad/missing evidence.
"""
import logging
from typing import Any, Dict, Optional

import httpx

from app.llm.tools import calculator, get_weather

logger = logging.getLogger("ai_assistant.agent_tools")


class ToolUnavailableError(Exception):
    pass


def wikipedia_search(query: str, fail_inject: bool = False) -> Dict[str, Any]:
    """Looks up a short extract from Wikipedia's public REST summary API —
    an external, independent source used to cross-check the internal KB."""
    if fail_inject:
        raise ToolUnavailableError("wikipedia_search is unavailable (injected failure)")
    try:
        resp = httpx.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}",
            timeout=5.0,
            headers={"User-Agent": "ai-assistant-fellowship-assignment/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "title": data.get("title"),
            "extract": data.get("extract", "")[:600],
            "url": data.get("content_urls", {}).get("desktop", {}).get("page"),
        }
    except Exception as e:  # noqa: BLE001
        return {"query": query, "error": str(e)}


# Single source of truth for "did the agent supply valid arguments?" — used
# both here (implicitly, via the handlers below) and by the evaluation
# harness's tool-call-correctness check (eval/run_eval.py).
AGENT_TOOL_REQUIRED_ARGS = {
    "rag_search": ["query"],
    "wikipedia_search": ["query"],
    "calculator": ["expression"],
    "get_weather": ["city"],
    "draft_answer": [],
    "ask_clarification": [],
    "finish": [],
}

AGENT_TOOL_DESCRIPTIONS = """\
- rag_search(query, top_k=4): search the internal ingested knowledge base.
- wikipedia_search(query): look up an independent external encyclopedia summary.
- calculator(expression): evaluate arithmetic.
- get_weather(city): current weather for a city.
- draft_answer(): propose a draft answer in `draft_answer` based on notes gathered so far (does not stop the loop).
- ask_clarification(): stop and ask the user a clarifying question (put the question in `draft_answer`).
- finish(): stop and return the final answer in `draft_answer`. Only allowed once evidence from at least
  two independent sources has been gathered and your confidence reflects how well they agree.
"""


def execute_agent_tool(
    name: str,
    arguments: Dict[str, Any],
    retriever=None,
    fail_inject_tool: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes a tool by name and ALWAYS returns a dict — errors are
    captured as {"error": ...} rather than raised, so the agent loop can
    treat a failed tool as an observation to reason about (see
    VerificationAgent._extract_note), not as a crash."""
    inject = bool(fail_inject_tool) and fail_inject_tool == name
    try:
        if name == "rag_search":
            if retriever is None:
                return {"error": "retriever not available"}
            if inject:
                # Simulate malformed retrieval output (one of the suggested failure injections).
                return {"error": "retriever returned malformed response (injected failure)"}
            query = arguments.get("query", "")
            top_k = arguments.get("top_k", 4)
            results = retriever.query(query, top_k=top_k)
            return {"results": results}

        if name == "wikipedia_search":
            return wikipedia_search(arguments.get("query", ""), fail_inject=inject)

        if name == "calculator":
            if inject:
                return {"error": "calculator tool unavailable (injected failure)"}
            return calculator(arguments.get("expression", ""))

        if name == "get_weather":
            if inject:
                return {"error": "weather tool unavailable (injected failure)"}
            return get_weather(arguments.get("city", ""))

        return {"error": f"unknown tool '{name}'"}
    except Exception as e:  # noqa: BLE001
        logger.warning("Tool '%s' raised: %s", name, e)
        return {"error": str(e)}
