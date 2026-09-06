"""
Tool / function-calling definitions.

Each tool has:
  - an OpenAI-style JSON schema describing it to the model
  - a Python callable that actually executes it

The `execute_tool` dispatcher is provider-agnostic: both the OpenAI and
Anthropic clients are normalised (see providers.py) to emit
{"name": ..., "arguments": {...}} tool calls that are run through here.
"""
import ast
import operator
from datetime import datetime, timezone
from typing import Any, Dict

import httpx

# ---------------------------------------------------------------------------
# 1. calculator — safe arithmetic evaluation (no eval())
# ---------------------------------------------------------------------------
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(expression, mode="eval").body
        return {"expression": expression, "result": _safe_eval(tree)}
    except Exception as e:  # noqa: BLE001
        return {"expression": expression, "error": str(e)}


# ---------------------------------------------------------------------------
# 2. get_current_time — simple deterministic tool, useful for demos/tests
# ---------------------------------------------------------------------------
def get_current_time() -> Dict[str, Any]:
    return {"utc_time": datetime.now(timezone.utc).isoformat()}


# ---------------------------------------------------------------------------
# 3. web_lookup — lightweight external tool call (weather via wttr.in)
# ---------------------------------------------------------------------------
def get_weather(city: str) -> Dict[str, Any]:
    try:
        resp = httpx.get(f"https://wttr.in/{city}?format=j1", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        current = data["current_condition"][0]
        return {
            "city": city,
            "temperature_c": current["temp_C"],
            "description": current["weatherDesc"][0]["value"],
        }
    except Exception as e:  # noqa: BLE001
        return {"city": city, "error": str(e)}


# ---------------------------------------------------------------------------
# 4. rag_search — populated at call time with the retriever (see main.py)
#    Registered separately because it needs the vector store instance.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4)'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current UTC date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city name.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search the internal knowledge base (ingested documents) for relevant context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 4},
                },
                "required": ["query"],
            },
        },
    },
]

_LOCAL_TOOLS = {
    "calculator": lambda args: calculator(**args),
    "get_current_time": lambda args: get_current_time(),
    "get_weather": lambda args: get_weather(**args),
}


def execute_tool(name: str, arguments: Dict[str, Any], retriever=None) -> Any:
    """Dispatch a tool call by name. `retriever` is injected for rag_search."""
    if name == "rag_search":
        if retriever is None:
            return {"error": "retriever not available"}
        query = arguments.get("query", "")
        top_k = arguments.get("top_k", 4)
        results = retriever.query(query, top_k=top_k)
        return {"results": results}

    handler = _LOCAL_TOOLS.get(name)
    if handler is None:
        return {"error": f"unknown tool '{name}'"}
    return handler(arguments)
