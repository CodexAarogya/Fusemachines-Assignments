"""
Provider abstraction layer.

Every provider (OpenAI cloud, Anthropic cloud, local vLLM) is wrapped so
callers get back the SAME normalised shape:

    {
        "content": str | None,
        "tool_calls": [{"name": str, "arguments": dict}],
        "usage": {"input_tokens": int, "output_tokens": int, "total_tokens": int},
        "raw": <sdk response>,
    }

`usage` is populated from each SDK's real token-accounting fields where
available, which is what the Task 3 evaluation harness sums for cost
accounting. vLLM's OpenAI-compatible server (`vllm serve ...`) is queried
with the *same* `openai.AsyncOpenAI` client, just pointed at a different
`base_url` — "local model via vLLM" is just another provider in the
fallback chain, not a special code path.
"""
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import anthropic
from openai import AsyncOpenAI

from app.config import Settings

logger = logging.getLogger("ai_assistant.providers")

_ZERO_USAGE = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


class LLMProvider(ABC):
    name: str

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ...

    @abstractmethod
    async def health(self) -> bool:
        ...


class OpenAICompatibleProvider(LLMProvider):
    """Works for OpenAI cloud AND any OpenAI-compatible server (vLLM)."""

    def __init__(self, name: str, api_key: str, model: str, base_url: Optional[str] = None):
        self.name = name
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key or "not-needed", base_url=base_url)

    async def chat(
        self,
        messages,
        tools=None,
        temperature=0.3,
        top_p=0.9,
        max_tokens=1024,
        json_schema=None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("title", "response"),
                    "schema": json_schema,
                    "strict": True,
                },
            }

        resp = await self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})

        usage = dict(_ZERO_USAGE)
        if getattr(resp, "usage", None):
            usage = {
                "input_tokens": resp.usage.prompt_tokens or 0,
                "output_tokens": resp.usage.completion_tokens or 0,
                "total_tokens": resp.usage.total_tokens or 0,
            }

        return {"content": msg.content, "tool_calls": tool_calls, "usage": usage, "raw": resp}

    async def health(self) -> bool:
        try:
            await self.client.models.list()
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("Health check failed for %s: %s", self.name, e)
            return False


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        self.name = "anthropic"
        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    def _convert_tools(self, tools):
        if not tools:
            return None
        converted = []
        for t in tools:
            fn = t["function"]
            converted.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return converted

    async def chat(
        self,
        messages,
        tools=None,
        temperature=0.3,
        top_p=0.9,
        max_tokens=1024,
        json_schema=None,
    ) -> Dict[str, Any]:
        system = ""
        converted_messages = []
        for m in messages:
            if m["role"] == "system":
                system += m["content"] + "\n"
            else:
                converted_messages.append({"role": m["role"], "content": m["content"]})

        if json_schema:
            system += (
                "\nYou MUST respond with ONLY valid JSON matching this schema, "
                f"no prose, no markdown fences:\n{json.dumps(json_schema)}"
            )

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            system=system.strip(),
            messages=converted_messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        anthropic_tools = self._convert_tools(tools)
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        resp = await self.client.messages.create(**kwargs)

        content_text = ""
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "arguments": block.input})

        usage = dict(_ZERO_USAGE)
        if getattr(resp, "usage", None):
            inp = resp.usage.input_tokens or 0
            out = resp.usage.output_tokens or 0
            usage = {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}

        return {"content": content_text or None, "tool_calls": tool_calls, "usage": usage, "raw": resp}

    async def health(self) -> bool:
        try:
            await self.client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("Health check failed for anthropic: %s", e)
            return False


def build_providers(settings: Settings) -> Dict[str, LLMProvider]:
    """Instantiate every provider that has credentials configured."""
    providers: Dict[str, LLMProvider] = {}

    if settings.openai_api_key:
        providers["openai"] = OpenAICompatibleProvider(
            name="openai", api_key=settings.openai_api_key, model=settings.openai_model
        )

    providers["local"] = OpenAICompatibleProvider(
        name="local",
        api_key=settings.local_llm_api_key,
        model=settings.local_llm_model,
        base_url=settings.local_llm_base_url,
    )

    if settings.anthropic_api_key:
        providers["anthropic"] = AnthropicProvider(
            api_key=settings.anthropic_api_key, model=settings.anthropic_model
        )

    return providers
