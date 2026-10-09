"""
MockOrchestrator — a deterministic, scripted stand-in for
app.reliability.fallback.FallbackOrchestrator, used ONLY by the evaluation
harness so it can run fully offline, free, and reproducibly (no API keys,
no network, same result every run).

It exposes the exact same `async def chat(...)` signature the real
orchestrator does, and VerificationAgent only ever calls `.chat(...)` on
whatever orchestrator it's given — so this is a drop-in replacement from
the agent's point of view. To evaluate against a REAL model instead, pass
`--live` to eval/run_eval.py, which builds a real FallbackOrchestrator
from your .env instead of this class (see run_eval.py).

Token accounting here is REAL, not faked: we tokenize the actual prompt
text sent to `.chat()` and the actual JSON the script would have the
model emit, using tiktoken (cl100k_base), so the harness's token numbers
reflect a realistic cost even though the "model" is scripted.
"""
import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger("ai_assistant.eval")

# tiktoken's encoder downloads its BPE merge table from a remote blob on
# first use. In an offline/sandboxed environment (no network egress) that
# download fails, so we fall back to a simple, documented approximation
# (~4 characters per token, the commonly-cited rule of thumb for English
# text with OpenAI's tokenizers) rather than letting the whole harness
# crash. Token counts are then clearly labeled as approximate in the report.
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
    TOKEN_COUNTING_METHOD = "tiktoken(cl100k_base)"

    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return len(_ENC.encode(text))

except Exception as e:  # noqa: BLE001
    logger.warning("tiktoken unavailable (%s); falling back to char-based token estimate.", e)
    TOKEN_COUNTING_METHOD = "approximate(len/4)"

    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)


class MockOrchestrator:
    def __init__(self, script: List[Dict[str, Any]], provider_name: str = "mock"):
        self.script = script
        self.provider_name = provider_name
        self._idx = 0
        self.calls = 0

    async def chat(self, messages, tools=None, temperature=0.2, top_p=0.9, max_tokens=600, json_schema=None):
        self.calls += 1
        prompt_text = "\n".join(m.get("content", "") for m in messages)
        input_tokens = count_tokens(prompt_text)

        if self._idx >= len(self.script):
            # Script exhausted without the test case reaching a terminal
            # action — this simulates a model that never converges, so the
            # agent's own max-iteration stopping condition must catch it.
            step = {
                "thought": "No more scripted reasoning available; stopping defensively.",
                "action": "finish",
                "action_input": {},
                "draft_answer": None,
                "confidence": 0.05,
            }
        else:
            step = self.script[self._idx]
            self._idx += 1

        if "__raw__" in step:
            # Simulate a model emitting malformed/non-schema-conforming output.
            content = step["__raw__"]
        else:
            content = json.dumps(step)
        output_tokens = count_tokens(content)
        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        return {"content": content, "tool_calls": [], "usage": usage, "provider_used": self.provider_name}
