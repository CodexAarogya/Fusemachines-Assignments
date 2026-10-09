"""
VerificationAgent — Task 3's agentic feature.

Feature: "cross-source verification + self-check before responding".
The agent gathers evidence from multiple independent sources (internal
RAG knowledge base + external Wikipedia summary, plus calculator/weather
when relevant), drafts an answer, and only finishes once it has checked
that answer against enough independent, agreeing evidence. If evidence
is thin, conflicting, or a tool fails, it must gather more, revise, or
ask the user — it cannot simply emit one answer after one retrieval.

Why a fixed pipeline is not sufficient (required one-sentence justification):
A fixed single-pass RAG pipeline always performs exactly one retrieval and
then answers, so it cannot recognize that the retrieved evidence is thin,
missing, or contradicted by an external source, and it cannot decide —
based on what it actually finds — to consult a second source, revise a
draft, or ask the user before responding.

Agentic pattern: SINGLE-AGENT loop (not multi-agent). See README.md
section "Agentic Pattern" for the justification using the five
structural-failures framework.

Context-engineering technique used: STRUCTURED EXTERNAL NOTES.
Unlike the W15 tool-calling loop in app/main.py (which appends the full
raw tool-call result into the message list every turn, so the prompt
grows without bound across iterations), this agent extracts a short
(1-2 sentence) EvidenceNote from each tool result and discards the raw
payload. Only the compact notes (capped at `agent_max_notes`, keeping the
most recent) are serialized into the next iteration's prompt. This keeps
prompt size roughly constant regardless of how many iterations run, which
matters here because cross-source verification can take several rounds of
tool calls whose raw outputs (full Wikipedia extracts, full RAG chunk
text) would otherwise dominate and eventually crowd out the actual
question and reasoning instructions (context saturation).
"""
import json
import logging
import time
from typing import Any, Dict, List, Optional

from app.agents.tools import execute_agent_tool
from app.schemas import (
    AGENT_STEP_JSON_SCHEMA,
    AgentStep,
    AgentTrace,
    EvidenceNote,
    StopReason,
)

logger = logging.getLogger("ai_assistant.agent")

AGENT_SYSTEM_PROMPT = """You are a careful research assistant that verifies answers across \
multiple independent sources before responding.

Available actions, each iteration pick exactly ONE:
- rag_search: search the internal knowledge base (ingested documents)
- wikipedia_search: look up an independent external encyclopedia summary
- calculator: evaluate arithmetic
- get_weather: current weather for a city
- draft_answer: propose a draft answer (does not stop the loop, use it to record progress)
- ask_clarification: stop and ask the user a clarifying question (put it in draft_answer)
- finish: stop and return the final answer (put it in draft_answer)

Rules you MUST follow:
1. You may only choose "finish" after you have gathered evidence from at least two
   DIFFERENT independent sources (e.g. rag_search AND wikipedia_search), and your
   confidence should reflect how well those sources agree. If they disagree, say so
   in the answer or gather more evidence instead of finishing.
2. If a tool returns an error, do not ignore it or answer as if it had succeeded —
   either try a different source/tool, lower your confidence, or ask the user.
3. Do not repeat the exact same action with the exact same input twice in a row.
4. Respond with ONLY the structured JSON decision for this step. Nothing else.

You will be shown the ORIGINAL QUESTION, a compact list of EVIDENCE NOTES gathered
so far (not full raw tool output — that has been summarized), and the ACTIONS
ALREADY TAKEN this run. Decide the next single action based on this state."""


class LoopDetectedError(Exception):
    pass


class VerificationAgent:
    def __init__(
        self,
        orchestrator,
        retriever,
        max_iterations: int = 6,
        min_sources: int = 2,
        confidence_threshold: float = 0.6,
        max_notes: int = 8,
        fail_inject_tool: Optional[str] = None,
        tool_executor=None,
        system_prompt: Optional[str] = None,
    ):
        self.orchestrator = orchestrator
        self.retriever = retriever
        self.max_iterations = max_iterations
        self.min_sources = min_sources
        self.confidence_threshold = confidence_threshold
        self.max_notes = max_notes
        self.fail_inject_tool = fail_inject_tool or None
        # Injectable so tests/the eval harness can run fully offline with a
        # deterministic fake tool layer instead of hitting live network tools
        # (e.g. the real wikipedia_search). Defaults to the real tools.
        self.tool_executor = tool_executor or execute_agent_tool
        self.system_prompt = system_prompt or AGENT_SYSTEM_PROMPT  # versioned prompts live in prompts/

    # ------------------------------------------------------------------
    # Prompt construction (context engineering happens here: we serialize
    # compact notes, never raw tool payloads)
    # ------------------------------------------------------------------
    def _build_prompt(self, query: str, notes: List[EvidenceNote], actions_taken: List[str], clarification_answer: Optional[str]) -> List[Dict[str, str]]:
        notes_block = "\n".join(f"- [{n.source}] {n.content}" for n in notes) or "(none yet)"
        actions_block = ", ".join(actions_taken) or "(none yet)"

        user_block = f"ORIGINAL QUESTION:\n{query}\n\nEVIDENCE NOTES SO FAR:\n{notes_block}\n\nACTIONS ALREADY TAKEN: {actions_block}"
        if clarification_answer:
            user_block += f"\n\nUSER'S ANSWER TO YOUR CLARIFYING QUESTION:\n{clarification_answer}"

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_block},
        ]

    # ------------------------------------------------------------------
    # Context engineering: raw tool output -> compact structured note
    # ------------------------------------------------------------------
    def _extract_note(self, iteration: int, action: str, raw_result: Dict[str, Any]) -> EvidenceNote:
        if "error" in raw_result:
            return EvidenceNote(iteration=iteration, source=action, content=f"FAILED: {raw_result['error']}", supports_answer=None)

        if action == "rag_search":
            results = raw_result.get("results", [])
            if not results:
                return EvidenceNote(iteration=iteration, source="internal_kb", content="No matching internal documents found.", supports_answer=None)
            top = results[0]
            snippet = top["text"][:220].replace("\n", " ")
            return EvidenceNote(iteration=iteration, source="internal_kb", content=f"(score={top['score']:.2f}) {snippet}", supports_answer=None)

        if action == "wikipedia_search":
            extract = raw_result.get("extract", "")
            title = raw_result.get("title", "unknown")
            if not extract:
                return EvidenceNote(iteration=iteration, source="wikipedia", content=f"No summary found for '{title}'.", supports_answer=None)
            return EvidenceNote(iteration=iteration, source="wikipedia", content=f"{title}: {extract[:220]}", supports_answer=None)

        if action == "calculator":
            return EvidenceNote(iteration=iteration, source="calculator", content=f"{raw_result.get('expression')} = {raw_result.get('result')}", supports_answer=None)

        if action == "get_weather":
            return EvidenceNote(iteration=iteration, source="weather", content=f"{raw_result.get('city')}: {raw_result.get('temperature_c')}C, {raw_result.get('description')}", supports_answer=None)

        return EvidenceNote(iteration=iteration, source=action, content=json.dumps(raw_result)[:200], supports_answer=None)

    def _cap_notes(self, notes: List[EvidenceNote]) -> List[EvidenceNote]:
        """Caps the notes list so the prompt stays bounded — part of the
        structured-external-notes technique (see module docstring)."""
        if len(notes) <= self.max_notes:
            return notes
        return notes[-self.max_notes :]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    async def run(self, query: str, clarification_answer: Optional[str] = None) -> Dict[str, Any]:
        start = time.perf_counter()
        notes: List[EvidenceNote] = []
        actions_taken: List[str] = []
        trace: List[AgentTrace] = []
        sources_used = set()
        total_tokens = 0
        last_draft: Optional[str] = None
        last_confidence = 0.0
        last_action_signature: Optional[str] = None

        for iteration in range(1, self.max_iterations + 1):
            messages = self._build_prompt(query, notes, actions_taken, clarification_answer if iteration == 1 else None)

            try:
                result = await self.orchestrator.chat(
                    messages=messages,
                    tools=None,
                    temperature=0.2,
                    top_p=0.9,
                    max_tokens=600,
                    json_schema=AGENT_STEP_JSON_SCHEMA,
                )
            except Exception as e:  # noqa: BLE001
                # Hard failure: the LLM layer itself is down (all providers failed).
                logger.error("Agent iteration %d: LLM call failed entirely: %s", iteration, e)
                trace.append(AgentTrace(iteration=iteration, thought="(LLM call failed)", action="error", action_input={}, observation=str(e)))
                return self._finalize(
                    stop_reason=StopReason.hard_failure, final_answer=last_draft, clarification_question=None,
                    confidence=last_confidence, sources_used=sources_used, notes=notes, trace=trace,
                    iterations_used=iteration, total_tokens=total_tokens, start=start,
                )

            usage = result.get("usage", {})
            total_tokens += usage.get("total_tokens", 0)

            try:
                step = AgentStep.model_validate_json(result["content"] or "{}")
            except Exception as e:  # noqa: BLE001
                # Soft failure: model didn't return valid JSON for this step.
                # Record it as a note and let the loop continue (it is not fatal).
                logger.warning("Agent iteration %d: invalid step JSON: %s", iteration, e)
                notes.append(EvidenceNote(iteration=iteration, source="agent", content=f"Produced invalid step JSON: {e}", supports_answer=False))
                trace.append(AgentTrace(iteration=iteration, thought="(invalid JSON)", action="invalid", action_input={}, observation=str(e), tokens_used=usage.get("total_tokens", 0)))
                continue

            action = step.action.value
            action_signature = f"{action}:{json.dumps(step.action_input, sort_keys=True)}"

            # --- Loop detection: same action+input repeated back-to-back ---
            if action_signature == last_action_signature:
                trace.append(AgentTrace(iteration=iteration, thought=step.thought, action=action, action_input=step.action_input, observation="LOOP DETECTED (repeated action) — stopping.", tokens_used=usage.get("total_tokens", 0)))
                return self._finalize(
                    stop_reason=StopReason.loop_detected, final_answer=last_draft, clarification_question=None,
                    confidence=last_confidence, sources_used=sources_used, notes=notes, trace=trace,
                    iterations_used=iteration, total_tokens=total_tokens, start=start,
                )
            last_action_signature = action_signature

            if step.draft_answer:
                last_draft = step.draft_answer
            last_confidence = step.confidence

            # --- Terminal actions ---
            if action == "ask_clarification":
                trace.append(AgentTrace(iteration=iteration, thought=step.thought, action=action, action_input=step.action_input, observation="Stopping to ask the user.", tokens_used=usage.get("total_tokens", 0)))
                return self._finalize(
                    stop_reason=StopReason.ask_clarification, final_answer=None, clarification_question=step.draft_answer or step.thought,
                    confidence=last_confidence, sources_used=sources_used, notes=notes, trace=trace,
                    iterations_used=iteration, total_tokens=total_tokens, start=start,
                )

            if action == "finish":
                # --- Self-check gate: do NOT just trust the model's own
                # "finish" claim (avoids the self-verification paradox —
                # a model grading its own work is a weak check). The
                # orchestrator enforces the cross-source rule in code.
                if len(sources_used) >= self.min_sources and last_confidence >= self.confidence_threshold and last_draft:
                    trace.append(AgentTrace(iteration=iteration, thought=step.thought, action=action, action_input=step.action_input, observation="Accepted: sufficient independent sources and confidence.", tokens_used=usage.get("total_tokens", 0)))
                    return self._finalize(
                        stop_reason=StopReason.finished, final_answer=last_draft, clarification_question=None,
                        confidence=last_confidence, sources_used=sources_used, notes=notes, trace=trace,
                        iterations_used=iteration, total_tokens=total_tokens, start=start,
                    )
                else:
                    reason = []
                    if len(sources_used) < self.min_sources:
                        reason.append(f"only {len(sources_used)}/{self.min_sources} independent sources used")
                    if last_confidence < self.confidence_threshold:
                        reason.append(f"confidence {last_confidence:.2f} below threshold {self.confidence_threshold}")
                    note_text = f"Rejected premature finish: {', '.join(reason)}. Continue gathering evidence."
                    notes.append(EvidenceNote(iteration=iteration, source="agent_self_check", content=note_text, supports_answer=False))
                    trace.append(AgentTrace(iteration=iteration, thought=step.thought, action=action, action_input=step.action_input, observation=note_text, tokens_used=usage.get("total_tokens", 0)))
                    actions_taken.append(f"finish(rejected@{iteration})")
                    continue

            if action == "draft_answer":
                trace.append(AgentTrace(iteration=iteration, thought=step.thought, action=action, action_input=step.action_input, observation="Draft recorded.", tokens_used=usage.get("total_tokens", 0)))
                actions_taken.append("draft_answer")
                continue

            # --- Tool actions ---
            raw_result = self.tool_executor(action, step.action_input, retriever=self.retriever, fail_inject_tool=self.fail_inject_tool)
            note = self._extract_note(iteration, action, raw_result)
            notes.append(note)
            notes = self._cap_notes(notes)
            if "FAILED" not in note.content:
                sources_used.add(action)
            actions_taken.append(action)

            trace.append(AgentTrace(
                iteration=iteration, thought=step.thought, action=action, action_input=step.action_input,
                observation=note.content, tokens_used=usage.get("total_tokens", 0),
                raw_result=json.dumps(raw_result, default=str)[:1500],
            ))

        # --- Max iterations reached: soft failure if we have a draft, else hard failure ---
        stop = StopReason.max_iterations
        return self._finalize(
            stop_reason=stop, final_answer=last_draft, clarification_question=None,
            confidence=last_confidence, sources_used=sources_used, notes=notes, trace=trace,
            iterations_used=self.max_iterations, total_tokens=total_tokens, start=start,
        )

    def _finalize(self, stop_reason, final_answer, clarification_question, confidence, sources_used, notes, trace, iterations_used, total_tokens, start) -> Dict[str, Any]:
        return {
            "final_answer": final_answer,
            "clarification_question": clarification_question,
            "stop_reason": stop_reason,
            "confidence": confidence,
            "sources_used": sorted(sources_used),
            "notes": notes,
            "trace": trace,
            "iterations_used": iterations_used,
            "total_tokens": total_tokens,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }
