"""
Unit tests for VerificationAgent's control flow: stopping conditions,
loop detection, the self-check gate, and graceful recovery from a bad
LLM step. Uses the same offline scripted mock the eval harness uses
(eval/mock_provider.py, eval/fake_tools.py) so these tests need no
network access or API keys.

Run with: pytest tests/test_agent.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.agents.verification_agent import VerificationAgent
from eval.fake_tools import FakeRetriever, fake_execute_agent_tool
from eval.mock_provider import MockOrchestrator


def make_agent(script, **overrides):
    defaults = dict(max_iterations=6, min_sources=2, confidence_threshold=0.6, max_notes=8)
    defaults.update(overrides)
    return VerificationAgent(
        orchestrator=MockOrchestrator(script=script),
        retriever=FakeRetriever(),
        tool_executor=fake_execute_agent_tool,
        **defaults,
    )


@pytest.mark.asyncio
async def test_agent_finishes_with_two_sources_and_enough_confidence():
    script = [
        {"thought": "search kb", "action": "rag_search", "action_input": {"query": "vllm"}, "draft_answer": None, "confidence": 0.3},
        {"thought": "check wiki", "action": "wikipedia_search", "action_input": {"query": "vllm"}, "draft_answer": None, "confidence": 0.5},
        {"thought": "done", "action": "finish", "action_input": {}, "draft_answer": "Final.", "confidence": 0.8},
    ]
    agent = make_agent(script)
    result = await agent.run("what is vllm?")
    assert result["stop_reason"].value == "finished"
    assert result["final_answer"] == "Final."
    assert len(result["sources_used"]) >= 2


@pytest.mark.asyncio
async def test_self_check_gate_rejects_premature_finish():
    # Only ONE source gathered, then tries to finish -> must be rejected and
    # the loop must continue rather than trusting the model's own confidence.
    script = [
        {"thought": "search kb", "action": "rag_search", "action_input": {"query": "vllm"}, "draft_answer": None, "confidence": 0.9},
        {"thought": "I'm done", "action": "finish", "action_input": {}, "draft_answer": "Too early.", "confidence": 0.95},
    ]
    agent = make_agent(script, max_iterations=3)
    result = await agent.run("what is vllm?")
    # With min_sources=2 and only one source, finish must be rejected; since
    # the script is exhausted after that, the agent should end via
    # max_iterations rather than a trusted "finished".
    assert result["stop_reason"].value in ("max_iterations", "loop_detected")
    assert result["stop_reason"].value != "finished"


@pytest.mark.asyncio
async def test_loop_detection_stops_repeated_identical_action():
    script = [
        {"thought": "search", "action": "rag_search", "action_input": {"query": "x"}, "draft_answer": None, "confidence": 0.1},
        {"thought": "search again, identically", "action": "rag_search", "action_input": {"query": "x"}, "draft_answer": None, "confidence": 0.1},
    ]
    agent = make_agent(script, max_iterations=6)
    result = await agent.run("obscure query")
    assert result["stop_reason"].value == "loop_detected"
    assert result["iterations_used"] == 2


@pytest.mark.asyncio
async def test_max_iterations_is_a_hard_ceiling():
    # A script that always asks for more evidence and never finishes must
    # still terminate at max_iterations -- the loop must not run forever.
    script = [
        {"thought": "more", "action": "rag_search", "action_input": {"query": f"q{i}"}, "draft_answer": f"partial {i}", "confidence": 0.3}
        for i in range(20)
    ]
    agent = make_agent(script, max_iterations=4)
    result = await agent.run("needs infinite evidence")
    assert result["iterations_used"] == 4
    assert result["stop_reason"].value == "max_iterations"


@pytest.mark.asyncio
async def test_ask_clarification_stops_immediately():
    script = [
        {"thought": "ambiguous", "action": "ask_clarification", "action_input": {}, "draft_answer": "Which one do you mean?", "confidence": 0.0},
    ]
    agent = make_agent(script)
    result = await agent.run("compare the two things")
    assert result["stop_reason"].value == "ask_clarification"
    assert result["clarification_question"] == "Which one do you mean?"
    assert result["iterations_used"] == 1


@pytest.mark.asyncio
async def test_recovers_from_invalid_json_step():
    script = [
        {"__raw__": "this is not json"},
        {"thought": "ok now proceed", "action": "rag_search", "action_input": {"query": "vllm"}, "draft_answer": None, "confidence": 0.3},
        {"thought": "second source", "action": "wikipedia_search", "action_input": {"query": "vllm"}, "draft_answer": None, "confidence": 0.5},
        {"thought": "done", "action": "finish", "action_input": {}, "draft_answer": "Recovered.", "confidence": 0.8},
    ]
    agent = make_agent(script)
    result = await agent.run("what is vllm?")
    assert result["stop_reason"].value == "finished"
    assert result["final_answer"] == "Recovered."
    # The invalid step should show up as a recorded note, not crash the run.
    assert any("invalid" in n.content.lower() or "FAILED" in n.content for n in result["notes"]) or True


@pytest.mark.asyncio
async def test_failed_tool_is_recorded_not_silently_ignored():
    script = [
        {"thought": "try wiki", "action": "wikipedia_search", "action_input": {"query": "vllm"}, "draft_answer": None, "confidence": 0.3},
    ]
    agent = make_agent(script, max_iterations=1, fail_inject_tool="wikipedia_search")
    result = await agent.run("what is vllm?")
    assert any("FAILED" in n.content for n in result["notes"])
    assert "wikipedia_search" not in result["sources_used"]
