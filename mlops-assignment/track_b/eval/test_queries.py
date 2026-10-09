"""
Evaluation dataset for the VerificationAgent (Task 3).

Each test case supplies a SCRIPT: the exact sequence of AgentStep-shaped
decisions MockOrchestrator will hand back, in order, standing in for what
an LLM would decide at each iteration. This keeps the harness deterministic
and offline while still exercising the real VerificationAgent control flow
(stopping conditions, self-check gate, note extraction, loop detection,
tool-arg handling) against real code, not a simulation of it.

`expected_outcome` is what a CORRECT agent should do for that query;
the harness grades `task_completion` by comparing it to what actually
happened (see eval/run_eval.py: `EXPECTED_VS_ACTUAL`).
"""

TEST_CASES = [
    {
        "id": "happy_path_cross_source",
        "query": "What technique does vLLM use to improve throughput, and is that confirmed by an external source?",
        "description": "Two independent, agreeing sources -> should finish with a confident answer.",
        "expected_outcome": "finished",
        "expected_tools": {"rag_search", "wikipedia_search"},
        "fail_inject_tool": None,
        "script": [
            {"thought": "Check the internal knowledge base first.", "action": "rag_search", "action_input": {"query": "vLLM throughput technique"}, "draft_answer": None, "confidence": 0.3},
            {"thought": "Cross-check against an independent external source before trusting one source alone.", "action": "wikipedia_search", "action_input": {"query": "vllm"}, "draft_answer": None, "confidence": 0.4},
            {"thought": "Both sources agree on PagedAttention; confidence is now high.", "action": "finish", "action_input": {}, "draft_answer": "vLLM improves throughput using PagedAttention, an efficient KV-cache management technique. This is confirmed both by the internal course notes and by an external Wikipedia summary.", "confidence": 0.9},
        ],
    },
    {
        "id": "calculator_tool_use",
        "query": "If I ingest 3 documents averaging 40 chunks each, how many total chunks is that, and does the internal KB design support that scale?",
        "description": "Requires the calculator tool with correctly-typed arguments, plus one more source before finishing.",
        "expected_outcome": "finished",
        "expected_tools": {"calculator", "rag_search"},
        "fail_inject_tool": None,
        "script": [
            {"thought": "First compute the arithmetic part.", "action": "calculator", "action_input": {"expression": "3 * 40"}, "draft_answer": None, "confidence": 0.3},
            {"thought": "Now check whether the internal docs say anything about scale.", "action": "rag_search", "action_input": {"query": "vector database scale chunks"}, "draft_answer": None, "confidence": 0.5},
            {"thought": "I have the arithmetic result and supporting context from two sources (calculator + KB).", "action": "finish", "action_input": {}, "draft_answer": "That's 120 total chunks. The internal notes describe Chroma as suitable for this kind of small-to-medium scale.", "confidence": 0.75},
        ],
    },
    {
        "id": "ambiguous_needs_clarification",
        "query": "Compare the two options and tell me which is better.",
        "description": "Query is underspecified (no options named) -> agent should recognize this and ask, not guess.",
        "expected_outcome": "ask_clarification",
        "expected_tools": set(),
        "fail_inject_tool": None,
        "script": [
            {"thought": "The query references 'the two options' but none were specified anywhere in the conversation. Guessing would be unreliable.", "action": "ask_clarification", "action_input": {}, "draft_answer": "Could you tell me which two options you'd like me to compare?", "confidence": 0.0},
        ],
    },
    {
        "id": "conflicting_evidence_lowers_confidence",
        "query": "Does vLLM require a GPU, and is that confirmed externally?",
        "description": "Internal notes and external source give partially conflicting detail -> agent should not finish prematurely; self-check gate rejects low-confidence finish and forces another round.",
        "expected_outcome": "finished",
        "expected_tools": {"rag_search", "wikipedia_search"},
        "fail_inject_tool": None,
        "script": [
            {"thought": "Check internal notes.", "action": "rag_search", "action_input": {"query": "vLLM GPU requirement"}, "draft_answer": None, "confidence": 0.3},
            {"thought": "Internal notes don't explicitly mention GPU requirement; check external source.", "action": "wikipedia_search", "action_input": {"query": "vllm"}, "draft_answer": "vLLM likely requires a GPU for practical throughput, consistent with its PagedAttention design, though the internal notes don't state this explicitly.", "confidence": 0.5},
            {"thought": "Confidence is below my own threshold given the partial mismatch; I should finish now since I have two independent sources and the evidence is reasonably consistent even if not word-for-word identical.", "action": "finish", "action_input": {}, "draft_answer": "vLLM is designed for GPU-accelerated serving; the internal notes focus on its throughput technique (PagedAttention) while the external summary confirms it is an LLM inference/serving engine, consistent with a GPU-based design.", "confidence": 0.65},
        ],
    },
    {
        "id": "failure_injection_wikipedia_down",
        "query": "What is Retrieval-Augmented Generation, confirmed by an external source?",
        "description": (
            "FAILURE INJECTION TEST: wikipedia_search is forced to fail, leaving only one "
            "of the two required independent sources available. The self-check gate will "
            "correctly refuse to let the agent 'finish' on one source pretending to be "
            "cross-verified, so the expected (correct) outcome is that the run does NOT "
            "end in a confident 'finished' state -- it should end via loop_detected or "
            "max_iterations instead, having been blocked from fabricating confirmation."
        ),
        "expected_outcome": "recognizes_failure",
        "expected_tools": {"rag_search", "wikipedia_search"},
        "fail_inject_tool": "wikipedia_search",
        "script": [
            {"thought": "Check internal notes first.", "action": "rag_search", "action_input": {"query": "retrieval augmented generation"}, "draft_answer": None, "confidence": 0.3},
            {"thought": "Now cross-check externally.", "action": "wikipedia_search", "action_input": {"query": "retrieval-augmented generation"}, "draft_answer": None, "confidence": 0.3},
            {"thought": "The external tool failed, so I only have one working source. I should not claim external confirmation I don't have. I'll answer from the internal source alone with reduced confidence rather than pretending it was cross-verified.", "action": "finish", "action_input": {}, "draft_answer": "RAG combines retrieval with generation to reduce hallucination (per internal notes). I was NOT able to independently confirm this externally right now because the external lookup tool failed.", "confidence": 0.5},
        ],
    },
    {
        "id": "loop_detection_repeated_action",
        "query": "What is the airspeed velocity of an unladen swallow, confirmed by two sources?",
        "description": "Out-of-scope query with nothing in the KB or Wikipedia fixtures. Scripted to repeat the identical action twice, triggering the agent's loop-detection guard.",
        "expected_outcome": "loop_detected",
        "expected_tools": {"rag_search"},
        "fail_inject_tool": None,
        "script": [
            {"thought": "Search internal KB.", "action": "rag_search", "action_input": {"query": "swallow airspeed"}, "draft_answer": None, "confidence": 0.1},
            {"thought": "Nothing relevant came back; try the exact same search again.", "action": "rag_search", "action_input": {"query": "swallow airspeed"}, "draft_answer": None, "confidence": 0.1},
        ],
    },
    {
        "id": "never_converges_hits_max_iterations",
        "query": "Give me a fully verified, numerically precise answer with five independent sources.",
        "description": "Script intentionally never reaches a terminal action within the iteration budget -> should hit max_iterations with a soft-failure (partial draft) outcome.",
        "expected_outcome": "max_iterations",
        "expected_tools": {"rag_search", "wikipedia_search"},
        "fail_inject_tool": None,
        "script": [
            {"thought": "Gather more.", "action": "rag_search", "action_input": {"query": "verification"}, "draft_answer": "Partial answer so far.", "confidence": 0.3},
            {"thought": "Still not enough sources.", "action": "wikipedia_search", "action_input": {"query": "mistral ai"}, "draft_answer": "Partial answer, more sources needed.", "confidence": 0.35},
            {"thought": "Need a third source, but none are available; keep trying anyway.", "action": "rag_search", "action_input": {"query": "five sources"}, "draft_answer": "Still gathering.", "confidence": 0.35},
            {"thought": "Try again.", "action": "wikipedia_search", "action_input": {"query": "llama (language model)"}, "draft_answer": "Still gathering, not yet confident.", "confidence": 0.4},
            {"thought": "One more attempt.", "action": "rag_search", "action_input": {"query": "numerically precise"}, "draft_answer": "Best available partial answer.", "confidence": 0.45},
        ],
    },
    {
        "id": "malformed_model_output_then_recovers",
        "query": "What does context saturation mean, confirmed by an external source if possible?",
        "description": "First iteration simulates the model returning invalid (non-schema) JSON -- a soft failure the agent must tolerate and recover from, not crash on.",
        "expected_outcome": "finished",
        "expected_tools": {"rag_search", "wikipedia_search"},
        "fail_inject_tool": None,
        "script": [
            {"__raw__": "Sure! Context saturation is when..."},
            {"thought": "Retry with proper structured output: check internal notes.", "action": "rag_search", "action_input": {"query": "context saturation"}, "draft_answer": None, "confidence": 0.3},
            {"thought": "No Wikipedia fixture exists for this specific term, so this source lookup will come back empty -- note that and proceed with appropriate confidence.", "action": "wikipedia_search", "action_input": {"query": "context engineering"}, "draft_answer": None, "confidence": 0.4},
            {"thought": "Internal source is clear; external lookup came back empty rather than contradicting it, so I'll finish with moderate-high confidence from the one strong source plus a documented second attempt.", "action": "finish", "action_input": {}, "draft_answer": "Context saturation is when appending full raw tool outputs to the conversation history each iteration makes the prompt grow unboundedly, eventually crowding out the original instructions.", "confidence": 0.7},
        ],
    },
    {
        "id": "malformed_tool_arguments",
        "query": "What's the weather for the trip, and how does that relate to deployment regions in the notes?",
        "description": "Tool-call-correctness probe: first call to get_weather is scripted WITHOUT the required 'city' argument.",
        "expected_outcome": "finished",
        "expected_tools": {"get_weather", "rag_search"},
        "fail_inject_tool": None,
        "script": [
            {"thought": "Check the weather.", "action": "get_weather", "action_input": {}, "draft_answer": None, "confidence": 0.2},
            {"thought": "That failed because I didn't specify a city; retry with one.", "action": "get_weather", "action_input": {"city": "Paris"}, "draft_answer": None, "confidence": 0.3},
            {"thought": "Also check internal notes for deployment-region context.", "action": "rag_search", "action_input": {"query": "deployment region"}, "draft_answer": None, "confidence": 0.4},
            {"thought": "I have weather plus internal context from two tool sources.", "action": "finish", "action_input": {}, "draft_answer": "Paris is currently 18C and partly cloudy; the internal notes don't tie weather to deployment regions directly, so that connection is weak.", "confidence": 0.6},
        ],
    },
]
