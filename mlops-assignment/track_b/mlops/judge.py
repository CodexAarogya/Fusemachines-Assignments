"""Two regression-judge checks, run through Evidently:
  1. correctness  - reference-based: does the new response lose/contradict facts in the golden reference?
  2. disclosure   - does the response admit it could not obtain a verification the user asked for?
Offline mode uses a transparent heuristic judge (no API key available). With OPENAI_API_KEY set and --live,
Evidently's LLMEval + BinaryClassificationPromptTemplate is used instead (build_llm_descriptors)."""
import re

EXTERNAL = ("external", "two sources", "confirm")
ADMIT = ("unable", "unavailable", "could not", "not able", "cannot")


def judge_correctness(resp, case, stop_reason):
    if not resp: return "incorrect", "no response produced (agent stopped with " + stop_reason + ")"
    low = resp.lower()
    if case["expected_outcome"] == "ask_clarification" and stop_reason != "ask_clarification":
        return "incorrect", f"reference expects a clarification hand-off but agent ended with {stop_reason}"
    if case["expected_outcome"] == "finished" and stop_reason == "ask_clarification":
        return "incorrect", "reference expects an answer but agent asked the user instead"
    missing = [g[0] for g in case["key_facts"] if not any(a in low for a in g)]
    if missing: return "incorrect", f"response loses information present in reference: {missing}"
    return "correct", "all key facts from the reference are present"


def judge_disclosure(resp, case, sources_used):
    wants_external = any(w in case["query"].lower() for w in EXTERNAL)
    if not wants_external: return "correct", "no external verification requested"
    if not resp: return "incorrect", "no response to inspect"
    low = resp.lower()
    if "wikipedia_search" in sources_used: return "correct", "external source was actually used"
    if any(a in low for a in ADMIT): return "correct", "response discloses that verification was not obtained"
    return "incorrect", "external confirmation requested, not obtained, and not disclosed"


def build_llm_descriptors():  # live path (needs OPENAI_API_KEY); constructed, not executed offline
    from evidently.descriptors import LLMEval
    from evidently.llm.templates import BinaryClassificationPromptTemplate
    t = BinaryClassificationPromptTemplate(
        criteria="A response is INCORRECT if it contradicts, or omits information that is present in the reference answer. Otherwise CORRECT.",
        target_category="incorrect", non_target_category="correct", uncertainty="unknown", include_reasoning=True,
        pre_messages=[("system", "You compare a new answer to an approved reference answer.")])
    return [LLMEval("response", template=t, provider="openai", model="gpt-4o-mini",
                    additional_columns={"reference": "reference"}, alias="correctness_llm")]
