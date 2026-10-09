"""Prompt-conditioned offline stand-in for the LLM (used when no API key / network is available).
It sees ONLY the prompt text the real model would see and obeys a rule only if that rule's wording is
present in the system prompt. It is a surrogate, not a real model: use `--live` for real behaviour."""
import json, re
from eval.mock_provider import count_tokens

KB = {"internal_kb": ("rag_search", "the internal KB"), "wikipedia": ("wikipedia_search", "Wikipedia"),
      "calculator": ("calculator", "the calculator"), "weather": ("get_weather", "the weather service")}


def wiki_term(q):
    for k, t in [("vllm", "vllm"), ("retrieval-augmented", "retrieval-augmented generation"), ("rag", "retrieval-augmented generation"),
                 ("context saturation", "context engineering")]:
        if k in q: return t
    return "unladen swallow" if "swallow" in q else q[:40]


def plan(query):
    q, pre = query.lower(), []
    m = re.search(r"(\d+) documents averaging (\d+)", q)
    if m: pre.append(("calculator", {"expression": f"{m[1]} * {m[2]}"}))
    if "weather" in q: pre.append(("get_weather", {"city": "Paris" if "paris" in q else "Tokyo"}))
    steps = pre + [("rag_search", {"query": query})]
    if any(w in q for w in ("external", "wikipedia", "two sources")): steps.append(("wikipedia_search", {"query": wiki_term(q)}))
    return steps


def clean(c): return re.sub(r"^\(score=[\d.]+\)\s*", "", c).strip()


class PromptPolicyOrchestrator:
    async def chat(self, messages, **kw):
        sysmsg, user = messages[0]["content"], messages[1]["content"]
        cross = "at least two independent sources" in sysmsg
        err = "never repeat the same call" in sysmsg
        clar = "underspecified" in sysmsg
        q = re.search(r"ORIGINAL QUESTION:\n(.*?)\n\nEVIDENCE NOTES", user, re.S).group(1)
        block = re.search(r"EVIDENCE NOTES SO FAR:\n(.*?)\n\nACTIONS", user, re.S).group(1)
        notes = re.findall(r"^- \[(.*?)\] (.*)$", block, re.M)
        ok = [(s, c) for s, c in notes if s in KB and not c.startswith("FAILED")]
        failed = [s for s, c in notes if c.startswith("FAILED")]
        done = {KB[s][0] for s, _ in ok}
        pending = [(a, i) for a, i in plan(q) if a not in done and a not in failed]
        step = self._decide(q, ok, failed, pending, cross, err, clar)
        content = json.dumps(step)
        i, o = count_tokens(sysmsg + user), count_tokens(content)
        return {"content": content, "tool_calls": [], "usage": {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}, "provider_used": "policy-sim"}

    def _say(self, ok):  # compose an answer from the notes
        return " ".join(f"According to {KB[s][1]}: {clean(c)}" for s, c in ok)

    def _decide(self, q, ok, failed, pending, cross, err, clar):
        S = lambda t, a, i=None, d=None, c=0.3: {"thought": t, "action": a, "action_input": i or {}, "draft_answer": d, "confidence": c}
        if clar and not ok and not failed and re.search(r"the two options|which is better", q.lower()):
            return S("The question refers to options the user never named; guessing would be unreliable.", "ask_clarification", d="Which two options would you like me to compare?")
        if not cross:  # v1 behaviour: first usable evidence is good enough
            if ok: return S("I have a result; answering.", "finish", d=clean(ok[0][1]), c=0.8)
            if pending: return S(f"Start with {pending[0][0]}.", pending[0][0], pending[0][1])
            return S("Retry the failed call.", "rag_search", {"query": q})
        if len(ok) >= 2: return S("Two independent sources collected; answering.", "finish", d=self._say(ok), c=0.85)
        if pending: return S(f"Need another source: {pending[0][0]}.", pending[0][0], pending[0][1])
        if not failed:  # plan exhausted with one source -> try an external check
            return S("Only one source so far; try Wikipedia.", "wikipedia_search", {"query": wiki_term(q.lower())})
        if err:
            gone = ", ".join(KB.get(f, (f, f))[1] if f in KB else f for f in failed)
            verified = self._say(ok) or "nothing reliable"
            return S("A source failed; I must not repeat it or claim verification I lack.", "ask_clarification",
                     d=f"I could only partly verify this. Verified: {verified} I was unable to verify it with {gone} (unavailable or no result). How would you like to proceed?", c=0.3)
        last = failed[-1]  # v2 behaviour: repeats the failing call
        return S("The call failed; try again.", last, {"query": wiki_term(q.lower())} if last == "wikipedia_search" else {"query": q})
