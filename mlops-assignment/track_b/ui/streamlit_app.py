"""
Streamlit UI for the AI Assistant.
Talks to the FastAPI backend over HTTP (BACKEND_URL).
Run with: streamlit run ui/streamlit_app.py
"""
import os
import time

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="centered")
st.title("🤖 AI Assistant")
st.caption(f"Backend: {BACKEND_URL}")

tab_chat, tab_agent = st.tabs(["💬 Chat (RAG + Tools)", "🧭 Verification Agent (Task 3)"])

# ---------------------------------------------------------------------------
# Tab 1: W15 single-pass chat
# ---------------------------------------------------------------------------
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Chat Settings")
        use_rag = st.checkbox("Use RAG (search ingested documents)", value=True)
        structured = st.checkbox("Force structured JSON output", value=False)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
        top_p = st.slider("Top-p", 0.0, 1.0, 0.9, 0.05)

        st.divider()
        st.subheader("📄 Ingest a document")
        uploaded_file = st.file_uploader("Upload .txt or .pdf", type=["txt", "pdf", "md"])
        if uploaded_file and st.button("Ingest"):
            with st.spinner("Ingesting..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                try:
                    resp = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=60)
                    resp.raise_for_status()
                    st.success(f"Ingested {resp.json()['chunks_created']} chunks.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Ingestion failed: {e}")

        st.divider()
        if st.button("Check backend health"):
            try:
                resp = requests.get(f"{BACKEND_URL}/health", timeout=10)
                st.json(resp.json())
            except Exception as e:  # noqa: BLE001
                st.error(f"Health check failed: {e}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])

    if prompt := st.chat_input("Ask something...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Thinking...")
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={
                        "message": prompt, "use_rag": use_rag, "structured": structured,
                        "temperature": temperature, "top_p": top_p,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()

                answer = data["answer"]
                if data.get("structured_output"):
                    answer += "\n\n```json\n" + str(data["structured_output"]) + "\n```"

                meta_bits = [f"provider={data['provider_used']}", f"latency={data['latency_ms']}ms", f"tokens={data.get('total_tokens', 0)}"]
                if data.get("cached"):
                    meta_bits.append("cached ✅")
                if data.get("citations"):
                    meta_bits.append(f"{len(data['citations'])} source(s)")
                if data.get("tool_calls"):
                    meta_bits.append(f"{len(data['tool_calls'])} tool call(s)")
                meta = " | ".join(meta_bits)

                placeholder.markdown(answer)
                st.caption(meta)
                st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta})

                if data.get("citations"):
                    with st.expander("Sources"):
                        for c in data["citations"]:
                            st.write(f"- {c['source']} (chunk {c['chunk_id']}, score={c['score']:.3f})")

            except requests.exceptions.RequestException as e:
                placeholder.error(f"Request failed: {e}")

# ---------------------------------------------------------------------------
# Tab 2: W16 agentic verification loop
# ---------------------------------------------------------------------------
with tab_agent:
    st.markdown(
        "Ask a question that benefits from **cross-source verification**. "
        "The agent will decide, step by step, whether to search the internal "
        "knowledge base, check Wikipedia, use a tool, revise its draft, ask "
        "you a clarifying question, or finish — based on what it finds."
    )

    if "agent_session_id" not in st.session_state:
        import uuid
        st.session_state.agent_session_id = uuid.uuid4().hex[:8]

    agent_query = st.text_input("Your question", key="agent_query")
    col1, col2 = st.columns([1, 3])
    run_clicked = col1.button("Run agent")
    if col2.button("New session"):
        import uuid
        st.session_state.agent_session_id = uuid.uuid4().hex[:8]
        st.session_state.pop("agent_result", None)

    if run_clicked and agent_query:
        with st.spinner("Agent is reasoning..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/agent/verify",
                    json={"query": agent_query, "session_id": st.session_state.agent_session_id},
                    timeout=120,
                )
                resp.raise_for_status()
                st.session_state.agent_result = resp.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")

    result = st.session_state.get("agent_result")
    if result:
        if result["stop_reason"] == "ask_clarification":
            st.warning(f"🤔 Agent needs clarification: **{result['clarification_question']}**")
            clarification = st.text_input("Your answer", key="clarification_input")
            if st.button("Submit clarification") and clarification:
                with st.spinner("Resuming..."):
                    resp = requests.post(
                        f"{BACKEND_URL}/agent/verify",
                        json={
                            "query": agent_query,
                            "session_id": st.session_state.agent_session_id,
                            "clarification_answer": clarification,
                        },
                        timeout=120,
                    )
                    st.session_state.agent_result = resp.json()
                    st.rerun()
        else:
            st.success(f"**Stop reason:** {result['stop_reason']} | confidence={result['confidence']:.2f}")
            st.markdown(f"### Answer\n{result.get('final_answer') or '(no answer produced)'}")

        meta = f"iterations={result['iterations_used']} | tokens={result['total_tokens']} | latency={result['latency_ms']}ms | sources={', '.join(result['sources_used']) or 'none'}"
        st.caption(meta)

        with st.expander("🔍 Agent trace (step by step)"):
            for t in result["trace"]:
                st.markdown(f"**Step {t['iteration']} — `{t['action']}`** (tokens: {t.get('tokens_used', 0)})")
                st.write(f"Thought: {t['thought']}")
                st.write(f"Observation: {t['observation']}")
                st.divider()

        with st.expander("📝 Structured notes (context-engineering artifact)"):
            for n in result["notes"]:
                st.write(f"- [{n['source']}] {n['content']}")
