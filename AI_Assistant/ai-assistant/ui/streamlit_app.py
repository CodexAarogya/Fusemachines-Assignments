"""
Streamlit UI for the AI Assistant.

Talks to the FastAPI backend over HTTP (BACKEND_URL). Run with:
    streamlit run ui/streamlit_app.py
or via the `ui` service in docker-compose.yml.
"""
import os
import time

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="centered")
st.title("🤖 AI Assistant")
st.caption(f"Backend: {BACKEND_URL}")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
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

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")
        t0 = time.time()
        try:
            resp = requests.post(
                f"{BACKEND_URL}/chat",
                json={
                    "message": prompt,
                    "use_rag": use_rag,
                    "structured": structured,
                    "temperature": temperature,
                    "top_p": top_p,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            answer = data["answer"]
            if data.get("structured_output"):
                answer += "\n\n```json\n" + str(data["structured_output"]) + "\n```"

            meta_bits = [f"provider={data['provider_used']}", f"latency={data['latency_ms']}ms"]
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
