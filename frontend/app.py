import os
import time
import streamlit as st
import requests
import pandas as pd
import base64
from io import BytesIO

API = os.getenv("API_URL", "http://host.docker.internal:8000")

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)

# ── Session state ──────────────────────────────────────────────
if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "preview" not in st.session_state:
    st.session_state.preview = None
if "indexed" not in st.session_state:
    st.session_state.indexed = False

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 AI Data Analyst")
    st.divider()

    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    if uploaded:
        if uploaded.name != st.session_state.filename:
            with st.spinner("Uploading and indexing..."):
                res = requests.post(
                    f"{API}/upload",
                    files={"file": (uploaded.name, uploaded.getvalue())}
                )
            if res.status_code == 200:
                data = res.json()
                st.session_state.dataset_id = data["dataset_id"]
                st.session_state.filename = data["filename"]
                st.session_state.preview = data["preview"]
                st.session_state.messages = []
                st.session_state.indexed = False
                st.success(f"✅ Uploaded: {data['filename']}")
                st.caption(f"{data['rows_count']} rows × {data['columns_count']} columns")
                st.info(data.get("message", ""))
            else:
                st.error("Upload failed")

    if st.session_state.dataset_id:
        st.divider()
        st.caption(f"**Active dataset:** {st.session_state.filename}")

        # Auto-poll index status
        if not st.session_state.indexed:
            try:
                res = requests.get(f"{API}/index-status/{st.session_state.dataset_id}", timeout=2)
                if res.status_code == 200 and res.json().get("indexed"):
                    st.session_state.indexed = True
                else:
                    st.warning("⏳ Building index...")
                    time.sleep(3)
                    st.rerun()
            except Exception:
                st.warning("⏳ Waiting for backend...")
                time.sleep(3)
                st.rerun()

        if st.session_state.indexed:
            st.success("✅ Index ready")

        if st.button("🗑️ Clear conversation"):
            requests.delete(f"{API}/chat/{st.session_state.dataset_id}")
            st.session_state.messages = []
            st.rerun()

        if st.button("🔄 Rebuild index"):
            with st.spinner("Rebuilding..."):
                requests.post(f"{API}/rebuild-index/{st.session_state.dataset_id}")
            st.success("Index rebuilt")
            st.session_state.indexed = True

# ── Main area ──────────────────────────────────────────────────
if not st.session_state.dataset_id:
    st.title("Welcome to AI Data Analyst 👋")
    st.write("Upload a CSV or Excel file from the sidebar to get started.")
    st.stop()

# Chat input must be outside tabs
question = st.chat_input("Ask anything about your data...")

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Chat", "🔍 Data Preview"])

with tab2:
    if st.session_state.preview:
        st.subheader("Dataset Preview")
        st.dataframe(pd.DataFrame(st.session_state.preview), use_container_width=True)

with tab1:
    st.subheader(f"Chat with **{st.session_state.filename}**")

    if not st.session_state.indexed:
        st.warning("⏳ Index is still building, please wait...")
        st.stop()

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("code"):
                with st.expander("🔧 Generated code"):
                    st.code(msg["code"], language="python")
            if msg.get("chart"):
                st.image(BytesIO(base64.b64decode(msg["chart"])), caption=msg.get("chart_description", "Chart"))

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = requests.post(
                    f"{API}/chat",
                    params={"dataset_id": st.session_state.dataset_id, "question": question}
                )

            if res.status_code == 200:
                data = res.json()
                answer = data.get("answer", "No response")
                code = data.get("code")
                mode = data.get("mode")

                st.markdown(answer)
                if code:
                    with st.expander("🔧 Generated code"):
                        st.code(code, language="python")

                # Try to generate visualization
                chart_data = None
                if any(word in question.lower() for word in ["plot", "chart", "graph", "visualize", "show", "distribution", "correlation"]):
                    try:
                        viz_res = requests.post(
                            f"{API}/visualize",
                            params={"dataset_id": st.session_state.dataset_id, "question": question}
                        )
                        if viz_res.status_code == 200:
                            chart_data = viz_res.json()
                            st.image(BytesIO(base64.b64decode(chart_data["chart"])), caption=chart_data.get("description", "Chart"))
                    except Exception:
                        pass  # Silently fail if visualization doesn't work

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "code": code,
                    "mode": mode,
                    "chart": chart_data["chart"] if chart_data else None,
                    "chart_description": chart_data.get("description") if chart_data else None
                })
            else:
                st.error(f"Error: {res.status_code} — {res.text}")
