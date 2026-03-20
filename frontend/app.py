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


def _select_dataset(dataset_id: int):
    """Load a previously uploaded dataset into session state."""
    res = requests.get(f"{API}/datasets/{dataset_id}/preview")
    if res.status_code == 200:
        data = res.json()
        st.session_state.dataset_id = data["dataset_id"]
        st.session_state.filename = data["filename"]
        st.session_state.preview = data["preview"]
        st.session_state.messages = []
        st.session_state.indexed = False

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 AI Data Analyst")
    st.divider()

    # ── Upload new file ──────────────────────────────────────
    st.subheader("⬆️ Upload New File")
    uploaded = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])

    if uploaded:
        if uploaded.name != st.session_state.filename:
            with st.spinner("Uploading..."):
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
                st.rerun()
            else:
                st.error("Upload failed")

    st.divider()

    # ── Previous uploads ─────────────────────────────────────
    st.subheader("🗂️ Previous Uploads")
    try:
        prev_res = requests.get(f"{API}/datasets", timeout=3)
        if prev_res.status_code == 200:
            datasets = prev_res.json()
            if datasets:
                for ds in datasets:
                    is_active = ds["id"] == st.session_state.dataset_id
                    with st.container():
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            name = f"🟢 {ds['filename']}" if is_active else ds["filename"]
                            st.caption(f"**{name}**\n{ds['rows_count'] or '?'} rows · {ds['created_at'][:10]}")
                        with c2:
                            if not is_active:
                                if st.button("Load", key=f"load_{ds['id']}"):
                                    _select_dataset(ds["id"])
                                    st.rerun()
            else:
                st.caption("No previous uploads yet.")
    except Exception:
        st.caption("Could not load previous uploads.")

    # ── Active dataset controls ───────────────────────────────
    if st.session_state.dataset_id:
        st.divider()
        st.caption(f"**Active:** {st.session_state.filename}")

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
tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔍 Data Preview", "📊 Analysis"])

with tab2:
    if st.session_state.preview:
        st.subheader("Dataset Preview")
        st.dataframe(pd.DataFrame(st.session_state.preview), use_container_width=True)

with tab3:
    if not st.session_state.dataset_id:
        st.info("Upload a dataset first.")
    else:
        if st.button("🔍 Run Analysis", type="primary"):
            with st.spinner("Analyzing..."):
                res = requests.get(f"{API}/analysis/{st.session_state.dataset_id}")
            if res.status_code != 200:
                st.error(f"Error: {res.text}")
            else:
                data = res.json()
                analysis = data["analysis"]
                plots = data["plots"]

                # ── Shape ──────────────────────────────────────────
                c1, c2 = st.columns(2)
                c1.metric("Rows", f"{analysis['shape']['rows']:,}")
                c2.metric("Columns", analysis['shape']['cols'])

                # ── Missing data ───────────────────────────────────
                if analysis["missing"]:
                    st.subheader("Missing Data")
                    st.dataframe(pd.DataFrame(analysis["missing"]).T, use_container_width=True)

                # ── Strong correlations ────────────────────────────
                strong = analysis["correlations"].get("strong_pairs", [])
                if strong:
                    st.subheader("Strong Column Relationships")
                    st.dataframe(pd.DataFrame(strong), use_container_width=True)

                # ── Outliers ───────────────────────────────────────
                if analysis["outliers"]:
                    st.subheader("Outliers Detected")
                    st.dataframe(pd.DataFrame(analysis["outliers"]).T, use_container_width=True)

                # ── Cat → Num relationships ────────────────────────
                if analysis["cat_num_relationships"]:
                    st.subheader("Categorical → Numeric Relationships (ANOVA p < 0.05)")
                    st.dataframe(pd.DataFrame(analysis["cat_num_relationships"]), use_container_width=True)

                # ── Plots ──────────────────────────────────────────
                if plots:
                    st.subheader("Key Plots")
                    cols = st.columns(2)
                    for i, (name, img) in enumerate(plots.items()):
                        with cols[i % 2]:
                            st.caption(name.replace("_", " ").title())
                            st.image(BytesIO(base64.b64decode(img)), use_container_width=True)

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
            answer_placeholder = st.empty()
            code_placeholder = st.empty()

            full_text = ""
            code, mode, answer = None, "analysis", ""
            with requests.post(
                f"{API}/chat/stream",
                params={"dataset_id": st.session_state.dataset_id, "question": question},
                stream=True,
                timeout=120
            ) as stream_res:
                if stream_res.status_code == 200:
                    for chunk in stream_res.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk.startswith("__CODE_RESULT__"):
                            import json as _json
                            code_data = _json.loads(chunk[len("__CODE_RESULT__"):])
                            answer = code_data["answer"]
                            code = code_data.get("code")
                            mode = "code"
                            answer_placeholder.markdown(answer)
                            if code:
                                with code_placeholder.expander("🔧 Generated code"):
                                    st.code(code, language="python")
                        else:
                            full_text += chunk
                            answer_placeholder.markdown(full_text + "▌")

                    if mode == "analysis":
                        answer = full_text
                        answer_placeholder.markdown(answer)
                else:
                    st.error(f"Error: {stream_res.status_code} — {stream_res.text}")
                    answer, code, mode = "Error", None, "analysis"

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
                            if chart_data.get("chart"):
                                st.image(BytesIO(base64.b64decode(chart_data["chart"])), caption=chart_data.get("description", "Chart"))
                    except Exception:
                        pass

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "code": code,
                    "mode": mode,
                    "chart": chart_data["chart"] if chart_data else None,
                    "chart_description": chart_data.get("description") if chart_data else None
                })
