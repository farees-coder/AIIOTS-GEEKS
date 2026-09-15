"""
PDF Question Answering with RAG — Groq + ChromaDB
"""
import os
import sys

# Prevent `transformers` from probing/loading TensorFlow's Keras 3 integration.
# This app is pure PyTorch (sentence-transformers) and never uses TF — but if
# TensorFlow happens to be installed in the environment, `transformers` will
# try to import its TF classes on load and crash with a Keras 3 incompatibility
# error. Setting USE_TF=0 makes `transformers` skip TF detection entirely.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
from src.pdf_processor import process_pdf, PDFProcessingError
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine, LLMError, RAGAnswer, GROQ_MODEL
from src.utils import FileUtils
from src.config import settings
from src.logger import logger

# ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A · Groq RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header {
    font-size: 2.5rem; font-weight: 800;
    background: linear-gradient(90deg, #f55036, #ff8c42);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.sub-header { font-size: 1rem; color: #666; margin-bottom: 1.6rem; }
.stat-card  { background:#fafafa; border-radius:10px; padding:14px 10px;
               text-align:center; border:1px solid #e0e0e0; }
.stat-num   { font-size:1.8rem; font-weight:800; color:#f55036; }
.stat-lbl   { font-size:0.75rem; color:#888; text-transform:uppercase; letter-spacing:.5px; }
.src-box    { background:#f8f9fa; border-left:4px solid #f55036;
               border-radius:0 8px 8px 0; padding:10px 14px; margin:6px 0; }
.src-title  { font-weight:700; color:#f55036; font-size:.87rem; }
.src-body   { color:#444; font-size:.80rem; margin-top:4px; line-height:1.55; }
.src-meta   { color:#999; font-size:.75rem; margin-top:4px; }
.conf-high  { color:#28a745; font-weight:700; }
.conf-mid   { color:#fd7e14; font-weight:700; }
.conf-low   { color:#dc3545; font-weight:700; }
.groq-badge { display:inline-block; background:#f55036; color:#fff;
               padding:3px 12px; border-radius:20px; font-size:.72rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────
def _conf_cls(c: float) -> str:
    return "conf-high" if c > 0.7 else "conf-mid" if c > 0.4 else "conf-low"


def _render_answer(ans: RAGAnswer):
    st.write(ans.answer)
    if ans.sources:
        with st.expander(f"📚 Sources  ({len(ans.sources)} pages)"):
            for s in ans.sources:
                pct = f"{s.get('similarity', 0):.0%}"
                st.markdown(f"""
<div class="src-box">
  <div class="src-title">📄 {s['display']}</div>
  <div class="src-body">{s.get('preview','')}</div>
  <div class="src-meta">Similarity: {pct}</div>
</div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.5, 1, 2])
    cc = _conf_cls(ans.confidence)
    c1.markdown(f'<span class="{cc}">● {ans.confidence:.0%} confidence</span>', unsafe_allow_html=True)
    c2.caption(f"📦 {ans.chunks_used} chunks")
    c3.caption(f"🤖 {ans.model}")


# ── Session state ────────────────────────────────────────────────
for k, v in {
    "vector_store":    None,
    "rag_engine":      None,
    "llm_error":       None,
    "chat_history":    [],
    "processed_files": set(),
    "upload_key":      0,
    "selected_doc":    "📚 All Documents",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.vector_store is None:
    with st.spinner("🔧 Loading vector store…"):
        st.session_state.vector_store = VectorStore()

if st.session_state.rag_engine is None:
    try:
        st.session_state.rag_engine = RAGEngine(st.session_state.vector_store)
        st.session_state.llm_error  = None
    except LLMError as e:
        st.session_state.llm_error = str(e)

_vs:     VectorStore      = st.session_state.vector_store
_eng:    RAGEngine | None = st.session_state.rag_engine
_llm_ok: bool             = _eng is not None


# ────────────────────────────────────────────────────────────────
#  SIDEBAR
# ────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Groq status ──────────────────────────────────────────
    st.markdown("## 🤖 Groq")
    if _llm_ok:
        st.markdown(
            '<span class="groq-badge">GROQ</span> '
            '<span style="color:#28a745;font-weight:600;font-size:.85rem"> ● Connected</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Model: `{GROQ_MODEL}`")
    else:
        st.markdown(
            '<span class="groq-badge">GROQ</span> '
            '<span style="color:#dc3545;font-weight:600;font-size:.85rem"> ● Disconnected</span>',
            unsafe_allow_html=True,
        )
        st.error(st.session_state.llm_error or "Unknown error")
        st.info("🔑 Get a free key at [console.groq.com](https://console.groq.com/keys)")

    # ── Retrieval settings ───────────────────────────────────
    with st.expander("⚙️ Retrieval Settings"):
        new_k = st.slider("Chunks to retrieve (top-k)", 1, 15, settings.top_k_results)
        new_t = st.slider("Similarity threshold", 0.05, 1.0,
                          float(settings.similarity_threshold), step=0.05)
        if new_k != settings.top_k_results:
            object.__setattr__(settings, "top_k_results", new_k)
        if new_t != settings.similarity_threshold:
            object.__setattr__(settings, "similarity_threshold", new_t)

    st.markdown("---")

    # ── Upload ───────────────────────────────────────────────
    st.markdown("## 📁 Upload PDFs")
    uploaded = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"up_{st.session_state.upload_key}",
    )

    if uploaded:
        st.markdown("---")
        for uf in uploaded:
            nc, bc = st.columns([3, 1])
            nc.write(f"📄 **{uf.name}**")
            nc.caption(FileUtils.format_file_size(uf.size))
            if uf.name in st.session_state.processed_files:
                bc.success("✅")
            elif bc.button("Index", key=f"idx_{uf.name}", use_container_width=True):
                bar = st.progress(0, text="Starting…")
                try:
                    bar.progress(15, text="Saving…")
                    fp = FileUtils.save_uploaded_file(uf)
                    bar.progress(45, text="Extracting text…")
                    chunks = process_pdf(fp)
                    bar.progress(80, text="Embedding & indexing…")
                    added = _vs.add_chunks(chunks)
                    bar.progress(100, text="Done!")
                    st.session_state.processed_files.add(uf.name)
                    st.success(f"✅ {added} chunks indexed")
                    logger.info("indexed", file=uf.name, chunks=added)
                    st.rerun()
                except PDFProcessingError as e:
                    bar.empty(); st.error(str(e))
                except Exception as e:
                    bar.empty(); st.error(f"Error: {e}")
                    logger.error("index_error", file=uf.name, error=str(e))

    st.markdown("---")

    # ── Knowledge base stats ─────────────────────────────────
    st.markdown("## 📊 Knowledge Base")
    try:
        kb = _vs.get_stats()
    except Exception:
        kb = {"total_chunks": 0, "unique_files": 0, "total_pages": 0, "files": []}

    if kb["total_chunks"] > 0:
        c1, c2, c3 = st.columns(3)
        for col, val, lbl in [(c1, kb["total_chunks"], "Chunks"),
                               (c2, kb["unique_files"],  "Files"),
                               (c3, kb["total_pages"],   "Pages")]:
            col.markdown(
                f'<div class="stat-card"><div class="stat-num">{val:,}</div>'
                f'<div class="stat-lbl">{lbl}</div></div>',
                unsafe_allow_html=True)

        if kb["files"]:
            with st.expander("📄 Indexed files"):
                for fn in kb["files"]:
                    fc, dc = st.columns([4, 1])
                    fc.markdown(f"`{fn}`")
                    if dc.button("🗑️", key=f"del_{fn}", help=f"Remove {fn}"):
                        _vs.delete_by_filename(fn)
                        st.session_state.processed_files.discard(fn)
                        FileUtils.delete_file(fn)
                        if st.session_state.selected_doc == fn:
                            st.session_state.selected_doc = "📚 All Documents"
                        st.rerun()
    else:
        st.info("📭 No documents indexed yet")

    st.markdown("---")

    # ── Danger zone ──────────────────────────────────────────
    with st.expander("⚠️ Danger Zone"):
        st.warning("Permanently deletes all data.")
        if st.button("🗑️ Clear Everything", type="primary", use_container_width=True):
            _vs.clear()
            FileUtils.clear_upload_dir()
            st.session_state.processed_files.clear()
            st.session_state.chat_history.clear()
            st.session_state.upload_key += 1
            st.session_state.selected_doc = "📚 All Documents"
            st.rerun()


# ────────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📚 PDF Question Answering</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Upload PDFs · Index them · Ask questions — powered by Groq + ChromaDB RAG'
    '</div>', unsafe_allow_html=True)

# Empty state
try:
    _kb = _vs.get_stats()
except Exception:
    _kb = {"total_chunks": 0}

if _kb["total_chunks"] == 0:
    st.info("👈 Upload and index PDF files from the sidebar to get started.")
    with st.expander("🚀 How it works", expanded=True):
        st.markdown("""
| Step | What happens |
|------|-------------|
| **1. Upload** | Drop a PDF in the sidebar |
| **2. Index**  | Text is extracted, split into chunks, embedded with `all-MiniLM-L6-v2`, stored in ChromaDB |
| **3. Ask**    | Your question is embedded, top-k similar chunks retrieved, sent to Groq for a grounded answer |
| **4. Cite**   | Expand **Sources** under any answer to see exact page excerpts and similarity scores |

> **Model:** `groq/compound` — Groq's best multi-model orchestration, free tier.
        """)
    st.stop()

if not _llm_ok:
    st.error(f"⚠️ Groq not connected — {st.session_state.llm_error}")

# ── Document scope selector ────────────────────────────────────────
doc_options = ["📚 All Documents"] + _kb.get("files", [])
if st.session_state.selected_doc not in doc_options:
    st.session_state.selected_doc = "📚 All Documents"

sc1, sc2 = st.columns([3, 2])
with sc1:
    st.selectbox(
        "🔎 Ask questions from",
        options=doc_options,
        key="selected_doc",
        help="Choose a specific PDF to scope your questions to just that document, "
             "or keep 'All Documents' to search across everything indexed.",
    )
with sc2:
    if st.session_state.selected_doc != "📚 All Documents":
        st.caption(f"📄 Scoped to **{st.session_state.selected_doc}**")
    else:
        st.caption("🌐 Searching across all indexed documents")

_scoped_filename = (
    None if st.session_state.selected_doc == "📚 All Documents"
    else st.session_state.selected_doc
)

# ── Chat ─────────────────────────────────────────────────────────
hc, bc = st.columns([5, 1])
hc.markdown("### 💬 Chat")
if st.session_state.chat_history and bc.button("🔄 Clear"):
    st.session_state.chat_history.clear()
    st.rerun()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            _render_answer(msg["content"])

_placeholder = (
    "Ask something about your documents…" if _scoped_filename is None
    else f"Ask something about {_scoped_filename}…"
)
query = st.chat_input(_placeholder, disabled=not _llm_ok)

if query:
    with st.chat_message("user", avatar="👤"):
        st.write(query)
        if _scoped_filename:
            st.caption(f"📄 Scoped to: {_scoped_filename}")
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching & generating…"):
            try:
                result = st.session_state.rag_engine.answer(query, filename=_scoped_filename)
                _render_answer(result)
                st.session_state.chat_history.append({"role": "assistant", "content": result})
                logger.info("query_done", query=query[:60], confidence=result.confidence)
            except LLMError as e:
                st.error(f"❌ Groq error: {e}")
                logger.error("llm_error", error=str(e))
            except Exception as e:
                st.error(f"❌ Error: {e}")
                logger.error("unexpected", error=str(e))

# # ── Footer ───────────────────────────────────────────────────────
# st.markdown("---")
# f1, f2, f3 = st.columns(3)
# f1.caption("🔧 ChromaDB · Sentence-Transformers · Groq · Streamlit")
# f2.caption("📖 [Groq Docs](https://docs.groq.com)")
# f3.caption("🔑 [Get Free API Key](https://console.groq.com/keys)")
