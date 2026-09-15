import streamlit as st
from typing import Dict, Any, List, Optional
import os

def render_header():
    st.markdown("================================================")
    st.markdown("<h1 style='text-align: center;'>DOCUMENTATION LEARNING ASSISTANT</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>AI-Powered Documentation Exploration & Contextual RAG</p>", unsafe_allow_html=True)
    st.markdown("================================================")

def render_url_input() -> str:
    with st.form("url_form"):
        url = st.text_input("Enter Documentation URL", placeholder="https://playwright.dev/docs/intro")
        submitted = st.form_submit_button("Process Documentation", type="primary")
        if submitted and url:
            return url.strip()
    return ""

def update_progress(placeholder, message: str, current: int = 0, total: int = 0):
    with placeholder.container():
        st.write(message)
        if total > 0:
            st.progress(min(current / total, 1.0))

def render_mode_selector(current_mode: str = "ask") -> str:
    """
    Renders the two primary Phase 2 navigation modes:
    [ 📖 Learn Documentation ]   [ 💬 Ask Questions ]
    """
    col1, col2 = st.columns(2)
    with col1:
        learn_selected = st.button(
            "📖 Learn Documentation",
            use_container_width=True,
            type="primary" if current_mode == "learn" else "secondary"
        )
    with col2:
        ask_selected = st.button(
            "💬 Ask Questions",
            use_container_width=True,
            type="primary" if current_mode == "ask" else "secondary"
        )

    if learn_selected:
        return "learn"
    elif ask_selected:
        return "ask"
    return current_mode

def render_learning_material(content: str):
    if content:
        st.markdown(content)
    else:
        st.info("No learning material generated yet. Click 'Generate Learning Material' to synthesize a structured guide.")

def render_page_browser(metadata: Optional[Dict[str, Any]], pages_dir: str):
    """
    Allows user to browse the original documentation structure and page contents.
    """
    if not metadata or "pages" not in metadata:
        st.info("No page index available.")
        return

    pages = [p for p in metadata["pages"] if p.get("status") == "success"]
    if not pages:
        st.info("No successful pages recorded.")
        return

    st.markdown("### 📑 Browse Original Documentation Pages")
    page_titles = [f"{p.get('title', 'Untitled')} ({p.get('path', '')})" for p in pages]
    
    selected_idx = st.selectbox(
        "Select a documentation page to inspect:",
        range(len(pages)),
        format_func=lambda i: page_titles[i]
    )

    if selected_idx is not None:
        selected_page = pages[selected_idx]
        url = selected_page.get("url", "")
        safe_filename = selected_page.get("url", "").replace("https://", "").replace("http://", "").replace("/", "_").replace("?", "_").replace("=", "_")
        if len(safe_filename) > 100:
            import hashlib
            h = hashlib.md5(url.encode('utf-8')).hexdigest()[:10]
            safe_filename = safe_filename[:80] + "_" + h
        filename = f"{safe_filename}.md"
        filepath = os.path.join(pages_dir, filename)

        st.markdown(f"**URL**: [{url}]({url})")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            with st.expander("View Cleaned Page Content", expanded=True):
                st.markdown(content)
        else:
            st.warning(f"File for page not found: {filename}")

def render_sources(sources: List[Dict[str, str]]):
    """
    Renders clickable, verified sources extracted from retrieved chunks.
    """
    if not sources:
        return
    st.markdown("##### 📚 Sources:")
    for s in sources:
        title = s.get("title", "Documentation")
        url = s.get("url", "")
        section = s.get("section")
        section_info = f" — *{section}*" if section and section != "General" else ""
        if url:
            st.markdown(f"- [{title}]({url}){section_info}")
        else:
            st.markdown(f"- **{title}**{section_info}")
