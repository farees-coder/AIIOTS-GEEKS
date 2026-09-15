import streamlit as st
import asyncio
import os
import urllib.parse
from dotenv import load_dotenv

# Force reload of .env file
load_dotenv(override=True)

from scraper.scraper import Scraper
from scraper.url_utils import validate_url
from knowledge.storage import KnowledgeStorage
from knowledge.parser import extract_title
from knowledge.structure import group_pages_into_sections
from knowledge.chunker import SectionChunker
from knowledge.vector_store import VectorStore
import importlib
import llm.prompts
importlib.reload(llm.prompts)

from llm.section_generator import SectionGenerator
from llm.final_generator import FinalGenerator
from llm.rag_generator import RAGGenerator
from frontend.ui import (
    render_header,
    render_url_input,
    update_progress,
    render_mode_selector,
    render_learning_material,
    render_page_browser,
    render_sources
)

# Initialize singletons
storage = KnowledgeStorage()
chunker = SectionChunker()
vector_store = VectorStore()
section_generator = SectionGenerator()
final_generator = FinalGenerator()
rag_generator = RAGGenerator()

def init_session_state():
    if "url" not in st.session_state:
        st.session_state.url = ""
    if "doc_id" not in st.session_state:
        st.session_state.doc_id = ""
    if "metadata" not in st.session_state:
        st.session_state.metadata = None
    if "stage" not in st.session_state:
        # Check if previous crawl exists
        if storage.has_documentation():
            meta = storage.get_metadata()
            st.session_state.metadata = meta
            st.session_state.url = meta.get("source_url", "")
            st.session_state.doc_id = storage.get_doc_id(st.session_state.url)
            st.session_state.stage = "ready"
        else:
            st.session_state.stage = "input"
    if "mode" not in st.session_state:
        st.session_state.mode = "ask" # 'ask' or 'learn'
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "all"

def do_crawl(url: str, placeholder):
    max_pages = int(os.getenv("MAX_PAGES", "1000"))
    max_depth = int(os.getenv("MAX_DEPTH", "20"))
    
    scraper = Scraper(start_url=url, max_pages=max_pages, max_depth=max_depth)
    
    progress_state = {
        "discovered": 1,
        "processed": 0,
        "failed": 0,
        "current_url": ""
    }
    
    failed_pages = []
    
    def on_page_crawled(crawled_url, visited_count, queue):
        progress_state["processed"] = visited_count
        progress_state["discovered"] = visited_count + len(queue)
        progress_state["current_url"] = crawled_url
        
        msg = f"""
        **Crawling documentation...**\n
        Pages discovered: {progress_state["discovered"]}
        Pages processed: {progress_state["processed"]}
        Pages remaining: {len(queue)}
        
        Current page:
        {crawled_url}
        """
        update_progress(placeholder, msg)
        
    def on_error(failed_url, error_msg):
        progress_state["failed"] += 1
        failed_pages.append({"url": failed_url, "error": error_msg})
        
    scraper.set_callbacks(on_page_crawled=on_page_crawled, on_error=on_error)
    
    cleaned_results = asyncio.run(scraper.scrape_and_clean())
    
    pages_info = []
    for page_url, content in cleaned_results.items():
        title = extract_title(content, page_url)
        storage.save_page(page_url, content, title)
        pages_info.append({"url": page_url, "title": title, "path": urllib.parse.urlparse(page_url).path, "status": "success"})
        
    for fail in failed_pages:
        pages_info.append({"url": fail["url"], "status": "failed", "error": fail["error"]})
        
    storage.save_metadata(url, pages_info)
    storage.build_master_documentation(url, pages_info)
    
    st.session_state.metadata = storage.get_metadata()
    st.session_state.doc_id = storage.get_doc_id(url)
    st.session_state.stage = "ready"
    st.session_state.messages = []
    st.rerun()

def ensure_indexed(doc_id: str, metadata: dict, placeholder) -> int:
    """Chunks and embeds cleaned documentation pages into ChromaDB if not already indexed."""
    if vector_store.is_indexed(doc_id):
        return vector_store.count(doc_id)

    pages = [p for p in metadata.get("pages", []) if p.get("status") == "success"]
    if not pages:
        return 0

    all_chunks = []
    total_pages = len(pages)

    update_progress(placeholder, f"Chunking {total_pages} cleaned documentation pages for vector search...", 0, total_pages)

    for idx, p in enumerate(pages):
        url = p["url"]
        title = p.get("title") or "Page"
        filename = storage._get_filename(url)
        filepath = os.path.join(storage.pages_dir, filename)

        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            page_id = filename.replace(".md", "")
            chunks = chunker.chunk_page(
                content=content,
                documentation_id=doc_id,
                page_id=page_id,
                url=url,
                page_title=title,
                page_order=idx
            )
            all_chunks.extend(chunks)

    update_progress(placeholder, f"Indexing {len(all_chunks)} chunks with all-MiniLM-L6-v2 embeddings in ChromaDB...", 1, 2)
    vector_store.add_chunks(doc_id, all_chunks)
    update_progress(placeholder, f"Successfully indexed {len(all_chunks)} chunks!", 2, 2)
    return len(all_chunks)

def do_generate_learning_guide(placeholder):
    metadata = st.session_state.metadata
    if not metadata:
        st.error("No metadata found. Please crawl again.")
        return
        
    pages_info = metadata.get("pages", [])
    success_urls = [p["url"] for p in pages_info if p.get("status") == "success"]
    
    sections = group_pages_into_sections(success_urls)
    section_summaries = []
    total_sections = len(sections)
    
    for i, (section_name, urls) in enumerate(sections.items()):
        msg = f"Generating learning material for section '{section_name}' ({i+1}/{total_sections})..."
        update_progress(placeholder, msg, i, total_sections)
        
        cached_summary = storage.get_section_summary(section_name)
        if cached_summary:
            section_summaries.append(cached_summary)
            continue
            
        section_content = f"# Section: {section_name}\n\n"
        for url in urls:
            filename = storage._get_filename(url)
            file_path = os.path.join(storage.pages_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    section_content += f.read() + "\n\n---\n\n"
                    
        summary = section_generator.generate_section_summary(section_content)
        if summary:
            storage.save_section_summary(section_name, summary)
            section_summaries.append(summary)
            
    parsed_start = urllib.parse.urlparse(st.session_state.url)
    domain = parsed_start.netloc + parsed_start.path
    
    update_progress(placeholder, "Combining sections into a final structured guide...", total_sections, total_sections)
    
    cached_final = storage.get_final_guide(domain)
    if not cached_final:
        final_summary = final_generator.generate_final_guide(section_summaries)
        if final_summary:
            storage.save_final_guide(domain, final_summary)
            
    update_progress(placeholder, "Done!", total_sections, total_sections)
    st.rerun()

def main():
    st.set_page_config(page_title="Documentation Learning Assistant", page_icon="📚", layout="wide")
    init_session_state()
    
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is missing. Please set it in the .env file.")
        st.stop()

    render_header()
    
    # Sidebar: Documentation management & switching
    with st.sidebar:
        st.markdown("### ⚙️ Documentation Info")
        if st.session_state.url:
            st.markdown(f"**Current URL:**\n`{st.session_state.url}`")
            if st.session_state.doc_id:
                chunk_cnt = vector_store.count(st.session_state.doc_id)
                st.markdown(f"**Chroma Collection:** `{st.session_state.doc_id}`")
                st.markdown(f"**Indexed Chunks:** `{chunk_cnt}`")
                
            st.markdown("---")
            if st.button("🔄 Switch / Process New URL"):
                st.session_state.stage = "input"
                st.session_state.messages = []
                st.rerun()
        else:
            st.info("No documentation currently loaded.")

    # Main Workflow
    if st.session_state.stage == "input":
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            st.markdown("### 📥 Enter Documentation to Process")
            st.write("Provide the starting documentation URL. The assistant will crawl, clean, index into ChromaDB, and prepare interactive RAG.")
            url = render_url_input()
            if url:
                if not validate_url(url):
                    st.error("Invalid URL provided. Please provide a valid http or https URL.")
                else:
                    st.session_state.url = url
                    st.session_state.doc_id = storage.get_doc_id(url)
                    st.session_state.stage = "crawling"
                    st.rerun()

    elif st.session_state.stage == "crawling":
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            st.info(f"Documentation URL: {st.session_state.url}")
            progress_placeholder = st.empty()
            do_crawl(st.session_state.url, progress_placeholder)

    elif st.session_state.stage == "ready":
        metadata = st.session_state.metadata
        doc_id = st.session_state.doc_id or storage.get_doc_id(st.session_state.url)
        
        # Ensure ChromaDB has indexed the chunks
        if not vector_store.is_indexed(doc_id):
            index_placeholder = st.empty()
            ensure_indexed(doc_id, metadata, index_placeholder)
            index_placeholder.empty()

        # Top summary banner
        col_info1, col_info2 = st.columns([3, 1])
        with col_info1:
            total_p = metadata.get("total_pages", 0) if metadata else 0
            st.success(f"✓ **Loaded:** {st.session_state.url} ({total_p} pages indexed in collection `{doc_id}`)")
        with col_info2:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        # Dual Mode Switcher: [ 📖 Learn Documentation ]   [ 💬 Ask Questions ]
        selected_mode = render_mode_selector(st.session_state.mode)
        if selected_mode != st.session_state.mode:
            st.session_state.mode = selected_mode
            st.rerun()

        st.markdown("---")

        # MODE 1: ASK QUESTIONS (RAG)
        if st.session_state.mode == "ask":
            st.markdown("### 💬 Ask Questions About This Documentation")
            
            # Page-specific dropdown filter
            pages = metadata.get("pages", []) if metadata else []
            success_pages = [p for p in pages if p.get("status") == "success"]
            
            filter_options = ["all"] + [p["url"] for p in success_pages]
            filter_labels = ["🌐 All Documentation (Entire Site)"] + [
                f"{p.get('title', 'Page')} ({p.get('path', '')})" for p in success_pages
            ]
            
            selected_filter = st.selectbox(
                "Filter question scope to specific page (optional):",
                range(len(filter_options)),
                format_func=lambda i: filter_labels[i],
                help="Select a specific topic or page to prioritize relevant context, or search all documentation."
            )
            page_filter_val = filter_options[selected_filter]

            # Render Conversation History
            chat_container = st.container()
            with chat_container:
                if not st.session_state.messages:
                    st.info("👋 Ask any question about the documentation above! The assistant uses RAG to retrieve only relevant context and cites source pages.")
                else:
                    for msg in st.session_state.messages:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        sources = msg.get("sources", [])
                        
                        with st.chat_message(role):
                            st.markdown(content)
                            if role == "assistant" and sources:
                                render_sources(sources)

            # User Input
            if prompt := st.chat_input("Ask a question about the documentation..."):
                # Append user query
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Searching documentation and generating answer..."):
                        # 1. Retrieve top 4-6 relevant chunks from ChromaDB
                        filter_arg = None if page_filter_val == "all" else page_filter_val
                        retrieved = vector_store.query(
                            doc_id=doc_id,
                            query_text=prompt,
                            n_results=5,
                            page_filter=filter_arg
                        )
                        
                        # 2. Call Groq with RAG context & history
                        res = rag_generator.generate_answer(
                            question=prompt,
                            retrieved_chunks=retrieved,
                            conversation_history=st.session_state.messages[:-1]
                        )
                        
                        answer = res.get("answer", "")
                        sources = res.get("sources", [])
                        
                        st.markdown(answer)
                        if sources:
                            render_sources(sources)

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                st.rerun()

        # MODE 2: LEARN DOCUMENTATION (Phase 1 guide + browser)
        elif st.session_state.mode == "learn":
            st.markdown("### 📖 Documentation Learning Guide")
            
            parsed_start = urllib.parse.urlparse(st.session_state.url)
            domain = parsed_start.netloc + parsed_start.path
            final_content = storage.get_final_guide(domain)
            
            tab1, tab2 = st.tabs(["📑 Master Learning Guide", "🔍 Browse Original Pages"])
            
            with tab1:
                if final_content:
                    render_learning_material(final_content)
                else:
                    st.info("A comprehensive learning guide has not been generated for this site yet.")
                    if st.button("Generate Master Learning Guide", type="primary"):
                        gen_placeholder = st.empty()
                        do_generate_learning_guide(gen_placeholder)
                        
            with tab2:
                render_page_browser(metadata, storage.pages_dir)

if __name__ == "__main__":
    main()
