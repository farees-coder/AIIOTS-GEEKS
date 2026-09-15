import os
import shutil
import pytest
from knowledge.vector_store import VectorStore

TEST_CHROMA_DIR = "data/test_chroma_pytest"

@pytest.fixture(scope="module")
def vector_store():
    if os.path.exists(TEST_CHROMA_DIR):
        shutil.rmtree(TEST_CHROMA_DIR, ignore_errors=True)
    vs = VectorStore(persist_dir=TEST_CHROMA_DIR)
    yield vs
    del vs
    if os.path.exists(TEST_CHROMA_DIR):
        shutil.rmtree(TEST_CHROMA_DIR, ignore_errors=True)

def test_add_and_query_chunks(vector_store):
    doc_id = "test_playwright_doc"
    chunks = [
        {
            "chunk_id": "page_intro_c0",
            "content": "[Playwright Intro > Getting Started]\nPlaywright lets you automate browsers including Chromium, Firefox and WebKit.",
            "metadata": {
                "documentation_id": doc_id,
                "page_id": "page_intro",
                "page_title": "Playwright Intro",
                "section": "Getting Started",
                "url": "https://playwright.dev/docs/intro",
                "chunk_id": "page_intro_c0",
                "page_order": 0,
                "chunk_index": 0
            }
        },
        {
            "chunk_id": "page_locators_c0",
            "content": "[Playwright Locators > Locating Elements]\nLocators are the central piece of Playwright auto-waiting and retry-ability. Use page.locator() to find elements.",
            "metadata": {
                "documentation_id": doc_id,
                "page_id": "page_locators",
                "page_title": "Playwright Locators",
                "section": "Locating Elements",
                "url": "https://playwright.dev/docs/locators",
                "chunk_id": "page_locators_c0",
                "page_order": 1,
                "chunk_index": 0
            }
        }
    ]

    # Add chunks
    added = vector_store.add_chunks(doc_id, chunks)
    assert added == 2
    assert vector_store.is_indexed(doc_id) is True
    assert vector_store.count(doc_id) == 2

    # Query general
    results = vector_store.query(doc_id, "How do locators work in Playwright?", n_results=2)
    assert len(results) > 0
    assert "locator" in results[0]["content"].lower()

    # Query with page filter
    filtered_results = vector_store.query(
        doc_id,
        "How do I install or get started?",
        n_results=1,
        page_filter="page_intro"
    )
    assert len(filtered_results) > 0
    assert filtered_results[0]["metadata"]["page_id"] == "page_intro"

def test_isolated_collections(vector_store):
    doc1 = "doc_alpha"
    doc2 = "doc_beta"

    vector_store.add_chunks(doc1, [{
        "chunk_id": "c1",
        "content": "Alpha specific documentation content.",
        "metadata": {"documentation_id": doc1, "url": "https://alpha.com", "page_id": "p1", "page_title": "Alpha", "section": "S1"}
    }])

    vector_store.add_chunks(doc2, [{
        "chunk_id": "c2",
        "content": "Beta specific documentation content.",
        "metadata": {"documentation_id": doc2, "url": "https://beta.com", "page_id": "p2", "page_title": "Beta", "section": "S2"}
    }])

    assert vector_store.count(doc1) == 1
    assert vector_store.count(doc2) == 1

    # Alpha collection should not retrieve Beta content
    alpha_res = vector_store.query(doc1, "Beta", n_results=5)
    for r in alpha_res:
        assert r["metadata"]["documentation_id"] == doc1
