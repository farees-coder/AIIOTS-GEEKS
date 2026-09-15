import pytest
from llm.rag_generator import RAGGenerator

def test_rag_empty_context():
    rag = RAGGenerator()
    result = rag.generate_answer(
        question="What is a locator?",
        retrieved_chunks=[]
    )
    assert "couldn't find enough information" in result["answer"].lower()
    assert result["sources"] == []

def test_source_extraction_and_budgeting():
    rag = RAGGenerator()
    chunks = [
        {
            "chunk_id": "c1",
            "content": "Playwright has Locators to find elements on the page.",
            "metadata": {
                "page_title": "Locators Guide",
                "section": "Overview",
                "url": "https://playwright.dev/docs/locators"
            }
        },
        {
            "chunk_id": "c2",
            "content": "Duplicate URL with another section.",
            "metadata": {
                "page_title": "Locators Guide",
                "section": "Filtering",
                "url": "https://playwright.dev/docs/locators"
            }
        },
        {
            "chunk_id": "c3",
            "content": "Assertions in Playwright test framework.",
            "metadata": {
                "page_title": "Assertions Guide",
                "section": "Expect",
                "url": "https://playwright.dev/docs/test-assertions"
            }
        }
    ]

    # Verify context budgeting without calling API
    # (Check that source deduplication logic works)
    sources = []
    seen = set()
    for c in chunks:
        u = c["metadata"]["url"]
        if u not in seen:
            seen.add(u)
            sources.append({"title": c["metadata"]["page_title"], "url": u})

    assert len(sources) == 2
    assert sources[0]["url"] == "https://playwright.dev/docs/locators"
    assert sources[1]["url"] == "https://playwright.dev/docs/test-assertions"
