import pytest
from knowledge.chunker import SectionChunker

def test_chunker_basic():
    chunker = SectionChunker(target_chunk_size=300, max_chunk_size=500)
    content = """# Playwright Intro

URL: https://playwright.dev/docs/intro

H2: Getting Started
Playwright is a framework for Web Testing and Automation.
It supports Chromium, Firefox, and WebKit.

H2: Installation
To install Playwright, run the following command:

CODE BLOCK:
```
npm init playwright@latest
```

Ensure Node.js 18+ is installed before proceeding.
"""
    chunks = chunker.chunk_page(
        content=content,
        documentation_id="playwright_dev",
        page_id="playwright_dev_docs_intro",
        url="https://playwright.dev/docs/intro",
        page_title="Playwright Intro",
        page_order=0
    )

    assert len(chunks) >= 2
    # Verify metadata
    for c in chunks:
        assert c["metadata"]["documentation_id"] == "playwright_dev"
        assert c["metadata"]["url"] == "https://playwright.dev/docs/intro"
        assert "chunk_id" in c
        assert "content" in c

    # Check that code block is intact in installation chunk
    install_chunk = next(c for c in chunks if "Installation" in c["metadata"]["section"])
    assert "npm init playwright@latest" in install_chunk["content"]

def test_chunker_empty():
    chunker = SectionChunker()
    chunks = chunker.chunk_page(
        content="",
        documentation_id="test",
        page_id="p1",
        url="https://test.com",
        page_title="Test"
    )
    assert chunks == []
