import pytest
from scraper.url_utils import validate_url, normalize_url, is_within_scope
from scraper.link_extractor import extract_links

def test_validate_url():
    assert validate_url("https://example.com/docs") == True
    assert validate_url("http://example.com") == True
    assert validate_url("ftp://example.com") == False
    assert validate_url("not_a_url") == False

def test_normalize_url():
    assert normalize_url("https://example.com/docs#section") == "https://example.com/docs"
    assert normalize_url("https://example.com/docs?query=1") == "https://example.com/docs"

def test_is_within_scope():
    start = "https://playwright.dev/docs/intro"
    
    # Same domain, within docs
    assert is_within_scope("https://playwright.dev/docs/locators", start) == True
    
    # Same domain, outside docs
    assert is_within_scope("https://playwright.dev/community", start) == False
    
    # Different domain
    assert is_within_scope("https://google.com/docs/intro", start) == False

def test_extract_links():
    html = """
    <html>
        <body>
            <a href="/docs/locators">Locators</a>
            <a href="https://playwright.dev/docs/actions">Actions</a>
            <a href="https://external.com">External</a>
            <a href="#id">Anchor</a>
        </body>
    </html>
    """
    
    start = "https://playwright.dev/docs/intro"
    base = "https://playwright.dev/docs/intro"
    
    links = extract_links(html, base, start)
    
    assert "https://playwright.dev/docs/locators" in links
    assert "https://playwright.dev/docs/actions" in links
    assert "https://external.com" not in links
    assert "https://playwright.dev/docs/intro" in links # Anchor resolved to base
