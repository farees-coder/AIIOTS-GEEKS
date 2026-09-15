from knowledge.structure import build_structure

def test_build_structure():
    urls = [
        "https://example.com/docs/intro",
        "https://example.com/docs/guides/locators",
        "https://example.com/docs/guides/actions"
    ]
    
    tree = build_structure(urls)
    
    assert "docs" in tree
    assert "intro" in tree["docs"]
    assert "https://example.com/docs/intro" in tree["docs"]["intro"]["_pages"]
    assert "guides" in tree["docs"]
    assert "locators" in tree["docs"]["guides"]
    assert "https://example.com/docs/guides/locators" in tree["docs"]["guides"]["locators"]["_pages"]
    assert "https://example.com/docs/guides/actions" in tree["docs"]["guides"]["actions"]["_pages"]
