from scraper.cleaner import clean_html
from knowledge.parser import extract_title

def test_clean_html():
    html = """
    <html>
        <head><title>Test</title></head>
        <body>
            <nav>Should be removed</nav>
            <main>
                <h1>Main Title</h1>
                <p>Some paragraph text.</p>
                <aside>Sidebar to ignore</aside>
                <div class="sidebar">Also ignore</div>
                <pre><code>print('hello')</code></pre>
            </main>
            <footer>Footer ignore</footer>
        </body>
    </html>
    """
    
    cleaned = clean_html(html)
    
    assert "H1: Main Title" in cleaned
    assert "Some paragraph text." in cleaned
    assert "print('hello')" in cleaned
    assert "Should be removed" not in cleaned
    assert "Sidebar to ignore" not in cleaned # wait, <aside> is removed completely
    assert "Also ignore" not in cleaned # .sidebar is removed
    assert "Footer ignore" not in cleaned

def test_extract_title():
    # Extracted from H1
    cleaned = "\n\nH1: My Custom Title\n\nSome text."
    assert extract_title(cleaned, "https://example.com/docs/intro") == "My Custom Title"
    
    # Extracted from URL fallback
    cleaned_no_h1 = "Just some text."
    assert extract_title(cleaned_no_h1, "https://example.com/docs/getting-started") == "Getting Started"
