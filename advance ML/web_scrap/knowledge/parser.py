import re

def extract_title(cleaned_content: str, url: str) -> str:
    """
    Extracts a title for the document.
    First tries to find an H1 tag.
    Fallback: Extract from the last part of the URL path.
    """
    # Look for H1 in cleaned content
    # Format is usually \n\nH1: Title\n
    h1_match = re.search(r"H1:\s*(.+)", cleaned_content)
    if h1_match:
        return h1_match.group(1).strip()
        
    # Fallback to URL path
    from urllib.parse import urlparse
    path = urlparse(url).path.strip("/")
    if path:
        parts = path.split("/")
        # Convert slug to title case (e.g., "getting-started" -> "Getting Started")
        title = parts[-1].replace("-", " ").title()
        return title
        
    return "Home"
