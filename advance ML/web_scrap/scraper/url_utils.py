import urllib.parse
from typing import Optional


def validate_url(url: str) -> bool:
    """Check if URL is valid (http or https)."""
    try:
        parsed = urllib.parse.urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def normalize_url(url: str) -> str:
    """Normalize URL by removing fragments and query parameters."""
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def get_domain(url: str) -> str:
    """Extract domain from URL."""
    parsed = urllib.parse.urlparse(url)
    return parsed.netloc


def get_base_path(url: str) -> str:
    """
    Extract the base documentation path.
    Example: https://playwright.dev/docs/intro -> /docs/
    If there is no specific doc path, returns /
    """
    parsed = urllib.parse.urlparse(url)
    path = parsed.path
    if not path or path == "/":
        return "/"
    
    # Try to find common docs path patterns
    parts = path.strip("/").split("/")
    if parts:
        # Just use the first part of the path as the scope
        return f"/{parts[0]}/"
    return "/"


def is_within_scope(url: str, start_url: str) -> bool:
    """
    Check if a URL is within the allowed domain and path scope of the starting URL.
    """
    if not validate_url(url):
        return False
        
    start_domain = get_domain(start_url)
    target_domain = get_domain(url)
    
    if start_domain != target_domain:
        return False
        
    start_path = urllib.parse.urlparse(start_url).path
    target_path = urllib.parse.urlparse(url).path
    
    # Very basic path scoping - can be improved
    # E.g., if start is /docs/intro, allow /docs/* but maybe not /api/* depending on requirements.
    # The requirement says:
    # Domain: playwright.dev
    # Documentation scope: /docs/*
    
    base_scope = get_base_path(start_url)
    if base_scope != "/" and not target_path.startswith(base_scope):
         return False

    return True

