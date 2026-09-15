import urllib.parse
from bs4 import BeautifulSoup
from typing import List, Set
from .url_utils import normalize_url, is_within_scope


def extract_links(html: str, base_url: str, start_url: str) -> List[str]:
    """
    Extract all valid links from HTML that fall within the scope.
    
    Args:
        html: The HTML content
        base_url: The URL of the page the HTML came from (for resolving relative links)
        start_url: The original starting URL (for scope checking)
        
    Returns:
        List of normalized, absolute URLs within scope.
    """
    soup = BeautifulSoup(html, "html.parser")
    links: Set[str] = set()
    
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        
        # Resolve relative URLs
        absolute_url = urllib.parse.urljoin(base_url, href)
        
        # Normalize (remove fragments/query params)
        normalized = normalize_url(absolute_url)
        
        # Check scope
        if is_within_scope(normalized, start_url):
            links.add(normalized)
            
    return list(links)
