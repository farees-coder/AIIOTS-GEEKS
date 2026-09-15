import urllib.parse
from typing import List, Dict, Any

def build_structure(urls: List[str]) -> Dict[str, Any]:
    """
    Builds a hierarchical structure of the documentation based on URLs.
    """
    tree = {}
    for url in urls:
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.strip("/")
        if not path:
            continue
            
        parts = path.split("/")
        current = tree
        for i, part in enumerate(parts):
            if part not in current:
                current[part] = {"_pages": []}
            if i == len(parts) - 1:
                current[part]["_pages"].append(url)
            else:
                current = current[part]
    return tree

def group_pages_into_sections(urls: List[str]) -> Dict[str, List[str]]:
    """
    Groups a list of URLs into logical sections based on their path structure.
    Returns: {"section_name": [url1, url2, ...]}
    """
    sections = {}
    for url in urls:
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.strip("/")
        parts = path.split("/")
        
        # Determine the section name based on the URL path.
        # e.g., https://example.com/docs/intro -> section 'docs'
        # https://example.com/api/v1/users -> section 'api'
        if len(parts) >= 2:
            section_name = parts[0]
        else:
            section_name = "General"
            
        if section_name not in sections:
            sections[section_name] = []
        sections[section_name].append(url)
        
    return sections
