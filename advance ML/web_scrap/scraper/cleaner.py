from bs4 import BeautifulSoup, Tag
from typing import Optional

def clean_html(html: str) -> str:
    """
    Cleans HTML to keep only structured documentation content.
    Removes sidebars, navigation, footers, scripts, styles, etc.
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove unwanted tags completely
    for tag in soup(["script", "style", "nav", "footer", "aside", "header", "noscript", "iframe"]):
        tag.decompose()
        
    # Attempt to find the main content area
    main_content: Optional[Tag] = None
    
    # Common main content selectors
    selectors = [
        "main", 
        "article", 
        ".main-content", 
        "#main-content",
        ".content",
        ".documentation",
        ".doc-content"
    ]
    
    for selector in selectors:
        element = soup.select_one(selector)
        if element:
            main_content = element
            break
            
    if not main_content:
        # Fallback to body if no main semantic tag is found
        main_content = soup.body
        
    if not main_content:
        return ""
        
    # Further clean the main content by removing likely unrelated elements by class/id
    unwanted_classes = ["sidebar", "toc", "table-of-contents", "cookie", "banner", "ads", "search"]
    for element in main_content.find_all(True):
        if getattr(element, 'attrs', None) is None:
            continue
        if element.get("class"):
            classes = " ".join(element.get("class")).lower()
            if any(unwanted in classes for unwanted in unwanted_classes):
                element.decompose()
                
    # Extract structural elements we care about
    # Headings, paragraphs, lists, code blocks, tables
    
    cleaned_parts = []
    
    # We want to preserve the basic structure
    for element in main_content.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "pre", "code", "table"]):
        # Avoid extracting elements that are inside other extracted elements (e.g., code inside pre, li inside ul)
        if element.find_parent(["pre", "ul", "ol", "table"]) and element.name not in ["pre", "ul", "ol", "table"]:
            continue
            
        text = element.get_text(separator=" ", strip=True)
        if not text:
            continue
            
        if element.name.startswith("h"):
            cleaned_parts.append(f"\n\n{element.name.upper()}: {text}\n")
        elif element.name in ["ul", "ol"]:
            items = []
            for li in element.find_all("li", recursive=False):
                li_text = li.get_text(separator=" ", strip=True)
                items.append(f"- {li_text}")
            if items:
                cleaned_parts.append("\n" + "\n".join(items) + "\n")
        elif element.name == "pre":
            cleaned_parts.append(f"\nCODE BLOCK:\n```\n{text}\n```\n")
        elif element.name == "p":
            cleaned_parts.append(f"{text}\n")
        elif element.name == "table":
            cleaned_parts.append(f"\nTABLE DATA:\n{text}\n")
            
    return "".join(cleaned_parts).strip()
