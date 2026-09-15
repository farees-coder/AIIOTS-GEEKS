import os
import json
import hashlib
from typing import Dict, List, Optional
from datetime import datetime

class KnowledgeStorage:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        
        # Documentation directories
        self.docs_dir = os.path.join(base_dir, "documentation")
        self.pages_dir = os.path.join(self.docs_dir, "pages")
        
        # Summaries directories
        self.summaries_dir = os.path.join(base_dir, "summaries")
        self.sections_dir = os.path.join(self.summaries_dir, "sections")
        self.final_dir = os.path.join(self.summaries_dir, "final")
        
        # Cache
        self.cache_dir = os.path.join(base_dir, "cache")
        
        # Ensure directories exist
        for directory in [self.pages_dir, self.sections_dir, self.final_dir, self.cache_dir]:
            os.makedirs(directory, exist_ok=True)
            
    def _get_hash(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()
        
    def get_doc_id(self, url: str) -> str:
        """Generates an isolated, safe documentation ID for ChromaDB collections."""
        import urllib.parse
        import re
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.replace(".", "_").replace("-", "_")
        path = parsed.path.strip("/").replace("/", "_").replace("-", "_")
        doc_id = f"{domain}_{path}" if path else domain
        doc_id = re.sub(r"[^a-zA-Z0-9_]", "", doc_id).strip("_")
        if len(doc_id) < 3:
            doc_id = f"doc_{doc_id}"
        if len(doc_id) > 60:
            doc_id = doc_id[:50] + "_" + self._get_hash(url)[:8]
        return doc_id.lower()

    def has_documentation(self) -> bool:
        """Checks if valid documentation has already been crawled and stored."""
        meta = self.get_metadata()
        return meta is not None and len(meta.get("pages", [])) > 0

    def _get_filename(self, url: str) -> str:
        # Convert URL to a safe filename
        safe_name = url.replace("https://", "").replace("http://", "").replace("/", "_").replace("?", "_").replace("=", "_")
        if len(safe_name) > 100:
            safe_name = safe_name[:80] + "_" + self._get_hash(url)[:10]
        return f"{safe_name}.md"

    def save_page(self, url: str, content: str, title: str = "") -> str:
        """Saves a single cleaned page as a markdown file."""
        filename = self._get_filename(url)
        file_path = os.path.join(self.pages_dir, filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# {title or url}\n\n")
            f.write(f"URL: {url}\n\n")
            f.write(content)
            
        return file_path
        
    def save_metadata(self, source_url: str, pages_info: List[Dict]):
        """Saves the metadata.json for the crawl."""
        file_path = os.path.join(self.docs_dir, "metadata.json")
        data = {
            "source_url": source_url,
            "crawl_date": datetime.now().isoformat(),
            "total_pages": len(pages_info),
            "pages": pages_info
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_metadata(self) -> Optional[Dict]:
        """Gets the metadata.json if it exists."""
        file_path = os.path.join(self.docs_dir, "metadata.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def build_master_documentation(self, source_url: str, pages_info: List[Dict]) -> str:
        """Combines all pages into a single documentation.md file."""
        master_file = os.path.join(self.docs_dir, "documentation.md")
        
        with open(master_file, "w", encoding="utf-8") as out_f:
            out_f.write("# Documentation\n\n")
            out_f.write(f"Source: {source_url}\n\n")
            
            for idx, page in enumerate(pages_info, 1):
                if page.get("status") == "success":
                    filename = self._get_filename(page["url"])
                    file_path = os.path.join(self.pages_dir, filename)
                    if os.path.exists(file_path):
                        out_f.write(f"---\n\n## Page {idx} - {page.get('title', page['url'])}\n\n")
                        with open(file_path, "r", encoding="utf-8") as in_f:
                            # Skip the first 4 lines containing the title and URL header we added in save_page
                            lines = in_f.readlines()[4:]
                            out_f.writelines(lines)
                        out_f.write("\n\n")
                        
        return master_file

    def get_master_documentation(self) -> Optional[str]:
        """Returns the content of the master documentation file if it exists."""
        master_file = os.path.join(self.docs_dir, "documentation.md")
        if os.path.exists(master_file):
            with open(master_file, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def save_section_summary(self, section_name: str, summary: str) -> str:
        safe_name = section_name.replace("/", "_").replace(" ", "_")
        file_path = os.path.join(self.sections_dir, f"{safe_name}.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(summary)
        return file_path
        
    def get_section_summary(self, section_name: str) -> Optional[str]:
        safe_name = section_name.replace("/", "_").replace(" ", "_")
        file_path = os.path.join(self.sections_dir, f"{safe_name}.md")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        return None
        
    def save_final_guide(self, domain: str, content: str) -> str:
        safe_name = domain.replace("/", "_").replace(":", "_")
        file_path = os.path.join(self.final_dir, f"{safe_name}_guide.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return file_path
        
    def get_final_guide(self, domain: str) -> Optional[str]:
        safe_name = domain.replace("/", "_").replace(":", "_")
        file_path = os.path.join(self.final_dir, f"{safe_name}_guide.md")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def clear_all(self):
        """Clears all stored data to allow a fresh crawl."""
        import shutil
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
        self.__init__(self.base_dir)
