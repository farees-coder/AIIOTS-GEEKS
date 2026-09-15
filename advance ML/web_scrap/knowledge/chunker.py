import re
from typing import List, Dict, Any, Optional

class SectionChunker:
    """
    Section-aware chunker for cleaned documentation pages.
    Preserves document title, section headings, URLs, code blocks, and ordering.
    """
    def __init__(
        self,
        target_chunk_size: int = 800,
        chunk_overlap: int = 100,
        max_chunk_size: int = 1200
    ):
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunk_size = max_chunk_size

    def chunk_page(
        self,
        content: str,
        documentation_id: str,
        page_id: str,
        url: str,
        page_title: str,
        page_order: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Chunks a single cleaned page into section-aware chunks with metadata.
        """
        if not content or not content.strip():
            return []

        # Parse title and URL from header if present
        lines = content.splitlines()
        body_lines = []
        in_header = True
        extracted_title = page_title
        extracted_url = url

        for line in lines:
            if in_header and line.startswith("# ") and not extracted_title:
                extracted_title = line[2:].strip()
                continue
            if in_header and line.startswith("URL: ") and not extracted_url:
                extracted_url = line[5:].strip()
                continue
            if in_header and (line.startswith("H1:") or line.startswith("H2:") or line.strip() == "---"):
                in_header = False
            elif in_header and line.strip() == "":
                continue
            else:
                in_header = False
                
            body_lines.append(line)

        body_text = "\n".join(body_lines).strip()
        if not body_text:
            body_text = content.strip()

        sections = self._split_into_sections(body_text, default_title=extracted_title or "Overview")
        
        chunks = []
        chunk_index = 0

        for section_title, section_text in sections:
            if not section_text.strip():
                continue
                
            # If section fits comfortably within max_chunk_size, keep it as a single chunk
            if len(section_text) <= self.max_chunk_size:
                formatted_content = f"[{extracted_title} > {section_title}]\n{section_text.strip()}"
                chunks.append({
                    "chunk_id": f"{page_id}_c{chunk_index}",
                    "content": formatted_content,
                    "metadata": {
                        "documentation_id": str(documentation_id),
                        "page_id": str(page_id),
                        "page_title": str(extracted_title or "Documentation"),
                        "section": str(section_title),
                        "url": str(extracted_url or ""),
                        "chunk_id": f"{page_id}_c{chunk_index}",
                        "page_order": int(page_order),
                        "chunk_index": int(chunk_index)
                    }
                })
                chunk_index += 1
            else:
                # Sub-chunk the section keeping code blocks intact
                sub_chunks = self._split_large_section(section_text)
                for sub in sub_chunks:
                    formatted_content = f"[{extracted_title} > {section_title}]\n{sub.strip()}"
                    chunks.append({
                        "chunk_id": f"{page_id}_c{chunk_index}",
                        "content": formatted_content,
                        "metadata": {
                            "documentation_id": str(documentation_id),
                            "page_id": str(page_id),
                            "page_title": str(extracted_title or "Documentation"),
                            "section": str(section_title),
                            "url": str(extracted_url or ""),
                            "chunk_id": f"{page_id}_c{chunk_index}",
                            "page_order": int(page_order),
                            "chunk_index": int(chunk_index)
                        }
                    })
                    chunk_index += 1

        return chunks

    def _split_into_sections(self, text: str, default_title: str) -> List[tuple]:
        """
        Splits text by headings (H1: ... H6: or # ... ######) into (section_title, text) pairs.
        """
        heading_pattern = re.compile(r'(?:\n|^)(?:H[1-6]:\s*(.+?)|(#{1,6})\s*(.+?))(?=\n|$)', re.IGNORECASE)
        
        matches = list(heading_pattern.finditer(text))
        if not matches:
            return [(default_title, text)]

        sections = []
        first_start = matches[0].start()
        if first_start > 0 and text[:first_start].strip():
            sections.append((default_title, text[:first_start].strip()))

        for i, match in enumerate(matches):
            h_text = match.group(1) or match.group(3)
            sec_title = h_text.strip() if h_text else default_title
            sec_title = re.sub(r'\s*​.*$', '', sec_title).strip()

            start_idx = match.end()
            end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sec_content = text[start_idx:end_idx].strip()
            if sec_content:
                sections.append((sec_title, sec_content))

        return sections

    def _split_large_section(self, text: str) -> List[str]:
        """
        Splits a large section into overlapping chunks without breaking code blocks.
        """
        blocks = []
        code_pattern = re.compile(r'(```[\s\S]*?```|CODE BLOCK:\s*```[\s\S]*?```)', re.DOTALL)
        
        last_idx = 0
        for match in code_pattern.finditer(text):
            pre_code = text[last_idx:match.start()].strip()
            if pre_code:
                blocks.append(pre_code)
            blocks.append(match.group(0).strip())
            last_idx = match.end()
            
        remaining = text[last_idx:].strip()
        if remaining:
            blocks.append(remaining)

        chunks = []
        current_chunk = []
        current_length = 0

        for block in blocks:
            block_len = len(block)
            if current_length + block_len > self.max_chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                if len(current_chunk[-1]) <= self.chunk_overlap * 2 and not current_chunk[-1].startswith("```"):
                    current_chunk = [current_chunk[-1], block]
                    current_length = len(current_chunk[0]) + block_len
                else:
                    current_chunk = [block]
                    current_length = block_len
            else:
                current_chunk.append(block)
                current_length += block_len

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks
