import os
import re
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions

class VectorStore:
    """
    Manages persistent ChromaDB vector storage, local embeddings (all-MiniLM-L6-v2),
    isolated collections per documentation_id, and metadata-aware retrieval.
    """
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "data/chromadb")
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.device = device or os.getenv("EMBEDDING_DEVICE", "cpu")

        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Use sentence-transformers or Chroma's built-in ONNX all-MiniLM-L6-v2 (CPU)
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.model_name,
                device=self.device
            )
        except Exception:
            # High-performance local ONNX all-MiniLM-L6-v2 on CPU
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    @staticmethod
    def sanitize_collection_name(name: str) -> str:
        """
        Ensures collection name satisfies ChromaDB constraints:
        - 3 to 63 characters
        - alphanumeric, underscores, hyphens
        - starts and ends with alphanumeric
        """
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        sanitized = sanitized.strip('_-')
        if len(sanitized) < 3:
            sanitized = f"doc_{sanitized}"
        if len(sanitized) > 63:
            sanitized = sanitized[:63].rstrip('_-')
        return sanitized.lower()

    def get_collection(self, doc_id: str):
        collection_name = self.sanitize_collection_name(doc_id)
        try:
            return self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_fn
            )
        except ValueError as ve:
            if "conflict" in str(ve).lower() or "embedding function" in str(ve).lower():
                # Re-align with current embedding function
                self.client.delete_collection(name=collection_name)
                return self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_fn,
                    metadata={"hnsw:space": "cosine", "documentation_id": doc_id}
                )
            raise ve
        except Exception:
            return self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine", "documentation_id": doc_id}
            )

    def is_indexed(self, doc_id: str) -> bool:
        """Checks if the documentation has already been embedded and indexed."""
        try:
            collection = self.get_collection(doc_id)
            return collection.count() > 0
        except Exception:
            return False

    def count(self, doc_id: str) -> int:
        try:
            collection = self.get_collection(doc_id)
            return collection.count()
        except Exception:
            return 0

    def add_chunks(self, doc_id: str, chunks: List[Dict[str, Any]], batch_size: int = 150) -> int:
        """
        Batches and upserts section chunks into the isolated collection.
        """
        if not chunks:
            return 0

        collection = self.get_collection(doc_id)

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        total = len(ids)
        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            collection.upsert(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )

        return total

    def query(
        self,
        doc_id: str,
        query_text: str,
        n_results: int = 5,
        page_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top-k relevant chunks for a question.
        Supports page-specific filtering with fallback to general collection.
        """
        collection = self.get_collection(doc_id)
        total_count = collection.count()
        if total_count == 0:
            return []

        actual_k = min(n_results, total_count)
        retrieved_chunks = []
        seen_ids = set()

        # 1. If page_filter is provided, prioritize chunks from that specific page
        if page_filter and page_filter != "all":
            try:
                # Filter by page_id or url
                where_clause = {
                    "$or": [
                        {"page_id": page_filter},
                        {"url": page_filter}
                    ]
                }
                res = collection.query(
                    query_texts=[query_text],
                    n_results=actual_k,
                    where=where_clause
                )
                if res and res.get("ids") and res["ids"][0]:
                    for i in range(len(res["ids"][0])):
                        cid = res["ids"][0][i]
                        seen_ids.add(cid)
                        retrieved_chunks.append({
                            "chunk_id": cid,
                            "content": res["documents"][0][i],
                            "metadata": res["metadatas"][0][i],
                            "distance": res["distances"][0][i] if "distances" in res and res["distances"] else None
                        })
            except Exception as e:
                print(f"Error querying with page_filter '{page_filter}': {e}")

        # 2. Fill remaining slots with top semantic results from the entire collection
        remaining_k = n_results - len(retrieved_chunks)
        if remaining_k > 0:
            try:
                res_general = collection.query(
                    query_texts=[query_text],
                    n_results=min(n_results + len(seen_ids), total_count)
                )
                if res_general and res_general.get("ids") and res_general["ids"][0]:
                    for i in range(len(res_general["ids"][0])):
                        cid = res_general["ids"][0][i]
                        if cid not in seen_ids:
                            seen_ids.add(cid)
                            retrieved_chunks.append({
                                "chunk_id": cid,
                                "content": res_general["documents"][0][i],
                                "metadata": res_general["metadatas"][0][i],
                                "distance": res_general["distances"][0][i] if "distances" in res_general and res_general["distances"] else None
                            })
                            if len(retrieved_chunks) >= n_results:
                                break
            except Exception as e:
                print(f"Error querying general collection: {e}")

        return retrieved_chunks

    def delete_collection(self, doc_id: str):
        collection_name = self.sanitize_collection_name(doc_id)
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass
