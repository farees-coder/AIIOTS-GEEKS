"""ChromaDB vector store with production features."""
import os

# Must be set before `sentence_transformers` (which pulls in `transformers`)
# is imported below — otherwise transformers may try to load its TensorFlow/
# Keras 3 integration and crash if TF happens to be installed. This app only
# uses the PyTorch backend, so TF support is unneeded.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from typing import List, Dict, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.logger import logger


class VectorStore:
    """Production-grade ChromaDB collection manager."""

    def __init__(self, collection_name: str = "pdf_knowledge"):
        """Initialize ChromaDB client and embedding model."""
        persist_dir = settings.chroma_persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Lazy-load embedding model
        self._embedding_model: SentenceTransformer | None = None
        logger.info("vector_store_initialized",
                    collection=collection_name,
                    persist_dir=persist_dir,
                    chunks=self.collection.count())

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model on first use."""
        if self._embedding_model is None:
            logger.info("loading_embedding_model", model=settings.embedding_model)
            self._embedding_model = SentenceTransformer(
                settings.embedding_model,
                device=settings.embedding_device
            )
        return self._embedding_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5)
    )
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True   # Required for cosine similarity
        )
        return embeddings.tolist()

    def add_chunks(self, chunks: List[Dict]) -> int:
        """
        Add chunks to the vector store, skipping duplicates.

        Args:
            chunks: List of dicts with 'id', 'text', 'metadata'

        Returns:
            Number of new chunks added
        """
        if not chunks:
            return 0

        # Check for existing IDs to avoid duplicates
        existing = self.collection.get(ids=[c["id"] for c in chunks])
        existing_ids = set(existing["ids"]) if existing["ids"] else set()
        new_chunks = [c for c in chunks if c["id"] not in existing_ids]

        if not new_chunks:
            logger.info("no_new_chunks", total=len(chunks))
            return 0

        # Batch embedding for memory efficiency
        batch_size = 64
        all_embeddings = []
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i:i + batch_size]
            texts = [c["text"] for c in batch]
            all_embeddings.extend(self.embed_texts(texts))
            logger.debug("embedded_batch", batch=i // batch_size + 1, size=len(batch))

        self.collection.add(
            ids=[c["id"] for c in new_chunks],
            documents=[c["text"] for c in new_chunks],
            embeddings=all_embeddings,
            metadatas=[c["metadata"] for c in new_chunks]
        )

        logger.info("chunks_indexed",
                    added=len(new_chunks),
                    skipped=len(chunks) - len(new_chunks))
        return len(new_chunks)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filter_dict: Optional[Dict] = None
    ) -> Dict:
        """
        Search for relevant chunks.

        Args:
            query: User question
            top_k: Number of results (defaults to settings.top_k_results)
            filter_dict: Optional ChromaDB metadata filter

        Returns:
            ChromaDB query results dict
        """
        k = top_k or settings.top_k_results
        # Cap top_k at actual collection size to avoid ChromaDB error
        total = self.collection.count()
        k = min(k, total) if total > 0 else k

        query_embedding = self.embed_texts([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter_dict,
            include=["documents", "metadatas", "distances"]
        )

        result_count = len(results.get("documents", [[]])[0])
        logger.info("search_complete", query=query[:50], results=result_count)
        return results

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        count = self.collection.count()
        if count == 0:
            return {"total_chunks": 0, "unique_files": 0, "total_pages": 0, "files": []}

        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        filenames: set = set()
        pages: set = set()
        for meta in all_meta:
            if meta:
                filenames.add(meta.get("filename", "unknown"))
                pages.add(f"{meta.get('filename', 'unknown')}:p{meta.get('page', 0)}")

        return {
            "total_chunks": count,
            "unique_files": len(filenames),
            "total_pages": len(pages),
            "files": sorted(list(filenames))
        }

    def delete_by_filename(self, filename: str) -> int:
        """Remove all chunks belonging to a specific file."""
        before = self.collection.count()
        self.collection.delete(where={"filename": filename})
        after = self.collection.count()
        deleted = before - after
        logger.info("deleted_file", filename=filename, chunks_deleted=deleted)
        return deleted

    def clear(self):
        """Delete all chunks from the collection."""
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.warning("collection_cleared")
