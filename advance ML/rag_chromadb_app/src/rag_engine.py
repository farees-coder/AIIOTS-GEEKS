"""RAG pipeline — Groq only."""
import re
from dataclasses import dataclass
from typing import Dict, List

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import settings
from src.logger import logger
from src.vector_store import VectorStore

# ── Verified Groq model (single choice, Groq compound) ────────
GROQ_MODEL = "groq/compound"


def _strip_think_tags(text: str) -> str:
    """Remove <think>…</think> blocks that some models emit."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"\*\*Answer\*\*\s*\n(.*)", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)
    return cleaned.strip()


@dataclass
class RAGAnswer:
    """Structured RAG response."""
    answer: str
    sources: List[Dict]
    chunks_used: int
    confidence: float
    model: str


class LLMError(Exception):
    pass


class RAGEngine:
    """Production RAG engine — Groq backend."""

    SYSTEM_PROMPT = (
        "You are a precise document Q&A assistant. "
        "Answer the user's question using ONLY the provided document excerpts.\n\n"
        "Rules:\n"
        "- Use ONLY information from the excerpts.\n"
        "- If the answer is not in the excerpts, say: "
        "\"I cannot find the answer in the provided documents.\"\n"
        "- Cite excerpt numbers like [1], [2] when referencing information.\n"
        "- Be concise but complete.\n"
        "- Do NOT make up information.\n"
        "- Respond directly — no preamble, no reasoning blocks."
    )

    def __init__(self, vector_store: VectorStore):
        if not settings.groq_api_key:
            raise LLMError(
                "GROQ_API_KEY missing in .env — "
                "get a free key at https://console.groq.com/keys"
            )
        self.vector_store = vector_store
        self.model = GROQ_MODEL
        self._client = OpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        logger.info("rag_engine_ready", model=self.model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_groq(self, user_prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=settings.groq_temperature,
                max_tokens=settings.groq_max_tokens,
            )
            raw = response.choices[0].message.content or ""
            return _strip_think_tags(raw)
        except Exception as e:
            msg = str(e)
            if "model_not_found" in msg or "404" in msg:
                raise LLMError(f"Model '{self.model}' not found on your Groq account.")
            raise LLMError(f"Groq API error: {msg}") from e

    def _build_context(self, chunks: List[str], metadatas: List[Dict]) -> str:
        parts = []
        for i, (doc, meta) in enumerate(zip(chunks, metadatas), 1):
            fn   = meta.get("filename", "Unknown")
            page = meta.get("page", "?")
            parts.append(f"[Excerpt {i}] From '{fn}' (Page {page}):\n{doc}")
        return "\n\n".join(parts)

    def answer(self, query: str, top_k: int | None = None, filename: str | None = None) -> RAGAnswer:
        """
        Full RAG pipeline: retrieve → generate → structure.

        Args:
            query: User question
            top_k: Number of chunks to retrieve
            filename: If set, restrict retrieval to chunks from this file only
        """

        # 1. Retrieve
        filter_dict = {"filename": filename} if filename else None
        results   = self.vector_store.search(query, top_k=top_k, filter_dict=filter_dict)
        docs_list = results.get("documents", [[]])[0]

        if not docs_list:
            scope = f" in '{filename}'" if filename else " in the uploaded documents"
            return RAGAnswer(
                answer=f"No relevant information found{scope}.",
                sources=[],
                chunks_used=0,
                confidence=0.0,
                model=self.model,
            )

        chunks    = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # 2. Confidence score
        confidence = round(max(0.0, 1.0 - sum(distances) / len(distances)), 3)

        # 3. Filter by threshold
        threshold = settings.similarity_threshold * 2
        filtered  = [(c, m, d) for c, m, d in zip(chunks, metadatas, distances) if d <= threshold]
        if not filtered:
            filtered = list(zip(chunks[:3], metadatas[:3], distances[:3]))

        f_chunks, f_meta, f_dist = zip(*filtered)

        # 4. Generate
        context     = self._build_context(list(f_chunks), list(f_meta))
        user_prompt = f"DOCUMENT EXCERPTS:\n{context}\n\nUSER QUESTION: {query}\n\nYour answer:"

        try:
            answer_text = self._call_groq(user_prompt)
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"Unexpected error: {e}") from e

        # 5. Deduplicated sources
        sources: List[Dict] = []
        seen: set = set()
        for chunk_text, meta, dist in zip(f_chunks, f_meta, f_dist):
            key = f"{meta.get('filename')}::p{meta.get('page')}"
            if key not in seen:
                seen.add(key)
                preview = chunk_text[:250].strip()
                if len(chunk_text) > 250:
                    preview += "…"
                sources.append({
                    "display":    f"{meta.get('filename', 'Unknown')} — Page {meta.get('page', '?')}",
                    "preview":    preview,
                    "similarity": round(1.0 - dist, 3),
                })

        logger.info("answer_generated", query=query[:60], chunks=len(f_chunks), confidence=confidence)

        return RAGAnswer(
            answer=answer_text,
            sources=sources,
            chunks_used=len(f_chunks),
            confidence=confidence,
            model=self.model,
        )
