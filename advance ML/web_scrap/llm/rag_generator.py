from typing import List, Dict, Any, Optional
import time
import re
from groq import RateLimitError
from .groq_client import get_groq_client, get_model_config
from .prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT

class RAGGenerator:
    """
    Handles RAG question answering using Groq with retrieved context chunks,
    source extraction, conversational history, and rate limit retries.
    """
    def __init__(self):
        self.client = get_groq_client()
        self.config = get_model_config()

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_context_chars: int = 12000
    ) -> Dict[str, Any]:
        """
        Generates a grounded answer using retrieved documentation chunks.
        Returns:
            {
                "answer": str,
                "sources": [{"title": str, "url": str}]
            }
        """
        if not retrieved_chunks:
            return {
                "answer": "I couldn't find enough information about this in the loaded documentation.",
                "sources": []
            }

        # 1. Format Context and Budget Tokens
        context_parts = []
        current_len = 0
        used_chunks = []

        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            title = meta.get("page_title", "Documentation")
            section = meta.get("section", "General")
            url = meta.get("url", "")
            content = chunk.get("content", "").strip()

            snippet = f"--- Source: {title} | Section: {section} ---\nURL: {url}\n{content}\n"
            snippet_len = len(snippet)

            if current_len + snippet_len > max_context_chars and context_parts:
                break

            context_parts.append(snippet)
            current_len += snippet_len
            used_chunks.append(chunk)

        context_str = "\n".join(context_parts)

        # 2. Format Conversation History (last 2-3 turns)
        history_block = ""
        if conversation_history:
            recent_turns = conversation_history[-4:]  # Last 2 user/assistant pairs
            formatted_turns = []
            for msg in recent_turns:
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "").strip()
                # Truncate any extremely long past assistant responses in history
                if len(content) > 500:
                    content = content[:500] + "..."
                formatted_turns.append(f"{role}: {content}")
                
            if formatted_turns:
                history_block = "RECENT CONVERSATION HISTORY:\n" + "\n".join(formatted_turns) + "\n\n"

        # 3. Assemble Prompt
        user_prompt = RAG_USER_PROMPT.format(
            context=context_str,
            history_block=history_block,
            question=question
        )

        # 4. Call Groq with Retry
        answer_text = self._call_groq(RAG_SYSTEM_PROMPT, user_prompt)
        if not answer_text:
            answer_text = "Sorry, I encountered an error communicating with the model. Please try again."

        # 5. Extract Unique Sources
        sources = []
        seen_urls = set()
        for chunk in used_chunks:
            meta = chunk.get("metadata", {})
            url = meta.get("url")
            title = meta.get("page_title") or meta.get("section") or "Documentation Page"
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "title": title,
                    "url": url,
                    "section": meta.get("section", "")
                })

        return {
            "answer": answer_text,
            "sources": sources
        }

    def _call_groq(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        max_retries = 10
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=self.config["model"],
                    temperature=self.config["temperature"],
                    max_tokens=self.config["max_tokens"],
                )
                return response.choices[0].message.content
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = 10
                    match = re.search(r"try again in ([0-9.]+)s", str(e))
                    if match:
                        wait_time = float(match.group(1)) + 2.0
                    else:
                        wait_time = 15 * (2 ** attempt)
                    print(f"Rate limit hit. Retrying in {wait_time:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print("Rate limit hit. Max retries reached.")
                    raise e
            except Exception as e:
                print(f"Error calling Groq API: {e}")
                # If model not found or invalid, provide a clear error message
                if "model" in str(e).lower():
                    raise e
                return f"Error from Groq API: {str(e)}"
        return None
