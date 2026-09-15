from typing import Optional, List
from .groq_client import get_groq_client, get_model_config
from .prompts import FINAL_SUMMARY_PROMPT
import time
from groq import RateLimitError

class FinalGenerator:
    def __init__(self):
        self.client = get_groq_client()
        self.config = get_model_config()

    def _call_groq(self, prompt: str) -> Optional[str]:
        max_retries = 10
        import re
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    model=self.config["model"],
                    temperature=self.config["temperature"],
                    max_tokens=self.config["max_tokens"],
                )
                return response.choices[0].message.content
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = 10  # fallback
                    match = re.search(r"try again in ([0-9.]+)s", str(e))
                    if match:
                        wait_time = float(match.group(1)) + 2.0
                    else:
                        wait_time = 15 * (2 ** attempt)
                    print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print("Rate limit hit. Max retries reached.")
                    raise e
            except Exception as e:
                print(f"Error calling Groq API: {e}")
                raise e
        return None

    def generate_final_guide(self, summaries: List[str]) -> Optional[str]:
        """
        Combines multiple section summaries into a single, unified learning guide.
        """
        combined_text = "\n\n---\n\n".join(summaries)
        max_chars = 10000
        if len(combined_text) > max_chars:
            combined_text = combined_text[:max_chars] + "\n\n...[Truncated due to length]..."
            
        prompt = FINAL_SUMMARY_PROMPT.format(summaries=combined_text)
        return self._call_groq(prompt)
