import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def get_groq_client() -> Groq:
    """Initializes and returns the Groq client based on environment variables."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment.")
        
    return Groq(api_key=api_key)

def get_model_config():
    """Returns the model configuration from environment variables."""
    return {
        "model": os.getenv("GROQ_MODEL", "groq/compound"),
        "temperature": float(os.getenv("GROQ_TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("GROQ_MAX_TOKENS", "2048"))
    }
