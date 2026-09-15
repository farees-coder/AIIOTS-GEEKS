"""Application configuration — Groq only."""
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Storage (always absolute) ───────────────────────────
    @property
    def chroma_persist_dir(self) -> str:
        return str(BASE_DIR / "chroma_db")

    @property
    def upload_dir(self) -> str:
        return str(BASE_DIR / "uploaded_pdfs")

    # ── Embeddings ──────────────────────────────────────────
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_device: str = Field(default="cpu")

    # ── Chunking ────────────────────────────────────────────
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=100, ge=0, le=500)

    # ── Retrieval ───────────────────────────────────────────
    top_k_results: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=2.0)

    # ── Groq ────────────────────────────────────────────────
    groq_api_key: str | None = Field(default=None)
    groq_model: str = Field(default="groq/compound")
    groq_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    groq_max_tokens: int = Field(default=2048, ge=1, le=8192)

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key)


settings = Settings()
