"""Utility functions."""
import os
import shutil
from pathlib import Path
from typing import List

from src.config import settings


class FileUtils:
    """File handling utilities."""

    @staticmethod
    def ensure_upload_dir() -> Path:
        """Create upload directory if it doesn't exist."""
        path = Path(settings.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def save_uploaded_file(uploaded_file, filename: str | None = None) -> Path:
        """
        Save an uploaded file to the upload directory.

        Args:
            uploaded_file: Streamlit UploadedFile object
            filename: Optional override filename

        Returns:
            Path to saved file
        """
        upload_dir = FileUtils.ensure_upload_dir()
        name = filename or getattr(uploaded_file, "name", "uploaded.pdf")
        file_path = upload_dir / name

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return file_path

    @staticmethod
    def get_uploaded_pdfs() -> List[str]:
        """Get list of already uploaded PDF files."""
        upload_dir = FileUtils.ensure_upload_dir()
        return sorted([
            f.name for f in upload_dir.iterdir()
            if f.suffix.lower() == ".pdf"
        ])

    @staticmethod
    def delete_file(filename: str) -> bool:
        """Delete an uploaded file. Returns True if deleted."""
        file_path = Path(settings.upload_dir) / filename
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable form."""
        if size_bytes == 0:
            return "0 B"
        for unit in ["B", "KB", "MB", "GB"]:
            if abs(size_bytes) < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024  # type: ignore[assignment]
        return f"{size_bytes:.1f} TB"

    @staticmethod
    def clear_upload_dir():
        """Remove all uploaded files."""
        upload_dir = Path(settings.upload_dir)
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
