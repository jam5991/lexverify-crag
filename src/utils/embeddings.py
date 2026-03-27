"""
Embedding Model Utilities.

Factory wrapper for OpenAI embedding models used in the retrieval pipeline.
"""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from src.config import Settings, get_settings


def get_embedding_model(settings: Settings | None = None) -> OpenAIEmbeddings:
    """Create and return the configured embedding model.

    Args:
        settings: Optional settings override.

    Returns:
        OpenAIEmbeddings instance configured with the specified model.
    """
    settings = settings or get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
