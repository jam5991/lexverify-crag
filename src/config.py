"""
LexVerify configuration module.

Loads settings from environment variables (via .env file) and provides
tunable hyperparameters for the CRAG pipeline.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # ── API Keys ──
    openai_api_key: str = Field(default="", description="OpenAI API key")
    tavily_api_key: str = Field(default="", description="Tavily API key for web search")
    pinecone_api_key: str = Field(default="", description="Pinecone API key")

    # ── Pinecone ──
    pinecone_index_name: str = Field(default="lexverify-legal", description="Pinecone index name")

    # ── Models ──
    embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model"
    )
    embedding_dimensions: int = Field(
        default=1024, description="Embedding vector dimensions (must match Pinecone index)"
    )
    generator_model: str = Field(default="gpt-4o", description="LLM for response generation")
    evaluator_model: str = Field(default="gpt-4o", description="LLM for CRAG evaluation/critic")

    # ── Retrieval ──
    top_k: int = Field(default=10, description="Number of documents to retrieve")

    # ── CRAG Thresholds ──
    confidence_threshold_high: float = Field(
        default=0.8,
        description="Above this → GENERATE (retrieved docs are sufficient)",
    )
    confidence_threshold_low: float = Field(
        default=0.4,
        description="Below this → REINDEX/ALERT (retrieved docs are poor quality)",
    )

    # ── Generation ──
    temperature: float = Field(default=0.1, description="LLM temperature for generation")
    max_tokens: int = Field(default=2048, description="Max tokens for generated response")

    # ── Distilled Critic ──
    use_distilled_critic: bool = Field(
        default=False, description="Use local model for fast first-pass evaluation"
    )
    distilled_critic_model: str = Field(
        default="phi3:mini", description="Ollama model name for distilled critic"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama server URL"
    )

    # ── GraphRAG ──
    use_graph_rag: bool = Field(
        default=True, description="Enable knowledge graph for Good Law verification"
    )


def get_settings() -> Settings:
    """Load and return application settings."""
    return Settings()
