"""
Legal Document Retriever.

Wraps Pinecone vector store for semantic retrieval of legal documents
with jurisdiction-aware metadata filtering.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


# ── Data Models ──


class RetrievedDocument(BaseModel):
    """A document retrieved from the vector store or web search."""

    content: str = Field(description="Text content of the chunk")
    source_uri: str = Field(default="", description="Source URL or document ID")
    jurisdiction: str = Field(default="", description="Jurisdiction of the source")
    doc_type: str = Field(default="", description="Type: statute, case_law, regulation")
    relevance_score: float = Field(default=0.0, description="Retrieval similarity score")

    def __str__(self) -> str:
        return f"[{self.doc_type}] ({self.jurisdiction}) {self.content[:120]}..."


class LegalRetriever:
    """Retrieves legal documents from Pinecone with jurisdiction filtering.

    Uses OpenAI text-embedding-3-small for query embedding and supports
    metadata filtering by jurisdiction and document type.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._index = None

    def _get_index(self):
        """Lazy-initialize the Pinecone index."""
        if self._index is None:
            from pinecone import Pinecone

            pc = Pinecone(api_key=self.settings.pinecone_api_key)
            self._index = pc.Index(self.settings.pinecone_index_name)
        return self._index

    def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding for a query string."""
        from openai import OpenAI

        client = OpenAI(api_key=self.settings.openai_api_key)
        response = client.embeddings.create(
            model=self.settings.embedding_model,
            input=text,
            dimensions=self.settings.embedding_dimensions,
        )
        return response.data[0].embedding

    async def retrieve(
        self,
        query: str,
        jurisdiction: str | None = None,
        doc_type: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedDocument]:
        """Retrieve relevant legal documents.

        Args:
            query: The legal question to search for.
            jurisdiction: Optional jurisdiction filter (e.g., "Florida").
            doc_type: Optional document type filter (e.g., "statute").
            top_k: Number of results (defaults to settings.top_k).

        Returns:
            List of RetrievedDocument ordered by relevance.
            Returns empty list if Pinecone is unavailable (triggers CRAG fallback).
        """
        k = top_k or self.settings.top_k

        try:
            query_embedding = self._get_embedding(query)

            # Build metadata filter
            metadata_filter = {}
            if jurisdiction:
                metadata_filter["jurisdiction"] = {"$eq": jurisdiction}
            if doc_type:
                metadata_filter["doc_type"] = {"$eq": doc_type}

            # Query Pinecone
            index = self._get_index()
            results = index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True,
                filter=metadata_filter if metadata_filter else None,
            )

            # Map to RetrievedDocument models
            documents = []
            for match in results.get("matches", []):
                meta = match.get("metadata", {})
                documents.append(
                    RetrievedDocument(
                        content=meta.get("text", ""),
                        source_uri=meta.get("source_uri", ""),
                        jurisdiction=meta.get("jurisdiction", ""),
                        doc_type=meta.get("doc_type", ""),
                        relevance_score=match.get("score", 0.0),
                    )
                )

            return documents

        except Exception as e:
            logger.warning(
                "Pinecone retrieval failed (will fall back to web search): %s", e
            )
            return []
