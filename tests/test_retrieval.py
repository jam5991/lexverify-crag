"""
Unit tests for LegalRetriever.

Tests jurisdiction filtering, result formatting, and error handling
with a mocked Pinecone client.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.retriever import LegalRetriever, RetrievedDocument


class TestRetrievedDocument:
    """Tests for the RetrievedDocument data model."""

    def test_create_document(self):
        doc = RetrievedDocument(
            content="Test legal content",
            source_uri="https://example.com/law",
            jurisdiction="Florida",
            doc_type="statute",
            relevance_score=0.95,
        )
        assert doc.content == "Test legal content"
        assert doc.jurisdiction == "Florida"
        assert doc.doc_type == "statute"
        assert doc.relevance_score == 0.95

    def test_document_defaults(self):
        doc = RetrievedDocument(content="Minimal document")
        assert doc.source_uri == ""
        assert doc.jurisdiction == ""
        assert doc.doc_type == ""
        assert doc.relevance_score == 0.0

    def test_document_str(self):
        doc = RetrievedDocument(
            content="A" * 200,
            doc_type="case_law",
            jurisdiction="Federal",
        )
        result = str(doc)
        assert "[case_law]" in result
        assert "(Federal)" in result
        assert "..." in result


class TestLegalRetriever:
    """Tests for the LegalRetriever with mocked external services."""

    @pytest.fixture
    def mock_settings(self):
        settings = MagicMock()
        settings.openai_api_key = "test-key"
        settings.pinecone_api_key = "test-pinecone-key"
        settings.pinecone_index_name = "test-index"
        settings.embedding_model = "text-embedding-3-small"
        settings.top_k = 5
        return settings

    @pytest.fixture
    def mock_pinecone_results(self):
        return {
            "matches": [
                {
                    "id": "doc1",
                    "score": 0.95,
                    "metadata": {
                        "text": "Florida Statute § 95.11(3) provides...",
                        "source_uri": "https://flsenate.gov/95.11",
                        "jurisdiction": "Florida",
                        "doc_type": "statute",
                    },
                },
                {
                    "id": "doc2",
                    "score": 0.82,
                    "metadata": {
                        "text": "In Smith v. Jones, the court held...",
                        "source_uri": "https://casetext.com/smith-v-jones",
                        "jurisdiction": "Florida",
                        "doc_type": "case_law",
                    },
                },
            ]
        }

    @pytest.mark.asyncio
    async def test_retrieve_formats_results(self, mock_settings, mock_pinecone_results):
        retriever = LegalRetriever(settings=mock_settings)

        mock_index = MagicMock()
        mock_index.query.return_value = mock_pinecone_results
        retriever._index = mock_index

        with patch.object(retriever, "_get_embedding", return_value=[0.1] * 1536):
            docs = await retriever.retrieve("test query", jurisdiction="Florida")

        assert len(docs) == 2
        assert docs[0].jurisdiction == "Florida"
        assert docs[0].doc_type == "statute"
        assert docs[0].relevance_score == 0.95
        assert docs[1].doc_type == "case_law"

    @pytest.mark.asyncio
    async def test_retrieve_applies_jurisdiction_filter(self, mock_settings):
        retriever = LegalRetriever(settings=mock_settings)

        mock_index = MagicMock()
        mock_index.query.return_value = {"matches": []}
        retriever._index = mock_index

        with patch.object(retriever, "_get_embedding", return_value=[0.1] * 1536):
            await retriever.retrieve("test", jurisdiction="Florida")

        call_kwargs = mock_index.query.call_args[1]
        assert call_kwargs["filter"] == {"jurisdiction": {"$eq": "Florida"}}

    @pytest.mark.asyncio
    async def test_retrieve_no_filter_when_no_jurisdiction(self, mock_settings):
        retriever = LegalRetriever(settings=mock_settings)

        mock_index = MagicMock()
        mock_index.query.return_value = {"matches": []}
        retriever._index = mock_index

        with patch.object(retriever, "_get_embedding", return_value=[0.1] * 1536):
            await retriever.retrieve("test")

        call_kwargs = mock_index.query.call_args[1]
        assert call_kwargs["filter"] is None

    @pytest.mark.asyncio
    async def test_retrieve_respects_top_k(self, mock_settings):
        mock_settings.top_k = 3
        retriever = LegalRetriever(settings=mock_settings)

        mock_index = MagicMock()
        mock_index.query.return_value = {"matches": []}
        retriever._index = mock_index

        with patch.object(retriever, "_get_embedding", return_value=[0.1] * 1536):
            await retriever.retrieve("test", top_k=7)

        call_kwargs = mock_index.query.call_args[1]
        assert call_kwargs["top_k"] == 7

    @pytest.mark.asyncio
    async def test_retrieve_empty_results(self, mock_settings):
        retriever = LegalRetriever(settings=mock_settings)

        mock_index = MagicMock()
        mock_index.query.return_value = {"matches": []}
        retriever._index = mock_index

        with patch.object(retriever, "_get_embedding", return_value=[0.1] * 1536):
            docs = await retriever.retrieve("obscure query")

        assert docs == []
