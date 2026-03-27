"""Tests for the Distilled Critic (fast evaluation with GPT-4o fallback)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.distilled_critic import DistilledCritic
from src.core.evaluator import CRAGAction, DocumentScore, DocumentScoresResponse, DocumentVerdict
from src.core.retriever import RetrievedDocument


@pytest.fixture
def settings():
    """Create test settings with distilled critic enabled."""
    from src.config import Settings

    return Settings(
        openai_api_key="test-key",
        use_distilled_critic=True,
        distilled_critic_model="phi3:mini",
        ollama_base_url="http://localhost:11434",
    )


@pytest.fixture
def critic(settings) -> DistilledCritic:
    return DistilledCritic(settings=settings)


@pytest.fixture
def sample_documents() -> list[RetrievedDocument]:
    return [
        RetrievedDocument(
            content="FL Statute § 95.11 — 2 year SOL for PI",
            source_uri="https://example.com/95.11",
            jurisdiction="Florida",
            doc_type="statute",
            relevance_score=0.95,
        ),
        RetrievedDocument(
            content="CA CCP § 335.1 — 2 year SOL for PI",
            source_uri="https://example.com/335.1",
            jurisdiction="California",
            doc_type="statute",
            relevance_score=0.80,
        ),
    ]


class TestDistilledCritic:
    @pytest.mark.asyncio
    async def test_evaluate_empty_documents(self, critic: DistilledCritic):
        result = await critic.evaluate("test query", [], jurisdiction="Florida")
        assert result.action == CRAGAction.REINDEX
        assert result.overall_confidence == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_falls_back_to_gpt4o_on_error(
        self, critic: DistilledCritic, sample_documents: list[RetrievedDocument]
    ):
        """When local model fails, should fall back to GPT-4o."""
        mock_scores = DocumentScoresResponse(
            scores=[
                DocumentScore(verdict=DocumentVerdict.CORRECT, confidence=0.9, reasoning="Good", is_good_law=True),
                DocumentScore(verdict=DocumentVerdict.CORRECT, confidence=0.85, reasoning="Good", is_good_law=True),
            ]
        )

        mock_full_evaluator = MagicMock()
        mock_full_evaluator.evaluate = AsyncMock(
            return_value=MagicMock(
                action=CRAGAction.GENERATE,
                overall_confidence=0.875,
                document_scores=mock_scores.scores,
                verified_documents=[0, 1],
                reasoning="GPT-4o fallback",
            )
        )

        # The local model will fail (Ollama not running)
        critic._full_evaluator = mock_full_evaluator

        result = await critic.evaluate("test query", sample_documents, "Florida")

        # Should have fallen back to GPT-4o
        mock_full_evaluator.evaluate.assert_called_once()
        assert result.action == CRAGAction.GENERATE

    def test_format_documents(self, critic: DistilledCritic, sample_documents: list[RetrievedDocument]):
        formatted = critic._format_documents(sample_documents)
        assert "[Doc 1]" in formatted
        assert "[Doc 2]" in formatted
        assert "Florida" in formatted

    def test_get_full_evaluator(self, critic: DistilledCritic):
        evaluator = critic._get_full_evaluator()
        assert evaluator is not None
        # Should return same instance on second call
        assert critic._get_full_evaluator() is evaluator
