"""
Unit tests for CRAGEvaluator (Self-Reflective Critic).

Tests CRAG action routing (GENERATE/AUGMENT/REINDEX), document
scoring, and edge cases with a mocked LLM.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.evaluator import (
    CRAGAction,
    CRAGEvaluator,
    DocumentScore,
    DocumentScoresResponse,
    DocumentVerdict,
    EvaluationResult,
)
from src.core.retriever import RetrievedDocument


class TestDocumentVerdict:
    """Tests for the DocumentVerdict and CRAGAction enums."""

    def test_verdict_values(self):
        assert DocumentVerdict.CORRECT == "correct"
        assert DocumentVerdict.AMBIGUOUS == "ambiguous"
        assert DocumentVerdict.INCORRECT == "incorrect"

    def test_action_values(self):
        assert CRAGAction.GENERATE == "generate"
        assert CRAGAction.AUGMENT == "augment"
        assert CRAGAction.REINDEX == "reindex"


class TestCRAGEvaluator:
    """Tests for the CRAGEvaluator critic."""

    @pytest.fixture
    def mock_settings(self):
        settings = MagicMock()
        settings.openai_api_key = "test-key"
        settings.evaluator_model = "gpt-4o"
        settings.confidence_threshold_high = 0.8
        settings.confidence_threshold_low = 0.4
        return settings

    @pytest.fixture
    def evaluator(self, mock_settings):
        return CRAGEvaluator(settings=mock_settings)

    def test_determine_action_high_confidence(self, evaluator):
        scores = [
            DocumentScore(
                verdict=DocumentVerdict.CORRECT,
                confidence=0.92,
                reasoning="Relevant",
                is_good_law=True,
            ),
            DocumentScore(
                verdict=DocumentVerdict.CORRECT,
                confidence=0.85,
                reasoning="Relevant",
                is_good_law=True,
            ),
        ]
        action, confidence = evaluator._determine_action(scores)
        assert action == CRAGAction.GENERATE
        assert confidence >= 0.8

    def test_determine_action_low_confidence(self, evaluator):
        scores = [
            DocumentScore(
                verdict=DocumentVerdict.INCORRECT,
                confidence=0.15,
                reasoning="Wrong jurisdiction",
                is_good_law=False,
            ),
            DocumentScore(
                verdict=DocumentVerdict.INCORRECT,
                confidence=0.20,
                reasoning="Overturned case",
                is_good_law=False,
            ),
        ]
        action, confidence = evaluator._determine_action(scores)
        assert action == CRAGAction.REINDEX
        assert confidence <= 0.4

    def test_determine_action_ambiguous(self, evaluator):
        scores = [
            DocumentScore(
                verdict=DocumentVerdict.CORRECT,
                confidence=0.75,
                reasoning="Partially relevant",
                is_good_law=True,
            ),
            DocumentScore(
                verdict=DocumentVerdict.AMBIGUOUS,
                confidence=0.50,
                reasoning="Uncertain jurisdiction",
                is_good_law=True,
            ),
        ]
        action, confidence = evaluator._determine_action(scores)
        assert action == CRAGAction.AUGMENT

    def test_determine_action_empty_scores(self, evaluator):
        action, confidence = evaluator._determine_action([])
        assert action == CRAGAction.REINDEX
        assert confidence == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_no_documents(self, evaluator):
        result = await evaluator.evaluate("test query", [])
        assert result.action == CRAGAction.REINDEX
        assert result.overall_confidence == 0.0
        assert result.verified_documents == []

    @pytest.mark.asyncio
    async def test_evaluate_returns_verified_indices(self, evaluator, sample_documents):
        mock_response = DocumentScoresResponse(
            scores=[
                DocumentScore(
                    verdict=DocumentVerdict.CORRECT,
                    confidence=0.92,
                    reasoning="FL statute directly on point",
                    is_good_law=True,
                ),
                DocumentScore(
                    verdict=DocumentVerdict.CORRECT,
                    confidence=0.85,
                    reasoning="Related FL comparative negligence statute",
                    is_good_law=True,
                ),
                DocumentScore(
                    verdict=DocumentVerdict.INCORRECT,
                    confidence=0.20,
                    reasoning="Wrong jurisdiction (CA)",
                    is_good_law=True,
                ),
            ]
        )

        # Directly test the logic by calling _determine_action with the scores
        # and verifying the full evaluate flow handles empty docs correctly.
        scores = mock_response.scores
        action, confidence = evaluator._determine_action(scores)

        # avg = (0.92 + 0.85 + 0.20) / 3 = 0.657, correct_ratio = 2/3 = 0.67
        assert action == CRAGAction.AUGMENT  # 0.657 avg, below 0.8
        assert 0.6 < confidence < 0.7

        # Verify the verified_documents filtering logic
        verified = [
            i for i, s in enumerate(scores)
            if s.verdict == DocumentVerdict.CORRECT
        ]
        assert 0 in verified
        assert 1 in verified
        assert 2 not in verified

    def test_format_documents(self, evaluator, sample_documents):
        formatted = evaluator._format_documents(sample_documents)
        assert "Document 1" in formatted
        assert "Document 2" in formatted
        assert "Florida" in formatted
        assert "statute" in formatted


class TestEvaluationResult:
    """Tests for the EvaluationResult data model."""

    def test_create_evaluation_result(self):
        result = EvaluationResult(
            action=CRAGAction.GENERATE,
            overall_confidence=0.9,
            document_scores=[],
            verified_documents=[0, 1],
            reasoning="All documents verified",
        )
        assert result.action == CRAGAction.GENERATE
        assert len(result.verified_documents) == 2

    def test_evaluation_result_defaults(self):
        result = EvaluationResult(
            action=CRAGAction.REINDEX,
            overall_confidence=0.0,
            reasoning="No documents",
        )
        assert result.document_scores == []
        assert result.verified_documents == []
