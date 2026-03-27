"""Tests for Multi-Step Reasoning (query decomposition and synthesis)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.multi_step import (
    DecompositionResult,
    MultiStepReasoner,
    SubQuery,
    SubQueryResult,
    SynthesizedResponse,
)


@pytest.fixture
def settings():
    from src.config import Settings

    return Settings(openai_api_key="test-key")


@pytest.fixture
def reasoner(settings) -> MultiStepReasoner:
    return MultiStepReasoner(settings=settings)


class TestSubQuery:
    def test_create_sub_query(self):
        sq = SubQuery(query="What is PI SOL in FL?", jurisdiction="Florida")
        assert sq.jurisdiction == "Florida"

    def test_sub_query_with_reasoning(self):
        sq = SubQuery(
            query="What is PI SOL?",
            jurisdiction="California",
            reasoning="Need to compare CA",
        )
        assert sq.reasoning == "Need to compare CA"


class TestDecompositionResult:
    def test_single_step_query(self):
        result = DecompositionResult(
            is_multi_step=False,
            sub_queries=[],
            original_query="Simple question",
        )
        assert not result.is_multi_step
        assert len(result.sub_queries) == 0

    def test_multi_step_query(self):
        result = DecompositionResult(
            is_multi_step=True,
            sub_queries=[
                SubQuery(query="FL SOL?", jurisdiction="Florida"),
                SubQuery(query="CA SOL?", jurisdiction="California"),
            ],
            original_query="Compare SOL across FL and CA",
        )
        assert result.is_multi_step
        assert len(result.sub_queries) == 2


class TestSubQueryResult:
    def test_create_result(self):
        sq = SubQuery(query="Test?", jurisdiction="FL")
        result = SubQueryResult(
            sub_query=sq,
            answer="2 years",
            is_grounded=True,
            grounded_ratio=1.0,
        )
        assert result.answer == "2 years"
        assert result.is_grounded


class TestMultiStepReasoner:
    @pytest.mark.asyncio
    async def test_run_sub_queries_empty(self, reasoner: MultiStepReasoner):
        decomposition = DecompositionResult(
            is_multi_step=True,
            sub_queries=[],
            original_query="test",
        )
        results = await reasoner.run_sub_queries(decomposition, AsyncMock())
        assert results == []

    @pytest.mark.asyncio
    async def test_run_sub_queries_parallel(self, reasoner: MultiStepReasoner):
        decomposition = DecompositionResult(
            is_multi_step=True,
            sub_queries=[
                SubQuery(query="FL SOL?", jurisdiction="Florida"),
                SubQuery(query="CA SOL?", jurisdiction="California"),
            ],
            original_query="Compare SOL",
        )

        mock_pipeline = AsyncMock(
            side_effect=[
                {"answer": "2 years in FL", "citations": [], "is_grounded": True, "grounded_ratio": 1.0},
                {"answer": "2 years in CA", "citations": [], "is_grounded": True, "grounded_ratio": 1.0},
            ]
        )

        results = await reasoner.run_sub_queries(decomposition, mock_pipeline)

        assert len(results) == 2
        assert results[0].answer == "2 years in FL"
        assert results[1].answer == "2 years in CA"
        assert mock_pipeline.call_count == 2

    @pytest.mark.asyncio
    async def test_run_sub_queries_handles_errors(self, reasoner: MultiStepReasoner):
        decomposition = DecompositionResult(
            is_multi_step=True,
            sub_queries=[
                SubQuery(query="FL SOL?", jurisdiction="Florida"),
            ],
            original_query="test",
        )

        mock_pipeline = AsyncMock(side_effect=RuntimeError("LLM error"))
        results = await reasoner.run_sub_queries(decomposition, mock_pipeline)

        assert len(results) == 1
        assert "Error" in results[0].answer

    @pytest.mark.asyncio
    async def test_synthesize_empty(self, reasoner: MultiStepReasoner):
        result = await reasoner.synthesize("test", [])
        assert result.answer == "No sub-query results to synthesize."
        assert result.jurisdictions_covered == []

    def test_synthesized_response_model(self):
        resp = SynthesizedResponse(
            answer="Comparison: FL=2yr, CA=2yr",
            sub_results=[],
            jurisdictions_covered=["Florida", "California"],
        )
        assert len(resp.jurisdictions_covered) == 2
