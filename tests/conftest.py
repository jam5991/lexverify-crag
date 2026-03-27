"""Shared pytest fixtures for LexVerify tests."""

from __future__ import annotations

import pytest

from src.core.evaluator import CRAGAction, DocumentScore, DocumentVerdict, EvaluationResult
from src.core.retriever import RetrievedDocument


@pytest.fixture
def sample_query() -> str:
    """Sample Florida personal injury query."""
    return "What is the statute of limitations for personal injury claims in Florida?"


@pytest.fixture
def sample_documents() -> list[RetrievedDocument]:
    """Sample retrieved documents for testing."""
    return [
        RetrievedDocument(
            content=(
                "Under Fla. Stat. § 95.11(3), the statute of limitations for personal "
                "injury actions in Florida is two years. This was amended by HB 837 "
                "(effective March 24, 2023), reducing it from four years to two years."
            ),
            source_uri="https://www.flsenate.gov/Laws/Statutes/2023/95.11",
            jurisdiction="Florida",
            doc_type="statute",
            relevance_score=0.95,
        ),
        RetrievedDocument(
            content=(
                "Florida Statute § 768.81 establishes the modified comparative negligence "
                "standard. Under HB 837, plaintiffs more than 50% at fault are barred "
                "from recovery."
            ),
            source_uri="https://www.flsenate.gov/Laws/Statutes/2023/768.81",
            jurisdiction="Florida",
            doc_type="statute",
            relevance_score=0.78,
        ),
        RetrievedDocument(
            content=(
                "California Code of Civil Procedure § 335.1 provides a two-year "
                "statute of limitations for personal injury actions in California."
            ),
            source_uri="https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml",
            jurisdiction="California",
            doc_type="statute",
            relevance_score=0.62,
        ),
    ]


@pytest.fixture
def high_confidence_evaluation() -> EvaluationResult:
    """Evaluation result where documents pass with high confidence."""
    return EvaluationResult(
        action=CRAGAction.GENERATE,
        overall_confidence=0.88,
        document_scores=[
            DocumentScore(
                verdict=DocumentVerdict.CORRECT,
                confidence=0.95,
                reasoning="Directly addresses FL statute of limitations",
                is_good_law=True,
            ),
            DocumentScore(
                verdict=DocumentVerdict.CORRECT,
                confidence=0.80,
                reasoning="Related FL statute, supports context",
                is_good_law=True,
            ),
        ],
        verified_documents=[0, 1],
        reasoning="Action=generate: 2/2 documents passed (avg confidence=0.88)",
    )


@pytest.fixture
def low_confidence_evaluation() -> EvaluationResult:
    """Evaluation result where documents fail — triggers web search."""
    return EvaluationResult(
        action=CRAGAction.AUGMENT,
        overall_confidence=0.45,
        document_scores=[
            DocumentScore(
                verdict=DocumentVerdict.AMBIGUOUS,
                confidence=0.45,
                reasoning="Wrong jurisdiction (CA instead of FL)",
                is_good_law=True,
            ),
        ],
        verified_documents=[],
        reasoning="Action=augment: 0/1 documents passed (avg confidence=0.45)",
    )
