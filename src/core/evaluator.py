"""
CRAG Evaluator — The Self-Reflective Critic.

This is the core of the Corrective RAG architecture. It evaluates retrieved
documents for legal relevance, recency, and jurisdictional accuracy, then
decides whether to GENERATE, AUGMENT (web search), or REINDEX (alert).
"""

from __future__ import annotations

from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.core.retriever import RetrievedDocument


# ── Data Models ──


class DocumentVerdict(str, Enum):
    """Verdict for a single retrieved document."""

    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


class CRAGAction(str, Enum):
    """Pipeline action based on aggregate evaluation."""

    GENERATE = "generate"  # High confidence — proceed to generation
    AUGMENT = "augment"  # Ambiguous — trigger web search fallback
    REINDEX = "reindex"  # Low confidence — alert and re-index


class DocumentScore(BaseModel):
    """Evaluation score for a single document."""

    verdict: DocumentVerdict = Field(description="Correct, ambiguous, or incorrect")
    confidence: float = Field(
        description="Confidence score 0-1 for this document's relevance"
    )
    reasoning: str = Field(description="Brief explanation of the verdict")
    is_good_law: bool = Field(
        default=True,
        description="Whether the cited law is still valid (not overturned/superseded)",
    )


class EvaluationResult(BaseModel):
    """Aggregate evaluation result across all retrieved documents."""

    action: CRAGAction = Field(description="Pipeline action to take")
    overall_confidence: float = Field(description="Average confidence across documents")
    document_scores: list[DocumentScore] = Field(
        default_factory=list, description="Per-document evaluation scores"
    )
    verified_documents: list[int] = Field(
        default_factory=list,
        description="Indices of documents that passed evaluation (verdict=correct)",
    )
    reasoning: str = Field(description="Summary of the evaluation decision")


# ── Evaluator ──

EVALUATOR_SYSTEM_PROMPT = """\
You are a Legal Document Relevance Critic for a Corrective RAG system.

Your role is to evaluate whether a retrieved legal document is relevant and \
reliable for answering the user's legal query.

For each document, assess:
1. **Legal Relevance**: Does the document directly address the legal question?
2. **Jurisdictional Match**: Is it from the correct jurisdiction?
3. **Good Law Status**: Is the cited law still valid (not overturned, superseded, or expired)?
4. **Recency**: Is the document current enough for the legal context?

Score each document as:
- **correct**: Highly relevant, correct jurisdiction, good law, current.
- **ambiguous**: Partially relevant or uncertain jurisdiction/validity.
- **incorrect**: Irrelevant, wrong jurisdiction, bad law, or outdated.

Provide a confidence score (0.0 to 1.0) and brief reasoning for each.
"""

EVALUATOR_USER_PROMPT = """\
**Legal Query**: {query}

**Target Jurisdiction**: {jurisdiction}

**Retrieved Documents**:
{documents}

Evaluate each document and provide your assessment.
"""


class CRAGEvaluator:
    """The Self-Reflective Critic that evaluates retrieved documents.

    Scores each document for legal relevance, jurisdictional accuracy,
    and Good Law status, then decides the pipeline action.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.llm = ChatOpenAI(
            model=self.settings.evaluator_model,
            api_key=self.settings.openai_api_key,
            temperature=0,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", EVALUATOR_SYSTEM_PROMPT),
            ("human", EVALUATOR_USER_PROMPT),
        ])

    def _format_documents(self, documents: list[RetrievedDocument]) -> str:
        """Format documents for the evaluator prompt."""
        parts = []
        for i, doc in enumerate(documents):
            parts.append(
                f"--- Document {i + 1} ---\n"
                f"Source: {doc.source_uri}\n"
                f"Jurisdiction: {doc.jurisdiction}\n"
                f"Type: {doc.doc_type}\n"
                f"Content:\n{doc.content}\n"
            )
        return "\n".join(parts)

    def _determine_action(self, scores: list[DocumentScore]) -> tuple[CRAGAction, float]:
        """Determine pipeline action from document scores."""
        if not scores:
            return CRAGAction.REINDEX, 0.0

        avg_confidence = sum(s.confidence for s in scores) / len(scores)
        correct_count = sum(1 for s in scores if s.verdict == DocumentVerdict.CORRECT)
        correct_ratio = correct_count / len(scores)

        if avg_confidence >= self.settings.confidence_threshold_high and correct_ratio >= 0.5:
            return CRAGAction.GENERATE, avg_confidence
        elif avg_confidence <= self.settings.confidence_threshold_low:
            return CRAGAction.REINDEX, avg_confidence
        else:
            return CRAGAction.AUGMENT, avg_confidence

    async def evaluate(
        self,
        query: str,
        documents: list[RetrievedDocument],
        jurisdiction: str = "",
    ) -> EvaluationResult:
        """Evaluate retrieved documents for the CRAG pipeline.

        Args:
            query: The original legal query.
            documents: Retrieved documents to evaluate.
            jurisdiction: Target jurisdiction for relevance assessment.

        Returns:
            EvaluationResult with action, scores, and verified document indices.
        """
        if not documents:
            return EvaluationResult(
                action=CRAGAction.REINDEX,
                overall_confidence=0.0,
                document_scores=[],
                verified_documents=[],
                reasoning="No documents retrieved — requires re-indexing or web search.",
            )

        # Build LLM with structured output for per-document scoring
        scored_llm = self.llm.with_structured_output(
            DocumentScoresResponse, method="json_schema"
        )
        chain = self.prompt | scored_llm

        docs_text = self._format_documents(documents)
        response: DocumentScoresResponse = await chain.ainvoke({
            "query": query,
            "jurisdiction": jurisdiction,
            "documents": docs_text,
        })

        scores = response.scores
        action, overall_confidence = self._determine_action(scores)

        # Identify verified (correct) documents
        verified = [
            i for i, s in enumerate(scores) if s.verdict == DocumentVerdict.CORRECT
        ]

        return EvaluationResult(
            action=action,
            overall_confidence=overall_confidence,
            document_scores=scores,
            verified_documents=verified,
            reasoning=(
                f"Action={action.value}: {len(verified)}/{len(scores)} documents passed "
                f"(avg confidence={overall_confidence:.2f})"
            ),
        )


class DocumentScoresResponse(BaseModel):
    """Structured LLM response containing per-document scores."""

    scores: list[DocumentScore] = Field(description="Evaluation scores for each document")
