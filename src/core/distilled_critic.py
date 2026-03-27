"""
Distilled Critic Model.

Fast first-pass evaluator using a local model via Ollama for reduced latency.
Falls back to GPT-4o when the local model's confidence is in the ambiguous range.
"""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.core.evaluator import (
    CRAGAction,
    CRAGEvaluator,
    DocumentScore,
    DocumentScoresResponse,
    DocumentVerdict,
    EvaluationResult,
)
from src.core.retriever import RetrievedDocument

logger = logging.getLogger(__name__)


FAST_EVALUATOR_PROMPT = """\
You are a legal document relevance scorer. For each document, rate:
- verdict: correct, ambiguous, or incorrect
- confidence: 0.0 to 1.0
- reasoning: one sentence
- is_good_law: true/false

Query: {query}
Jurisdiction: {jurisdiction}
Documents:
{documents}
"""


class DistilledCritic:
    """Fast local-model critic with GPT-4o escalation.

    Uses a lightweight model via Ollama for initial document scoring.
    When confidence falls in the ambiguous range (0.4–0.8), escalates
    to the full GPT-4o CRAGEvaluator for a second opinion.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._local_llm = None
        self._full_evaluator = None

    def _get_local_llm(self):
        """Lazy-initialize the Ollama-based local LLM."""
        if self._local_llm is None:
            try:
                from langchain_ollama import ChatOllama

                self._local_llm = ChatOllama(
                    model=self.settings.distilled_critic_model,
                    base_url=self.settings.ollama_base_url,
                    temperature=0,
                )
                logger.info(
                    "Distilled critic initialized: model=%s, url=%s",
                    self.settings.distilled_critic_model,
                    self.settings.ollama_base_url,
                )
            except ImportError:
                logger.warning(
                    "langchain-ollama not installed. Install with: "
                    "pip install langchain-ollama"
                )
                raise
        return self._local_llm

    def _get_full_evaluator(self) -> CRAGEvaluator:
        """Get the full GPT-4o evaluator for escalation."""
        if self._full_evaluator is None:
            self._full_evaluator = CRAGEvaluator(settings=self.settings)
        return self._full_evaluator

    def _format_documents(self, documents: list[RetrievedDocument]) -> str:
        """Format documents for the local model prompt."""
        parts = []
        for i, doc in enumerate(documents):
            parts.append(
                f"[Doc {i + 1}] ({doc.jurisdiction} | {doc.doc_type}) "
                f"{doc.content[:300]}..."
            )
        return "\n".join(parts)

    async def evaluate(
        self,
        query: str,
        documents: list[RetrievedDocument],
        jurisdiction: str = "",
    ) -> EvaluationResult:
        """Fast evaluation with optional GPT-4o escalation.

        Phase 1: Run documents through the local model for quick scoring.
        Phase 2: If overall confidence is ambiguous (0.4–0.8), escalate
                 to the full GPT-4o evaluator.

        Args:
            query: Legal query.
            documents: Retrieved documents to evaluate.
            jurisdiction: Target jurisdiction.

        Returns:
            EvaluationResult from either the local model or GPT-4o.
        """
        if not documents:
            return EvaluationResult(
                action=CRAGAction.REINDEX,
                overall_confidence=0.0,
                document_scores=[],
                verified_documents=[],
                reasoning="No documents retrieved.",
            )

        try:
            # Phase 1: Fast local evaluation
            logger.info("Distilled critic: fast evaluation with %s", self.settings.distilled_critic_model)
            local_result = await self._local_evaluate(query, documents, jurisdiction)

            # Phase 2: Escalate if ambiguous
            if local_result.action == CRAGAction.AUGMENT:
                logger.info(
                    "Distilled critic: ambiguous result (%.2f), escalating to GPT-4o",
                    local_result.overall_confidence,
                )
                full_evaluator = self._get_full_evaluator()
                return await full_evaluator.evaluate(query, documents, jurisdiction)

            return local_result

        except Exception as e:
            # If local model fails, fall back to full evaluator
            logger.warning(
                "Distilled critic failed (%s), falling back to GPT-4o", e
            )
            full_evaluator = self._get_full_evaluator()
            return await full_evaluator.evaluate(query, documents, jurisdiction)

    async def _local_evaluate(
        self,
        query: str,
        documents: list[RetrievedDocument],
        jurisdiction: str,
    ) -> EvaluationResult:
        """Run evaluation through the local model."""
        llm = self._get_local_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("human", FAST_EVALUATOR_PROMPT),
        ])

        scored_llm = llm.with_structured_output(DocumentScoresResponse)
        chain = prompt | scored_llm

        docs_text = self._format_documents(documents)
        response: DocumentScoresResponse = await chain.ainvoke({
            "query": query,
            "jurisdiction": jurisdiction,
            "documents": docs_text,
        })

        scores = response.scores
        if not scores:
            return EvaluationResult(
                action=CRAGAction.AUGMENT,
                overall_confidence=0.5,
                document_scores=[],
                verified_documents=[],
                reasoning="Local model returned no scores — escalating.",
            )

        avg_confidence = sum(s.confidence for s in scores) / len(scores)
        correct_count = sum(1 for s in scores if s.verdict == DocumentVerdict.CORRECT)
        correct_ratio = correct_count / len(scores)

        if avg_confidence >= self.settings.confidence_threshold_high and correct_ratio >= 0.5:
            action = CRAGAction.GENERATE
        elif avg_confidence <= self.settings.confidence_threshold_low:
            action = CRAGAction.REINDEX
        else:
            action = CRAGAction.AUGMENT

        verified = [i for i, s in enumerate(scores) if s.verdict == DocumentVerdict.CORRECT]

        return EvaluationResult(
            action=action,
            overall_confidence=avg_confidence,
            document_scores=scores,
            verified_documents=verified,
            reasoning=(
                f"[Distilled] Action={action.value}: {len(verified)}/{len(scores)} passed "
                f"(avg confidence={avg_confidence:.2f})"
            ),
        )
