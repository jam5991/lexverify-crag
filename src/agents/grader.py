"""
Hallucination Grader.

Post-generation check that verifies the generated response is grounded
in the source documents. Uses NLI-style reasoning to detect ungrounded claims.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.core.retriever import RetrievedDocument


# ── Data Models ──


class SentenceVerdict(BaseModel):
    """Grounding verdict for a single sentence."""

    sentence: str = Field(description="The sentence being evaluated")
    is_grounded: bool = Field(description="Whether the sentence is grounded in sources")
    source_index: int | None = Field(
        default=None, description="Index of the supporting source document (if grounded)"
    )
    reasoning: str = Field(description="Explanation of the verdict")


class GradingResult(BaseModel):
    """Overall hallucination grading result."""

    is_grounded: bool = Field(description="Whether the entire response is grounded")
    grounded_ratio: float = Field(
        description="Fraction of sentences that are grounded (0.0 to 1.0)"
    )
    sentence_verdicts: list[SentenceVerdict] = Field(
        default_factory=list, description="Per-sentence grounding verdicts"
    )
    ungrounded_claims: list[str] = Field(
        default_factory=list, description="Sentences that are NOT grounded in sources"
    )


class GradingResponse(BaseModel):
    """Structured LLM response for hallucination grading."""

    verdicts: list[SentenceVerdict] = Field(
        description="Per-sentence grounding verdicts"
    )


# ── Grader ──

GRADER_SYSTEM_PROMPT = """\
You are a hallucination detection system for legal responses.

Your task is to verify whether each sentence in a generated legal response \
is grounded in (supported by) the provided source documents.

For each sentence:
1. Determine if the claim is directly supported by at least one source document.
2. If supported, identify which source document (by index, starting from 1).
3. If NOT supported, mark it as ungrounded.

Consider a sentence grounded if:
- The factual claim can be directly traced to a source document.
- Legal citations match what's in the sources.
- The legal conclusion logically follows from the source materials.

Consider a sentence ungrounded if:
- It contains facts, case names, or statute numbers not in any source.
- It makes legal claims beyond what the sources support.
- It contains fabricated citations or holdings.

Note: General transitional phrases and concluding remarks that don't make \
factual claims should be marked as grounded.
"""

GRADER_USER_PROMPT = """\
**Generated Response**:
{response}

**Source Documents**:
{documents}

Evaluate each sentence in the response for grounding in the source documents.
"""


class HallucinationGrader:
    """Grades generated responses for hallucination by checking source grounding.

    Uses NLI-style reasoning to verify each sentence in the generated
    response is supported by the provided source documents.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.llm = ChatOpenAI(
            model=self.settings.evaluator_model,
            api_key=self.settings.openai_api_key,
            temperature=0,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", GRADER_SYSTEM_PROMPT),
            ("human", GRADER_USER_PROMPT),
        ])

    def _format_documents(self, documents: list[RetrievedDocument]) -> str:
        """Format source documents for the grader prompt."""
        parts = []
        for i, doc in enumerate(documents):
            parts.append(
                f"--- [Source {i + 1}] ---\n"
                f"URI: {doc.source_uri}\n"
                f"Content:\n{doc.content}\n"
            )
        return "\n".join(parts)

    async def grade(
        self,
        response: str,
        source_documents: list[RetrievedDocument],
    ) -> GradingResult:
        """Grade a generated response for hallucination.

        Args:
            response: The generated legal response text.
            source_documents: Documents the response should be grounded in.

        Returns:
            GradingResult with per-sentence verdicts and ungrounded claims.
        """
        if not response.strip():
            return GradingResult(
                is_grounded=True,
                grounded_ratio=1.0,
                sentence_verdicts=[],
                ungrounded_claims=[],
            )

        docs_text = self._format_documents(source_documents)

        structured_llm = self.llm.with_structured_output(GradingResponse)
        chain = self.prompt | structured_llm

        result: GradingResponse = await chain.ainvoke({
            "response": response,
            "documents": docs_text,
        })

        verdicts = result.verdicts
        ungrounded = [v.sentence for v in verdicts if not v.is_grounded]
        grounded_count = sum(1 for v in verdicts if v.is_grounded)
        total = len(verdicts) if verdicts else 1

        return GradingResult(
            is_grounded=len(ungrounded) == 0,
            grounded_ratio=grounded_count / total,
            sentence_verdicts=verdicts,
            ungrounded_claims=ungrounded,
        )
