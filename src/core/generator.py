"""
Legal Response Generator.

Synthesizes verified legal documents into a final response with
citation attribution — every claim mapped to a source URI.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.core.retriever import RetrievedDocument


# ── Data Models ──


class Citation(BaseModel):
    """A citation linking a claim to its source."""

    claim: str = Field(description="The specific legal claim or statement")
    source_uri: str = Field(description="Source URI or document ID")
    source_text: str = Field(default="", description="Relevant excerpt from the source")


class GeneratedResponse(BaseModel):
    """Final verified response with citation attribution."""

    answer: str = Field(description="The complete legal response")
    citations: list[Citation] = Field(
        default_factory=list, description="Citations for each claim"
    )
    jurisdiction: str = Field(default="", description="Applicable jurisdiction")
    confidence_note: str = Field(
        default="",
        description="Any caveats about the response's reliability",
    )


# ── Generator ──

GENERATOR_SYSTEM_PROMPT = """\
You are a legal research assistant that produces precise, well-cited responses.

Rules:
1. **Every factual claim must be attributed** to a specific source document.
2. Use inline citations in the format [Source N] where N corresponds to the \
document number.
3. Never fabricate case names, statute numbers, or legal holdings.
4. If the provided documents are insufficient, explicitly state what is missing.
5. Clearly identify the applicable jurisdiction.
6. Use precise legal language but ensure the response is understandable.
"""

GENERATOR_USER_PROMPT = """\
**Legal Query**: {query}

**Jurisdiction**: {jurisdiction}

**Verified Source Documents**:
{documents}

Provide a comprehensive legal response with proper citations to the source \
documents above. For each factual claim, provide the source document number.
"""


class LegalGenerator:
    """Generates legal responses with citation attribution.

    Takes verified documents from the CRAG evaluator and produces
    a structured response where every claim is mapped to a source.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.llm = ChatOpenAI(
            model=self.settings.generator_model,
            api_key=self.settings.openai_api_key,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", GENERATOR_SYSTEM_PROMPT),
            ("human", GENERATOR_USER_PROMPT),
        ])

    def _format_documents(self, documents: list[RetrievedDocument]) -> str:
        """Format verified documents for the generation prompt."""
        parts = []
        for i, doc in enumerate(documents):
            parts.append(
                f"--- [Source {i + 1}] ---\n"
                f"URI: {doc.source_uri}\n"
                f"Jurisdiction: {doc.jurisdiction}\n"
                f"Type: {doc.doc_type}\n"
                f"Content:\n{doc.content}\n"
            )
        return "\n".join(parts)

    async def generate(
        self,
        query: str,
        verified_documents: list[RetrievedDocument],
        jurisdiction: str = "",
    ) -> GeneratedResponse:
        """Generate a legal response with citation attribution.

        Args:
            query: The original legal question.
            verified_documents: Documents that passed CRAG evaluation.
            jurisdiction: Target jurisdiction.

        Returns:
            GeneratedResponse with answer text and per-claim citations.
        """
        if not verified_documents:
            return GeneratedResponse(
                answer=(
                    "I was unable to find sufficiently reliable sources to answer "
                    "this legal question. Please consult a licensed attorney in the "
                    "relevant jurisdiction for accurate legal guidance."
                ),
                citations=[],
                jurisdiction=jurisdiction,
                confidence_note="No verified documents available.",
            )

        docs_text = self._format_documents(verified_documents)

        # Use structured output for citations
        structured_llm = self.llm.with_structured_output(GeneratedResponse)
        chain = self.prompt | structured_llm

        response: GeneratedResponse = await chain.ainvoke({
            "query": query,
            "jurisdiction": jurisdiction,
            "documents": docs_text,
        })

        response.jurisdiction = jurisdiction
        return response
