"""
Jurisdictional Router.

Classifies a legal query by jurisdiction (federal vs. state, specific state)
and produces routing metadata used to filter downstream retrieval.
"""

from __future__ import annotations

from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import Settings, get_settings


# ── Data Models ──


class JurisdictionLevel(str, Enum):
    """Federal vs. state jurisdiction."""

    FEDERAL = "federal"
    STATE = "state"
    BOTH = "both"


class RoutingResult(BaseModel):
    """Result of jurisdictional classification."""

    level: JurisdictionLevel = Field(description="Federal, state, or both")
    state: str | None = Field(default=None, description="Specific state (e.g., 'Florida')")
    area_of_law: str = Field(description="Legal area (e.g., 'personal injury', 'contract law')")
    reasoning: str = Field(description="Brief explanation of the routing decision")


# ── Router ──

ROUTER_SYSTEM_PROMPT = """\
You are a legal jurisdiction classifier. Given a legal query, determine:
1. Whether it falls under federal law, state law, or both.
2. If state law, which specific state.
3. The area of law (e.g., personal injury, contract, criminal, family).

Respond with a structured classification. If the jurisdiction is unclear, \
default to "both" with the most likely state.
"""


class JurisdictionalRouter:
    """Routes legal queries to the appropriate jurisdiction for retrieval filtering."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.llm = ChatOpenAI(
            model=self.settings.evaluator_model,
            api_key=self.settings.openai_api_key,
            temperature=0,
        ).with_structured_output(RoutingResult)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_SYSTEM_PROMPT),
            (
                "human",
                "Legal query: {query}\n"
                "{jurisdiction_hint}",
            ),
        ])
        self.chain = self.prompt | self.llm

    async def route(
        self,
        query: str,
        jurisdiction_hint: str | None = None,
    ) -> RoutingResult:
        """Classify a query's jurisdiction.

        Args:
            query: The legal question.
            jurisdiction_hint: Optional hint (e.g., "Florida").

        Returns:
            RoutingResult with jurisdiction metadata.
        """
        hint_text = f"Jurisdiction hint: {jurisdiction_hint}" if jurisdiction_hint else ""
        result = await self.chain.ainvoke({"query": query, "jurisdiction_hint": hint_text})
        return result
