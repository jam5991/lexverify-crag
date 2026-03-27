"""
Multi-Step Reasoning.

Decomposes complex cross-jurisdiction or comparative legal queries
into sub-queries, runs each through the CRAG pipeline in parallel,
and synthesizes a unified comparative response.
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


# ── Data Models ──


class SubQuery(BaseModel):
    """A decomposed sub-query with jurisdiction context."""

    query: str = Field(description="The specific legal sub-question")
    jurisdiction: str = Field(description="Target jurisdiction for this sub-query")
    reasoning: str = Field(default="", description="Why this sub-query was created")


class DecompositionResult(BaseModel):
    """Result of query decomposition."""

    is_multi_step: bool = Field(description="Whether the query requires multi-step reasoning")
    sub_queries: list[SubQuery] = Field(default_factory=list, description="Decomposed sub-queries")
    original_query: str = Field(default="", description="The original query")
    reasoning: str = Field(default="", description="Explanation of the decomposition")


class SubQueryResult(BaseModel):
    """Result from running a single sub-query through the CRAG pipeline."""

    sub_query: SubQuery
    answer: str = ""
    citations: list[dict] = Field(default_factory=list)
    is_grounded: bool = False
    grounded_ratio: float = 0.0


class SynthesizedResponse(BaseModel):
    """Final synthesized comparative response."""

    answer: str = Field(description="Unified comparative answer")
    sub_results: list[SubQueryResult] = Field(default_factory=list)
    jurisdictions_covered: list[str] = Field(default_factory=list)


# ── Prompts ──

DECOMPOSITION_PROMPT = """\
You are a legal query decomposition expert. Given a complex legal query, \
determine if it requires multi-step reasoning and, if so, break it down \
into simpler sub-queries.

A query requires multi-step reasoning if it:
1. Compares laws across multiple jurisdictions (e.g., "Compare X across FL, CA, NY")
2. Asks about multiple legal topics that require separate analysis
3. Involves temporal analysis (e.g., "How has X changed over time")

For each sub-query, specify:
- The specific legal question
- The target jurisdiction
- Why this sub-query is needed

If the query is simple enough for a single-step answer, set is_multi_step to false.
"""

SYNTHESIS_PROMPT = """\
You are a legal analysis synthesizer. You have received answers to multiple \
sub-queries about different jurisdictions or legal topics. Synthesize these \
into a single, cohesive comparative response.

Rules:
1. Organize the comparison in a clear, structured format.
2. Highlight key differences and similarities between jurisdictions.
3. Use proper citations from each sub-query's results.
4. Include a summary comparison table when appropriate.
5. Note any caveats or limitations in the comparison.

**Original Query**: {original_query}

**Sub-Query Results**:
{sub_results}

Provide a comprehensive comparative analysis.
"""


class MultiStepReasoner:
    """Decomposes complex queries and synthesizes multi-jurisdiction responses.

    Handles cross-jurisdiction comparisons and multi-topic queries by:
    1. Decomposing the query into targeted sub-queries
    2. Running each sub-query through the CRAG pipeline
    3. Synthesizing a unified comparative response
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.llm = ChatOpenAI(
            model=self.settings.generator_model,
            api_key=self.settings.openai_api_key,
            temperature=0,
        )

    async def decompose(self, query: str) -> DecompositionResult:
        """Decompose a complex query into sub-queries.

        Args:
            query: The complex legal query.

        Returns:
            DecompositionResult with sub-queries and jurisdiction tags.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", DECOMPOSITION_PROMPT),
            ("human", "Legal query: {query}"),
        ])

        structured_llm = self.llm.with_structured_output(DecompositionResult)
        chain = prompt | structured_llm

        result: DecompositionResult = await chain.ainvoke({"query": query})
        result.original_query = query

        logger.info(
            "Decomposed query into %d sub-queries (multi_step=%s)",
            len(result.sub_queries),
            result.is_multi_step,
        )

        return result

    async def run_sub_queries(
        self,
        decomposition: DecompositionResult,
        pipeline_fn,
    ) -> list[SubQueryResult]:
        """Run all sub-queries through the CRAG pipeline in parallel.

        Args:
            decomposition: The decomposed query.
            pipeline_fn: Async function that runs a single query through the pipeline.
                          Signature: async fn(query: str, jurisdiction: str) -> dict

        Returns:
            List of SubQueryResult from each sub-query.
        """
        if not decomposition.sub_queries:
            return []

        # Run sub-queries in parallel
        tasks = [
            pipeline_fn(sq.query, sq.jurisdiction)
            for sq in decomposition.sub_queries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        sub_results = []
        for sq, result in zip(decomposition.sub_queries, results):
            if isinstance(result, Exception):
                logger.error("Sub-query failed for %s: %s", sq.jurisdiction, result)
                sub_results.append(SubQueryResult(
                    sub_query=sq,
                    answer=f"Error processing query for {sq.jurisdiction}: {result}",
                ))
            else:
                sub_results.append(SubQueryResult(
                    sub_query=sq,
                    answer=result.get("answer", ""),
                    citations=result.get("citations", []),
                    is_grounded=result.get("is_grounded", False),
                    grounded_ratio=result.get("grounded_ratio", 0.0),
                ))

        return sub_results

    async def synthesize(
        self,
        original_query: str,
        sub_results: list[SubQueryResult],
    ) -> SynthesizedResponse:
        """Synthesize sub-query results into a unified comparative response.

        Args:
            original_query: The original complex query.
            sub_results: Results from each sub-query.

        Returns:
            SynthesizedResponse with unified answer.
        """
        if not sub_results:
            return SynthesizedResponse(
                answer="No sub-query results to synthesize.",
                sub_results=[],
                jurisdictions_covered=[],
            )

        # Format sub-results for synthesis prompt
        parts = []
        for i, sr in enumerate(sub_results, 1):
            parts.append(
                f"--- [{sr.sub_query.jurisdiction}] Sub-Query {i} ---\n"
                f"Question: {sr.sub_query.query}\n"
                f"Answer: {sr.answer}\n"
                f"Grounded: {sr.grounded_ratio:.0%}\n"
            )
        sub_results_text = "\n".join(parts)

        prompt = ChatPromptTemplate.from_messages([
            ("human", SYNTHESIS_PROMPT),
        ])

        response = await self.llm.ainvoke(
            prompt.format_messages(
                original_query=original_query,
                sub_results=sub_results_text,
            )
        )

        jurisdictions = list({sr.sub_query.jurisdiction for sr in sub_results})

        return SynthesizedResponse(
            answer=response.content,
            sub_results=sub_results,
            jurisdictions_covered=jurisdictions,
        )
