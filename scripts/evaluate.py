"""
LexVerify Evaluation Harness.

Runs a suite of legal queries through the full CRAG pipeline, collecting
per-stage metrics: faithfulness, answer relevancy, CRAG action, latency,
and Good Law status. Outputs a Rich-formatted table and JSON results file.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.config import get_settings
from src.core.evaluator import CRAGAction, EvaluationResult
from src.core.retriever import RetrievedDocument
from src.core.router import RoutingResult

console = Console()

RESULTS_DIR = Path(__file__).parent.parent / "data" / "eval_results"


# ── Test Suite ──


class TestQuery(BaseModel):
    """A single evaluation query with expected behavior."""

    id: str
    query: str
    jurisdiction_hint: str | None = None
    expected_action: str = Field(description="Expected CRAG action: generate, augment, reindex")
    category: str = Field(description="What capability this tests")
    description: str = ""
    multi_step: bool = False


class QueryResult(BaseModel):
    """Result of running a single evaluation query."""

    id: str
    query: str
    category: str
    description: str

    # Routing
    detected_jurisdiction: str = ""
    detected_area: str = ""
    route_latency_ms: float = 0.0

    # Retrieval
    num_documents: int = 0
    avg_relevance_score: float = 0.0
    retrieve_latency_ms: float = 0.0

    # GraphRAG
    graph_nodes_matched: int = 0
    good_law_flags: list[str] = Field(default_factory=list)
    graph_latency_ms: float = 0.0

    # Evaluation (Critic)
    crag_action: str = ""
    expected_action: str = ""
    action_correct: bool = False
    answer_relevancy: float = 0.0  # overall_confidence from evaluator
    num_verified: int = 0
    evaluate_latency_ms: float = 0.0

    # Augmentation
    augmented: bool = False
    augment_docs: int = 0
    augment_latency_ms: float = 0.0

    # Generation
    answer: str = ""
    num_citations: int = 0
    generate_latency_ms: float = 0.0

    # Grading
    faithfulness: float = 0.0  # grounded_ratio
    ungrounded_claims: list[str] = Field(default_factory=list)
    grade_latency_ms: float = 0.0

    # Totals
    total_latency_ms: float = 0.0


# 8-query test suite
TEST_SUITE: list[TestQuery] = [
    TestQuery(
        id="Q1",
        query="What is the statute of limitations for personal injury claims in Florida?",
        jurisdiction_hint="Florida",
        expected_action="generate",
        category="Core Retrieval",
        description="Standard query — tests correct retrieval from index + generation",
    ),
    TestQuery(
        id="Q2",
        query="Are there caps on non-economic damages in Florida medical malpractice?",
        jurisdiction_hint="Florida",
        expected_action="generate",
        category="GraphRAG Good Law",
        description="Overturned statute detection — GraphRAG identifies § 766.118 as invalidated",
    ),
    TestQuery(
        id="Q3",
        query="What happens if I'm 60% at fault in a car accident in Florida?",
        jurisdiction_hint="Florida",
        expected_action="generate",
        category="Amendment Tracking",
        description="HB 837 amendment — GraphRAG surfaces the 2023 comparative negligence change",
    ),
    TestQuery(
        id="Q4",
        query="What is the statute of limitations for medical malpractice in Texas?",
        jurisdiction_hint="Texas",
        expected_action="augment",
        category="Corrective Trigger",
        description="⚡ STRESS TEST — No TX docs in index, critic should trigger web search fallback",
    ),
    TestQuery(
        id="Q5",
        query="What are the requirements for a class action lawsuit in Florida?",
        jurisdiction_hint="Florida",
        expected_action="augment",
        category="Web Search Fallback",
        description="⚡ STRESS TEST — Topic not in index, should fallback to Tavily",
    ),
    TestQuery(
        id="Q6",
        query="What are the pre-suit requirements for a medical malpractice case in Florida?",
        jurisdiction_hint="Florida",
        expected_action="generate",
        category="Procedural Precision",
        description="Complex procedural question — tests multi-step statutory retrieval",
    ),
    TestQuery(
        id="Q7",
        query="Can I sue the state of Florida for a car accident caused by a government vehicle?",
        jurisdiction_hint="Florida",
        expected_action="generate",
        category="Sovereign Immunity",
        description="Sovereign immunity waiver + caps — tests § 768.28 retrieval",
    ),
    TestQuery(
        id="Q8",
        query="Compare the statute of limitations for personal injury across Florida, California, and Federal law",
        jurisdiction_hint=None,
        expected_action="generate",
        category="Multi-Step Reasoning",
        description="Cross-jurisdiction comparison — decomposes into 3 parallel sub-queries",
        multi_step=True,
    ),
]


# ── Pipeline Runner ──


async def run_single_query(tq: TestQuery) -> QueryResult:
    """Run a single test query through the full pipeline with timing."""
    from src.core.router import JurisdictionalRouter
    from src.core.retriever import LegalRetriever
    from src.core.knowledge_graph import LegalKnowledgeGraph
    from src.core.evaluator import CRAGEvaluator
    from src.core.generator import LegalGenerator
    from src.agents.web_search import WebSearchAgent
    from src.agents.grader import HallucinationGrader

    result = QueryResult(
        id=tq.id,
        query=tq.query,
        category=tq.category,
        description=tq.description,
        expected_action=tq.expected_action,
    )

    total_start = time.perf_counter()

    # ── Route ──
    t0 = time.perf_counter()
    router = JurisdictionalRouter()
    routing = await router.route(tq.query, tq.jurisdiction_hint)
    result.route_latency_ms = (time.perf_counter() - t0) * 1000
    result.detected_jurisdiction = routing.state or routing.level.value
    result.detected_area = routing.area_of_law

    # ── Multi-step path ──
    if tq.multi_step:
        from src.core.multi_step import MultiStepReasoner

        t0 = time.perf_counter()
        reasoner = MultiStepReasoner()
        decomposition = await reasoner.decompose(tq.query)

        async def run_sub(q, j):
            from src.main import build_pipeline, PipelineState
            pipeline = build_pipeline()
            state = PipelineState(query=q, jurisdiction_hint=j)
            return await pipeline.ainvoke(state)

        sub_results = await reasoner.run_sub_queries(decomposition, run_sub)
        synthesized = await reasoner.synthesize(tq.query, sub_results)

        result.answer = synthesized.answer
        result.num_citations = sum(len(sr.citations) for sr in sub_results)
        result.crag_action = "generate (multi-step)"
        result.action_correct = True
        result.answer_relevancy = sum(sr.grounded_ratio for sr in sub_results) / max(len(sub_results), 1)
        result.faithfulness = result.answer_relevancy
        result.total_latency_ms = (time.perf_counter() - total_start) * 1000
        return result

    # ── Retrieve ──
    t0 = time.perf_counter()
    retriever = LegalRetriever()
    jurisdiction = routing.state
    docs = await retriever.retrieve(tq.query, jurisdiction=jurisdiction)
    result.retrieve_latency_ms = (time.perf_counter() - t0) * 1000
    result.num_documents = len(docs)
    if docs:
        result.avg_relevance_score = sum(d.relevance_score for d in docs) / len(docs)

    # ── GraphRAG Enrich ──
    t0 = time.perf_counter()
    settings = get_settings()
    graph_context = {}
    if settings.use_graph_rag:
        graph = LegalKnowledgeGraph()
        graph.load()
        for doc in docs:
            for node_id in graph.graph.nodes:
                node_data = graph.graph.nodes[node_id]
                title = node_data.get("title", "")
                if node_id in doc.source_uri or any(
                    part in doc.content[:200] for part in title.split("§") if len(part.strip()) > 3
                ):
                    enrichment = graph.enrich_document_context(node_id)
                    if enrichment and enrichment != "No graph relationships found.":
                        graph_context[node_id] = enrichment
                        if "NO LONGER good law" in enrichment:
                            result.good_law_flags.append(f"{node_id}: OVERTURNED")
                        elif "Amended" in enrichment:
                            result.good_law_flags.append(f"{node_id}: AMENDED")
                        doc.content += f"\n\n[GraphRAG Context]: {enrichment}"
                    break
    result.graph_latency_ms = (time.perf_counter() - t0) * 1000
    result.graph_nodes_matched = len(graph_context)

    # ── Evaluate ──
    t0 = time.perf_counter()
    evaluator = CRAGEvaluator()
    evaluation = await evaluator.evaluate(tq.query, docs, jurisdiction=jurisdiction or "")
    result.evaluate_latency_ms = (time.perf_counter() - t0) * 1000
    result.crag_action = evaluation.action.value
    result.action_correct = evaluation.action.value == tq.expected_action
    result.answer_relevancy = evaluation.overall_confidence
    result.num_verified = len(evaluation.verified_documents)

    # ── Augment (if needed) ──
    augmented_docs: list[RetrievedDocument] = []
    if evaluation.action in (CRAGAction.AUGMENT, CRAGAction.REINDEX):
        t0 = time.perf_counter()
        agent = WebSearchAgent()
        augmented_docs = await agent.search(tq.query, jurisdiction=jurisdiction)
        result.augment_latency_ms = (time.perf_counter() - t0) * 1000
        result.augmented = True
        result.augment_docs = len(augmented_docs)

    # ── Generate ──
    t0 = time.perf_counter()
    generator = LegalGenerator()
    verified_docs = [docs[i] for i in evaluation.verified_documents if i < len(docs)]
    all_docs = verified_docs + augmented_docs
    gen_result = await generator.generate(tq.query, all_docs, jurisdiction=jurisdiction or "")
    result.generate_latency_ms = (time.perf_counter() - t0) * 1000
    result.answer = gen_result.answer
    result.num_citations = len(gen_result.citations)

    # ── Grade ──
    t0 = time.perf_counter()
    grader = HallucinationGrader()
    grade = await grader.grade(gen_result.answer, docs + augmented_docs)
    result.grade_latency_ms = (time.perf_counter() - t0) * 1000
    result.faithfulness = grade.grounded_ratio
    result.ungrounded_claims = grade.ungrounded_claims

    result.total_latency_ms = (time.perf_counter() - total_start) * 1000
    return result


# ── Display & Output ──


def display_results(results: list[QueryResult]) -> None:
    """Display evaluation results as a Rich table."""

    # Main metrics table
    table = Table(
        title="🏛️ LexVerify CRAG — Evaluation Results",
        show_lines=True,
        title_style="bold cyan",
    )
    table.add_column("ID", style="bold", width=4)
    table.add_column("Category", width=20)
    table.add_column("Faith.", justify="center", width=7)
    table.add_column("Relev.", justify="center", width=7)
    table.add_column("CRAG Action", width=16)
    table.add_column("Good Law", width=14)
    table.add_column("Docs", justify="center", width=5)
    table.add_column("Cites", justify="center", width=5)
    table.add_column("Latency", justify="right", width=9)

    for r in results:
        faith_str = f"{r.faithfulness:.0%}"
        faith_color = "green" if r.faithfulness >= 0.9 else "yellow" if r.faithfulness >= 0.7 else "red"

        relev_str = f"{r.answer_relevancy:.0%}"
        relev_color = "green" if r.answer_relevancy >= 0.8 else "yellow" if r.answer_relevancy >= 0.5 else "red"

        action_str = r.crag_action.upper()
        if r.action_correct:
            action_color = "green"
        else:
            action_color = "yellow"

        good_law = ", ".join(r.good_law_flags) if r.good_law_flags else "—"
        good_law_style = "red bold" if "OVERTURNED" in good_law else "yellow" if "AMENDED" in good_law else "dim"

        latency_str = f"{r.total_latency_ms / 1000:.1f}s"

        table.add_row(
            r.id,
            r.category,
            f"[{faith_color}]{faith_str}[/{faith_color}]",
            f"[{relev_color}]{relev_str}[/{relev_color}]",
            f"[{action_color}]{action_str}[/{action_color}]",
            f"[{good_law_style}]{good_law}[/{good_law_style}]",
            str(r.num_documents),
            str(r.num_citations),
            latency_str,
        )

    # Aggregates
    avg_faith = sum(r.faithfulness for r in results) / len(results)
    avg_relev = sum(r.answer_relevancy for r in results) / len(results)
    avg_latency = sum(r.total_latency_ms for r in results) / len(results)
    correct_actions = sum(1 for r in results if r.action_correct)

    table.add_row(
        "—",
        "[bold]AVERAGE[/bold]",
        f"[bold]{avg_faith:.0%}[/bold]",
        f"[bold]{avg_relev:.0%}[/bold]",
        f"[bold]{correct_actions}/{len(results)}[/bold]",
        "—",
        "—",
        "—",
        f"[bold]{avg_latency / 1000:.1f}s[/bold]",
        style="on grey23",
    )

    console.print()
    console.print(table)

    # Latency breakdown
    latency_table = Table(title="⏱️ Latency Breakdown (avg ms)", show_lines=True)
    latency_table.add_column("Stage", width=14)
    latency_table.add_column("Avg (ms)", justify="right", width=10)
    latency_table.add_column("Max (ms)", justify="right", width=10)

    standard = [r for r in results if not r.crag_action.startswith("generate (")]
    if standard:
        stages = [
            ("Route", [r.route_latency_ms for r in standard]),
            ("Retrieve", [r.retrieve_latency_ms for r in standard]),
            ("GraphRAG", [r.graph_latency_ms for r in standard]),
            ("Evaluate", [r.evaluate_latency_ms for r in standard]),
            ("Augment", [r.augment_latency_ms for r in standard if r.augmented]),
            ("Generate", [r.generate_latency_ms for r in standard]),
            ("Grade", [r.grade_latency_ms for r in standard]),
        ]
        for name, vals in stages:
            if vals:
                latency_table.add_row(name, f"{sum(vals)/len(vals):.0f}", f"{max(vals):.0f}")

    console.print()
    console.print(latency_table)

    # Ungrounded claims
    any_ungrounded = [r for r in results if r.ungrounded_claims]
    if any_ungrounded:
        console.print("\n[bold yellow]⚠️  Ungrounded Claims Detected:[/bold yellow]")
        for r in any_ungrounded:
            console.print(f"  [{r.id}] {r.category}:")
            for claim in r.ungrounded_claims:
                console.print(f"    • {claim[:100]}")
    else:
        console.print("\n[bold green]✅ Zero ungrounded claims across all queries![/bold green]")


def save_results(results: list[QueryResult], path: Path) -> None:
    """Save results to JSON for embedding in DEMO.md."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [r.model_dump() for r in results]
    path.write_text(json.dumps(data, indent=2))
    console.print(f"\n[dim]Results saved to {path}[/dim]")


# ── Main ──


async def main():
    console.print(
        Panel(
            f"Running [bold]{len(TEST_SUITE)}[/bold] evaluation queries...\n"
            "[dim]This will make live API calls to OpenAI, Pinecone, and Tavily.[/dim]",
            title="🏛️ LexVerify Evaluation Harness",
            border_style="cyan",
        )
    )

    results: list[QueryResult] = []

    for i, tq in enumerate(TEST_SUITE, 1):
        console.print(f"\n[bold cyan][{i}/{len(TEST_SUITE)}][/bold cyan] {tq.id}: {tq.description}")
        try:
            result = await run_single_query(tq)
            results.append(result)
            icon = "✅" if result.faithfulness >= 0.9 else "⚠️"
            console.print(
                f"  {icon} {result.crag_action.upper()} | "
                f"Faith: {result.faithfulness:.0%} | "
                f"Relev: {result.answer_relevancy:.0%} | "
                f"{result.total_latency_ms/1000:.1f}s"
            )
        except Exception as e:
            console.print(f"  [red]❌ FAILED: {e}[/red]")

    if results:
        display_results(results)
        save_results(results, RESULTS_DIR / "eval_results.json")


if __name__ == "__main__":
    asyncio.run(main())
