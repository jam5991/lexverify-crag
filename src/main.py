"""
LexVerify CLI — Main entry point for the CRAG pipeline.

Assembles the corrective RAG pipeline using LangGraph and provides
a CLI interface for running legal queries. Supports:
- GraphRAG-enhanced Good Law verification
- Distilled critic fast-pass evaluation (--fast)
- Multi-step cross-jurisdiction comparison (compare command)
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

import typer
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import get_settings
from src.core.evaluator import CRAGAction, EvaluationResult
from src.core.retriever import RetrievedDocument
from src.core.router import RoutingResult
from src.utils.logger import setup_logging

app = typer.Typer(
    name="lexverify",
    help="LexVerify: Corrective RAG for Legal Citation Integrity",
    add_completion=False,
)
console = Console()


# ── LangGraph Pipeline State ──


class PipelineState(BaseModel):
    """State passed through the LangGraph CRAG pipeline."""

    query: str = ""
    jurisdiction_hint: str | None = None
    use_fast_critic: bool = False

    # After routing
    routing: RoutingResult | None = None

    # After retrieval
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)

    # After graph enrichment
    graph_context: dict[str, str] = Field(
        default_factory=dict, description="Graph-derived context per document ID"
    )

    # After evaluation
    evaluation: EvaluationResult | None = None

    # After augmentation (web search)
    augmented_documents: list[RetrievedDocument] = Field(default_factory=list)

    # After generation
    answer: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)

    # After hallucination grading
    is_grounded: bool = False
    grounded_ratio: float = 0.0
    ungrounded_claims: list[str] = Field(default_factory=list)


# ── Pipeline Node Functions ──


async def route_query(state: PipelineState) -> dict:
    """Route the query to the appropriate jurisdiction."""
    from src.core.router import JurisdictionalRouter

    router = JurisdictionalRouter()
    result = await router.route(state.query, state.jurisdiction_hint)
    return {"routing": result}


async def retrieve_documents(state: PipelineState) -> dict:
    """Retrieve relevant legal documents from Pinecone."""
    from src.core.retriever import LegalRetriever

    retriever = LegalRetriever()
    jurisdiction = state.routing.state if state.routing else None
    docs = await retriever.retrieve(state.query, jurisdiction=jurisdiction)
    return {"retrieved_documents": docs}


async def enrich_with_graph(state: PipelineState) -> dict:
    """Enrich retrieved documents with knowledge graph context."""
    settings = get_settings()
    if not settings.use_graph_rag:
        return {"graph_context": {}}

    from src.core.knowledge_graph import LegalKnowledgeGraph

    graph = LegalKnowledgeGraph()
    graph.load()

    context = {}
    for doc in state.retrieved_documents:
        # Try to match document to graph node by source_uri or content
        for node_id in graph.graph.nodes:
            node_data = graph.graph.nodes[node_id]
            title = node_data.get("title", "")
            # Match by node ID pattern in document source URI
            if node_id in doc.source_uri or any(
                part in doc.content[:200] for part in title.split("§") if len(part.strip()) > 3
            ):
                enrichment = graph.enrich_document_context(node_id)
                if enrichment and enrichment != "No graph relationships found.":
                    context[node_id] = enrichment
                    doc.content += f"\n\n[GraphRAG Context]: {enrichment}"
                break

    return {"graph_context": context}


async def evaluate_documents(state: PipelineState) -> dict:
    """Evaluate retrieved documents with the CRAG critic."""
    settings = get_settings()
    jurisdiction = state.routing.state if state.routing else ""

    if state.use_fast_critic and settings.use_distilled_critic:
        from src.core.distilled_critic import DistilledCritic

        evaluator = DistilledCritic()
    else:
        from src.core.evaluator import CRAGEvaluator

        evaluator = CRAGEvaluator()

    result = await evaluator.evaluate(
        state.query, state.retrieved_documents, jurisdiction=jurisdiction or ""
    )
    return {"evaluation": result}


async def augment_with_search(state: PipelineState) -> dict:
    """Augment documents via web search fallback."""
    from src.agents.web_search import WebSearchAgent

    agent = WebSearchAgent()
    jurisdiction = state.routing.state if state.routing else None
    docs = await agent.search(state.query, jurisdiction=jurisdiction)
    return {"augmented_documents": docs}


async def generate_response(state: PipelineState) -> dict:
    """Generate the final response with citations."""
    from src.core.generator import LegalGenerator

    generator = LegalGenerator()

    # Use verified docs from evaluation + any augmented docs
    verified_docs = []
    if state.evaluation and state.evaluation.verified_documents:
        verified_docs = [
            state.retrieved_documents[i]
            for i in state.evaluation.verified_documents
            if i < len(state.retrieved_documents)
        ]

    # Add augmented documents
    all_docs = verified_docs + state.augmented_documents

    jurisdiction = state.routing.state if state.routing else ""
    response = await generator.generate(state.query, all_docs, jurisdiction=jurisdiction or "")

    return {
        "answer": response.answer,
        "citations": [c.model_dump() for c in response.citations],
    }


async def grade_response(state: PipelineState) -> dict:
    """Grade the response for hallucination."""
    from src.agents.grader import HallucinationGrader

    grader = HallucinationGrader()

    # Combine all source documents
    all_docs = state.retrieved_documents + state.augmented_documents
    result = await grader.grade(state.answer, all_docs)

    return {
        "is_grounded": result.is_grounded,
        "grounded_ratio": result.grounded_ratio,
        "ungrounded_claims": result.ungrounded_claims,
    }


def should_augment(state: PipelineState) -> str:
    """Conditional edge: decide whether to augment or generate."""
    if state.evaluation is None:
        return "augment"
    if state.evaluation.action == CRAGAction.AUGMENT:
        return "augment"
    if state.evaluation.action == CRAGAction.REINDEX:
        return "augment"  # Also try web search for reindex cases
    return "generate"


def build_pipeline():
    """Build the LangGraph CRAG pipeline with GraphRAG enrichment."""
    from langgraph.graph import END, StateGraph

    workflow = StateGraph(PipelineState)

    # Add nodes
    workflow.add_node("route", route_query)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("graph_enrich", enrich_with_graph)
    workflow.add_node("evaluate", evaluate_documents)
    workflow.add_node("augment", augment_with_search)
    workflow.add_node("generate", generate_response)
    workflow.add_node("grade", grade_response)

    # Define edges
    workflow.set_entry_point("route")
    workflow.add_edge("route", "retrieve")
    workflow.add_edge("retrieve", "graph_enrich")
    workflow.add_edge("graph_enrich", "evaluate")

    # Conditional: evaluate → augment or generate
    workflow.add_conditional_edges(
        "evaluate",
        should_augment,
        {"augment": "augment", "generate": "generate"},
    )
    workflow.add_edge("augment", "generate")
    workflow.add_edge("generate", "grade")
    workflow.add_edge("grade", END)

    return workflow.compile()


# ── Display Helpers ──


def display_result(result: dict) -> None:
    """Display pipeline results with Rich formatting."""
    if result.get("answer"):
        console.print()
        console.print(Panel(Markdown(result["answer"]), title="📋 Legal Response", border_style="green"))

        if result.get("citations"):
            console.print("\n[bold]📚 Citations:[/bold]")
            for i, cit in enumerate(result["citations"], 1):
                console.print(f"  [{i}] {cit.get('source_uri', 'N/A')} — {cit.get('claim', '')[:80]}")

        # Graph context
        if result.get("graph_context"):
            console.print("\n[bold]🔗 Knowledge Graph Context:[/bold]")
            for node_id, ctx in result["graph_context"].items():
                console.print(f"  • [dim]{node_id}[/dim]: {ctx}")

        if result.get("is_grounded") is not None:
            grounded = result["is_grounded"]
            ratio = result.get("grounded_ratio", 0)
            icon = "✅" if grounded else "⚠️"
            color = "green" if grounded else "yellow"
            console.print(
                f"\n{icon} [bold {color}]Grounding Score:[/bold {color}] "
                f"{ratio:.0%} of claims verified"
            )

            if result.get("ungrounded_claims"):
                console.print("\n[bold yellow]⚠️  Ungrounded Claims:[/bold yellow]")
                for claim in result["ungrounded_claims"]:
                    console.print(f"  • {claim}")
    else:
        console.print("[yellow]No response generated.[/yellow]")


# ── CLI Commands ──


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Legal question to answer")],
    jurisdiction: Annotated[
        str | None,
        typer.Option("--jurisdiction", "-j", help="Jurisdiction hint (e.g., 'Florida')"),
    ] = None,
    fast: Annotated[
        bool, typer.Option("--fast", "-f", help="Use distilled critic for faster evaluation")
    ] = False,
    multi_step: Annotated[
        bool, typer.Option("--multi-step", "-m", help="Force multi-step reasoning")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """Run a legal query through the LexVerify CRAG pipeline."""
    setup_logging(level="DEBUG" if verbose else "INFO")
    settings = get_settings()

    # Validate API keys
    if not settings.openai_api_key or settings.openai_api_key.startswith("sk-your"):
        console.print(
            Panel(
                "[red]Missing OpenAI API key.[/red]\n\n"
                "Copy .env.example to .env and fill in your API keys:\n"
                "[dim]cp .env.example .env[/dim]",
                title="⚠️  Configuration Error",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    mode_tags = []
    if settings.use_graph_rag:
        mode_tags.append("GraphRAG")
    if fast:
        mode_tags.append("Fast Critic")
    if multi_step:
        mode_tags.append("Multi-Step")
    mode_str = " + ".join(mode_tags) if mode_tags else "Standard"

    console.print(
        Panel(
            f"[bold cyan]Query:[/bold cyan] {question}\n"
            f"[bold cyan]Jurisdiction:[/bold cyan] {jurisdiction or 'Auto-detect'}\n"
            f"[bold cyan]Mode:[/bold cyan] {mode_str}",
            title="🔍 LexVerify CRAG Pipeline",
            border_style="cyan",
        )
    )

    async def _run():
        # Check if multi-step reasoning is needed
        if multi_step:
            return await _run_multi_step(question, jurisdiction)

        pipeline = build_pipeline()
        initial_state = PipelineState(
            query=question,
            jurisdiction_hint=jurisdiction,
            use_fast_critic=fast,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running CRAG pipeline...", total=None)
            result = await pipeline.ainvoke(initial_state)
            progress.update(task, description="Complete!")

        return result

    result = asyncio.run(_run())
    display_result(result)


@app.command()
def compare(
    question: Annotated[str, typer.Argument(help="Comparative legal question")],
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable verbose logging")
    ] = False,
) -> None:
    """Run a multi-step comparative legal analysis across jurisdictions."""
    setup_logging(level="DEBUG" if verbose else "INFO")
    settings = get_settings()

    if not settings.openai_api_key or settings.openai_api_key.startswith("sk-your"):
        console.print(Panel("[red]Missing OpenAI API key.[/red]", title="⚠️  Error", border_style="red"))
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold cyan]Query:[/bold cyan] {question}\n"
            f"[bold cyan]Mode:[/bold cyan] Multi-Step Comparative Analysis",
            title="⚖️  LexVerify Multi-Step Reasoning",
            border_style="cyan",
        )
    )

    async def _run():
        return await _run_multi_step(question, jurisdiction=None)

    result = asyncio.run(_run())
    display_result(result)


async def _run_multi_step(question: str, jurisdiction: str | None) -> dict:
    """Execute the multi-step reasoning pipeline."""
    from src.core.multi_step import MultiStepReasoner

    reasoner = MultiStepReasoner()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Decomposing query...", total=None)

        # Step 1: Decompose
        decomposition = await reasoner.decompose(question)

        if not decomposition.is_multi_step or not decomposition.sub_queries:
            # Not actually multi-step — run as single query
            progress.update(task, description="Single-step query, running standard pipeline...")
            pipeline = build_pipeline()
            initial_state = PipelineState(
                query=question,
                jurisdiction_hint=jurisdiction,
            )
            result = await pipeline.ainvoke(initial_state)
            progress.update(task, description="Complete!")
            return result

        console.print(f"\n[dim]Decomposed into {len(decomposition.sub_queries)} sub-queries:[/dim]")
        for i, sq in enumerate(decomposition.sub_queries, 1):
            console.print(f"  [dim]{i}. [{sq.jurisdiction}] {sq.query}[/dim]")

        # Step 2: Run sub-queries in parallel
        progress.update(
            task, description=f"Running {len(decomposition.sub_queries)} sub-queries in parallel..."
        )

        async def run_single(q: str, j: str) -> dict:
            pipeline = build_pipeline()
            state = PipelineState(query=q, jurisdiction_hint=j)
            return await pipeline.ainvoke(state)

        sub_results = await reasoner.run_sub_queries(decomposition, run_single)

        # Step 3: Synthesize
        progress.update(task, description="Synthesizing comparative analysis...")
        synthesized = await reasoner.synthesize(question, sub_results)

        progress.update(task, description="Complete!")

    return {
        "answer": synthesized.answer,
        "citations": [],
        "is_grounded": all(sr.is_grounded for sr in sub_results),
        "grounded_ratio": (
            sum(sr.grounded_ratio for sr in sub_results) / len(sub_results)
            if sub_results
            else 0.0
        ),
        "ungrounded_claims": [],
        "graph_context": {},
    }


@app.command()
def info() -> None:
    """Display LexVerify configuration info."""
    settings = get_settings()
    console.print(
        Panel(
            f"[bold]Generator Model:[/bold] {settings.generator_model}\n"
            f"[bold]Evaluator Model:[/bold] {settings.evaluator_model}\n"
            f"[bold]Embedding Model:[/bold] {settings.embedding_model}\n"
            f"[bold]Top-K Retrieval:[/bold] {settings.top_k}\n"
            f"[bold]Confidence High:[/bold] {settings.confidence_threshold_high}\n"
            f"[bold]Confidence Low:[/bold] {settings.confidence_threshold_low}\n"
            f"[bold]Pinecone Index:[/bold] {settings.pinecone_index_name}\n"
            f"[bold]GraphRAG:[/bold] {'✅ Enabled' if settings.use_graph_rag else '❌ Disabled'}\n"
            f"[bold]Distilled Critic:[/bold] {'✅ ' + settings.distilled_critic_model if settings.use_distilled_critic else '❌ Disabled'}\n"
            f"[bold]OpenAI Key:[/bold] {'✅ Set' if settings.openai_api_key else '❌ Missing'}\n"
            f"[bold]Tavily Key:[/bold] {'✅ Set' if settings.tavily_api_key else '❌ Missing'}\n"
            f"[bold]Pinecone Key:[/bold] {'✅ Set' if settings.pinecone_api_key else '❌ Missing'}",
            title="⚙️  LexVerify Configuration",
            border_style="blue",
        )
    )


if __name__ == "__main__":
    app()
