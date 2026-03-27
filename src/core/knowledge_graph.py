"""
Legal Knowledge Graph.

NetworkX-based knowledge graph capturing legal relationships between
cases and statutes: OVERTURNED_BY, AFFIRMED_BY, AMENDED_BY, CITES,
SUPERSEDED_BY. Enables graph-based Good Law verification for the
CRAG evaluator.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path

import networkx as nx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default graph data file
DEFAULT_GRAPH_DATA = Path(__file__).parent.parent.parent / "data" / "processed" / "legal_graph_data.json"


class RelationshipType(str, Enum):
    """Types of legal relationships between documents."""

    OVERTURNED_BY = "overturned_by"
    AFFIRMED_BY = "affirmed_by"
    AMENDED_BY = "amended_by"
    SUPERSEDED_BY = "superseded_by"
    CITES = "cites"
    MODIFIES = "modifies"


class GraphNode(BaseModel):
    """A node in the legal knowledge graph."""

    id: str
    title: str = ""
    jurisdiction: str = ""
    doc_type: str = ""
    year: int | None = None
    is_active: bool = True  # False if overturned/superseded


class GraphRelationship(BaseModel):
    """An edge in the legal knowledge graph."""

    source: str = Field(description="Source document ID")
    target: str = Field(description="Target document ID")
    relationship: RelationshipType
    description: str = Field(default="", description="Brief description of the relationship")
    year: int | None = Field(default=None, description="Year the relationship was established")


class GraphQueryResult(BaseModel):
    """Result of a graph query."""

    is_good_law: bool = Field(description="Whether the document is still valid law")
    relationships: list[dict] = Field(default_factory=list, description="Related documents")
    overturned_by: list[str] = Field(default_factory=list, description="Cases that overturned this")
    amended_by: list[str] = Field(default_factory=list, description="Laws that amended this")
    citation_chain: list[str] = Field(default_factory=list, description="Full citation lineage")
    context: str = Field(default="", description="Narrative context from graph relationships")


class LegalKnowledgeGraph:
    """NetworkX-based knowledge graph for legal document relationships.

    Captures OVERTURNED_BY, AFFIRMED_BY, AMENDED_BY, CITES relationships
    to enable graph-based Good Law verification in the CRAG evaluator.
    """

    def __init__(self, graph_data_path: Path | str | None = None) -> None:
        self.graph = nx.DiGraph()
        self._loaded = False
        self._graph_data_path = Path(graph_data_path) if graph_data_path else DEFAULT_GRAPH_DATA

    def load(self, graph_data_path: Path | str | None = None) -> None:
        """Load graph data from a JSON file.

        Args:
            graph_data_path: Path to the graph data JSON file.
        """
        path = Path(graph_data_path) if graph_data_path else self._graph_data_path
        if not path.exists():
            logger.warning("Graph data file not found: %s", path)
            return

        with open(path) as f:
            data = json.load(f)

        # Add nodes
        for node_data in data.get("nodes", []):
            node = GraphNode(**node_data)
            self.graph.add_node(
                node.id,
                title=node.title,
                jurisdiction=node.jurisdiction,
                doc_type=node.doc_type,
                year=node.year,
                is_active=node.is_active,
            )

        # Add edges
        for edge_data in data.get("edges", []):
            rel = GraphRelationship(**edge_data)
            self.graph.add_edge(
                rel.source,
                rel.target,
                relationship=rel.relationship,
                description=rel.description,
                year=rel.year,
            )

        self._loaded = True
        logger.info(
            "Loaded knowledge graph: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def _ensure_loaded(self) -> None:
        """Lazy-load the graph if not already loaded."""
        if not self._loaded:
            self.load()

    def is_good_law(self, doc_id: str) -> bool:
        """Check if a document is still valid law by traversing the graph.

        A document is NOT good law if:
        - It has an outgoing OVERTURNED_BY edge
        - It has an outgoing SUPERSEDED_BY edge
        - Its node is_active flag is False

        Args:
            doc_id: Document identifier.

        Returns:
            True if the document is still good law.
        """
        self._ensure_loaded()

        if doc_id not in self.graph:
            return True  # Unknown documents assumed valid

        node_data = self.graph.nodes[doc_id]
        if not node_data.get("is_active", True):
            return False

        # Check for overturning/superseding edges
        for _, target, data in self.graph.out_edges(doc_id, data=True):
            rel = data.get("relationship")
            if rel in (RelationshipType.OVERTURNED_BY, RelationshipType.SUPERSEDED_BY):
                return False

        return True

    def query_relationships(self, doc_id: str) -> GraphQueryResult:
        """Query all relationships for a document.

        Args:
            doc_id: Document identifier.

        Returns:
            GraphQueryResult with relationship details.
        """
        self._ensure_loaded()

        if doc_id not in self.graph:
            return GraphQueryResult(
                is_good_law=True,
                context=f"Document '{doc_id}' not found in knowledge graph.",
            )

        relationships = []
        overturned_by = []
        amended_by = []

        # Outgoing edges (this doc → related doc)
        for _, target, data in self.graph.out_edges(doc_id, data=True):
            rel_type = data.get("relationship", "")
            target_data = self.graph.nodes.get(target, {})
            relationships.append({
                "target_id": target,
                "target_title": target_data.get("title", target),
                "relationship": rel_type.value if isinstance(rel_type, RelationshipType) else str(rel_type),
                "direction": "outgoing",
                "description": data.get("description", ""),
            })
            if rel_type == RelationshipType.OVERTURNED_BY:
                overturned_by.append(target_data.get("title", target))
            if rel_type in (RelationshipType.AMENDED_BY, RelationshipType.MODIFIES):
                amended_by.append(target_data.get("title", target))

        # Incoming edges (related doc → this doc)
        for source, _, data in self.graph.in_edges(doc_id, data=True):
            rel_type = data.get("relationship", "")
            source_data = self.graph.nodes.get(source, {})
            relationships.append({
                "source_id": source,
                "source_title": source_data.get("title", source),
                "relationship": rel_type.value if isinstance(rel_type, RelationshipType) else str(rel_type),
                "direction": "incoming",
                "description": data.get("description", ""),
            })

        good_law = self.is_good_law(doc_id)
        citation_chain = self.get_citation_chain(doc_id)

        # Build narrative context
        context_parts = []
        if not good_law:
            context_parts.append(f"⚠️ This document is NO LONGER good law.")
        if overturned_by:
            context_parts.append(f"Overturned by: {', '.join(overturned_by)}.")
        if amended_by:
            context_parts.append(f"Amended/modified by: {', '.join(amended_by)}.")
        if citation_chain:
            context_parts.append(f"Citation chain: {' → '.join(citation_chain)}.")

        return GraphQueryResult(
            is_good_law=good_law,
            relationships=relationships,
            overturned_by=overturned_by,
            amended_by=amended_by,
            citation_chain=citation_chain,
            context=" ".join(context_parts) if context_parts else "No graph relationships found.",
        )

    def get_citation_chain(self, doc_id: str, max_depth: int = 5) -> list[str]:
        """Traverse the citation chain for a document.

        Follows CITES edges to build the citation lineage.

        Args:
            doc_id: Starting document.
            max_depth: Maximum traversal depth.

        Returns:
            List of document titles in the citation chain.
        """
        self._ensure_loaded()

        chain = []
        visited = set()
        current = doc_id

        for _ in range(max_depth):
            if current in visited or current not in self.graph:
                break
            visited.add(current)

            node_data = self.graph.nodes[current]
            chain.append(node_data.get("title", current))

            # Follow CITES edges
            cites_targets = [
                target
                for _, target, data in self.graph.out_edges(current, data=True)
                if data.get("relationship") == RelationshipType.CITES
            ]

            if not cites_targets:
                break
            current = cites_targets[0]  # Follow first citation

        return chain

    def enrich_document_context(self, doc_id: str) -> str:
        """Generate enrichment context for a retrieved document.

        Used by the pipeline to add graph context before evaluation.

        Args:
            doc_id: Document identifier (matched against graph node IDs).

        Returns:
            Context string describing the document's graph relationships.
        """
        result = self.query_relationships(doc_id)
        return result.context
