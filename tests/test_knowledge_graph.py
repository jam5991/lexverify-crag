"""Tests for the Legal Knowledge Graph (GraphRAG integration)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.core.knowledge_graph import (
    GraphNode,
    GraphQueryResult,
    GraphRelationship,
    LegalKnowledgeGraph,
    RelationshipType,
)


@pytest.fixture
def sample_graph_data(tmp_path: Path) -> Path:
    """Create a sample graph data file for testing."""
    data = {
        "nodes": [
            {"id": "statute-a", "title": "Statute A", "jurisdiction": "Florida", "doc_type": "statute", "year": 2020, "is_active": True},
            {"id": "statute-b", "title": "Statute B (Superseded)", "jurisdiction": "Florida", "doc_type": "statute", "year": 2010, "is_active": False},
            {"id": "case-1", "title": "Case One v. State", "jurisdiction": "Florida", "doc_type": "case_law", "year": 2022, "is_active": True},
            {"id": "case-2", "title": "Case Two v. State", "jurisdiction": "Florida", "doc_type": "case_law", "year": 2023, "is_active": True},
        ],
        "edges": [
            {"source": "statute-b", "target": "case-1", "relationship": "overturned_by", "description": "Case One struck down Statute B", "year": 2022},
            {"source": "statute-a", "target": "case-2", "relationship": "affirmed_by", "description": "Case Two affirmed Statute A", "year": 2023},
            {"source": "case-2", "target": "case-1", "relationship": "cites", "description": "Case Two cited Case One", "year": 2023},
            {"source": "statute-a", "target": "statute-b", "relationship": "amended_by", "description": "Statute B was amended by Statute A", "year": 2020},
        ],
    }
    path = tmp_path / "test_graph.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def graph(sample_graph_data: Path) -> LegalKnowledgeGraph:
    """Create and load a knowledge graph from sample data."""
    g = LegalKnowledgeGraph(graph_data_path=sample_graph_data)
    g.load()
    return g


class TestGraphNodes:
    def test_node_creation(self):
        node = GraphNode(id="test", title="Test Node", jurisdiction="FL")
        assert node.id == "test"
        assert node.is_active is True

    def test_inactive_node(self):
        node = GraphNode(id="old", title="Old", is_active=False)
        assert node.is_active is False


class TestGraphRelationships:
    def test_relationship_creation(self):
        rel = GraphRelationship(
            source="a", target="b",
            relationship=RelationshipType.OVERTURNED_BY,
            description="B overturned A",
        )
        assert rel.relationship == RelationshipType.OVERTURNED_BY


class TestLegalKnowledgeGraph:
    def test_load_graph(self, graph: LegalKnowledgeGraph):
        assert graph.graph.number_of_nodes() == 4
        assert graph.graph.number_of_edges() == 4

    def test_is_good_law_active_statute(self, graph: LegalKnowledgeGraph):
        assert graph.is_good_law("statute-a") is True

    def test_is_good_law_overturned(self, graph: LegalKnowledgeGraph):
        # statute-b has an overturned_by edge AND is_active=False
        assert graph.is_good_law("statute-b") is False

    def test_is_good_law_unknown_doc(self, graph: LegalKnowledgeGraph):
        assert graph.is_good_law("nonexistent") is True

    def test_query_relationships(self, graph: LegalKnowledgeGraph):
        result = graph.query_relationships("statute-b")
        assert result.is_good_law is False
        assert len(result.overturned_by) > 0

    def test_query_relationships_unknown(self, graph: LegalKnowledgeGraph):
        result = graph.query_relationships("nonexistent")
        assert result.is_good_law is True
        assert "not found" in result.context

    def test_citation_chain(self, graph: LegalKnowledgeGraph):
        chain = graph.get_citation_chain("case-2")
        assert len(chain) >= 1
        assert chain[0] == "Case Two v. State"

    def test_enrich_document_context(self, graph: LegalKnowledgeGraph):
        context = graph.enrich_document_context("statute-b")
        assert "NO LONGER good law" in context

    def test_enrich_active_document(self, graph: LegalKnowledgeGraph):
        context = graph.enrich_document_context("statute-a")
        # Active statute with affirmed_by — should have some context
        assert isinstance(context, str)

    def test_empty_graph(self, tmp_path: Path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"nodes": [], "edges": []}))
        g = LegalKnowledgeGraph(graph_data_path=path)
        g.load()
        assert g.graph.number_of_nodes() == 0
        assert g.is_good_law("anything") is True
