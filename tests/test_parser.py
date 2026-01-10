"""
Unit tests for L2 Parsing Layer (policy_parser_agent.py)
"""

import pytest
from unittest.mock import MagicMock, patch

from aegislang.agents.policy_parser_agent import (
    PolicyParserAgent,
    ParsedClause,
    ParsedClauseCollection,
    ClauseType,
    ActorEntity,
    ActionPhrase,
    MockLLMClient,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_parser():
    """Create parser with mock LLM client."""
    return PolicyParserAgent(use_mock=True)


@pytest.fixture
def sample_obligation_clause():
    """Sample obligation clause text."""
    return "Financial institutions must verify customer identity before account creation."


@pytest.fixture
def sample_prohibition_clause():
    """Sample prohibition clause text."""
    return "Employees shall not access customer data without authorization."


@pytest.fixture
def sample_conditional_clause():
    """Sample conditional clause text."""
    return "If a transaction exceeds $10,000, the bank must file a report within 24 hours."


@pytest.fixture
def sample_ingested_document():
    """Sample ingested document structure."""
    return {
        "doc_id": "TEST_DOC_001",
        "metadata": {
            "source_file": "test.md",
            "document_type": "markdown",
        },
        "sections": [
            {
                "section_id": "TEST_DOC_001_S001",
                "section_title": "Requirements",
                "text_chunks": [
                    {
                        "chunk_id": "TEST_DOC_001_S001_C001",
                        "text": "Financial institutions must verify customer identity. Records must be maintained for 5 years.",
                        "token_count": 15,
                    }
                ],
            }
        ],
    }


# =============================================================================
# MockLLMClient Tests
# =============================================================================

class TestMockLLMClient:
    """Tests for MockLLMClient."""

    def test_parse_obligation(self):
        """Test parsing obligation clause."""
        client = MockLLMClient()
        result = client.parse_clause(
            "Financial institutions must verify customer identity."
        )

        assert result["type"] == "obligation"
        assert "actor" in result
        assert "action" in result
        assert "confidence" in result

    def test_parse_prohibition(self):
        """Test parsing prohibition clause."""
        client = MockLLMClient()
        result = client.parse_clause(
            "Employees shall not access customer data without authorization."
        )

        assert result["type"] == "prohibition"

    def test_parse_permission(self):
        """Test parsing permission clause."""
        client = MockLLMClient()
        result = client.parse_clause(
            "Customers may request data deletion at any time."
        )

        assert result["type"] == "permission"

    def test_parse_conditional(self):
        """Test parsing conditional clause."""
        client = MockLLMClient()
        result = client.parse_clause(
            "If the amount exceeds $10,000, a report must be filed."
        )

        assert result["type"] == "conditional"

    def test_parse_definition(self):
        """Test parsing definition clause."""
        client = MockLLMClient()
        result = client.parse_clause(
            "Customer means any individual who holds an account."
        )

        assert result["type"] == "definition"

    def test_extract_actor(self):
        """Test actor extraction."""
        client = MockLLMClient()
        result = client.parse_clause(
            "Financial institutions must report suspicious activity."
        )

        assert result["actor"]["entity"]
        assert isinstance(result["actor"]["entity"], str)

    def test_extract_condition(self):
        """Test condition extraction."""
        client = MockLLMClient()
        result = client.parse_clause(
            "Before account creation, identity must be verified."
        )

        assert result.get("condition") is not None or result["type"] == "obligation"

    def test_extract_temporal(self):
        """Test temporal scope extraction."""
        client = MockLLMClient()
        result = client.parse_clause(
            "Records must be maintained for at least 5 years."
        )

        temporal = result.get("temporal_scope")
        if temporal:
            assert temporal.get("duration") or temporal.get("deadline") or temporal.get("frequency")


# =============================================================================
# PolicyParserAgent Tests
# =============================================================================

class TestPolicyParserAgent:
    """Tests for PolicyParserAgent."""

    def test_initialization_mock(self):
        """Test initialization with mock client."""
        parser = PolicyParserAgent(use_mock=True)
        assert parser.llm_client is not None

    def test_initialization_anthropic(self):
        """Test initialization attempt with Anthropic (will fail without key)."""
        with pytest.raises((ValueError, ImportError)):
            PolicyParserAgent(llm_provider="anthropic")

    def test_parse_clause_returns_parsed_clause(self, mock_parser, sample_obligation_clause):
        """Test that parse_clause returns ParsedClause."""
        result = mock_parser.parse_clause(
            clause_text=sample_obligation_clause,
            clause_id="TEST_CL001",
            source_chunk_id="TEST_C001",
        )

        assert isinstance(result, ParsedClause)
        assert result.clause_id == "TEST_CL001"
        assert result.source_chunk_id == "TEST_C001"
        assert result.source_text == sample_obligation_clause

    def test_parse_clause_type_detection(self, mock_parser):
        """Test clause type detection."""
        obligation = mock_parser.parse_clause(
            "Banks must report transactions.",
            "CL001", "C001"
        )
        assert obligation.type == ClauseType.OBLIGATION

        prohibition = mock_parser.parse_clause(
            "Staff shall not share passwords.",
            "CL002", "C002"
        )
        assert prohibition.type == ClauseType.PROHIBITION

    def test_parse_clause_has_confidence(self, mock_parser, sample_obligation_clause):
        """Test that parsed clause has confidence score."""
        result = mock_parser.parse_clause(
            sample_obligation_clause, "CL001", "C001"
        )

        assert 0.0 <= result.confidence <= 1.0

    def test_parse_clause_has_actor(self, mock_parser, sample_obligation_clause):
        """Test that parsed clause has actor."""
        result = mock_parser.parse_clause(
            sample_obligation_clause, "CL001", "C001"
        )

        assert result.actor is not None
        assert isinstance(result.actor, ActorEntity)
        assert result.actor.entity

    def test_parse_clause_has_action(self, mock_parser, sample_obligation_clause):
        """Test that parsed clause has action."""
        result = mock_parser.parse_clause(
            sample_obligation_clause, "CL001", "C001"
        )

        assert result.action is not None
        assert isinstance(result.action, ActionPhrase)
        assert result.action.verb

    def test_parse_text_chunk(self, mock_parser):
        """Test parsing a text chunk with multiple clauses."""
        chunk_text = """
        Financial institutions must verify identity.
        Banks shall not process unauthorized transactions.
        """

        results = mock_parser.parse_text_chunk(
            chunk_text=chunk_text,
            chunk_id="C001",
            doc_id="DOC001",
        )

        assert len(results) >= 1
        for result in results:
            assert isinstance(result, ParsedClause)

    def test_parse_ingested_document(self, mock_parser, sample_ingested_document):
        """Test parsing an ingested document."""
        result = mock_parser.parse_ingested_document(sample_ingested_document)

        assert isinstance(result, ParsedClauseCollection)
        assert result.doc_id == "TEST_DOC_001"
        assert result.parse_timestamp
        assert len(result.clauses) >= 1

    def test_split_into_clauses(self, mock_parser):
        """Test clause splitting logic."""
        text = "Banks must verify identity. Customers may request data."
        clauses = mock_parser._split_into_clauses(text)

        assert len(clauses) >= 1
        for clause in clauses:
            assert clause.strip()


# =============================================================================
# ParsedClause Model Tests
# =============================================================================

class TestParsedClauseModel:
    """Tests for ParsedClause Pydantic model."""

    def test_valid_parsed_clause(self):
        """Test creating a valid ParsedClause."""
        clause = ParsedClause(
            clause_id="TEST_CL001",
            source_chunk_id="TEST_C001",
            source_text="Test clause text",
            type=ClauseType.OBLIGATION,
            actor=ActorEntity(entity="test entity"),
            action=ActionPhrase(verb="test"),
            confidence=0.85,
        )

        assert clause.clause_id == "TEST_CL001"
        assert clause.type == ClauseType.OBLIGATION
        assert clause.confidence == 0.85

    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence
        clause = ParsedClause(
            clause_id="CL001",
            source_chunk_id="C001",
            source_text="Test",
            type=ClauseType.OBLIGATION,
            actor=ActorEntity(entity="entity"),
            action=ActionPhrase(verb="act"),
            confidence=0.999,
        )
        assert clause.confidence == 1.0  # Rounded to 2 decimals

        # Invalid confidence (out of range)
        with pytest.raises(ValueError):
            ParsedClause(
                clause_id="CL001",
                source_chunk_id="C001",
                source_text="Test",
                type=ClauseType.OBLIGATION,
                actor=ActorEntity(entity="entity"),
                action=ActionPhrase(verb="act"),
                confidence=1.5,
            )

    def test_optional_fields(self):
        """Test optional fields."""
        clause = ParsedClause(
            clause_id="CL001",
            source_chunk_id="C001",
            source_text="Test",
            type=ClauseType.OBLIGATION,
            actor=ActorEntity(entity="entity"),
            action=ActionPhrase(verb="act"),
            confidence=0.8,
        )

        assert clause.object is None
        assert clause.condition is None
        assert clause.temporal_scope is None
        assert clause.cross_references == []

    def test_clause_type_enum(self):
        """Test ClauseType enum values."""
        assert ClauseType.OBLIGATION.value == "obligation"
        assert ClauseType.PROHIBITION.value == "prohibition"
        assert ClauseType.PERMISSION.value == "permission"
        assert ClauseType.CONDITIONAL.value == "conditional"
        assert ClauseType.DEFINITION.value == "definition"
        assert ClauseType.EXCEPTION.value == "exception"


# =============================================================================
# Integration Tests
# =============================================================================

class TestParserIntegration:
    """Integration tests for the parser."""

    def test_full_parsing_pipeline(self, mock_parser, sample_ingested_document):
        """Test full parsing pipeline."""
        result = mock_parser.parse_ingested_document(sample_ingested_document)

        # Verify structure
        assert result.doc_id
        assert result.clauses

        # Verify each clause
        for clause in result.clauses:
            assert clause.clause_id
            assert clause.type
            assert clause.actor
            assert clause.action
            assert 0 <= clause.confidence <= 1

    def test_regulatory_clause_examples(self, mock_parser):
        """Test with various regulatory clause examples."""
        examples = [
            ("Banks must maintain capital ratios above 8%.", ClauseType.OBLIGATION),
            ("Institutions shall not engage in insider trading.", ClauseType.PROHIBITION),
            ("Customers may opt out of data sharing.", ClauseType.PERMISSION),
        ]

        for text, expected_type in examples:
            result = mock_parser.parse_clause(text, "CL", "C")
            assert result.type == expected_type, f"Failed for: {text}"
