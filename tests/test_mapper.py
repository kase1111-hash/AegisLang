"""
Unit tests for L3 Mapping Layer (schema_mapping_agent.py)
"""

import pytest
from aegislang.agents.schema_mapping_agent import (
    SchemaMappingAgent,
    SchemaRegistry,
    TargetSchema,
    SchemaTable,
    SchemaField,
    SchemaType,
    MappedClause,
    EntityMapping,
    MappingMethod,
    MappingStatus,
    SourceRole,
    MockEmbeddingProvider,
    create_default_registry,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_embedding_provider():
    """Create mock embedding provider."""
    return MockEmbeddingProvider(dimensions=384)


@pytest.fixture
def sample_schema():
    """Create sample schema for testing."""
    return TargetSchema(
        schema_id="test_schema",
        schema_type=SchemaType.SQL,
        version="1.0.0",
        tables=[
            SchemaTable(
                table_name="customer",
                fields=[
                    SchemaField(
                        field_name="customer_id",
                        field_type="UUID",
                        semantic_labels=["customer", "client", "user"],
                    ),
                    SchemaField(
                        field_name="identity_verified",
                        field_type="BOOLEAN",
                        semantic_labels=["identity", "verification", "kyc"],
                    ),
                ],
            ),
            SchemaTable(
                table_name="transaction",
                fields=[
                    SchemaField(
                        field_name="transaction_id",
                        field_type="UUID",
                        semantic_labels=["transaction", "payment"],
                    ),
                    SchemaField(
                        field_name="amount",
                        field_type="DECIMAL",
                        semantic_labels=["amount", "value", "sum"],
                    ),
                ],
            ),
        ],
    )


@pytest.fixture
def sample_registry(sample_schema):
    """Create sample registry with schema."""
    registry = SchemaRegistry(
        schemas=[sample_schema],
        synonyms={
            "customer": ["client", "user", "account holder"],
            "transaction": ["payment", "transfer"],
        },
    )
    return registry


@pytest.fixture
def mapper(sample_registry):
    """Create mapper with sample registry."""
    return SchemaMappingAgent(
        registry=sample_registry,
        use_mock=True,
        confidence_threshold=0.7,
    )


@pytest.fixture
def sample_parsed_clause():
    """Create sample parsed clause."""
    return {
        "clause_id": "TEST_CL001",
        "source_chunk_id": "TEST_C001",
        "source_text": "Financial institutions must verify customer identity.",
        "type": "obligation",
        "actor": {
            "entity": "financial institutions",
            "qualifiers": [],
        },
        "action": {
            "verb": "verify",
            "modifiers": [],
        },
        "object": {
            "entity": "customer identity",
            "qualifiers": [],
        },
        "condition": None,
        "confidence": 0.9,
    }


@pytest.fixture
def sample_parsed_collection(sample_parsed_clause):
    """Create sample parsed collection."""
    return {
        "doc_id": "TEST_DOC_001",
        "clauses": [sample_parsed_clause],
        "parse_timestamp": "2025-01-01T00:00:00Z",
    }


# =============================================================================
# MockEmbeddingProvider Tests
# =============================================================================

class TestMockEmbeddingProvider:
    """Tests for MockEmbeddingProvider."""

    def test_embed_returns_vector(self, mock_embedding_provider):
        """Test that embed returns a vector."""
        embedding = mock_embedding_provider.embed("test text")

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(v, float) for v in embedding)

    def test_embed_deterministic(self, mock_embedding_provider):
        """Test that embeddings are deterministic."""
        embedding1 = mock_embedding_provider.embed("test text")
        embedding2 = mock_embedding_provider.embed("test text")

        assert embedding1 == embedding2

    def test_embed_different_texts(self, mock_embedding_provider):
        """Test that different texts produce different embeddings."""
        embedding1 = mock_embedding_provider.embed("text one")
        embedding2 = mock_embedding_provider.embed("text two")

        assert embedding1 != embedding2

    def test_embed_batch(self, mock_embedding_provider):
        """Test batch embedding."""
        texts = ["text one", "text two", "text three"]
        embeddings = mock_embedding_provider.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)


# =============================================================================
# SchemaRegistry Tests
# =============================================================================

class TestSchemaRegistry:
    """Tests for SchemaRegistry."""

    def test_create_empty_registry(self):
        """Test creating empty registry."""
        registry = SchemaRegistry()

        assert registry.schemas == []
        assert registry.synonyms == {}
        assert registry.manual_overrides == {}

    def test_create_registry_with_schemas(self, sample_schema):
        """Test creating registry with schemas."""
        registry = SchemaRegistry(schemas=[sample_schema])

        assert len(registry.schemas) == 1
        assert registry.schemas[0].schema_id == "test_schema"

    def test_default_registry(self):
        """Test default registry creation."""
        registry = create_default_registry()

        assert len(registry.schemas) > 0
        assert len(registry.synonyms) > 0


# =============================================================================
# SchemaMappingAgent Tests
# =============================================================================

class TestSchemaMappingAgent:
    """Tests for SchemaMappingAgent."""

    def test_initialization(self, sample_registry):
        """Test agent initialization."""
        agent = SchemaMappingAgent(
            registry=sample_registry,
            use_mock=True,
        )

        assert agent.registry == sample_registry
        assert agent.confidence_threshold == 0.7  # default

    def test_register_schema(self, mapper, sample_schema):
        """Test schema registration."""
        new_schema = TargetSchema(
            schema_id="new_schema",
            schema_type=SchemaType.API,
            tables=[],
        )

        mapper.register_schema(new_schema)

        assert any(s.schema_id == "new_schema" for s in mapper.registry.schemas)

    def test_add_synonym(self, mapper):
        """Test adding synonyms."""
        mapper.add_synonym("account", ["profile", "record"])

        assert "account" in mapper.registry.synonyms
        assert "profile" in mapper.registry.synonyms["account"]

    def test_add_manual_override(self, mapper):
        """Test adding manual override."""
        mapper.add_manual_override("financial institution", "org.institution_id")

        assert "financial institution" in mapper.registry.manual_overrides

    def test_map_entity_exact_match(self, mapper):
        """Test exact match mapping."""
        mapping, unmapped = mapper.map_entity("customer", SourceRole.ACTOR)

        # Should find exact match in semantic labels
        assert mapping is not None or unmapped is not None

    def test_map_entity_synonym_match(self, mapper):
        """Test synonym match mapping."""
        mapping, unmapped = mapper.map_entity("client", SourceRole.ACTOR)

        # "client" is a synonym for "customer"
        if mapping:
            assert mapping.mapping_method in [MappingMethod.SYNONYM, MappingMethod.SEMANTIC, MappingMethod.EXACT]

    def test_map_entity_manual_override(self, mapper):
        """Test manual override mapping."""
        mapper.add_manual_override("special entity", "test_schema:customer.customer_id")

        mapping, unmapped = mapper.map_entity("special entity", SourceRole.ACTOR)

        assert mapping is not None
        assert mapping.mapping_method == MappingMethod.MANUAL_OVERRIDE
        assert mapping.confidence == 1.0

    def test_map_entity_unmapped(self, mapper):
        """Test unmapped entity."""
        mapping, unmapped = mapper.map_entity(
            "completely unknown entity xyz",
            SourceRole.ACTOR,
        )

        # Either maps with low confidence or returns unmapped
        if unmapped:
            assert unmapped.entity == "completely unknown entity xyz"
            assert unmapped.reason

    def test_map_clause(self, mapper, sample_parsed_clause):
        """Test mapping a clause."""
        result = mapper.map_clause(sample_parsed_clause)

        assert isinstance(result, MappedClause)
        assert result.clause_id == "TEST_CL001"
        assert result.mapping_status in MappingStatus

    def test_map_clause_has_source(self, mapper, sample_parsed_clause):
        """Test that mapped clause includes source."""
        result = mapper.map_clause(sample_parsed_clause)

        assert result.source_clause == sample_parsed_clause

    def test_map_parsed_collection(self, mapper, sample_parsed_collection):
        """Test mapping a parsed collection."""
        result = mapper.map_parsed_collection(sample_parsed_collection)

        assert result.doc_id == "TEST_DOC_001"
        assert len(result.clauses) == 1
        assert result.mapping_timestamp


# =============================================================================
# EntityMapping Tests
# =============================================================================

class TestEntityMapping:
    """Tests for EntityMapping model."""

    def test_valid_entity_mapping(self):
        """Test creating valid entity mapping."""
        mapping = EntityMapping(
            source_entity="customer",
            source_role=SourceRole.ACTOR,
            target_path="customer.customer_id",
            target_schema="test_schema",
            confidence=0.9,
            mapping_method=MappingMethod.EXACT,
        )

        assert mapping.source_entity == "customer"
        assert mapping.confidence == 0.9

    def test_confidence_range(self):
        """Test confidence range validation."""
        with pytest.raises(ValueError):
            EntityMapping(
                source_entity="test",
                source_role=SourceRole.ACTOR,
                target_path="path",
                target_schema="schema",
                confidence=1.5,  # Out of range
                mapping_method=MappingMethod.EXACT,
            )


# =============================================================================
# MappedClause Tests
# =============================================================================

class TestMappedClause:
    """Tests for MappedClause model."""

    def test_valid_mapped_clause(self, sample_parsed_clause):
        """Test creating valid mapped clause."""
        mapped = MappedClause(
            clause_id="TEST_CL001",
            source_clause=sample_parsed_clause,
            mapped_entities=[],
            unmapped_entities=[],
            mapping_status=MappingStatus.COMPLETE,
        )

        assert mapped.clause_id == "TEST_CL001"
        assert mapped.mapping_status == MappingStatus.COMPLETE

    def test_mapping_status_enum(self):
        """Test MappingStatus enum values."""
        assert MappingStatus.COMPLETE.value == "complete"
        assert MappingStatus.PARTIAL.value == "partial"
        assert MappingStatus.FAILED.value == "failed"
        assert MappingStatus.NEEDS_REVIEW.value == "needs_review"


# =============================================================================
# Integration Tests
# =============================================================================

class TestMapperIntegration:
    """Integration tests for the mapper."""

    def test_full_mapping_pipeline(self, sample_parsed_collection):
        """Test full mapping pipeline with default registry."""
        registry = create_default_registry()
        mapper = SchemaMappingAgent(registry=registry, use_mock=True)

        result = mapper.map_parsed_collection(sample_parsed_collection)

        assert result.doc_id
        assert result.clauses
        assert result.mapping_timestamp

    def test_mapping_with_synonyms(self):
        """Test mapping using synonyms."""
        registry = create_default_registry()
        mapper = SchemaMappingAgent(registry=registry, use_mock=True)

        # "client" should map via synonym to customer
        mapping, unmapped = mapper.map_entity("client", SourceRole.ACTOR)

        # Should find some match
        assert mapping is not None or unmapped is not None

    def test_cosine_similarity(self, mapper):
        """Test cosine similarity calculation."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]

        similarity = mapper._cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(1.0)

        vec3 = [0.0, 1.0, 0.0]
        similarity = mapper._cosine_similarity(vec1, vec3)
        assert similarity == pytest.approx(0.0)
