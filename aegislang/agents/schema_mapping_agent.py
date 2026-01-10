"""
AegisLang L3 Mapping Layer - Schema Mapping Agent

Purpose: Align natural-language entities from parsed clauses to operational system schemas.

Functional Requirements:
- MAP-001: Match regulatory entities to schema field paths
- MAP-002: Use semantic embeddings for fuzzy matching
- MAP-003: Support multiple target schema formats (SQL, API, Object)
- MAP-004: Maintain Schema Registry with versioning
- MAP-005: Handle synonym resolution
- MAP-006: Support manual mapping overrides
- MAP-007: Confidence scoring for each mapping
- MAP-008: Detect unmappable entities and flag for review
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# -----------------------------------------------------------------------------
# Schema Definitions
# -----------------------------------------------------------------------------


class SchemaType(str, Enum):
    """Types of target schemas."""

    SQL = "sql"
    API = "api"
    OBJECT = "object"


class MappingMethod(str, Enum):
    """Method used to establish mapping."""

    EXACT = "exact"
    SYNONYM = "synonym"
    SEMANTIC = "semantic"
    MANUAL_OVERRIDE = "manual_override"


class MappingStatus(str, Enum):
    """Status of the mapping operation."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


class SourceRole(str, Enum):
    """Role of the source entity in the clause."""

    ACTOR = "actor"
    OBJECT = "object"
    CONDITION_SUBJECT = "condition_subject"


class SchemaField(BaseModel):
    """A field within a schema table/object."""

    field_name: str = Field(..., description="Name of the field")
    field_type: str = Field(..., description="Data type of the field")
    semantic_labels: list[str] = Field(
        default_factory=list, description="Semantic labels for matching"
    )
    description: str | None = Field(default=None, description="Field description")
    embedding: list[float] | None = Field(
        default=None, description="Pre-computed embedding vector"
    )


class SchemaTable(BaseModel):
    """A table/object in the schema."""

    table_name: str = Field(..., description="Name of the table/object")
    fields: list[SchemaField] = Field(
        default_factory=list, description="Fields in the table"
    )
    description: str | None = Field(default=None, description="Table description")


class TargetSchema(BaseModel):
    """A target schema for entity mapping."""

    schema_id: str = Field(..., description="Unique schema identifier")
    schema_type: SchemaType = Field(..., description="Type of schema")
    version: str = Field(default="1.0.0", description="Schema version")
    tables: list[SchemaTable] = Field(
        default_factory=list, description="Tables/objects in schema"
    )


class SchemaRegistry(BaseModel):
    """Registry of available target schemas."""

    registry_version: str = Field(default="1.0.0", description="Registry version")
    schemas: list[TargetSchema] = Field(
        default_factory=list, description="Registered schemas"
    )
    synonyms: dict[str, list[str]] = Field(
        default_factory=dict, description="Synonym mappings"
    )
    manual_overrides: dict[str, str] = Field(
        default_factory=dict, description="Manual entity-to-path overrides"
    )


class EntityMapping(BaseModel):
    """A single entity-to-schema mapping."""

    source_entity: str = Field(..., description="The source entity from clause")
    source_role: SourceRole = Field(..., description="Role in the clause")
    target_path: str = Field(
        ..., description="Dot-notation path to schema field"
    )
    target_schema: str = Field(..., description="Schema ID")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Mapping confidence"
    )
    mapping_method: MappingMethod = Field(
        ..., description="Method used for mapping"
    )


class SuggestedMatch(BaseModel):
    """A suggested match for an unmapped entity."""

    target_path: str = Field(..., description="Suggested target path")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Match confidence")


class UnmappedEntity(BaseModel):
    """An entity that could not be mapped."""

    entity: str = Field(..., description="The unmapped entity")
    reason: str = Field(..., description="Reason for failure")
    suggested_matches: list[SuggestedMatch] = Field(
        default_factory=list, description="Potential matches below threshold"
    )


class MappedClause(BaseModel):
    """Output schema for a mapped clause."""

    clause_id: str = Field(..., description="Source clause ID")
    source_clause: dict[str, Any] = Field(
        ..., description="Original parsed clause data"
    )
    mapped_entities: list[EntityMapping] = Field(
        default_factory=list, description="Successfully mapped entities"
    )
    unmapped_entities: list[UnmappedEntity] = Field(
        default_factory=list, description="Entities that could not be mapped"
    )
    mapping_status: MappingStatus = Field(..., description="Overall mapping status")


class MappedClauseCollection(BaseModel):
    """Collection of mapped clauses."""

    doc_id: str = Field(..., description="Source document ID")
    target_schema: str = Field(..., description="Target schema used")
    clauses: list[MappedClause] = Field(
        default_factory=list, description="Mapped clauses"
    )
    mapping_timestamp: str = Field(..., description="ISO 8601 timestamp")


# -----------------------------------------------------------------------------
# Embedding Provider
# -----------------------------------------------------------------------------


class BaseEmbeddingProvider:
    """Base class for embedding providers."""

    def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embeddings provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
    ):
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            ) from e

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model

    def embed(self, text: str) -> list[float]:
        """Generate embedding using OpenAI."""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for batch."""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [item.embedding for item in response.data]


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Sentence Transformers embedding provider (local)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers required. "
                "Install with: pip install sentence-transformers"
            ) from e

        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        """Generate embedding using Sentence Transformers."""
        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for batch."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions
        self._cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        """Generate deterministic mock embedding."""
        if text in self._cache:
            return self._cache[text]

        # Generate deterministic embedding based on text hash
        import hashlib

        hash_bytes = hashlib.sha256(text.encode()).digest()
        embedding = []
        for i in range(self.dimensions):
            byte_idx = i % len(hash_bytes)
            # Normalize to [-1, 1]
            value = (hash_bytes[byte_idx] / 255.0) * 2 - 1
            embedding.append(value)

        self._cache[text] = embedding
        return embedding


# -----------------------------------------------------------------------------
# Schema Mapping Agent
# -----------------------------------------------------------------------------


class SchemaMappingAgent:
    """
    L3 Mapping Layer Agent.

    Maps natural-language entities from parsed clauses to operational
    system schemas using semantic similarity and synonym resolution.
    """

    def __init__(
        self,
        registry: SchemaRegistry | None = None,
        embedding_provider: BaseEmbeddingProvider | None = None,
        confidence_threshold: float = 0.7,
        use_mock: bool = False,
    ):
        """
        Initialize the mapping agent.

        Args:
            registry: Schema registry to use
            embedding_provider: Provider for semantic embeddings
            confidence_threshold: Minimum confidence for valid mapping
            use_mock: Use mock embedding provider
        """
        self.registry = registry or SchemaRegistry()
        self.confidence_threshold = confidence_threshold

        if use_mock or embedding_provider is None:
            self.embedding_provider: BaseEmbeddingProvider = MockEmbeddingProvider()
            logger.info("schema_mapper_initialized", provider="mock")
        else:
            self.embedding_provider = embedding_provider
            logger.info("schema_mapper_initialized", provider=type(embedding_provider).__name__)

        # Pre-compute schema field embeddings
        self._field_embeddings: dict[str, tuple[list[float], str]] = {}
        self._build_field_index()

    def _build_field_index(self) -> None:
        """Build embedding index for all schema fields."""
        for schema in self.registry.schemas:
            for table in schema.tables:
                for field in table.fields:
                    path = f"{table.table_name}.{field.field_name}"
                    full_path = f"{schema.schema_id}:{path}"

                    # Create text representation for embedding
                    text_parts = [field.field_name]
                    text_parts.extend(field.semantic_labels)
                    if field.description:
                        text_parts.append(field.description)

                    text = " ".join(text_parts)

                    # Use pre-computed embedding or generate new one
                    if field.embedding:
                        embedding = field.embedding
                    else:
                        embedding = self.embedding_provider.embed(text)

                    self._field_embeddings[full_path] = (embedding, path)

        logger.info(
            "field_index_built",
            field_count=len(self._field_embeddings),
        )

    def register_schema(self, schema: TargetSchema) -> None:
        """Register a new schema in the registry."""
        # Remove existing schema with same ID
        self.registry.schemas = [
            s for s in self.registry.schemas if s.schema_id != schema.schema_id
        ]
        self.registry.schemas.append(schema)

        # Rebuild index
        self._build_field_index()

        logger.info("schema_registered", schema_id=schema.schema_id)

    def add_synonym(self, term: str, synonyms: list[str]) -> None:
        """Add synonyms for a term."""
        if term in self.registry.synonyms:
            existing = set(self.registry.synonyms[term])
            existing.update(synonyms)
            self.registry.synonyms[term] = list(existing)
        else:
            self.registry.synonyms[term] = synonyms

    def add_manual_override(self, entity: str, target_path: str) -> None:
        """Add a manual mapping override."""
        self.registry.manual_overrides[entity.lower()] = target_path
        logger.info(
            "manual_override_added",
            entity=entity,
            target_path=target_path,
        )

    def map_entity(
        self,
        entity: str,
        role: SourceRole,
        target_schema_id: str | None = None,
    ) -> tuple[EntityMapping | None, UnmappedEntity | None]:
        """
        Map a single entity to a schema field.

        Args:
            entity: The entity text to map
            role: Role of the entity in the clause
            target_schema_id: Specific schema to map to (optional)

        Returns:
            Tuple of (mapping, unmapped) - one will be None
        """
        entity_lower = entity.lower()

        # Check manual overrides first
        if entity_lower in self.registry.manual_overrides:
            target_path = self.registry.manual_overrides[entity_lower]
            schema_id = target_path.split(":")[0] if ":" in target_path else target_schema_id or "default"
            path = target_path.split(":")[-1]

            mapping = EntityMapping(
                source_entity=entity,
                source_role=role,
                target_path=path,
                target_schema=schema_id,
                confidence=1.0,
                mapping_method=MappingMethod.MANUAL_OVERRIDE,
            )
            return mapping, None

        # Try exact match
        exact_match = self._find_exact_match(entity, target_schema_id)
        if exact_match:
            mapping = EntityMapping(
                source_entity=entity,
                source_role=role,
                target_path=exact_match[1],
                target_schema=exact_match[0],
                confidence=0.95,
                mapping_method=MappingMethod.EXACT,
            )
            return mapping, None

        # Try synonym match
        synonym_match = self._find_synonym_match(entity, target_schema_id)
        if synonym_match:
            mapping = EntityMapping(
                source_entity=entity,
                source_role=role,
                target_path=synonym_match[1],
                target_schema=synonym_match[0],
                confidence=0.85,
                mapping_method=MappingMethod.SYNONYM,
            )
            return mapping, None

        # Try semantic match
        semantic_matches = self._find_semantic_matches(entity, target_schema_id)

        if semantic_matches and semantic_matches[0][2] >= self.confidence_threshold:
            best = semantic_matches[0]
            mapping = EntityMapping(
                source_entity=entity,
                source_role=role,
                target_path=best[1],
                target_schema=best[0],
                confidence=best[2],
                mapping_method=MappingMethod.SEMANTIC,
            )
            return mapping, None

        # Entity could not be mapped
        suggestions = [
            SuggestedMatch(
                target_path=f"{m[0]}:{m[1]}",
                confidence=max(0.0, m[2]),  # Clamp negative similarities to 0
            )
            for m in semantic_matches[:3]
        ]

        unmapped = UnmappedEntity(
            entity=entity,
            reason=f"No match found above threshold ({self.confidence_threshold})",
            suggested_matches=suggestions,
        )
        return None, unmapped

    def _find_exact_match(
        self, entity: str, target_schema_id: str | None
    ) -> tuple[str, str] | None:
        """Find exact match in schema fields."""
        entity_lower = entity.lower().replace(" ", "_")

        for schema in self.registry.schemas:
            if target_schema_id and schema.schema_id != target_schema_id:
                continue

            for table in schema.tables:
                for field in table.fields:
                    if field.field_name.lower() == entity_lower:
                        return schema.schema_id, f"{table.table_name}.{field.field_name}"

                    # Check semantic labels
                    for label in field.semantic_labels:
                        if label.lower() == entity_lower:
                            return schema.schema_id, f"{table.table_name}.{field.field_name}"

        return None

    def _find_synonym_match(
        self, entity: str, target_schema_id: str | None
    ) -> tuple[str, str] | None:
        """Find match via synonym resolution."""
        entity_lower = entity.lower()

        # Find synonyms for the entity
        related_terms = [entity_lower]
        for term, syns in self.registry.synonyms.items():
            if entity_lower == term.lower():
                related_terms.extend(s.lower() for s in syns)
            elif entity_lower in [s.lower() for s in syns]:
                related_terms.append(term.lower())
                related_terms.extend(s.lower() for s in syns)

        # Try exact match with synonyms
        for term in related_terms:
            match = self._find_exact_match(term, target_schema_id)
            if match:
                return match

        return None

    def _find_semantic_matches(
        self, entity: str, target_schema_id: str | None
    ) -> list[tuple[str, str, float]]:
        """Find matches using semantic similarity."""
        entity_embedding = self.embedding_provider.embed(entity)

        matches: list[tuple[str, str, float]] = []

        for full_path, (field_embedding, path) in self._field_embeddings.items():
            schema_id = full_path.split(":")[0]

            if target_schema_id and schema_id != target_schema_id:
                continue

            similarity = self._cosine_similarity(entity_embedding, field_embedding)
            matches.append((schema_id, path, similarity))

        # Sort by similarity descending
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def map_clause(
        self,
        parsed_clause: dict[str, Any],
        target_schema_id: str | None = None,
    ) -> MappedClause:
        """
        Map all entities in a parsed clause.

        Args:
            parsed_clause: ParsedClause as dictionary
            target_schema_id: Target schema to map to

        Returns:
            MappedClause with mappings and unmapped entities
        """
        clause_id = parsed_clause["clause_id"]
        mapped_entities: list[EntityMapping] = []
        unmapped_entities: list[UnmappedEntity] = []

        # Map actor
        if "actor" in parsed_clause and parsed_clause["actor"]:
            actor_entity = parsed_clause["actor"].get("entity", "")
            if actor_entity:
                mapping, unmapped = self.map_entity(
                    actor_entity, SourceRole.ACTOR, target_schema_id
                )
                if mapping:
                    mapped_entities.append(mapping)
                if unmapped:
                    unmapped_entities.append(unmapped)

        # Map object
        if "object" in parsed_clause and parsed_clause["object"]:
            obj_entity = parsed_clause["object"].get("entity", "")
            if obj_entity:
                mapping, unmapped = self.map_entity(
                    obj_entity, SourceRole.OBJECT, target_schema_id
                )
                if mapping:
                    mapped_entities.append(mapping)
                if unmapped:
                    unmapped_entities.append(unmapped)

        # Map condition subject if present
        if "condition" in parsed_clause and parsed_clause["condition"]:
            trigger = parsed_clause["condition"].get("trigger", "")
            if trigger:
                # Extract subject from trigger
                words = trigger.split()
                if len(words) > 0:
                    subject = " ".join(words[:3])  # Take first few words
                    mapping, unmapped = self.map_entity(
                        subject, SourceRole.CONDITION_SUBJECT, target_schema_id
                    )
                    if mapping:
                        mapped_entities.append(mapping)
                    # Don't add condition subjects to unmapped - they're often phrases

        # Determine status
        if not unmapped_entities:
            status = MappingStatus.COMPLETE
        elif mapped_entities:
            status = MappingStatus.PARTIAL
        else:
            status = MappingStatus.NEEDS_REVIEW

        # Check for low confidence mappings
        low_confidence = any(
            m.confidence < 0.8 for m in mapped_entities
        )
        if low_confidence and status == MappingStatus.COMPLETE:
            status = MappingStatus.NEEDS_REVIEW

        return MappedClause(
            clause_id=clause_id,
            source_clause=parsed_clause,
            mapped_entities=mapped_entities,
            unmapped_entities=unmapped_entities,
            mapping_status=status,
        )

    def map_parsed_collection(
        self,
        parsed_collection: dict[str, Any],
        target_schema_id: str | None = None,
    ) -> MappedClauseCollection:
        """
        Map all clauses in a parsed collection.

        Args:
            parsed_collection: ParsedClauseCollection as dictionary
            target_schema_id: Target schema to map to

        Returns:
            MappedClauseCollection with all mappings
        """
        doc_id = parsed_collection["doc_id"]
        mapped_clauses: list[MappedClause] = []

        for clause in parsed_collection.get("clauses", []):
            mapped = self.map_clause(clause, target_schema_id)
            mapped_clauses.append(mapped)

        collection = MappedClauseCollection(
            doc_id=doc_id,
            target_schema=target_schema_id or "default",
            clauses=mapped_clauses,
            mapping_timestamp=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            "collection_mapped",
            doc_id=doc_id,
            total_clauses=len(mapped_clauses),
            complete=sum(1 for c in mapped_clauses if c.mapping_status == MappingStatus.COMPLETE),
            partial=sum(1 for c in mapped_clauses if c.mapping_status == MappingStatus.PARTIAL),
            needs_review=sum(1 for c in mapped_clauses if c.mapping_status == MappingStatus.NEEDS_REVIEW),
        )

        return collection

    def load_registry(self, path: str | Path) -> None:
        """Load schema registry from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        self.registry = SchemaRegistry(**data)
        self._build_field_index()
        logger.info("registry_loaded", path=str(path))

    def save_registry(self, path: str | Path) -> None:
        """Save schema registry to JSON file."""
        path = Path(path)
        path.write_text(self.registry.model_dump_json(indent=2))
        logger.info("registry_saved", path=str(path))


# -----------------------------------------------------------------------------
# Default Schema Registry
# -----------------------------------------------------------------------------


def create_default_registry() -> SchemaRegistry:
    """Create a default schema registry with common compliance schemas."""
    return SchemaRegistry(
        registry_version="1.0.0",
        schemas=[
            TargetSchema(
                schema_id="kyc_schema",
                schema_type=SchemaType.SQL,
                version="1.0.0",
                tables=[
                    SchemaTable(
                        table_name="customer",
                        fields=[
                            SchemaField(
                                field_name="customer_id",
                                field_type="UUID",
                                semantic_labels=["customer", "client", "account holder", "user"],
                            ),
                            SchemaField(
                                field_name="identity_verified",
                                field_type="BOOLEAN",
                                semantic_labels=["identity", "verification", "kyc", "verified"],
                            ),
                            SchemaField(
                                field_name="verification_date",
                                field_type="TIMESTAMP",
                                semantic_labels=["date", "timestamp", "when verified"],
                            ),
                        ],
                    ),
                    SchemaTable(
                        table_name="transaction",
                        fields=[
                            SchemaField(
                                field_name="transaction_id",
                                field_type="UUID",
                                semantic_labels=["transaction", "payment", "transfer"],
                            ),
                            SchemaField(
                                field_name="amount",
                                field_type="DECIMAL",
                                semantic_labels=["amount", "value", "sum", "total"],
                            ),
                            SchemaField(
                                field_name="reported",
                                field_type="BOOLEAN",
                                semantic_labels=["report", "reported", "flagged", "suspicious"],
                            ),
                        ],
                    ),
                ],
            ),
            TargetSchema(
                schema_id="org_schema",
                schema_type=SchemaType.SQL,
                version="1.0.0",
                tables=[
                    SchemaTable(
                        table_name="institution",
                        fields=[
                            SchemaField(
                                field_name="institution_id",
                                field_type="UUID",
                                semantic_labels=[
                                    "institution", "bank", "financial institution",
                                    "organization", "company"
                                ],
                            ),
                            SchemaField(
                                field_name="license_status",
                                field_type="VARCHAR",
                                semantic_labels=["license", "licensed", "authorized"],
                            ),
                        ],
                    ),
                    SchemaTable(
                        table_name="employee",
                        fields=[
                            SchemaField(
                                field_name="employee_id",
                                field_type="UUID",
                                semantic_labels=["employee", "staff", "worker", "personnel"],
                            ),
                            SchemaField(
                                field_name="access_level",
                                field_type="VARCHAR",
                                semantic_labels=["access", "permission", "authorization"],
                            ),
                        ],
                    ),
                ],
            ),
            TargetSchema(
                schema_id="records_schema",
                schema_type=SchemaType.SQL,
                version="1.0.0",
                tables=[
                    SchemaTable(
                        table_name="audit_record",
                        fields=[
                            SchemaField(
                                field_name="record_id",
                                field_type="UUID",
                                semantic_labels=["record", "log", "entry"],
                            ),
                            SchemaField(
                                field_name="retention_period",
                                field_type="INTERVAL",
                                semantic_labels=["retention", "keep", "maintain", "duration"],
                            ),
                            SchemaField(
                                field_name="created_at",
                                field_type="TIMESTAMP",
                                semantic_labels=["created", "timestamp", "date"],
                            ),
                        ],
                    ),
                ],
            ),
        ],
        synonyms={
            "customer": ["client", "account holder", "consumer", "user", "individual"],
            "transaction": ["payment", "transfer", "wire", "remittance"],
            "financial institution": ["bank", "institution", "firm", "company"],
            "employee": ["staff", "worker", "personnel", "team member"],
            "record": ["document", "file", "log", "entry"],
            "verify": ["check", "confirm", "validate", "authenticate"],
            "report": ["submit", "file", "disclose", "notify"],
        },
    )


# -----------------------------------------------------------------------------
# Event Publishing (Agent-OS Integration)
# -----------------------------------------------------------------------------


async def publish_mapped_event(
    collection: MappedClauseCollection,
    redis_url: str | None = None,
) -> None:
    """Publish policy.mapped event to Agent-OS event bus."""
    if redis_url is None:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    try:
        import redis.asyncio as redis_async

        client = redis_async.from_url(redis_url)
        await client.publish(
            "policy.mapped",
            collection.model_dump_json(),
        )
        await client.aclose()

        logger.info(
            "event_published",
            topic="policy.mapped",
            doc_id=collection.doc_id,
            clause_count=len(collection.clauses),
        )
    except Exception as e:
        logger.warning(
            "event_publish_failed",
            topic="policy.mapped",
            error=str(e),
        )


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------


def main() -> None:
    """Command-line interface for schema mapping."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="AegisLang Schema Mapper - L3 Mapping Layer"
    )
    parser.add_argument(
        "input",
        help="Input file (JSON from L2 parser)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--registry",
        help="Path to schema registry JSON file",
    )
    parser.add_argument(
        "--schema",
        help="Target schema ID to map to",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for mapping (default: 0.7)",
    )

    args = parser.parse_args()

    # Initialize agent
    if args.registry:
        registry_path = Path(args.registry)
        if not registry_path.exists():
            print(f"Error: Registry file not found: {args.registry}", file=sys.stderr)
            sys.exit(1)
        registry_data = json.loads(registry_path.read_text())
        registry = SchemaRegistry(**registry_data)
    else:
        registry = create_default_registry()

    agent = SchemaMappingAgent(
        registry=registry,
        confidence_threshold=args.threshold,
        use_mock=True,
    )

    # Read input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        parsed_data = json.loads(input_path.read_text())
        result = agent.map_parsed_collection(parsed_data, args.schema)
        output_json = result.model_dump_json(indent=2)

        if args.output:
            Path(args.output).write_text(output_json)
            print(f"Output written to: {args.output}", file=sys.stderr)
        else:
            print(output_json)

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
