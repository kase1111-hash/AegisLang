"""
AegisLang L5 Validation Layer - Trace Validator Agent

Purpose: Ensure semantic traceability, validate correctness, and emit
provenance metadata.

Functional Requirements:
- VAL-001: Validate clause→artifact provenance chain completeness
- VAL-002: Check artifact syntax validity
- VAL-003: Compute confidence scores for trace links
- VAL-004: Detect semantic drift between clause and artifact
- VAL-005: Generate lineage metadata for audit
- VAL-006: Publish validated artifacts to downstream systems
- VAL-007: Flag low-confidence traces for human review
- VAL-008: Maintain provenance graph in graph database
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
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


class ValidationStatus(str, Enum):
    """Validation result status."""

    PASSED = "passed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


class CheckResult(BaseModel):
    """Result of a single validation check."""

    check_name: str = Field(..., description="Name of the check")
    passed: bool = Field(..., description="Whether the check passed")
    message: str = Field(..., description="Check result message")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional details"
    )


class Lineage(BaseModel):
    """Complete lineage from source to artifact."""

    document_id: str = Field(..., description="Source document ID")
    section_id: str | None = Field(default=None, description="Section ID")
    chunk_id: str | None = Field(default=None, description="Chunk ID")
    clause_id: str = Field(..., description="Parsed clause ID")
    mapping_id: str | None = Field(default=None, description="Mapping ID")
    artifact_id: str = Field(..., description="Generated artifact ID")


class ValidationResult(BaseModel):
    """Output schema for validation result."""

    trace_id: str = Field(..., description="Unique trace identifier")
    source_clause: str = Field(..., description="Source clause ID")
    generated_artifact: str = Field(..., description="Artifact file path")
    validation_status: ValidationStatus = Field(..., description="Overall status")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Trace confidence"
    )
    validation_checks: list[CheckResult] = Field(
        default_factory=list, description="Individual check results"
    )
    lineage: Lineage = Field(..., description="Complete lineage chain")
    review_flags: list[str] = Field(
        default_factory=list, description="Flags for human review"
    )
    validated_at: str = Field(..., description="ISO 8601 timestamp")
    validated_by: str = Field(
        default="aegislang-validator-v1.0.0", description="Validator identifier"
    )


class ValidationResultCollection(BaseModel):
    """Collection of validation results."""

    doc_id: str = Field(..., description="Source document ID")
    results: list[ValidationResult] = Field(
        default_factory=list, description="Validation results"
    )
    summary: dict[str, int] = Field(
        default_factory=dict, description="Status summary counts"
    )
    validation_timestamp: str = Field(..., description="ISO 8601 timestamp")


class ProvenanceNode(BaseModel):
    """Node in the provenance graph."""

    node_id: str = Field(..., description="Unique node identifier")
    node_type: str = Field(..., description="Type: document, section, chunk, clause, mapping, artifact")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Node properties"
    )
    created_at: str = Field(..., description="ISO 8601 timestamp")


class ProvenanceEdge(BaseModel):
    """Edge in the provenance graph."""

    edge_id: str = Field(..., description="Unique edge identifier")
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    relationship: str = Field(..., description="Relationship type")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Edge properties"
    )


class ProvenanceGraph(BaseModel):
    """Complete provenance graph."""

    graph_id: str = Field(..., description="Unique graph identifier")
    doc_id: str = Field(..., description="Source document ID")
    nodes: list[ProvenanceNode] = Field(
        default_factory=list, description="Graph nodes"
    )
    edges: list[ProvenanceEdge] = Field(
        default_factory=list, description="Graph edges"
    )
    created_at: str = Field(..., description="ISO 8601 timestamp")


# -----------------------------------------------------------------------------
# Validation Configuration
# -----------------------------------------------------------------------------


class ValidationConfig(BaseModel):
    """Configuration for validation thresholds."""

    confidence_threshold: float = Field(
        default=0.85, description="Minimum confidence for auto-approval"
    )
    review_threshold: float = Field(
        default=0.70, description="Below this, flag for human review"
    )
    block_threshold: float = Field(
        default=0.50, description="Below this, block artifact"
    )
    require_syntax_valid: bool = Field(
        default=True, description="Require valid syntax"
    )
    require_complete_chain: bool = Field(
        default=True, description="Require complete provenance chain"
    )


# -----------------------------------------------------------------------------
# Validation Checks
# -----------------------------------------------------------------------------


class ValidationChecks:
    """Collection of validation check functions."""

    @staticmethod
    def check_chain_completeness(
        artifact: dict[str, Any],
        mapped_clause: dict[str, Any] | None,
        parsed_clause: dict[str, Any] | None,
    ) -> CheckResult:
        """Check that all pipeline stages have output."""
        missing = []

        if not parsed_clause:
            missing.append("parsed_clause")
        if not mapped_clause:
            missing.append("mapped_clause")
        if not artifact:
            missing.append("artifact")

        if missing:
            return CheckResult(
                check_name="chain_completeness",
                passed=False,
                message=f"Missing pipeline stages: {', '.join(missing)}",
                details={"missing_stages": missing},
            )

        # Check for required fields
        if parsed_clause and not parsed_clause.get("source_text"):
            missing.append("source_text in parsed_clause")
        if mapped_clause and not mapped_clause.get("clause_id"):
            missing.append("clause_id in mapped_clause")
        if artifact and not artifact.get("content"):
            missing.append("content in artifact")

        if missing:
            return CheckResult(
                check_name="chain_completeness",
                passed=False,
                message=f"Missing required fields: {', '.join(missing)}",
                details={"missing_fields": missing},
            )

        return CheckResult(
            check_name="chain_completeness",
            passed=True,
            message="All pipeline stages present with required fields",
        )

    @staticmethod
    def check_syntax_validity(artifact: dict[str, Any]) -> CheckResult:
        """Check artifact syntax is valid."""
        syntax_valid = artifact.get("syntax_valid", False)
        warnings = artifact.get("warnings", [])

        if not syntax_valid:
            return CheckResult(
                check_name="syntax_validity",
                passed=False,
                message="Artifact has syntax errors",
                details={"warnings": warnings},
            )

        if warnings:
            return CheckResult(
                check_name="syntax_validity",
                passed=True,
                message=f"Valid syntax with {len(warnings)} warning(s)",
                details={"warnings": warnings},
            )

        return CheckResult(
            check_name="syntax_validity",
            passed=True,
            message="Artifact syntax is valid",
        )

    @staticmethod
    def check_confidence_threshold(
        confidence: float,
        threshold: float,
    ) -> CheckResult:
        """Check confidence meets threshold."""
        passed = confidence >= threshold

        return CheckResult(
            check_name="confidence_threshold",
            passed=passed,
            message=f"Confidence {confidence:.2f} {'meets' if passed else 'below'} threshold {threshold:.2f}",
            details={
                "confidence": confidence,
                "threshold": threshold,
            },
        )

    @staticmethod
    def check_semantic_alignment(
        parsed_clause: dict[str, Any],
        artifact: dict[str, Any],
    ) -> CheckResult:
        """Check semantic alignment between clause and artifact."""
        source_text = parsed_clause.get("source_text", "")
        artifact_content = artifact.get("content", "")

        # Extract key terms from source
        clause_type = parsed_clause.get("type", "")
        actor = parsed_clause.get("actor", {}).get("entity", "")
        action = parsed_clause.get("action", {}).get("verb", "")

        # Check if key terms appear in artifact
        checks = []
        content_lower = artifact_content.lower()

        if clause_type and clause_type.lower() in content_lower:
            checks.append(("type", True))
        else:
            checks.append(("type", False))

        if actor and actor.lower() in content_lower:
            checks.append(("actor", True))
        else:
            checks.append(("actor", False))

        if action and action.lower() in content_lower:
            checks.append(("action", True))
        else:
            checks.append(("action", False))

        passed_count = sum(1 for _, passed in checks if passed)
        total = len(checks)

        if passed_count == total:
            return CheckResult(
                check_name="semantic_alignment",
                passed=True,
                message="All key terms preserved in artifact",
                details={"checks": dict(checks)},
            )
        elif passed_count >= total // 2:
            return CheckResult(
                check_name="semantic_alignment",
                passed=True,
                message=f"Partial alignment: {passed_count}/{total} key terms found",
                details={"checks": dict(checks)},
            )
        else:
            return CheckResult(
                check_name="semantic_alignment",
                passed=False,
                message=f"Poor alignment: only {passed_count}/{total} key terms found",
                details={"checks": dict(checks)},
            )

    @staticmethod
    def check_cross_reference_integrity(
        parsed_clause: dict[str, Any],
        all_clause_ids: set[str],
    ) -> CheckResult:
        """Check that cross-references point to valid clauses."""
        cross_refs = parsed_clause.get("cross_references", [])

        if not cross_refs:
            return CheckResult(
                check_name="cross_reference_integrity",
                passed=True,
                message="No cross-references to validate",
            )

        invalid_refs = [ref for ref in cross_refs if ref not in all_clause_ids]

        if invalid_refs:
            return CheckResult(
                check_name="cross_reference_integrity",
                passed=False,
                message=f"Invalid cross-references: {', '.join(invalid_refs)}",
                details={"invalid_refs": invalid_refs},
            )

        return CheckResult(
            check_name="cross_reference_integrity",
            passed=True,
            message=f"All {len(cross_refs)} cross-references valid",
        )


# -----------------------------------------------------------------------------
# Trace Validator Agent
# -----------------------------------------------------------------------------


class TraceValidatorAgent:
    """
    L5 Validation Layer Agent.

    Ensures semantic traceability, validates correctness, and emits
    provenance metadata for audit.
    """

    def __init__(
        self,
        config: ValidationConfig | None = None,
        graph_store: Any | None = None,  # Neo4j/ArangoDB client
    ):
        """
        Initialize the validator agent.

        Args:
            config: Validation configuration
            graph_store: Graph database client for provenance storage
        """
        self.config = config or ValidationConfig()
        self.graph_store = graph_store
        self._checks = ValidationChecks()

        logger.info(
            "trace_validator_initialized",
            confidence_threshold=self.config.confidence_threshold,
            review_threshold=self.config.review_threshold,
        )

    def validate_artifact(
        self,
        artifact: dict[str, Any],
        mapped_clause: dict[str, Any],
        parsed_clause: dict[str, Any],
        doc_id: str,
        all_clause_ids: set[str] | None = None,
    ) -> ValidationResult:
        """
        Validate a single artifact and its provenance chain.

        Args:
            artifact: CompiledArtifact as dictionary
            mapped_clause: MappedClause as dictionary
            parsed_clause: ParsedClause as dictionary
            doc_id: Source document ID
            all_clause_ids: Set of all clause IDs for cross-reference checking

        Returns:
            ValidationResult with checks and lineage
        """
        clause_id = artifact.get("clause_id", "unknown")
        artifact_id = artifact.get("artifact_id", "unknown")

        logger.debug(
            "validating_artifact",
            clause_id=clause_id,
            artifact_id=artifact_id,
        )

        all_clause_ids = all_clause_ids or set()
        checks: list[CheckResult] = []
        review_flags: list[str] = []

        # Run validation checks
        # 1. Chain completeness
        chain_check = self._checks.check_chain_completeness(
            artifact, mapped_clause, parsed_clause
        )
        checks.append(chain_check)
        if not chain_check.passed:
            review_flags.append("incomplete_chain")

        # 2. Syntax validity
        syntax_check = self._checks.check_syntax_validity(artifact)
        checks.append(syntax_check)
        if not syntax_check.passed:
            review_flags.append("syntax_error")

        # 3. Compute trace confidence
        confidences = []

        # Clause extraction confidence
        if parsed_clause:
            clause_confidence = parsed_clause.get("confidence", 0.5)
            confidences.append(clause_confidence)

        # Mapping confidence
        if mapped_clause:
            mapping_confidences = [
                m.get("confidence", 0.5)
                for m in mapped_clause.get("mapped_entities", [])
            ]
            if mapping_confidences:
                confidences.extend(mapping_confidences)

        # Calculate aggregate confidence
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
        else:
            avg_confidence = 0.5

        # 4. Confidence threshold check
        confidence_check = self._checks.check_confidence_threshold(
            avg_confidence, self.config.confidence_threshold
        )
        checks.append(confidence_check)

        if avg_confidence < self.config.review_threshold:
            review_flags.append("low_confidence")
        elif avg_confidence < self.config.confidence_threshold:
            review_flags.append("below_threshold")

        # 5. Semantic alignment
        if parsed_clause and artifact:
            alignment_check = self._checks.check_semantic_alignment(
                parsed_clause, artifact
            )
            checks.append(alignment_check)
            if not alignment_check.passed:
                review_flags.append("semantic_drift")

        # 6. Cross-reference integrity
        if parsed_clause:
            ref_check = self._checks.check_cross_reference_integrity(
                parsed_clause, all_clause_ids
            )
            checks.append(ref_check)
            if not ref_check.passed:
                review_flags.append("invalid_references")

        # Determine overall status
        critical_failures = [
            c for c in checks
            if not c.passed and c.check_name in ["chain_completeness", "syntax_validity"]
        ]

        if critical_failures and self.config.require_complete_chain:
            status = ValidationStatus.FAILED
        elif review_flags:
            status = ValidationStatus.NEEDS_REVIEW
        else:
            status = ValidationStatus.PASSED

        # Block if below block threshold
        if avg_confidence < self.config.block_threshold:
            status = ValidationStatus.FAILED
            review_flags.append("blocked_low_confidence")

        # Build lineage
        lineage = Lineage(
            document_id=doc_id,
            section_id=self._extract_section_id(clause_id),
            chunk_id=parsed_clause.get("source_chunk_id") if parsed_clause else None,
            clause_id=clause_id,
            mapping_id=f"map_{clause_id}" if mapped_clause else None,
            artifact_id=artifact_id,
        )

        # Generate trace ID
        trace_id = f"TRC_{uuid.uuid4().hex[:8].upper()}"

        result = ValidationResult(
            trace_id=trace_id,
            source_clause=clause_id,
            generated_artifact=artifact.get("file_path", "unknown"),
            validation_status=status,
            confidence_score=round(avg_confidence, 3),
            validation_checks=checks,
            lineage=lineage,
            review_flags=review_flags,
            validated_at=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            "artifact_validated",
            trace_id=trace_id,
            status=status.value,
            confidence=avg_confidence,
            flags=review_flags,
        )

        return result

    def _extract_section_id(self, clause_id: str) -> str | None:
        """Extract section ID from clause ID."""
        # Clause IDs typically follow pattern: DOC_S001_C001_CL001
        parts = clause_id.split("_")
        for i, part in enumerate(parts):
            if part.startswith("S") and part[1:].isdigit():
                return "_".join(parts[:i + 1])
        return None

    def validate_compiled_collection(
        self,
        compiled_collection: dict[str, Any],
        mapped_collection: dict[str, Any],
        parsed_collection: dict[str, Any],
    ) -> ValidationResultCollection:
        """
        Validate all artifacts in a compiled collection.

        Args:
            compiled_collection: CompiledArtifactCollection as dictionary
            mapped_collection: MappedClauseCollection as dictionary
            parsed_collection: ParsedClauseCollection as dictionary

        Returns:
            ValidationResultCollection with all results
        """
        doc_id = compiled_collection.get("doc_id", "unknown")

        # Build lookup maps
        parsed_by_id = {
            c["clause_id"]: c
            for c in parsed_collection.get("clauses", [])
        }
        mapped_by_id = {
            c["clause_id"]: c
            for c in mapped_collection.get("clauses", [])
        }
        all_clause_ids = set(parsed_by_id.keys())

        results: list[ValidationResult] = []

        for artifact in compiled_collection.get("artifacts", []):
            clause_id = artifact.get("clause_id", "")
            parsed = parsed_by_id.get(clause_id)
            mapped = mapped_by_id.get(clause_id)

            result = self.validate_artifact(
                artifact=artifact,
                mapped_clause=mapped,
                parsed_clause=parsed,
                doc_id=doc_id,
                all_clause_ids=all_clause_ids,
            )
            results.append(result)

        # Build summary
        summary = {
            "total": len(results),
            "passed": sum(1 for r in results if r.validation_status == ValidationStatus.PASSED),
            "failed": sum(1 for r in results if r.validation_status == ValidationStatus.FAILED),
            "needs_review": sum(1 for r in results if r.validation_status == ValidationStatus.NEEDS_REVIEW),
        }

        collection = ValidationResultCollection(
            doc_id=doc_id,
            results=results,
            summary=summary,
            validation_timestamp=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            "collection_validated",
            doc_id=doc_id,
            **summary,
        )

        return collection

    def build_provenance_graph(
        self,
        validation_results: ValidationResultCollection,
        include_content: bool = False,
    ) -> ProvenanceGraph:
        """
        Build a provenance graph from validation results.

        Args:
            validation_results: Validated results
            include_content: Include full content in nodes

        Returns:
            ProvenanceGraph for visualization/storage
        """
        doc_id = validation_results.doc_id
        nodes: list[ProvenanceNode] = []
        edges: list[ProvenanceEdge] = []
        timestamp = datetime.now(timezone.utc).isoformat()

        # Create document node
        doc_node = ProvenanceNode(
            node_id=f"doc:{doc_id}",
            node_type="document",
            properties={"doc_id": doc_id},
            created_at=timestamp,
        )
        nodes.append(doc_node)

        seen_sections: set[str] = set()
        seen_chunks: set[str] = set()

        for result in validation_results.results:
            lineage = result.lineage

            # Section node
            if lineage.section_id and lineage.section_id not in seen_sections:
                section_node = ProvenanceNode(
                    node_id=f"section:{lineage.section_id}",
                    node_type="section",
                    properties={"section_id": lineage.section_id},
                    created_at=timestamp,
                )
                nodes.append(section_node)
                seen_sections.add(lineage.section_id)

                # Document -> Section edge
                edges.append(ProvenanceEdge(
                    edge_id=f"e:{doc_id}->{lineage.section_id}",
                    source_id=f"doc:{doc_id}",
                    target_id=f"section:{lineage.section_id}",
                    relationship="CONTAINS_SECTION",
                    properties={},
                ))

            # Chunk node
            if lineage.chunk_id and lineage.chunk_id not in seen_chunks:
                chunk_node = ProvenanceNode(
                    node_id=f"chunk:{lineage.chunk_id}",
                    node_type="chunk",
                    properties={"chunk_id": lineage.chunk_id},
                    created_at=timestamp,
                )
                nodes.append(chunk_node)
                seen_chunks.add(lineage.chunk_id)

                # Section -> Chunk edge
                if lineage.section_id:
                    edges.append(ProvenanceEdge(
                        edge_id=f"e:{lineage.section_id}->{lineage.chunk_id}",
                        source_id=f"section:{lineage.section_id}",
                        target_id=f"chunk:{lineage.chunk_id}",
                        relationship="CONTAINS_CHUNK",
                        properties={},
                    ))

            # Clause node
            clause_node = ProvenanceNode(
                node_id=f"clause:{lineage.clause_id}",
                node_type="clause",
                properties={
                    "clause_id": lineage.clause_id,
                    "confidence": result.confidence_score,
                    "status": result.validation_status.value,
                },
                created_at=timestamp,
            )
            nodes.append(clause_node)

            # Chunk -> Clause edge
            if lineage.chunk_id:
                edges.append(ProvenanceEdge(
                    edge_id=f"e:{lineage.chunk_id}->{lineage.clause_id}",
                    source_id=f"chunk:{lineage.chunk_id}",
                    target_id=f"clause:{lineage.clause_id}",
                    relationship="PARSED_TO",
                    properties={},
                ))

            # Artifact node
            artifact_node = ProvenanceNode(
                node_id=f"artifact:{lineage.artifact_id}",
                node_type="artifact",
                properties={
                    "artifact_id": lineage.artifact_id,
                    "file_path": result.generated_artifact,
                    "trace_id": result.trace_id,
                },
                created_at=timestamp,
            )
            nodes.append(artifact_node)

            # Clause -> Artifact edge
            edges.append(ProvenanceEdge(
                edge_id=f"e:{lineage.clause_id}->{lineage.artifact_id}",
                source_id=f"clause:{lineage.clause_id}",
                target_id=f"artifact:{lineage.artifact_id}",
                relationship="COMPILED_TO",
                properties={
                    "confidence": result.confidence_score,
                    "validated": result.validation_status == ValidationStatus.PASSED,
                },
            ))

        graph = ProvenanceGraph(
            graph_id=f"graph:{doc_id}:{uuid.uuid4().hex[:8]}",
            doc_id=doc_id,
            nodes=nodes,
            edges=edges,
            created_at=timestamp,
        )

        logger.info(
            "provenance_graph_built",
            graph_id=graph.graph_id,
            nodes=len(nodes),
            edges=len(edges),
        )

        return graph

    async def store_provenance_graph(
        self,
        graph: ProvenanceGraph,
    ) -> bool:
        """
        Store provenance graph in graph database.

        Args:
            graph: ProvenanceGraph to store

        Returns:
            Success status
        """
        if not self.graph_store:
            logger.warning("no_graph_store_configured")
            return False

        try:
            # Neo4j implementation
            if hasattr(self.graph_store, "run"):
                async with self.graph_store.session() as session:
                    # Create nodes
                    for node in graph.nodes:
                        await session.run(
                            f"""
                            MERGE (n:{node.node_type} {{node_id: $node_id}})
                            SET n += $properties
                            SET n.created_at = $created_at
                            """,
                            node_id=node.node_id,
                            properties=node.properties,
                            created_at=node.created_at,
                        )

                    # Create edges
                    for edge in graph.edges:
                        await session.run(
                            f"""
                            MATCH (a {{node_id: $source_id}})
                            MATCH (b {{node_id: $target_id}})
                            MERGE (a)-[r:{edge.relationship}]->(b)
                            SET r += $properties
                            """,
                            source_id=edge.source_id,
                            target_id=edge.target_id,
                            properties=edge.properties,
                        )

                logger.info("provenance_graph_stored", graph_id=graph.graph_id)
                return True

        except Exception as e:
            logger.error("provenance_store_failed", error=str(e))
            return False

        return False

    def export_graph_json(self, graph: ProvenanceGraph) -> str:
        """Export provenance graph as JSON."""
        return graph.model_dump_json(indent=2)

    def export_graph_dot(self, graph: ProvenanceGraph) -> str:
        """Export provenance graph as DOT format for Graphviz."""
        lines = ["digraph Provenance {"]
        lines.append("  rankdir=TB;")
        lines.append("  node [shape=box];")
        lines.append("")

        # Define node styles
        styles = {
            "document": 'style=filled,fillcolor="#e3f2fd"',
            "section": 'style=filled,fillcolor="#fff3e0"',
            "chunk": 'style=filled,fillcolor="#f3e5f5"',
            "clause": 'style=filled,fillcolor="#e8f5e9"',
            "artifact": 'style=filled,fillcolor="#fce4ec"',
        }

        # Add nodes
        for node in graph.nodes:
            style = styles.get(node.node_type, "")
            label = node.node_id.split(":")[-1]
            lines.append(f'  "{node.node_id}" [label="{label}",{style}];')

        lines.append("")

        # Add edges
        for edge in graph.edges:
            lines.append(f'  "{edge.source_id}" -> "{edge.target_id}" [label="{edge.relationship}"];')

        lines.append("}")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Event Publishing (Agent-OS Integration)
# -----------------------------------------------------------------------------


async def publish_validated_event(
    collection: ValidationResultCollection,
    redis_url: str | None = None,
) -> None:
    """Publish policy.validated event to Agent-OS event bus."""
    if redis_url is None:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    try:
        import redis.asyncio as redis_async

        client = redis_async.from_url(redis_url)
        await client.publish(
            "policy.validated",
            collection.model_dump_json(),
        )
        await client.aclose()

        logger.info(
            "event_published",
            topic="policy.validated",
            doc_id=collection.doc_id,
            result_count=len(collection.results),
        )
    except Exception as e:
        logger.warning(
            "event_publish_failed",
            topic="policy.validated",
            error=str(e),
        )


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------


def main() -> None:
    """Command-line interface for validation."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="AegisLang Trace Validator - L5 Validation Layer"
    )
    parser.add_argument(
        "--compiled",
        required=True,
        help="Compiled artifacts JSON (from L4 compiler)",
    )
    parser.add_argument(
        "--mapped",
        required=True,
        help="Mapped clauses JSON (from L3 mapper)",
    )
    parser.add_argument(
        "--parsed",
        required=True,
        help="Parsed clauses JSON (from L2 parser)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file (default: stdout)",
    )
    parser.add_argument(
        "--graph",
        help="Output provenance graph JSON to file",
    )
    parser.add_argument(
        "--graph-dot",
        help="Output provenance graph DOT to file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.85,
        help="Confidence threshold (default: 0.85)",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.70,
        help="Review threshold (default: 0.70)",
    )

    args = parser.parse_args()

    # Load input files
    paths = {
        "compiled": Path(args.compiled),
        "mapped": Path(args.mapped),
        "parsed": Path(args.parsed),
    }

    for name, path in paths.items():
        if not path.exists():
            print(f"Error: {name} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    try:
        compiled_data = json.loads(paths["compiled"].read_text())
        mapped_data = json.loads(paths["mapped"].read_text())
        parsed_data = json.loads(paths["parsed"].read_text())
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize validator
    config = ValidationConfig(
        confidence_threshold=args.confidence_threshold,
        review_threshold=args.review_threshold,
    )
    agent = TraceValidatorAgent(config=config)

    # Validate
    results = agent.validate_compiled_collection(
        compiled_data, mapped_data, parsed_data
    )

    # Output results
    output_json = results.model_dump_json(indent=2)
    if args.output:
        Path(args.output).write_text(output_json)
        print(f"Results written to: {args.output}", file=sys.stderr)
    else:
        print(output_json)

    # Output provenance graph
    if args.graph or args.graph_dot:
        graph = agent.build_provenance_graph(results)

        if args.graph:
            Path(args.graph).write_text(agent.export_graph_json(graph))
            print(f"Graph JSON written to: {args.graph}", file=sys.stderr)

        if args.graph_dot:
            Path(args.graph_dot).write_text(agent.export_graph_dot(graph))
            print(f"Graph DOT written to: {args.graph_dot}", file=sys.stderr)

    # Print summary
    print(f"\nValidation Summary:", file=sys.stderr)
    for key, value in results.summary.items():
        print(f"  {key}: {value}", file=sys.stderr)


if __name__ == "__main__":
    main()
