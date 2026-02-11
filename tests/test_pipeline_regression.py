"""
Pipeline output regression tests.

These tests run the AML pipeline on the Phase 3 regulatory documents and verify
that output characteristics remain stable across code changes. They test that:

- The same document produces the same clause count
- Clause type distributions are stable
- Mapping success does not regress
"""

from pathlib import Path

import pytest

from aegislang.agents.aegis_ingestor import AegisIngestor
from aegislang.agents.policy_parser_agent import PolicyParserAgent
from aegislang.agents.schema_mapping_agent import (
    MappingStatus,
    SchemaField,
    SchemaMappingAgent,
    SchemaRegistry,
    SchemaTable,
    SchemaType,
    TargetSchema,
)

REGULATIONS_DIR = Path(__file__).parent.parent / "examples" / "regulations"


def _create_aml_registry() -> SchemaRegistry:
    """Minimal AML schema for regression testing."""
    return SchemaRegistry(
        registry_version="1.0.0",
        schemas=[
            TargetSchema(
                schema_id="aml_schema",
                schema_type=SchemaType.SQL,
                version="1.0.0",
                tables=[
                    SchemaTable(
                        table_name="customers",
                        fields=[
                            SchemaField(
                                field_name="identity_verified",
                                field_type="BOOLEAN",
                                semantic_labels=[
                                    "identity", "verification", "kyc",
                                    "verified", "customer identity",
                                ],
                            ),
                            SchemaField(
                                field_name="risk_level",
                                field_type="VARCHAR(20)",
                                semantic_labels=[
                                    "risk", "risk level", "risk rating",
                                    "risk profile",
                                ],
                            ),
                            SchemaField(
                                field_name="beneficial_owner",
                                field_type="VARCHAR(255)",
                                semantic_labels=[
                                    "beneficial owner", "UBO", "ownership",
                                ],
                            ),
                        ],
                    ),
                    SchemaTable(
                        table_name="transactions",
                        fields=[
                            SchemaField(
                                field_name="suspicious_flag",
                                field_type="BOOLEAN",
                                semantic_labels=[
                                    "suspicious", "suspicious activity",
                                    "flagged", "SAR",
                                ],
                            ),
                            SchemaField(
                                field_name="amount",
                                field_type="DECIMAL(18,2)",
                                semantic_labels=[
                                    "amount", "transaction amount",
                                    "threshold", "value",
                                ],
                            ),
                        ],
                    ),
                    SchemaTable(
                        table_name="audit_log",
                        fields=[
                            SchemaField(
                                field_name="action",
                                field_type="VARCHAR(100)",
                                semantic_labels=[
                                    "action", "event", "activity",
                                    "procedures", "due diligence",
                                ],
                            ),
                            SchemaField(
                                field_name="actor",
                                field_type="VARCHAR(255)",
                                semantic_labels=[
                                    "actor", "staff", "employee",
                                    "officer",
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
        synonyms={
            "customer": ["client", "account holder", "individual"],
            "identity": ["identification", "CIP", "KYC", "verification"],
            "suspicious": ["flagged", "alert", "SAR"],
            "record": ["documentation", "information"],
        },
    )


def _run_pipeline_stages(doc_path: Path):
    """Run ingest + parse + map pipeline stages on a document."""
    ingestor = AegisIngestor()
    ingested = ingestor.ingest(doc_path)

    parser = PolicyParserAgent(use_mock=True)
    parsed = parser.parse_ingested_document(ingested.model_dump())

    registry = _create_aml_registry()
    mapper = SchemaMappingAgent(registry=registry, use_mock=True)
    mapped = mapper.map_parsed_collection(
        parsed.model_dump(), target_schema_id="aml_schema"
    )

    return ingested, parsed, mapped


class TestFinCENCDDRegression:
    """Regression tests for FinCEN CDD Rule processing."""

    @pytest.fixture(scope="class")
    def pipeline_result(self):
        doc = REGULATIONS_DIR / "fincen_cdd_rule.md"
        if not doc.exists():
            pytest.skip("Regulation document not found")
        return _run_pipeline_stages(doc)

    def test_clause_count_stable(self, pipeline_result):
        _, parsed, _ = pipeline_result
        assert len(parsed.clauses) >= 10, (
            f"Expected at least 10 clauses, got {len(parsed.clauses)}"
        )

    def test_obligation_clauses_dominant(self, pipeline_result):
        _, parsed, _ = pipeline_result
        types = [c.type.value for c in parsed.clauses]
        obligation_count = types.count("obligation")
        assert obligation_count >= 8, (
            f"Expected at least 8 obligation clauses, got {obligation_count}"
        )

    def test_prohibition_clause_present(self, pipeline_result):
        _, parsed, _ = pipeline_result
        types = [c.type.value for c in parsed.clauses]
        assert "prohibition" in types, "Expected at least one prohibition clause"

    def test_mapping_does_not_regress(self, pipeline_result):
        _, _, mapped = pipeline_result
        partial_or_complete = sum(
            1 for c in mapped.clauses
            if c.mapping_status in (MappingStatus.COMPLETE, MappingStatus.PARTIAL)
        )
        # With improved mock extraction, we expect at least some mappings
        assert partial_or_complete >= 1, (
            f"Expected at least 1 mapped clause, got {partial_or_complete}"
        )

    def test_all_clauses_have_confidence(self, pipeline_result):
        _, parsed, _ = pipeline_result
        for clause in parsed.clauses:
            assert 0.0 < clause.confidence <= 1.0, (
                f"Clause {clause.clause_id} has invalid confidence: {clause.confidence}"
            )


class TestFFIECCIPRegression:
    """Regression tests for FFIEC CIP Manual processing."""

    @pytest.fixture(scope="class")
    def pipeline_result(self):
        doc = REGULATIONS_DIR / "ffiec_cip_manual.md"
        if not doc.exists():
            pytest.skip("Regulation document not found")
        return _run_pipeline_stages(doc)

    def test_clause_count_stable(self, pipeline_result):
        _, parsed, _ = pipeline_result
        assert len(parsed.clauses) >= 12, (
            f"Expected at least 12 clauses, got {len(parsed.clauses)}"
        )

    def test_permission_clauses_present(self, pipeline_result):
        _, parsed, _ = pipeline_result
        types = [c.type.value for c in parsed.clauses]
        permission_count = types.count("permission")
        assert permission_count >= 1, "Expected at least one permission clause"

    def test_sections_extracted(self, pipeline_result):
        ingested, _, _ = pipeline_result
        assert len(ingested.sections) >= 5, (
            f"Expected at least 5 sections, got {len(ingested.sections)}"
        )


class TestFATFRec10Regression:
    """Regression tests for FATF Recommendation 10 processing."""

    @pytest.fixture(scope="class")
    def pipeline_result(self):
        doc = REGULATIONS_DIR / "fatf_recommendation_10.md"
        if not doc.exists():
            pytest.skip("Regulation document not found")
        return _run_pipeline_stages(doc)

    def test_clause_count_stable(self, pipeline_result):
        _, parsed, _ = pipeline_result
        assert len(parsed.clauses) >= 14, (
            f"Expected at least 14 clauses, got {len(parsed.clauses)}"
        )

    def test_multiple_clause_types(self, pipeline_result):
        _, parsed, _ = pipeline_result
        types = {c.type.value for c in parsed.clauses}
        assert len(types) >= 3, (
            f"Expected at least 3 clause types, got: {types}"
        )

    def test_conditional_clause_present(self, pipeline_result):
        _, parsed, _ = pipeline_result
        types = [c.type.value for c in parsed.clauses]
        assert "conditional" in types, "Expected at least one conditional clause"

    def test_actors_extracted(self, pipeline_result):
        _, parsed, _ = pipeline_result
        actors = [c.actor.entity for c in parsed.clauses if c.actor]
        non_default = [a for a in actors if a != "unspecified entity"]
        assert len(non_default) >= len(parsed.clauses) * 0.5, (
            f"Expected at least 50% of clauses to have actors, "
            f"got {len(non_default)}/{len(parsed.clauses)}"
        )
