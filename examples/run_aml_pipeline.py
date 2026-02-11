#!/usr/bin/env python3
"""
AegisLang AML/KYC Pipeline Runner.

Processes all regulatory documents in examples/regulations/ through the full
AegisLang pipeline (ingest → parse → map → compile → validate) and saves
artifacts to examples/output/.

Usage:
    python examples/run_aml_pipeline.py
"""

import json
import sys
from pathlib import Path

# Ensure aegislang is importable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from aegislang.agents.aegis_ingestor import AegisIngestor
from aegislang.agents.policy_parser_agent import PolicyParserAgent
from aegislang.agents.schema_mapping_agent import (
    SchemaField,
    SchemaMappingAgent,
    SchemaRegistry,
    SchemaTable,
    SchemaType,
    TargetSchema,
)
from aegislang.agents.compiler_agent import ArtifactFormat, CompilerAgent
from aegislang.agents.trace_validator_agent import TraceValidatorAgent


# =============================================================================
# AML Target Schema (realistic financial services database)
# =============================================================================

def create_aml_registry() -> SchemaRegistry:
    """Build a realistic AML/KYC database schema for entity mapping."""
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
                        description="Customer master table with KYC status",
                        fields=[
                            SchemaField(
                                field_name="id",
                                field_type="UUID",
                                semantic_labels=["customer", "client", "account holder"],
                                description="Primary key",
                            ),
                            SchemaField(
                                field_name="name",
                                field_type="VARCHAR(255)",
                                semantic_labels=["name", "customer name", "full name"],
                                description="Customer full legal name",
                            ),
                            SchemaField(
                                field_name="risk_level",
                                field_type="VARCHAR(20)",
                                semantic_labels=[
                                    "risk", "risk level", "risk rating",
                                    "risk profile", "risk category",
                                ],
                                description="Customer risk rating (low/medium/high)",
                            ),
                            SchemaField(
                                field_name="identity_verified",
                                field_type="BOOLEAN",
                                semantic_labels=[
                                    "identity", "verification", "kyc", "verified",
                                    "identity verified", "CIP", "identification",
                                ],
                                description="Whether CIP verification is complete",
                            ),
                            SchemaField(
                                field_name="verification_date",
                                field_type="TIMESTAMP",
                                semantic_labels=[
                                    "verification date", "verified date",
                                    "date verified", "CIP date",
                                ],
                                description="Date identity was verified",
                            ),
                            SchemaField(
                                field_name="beneficial_owner",
                                field_type="VARCHAR(255)",
                                semantic_labels=[
                                    "beneficial owner", "ultimate beneficial owner",
                                    "UBO", "ownership",
                                ],
                                description="Beneficial owner of legal entity",
                            ),
                            SchemaField(
                                field_name="pep_status",
                                field_type="BOOLEAN",
                                semantic_labels=[
                                    "politically exposed", "PEP",
                                    "politically exposed person",
                                ],
                                description="Politically exposed person flag",
                            ),
                        ],
                    ),
                    SchemaTable(
                        table_name="accounts",
                        description="Customer accounts",
                        fields=[
                            SchemaField(
                                field_name="id",
                                field_type="UUID",
                                semantic_labels=["account", "account id"],
                                description="Primary key",
                            ),
                            SchemaField(
                                field_name="customer_id",
                                field_type="UUID",
                                semantic_labels=["customer", "account holder"],
                                description="FK to customers",
                            ),
                            SchemaField(
                                field_name="balance",
                                field_type="DECIMAL(18,2)",
                                semantic_labels=["balance", "account balance"],
                                description="Current account balance",
                            ),
                            SchemaField(
                                field_name="opened_date",
                                field_type="TIMESTAMP",
                                semantic_labels=[
                                    "opened", "account opening", "date opened",
                                ],
                                description="Account opening date",
                            ),
                            SchemaField(
                                field_name="status",
                                field_type="VARCHAR(20)",
                                semantic_labels=[
                                    "status", "account status", "active",
                                ],
                                description="Account status (active/closed/frozen)",
                            ),
                        ],
                    ),
                    SchemaTable(
                        table_name="transactions",
                        description="Financial transactions",
                        fields=[
                            SchemaField(
                                field_name="id",
                                field_type="UUID",
                                semantic_labels=["transaction", "payment", "transfer"],
                                description="Primary key",
                            ),
                            SchemaField(
                                field_name="account_id",
                                field_type="UUID",
                                semantic_labels=["account"],
                                description="FK to accounts",
                            ),
                            SchemaField(
                                field_name="amount",
                                field_type="DECIMAL(18,2)",
                                semantic_labels=[
                                    "amount", "value", "sum", "total",
                                    "transaction amount", "threshold",
                                ],
                                description="Transaction amount",
                            ),
                            SchemaField(
                                field_name="type",
                                field_type="VARCHAR(50)",
                                semantic_labels=[
                                    "type", "transaction type",
                                    "transfer type", "category",
                                ],
                                description="Transaction type",
                            ),
                            SchemaField(
                                field_name="timestamp",
                                field_type="TIMESTAMP",
                                semantic_labels=[
                                    "timestamp", "date", "time",
                                    "transaction date",
                                ],
                                description="Transaction timestamp",
                            ),
                            SchemaField(
                                field_name="suspicious_flag",
                                field_type="BOOLEAN",
                                semantic_labels=[
                                    "suspicious", "flagged", "suspicious activity",
                                    "SAR", "reported", "alert",
                                ],
                                description="Suspicious activity flag",
                            ),
                        ],
                    ),
                    SchemaTable(
                        table_name="sar_reports",
                        description="Suspicious Activity Reports",
                        fields=[
                            SchemaField(
                                field_name="id",
                                field_type="UUID",
                                semantic_labels=[
                                    "SAR", "suspicious activity report",
                                    "report",
                                ],
                                description="Primary key",
                            ),
                            SchemaField(
                                field_name="transaction_id",
                                field_type="UUID",
                                semantic_labels=["transaction", "related transaction"],
                                description="FK to transactions",
                            ),
                            SchemaField(
                                field_name="filed_date",
                                field_type="TIMESTAMP",
                                semantic_labels=[
                                    "filed", "filing date", "report date",
                                ],
                                description="Date SAR was filed",
                            ),
                            SchemaField(
                                field_name="fiu_reference",
                                field_type="VARCHAR(100)",
                                semantic_labels=[
                                    "FIU", "FinCEN", "reference number",
                                ],
                                description="Financial Intelligence Unit reference",
                            ),
                        ],
                    ),
                    SchemaTable(
                        table_name="audit_log",
                        description="Compliance audit trail",
                        fields=[
                            SchemaField(
                                field_name="id",
                                field_type="UUID",
                                semantic_labels=["audit", "log entry"],
                                description="Primary key",
                            ),
                            SchemaField(
                                field_name="entity_type",
                                field_type="VARCHAR(50)",
                                semantic_labels=["entity", "record type"],
                                description="Type of entity (customer/transaction/etc)",
                            ),
                            SchemaField(
                                field_name="entity_id",
                                field_type="UUID",
                                semantic_labels=["entity id", "record id"],
                                description="ID of the audited entity",
                            ),
                            SchemaField(
                                field_name="action",
                                field_type="VARCHAR(100)",
                                semantic_labels=[
                                    "action", "event", "activity",
                                    "compliance action",
                                ],
                                description="Action performed",
                            ),
                            SchemaField(
                                field_name="timestamp",
                                field_type="TIMESTAMP",
                                semantic_labels=["timestamp", "date", "when"],
                                description="When the action occurred",
                            ),
                            SchemaField(
                                field_name="actor",
                                field_type="VARCHAR(255)",
                                semantic_labels=[
                                    "actor", "user", "staff", "employee",
                                    "officer", "personnel",
                                ],
                                description="Who performed the action",
                            ),
                        ],
                    ),
                ],
            ),
        ],
        synonyms={
            "customer": ["client", "account holder", "individual", "person", "user"],
            "bank": ["institution", "financial institution", "covered institution"],
            "transaction": ["payment", "transfer", "wire", "remittance"],
            "suspicious": ["flagged", "alert", "suspicious activity"],
            "identity": ["identification", "CIP", "KYC", "verification"],
            "employee": ["staff", "personnel", "officer", "manager"],
            "report": ["SAR", "suspicious activity report", "filing"],
            "record": ["document", "information", "data", "documentation"],
        },
    )


# =============================================================================
# Pipeline Runner
# =============================================================================

def run_pipeline(
    doc_path: Path,
    registry: SchemaRegistry,
    output_dir: Path,
) -> dict:
    """Run the full AegisLang pipeline on a single document."""
    doc_name = doc_path.stem
    print(f"\n{'='*60}")
    print(f"Processing: {doc_name}")
    print(f"{'='*60}")

    # Step 1: Ingest
    print("  [1/5] Ingesting document...")
    ingestor = AegisIngestor()
    ingested = ingestor.ingest(doc_path)
    ingested_data = ingested.model_dump()
    print(f"         Sections: {len(ingested.sections)}, "
          f"Chunks: {sum(len(s.text_chunks) for s in ingested.sections)}")

    # Step 2: Parse
    print("  [2/5] Parsing clauses (mock LLM)...")
    parser = PolicyParserAgent(use_mock=True)
    parsed = parser.parse_ingested_document(ingested_data)
    parsed_data = parsed.model_dump()
    clause_types = {}
    for c in parsed.clauses:
        t = c.type.value
        clause_types[t] = clause_types.get(t, 0) + 1
    print(f"         Clauses: {len(parsed.clauses)}")
    print(f"         Types: {clause_types}")

    # Step 3: Map
    print("  [3/5] Mapping to AML schema...")
    mapper = SchemaMappingAgent(registry=registry, use_mock=True)
    mapped = mapper.map_parsed_collection(parsed_data, target_schema_id="aml_schema")
    mapped_data = mapped.model_dump()
    mapped_count = sum(1 for c in mapped.clauses if c.mapping_status.value == "complete")
    partial_count = sum(1 for c in mapped.clauses if c.mapping_status.value == "partial")
    unmapped_count = sum(1 for c in mapped.clauses if c.mapping_status.value in ("failed", "needs_review"))
    print(f"         Fully mapped: {mapped_count}, "
          f"Partial: {partial_count}, Unmapped: {unmapped_count}")

    # Step 4: Compile
    print("  [4/5] Compiling artifacts (YAML, SQL, Python)...")
    compiler = CompilerAgent()
    formats = [ArtifactFormat.YAML, ArtifactFormat.SQL, ArtifactFormat.PYTHON]
    compiled = compiler.compile_mapped_collection(mapped_data, formats=formats)
    compiled_data = compiled.model_dump()
    artifact_counts = {}
    for a in compiled.artifacts:
        fmt = a.format.value
        artifact_counts[fmt] = artifact_counts.get(fmt, 0) + 1
    print(f"         Artifacts: {artifact_counts}")

    # Step 5: Validate
    print("  [5/5] Validating traceability...")
    validator = TraceValidatorAgent()
    validated = validator.validate_compiled_collection(
        compiled_data, mapped_data, parsed_data
    )
    print(f"         Summary: {validated.summary}")

    # Save outputs
    doc_output_dir = output_dir / doc_name
    doc_output_dir.mkdir(parents=True, exist_ok=True)

    # Save each artifact
    for artifact in compiled.artifacts:
        ext = artifact.format.value
        clause_id = artifact.clause_id.replace("/", "_")
        artifact_path = doc_output_dir / f"{clause_id}.{ext}"
        artifact_path.write_text(artifact.content)

    # Save full pipeline output as JSON
    pipeline_output = {
        "document": doc_name,
        "source_path": str(doc_path),
        "ingestion": {
            "sections": len(ingested.sections),
            "total_chunks": sum(len(s.text_chunks) for s in ingested.sections),
        },
        "parsing": {
            "total_clauses": len(parsed.clauses),
            "clause_types": clause_types,
        },
        "mapping": {
            "target_schema": "aml_schema",
            "fully_mapped": mapped_count,
            "partially_mapped": partial_count,
            "unmapped": unmapped_count,
        },
        "compilation": {
            "artifact_counts": artifact_counts,
            "total_artifacts": len(compiled.artifacts),
        },
        "validation": validated.summary,
    }

    summary_path = doc_output_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(pipeline_output, indent=2))

    print(f"  Output saved to: {doc_output_dir}/")
    return pipeline_output


def main():
    regulations_dir = Path(__file__).parent / "regulations"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    if not regulations_dir.exists():
        print(f"Error: {regulations_dir} not found")
        sys.exit(1)

    docs = sorted(regulations_dir.glob("*.md"))
    if not docs:
        print(f"No .md files found in {regulations_dir}")
        sys.exit(1)

    print(f"AegisLang AML/KYC Pipeline Runner")
    print(f"Documents: {len(docs)}")
    print(f"Output: {output_dir}")

    # Build AML schema registry
    registry = create_aml_registry()

    # Process each document
    all_results = []
    for doc_path in docs:
        result = run_pipeline(doc_path, registry, output_dir)
        all_results.append(result)

    # Print summary
    print(f"\n{'='*60}")
    print("Pipeline Summary")
    print(f"{'='*60}")
    total_clauses = sum(r["parsing"]["total_clauses"] for r in all_results)
    total_artifacts = sum(r["compilation"]["total_artifacts"] for r in all_results)
    total_mapped = sum(r["mapping"]["fully_mapped"] for r in all_results)
    total_partial = sum(r["mapping"]["partially_mapped"] for r in all_results)
    total_unmapped = sum(r["mapping"]["unmapped"] for r in all_results)

    print(f"Documents processed: {len(all_results)}")
    print(f"Total clauses extracted: {total_clauses}")
    print(f"Total artifacts generated: {total_artifacts}")
    print(f"Mapping: {total_mapped} fully mapped, "
          f"{total_partial} partial, {total_unmapped} unmapped")

    mapping_success = (total_mapped + total_partial) / max(total_clauses, 1) * 100
    print(f"Mapping success rate: {mapping_success:.1f}%")

    # Save aggregate summary
    aggregate = {
        "documents_processed": len(all_results),
        "total_clauses": total_clauses,
        "total_artifacts": total_artifacts,
        "mapping_fully_mapped": total_mapped,
        "mapping_partially_mapped": total_partial,
        "mapping_unmapped": total_unmapped,
        "mapping_success_rate_pct": round(mapping_success, 1),
        "per_document": all_results,
    }
    aggregate_path = output_dir / "aggregate_summary.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2))
    print(f"\nAggregate summary saved to: {aggregate_path}")


if __name__ == "__main__":
    main()
