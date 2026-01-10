"""
Integration tests for AegisLang pipeline.

Tests the full flow from ingestion through validation.
"""

import pytest
import tempfile
from pathlib import Path


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_policy_document(tmp_path):
    """Create a comprehensive sample policy document."""
    content = """# Anti-Money Laundering Policy

## 1. Introduction

This policy establishes the requirements for anti-money laundering compliance
at all financial institutions under our jurisdiction.

## 2. Customer Due Diligence

### 2.1 Identity Verification

Financial institutions must verify customer identity before opening any account.
This verification must be completed within 5 business days of account application.

### 2.2 Enhanced Due Diligence

Banks shall perform enhanced due diligence for high-risk customers.
High-risk customers include politically exposed persons and their associates.

## 3. Transaction Monitoring

### 3.1 Reporting Requirements

Institutions must report suspicious transactions exceeding $10,000 within 24 hours.
All reports shall be submitted to the Financial Intelligence Unit.

### 3.2 Prohibited Activities

Employees shall not process transactions without proper authorization.
Staff must not share customer information with unauthorized parties.

## 4. Record Keeping

All customer records must be maintained for at least 5 years.
Transaction records shall be retained for 7 years from the date of transaction.

## 5. Exceptions

Notwithstanding section 2.1, temporary accounts may be opened pending verification
if the transaction amount is below $1,000.
"""
    file_path = tmp_path / "aml_policy.md"
    file_path.write_text(content)
    return file_path


# =============================================================================
# End-to-End Pipeline Tests
# =============================================================================

class TestEndToEndPipeline:
    """End-to-end tests for the full pipeline."""

    def test_ingest_to_parse(self, sample_policy_document):
        """Test ingestion followed by parsing."""
        from aegislang.agents.aegis_ingestor import AegisIngestor
        from aegislang.agents.policy_parser_agent import PolicyParserAgent

        # Ingest
        ingestor = AegisIngestor()
        ingested = ingestor.ingest(sample_policy_document)

        assert ingested.doc_id
        assert len(ingested.sections) > 0

        # Parse
        parser = PolicyParserAgent(use_mock=True)
        parsed = parser.parse_ingested_document(ingested.model_dump())

        assert parsed.doc_id == ingested.doc_id
        assert len(parsed.clauses) > 0

        # Verify clause types are detected
        clause_types = {c.type.value for c in parsed.clauses}
        assert len(clause_types) > 0

    def test_ingest_to_map(self, sample_policy_document):
        """Test ingestion through mapping."""
        from aegislang.agents.aegis_ingestor import AegisIngestor
        from aegislang.agents.policy_parser_agent import PolicyParserAgent
        from aegislang.agents.schema_mapping_agent import (
            SchemaMappingAgent,
            create_default_registry,
        )

        # Ingest
        ingestor = AegisIngestor()
        ingested = ingestor.ingest(sample_policy_document)

        # Parse
        parser = PolicyParserAgent(use_mock=True)
        parsed = parser.parse_ingested_document(ingested.model_dump())

        # Map
        mapper = SchemaMappingAgent(
            registry=create_default_registry(),
            use_mock=True,
        )
        mapped = mapper.map_parsed_collection(parsed.model_dump())

        assert mapped.doc_id == ingested.doc_id
        assert len(mapped.clauses) > 0

        # Check mapping status distribution
        statuses = {c.mapping_status.value for c in mapped.clauses}
        assert len(statuses) > 0

    def test_ingest_to_compile(self, sample_policy_document):
        """Test ingestion through compilation."""
        from aegislang.agents.aegis_ingestor import AegisIngestor
        from aegislang.agents.policy_parser_agent import PolicyParserAgent
        from aegislang.agents.schema_mapping_agent import (
            SchemaMappingAgent,
            create_default_registry,
        )
        from aegislang.agents.compiler_agent import CompilerAgent, ArtifactFormat

        # Ingest
        ingestor = AegisIngestor()
        ingested = ingestor.ingest(sample_policy_document)

        # Parse
        parser = PolicyParserAgent(use_mock=True)
        parsed = parser.parse_ingested_document(ingested.model_dump())

        # Map
        mapper = SchemaMappingAgent(
            registry=create_default_registry(),
            use_mock=True,
        )
        mapped = mapper.map_parsed_collection(parsed.model_dump())

        # Compile
        compiler = CompilerAgent()
        compiled = compiler.compile_mapped_collection(
            mapped.model_dump(),
            formats=[ArtifactFormat.YAML, ArtifactFormat.JSON],
        )

        assert compiled.doc_id == ingested.doc_id
        assert len(compiled.artifacts) > 0

        # Check formats generated
        formats = {a.format.value for a in compiled.artifacts}
        assert "yaml" in formats
        assert "json" in formats

    def test_full_pipeline_with_validation(self, sample_policy_document):
        """Test complete pipeline including validation."""
        from aegislang.agents.aegis_ingestor import AegisIngestor
        from aegislang.agents.policy_parser_agent import PolicyParserAgent
        from aegislang.agents.schema_mapping_agent import (
            SchemaMappingAgent,
            create_default_registry,
        )
        from aegislang.agents.compiler_agent import CompilerAgent, ArtifactFormat
        from aegislang.agents.trace_validator_agent import TraceValidatorAgent

        # Ingest
        ingestor = AegisIngestor()
        ingested = ingestor.ingest(sample_policy_document)

        # Parse
        parser = PolicyParserAgent(use_mock=True)
        parsed = parser.parse_ingested_document(ingested.model_dump())

        # Map
        mapper = SchemaMappingAgent(
            registry=create_default_registry(),
            use_mock=True,
        )
        mapped = mapper.map_parsed_collection(parsed.model_dump())

        # Compile
        compiler = CompilerAgent()
        compiled = compiler.compile_mapped_collection(
            mapped.model_dump(),
            formats=[ArtifactFormat.YAML],
        )

        # Validate
        validator = TraceValidatorAgent()
        validated = validator.validate_compiled_collection(
            compiled.model_dump(),
            mapped.model_dump(),
            parsed.model_dump(),
        )

        assert validated.doc_id == ingested.doc_id
        assert len(validated.results) > 0
        assert "total" in validated.summary

    def test_provenance_graph_generation(self, sample_policy_document):
        """Test provenance graph generation."""
        from aegislang.agents.aegis_ingestor import AegisIngestor
        from aegislang.agents.policy_parser_agent import PolicyParserAgent
        from aegislang.agents.schema_mapping_agent import (
            SchemaMappingAgent,
            create_default_registry,
        )
        from aegislang.agents.compiler_agent import CompilerAgent, ArtifactFormat
        from aegislang.agents.trace_validator_agent import TraceValidatorAgent

        # Run pipeline
        ingestor = AegisIngestor()
        ingested = ingestor.ingest(sample_policy_document)

        parser = PolicyParserAgent(use_mock=True)
        parsed = parser.parse_ingested_document(ingested.model_dump())

        mapper = SchemaMappingAgent(registry=create_default_registry(), use_mock=True)
        mapped = mapper.map_parsed_collection(parsed.model_dump())

        compiler = CompilerAgent()
        compiled = compiler.compile_mapped_collection(
            mapped.model_dump(),
            formats=[ArtifactFormat.YAML],
        )

        validator = TraceValidatorAgent()
        validated = validator.validate_compiled_collection(
            compiled.model_dump(),
            mapped.model_dump(),
            parsed.model_dump(),
        )

        # Generate provenance graph
        graph = validator.build_provenance_graph(validated)

        assert graph.doc_id == ingested.doc_id
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0

        # Verify node types
        node_types = {n.node_type for n in graph.nodes}
        assert "document" in node_types
        assert "artifact" in node_types

    def test_graph_export_formats(self, sample_policy_document):
        """Test graph export to different formats."""
        from aegislang.agents.aegis_ingestor import AegisIngestor
        from aegislang.agents.policy_parser_agent import PolicyParserAgent
        from aegislang.agents.schema_mapping_agent import (
            SchemaMappingAgent,
            create_default_registry,
        )
        from aegislang.agents.compiler_agent import CompilerAgent, ArtifactFormat
        from aegislang.agents.trace_validator_agent import TraceValidatorAgent

        # Run pipeline (abbreviated)
        ingestor = AegisIngestor()
        ingested = ingestor.ingest(sample_policy_document)

        parser = PolicyParserAgent(use_mock=True)
        parsed = parser.parse_ingested_document(ingested.model_dump())

        mapper = SchemaMappingAgent(registry=create_default_registry(), use_mock=True)
        mapped = mapper.map_parsed_collection(parsed.model_dump())

        compiler = CompilerAgent()
        compiled = compiler.compile_mapped_collection(
            mapped.model_dump(),
            formats=[ArtifactFormat.YAML],
        )

        validator = TraceValidatorAgent()
        validated = validator.validate_compiled_collection(
            compiled.model_dump(),
            mapped.model_dump(),
            parsed.model_dump(),
        )

        graph = validator.build_provenance_graph(validated)

        # Test JSON export
        json_output = validator.export_graph_json(graph)
        assert "nodes" in json_output
        assert "edges" in json_output

        # Test DOT export
        dot_output = validator.export_graph_dot(graph)
        assert "digraph" in dot_output
        assert "->" in dot_output


# =============================================================================
# Clause Type Detection Tests
# =============================================================================

class TestClauseTypeDetection:
    """Tests for clause type detection across the pipeline."""

    def test_obligation_detection(self):
        """Test detection of obligation clauses."""
        from aegislang.agents.policy_parser_agent import PolicyParserAgent, ClauseType

        parser = PolicyParserAgent(use_mock=True)

        obligation_texts = [
            "Banks must verify customer identity.",
            "Institutions shall report suspicious activity.",
            "Organizations are required to maintain records.",
        ]

        for text in obligation_texts:
            result = parser.parse_clause(text, "CL", "C")
            assert result.type == ClauseType.OBLIGATION, f"Failed for: {text}"

    def test_prohibition_detection(self):
        """Test detection of prohibition clauses."""
        from aegislang.agents.policy_parser_agent import PolicyParserAgent, ClauseType

        parser = PolicyParserAgent(use_mock=True)

        prohibition_texts = [
            "Employees shall not share passwords.",
            "Staff must not access unauthorized systems.",
            "Trading on insider information is prohibited.",
        ]

        for text in prohibition_texts:
            result = parser.parse_clause(text, "CL", "C")
            assert result.type == ClauseType.PROHIBITION, f"Failed for: {text}"

    def test_permission_detection(self):
        """Test detection of permission clauses."""
        from aegislang.agents.policy_parser_agent import PolicyParserAgent, ClauseType

        parser = PolicyParserAgent(use_mock=True)

        permission_texts = [
            "Customers may request data deletion.",
            "Users are permitted to access their records.",
        ]

        for text in permission_texts:
            result = parser.parse_clause(text, "CL", "C")
            assert result.type == ClauseType.PERMISSION, f"Failed for: {text}"


# =============================================================================
# Artifact Generation Tests
# =============================================================================

class TestArtifactGeneration:
    """Tests for artifact generation."""

    def test_yaml_artifact_structure(self, sample_policy_document):
        """Test YAML artifact structure."""
        from aegislang.agents.aegis_ingestor import AegisIngestor
        from aegislang.agents.policy_parser_agent import PolicyParserAgent
        from aegislang.agents.schema_mapping_agent import (
            SchemaMappingAgent,
            create_default_registry,
        )
        from aegislang.agents.compiler_agent import CompilerAgent, ArtifactFormat
        import yaml

        # Run pipeline
        ingestor = AegisIngestor()
        ingested = ingestor.ingest(sample_policy_document)

        parser = PolicyParserAgent(use_mock=True)
        parsed = parser.parse_ingested_document(ingested.model_dump())

        mapper = SchemaMappingAgent(registry=create_default_registry(), use_mock=True)
        mapped = mapper.map_parsed_collection(parsed.model_dump())

        compiler = CompilerAgent()
        compiled = compiler.compile_mapped_collection(
            mapped.model_dump(),
            formats=[ArtifactFormat.YAML],
        )

        # Verify YAML artifacts
        yaml_artifacts = [a for a in compiled.artifacts if a.format == ArtifactFormat.YAML]
        assert len(yaml_artifacts) > 0

        for artifact in yaml_artifacts:
            # Should be valid YAML
            assert artifact.syntax_valid
            parsed_yaml = yaml.safe_load(artifact.content)
            assert parsed_yaml is not None

    def test_sql_artifact_structure(self, sample_policy_document):
        """Test SQL artifact structure."""
        from aegislang.agents.aegis_ingestor import AegisIngestor
        from aegislang.agents.policy_parser_agent import PolicyParserAgent
        from aegislang.agents.schema_mapping_agent import (
            SchemaMappingAgent,
            create_default_registry,
        )
        from aegislang.agents.compiler_agent import CompilerAgent, ArtifactFormat

        # Run pipeline
        ingestor = AegisIngestor()
        ingested = ingestor.ingest(sample_policy_document)

        parser = PolicyParserAgent(use_mock=True)
        parsed = parser.parse_ingested_document(ingested.model_dump())

        mapper = SchemaMappingAgent(registry=create_default_registry(), use_mock=True)
        mapped = mapper.map_parsed_collection(parsed.model_dump())

        compiler = CompilerAgent()
        compiled = compiler.compile_mapped_collection(
            mapped.model_dump(),
            formats=[ArtifactFormat.SQL],
        )

        # Verify SQL artifacts
        sql_artifacts = [a for a in compiled.artifacts if a.format == ArtifactFormat.SQL]
        assert len(sql_artifacts) > 0

        for artifact in sql_artifacts:
            # Should contain SQL keywords
            content_upper = artifact.content.upper()
            assert "ALTER TABLE" in content_upper or "CREATE" in content_upper


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidationIntegration:
    """Integration tests for validation."""

    def test_validation_summary(self, sample_policy_document):
        """Test validation summary generation."""
        from aegislang.agents.aegis_ingestor import AegisIngestor
        from aegislang.agents.policy_parser_agent import PolicyParserAgent
        from aegislang.agents.schema_mapping_agent import (
            SchemaMappingAgent,
            create_default_registry,
        )
        from aegislang.agents.compiler_agent import CompilerAgent, ArtifactFormat
        from aegislang.agents.trace_validator_agent import TraceValidatorAgent

        # Run pipeline
        ingestor = AegisIngestor()
        ingested = ingestor.ingest(sample_policy_document)

        parser = PolicyParserAgent(use_mock=True)
        parsed = parser.parse_ingested_document(ingested.model_dump())

        mapper = SchemaMappingAgent(registry=create_default_registry(), use_mock=True)
        mapped = mapper.map_parsed_collection(parsed.model_dump())

        compiler = CompilerAgent()
        compiled = compiler.compile_mapped_collection(
            mapped.model_dump(),
            formats=[ArtifactFormat.YAML],
        )

        validator = TraceValidatorAgent()
        validated = validator.validate_compiled_collection(
            compiled.model_dump(),
            mapped.model_dump(),
            parsed.model_dump(),
        )

        # Check summary
        summary = validated.summary
        assert "total" in summary
        assert "passed" in summary
        assert "failed" in summary
        assert "needs_review" in summary
        assert summary["total"] == len(validated.results)
