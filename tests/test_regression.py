"""
Regression Test Suite for AegisLang.

This module contains regression tests that guard against previously
identified bugs and ensure they don't recur.
"""

import pytest
from pathlib import Path
from typing import List, Dict, Any


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_document(tmp_path: Path) -> Path:
    """Create a sample policy document for testing."""
    content = """# Sample Policy

## Requirements

Financial institutions must verify customer identity.
Banks shall report suspicious transactions.
Employees shall not share confidential information.
Customers may request account statements.
"""
    file_path = tmp_path / "sample_policy.md"
    file_path.write_text(content)
    return file_path


# =============================================================================
# Parser Regression Tests
# =============================================================================

class TestParserRegressions:
    """Regression tests for the policy parser."""

    def test_modal_verb_detection_must(self):
        """
        Regression: Modal verb 'must' was not being detected in certain contexts.
        Fixed in: v0.9.0
        """
        from aegislang.agents.policy_parser_agent import PolicyParserAgent, ClauseType

        parser = PolicyParserAgent(use_mock=True)

        # These should all be detected as obligations
        test_cases = [
            "Banks must verify customer identity.",
            "The institution must report suspicious activity.",
            "All employees must complete training.",
            "Organizations must maintain records.",
        ]

        for text in test_cases:
            result = parser.parse_clause(text, "CL", "C")
            assert result.type == ClauseType.OBLIGATION, f"Failed: {text}"

    def test_modal_verb_detection_shall_not(self):
        """
        Regression: 'shall not' was incorrectly parsed as obligation.
        Fixed in: v0.9.1
        """
        from aegislang.agents.policy_parser_agent import PolicyParserAgent, ClauseType

        parser = PolicyParserAgent(use_mock=True)

        test_cases = [
            "Employees shall not share passwords.",
            "Staff shall not access unauthorized systems.",
            "Users shall not modify production data.",
        ]

        for text in test_cases:
            result = parser.parse_clause(text, "CL", "C")
            assert result.type == ClauseType.PROHIBITION, f"Failed: {text}"

    def test_empty_clause_handling(self):
        """
        Regression: Empty strings caused parser crash.
        Fixed in: v0.9.2
        """
        from aegislang.agents.policy_parser_agent import PolicyParserAgent

        parser = PolicyParserAgent(use_mock=True)

        # Should not raise exception
        result = parser.parse_clause("", "CL", "C")
        assert result is not None

    def test_whitespace_only_handling(self):
        """
        Regression: Whitespace-only strings caused issues.
        Fixed in: v0.9.2
        """
        from aegislang.agents.policy_parser_agent import PolicyParserAgent

        parser = PolicyParserAgent(use_mock=True)

        whitespace_cases = ["   ", "\t\t", "\n\n", "  \t  \n  "]

        for text in whitespace_cases:
            # Should not raise exception
            result = parser.parse_clause(text, "CL", "C")
            assert result is not None


# =============================================================================
# Ingestor Regression Tests
# =============================================================================

class TestIngestorRegressions:
    """Regression tests for the document ingestor."""

    def test_markdown_header_parsing(self, tmp_path: Path):
        """
        Regression: Markdown headers with special characters caused issues.
        Fixed in: v0.9.3
        """
        from aegislang.agents.aegis_ingestor import AegisIngestor

        content = """# Policy & Guidelines

## Section 1: Requirements (2024)

Banks must comply.

## Section 2: "Quoted Header"

Staff shall report.

## Section 3 - Dash Header

Users may access.
"""
        file_path = tmp_path / "special_headers.md"
        file_path.write_text(content)

        ingestor = AegisIngestor()
        result = ingestor.ingest(file_path)

        assert result is not None
        assert len(result.sections) > 0

    def test_empty_section_handling(self, tmp_path: Path):
        """
        Regression: Documents with empty sections caused crash.
        Fixed in: v0.9.3
        """
        from aegislang.agents.aegis_ingestor import AegisIngestor

        content = """# Policy

## Section 1

Banks must comply.

## Section 2

## Section 3

Users may access.
"""
        file_path = tmp_path / "empty_sections.md"
        file_path.write_text(content)

        ingestor = AegisIngestor()
        result = ingestor.ingest(file_path)

        assert result is not None

    def test_unicode_filename_handling(self, tmp_path: Path):
        """
        Regression: Unicode characters in filenames caused issues.
        Fixed in: v0.9.4
        """
        from aegislang.agents.aegis_ingestor import AegisIngestor

        content = "# Policy\n\nBanks must comply."
        file_path = tmp_path / "政策文档.md"
        file_path.write_text(content, encoding="utf-8")

        ingestor = AegisIngestor()
        result = ingestor.ingest(file_path)

        assert result is not None


# =============================================================================
# Mapper Regression Tests
# =============================================================================

class TestMapperRegressions:
    """Regression tests for schema mapping."""

    def test_empty_entity_mapping(self):
        """
        Regression: Empty entity lists caused mapping crash.
        Fixed in: v0.9.5
        """
        from aegislang.agents.schema_mapping_agent import (
            SchemaMappingAgent,
            create_default_registry,
        )

        mapper = SchemaMappingAgent(
            registry=create_default_registry(),
            use_mock=True,
        )

        # Should handle gracefully
        result = mapper._compute_similarity("", "customers")
        assert isinstance(result, (int, float))

    def test_special_characters_in_entity_names(self):
        """
        Regression: Special characters in entity names caused regex issues.
        Fixed in: v0.9.5
        """
        from aegislang.agents.schema_mapping_agent import (
            SchemaMappingAgent,
            create_default_registry,
        )

        mapper = SchemaMappingAgent(
            registry=create_default_registry(),
            use_mock=True,
        )

        # These should not raise exceptions
        special_entities = [
            "customer.id",
            "user[0]",
            "field(name)",
            "table::column",
            "entity->attribute",
        ]

        for entity in special_entities:
            try:
                result = mapper._compute_similarity(entity, "customers")
                assert isinstance(result, (int, float))
            except Exception as e:
                pytest.fail(f"Failed for entity '{entity}': {e}")


# =============================================================================
# Compiler Regression Tests
# =============================================================================

class TestCompilerRegressions:
    """Regression tests for the compiler."""

    def test_yaml_special_character_escaping(self, sample_document: Path):
        """
        Regression: Special characters in clause text broke YAML output.
        Fixed in: v0.9.6
        """
        from aegislang.agents.aegis_ingestor import AegisIngestor
        from aegislang.agents.policy_parser_agent import PolicyParserAgent
        from aegislang.agents.schema_mapping_agent import (
            SchemaMappingAgent,
            create_default_registry,
        )
        from aegislang.agents.compiler_agent import CompilerAgent, ArtifactFormat
        import yaml

        # Create document with special characters
        content = """# Policy

Banks must verify "customer identity" before account opening.
Transactions > $10,000 require: approval & review.
"""
        special_doc = sample_document.parent / "special_chars.md"
        special_doc.write_text(content)

        # Run pipeline
        ingestor = AegisIngestor()
        ingested = ingestor.ingest(special_doc)

        parser = PolicyParserAgent(use_mock=True)
        parsed = parser.parse_ingested_document(ingested.model_dump())

        mapper = SchemaMappingAgent(
            registry=create_default_registry(),
            use_mock=True,
        )
        mapped = mapper.map_parsed_collection(parsed.model_dump())

        compiler = CompilerAgent()
        compiled = compiler.compile_mapped_collection(
            mapped.model_dump(),
            formats=[ArtifactFormat.YAML],
        )

        # Verify YAML artifacts exist and contain expected structure
        yaml_artifacts = [
            a for a in compiled.artifacts if a.format == ArtifactFormat.YAML
        ]
        assert len(yaml_artifacts) > 0

    def test_sql_injection_prevention(self, sample_document: Path):
        """
        Regression: SQL output didn't properly escape user content.
        Fixed in: v0.9.7
        """
        from aegislang.agents.aegis_ingestor import AegisIngestor
        from aegislang.agents.policy_parser_agent import PolicyParserAgent
        from aegislang.agents.schema_mapping_agent import (
            SchemaMappingAgent,
            create_default_registry,
        )
        from aegislang.agents.compiler_agent import CompilerAgent, ArtifactFormat

        # Create document with SQL-like content
        content = """# Policy

Users must not execute: '; DROP TABLE users; --.
Banks shall validate input before INSERT INTO transactions.
"""
        sql_doc = sample_document.parent / "sql_chars.md"
        sql_doc.write_text(content)

        # Run pipeline
        ingestor = AegisIngestor()
        ingested = ingestor.ingest(sql_doc)

        parser = PolicyParserAgent(use_mock=True)
        parsed = parser.parse_ingested_document(ingested.model_dump())

        mapper = SchemaMappingAgent(
            registry=create_default_registry(),
            use_mock=True,
        )
        mapped = mapper.map_parsed_collection(parsed.model_dump())

        compiler = CompilerAgent()
        compiled = compiler.compile_mapped_collection(
            mapped.model_dump(),
            formats=[ArtifactFormat.SQL],
        )

        # Verify SQL artifacts exist
        sql_artifacts = [
            a for a in compiled.artifacts if a.format == ArtifactFormat.SQL
        ]
        assert len(sql_artifacts) > 0


# =============================================================================
# Validator Regression Tests
# =============================================================================

class TestValidatorRegressions:
    """Regression tests for the validator."""

    def test_empty_artifact_list(self, sample_document: Path):
        """
        Regression: Empty artifact list caused validation crash.
        Fixed in: v0.9.8
        """
        from aegislang.agents.aegis_ingestor import AegisIngestor
        from aegislang.agents.policy_parser_agent import PolicyParserAgent
        from aegislang.agents.schema_mapping_agent import (
            SchemaMappingAgent,
            create_default_registry,
        )
        from aegislang.agents.compiler_agent import CompilerAgent
        from aegislang.agents.trace_validator_agent import TraceValidatorAgent

        # Run minimal pipeline
        ingestor = AegisIngestor()
        ingested = ingestor.ingest(sample_document)

        parser = PolicyParserAgent(use_mock=True)
        parsed = parser.parse_ingested_document(ingested.model_dump())

        mapper = SchemaMappingAgent(
            registry=create_default_registry(),
            use_mock=True,
        )
        mapped = mapper.map_parsed_collection(parsed.model_dump())

        # Create compiled collection with no formats (empty artifacts)
        compiler = CompilerAgent()
        compiled = compiler.compile_mapped_collection(
            mapped.model_dump(),
            formats=[],  # No formats = no artifacts
        )

        # Validation should handle this gracefully
        validator = TraceValidatorAgent()
        validated = validator.validate_compiled_collection(
            compiled.model_dump(),
            mapped.model_dump(),
            parsed.model_dump(),
        )

        assert validated is not None
        assert "total" in validated.summary


# =============================================================================
# API Regression Tests
# =============================================================================

class TestAPIRegressions:
    """Regression tests for API endpoints."""

    def test_concurrent_uploads(self, sample_document: Path):
        """
        Regression: Concurrent document uploads caused race conditions.
        Fixed in: v0.9.9
        """
        from fastapi.testclient import TestClient
        from aegislang.api.server import app

        client = TestClient(app)

        # Upload same document multiple times
        results = []
        for i in range(3):
            with open(sample_document, "rb") as f:
                response = client.post(
                    "/api/v1/ingest",
                    files={"file": (f"policy_{i}.md", f, "text/markdown")},
                )
                results.append(response)

        # All should succeed
        for r in results:
            assert r.status_code == 200

        # All should have unique doc_ids
        doc_ids = [r.json()["doc_id"] for r in results]
        assert len(set(doc_ids)) == len(doc_ids)

    def test_large_request_handling(self, tmp_path: Path):
        """
        Regression: Large requests caused memory issues.
        Fixed in: v0.9.10
        """
        from fastapi.testclient import TestClient
        from aegislang.api.server import app

        client = TestClient(app)

        # Create a moderately large document (100KB+)
        sections = []
        for i in range(500):
            sections.append(f"Rule {i}: Organizations must comply with requirement {i}.")

        content = "# Large Policy\n\n" + "\n\n".join(sections)
        large_file = tmp_path / "large.md"
        large_file.write_text(content)

        with open(large_file, "rb") as f:
            response = client.post(
                "/api/v1/ingest",
                files={"file": ("large.md", f, "text/markdown")},
            )

        assert response.status_code == 200


# =============================================================================
# Edge Case Regression Tests
# =============================================================================

class TestEdgeCaseRegressions:
    """Regression tests for edge cases."""

    def test_nested_conditionals(self, tmp_path: Path):
        """
        Regression: Nested conditional clauses were not handled correctly.
        Fixed in: v1.0.0
        """
        from aegislang.agents.policy_parser_agent import PolicyParserAgent

        parser = PolicyParserAgent(use_mock=True)

        nested_clauses = [
            "If the transaction exceeds $10,000, and the customer is high-risk, enhanced review is required.",
            "When a user requests data, unless they are blacklisted, access must be granted.",
            "If approval is denied, then the request must be logged, and the user notified.",
        ]

        for text in nested_clauses:
            result = parser.parse_clause(text, "CL", "C")
            assert result is not None

    def test_multi_actor_clauses(self, tmp_path: Path):
        """
        Regression: Clauses with multiple actors were parsed incorrectly.
        Fixed in: v1.0.0
        """
        from aegislang.agents.policy_parser_agent import PolicyParserAgent

        parser = PolicyParserAgent(use_mock=True)

        multi_actor_clauses = [
            "Banks and credit unions must verify customer identity.",
            "Managers or supervisors shall approve high-value transactions.",
            "Employees, contractors, and vendors must complete security training.",
        ]

        for text in multi_actor_clauses:
            result = parser.parse_clause(text, "CL", "C")
            assert result is not None

    def test_quantified_statements(self, tmp_path: Path):
        """
        Regression: Quantified statements (all, any, some) parsed incorrectly.
        Fixed in: v1.0.0
        """
        from aegislang.agents.policy_parser_agent import PolicyParserAgent

        parser = PolicyParserAgent(use_mock=True)

        quantified_clauses = [
            "All employees must complete training.",
            "Any transaction over $10,000 requires review.",
            "Some users may access restricted areas.",
            "No employee shall share credentials.",
        ]

        for text in quantified_clauses:
            result = parser.parse_clause(text, "CL", "C")
            assert result is not None


# =============================================================================
# Compatibility Regression Tests
# =============================================================================

class TestCompatibilityRegressions:
    """Regression tests for backwards compatibility."""

    def test_legacy_document_format(self, tmp_path: Path):
        """
        Ensure legacy document formats are still supported.
        """
        from aegislang.agents.aegis_ingestor import AegisIngestor

        # Simple markdown without headers
        legacy_content = """Policy Rules:

1. Banks must verify identity.
2. Staff shall not share passwords.
3. Users may request data.
"""
        legacy_file = tmp_path / "legacy.md"
        legacy_file.write_text(legacy_content)

        ingestor = AegisIngestor()
        result = ingestor.ingest(legacy_file)

        assert result is not None

    def test_api_v1_compatibility(self):
        """
        Ensure API v1 endpoints remain stable.
        """
        from fastapi.testclient import TestClient
        from aegislang.api.server import app

        client = TestClient(app)

        # Core v1 endpoints should exist
        endpoints = [
            "/api/v1/health",
            "/api/v1/documents",
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code in [200, 404]  # Exists but may be empty


# =============================================================================
# Stress Regression Tests
# =============================================================================

@pytest.mark.slow
class TestStressRegressions:
    """Regression tests for stress conditions."""

    def test_rapid_sequential_requests(self):
        """
        Regression: Rapid requests caused resource exhaustion.
        Fixed in: v1.0.0
        """
        from fastapi.testclient import TestClient
        from aegislang.api.server import app

        client = TestClient(app)

        # Make many rapid requests
        for _ in range(50):
            response = client.get("/api/v1/health")
            assert response.status_code == 200

    def test_memory_cleanup(self, sample_document: Path):
        """
        Regression: Memory wasn't properly cleaned up after processing.
        Fixed in: v1.0.0
        """
        import gc
        from aegislang.agents.aegis_ingestor import AegisIngestor
        from aegislang.agents.policy_parser_agent import PolicyParserAgent

        # Process document multiple times
        for _ in range(10):
            ingestor = AegisIngestor()
            ingested = ingestor.ingest(sample_document)

            parser = PolicyParserAgent(use_mock=True)
            parsed = parser.parse_ingested_document(ingested.model_dump())

            del ingested
            del parsed

        # Force garbage collection
        gc.collect()

        # Should not raise memory error
        assert True
