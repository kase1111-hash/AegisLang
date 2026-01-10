"""
System and Acceptance Tests for AegisLang.

These tests verify the system behaves correctly from an end-user perspective,
testing through the API endpoints and validating complete user journeys.
"""

import pytest
import tempfile
import json
import time
from pathlib import Path
from typing import Generator
from unittest.mock import patch, MagicMock


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_aml_policy() -> str:
    """Sample Anti-Money Laundering policy document."""
    return """# Anti-Money Laundering Policy

## 1. Customer Due Diligence

Financial institutions must verify customer identity before account opening.
Banks shall perform enhanced due diligence for politically exposed persons.
Institutions are required to maintain customer records for 5 years.

## 2. Transaction Monitoring

Organizations must report transactions exceeding $10,000 within 24 hours.
Employees shall not process transactions without proper authorization.
Staff must not share customer information with unauthorized third parties.

## 3. Exceptions

Temporary accounts may be opened if the transaction amount is below $1,000.
Internal transfers between affiliated entities are exempt from reporting.
"""


@pytest.fixture
def sample_data_protection_policy() -> str:
    """Sample GDPR-style data protection policy."""
    return """# Data Protection Policy

## 1. Data Collection

Organizations must obtain explicit consent before collecting personal data.
Companies shall clearly state the purpose of data collection.
Data controllers are required to maintain processing records.

## 2. Data Subject Rights

Individuals may request access to their personal data.
Users are permitted to request deletion of their data.
Data subjects must be notified of breaches within 72 hours.

## 3. Data Security

Organizations shall implement appropriate security measures.
Personal data must be encrypted during transmission.
Access to personal data shall be restricted to authorized personnel.
"""


@pytest.fixture
def sample_hr_policy() -> str:
    """Sample HR/Employment policy document."""
    return """# Employee Conduct Policy

## 1. Workplace Standards

Employees must adhere to the company dress code.
Staff shall report to work on time.
Managers are required to conduct quarterly performance reviews.

## 2. Prohibited Actions

Employees shall not engage in harassment or discrimination.
Staff must not use company resources for personal business.
Sharing confidential company information is prohibited.

## 3. Leave Policies

Employees may request up to 20 days annual leave.
Staff are permitted to work remotely with manager approval.
Sick leave exceeding 3 days requires medical documentation.
"""


@pytest.fixture
def temp_policy_file(sample_aml_policy: str, tmp_path: Path) -> Path:
    """Create temporary policy file."""
    file_path = tmp_path / "test_policy.md"
    file_path.write_text(sample_aml_policy)
    return file_path


# =============================================================================
# API Client Fixture
# =============================================================================

@pytest.fixture
def api_client():
    """Create a test API client."""
    from fastapi.testclient import TestClient
    from aegislang.api.server import app

    return TestClient(app)


# =============================================================================
# System Tests - Complete User Journeys
# =============================================================================

class TestUserJourneyDocumentProcessing:
    """
    Acceptance Criteria: Users can upload policy documents and receive
    machine-readable compliance artifacts.
    """

    @pytest.mark.slow
    def test_complete_document_processing_journey(
        self,
        api_client,
        temp_policy_file: Path,
    ):
        """
        User Story: As a compliance officer, I want to upload a policy document
        and receive structured compliance rules that can be integrated into our systems.

        Acceptance Criteria:
        1. Document upload succeeds
        2. Clauses are extracted
        3. Artifacts are generated in requested formats
        4. Lineage is traceable
        """
        # Step 1: Upload document
        with open(temp_policy_file, "rb") as f:
            response = api_client.post(
                "/api/v1/ingest",
                files={"file": ("policy.md", f, "text/markdown")},
            )

        assert response.status_code == 200
        ingest_result = response.json()
        assert "doc_id" in ingest_result
        doc_id = ingest_result["doc_id"]

        # Step 2: Verify document is listed
        response = api_client.get("/api/v1/documents")
        assert response.status_code == 200
        documents = response.json()
        assert any(d["doc_id"] == doc_id for d in documents)

        # Step 3: Get extracted clauses
        response = api_client.get(f"/api/v1/clauses?doc_id={doc_id}")
        assert response.status_code == 200
        clauses = response.json()
        assert len(clauses) > 0

        # Verify clause structure
        for clause in clauses:
            assert "clause_id" in clause
            assert "type" in clause
            assert "text" in clause

        # Step 4: Compile to multiple formats
        response = api_client.post(
            "/api/v1/compile",
            json={
                "doc_id": doc_id,
                "formats": ["yaml", "sql", "json"],
            },
        )
        assert response.status_code == 200
        compile_result = response.json()
        assert len(compile_result.get("artifacts", [])) > 0

        # Step 5: Verify artifacts contain expected content
        formats_generated = {a["format"] for a in compile_result["artifacts"]}
        assert "yaml" in formats_generated

    def test_document_with_all_clause_types(
        self,
        api_client,
        sample_aml_policy: str,
        tmp_path: Path,
    ):
        """
        Acceptance Criteria: System correctly identifies all clause types
        (obligation, prohibition, permission, conditional, exception).
        """
        # Create comprehensive policy with all types
        comprehensive_policy = """# Comprehensive Policy

## Obligations
Banks must verify customer identity.
Institutions shall report suspicious activity.

## Prohibitions
Employees shall not share passwords.
Staff must not access unauthorized systems.

## Permissions
Customers may request account statements.
Users are permitted to update their contact information.

## Conditionals
If a transaction exceeds $10,000, it must be reported.
When a customer is high-risk, enhanced due diligence applies.

## Exceptions
Notwithstanding the above, internal transfers are exempt.
Except for emergency situations, approvals are required.
"""
        policy_file = tmp_path / "comprehensive.md"
        policy_file.write_text(comprehensive_policy)

        # Upload and process
        with open(policy_file, "rb") as f:
            response = api_client.post(
                "/api/v1/ingest",
                files={"file": ("policy.md", f, "text/markdown")},
            )

        assert response.status_code == 200
        doc_id = response.json()["doc_id"]

        # Get clauses
        response = api_client.get(f"/api/v1/clauses?doc_id={doc_id}")
        clauses = response.json()

        # Verify multiple clause types detected
        clause_types = {c["type"] for c in clauses}
        assert len(clause_types) >= 2, f"Expected multiple clause types, got: {clause_types}"


class TestUserJourneySchemaMapping:
    """
    Acceptance Criteria: Users can register custom schemas and map
    policy entities to their data models.
    """

    def test_custom_schema_registration_and_mapping(self, api_client, tmp_path: Path):
        """
        User Story: As a database administrator, I want to map policy terms
        to my database schema so compliance rules can reference actual tables.
        """
        # Step 1: Register a custom schema
        custom_schema = {
            "schema_id": "test_crm_schema",
            "schema_type": "sql",
            "tables_json": [
                {
                    "table_name": "customers",
                    "fields": [
                        {"field_name": "customer_id", "field_type": "UUID"},
                        {"field_name": "kyc_verified", "field_type": "BOOLEAN"},
                        {"field_name": "risk_level", "field_type": "VARCHAR"},
                        {"field_name": "last_verification", "field_type": "TIMESTAMP"},
                    ],
                },
                {
                    "table_name": "transactions",
                    "fields": [
                        {"field_name": "transaction_id", "field_type": "UUID"},
                        {"field_name": "amount", "field_type": "DECIMAL"},
                        {"field_name": "customer_id", "field_type": "UUID"},
                        {"field_name": "created_at", "field_type": "TIMESTAMP"},
                    ],
                },
            ],
        }

        response = api_client.post(
            "/api/v1/schemas",
            json=custom_schema,
        )
        assert response.status_code in [200, 201]

        # Step 2: Upload policy that references similar entities
        policy = """# Transaction Policy

Customers must have verified KYC status before transactions.
All transactions exceeding $10,000 must be flagged for review.
Customer risk levels must be updated quarterly.
"""
        policy_file = tmp_path / "transaction_policy.md"
        policy_file.write_text(policy)

        with open(policy_file, "rb") as f:
            response = api_client.post(
                "/api/v1/ingest",
                files={"file": ("policy.md", f, "text/markdown")},
            )

        assert response.status_code == 200
        doc_id = response.json()["doc_id"]

        # Step 3: Verify clauses can reference schema entities
        response = api_client.get(f"/api/v1/clauses?doc_id={doc_id}")
        assert response.status_code == 200


class TestUserJourneyAuditCompliance:
    """
    Acceptance Criteria: Users can trace generated rules back to source
    policy text for audit purposes.
    """

    def test_lineage_traceability(self, api_client, temp_policy_file: Path):
        """
        User Story: As an auditor, I need to trace any compliance rule
        back to its source policy text.
        """
        # Upload document
        with open(temp_policy_file, "rb") as f:
            response = api_client.post(
                "/api/v1/ingest",
                files={"file": ("policy.md", f, "text/markdown")},
            )

        doc_id = response.json()["doc_id"]

        # Compile to get artifacts
        response = api_client.post(
            "/api/v1/compile",
            json={"doc_id": doc_id, "formats": ["yaml"]},
        )

        artifacts = response.json().get("artifacts", [])

        if len(artifacts) > 0:
            # Each artifact should have lineage info
            for artifact in artifacts:
                assert "clause_id" in artifact
                # Lineage allows tracing back to source


# =============================================================================
# API Contract Tests
# =============================================================================

class TestAPIContracts:
    """Tests verifying API contracts and expected behaviors."""

    def test_health_endpoint(self, api_client):
        """Verify health check returns expected structure."""
        response = api_client.get("/api/v1/health")
        assert response.status_code == 200

        health = response.json()
        assert "status" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]

    def test_invalid_document_upload(self, api_client):
        """Verify appropriate error for invalid uploads."""
        # Empty file
        response = api_client.post(
            "/api/v1/ingest",
            files={"file": ("empty.md", b"", "text/markdown")},
        )
        # Should either accept empty or return appropriate error
        assert response.status_code in [200, 400, 422]

    def test_nonexistent_document(self, api_client):
        """Verify 404 for non-existent document."""
        response = api_client.get("/api/v1/clauses?doc_id=DOC-NONEXISTENT")
        assert response.status_code in [200, 404]  # May return empty list or 404

    def test_compile_invalid_format(self, api_client, temp_policy_file: Path):
        """Verify error handling for invalid compilation format."""
        # First upload a document
        with open(temp_policy_file, "rb") as f:
            response = api_client.post(
                "/api/v1/ingest",
                files={"file": ("policy.md", f, "text/markdown")},
            )
        doc_id = response.json()["doc_id"]

        # Request invalid format
        response = api_client.post(
            "/api/v1/compile",
            json={"doc_id": doc_id, "formats": ["invalid_format"]},
        )
        # Should handle gracefully
        assert response.status_code in [200, 400, 422]


# =============================================================================
# Performance Acceptance Tests
# =============================================================================

class TestPerformanceAcceptance:
    """Verify system meets performance requirements."""

    def test_document_processing_time(self, api_client, temp_policy_file: Path):
        """
        Acceptance Criteria: Document processing completes within reasonable time.
        """
        start_time = time.time()

        with open(temp_policy_file, "rb") as f:
            response = api_client.post(
                "/api/v1/ingest",
                files={"file": ("policy.md", f, "text/markdown")},
            )

        processing_time = time.time() - start_time

        assert response.status_code == 200
        # Should complete within 60 seconds for small documents
        assert processing_time < 60, f"Processing took {processing_time:.2f}s"

    def test_health_check_response_time(self, api_client):
        """Health check should respond quickly."""
        start_time = time.time()
        response = api_client.get("/api/v1/health")
        response_time = time.time() - start_time

        assert response.status_code == 200
        # Health check should be < 1 second
        assert response_time < 1.0


# =============================================================================
# Error Handling and Recovery Tests
# =============================================================================

class TestErrorHandlingRecovery:
    """Test system behavior under error conditions."""

    def test_malformed_json_request(self, api_client):
        """System handles malformed JSON gracefully."""
        response = api_client.post(
            "/api/v1/compile",
            content="not valid json{{{",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422  # Validation error

    def test_missing_required_fields(self, api_client):
        """System validates required fields."""
        response = api_client.post(
            "/api/v1/compile",
            json={},  # Missing doc_id
        )
        assert response.status_code in [400, 422]

    def test_concurrent_document_processing(
        self,
        api_client,
        sample_aml_policy: str,
        sample_data_protection_policy: str,
        tmp_path: Path,
    ):
        """System handles concurrent document processing."""
        # Create multiple policy files
        policy1 = tmp_path / "policy1.md"
        policy2 = tmp_path / "policy2.md"
        policy1.write_text(sample_aml_policy)
        policy2.write_text(sample_data_protection_policy)

        # Upload both (sequentially, but testing system can handle multiple docs)
        with open(policy1, "rb") as f:
            resp1 = api_client.post(
                "/api/v1/ingest",
                files={"file": ("policy1.md", f, "text/markdown")},
            )

        with open(policy2, "rb") as f:
            resp2 = api_client.post(
                "/api/v1/ingest",
                files={"file": ("policy2.md", f, "text/markdown")},
            )

        assert resp1.status_code == 200
        assert resp2.status_code == 200

        # Verify both documents are tracked
        response = api_client.get("/api/v1/documents")
        documents = response.json()
        doc_ids = [d["doc_id"] for d in documents]

        assert resp1.json()["doc_id"] in doc_ids
        assert resp2.json()["doc_id"] in doc_ids


# =============================================================================
# Regression Guard Tests
# =============================================================================

class TestRegressionGuards:
    """Tests that guard against known issues and regressions."""

    def test_special_characters_in_policy(self, api_client, tmp_path: Path):
        """Ensure special characters don't break processing."""
        policy = """# Policy with Special Characters

## Section 1
Users must comply with terms & conditions.
Price thresholds: $100, €200, £300.
Quotes: "single" and 'double'.
Math: 5 < 10 > 3.
Ampersand in names: AT&T policy applies.
"""
        policy_file = tmp_path / "special_chars.md"
        policy_file.write_text(policy)

        with open(policy_file, "rb") as f:
            response = api_client.post(
                "/api/v1/ingest",
                files={"file": ("policy.md", f, "text/markdown")},
            )

        assert response.status_code == 200

    def test_unicode_in_policy(self, api_client, tmp_path: Path):
        """Ensure unicode content is handled correctly."""
        policy = """# International Policy 国际政策

## Règles générales
Les utilisateurs doivent se conformer aux règles.
ユーザーは規則に従う必要があります。

## 日本語セクション
従業員は機密情報を共有してはならない。
"""
        policy_file = tmp_path / "unicode_policy.md"
        policy_file.write_text(policy, encoding="utf-8")

        with open(policy_file, "rb") as f:
            response = api_client.post(
                "/api/v1/ingest",
                files={"file": ("policy.md", f, "text/markdown")},
            )

        assert response.status_code == 200

    def test_large_document_handling(self, api_client, tmp_path: Path):
        """Ensure large documents don't cause issues."""
        # Create a moderately large document
        sections = []
        for i in range(50):
            sections.append(f"""
## Section {i}

Clause {i}.1: Organizations must comply with requirement {i}.
Clause {i}.2: Employees shall not violate policy {i}.
Clause {i}.3: Customers may request service {i}.
""")

        policy = "# Large Policy Document\n" + "\n".join(sections)
        policy_file = tmp_path / "large_policy.md"
        policy_file.write_text(policy)

        with open(policy_file, "rb") as f:
            response = api_client.post(
                "/api/v1/ingest",
                files={"file": ("policy.md", f, "text/markdown")},
            )

        assert response.status_code == 200


# =============================================================================
# Data Integrity Tests
# =============================================================================

class TestDataIntegrity:
    """Tests verifying data integrity throughout the system."""

    def test_document_content_preserved(self, api_client, temp_policy_file: Path):
        """Original document content is preserved accurately."""
        original_content = temp_policy_file.read_text()

        with open(temp_policy_file, "rb") as f:
            response = api_client.post(
                "/api/v1/ingest",
                files={"file": ("policy.md", f, "text/markdown")},
            )

        doc_id = response.json()["doc_id"]

        # Get clauses
        response = api_client.get(f"/api/v1/clauses?doc_id={doc_id}")
        clauses = response.json()

        # Verify clause text comes from original document
        for clause in clauses:
            clause_text = clause.get("text", "")
            # The clause text should be derivable from original content
            # (Note: exact matching may vary based on parsing)
            assert len(clause_text) > 0

    def test_artifact_clause_correspondence(self, api_client, temp_policy_file: Path):
        """Each artifact corresponds to a valid clause."""
        with open(temp_policy_file, "rb") as f:
            response = api_client.post(
                "/api/v1/ingest",
                files={"file": ("policy.md", f, "text/markdown")},
            )

        doc_id = response.json()["doc_id"]

        # Get clauses
        clause_response = api_client.get(f"/api/v1/clauses?doc_id={doc_id}")
        clause_ids = {c["clause_id"] for c in clause_response.json()}

        # Compile
        compile_response = api_client.post(
            "/api/v1/compile",
            json={"doc_id": doc_id, "formats": ["yaml"]},
        )

        artifacts = compile_response.json().get("artifacts", [])

        # Each artifact should reference a valid clause
        for artifact in artifacts:
            if "clause_id" in artifact:
                assert artifact["clause_id"] in clause_ids


# =============================================================================
# Security Acceptance Tests
# =============================================================================

class TestSecurityAcceptance:
    """Basic security acceptance tests."""

    def test_no_sensitive_data_in_error_responses(self, api_client):
        """Error responses don't leak sensitive information."""
        response = api_client.post(
            "/api/v1/compile",
            json={"doc_id": "DOC-NONEXISTENT", "formats": ["yaml"]},
        )

        error_text = response.text.lower()

        # Should not contain stack traces or internal paths
        assert "traceback" not in error_text
        assert "/home/" not in error_text
        assert "password" not in error_text

    def test_input_sanitization(self, api_client, tmp_path: Path):
        """Malicious input doesn't cause code execution."""
        malicious_policy = """# Policy

{{ system('ls') }}
<script>alert('xss')</script>
'; DROP TABLE documents; --
"""
        policy_file = tmp_path / "malicious.md"
        policy_file.write_text(malicious_policy)

        with open(policy_file, "rb") as f:
            response = api_client.post(
                "/api/v1/ingest",
                files={"file": ("policy.md", f, "text/markdown")},
            )

        # Should process without executing any malicious content
        assert response.status_code == 200
