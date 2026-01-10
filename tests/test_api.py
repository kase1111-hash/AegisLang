"""
API tests for AegisLang REST API.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import json


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def client():
    """Create test client."""
    from aegislang.api.server import app
    return TestClient(app)


@pytest.fixture
def sample_markdown_content():
    """Sample markdown content for upload."""
    return """# Test Policy

## Requirements

Financial institutions must verify customer identity.
Banks shall report suspicious transactions.
"""


@pytest.fixture
def sample_markdown_file(tmp_path, sample_markdown_content):
    """Create sample markdown file."""
    file_path = tmp_path / "test_policy.md"
    file_path.write_text(sample_markdown_content)
    return file_path


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health check returns healthy status."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data

    def test_health_check_version(self, client):
        """Test health check returns version."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "1.0.0"


# =============================================================================
# Document Ingestion Tests
# =============================================================================

class TestDocumentIngestion:
    """Tests for document ingestion endpoint."""

    def test_ingest_markdown_file(self, client, sample_markdown_file):
        """Test ingesting a markdown file."""
        with open(sample_markdown_file, "rb") as f:
            response = client.post(
                "/api/v1/ingest",
                files={"file": ("test_policy.md", f, "text/markdown")},
                data={"metadata": json.dumps({"document_name": "Test Policy"})},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert "job_id" in data
        assert "doc_id" in data
        assert "webhook_url" in data

    def test_ingest_with_metadata(self, client, sample_markdown_file):
        """Test ingestion with custom metadata."""
        metadata = {
            "document_name": "AML Policy",
            "document_type": "regulation",
            "jurisdiction": "US",
        }

        with open(sample_markdown_file, "rb") as f:
            response = client.post(
                "/api/v1/ingest",
                files={"file": ("policy.md", f, "text/markdown")},
                data={"metadata": json.dumps(metadata)},
            )

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data

    def test_ingest_unsupported_format(self, client, tmp_path):
        """Test ingestion with unsupported file format."""
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_text("content")

        with open(unsupported_file, "rb") as f:
            response = client.post(
                "/api/v1/ingest",
                files={"file": ("test.xyz", f, "application/octet-stream")},
            )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data


# =============================================================================
# Job Status Tests
# =============================================================================

class TestJobStatus:
    """Tests for job status endpoint."""

    def test_get_job_status(self, client, sample_markdown_file):
        """Test getting job status after ingestion."""
        # First, create a job
        with open(sample_markdown_file, "rb") as f:
            ingest_response = client.post(
                "/api/v1/ingest",
                files={"file": ("test.md", f, "text/markdown")},
            )

        job_id = ingest_response.json()["job_id"]

        # Get job status
        response = client.get(f"/api/v1/jobs/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "status" in data
        assert "created_at" in data

    def test_get_nonexistent_job(self, client):
        """Test getting non-existent job."""
        response = client.get("/api/v1/jobs/nonexistent_job_123")

        assert response.status_code == 404


# =============================================================================
# Document Retrieval Tests
# =============================================================================

class TestDocumentRetrieval:
    """Tests for document retrieval endpoints."""

    def test_get_nonexistent_document(self, client):
        """Test getting non-existent document."""
        response = client.get("/api/v1/documents/NONEXISTENT_DOC")

        assert response.status_code == 404

    def test_get_nonexistent_clauses(self, client):
        """Test getting clauses for non-existent document."""
        response = client.get("/api/v1/clauses/NONEXISTENT_DOC")

        assert response.status_code == 404


# =============================================================================
# Schema Registry Tests
# =============================================================================

class TestSchemaRegistry:
    """Tests for schema registry endpoints."""

    def test_register_schema(self, client):
        """Test registering a new schema."""
        schema_data = {
            "schema_id": "test_schema_001",
            "schema_type": "sql",
            "version": "1.0.0",
            "tables": [
                {
                    "table_name": "customers",
                    "fields": [
                        {"field_name": "id", "field_type": "UUID"},
                        {"field_name": "name", "field_type": "VARCHAR"},
                    ],
                }
            ],
        }

        response = client.post("/api/v1/schemas", json=schema_data)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "registered"
        assert data["schema_id"] == "test_schema_001"

    def test_list_schemas(self, client):
        """Test listing schemas."""
        # Register a schema first
        schema_data = {
            "schema_id": "list_test_schema",
            "schema_type": "sql",
            "version": "1.0.0",
            "tables": [],
        }
        client.post("/api/v1/schemas", json=schema_data)

        # List schemas
        response = client.get("/api/v1/schemas")

        assert response.status_code == 200
        data = response.json()
        assert "schemas" in data
        assert "count" in data

    def test_get_schema(self, client):
        """Test getting a specific schema."""
        # Register a schema first
        schema_data = {
            "schema_id": "get_test_schema",
            "schema_type": "sql",
            "version": "2.0.0",
            "tables": [],
        }
        client.post("/api/v1/schemas", json=schema_data)

        # Get the schema
        response = client.get("/api/v1/schemas/get_test_schema")

        assert response.status_code == 200
        data = response.json()
        assert data["schema_id"] == "get_test_schema"
        assert data["version"] == "2.0.0"

    def test_get_nonexistent_schema(self, client):
        """Test getting non-existent schema."""
        response = client.get("/api/v1/schemas/nonexistent_schema")

        assert response.status_code == 404


# =============================================================================
# Compilation Tests
# =============================================================================

class TestCompilation:
    """Tests for compilation endpoint."""

    def test_compile_nonexistent_document(self, client):
        """Test compiling non-existent document."""
        response = client.post(
            "/api/v1/compile",
            json={
                "doc_id": "NONEXISTENT_DOC",
                "output_formats": ["yaml"],
            },
        )

        assert response.status_code == 404


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_json_metadata(self, client, sample_markdown_file):
        """Test ingestion with invalid JSON metadata."""
        with open(sample_markdown_file, "rb") as f:
            response = client.post(
                "/api/v1/ingest",
                files={"file": ("test.md", f, "text/markdown")},
                data={"metadata": "not valid json{"},
            )

        # Should still accept (invalid JSON becomes empty dict)
        assert response.status_code == 200

    def test_missing_file(self, client):
        """Test ingestion without file."""
        response = client.post("/api/v1/ingest")

        assert response.status_code == 422  # Validation error


# =============================================================================
# OpenAPI Documentation Tests
# =============================================================================

class TestOpenAPI:
    """Tests for OpenAPI documentation."""

    def test_openapi_schema(self, client):
        """Test OpenAPI schema is available."""
        response = client.get("/api/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
        assert "info" in data

    def test_swagger_docs(self, client):
        """Test Swagger docs are available."""
        response = client.get("/api/docs")

        assert response.status_code == 200

    def test_redoc_docs(self, client):
        """Test ReDoc docs are available."""
        response = client.get("/api/redoc")

        assert response.status_code == 200
