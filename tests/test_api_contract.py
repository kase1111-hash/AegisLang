"""
Tests that the REST API behaves as documented in docs/API.md.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


POLICY = """# Test Policy

## Requirements

Financial institutions must verify customer identity.
Banks shall report suspicious transactions.
"""


@pytest.fixture
def client():
    from aegislang.api.server import app
    return TestClient(app)


@pytest.fixture
def policy_file(tmp_path: Path) -> Path:
    path = tmp_path / "policy.md"
    path.write_text(POLICY)
    return path


def _ingest(client: TestClient, policy_file: Path) -> str:
    with open(policy_file, "rb") as f:
        response = client.post(
            "/api/v1/ingest", files={"file": ("policy.md", f, "text/markdown")}
        )
    assert response.status_code == 200
    doc_id = response.json()["doc_id"]
    job = client.get(response.json()["status_url"]).json()
    assert job["status"] == "completed", job
    return doc_id


def _compile(client: TestClient, doc_id: str, **kwargs) -> dict:
    response = client.post("/api/v1/compile", json={"doc_id": doc_id, **kwargs})
    assert response.status_code == 200, response.json()
    job = client.get(response.json()["status_url"]).json()
    assert job["status"] == "completed", job
    return job


class TestAuthentication:
    def test_development_key_is_stable(self, monkeypatch):
        """The generated development key must keep working across requests."""
        from aegislang.api import server

        monkeypatch.delenv("AEGISLANG_DISABLE_AUTH", raising=False)
        monkeypatch.delenv("AEGISLANG_API_KEYS", raising=False)

        first = server.get_valid_api_keys()
        second = server.get_valid_api_keys()
        assert first == second
        assert len(first) == 1

        client = TestClient(server.app)
        key = next(iter(first))
        assert client.get("/api/v1/schemas", headers={"X-API-Key": key}).status_code == 200
        assert client.get("/api/v1/schemas").status_code == 401
        assert client.get("/api/v1/schemas", headers={"X-API-Key": "wrong"}).status_code == 403

    def test_configured_keys(self, monkeypatch):
        from aegislang.api import server

        monkeypatch.delenv("AEGISLANG_DISABLE_AUTH", raising=False)
        monkeypatch.setenv("AEGISLANG_API_KEYS", "key1, key2")
        client = TestClient(server.app)

        assert client.get("/api/v1/schemas", headers={"X-API-Key": "key2"}).status_code == 200
        assert client.get("/api/v1/health").status_code == 200


class TestErrorFormat:
    def test_unknown_route_uses_error_format(self, client):
        response = client.get("/api/v1/does-not-exist")
        assert response.status_code == 404
        body = response.json()
        assert body["status_code"] == 404
        assert "error" in body

    def test_validation_error_uses_error_format(self, client):
        response = client.post("/api/v1/compile", json={})
        assert response.status_code == 422
        body = response.json()
        assert body["status_code"] == 422
        assert body["error_code"] == "VALIDATION_ERROR"
        assert body["details"]["errors"]

    def test_rate_limit_sets_retry_after(self, client, monkeypatch):
        from aegislang.api.server import _rate_limiter

        monkeypatch.setattr(_rate_limiter, "requests_per_minute", 1)
        assert client.get("/api/v1/schemas").status_code == 200
        response = client.get("/api/v1/schemas")
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "60"
        assert response.json()["status_code"] == 429


class TestIngestValidation:
    def test_metadata_must_be_object(self, client, policy_file):
        with open(policy_file, "rb") as f:
            response = client.post(
                "/api/v1/ingest",
                files={"file": ("policy.md", f, "text/markdown")},
                data={"metadata": "[1, 2]"},
            )
        assert response.status_code == 400

    def test_metadata_is_stored(self, client, policy_file):
        with open(policy_file, "rb") as f:
            response = client.post(
                "/api/v1/ingest",
                files={"file": ("policy.md", f, "text/markdown")},
                data={"metadata": '{"document_name": "AML Policy", "jurisdiction": "US"}'},
            )
        doc_id = response.json()["doc_id"]
        doc = client.get(f"/api/v1/documents/{doc_id}").json()
        assert doc["metadata"]["document_name"] == "AML Policy"
        assert doc["metadata"]["jurisdiction"] == "US"


class TestCompileValidation:
    def test_unsupported_output_format_rejected(self, client, policy_file):
        doc_id = _ingest(client, policy_file)
        response = client.post(
            "/api/v1/compile", json={"doc_id": doc_id, "output_formats": ["xml"]}
        )
        assert response.status_code == 422

    def test_unknown_target_schema_rejected(self, client, policy_file):
        doc_id = _ingest(client, policy_file)
        response = client.post(
            "/api/v1/compile", json={"doc_id": doc_id, "target_schema": "no_such_schema"}
        )
        assert response.status_code == 404

    def test_built_in_target_schema_accepted(self, client, policy_file):
        doc_id = _ingest(client, policy_file)
        _compile(client, doc_id, target_schema="kyc_schema")


class TestSchemas:
    def test_invalid_schema_type_rejected(self, client):
        response = client.post(
            "/api/v1/schemas",
            json={"schema_id": "bad", "schema_type": "bogus", "tables": []},
        )
        assert response.status_code == 422

    def test_invalid_table_definition_rejected(self, client):
        response = client.post(
            "/api/v1/schemas",
            json={"schema_id": "bad", "schema_type": "sql", "tables": [{"anything": 1}]},
        )
        assert response.status_code == 422

    def test_registered_schema_is_used_for_mapping(self, client, policy_file):
        response = client.post(
            "/api/v1/schemas",
            json={
                "schema_id": "contract_custom_schema",
                "schema_type": "sql",
                "tables": [
                    {
                        "table_name": "bank_customer",
                        "fields": [
                            {
                                "field_name": "customer_identity",
                                "field_type": "VARCHAR",
                                "semantic_labels": ["customer identity"],
                            },
                            {
                                "field_name": "institution",
                                "field_type": "VARCHAR",
                                "semantic_labels": ["financial institutions", "banks"],
                            },
                        ],
                    }
                ],
            },
        )
        assert response.status_code == 200

        from aegislang.api.server import build_schema_registry, storage

        registry = build_schema_registry(storage)
        schema_ids = {s.schema_id for s in registry.schemas}
        assert "contract_custom_schema" in schema_ids
        assert "kyc_schema" in schema_ids

        from aegislang.agents.schema_mapping_agent import SchemaMappingAgent

        doc_id = _ingest(client, policy_file)
        _compile(client, doc_id, target_schema="contract_custom_schema")

        clauses = client.get(f"/api/v1/clauses/{doc_id}").json()["clauses"]
        mapper = SchemaMappingAgent(registry=registry, use_mock=True)
        mapped = mapper.map_parsed_collection(
            {"doc_id": doc_id, "clauses": clauses}, "contract_custom_schema"
        )
        targets = {
            (m.target_schema, m.target_path)
            for clause in mapped.clauses
            for m in clause.mapped_entities
        }
        assert ("contract_custom_schema", "bank_customer.customer_identity") in targets


class TestRules:
    def test_rule_returns_all_artifacts_for_clause(self, client, policy_file):
        doc_id = _ingest(client, policy_file)
        _compile(client, doc_id, output_formats=["yaml", "sql", "python"])

        clauses = client.get(f"/api/v1/clauses/{doc_id}").json()["clauses"]
        assert clauses
        clause_id = clauses[0]["clause_id"]

        rule = client.get(f"/api/v1/rules/{clause_id}").json()
        formats = sorted(a["format"] for a in rule["artifacts"])
        assert formats == ["python", "sql", "yaml"]


class TestSqliteBackend:
    def test_full_pipeline_with_sqlite(self, client, policy_file, tmp_path, monkeypatch):
        from aegislang.api import server
        from aegislang.api.sqlite_storage import SqliteStorage

        sqlite_storage = SqliteStorage(db_path=str(tmp_path / "test.db"))
        monkeypatch.setattr(server, "storage", sqlite_storage)
        server.app.dependency_overrides[server.get_storage] = lambda: sqlite_storage
        try:
            doc_id = _ingest(client, policy_file)
            job = _compile(client, doc_id, output_formats=["yaml"])
            assert job["result"]["clauses_parsed"] > 0

            clauses = client.get(f"/api/v1/clauses/{doc_id}").json()
            assert clauses["clause_count"] == job["result"]["clauses_parsed"]
            assert len(client.get("/api/v1/documents").json()) == 1
        finally:
            server.app.dependency_overrides.clear()
            sqlite_storage.close()

    def test_empty_clause_list_is_present(self, tmp_path):
        from aegislang.api.sqlite_storage import SqliteStorage

        sqlite_storage = SqliteStorage(db_path=str(tmp_path / "test.db"))
        sqlite_storage.store_clauses("DOC", [])
        assert "DOC" in sqlite_storage.clauses
        assert sqlite_storage.clauses["DOC"] == []
        sqlite_storage.close()

    def test_data_persists_across_instances(self, tmp_path):
        from aegislang.api.sqlite_storage import SqliteStorage

        db_path = str(tmp_path / "test.db")
        first = SqliteStorage(db_path=db_path)
        first.store_document("DOC", {"doc_id": "DOC", "sections": []})
        first.store_artifacts("DOC", [{"artifact_id": "A1"}, {"artifact_id": "A2"}])
        first.close()

        second = SqliteStorage(db_path=db_path)
        assert second.documents["DOC"]["doc_id"] == "DOC"
        assert [a["artifact_id"] for a in second.artifacts["DOC"]] == ["A1", "A2"]
        second.close()
