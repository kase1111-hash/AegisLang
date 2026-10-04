"""
Tests that pipeline behaviour matches the feature descriptions in SPEC.md.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aegislang.agents.aegis_ingestor import AegisIngestor
from aegislang.agents.compiler_agent import ArtifactFormat, CompilerAgent
from aegislang.agents.policy_parser_agent import MockLLMClient, PolicyParserAgent
from aegislang.agents.schema_mapping_agent import (
    SchemaField,
    SchemaMappingAgent,
    SchemaTable,
    SchemaType,
    TargetSchema,
    create_default_registry,
)
from aegislang.agents.trace_validator_agent import TraceValidatorAgent

REPO_ROOT = Path(__file__).parent.parent


# =============================================================================
# PRS-001: clause type taxonomy (SPEC section 3.2)
# =============================================================================


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Banks must verify customer identity.", "obligation"),
        ("Banks shall report suspicious transactions.", "obligation"),
        ("The bank is required to keep records.", "obligation"),
        ("Banks must not open anonymous accounts.", "prohibition"),
        ("Staff shall not disclose a filing.", "prohibition"),
        ("Institutions should not open the account.", "prohibition"),
        ("The bank is prohibited from opening accounts.", "prohibition"),
        ("Customers may request data deletion.", "permission"),
        ("The bank is permitted to rely on third parties.", "permission"),
        ("Institutions can rely on third parties.", "permission"),
        ("If the amount exceeds $10,000, a report must be filed.", "conditional"),
        ("When risk is high, enhanced due diligence must be applied.", "conditional"),
        ("Where lower risks exist, simplified measures apply.", "conditional"),
        ("Unless exempt, banks must file reports.", "conditional"),
        ("Customer means any individual who holds an account.", "definition"),
        ("Beneficial owner refers to the natural person who owns the entity.", "definition"),
        ("A covered institution is defined as a bank.", "definition"),
        ("Notwithstanding paragraph (a), banks need not verify existing customers.", "exception"),
    ],
)
def test_mock_parser_follows_taxonomy(text, expected):
    assert MockLLMClient().parse_clause(text)["type"] == expected


@pytest.mark.parametrize(
    "text",
    [
        "Banks will verify the customer.",  # "verify" contains "if"
        "Records are kept by means of a ledger.",  # "by means of" is not "means"
    ],
)
def test_keywords_match_whole_words_only(text):
    result = MockLLMClient().parse_clause(text)
    assert result["type"] not in ("conditional", "definition")


# =============================================================================
# PRS-006: temporal scope extraction
# =============================================================================


@pytest.mark.parametrize(
    "text, field, value",
    [
        ("Records must be retained for at least five years.", "duration", "five years"),
        ("Records shall be kept for a period of 5 years.", "duration", "5 years"),
        ("Reports must be filed within 30 calendar days.", "deadline", "30 calendar days"),
        ("Institutions shall review accounts annually.", "frequency", "annually"),
    ],
)
def test_temporal_scope_extracted(text, field, value):
    temporal = MockLLMClient().parse_clause(text)["temporal_scope"]
    assert temporal is not None
    assert temporal[field] == value


def test_temporal_scope_ignores_non_temporal_by():
    text = "Institutions may post a notice in the lobby or on the website."
    assert MockLLMClient().parse_clause(text)["temporal_scope"] is None


# =============================================================================
# ING-002 / ING-007: DOCX hierarchy
# =============================================================================


def test_docx_deep_headings_are_clamped(tmp_path):
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_heading("Policy", level=1)
    document.add_heading("Deep section", level=8)
    document.add_paragraph("Banks must verify customer identity.")
    path = tmp_path / "deep.docx"
    document.save(path)

    result = AegisIngestor().ingest(path)
    assert all(s.hierarchy_level <= 6 for s in result.sections)


def test_document_ids_are_stable(tmp_path):
    path = tmp_path / "policy.md"
    path.write_text("# Policy\n\nBanks must verify customer identity.\n")

    first = AegisIngestor().ingest(path).doc_id
    second = AegisIngestor().ingest(path).doc_id
    assert first == second

    path.write_text("# Policy\n\nBanks must report suspicious activity.\n")
    assert AegisIngestor().ingest(path).doc_id != first


# =============================================================================
# MAP-004: schema replacement
# =============================================================================


def test_replacing_schema_removes_stale_fields():
    mapper = SchemaMappingAgent(use_mock=True)
    mapper.register_schema(
        TargetSchema(
            schema_id="x",
            schema_type=SchemaType.SQL,
            version="1.0.0",
            tables=[
                SchemaTable(
                    table_name="t",
                    fields=[
                        SchemaField(field_name="old_field", field_type="TEXT"),
                    ],
                )
            ],
        )
    )
    mapper.register_schema(
        TargetSchema(
            schema_id="x",
            schema_type=SchemaType.SQL,
            version="2.0.0",
            tables=[
                SchemaTable(
                    table_name="t",
                    fields=[
                        SchemaField(field_name="new_field", field_type="TEXT"),
                    ],
                )
            ],
        )
    )

    assert "x:t.old_field" not in mapper._field_embeddings
    assert "x:t.new_field" in mapper._field_embeddings


# =============================================================================
# L4 compilation
# =============================================================================


def _compile_policy(tmp_path, formats, compiler=None):
    path = tmp_path / "policy.md"
    path.write_text(
        "# Policy\n\n## Rules\n\n"
        "Banks must verify customer identity.\n"
        "Banks must not open anonymous accounts.\n"
        "Customers may request copies of their records.\n"
    )
    doc = AegisIngestor().ingest(path).model_dump()
    parsed = PolicyParserAgent(use_mock=True).parse_ingested_document(doc).model_dump()
    mapped = (
        SchemaMappingAgent(registry=create_default_registry(), use_mock=True)
        .map_parsed_collection(parsed)
        .model_dump()
    )
    compiled = (compiler or CompilerAgent()).compile_mapped_collection(mapped, formats)
    return parsed, mapped, compiled


def test_artifacts_use_plain_clause_types(tmp_path):
    """model_dump() collections must not render enum reprs into artifacts."""
    _, _, compiled = _compile_policy(
        tmp_path, [ArtifactFormat.YAML, ArtifactFormat.SQL, ArtifactFormat.PYTHON]
    )
    for artifact in compiled.artifacts:
        assert "ClauseType." not in artifact.content
        assert "ClauseType." not in artifact.template_used


def test_sql_comment_only_for_generated_constraints(tmp_path):
    _, _, compiled = _compile_policy(tmp_path, [ArtifactFormat.SQL])
    for artifact in compiled.artifacts:
        if "ADD CONSTRAINT" not in artifact.content:
            assert "COMMENT ON CONSTRAINT" not in artifact.content


def test_template_directory_overrides_by_clause_type(tmp_path):
    templates = tmp_path / "templates" / "yaml"
    templates.mkdir(parents=True)
    (templates / "obligation.yaml.j2").write_text("custom: true\nclause: {{ clause.clause_id }}\n")
    compiler = CompilerAgent(templates_dir=tmp_path / "templates")
    _, _, compiled = _compile_policy(tmp_path, [ArtifactFormat.YAML], compiler)

    obligation = [a for a in compiled.artifacts if a.template_used == "yaml/obligation"]
    assert obligation
    assert all(a.content.startswith("custom: true") for a in obligation)


def test_repository_templates_render(tmp_path):
    compiler = CompilerAgent(templates_dir=REPO_ROOT / "templates")
    _, _, compiled = _compile_policy(
        tmp_path, [ArtifactFormat.YAML, ArtifactFormat.SQL, ArtifactFormat.PYTHON], compiler
    )
    assert compiled.artifacts
    assert all(a.syntax_valid for a in compiled.artifacts)


def test_templates_are_sandboxed():
    from jinja2.exceptions import SecurityError

    registry = CompilerAgent().template_registry
    with pytest.raises(SecurityError):
        registry.render("{{ ''.__class__.__mro__[1].__subclasses__() }}", {})


# =============================================================================
# VAL-005: lineage
# =============================================================================


def test_lineage_section_id_matches_source_section(tmp_path):
    parsed, mapped, compiled = _compile_policy(tmp_path, [ArtifactFormat.YAML])
    validated = TraceValidatorAgent().validate_compiled_collection(
        compiled.model_dump(), mapped, parsed
    )
    chunk_to_section = {}
    path = tmp_path / "policy.md"
    for section in AegisIngestor().ingest(path).sections:
        for chunk in section.text_chunks:
            chunk_to_section[chunk.chunk_id] = section.section_id

    assert validated.results
    for result in validated.results:
        assert result.lineage.section_id == chunk_to_section[result.lineage.chunk_id]


# =============================================================================
# API: provider selection, health, trace endpoint
# =============================================================================


@pytest.fixture
def client():
    from aegislang.api.server import app

    return TestClient(app)


def test_health_reports_llm_provider(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert client.get("/api/v1/health").json()["llm_provider"] == "mock"

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert client.get("/api/v1/health").json()["llm_provider"] == "openai"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert client.get("/api/v1/health").json()["llm_provider"] == "anthropic"


def test_openai_key_selects_openai_parser(monkeypatch):
    from aegislang.api import server

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    created = {}

    class FakeParser:
        def __init__(self, llm_provider="anthropic", use_mock=False, **kwargs):
            created["provider"] = llm_provider
            created["use_mock"] = use_mock
            raise RuntimeError("stop after construction")

    monkeypatch.setattr("aegislang.agents.policy_parser_agent.PolicyParserAgent", FakeParser)
    storage = server.Storage()
    storage.store_document("DOC", {"doc_id": "DOC", "sections": [], "metadata": {}})
    job_id = storage.create_job("cmp")
    server.process_compilation(job_id, "DOC", ["yaml"], None, 0.85, storage)

    assert created == {"provider": "openai", "use_mock": False}


def test_trace_endpoint_returns_lineage(client, tmp_path):
    path = tmp_path / "policy.md"
    path.write_text("# Policy\n\n## Rules\n\nBanks must verify customer identity.\n")
    with open(path, "rb") as f:
        doc_id = client.post(
            "/api/v1/ingest", files={"file": ("policy.md", f, "text/markdown")}
        ).json()["doc_id"]

    assert client.get(f"/api/v1/trace/{doc_id}").status_code == 404

    client.post("/api/v1/compile", json={"doc_id": doc_id, "output_formats": ["yaml"]})
    trace = client.get(f"/api/v1/trace/{doc_id}").json()

    assert trace["doc_id"] == doc_id
    assert trace["summary"]["total"] == len(trace["results"]) > 0
    lineage = trace["results"][0]["lineage"]
    assert lineage["section_id"].startswith("POLICY_S")
    node_types = {n["node_type"] for n in trace["provenance_graph"]["nodes"]}
    assert {"document", "clause", "artifact"} <= node_types

    doc = client.get(f"/api/v1/documents/{doc_id}").json()
    assert doc["metadata"]["source_file"] == "policy.md"


# =============================================================================
# LLM providers: calls must match the installed SDK signatures
# =============================================================================


class _FakeTextBlock:
    type = "text"
    text = '{"type": "obligation", "actor": {"entity": "Banks"}, "confidence": 0.9}'


def test_anthropic_client_matches_installed_sdk(monkeypatch):
    import inspect

    anthropic = pytest.importorskip("anthropic")
    from aegislang.agents.policy_parser_agent import AnthropicClient

    client = AnthropicClient(api_key="sk-ant-test")
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return type("Message", (), {"content": [_FakeTextBlock()]})()

    monkeypatch.setattr(client.client.messages, "create", fake_create)
    result = client.parse_clause("Banks must verify customer identity.")

    accepted = inspect.signature(anthropic.resources.messages.Messages.create).parameters
    assert set(captured) <= set(accepted), set(captured) - set(accepted)
    assert captured["extra_body"] == {"temperature": 0.1}
    assert result["type"] == "obligation"

    client.temperature = None
    client.parse_clause("Banks must verify customer identity.")
    assert captured["extra_body"] is None


def test_openai_client_matches_installed_sdk(monkeypatch):
    import inspect

    openai = pytest.importorskip("openai")
    from aegislang.agents.policy_parser_agent import OpenAIClient

    client = OpenAIClient(api_key="sk-test")
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        message = type("Msg", (), {"content": _FakeTextBlock.text})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()

    monkeypatch.setattr(client.client.chat.completions, "create", fake_create)
    result = client.parse_clause("Banks must verify customer identity.")

    accepted = inspect.signature(openai.resources.chat.completions.Completions.create).parameters
    assert set(captured) <= set(accepted), set(captured) - set(accepted)
    assert result["type"] == "obligation"
