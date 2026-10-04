"""
Tests for the per-agent command-line interfaces (python -m aegislang.agents.<agent>).

Runs the documented L1 -> L5 chain end to end on a small policy document.
"""

import json
import sys
from pathlib import Path

import pytest

from aegislang.agents import (
    aegis_ingestor,
    compiler_agent,
    policy_parser_agent,
    schema_mapping_agent,
    trace_validator_agent,
)

POLICY = """# Customer Due Diligence Policy

## Identification

Financial institutions must verify customer identity within 30 days.
Banks must not open anonymous accounts.

## Records

Institutions shall retain records for at least five years.
Institutions may rely on third parties to perform verification.
"""


def _run(monkeypatch, module, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", [module.__name__, *args])
    module.main()


@pytest.fixture
def chain(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    """Run the full CLI chain and return the produced files."""
    files = {
        "policy": tmp_path / "policy.md",
        "ingested": tmp_path / "ingested.json",
        "parsed": tmp_path / "parsed.json",
        "mapped": tmp_path / "mapped.json",
        "compiled": tmp_path / "compiled.json",
        "artifacts": tmp_path / "artifacts",
        "validated": tmp_path / "validated.json",
        "graph": tmp_path / "graph.json",
        "graph_dot": tmp_path / "graph.dot",
    }
    files["policy"].write_text(POLICY)

    _run(monkeypatch, aegis_ingestor, str(files["policy"]), "-o", str(files["ingested"]))
    _run(
        monkeypatch,
        policy_parser_agent,
        str(files["ingested"]),
        "--provider",
        "mock",
        "-o",
        str(files["parsed"]),
    )
    _run(
        monkeypatch,
        schema_mapping_agent,
        str(files["parsed"]),
        "--schema",
        "kyc_schema",
        "-o",
        str(files["mapped"]),
    )
    _run(
        monkeypatch,
        compiler_agent,
        str(files["mapped"]),
        "--formats",
        "yaml",
        "sql",
        "python",
        "-o",
        str(files["artifacts"]),
    )
    return files


def test_ingestor_cli_output(chain):
    ingested = json.loads(chain["ingested"].read_text())
    assert ingested["doc_id"].startswith("POLICY_")
    assert len(ingested["sections"]) >= 2


def test_parser_cli_output(chain):
    parsed = json.loads(chain["parsed"].read_text())
    types = sorted(c["type"] for c in parsed["clauses"])
    assert types == ["obligation", "obligation", "permission", "prohibition"]


def test_mapper_cli_output(chain):
    mapped = json.loads(chain["mapped"].read_text())
    assert mapped["target_schema"] == "kyc_schema"
    assert len(mapped["clauses"]) == 4


def test_compiler_cli_writes_artifacts(chain):
    written = sorted(p.suffix for p in chain["artifacts"].rglob("*") if p.is_file())
    assert written.count(".yaml") == 4
    assert written.count(".sql") == 4
    assert written.count(".py") == 4


def test_compiler_cli_dry_run(chain, monkeypatch, capsys):
    _run(monkeypatch, compiler_agent, str(chain["mapped"]), "--formats", "json", "--dry-run")
    compiled = json.loads(capsys.readouterr().out)
    assert {a["format"] for a in compiled["artifacts"]} == {"json"}
    chain["compiled"].write_text(json.dumps(compiled))


def test_validator_cli_outputs(chain, monkeypatch, capsys):
    _run(monkeypatch, compiler_agent, str(chain["mapped"]), "--formats", "yaml", "--dry-run")
    chain["compiled"].write_text(capsys.readouterr().out)

    _run(
        monkeypatch,
        trace_validator_agent,
        "--compiled",
        str(chain["compiled"]),
        "--mapped",
        str(chain["mapped"]),
        "--parsed",
        str(chain["parsed"]),
        "-o",
        str(chain["validated"]),
        "--graph",
        str(chain["graph"]),
        "--graph-dot",
        str(chain["graph_dot"]),
    )

    validated = json.loads(chain["validated"].read_text())
    assert validated["summary"]["total"] == 4
    graph = json.loads(chain["graph"].read_text())
    assert graph["nodes"] and graph["edges"]
    assert chain["graph_dot"].read_text().startswith("digraph")


def test_parser_cli_raw_text(tmp_path, monkeypatch, capsys):
    text = tmp_path / "clause.txt"
    text.write_text("Banks shall report suspicious transactions.")
    _run(monkeypatch, policy_parser_agent, str(text), "--provider", "mock", "--raw")
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["clauses"][0]["type"] == "obligation"


def test_mapper_cli_custom_registry(chain, tmp_path, monkeypatch, capsys):
    registry = schema_mapping_agent.create_default_registry()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(registry.model_dump_json())

    _run(
        monkeypatch,
        schema_mapping_agent,
        str(chain["parsed"]),
        "--registry",
        str(registry_path),
        "--threshold",
        "0.9",
    )
    mapped = json.loads(capsys.readouterr().out)
    assert len(mapped["clauses"]) == 4


@pytest.mark.parametrize(
    "module, args",
    [
        (aegis_ingestor, ["missing.md"]),
        (policy_parser_agent, ["missing.json", "--provider", "mock"]),
        (schema_mapping_agent, ["missing.json"]),
        (compiler_agent, ["missing.json"]),
        (
            trace_validator_agent,
            ["--compiled", "missing.json", "--mapped", "missing.json", "--parsed", "missing.json"],
        ),
    ],
)
def test_cli_missing_input_exits_with_error(module, args, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        _run(monkeypatch, module, *args)
    assert exc.value.code != 0
