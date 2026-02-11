# AegisLang

**Language in. Compliance out.**

AegisLang is a multi-agent semantic compiler that transforms natural-language policy documents into executable control logic with full clause-to-artifact traceability. Feed it a regulation (PDF, DOCX, Markdown, or HTML), and it produces YAML rules, SQL constraints, and Python compliance tests — each linked back to its source clause.

## How It Works

```
policy_doc → AegisIngestor → PolicyParser → SchemaMapper → Compiler → TraceValidator
```

| Stage | Agent | Input | Output |
|-------|-------|-------|--------|
| **L1 Ingest** | `AegisIngestor` | Raw document (PDF/DOCX/MD/HTML) | Structured sections + text chunks |
| **L2 Parse** | `PolicyParserAgent` | Text chunks | Typed clauses (obligation/prohibition/permission/conditional) |
| **L3 Map** | `SchemaMappingAgent` | Parsed clauses | Entity-to-schema-field mappings |
| **L4 Compile** | `CompilerAgent` | Mapped clauses | YAML, SQL, Python artifacts via Jinja2 templates |
| **L5 Validate** | `TraceValidatorAgent` | Artifacts + source data | Provenance traces + validation results |

## Current Status

**Version: 0.1.0 (Alpha)**

| What works | What doesn't (yet) |
|------------|-------------------|
| Full 5-stage pipeline end-to-end | Real LLM extraction not validated on production docs |
| Mock LLM mode for offline dev/testing | Schema mapping depends on LLM entity extraction quality |
| PDF, DOCX, Markdown, HTML ingestion | No persistent storage (in-memory only) |
| YAML, SQL, Python artifact generation | SQL artifacts reference generic tables without mapping |
| Clause-to-artifact traceability (100%) | Cross-reference resolution between clauses |
| REST API with OpenAPI docs | No web UI |
| 139 tests, all passing | No output regression tests from real documents |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run the API server
python -m aegislang.api.server

# Or run the AML pipeline demo
python examples/run_aml_pipeline.py
```

The API serves at `http://localhost:8080` with Swagger docs at `/docs`.

## Supported Domain: AML/KYC

AegisLang has been evaluated against Anti-Money Laundering / Know Your Customer regulations:

- **FinCEN CDD Rule** (31 CFR 1010.230)
- **FFIEC BSA/AML CIP Manual**
- **FATF Recommendation 10**

Pipeline results (mock LLM mode):

| Metric | Result |
|--------|--------|
| Documents processed | 3 |
| Clauses extracted | 49 |
| Artifacts generated | 147 (YAML + SQL + Python) |
| Clause type detection | 40 obligation, 5 permission, 2 prohibition, 2 conditional |
| Traceability | 100% — every artifact links to source clause |

See [`examples/aml_evaluation.md`](examples/aml_evaluation.md) for the full quality assessment.

## Architecture

```
aegislang/
├── agents/                  # Pipeline agents (L1-L5)
│   ├── aegis_ingestor.py    # L1: Document ingestion + chunking
│   ├── policy_parser_agent.py   # L2: Clause extraction (LLM-driven)
│   ├── schema_mapping_agent.py  # L3: Entity-to-schema mapping
│   ├── compiler_agent.py    # L4: Artifact generation (Jinja2)
│   └── trace_validator_agent.py # L5: Provenance validation
├── api/
│   └── server.py            # FastAPI REST API
├── core/
│   ├── errors.py            # Error handling
│   └── logging.py           # Structured logging (structlog)
└── templates/               # Jinja2 compilation templates
    ├── yaml/
    ├── sql/
    └── python/
```

Each agent is independently testable with mock providers. The `use_mock=True` flag on parser and mapper agents enables offline development without LLM API keys.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/ingest` | Upload and ingest a document |
| `GET` | `/api/v1/documents` | List all documents |
| `GET` | `/api/v1/documents/{doc_id}` | Get document details |
| `GET` | `/api/v1/clauses/{doc_id}` | Get extracted clauses |
| `POST` | `/api/v1/compile` | Compile document to artifacts |
| `POST` | `/api/v1/schemas` | Register a target schema |
| `GET` | `/api/v1/schemas` | List registered schemas |
| `GET` | `/api/v1/jobs/{job_id}` | Check async job status |

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for planned features.

## Contributing

```bash
# Run tests
python -m pytest tests/ -v

# Install optional ML dependencies (for SentenceTransformer embeddings)
pip install -r requirements-ml.txt
```

## License

MIT
