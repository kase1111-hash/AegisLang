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
| **L2 Parse** | `PolicyParserAgent` | Text chunks | Typed clauses (obligation/prohibition/permission/conditional/definition/exception) |
| **L3 Map** | `SchemaMappingAgent` | Parsed clauses | Entity-to-schema-field mappings |
| **L4 Compile** | `CompilerAgent` | Mapped clauses | YAML, SQL, Python artifacts via Jinja2 templates |
| **L5 Validate** | `TraceValidatorAgent` | Artifacts + source data | Validation results + provenance graph |

## Current Status

**Version: 0.1.0 (Alpha)**

| What works | What doesn't (yet) |
|------------|-------------------|
| Full 5-stage pipeline end-to-end (library, per-agent CLIs, REST API) | Real LLM extraction not validated on production docs (the AML results below use the mock parser) |
| Mock (regex-based) parser for offline dev/testing | Schema mapping in the API and CLI uses mock embeddings (deterministic hashes, not semantic); real embedding providers only via library code |
| PDF, DOCX, Markdown, HTML ingestion | No OCR; PDF section hierarchy only from `#`-style heading lines |
| YAML, SQL, Python artifact generation | SQL artifacts fall back to a generic `compliance_table` when entities don't map |
| Clause-to-artifact traceability (validation results + provenance graph via `GET /api/v1/trace/{doc_id}`) | Cross-reference resolution between clauses (not implemented; `cross_references` is always empty) |
| REST API with OpenAPI docs, API-key auth, rate limiting | No web UI; no user accounts or roles |
| SQLite persistence (opt-in via `AEGISLANG_STORAGE_BACKEND=sqlite`) | Default in-memory storage is lost on restart |
| 252 tests, all passing, including output regression tests on the real AML documents | Terraform/Rego/JSON output is experimental and CLI/library only |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run the API server (local development, no API key needed)
AEGISLANG_DISABLE_AUTH=true python -m aegislang.api.server

# Or require API keys (clients send X-API-Key: my-secret-key)
AEGISLANG_API_KEYS=my-secret-key python -m aegislang.api.server

# Or run the AML pipeline demo (no server needed)
python examples/run_aml_pipeline.py
```

The API serves at `http://localhost:8080` with Swagger docs at `/api/docs` (ReDoc at `/api/redoc`). If you set neither `AEGISLANG_DISABLE_AUTH=true` nor `AEGISLANG_API_KEYS`, the server generates a random development key at startup and logs it (look for `no_api_keys_configured`). Python does not read `.env` files, so export variables in your shell.

Without `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`, the API parses clauses with the mock parser; `GET /api/v1/health` reports which provider is active (`llm_provider`). See [`docs/API.md`](docs/API.md) for a complete ingest → compile → clauses/rules/trace walkthrough.

### Docker

```bash
echo "AEGISLANG_API_KEYS=change-me" > .env   # optionally add ANTHROPIC_API_KEY=...
docker compose up -d
curl http://localhost:8080/api/v1/health
```

`docker compose` reads `.env` and passes the keys to the container. The `aegislang` service runs a single worker with SQLite storage on the `aegislang-data` volume. A hot-reload dev server (auth disabled by default, port 8081) is available with `docker compose --profile dev up aegislang-dev`.

## Supported Domain: AML/KYC

AegisLang has been evaluated against Anti-Money Laundering / Know Your Customer regulations:

- **FinCEN CDD Rule** (31 CFR 1010.230)
- **FFIEC BSA/AML CIP Manual**
- **FATF Recommendation 10**

Pipeline results (mock LLM mode, `python examples/run_aml_pipeline.py`):

| Metric | Result |
|--------|--------|
| Documents processed | 3 |
| Clauses extracted | 49 |
| Artifacts generated | 147 (49 each YAML + SQL + Python), all syntax-valid |
| Clause type detection | 40 obligation, 5 permission, 4 prohibition (0 conditional, definition or exception) |
| Schema mapping | 1 fully mapped, 8 partially mapped, 40 unmapped (18.4% with any mapping) |
| Validation | 147 traces: 123 passed, 0 failed, 24 needs review |
| Traceability | 100% — every artifact links to its source clause (document → section → chunk → clause → mapping → artifact) |

Low mapping coverage is expected with mock embeddings. See [`examples/aml_evaluation.md`](examples/aml_evaluation.md) for the full quality assessment.

## Architecture

```
aegislang/
├── agents/                    # Pipeline agents (L1-L5)
│   ├── aegis_ingestor.py      # L1: Document ingestion + chunking
│   ├── policy_parser_agent.py # L2: Clause extraction (LLM or mock parser)
│   ├── schema_mapping_agent.py# L3: Entity-to-schema mapping
│   ├── compiler_agent.py      # L4: Artifact generation (built-in Jinja2 templates)
│   └── trace_validator_agent.py # L5: Provenance validation
├── api/
│   ├── server.py              # FastAPI REST API
│   └── sqlite_storage.py     # SQLite persistent storage backend
└── core/
    ├── errors.py              # Error handling
    └── logging.py             # Structured logging (structlog)

templates/                     # Optional Jinja2 template overrides (project root)
├── yaml/
├── sql/
└── python/
```

The built-in templates live in `compiler_agent.py`. The `templates/` directory is not loaded by default: pass `CompilerAgent(templates_dir=Path("templates"))` in library code or `--templates templates` to the compiler CLI. The REST API always uses the built-in templates.

Each agent is independently testable and has its own CLI (`python -m aegislang.agents.<agent> --help`). Only the parser calls an LLM: `PolicyParserAgent(use_mock=True)` swaps in the regex-based mock parser, and the API does this automatically when no `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is set. The mapper never calls an LLM. It always uses deterministic mock embeddings unless you pass an `embedding_provider` (e.g. `SentenceTransformerProvider`, which needs `requirements-ml.txt`) in library code.

## API Endpoints

All endpoints except `/api/v1/health` require an `X-API-Key` header (unless auth is disabled).

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Health check (version, active LLM provider) |
| `POST` | `/api/v1/ingest` | Upload a document (async ingestion job) |
| `GET` | `/api/v1/documents` | List all documents |
| `GET` | `/api/v1/documents/{doc_id}` | Get document details |
| `POST` | `/api/v1/compile` | Parse, map, compile and validate a document (async job) |
| `GET` | `/api/v1/jobs/{job_id}` | Check async job status |
| `GET` | `/api/v1/jobs/{job_id}/stream` | Stream job status (Server-Sent Events) |
| `GET` | `/api/v1/clauses/{doc_id}` | Get extracted clauses (after a compile job) |
| `GET` | `/api/v1/rules/{clause_id}` | Get generated artifacts for a clause |
| `GET` | `/api/v1/trace/{doc_id}` | Get validation results and provenance graph |
| `POST` | `/api/v1/schemas` | Register or replace a target schema |
| `GET` | `/api/v1/schemas` | List registered schemas |
| `GET` | `/api/v1/schemas/{schema_id}` | Get a registered schema |

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for planned features.

## Contributing

```bash
# Install runtime + dev dependencies (pytest, pytest-cov, pytest-asyncio, httpx, black, ruff, mypy)
make dev-install

# Run tests
python -m pytest tests/ -v

# Install optional ML dependencies (for SentenceTransformer embeddings)
pip install -r requirements-ml.txt
```

`pytest` and `httpx` are not in `requirements.txt`, so run `make dev-install` before running the tests. `make check-all` runs lint (Ruff), type checking (MyPy), the Bandit security check and the tests; all of them pass, and CI runs the same Ruff, Black and MyPy checks.

## License

MIT
