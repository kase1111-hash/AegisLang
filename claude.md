# AegisLang

AegisLang is a natural language policy compiler for compliance automation. It transforms regulatory and policy documents (PDF, DOCX, Markdown, HTML) into executable control logic while maintaining traceability from source clause to generated code. Version 0.1.0 (Alpha).

## Quick Reference

```bash
# Install dependencies (requirements.txt + pytest, black, ruff, mypy, httpx)
make dev-install

# Run the API server (port 8080, in-memory storage)
make run

# Run tests
make test                 # All tests
make test-cov             # With coverage (fails under 70%)
make test-fast            # Skip tests marked slow

# Code quality (all must pass; CI runs ruff, black --check and mypy)
make lint                 # Run Ruff linter
make format               # Format with Black
make type-check           # MyPy (non-blocking in CI)
make security-check       # Bandit + Safety (Bandit passes)
make check-all            # lint, type-check, security-check, test

# Docker
make docker-up            # Start the API on :8080 (SQLite on a named volume)
make docker-up-dev        # Also start the hot-reload dev server on :8081 (profile "dev")
make docker-down          # Stop services
```

## Project Structure

```
aegislang/                 # Main source code
├── agents/               # Multi-agent pipeline (L1-L5); __init__.py is empty, import from submodules
│   ├── aegis_ingestor.py        # L1: Document ingestion
│   ├── policy_parser_agent.py   # L2: Semantic parsing
│   ├── schema_mapping_agent.py  # L3: Entity mapping
│   ├── compiler_agent.py        # L4: Code generation
│   └── trace_validator_agent.py # L5: Validation & lineage
├── api/                  # FastAPI REST server
│   ├── server.py                # Endpoints, auth, rate limiting, in-memory storage
│   └── sqlite_storage.py        # Optional SQLite storage backend
├── core/                 # Shared utilities
│   ├── errors.py                # Error types and the API error format
│   └── logging.py               # structlog setup, request context, Sentry
└── config/               # Empty package (configuration is via environment variables)

templates/                # Optional Jinja2 template overrides (not loaded by default)
├── yaml/                 # YAML compliance rules
├── sql/                  # SQL constraints & triggers
└── python/               # Python validators

examples/                 # AML/KYC regulations, run_aml_pipeline.py, generated output

tests/                    # Test suite
├── test_*.py             # Unit, API, CLI, system and regression tests
├── test_integration.py   # Integration tests
├── test_pipeline_regression.py  # AML pipeline output regression
└── performance/          # Load tests (Locust; not part of the normal run)
```

## Architecture

5-layer multi-agent pipeline:

| Layer | Agent | Purpose |
|-------|-------|---------|
| L1 | `aegis_ingestor.py` | Parse documents, chunk text, extract metadata |
| L2 | `policy_parser_agent.py` | Extract semantic clauses (obligation, prohibition, permission, conditional, definition, exception) via Anthropic, OpenAI or the keyword-based mock parser |
| L3 | `schema_mapping_agent.py` | Map entities to target schemas (exact, synonym, embedding similarity; deterministic mock embeddings unless a provider is passed in code) |
| L4 | `compiler_agent.py` | Generate artifacts (YAML, SQL, Python; experimental Terraform, Rego, JSON) |
| L5 | `trace_validator_agent.py` | Verify traceability, emit provenance graph |

### Output Formats

- **REST API** (`POST /api/v1/compile`): `yaml`, `sql`, `python` only (anything else is a 422)
- **CLI / library** (`python -m aegislang.agents.compiler_agent --formats ...`): additionally `terraform`, `rego`, `json` (experimental: Terraform emits a placeholder policy resource; syntax checks for these are basic)

Each agent also has a CLI (`python -m aegislang.agents.<agent> --help`); CLIs log to stderr and write JSON to stdout or `-o`.

## Coding Standards

- **Python 3.11+** required
- **Type hints** required for all functions. `pyproject.toml` enables strict-leaning MyPy flags (e.g. `disallow_untyped_defs`, `disallow_untyped_decorators`), though not `strict = true`. `mypy aegislang/` is clean and blocking in CI; run it in an environment with `requirements.txt` installed so FastAPI's types are visible
- **Docstrings** Google-style for public APIs
- **Line length** 100 characters
- **Formatting** Black (`make format`); Ruff for linting
- **Lint/format are clean**: `ruff check .` and `black --check .` pass. Rule exceptions are scoped in `pyproject.toml` with a reason (e.g. lazy imports of optional dependencies, FastAPI auth dependencies); add new ones the same way rather than with blanket ignores

### Patterns

- Use **Pydantic models** for all data schemas with Field descriptions
- Each agent is a discrete processing layer with defined input/output types
- Built-in **Jinja2 templates** are Python strings in `compiler_agent.py`. `templates/` holds optional overrides, loaded only with `CompilerAgent(templates_dir=...)` or `--templates`. Templates render in a Jinja2 `SandboxedEnvironment`
- All transformations include **confidence scores**
- Use **structlog** for logging with context variables (request IDs are bound by the API middleware)

### Example Model Pattern

```python
class TextChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Chunk text content")
    token_count: int = Field(..., description="Number of tokens in chunk")
    embedding_vector: list[float] | None = Field(default=None)
```

## Testing

```bash
pytest tests/                    # Run all tests (252)
pytest tests/test_parser.py      # Single test file
pytest -m "not slow"             # Skip slow tests
pytest --cov=aegislang           # With coverage
```

Test markers:
- `@pytest.mark.slow` - Long-running tests (the only marker currently used)
- `pyproject.toml` also declares `integration`, `unit` and `security`, but no tests use them; select tests by file instead (`make test-unit`, `make test-integration`)

Coverage target: **70% minimum**, enforced by `fail_under` in `pyproject.toml` (currently ~73%)

## Environment Variables

The Python code does not load `.env`; Docker Compose substitutes values from it. `config.yaml` is a reference file only (never loaded).

- `ANTHROPIC_API_KEY` - Claude API key (optional; the API uses Anthropic when set)
- `OPENAI_API_KEY` - OpenAI API key (optional; used when only this key is set). With neither key, the mock parser is used
- `AEGISLANG_API_KEYS` - Comma-separated API keys for the `X-API-Key` header (if unset, a random dev key is generated and logged at startup)
- `AEGISLANG_DISABLE_AUTH` - `true` disables authentication (development only)
- `AEGISLANG_RATE_LIMIT_MINUTE` / `AEGISLANG_RATE_LIMIT_HOUR` - Per-key rate limits (60 / 1000)
- `AEGISLANG_STORAGE_BACKEND` - `memory` (default; forces 1 worker) or `sqlite`
- `AEGISLANG_SQLITE_PATH` - SQLite file (default `aegislang_data.db`)
- `AEGISLANG_JOB_TTL_SECONDS` - Finished-job retention (default 86400)
- `AEGISLANG_MAX_FILE_SIZE` - Upload limit in bytes (default 50 MB)
- `AEGISLANG_ENV` - `production` hides internal error details and logs JSON
- `LOG_LEVEL` / `AEGISLANG_LOG_LEVEL` - Log level (`AEGISLANG_LOG_LEVEL` wins); `AEGISLANG_LOG_FILE` adds a JSON log file
- `SENTRY_DSN` - Enables Sentry; `AEGISLANG_VERSION` sets the reported release
- `CORS_ORIGINS` - Comma-separated allowed origins (default `http://localhost:3000`)
- `HOST`, `PORT`, `WORKERS`, `RELOAD` - Server settings (0.0.0.0, 8080, 4, false)

## Services (Docker)

- **aegislang** (8080) - REST API server; SQLite storage on the `aegislang-data` volume
- **aegislang-dev** (8081, profile `dev`) - Hot-reload server with source mounted read-only; auth disabled by default

No PostgreSQL, Redis or Neo4j is used. Run a single instance only (the rate limiter is in-memory and SQLite is a local file).

## Key Files

- `Makefile` - Build automation commands
- `pyproject.toml` - Python config, tool settings
- `config.yaml` - Reference configuration (not loaded by the code)
- `docker-compose.yml` - Service orchestration
- `requirements.txt` - Production dependencies
- `examples/run_aml_pipeline.py` - Runs the full pipeline on the AML/KYC example documents
