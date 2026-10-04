# Changelog

All notable changes to AegisLang will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Fixes that make the code behave as `SPEC.md` and the documentation describe.

### Added
- `GET /api/v1/trace/{doc_id}`: validation results and provenance graph for a compiled document (persisted with the SQLite backend)
- `llm_provider` field in `GET /api/v1/health` (`anthropic`, `openai` or `mock`)
- Tests for the per-agent CLIs (`tests/test_cli.py`); total coverage is above the enforced 70% minimum

### Fixed

#### API
- The development API key (used when `AEGISLANG_API_KEYS` is unset) is generated once per process at startup and shared by all workers; previously a new key was generated per request, so the logged key never worked
- The SQLite storage backend (`AEGISLANG_STORAGE_BACKEND=sqlite`) now persists documents, clauses and artifacts (the store methods were missing), including empty clause lists
- `POST /compile` uses schemas registered via `POST /schemas` (merged with the built-in schemas); an unknown `target_schema` returns 404
- `GET /rules/{clause_id}` returns every artifact compiled for the clause (one per format)
- All errors, including unknown routes (404/405) and request validation errors (422), use the documented error format (`error`, `status_code`, `request_id`, optional `error_code` and `details`); 429 responses include a `Retry-After` header
- Input is validated up front: `output_formats` must be a subset of `yaml`/`sql`/`python`, schema bodies are validated, and ingest `metadata` must be a JSON object
- The OpenAI parser is selected when only `OPENAI_API_KEY` is set
- The Anthropic parser works with the `anthropic` 1.x SDK, which removed the `temperature` keyword: temperature is now sent in the request body (set `temperature=None` for models that reject sampling parameters); previously every clause failed to parse and compile jobs completed with 0 clauses
- `/health` reports the package version (0.1.0) instead of a hard-coded 1.0.0
- `python -m aegislang.api.server` now calls `setup_logging()`, so `LOG_LEVEL`/`AEGISLANG_LOG_LEVEL`, `AEGISLANG_LOG_FILE` and `SENTRY_DSN` take effect
- The uploaded file name (not the temporary path) drives section IDs and `source_file`

#### Pipeline
- The mock parser follows the SPEC clause-type taxonomy with whole-word matching ("verify" is no longer detected as conditional; adds "can", "refers to", "where", "should not", "notwithstanding"; "unless" is conditional)
- Temporal extraction handles number words ("five years") and no longer treats every "by"/"before" as a deadline; conditions are extracted at the end of sentences too
- Ingestor document IDs are derived from the file name plus a content hash, so they are stable across runs
- DOCX headings 7-9 no longer crash ingestion (clamped to level 6)
- Re-registering a schema drops stale fields from the mapping index
- Validation lineage `section_id` points at the real source section
- Artifacts render enum values (`obligation`) instead of `ClauseType.OBLIGATION`
- SQL artifacts only emit `COMMENT ON CONSTRAINT` for constraints that were actually generated
- The template directory loader maps `obligation.yaml.j2` to the `obligation` template
- CLI tools (`python -m aegislang.agents.<agent>`) log to stderr, so JSON written to stdout stays machine-readable

### Changed
- Jinja2 templates render in a `SandboxedEnvironment`
- Docker Compose simplified to the `aegislang` service (SQLite on the `aegislang-data` volume, passes `AEGISLANG_API_KEYS`) and the `aegislang-dev` profile; the unused PostgreSQL, Redis and Neo4j services were removed, along with the Makefile `db-*` targets and `scripts/init-db.sql`
- The production Docker image installs `curl` so the `HEALTHCHECK` works
- The AML example runner writes `.py` files and replaces stale output; `examples/output` was regenerated

### Removed
- Dead event-publishing helpers (`publish_*_event`) that imported a missing module

### Changed (code quality)
- `ruff check .`, `black --check .` and `mypy aegislang/` are clean; the CI lint job installs the runtime dependencies so MyPy sees FastAPI's types, and MyPy is now blocking. Rule exceptions are scoped in `pyproject.toml` with reasons

### Security
- Bandit findings resolved so the Bandit step of `make security-check` passes (justified `# nosec` for constant SQLite table names (B608), the `HOST` default (B104) and a best-effort Sentry breadcrumb (B110))

### Planned
See [ROADMAP.md](ROADMAP.md). Highlights:
- Real LLM validation and prompt tuning on AML/KYC documents
- Cross-reference resolution between clauses
- Terraform and OPA/Rego output format improvements
- Rule drift detection
- Audit chain visualizer (web UI for clause-to-artifact lineage)
- RAG-based policy retrieval

---

## [0.1.0] - 2026-01-10

Initial alpha release (not tagged). This entry summarizes the code as it stood before the Unreleased fixes (development continued through February 2026 without a version bump).

### Added

#### Core Pipeline (L1-L5 Agents)
- **L1 Ingestion Layer** (`aegis_ingestor.py`): Document parsing for PDF, DOCX, Markdown, and HTML formats with metadata extraction and content chunking
- **L2 Parsing Layer** (`policy_parser_agent.py`): LLM-powered clause extraction (Anthropic or OpenAI, with a keyword-based mock parser for offline use) supporting obligations, prohibitions, permissions, conditionals, definitions and exceptions with structured output
- **L3 Mapping Layer** (`schema_mapping_agent.py`): Entity mapping by exact, synonym and embedding-similarity match with configurable thresholds and manual override support
- **L4 Compilation Layer** (`compiler_agent.py`): Multi-format code generation with built-in Jinja2 templates for YAML, SQL and Python; experimental Terraform, Rego and JSON output in the CLI/library
- **L5 Validation Layer** (`trace_validator_agent.py`): Artifact validation with confidence scoring, lineage tracking, and review flag generation

#### Templates
- Reference Jinja2 templates in `templates/`: YAML (obligation, prohibition, conditional), SQL (check constraint, trigger, audit table) and Python (test stub, validator class)
- These are optional overrides: the compiler uses its built-in templates unless a templates directory is passed (`CompilerAgent(templates_dir=...)` or `--templates`)

#### API & Infrastructure
- REST API server with FastAPI (`/api/v1/`)
- Endpoints: `/health`, `/ingest`, `/documents`, `/documents/{doc_id}`, `/compile`, `/jobs/{job_id}`, `/jobs/{job_id}/stream`, `/clauses/{doc_id}`, `/rules/{clause_id}`, `/schemas`, `/schemas/{schema_id}`
- API key authentication (`X-API-Key`) and per-key rate limiting
- In-memory storage, with an opt-in SQLite backend
- Docker multi-stage build for production deployment
- Docker Compose file (it also defined PostgreSQL, Redis and Neo4j services that the application never used; removed in Unreleased)
- CI/CD pipeline with GitHub Actions

#### Developer Experience
- Makefile with 30+ automation targets
- Structured logging (structlog) with optional Sentry integration (not wired into the server until Unreleased)
- Unit, integration, system and regression tests
- API documentation

#### Configuration
- Environment-variable configuration; `AEGISLANG_ENV=production` hides internal error details and switches logs to JSON (`config.yaml` is a reference file only)
- Secrets read from environment variables
- Semantic versioning (VERSION file)

### Security
- Non-root Docker user
- Input validation and sanitization
- SQL injection prevention via parameterized queries
- CORS configuration for API

### Documentation
- README with setup guide and ecosystem links
- API documentation (docs/API.md)
- SPEC.md with detailed requirements
- CONTRIBUTING.md, SECURITY.md, and docs/ (architecture, FAQ, troubleshooting, compliance review, security scan)

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 0.1.0 | 2026-01-10 | Initial alpha with the L1-L5 pipeline |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
