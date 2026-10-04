# AegisLang Technical Specification

**Version:** 0.1.0
**Status:** Alpha
**Last Updated:** October 2026

---

## 1. Executive Summary

### 1.1 Purpose

AegisLang is a multi-agent semantic compiler that transforms unstructured regulatory and policy text into executable controls, workflows, and audit artifacts. The system maintains complete semantic traceability from source clause to generated code.

### 1.2 Tagline

**Language in. Compliance out.**

### 1.3 Core Value Proposition

- **Eliminates manual translation** of regulatory requirements into technical controls
- **Maintains provenance** from policy clause to executable artifact
- **Enables audit transparency** via traceable lineage graphs

### 1.4 Current Domain Focus

| Domain | Status | Example Regulations |
|--------|--------|---------------------|
| Financial Services (AML/KYC) | Evaluated (mock LLM) | FinCEN CDD, FFIEC CIP, FATF Rec. 10 |
| Data Privacy (GDPR, CCPA) | Not yet evaluated | — |
| Information Security (NIST, ISO) | Not yet evaluated | — |

---

## 2. System Architecture

### 2.1 Architectural Pattern

AegisLang employs a **layered multi-agent architecture**. Each layer operates as a discrete agent, callable independently (Python library or `python -m aegislang.agents.<agent>` CLI) or as part of the full pipeline (REST API, `examples/run_aml_pipeline.py`).

### 2.2 Layer Specification

| Layer | Purpose | Agent Module | Input | Output |
|-------|---------|------------|-------|--------|
| **L1: Ingestion** | Collect and preprocess policy documents | `aegis_ingestor.py` | Raw documents (PDF, DOCX, MD, HTML) | Normalized JSON sections + text chunks |
| **L2: Parsing** | Extract obligations, conditions, actors | `policy_parser_agent.py` | Text chunks | Semantic clause structures |
| **L3: Mapping** | Link entities to system schemas | `schema_mapping_agent.py` | Clause structures | Entity-to-field mappings |
| **L4: Compilation** | Generate executable artifacts | `compiler_agent.py` | Mapped clauses | YAML, SQL, Python artifacts (Terraform/Rego/JSON experimental) |
| **L5: Validation** | Verify correctness, emit provenance | `trace_validator_agent.py` | Artifacts + source clauses | Validation results + provenance graph |

### 2.3 Data Flow

```
     ┌──────────────┐
     │   Document   │  PDF, DOCX, MD, HTML
     │    Source     │
     └──────┬───────┘
            │
            ▼
┌──────────────────────┐
│   Document Ingestor  │  Tokenization via tiktoken (char estimate fallback)
│       (L1)           │  Section hierarchy extraction
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐     ┌─────────────────┐
│  Policy Parser Agent │────▶│    LLM API      │  Anthropic/OpenAI
│       (L2)           │     │  (or Mock)      │  or regex mock parser
└──────────┬───────────┘     └─────────────────┘
           │
           ▼
┌──────────────────────┐     ┌─────────────────┐
│ Schema Mapping Agent │────▶│ Schema Registry │  Built-in schemas + API-registered
│       (L3)           │     │  (Pydantic)     │  synonym maps; mock embeddings
└──────────┬───────────┘     └─────────────────┘  by default (no LLM calls)
           │
           ▼
┌──────────────────────┐     ┌─────────────────┐
│   Compiler Agent     │────▶│   Templates     │  Built-in Jinja2 templates
│       (L4)           │     │  (built-in +    │  (compiler_agent.py); optional
└──────────┬───────────┘     │   overrides)    │  templates/ overrides
           │                 └─────────────────┘
           ▼
┌──────────────────────┐
│ Trace Validator Agent│  Provenance chain validation
│       (L5)           │  Confidence scoring, provenance graph
└──────────┬───────────┘
           │
           ▼
     ┌─────┴─────┐
     │ Artifacts │  YAML rules, SQL constraints, Python tests
     └───────────┘  + validation results / provenance graph (GET /trace/{doc_id})
```

---

## 3. Component Specifications

### 3.1 Ingestion Layer (L1)

**Module:** `aegislang/agents/aegis_ingestor.py`

**Implemented features:**

| ID | Feature | Status |
|----|---------|--------|
| ING-001 | Parse PDF documents (via pdfminer.six) | Implemented |
| ING-002 | Parse DOCX documents (via python-docx) | Implemented |
| ING-003 | Parse Markdown files | Implemented |
| ING-004 | Parse HTML (via beautifulsoup4) | Implemented |
| ING-006 | Paragraph/sentence-aware text chunking with token targets (tiktoken `cl100k_base`; ~4 chars/token estimate if the encoding cannot be loaded). Defaults: target 768, min 256, max 1024, overlap 64 tokens | Implemented |
| ING-007 | Document hierarchy preservation (levels 1–6): Markdown `#` headings, HTML `h1`–`h6`, DOCX `Heading N` styles (levels above 6 clamped to 6). PDF hierarchy is built only from lines in the extracted text that start with `#`-style headings; otherwise the whole PDF becomes one section | Implemented (PDF: limited) |
| ING-008 | Standardized JSON output (`doc_id`, `metadata{source_file, ingestion_timestamp, document_type, page_count, language, hash}`, `sections[{section_id, section_title, parent_section, hierarchy_level, text_chunks[{chunk_id, text, token_count, embedding_vector}]}]`) | Implemented |

**ID formats:** the library `doc_id` is the normalized file name plus the first 6 hex chars of the content SHA-256 (stable across runs). The REST API instead assigns `{FILE_STEM}_{6 random hex}` per upload. Section IDs are `{NAME}_S001`, chunk IDs `{NAME}_S001_C000`. `language` is always `"en"`.

**Not implemented:** ING-005 (OCR for scanned documents).

### 3.2 Parsing Layer (L2)

**Module:** `aegislang/agents/policy_parser_agent.py`

**Implemented features:**

| ID | Feature | Status |
|----|---------|--------|
| PRS-001 | Clause type detection (obligation, prohibition, permission, conditional, definition, exception) | Implemented (mock parser: keyword taxonomy below) |
| PRS-002 | Actor entity extraction | Implemented (mock: words before the first modal verb) |
| PRS-003 | Action/verb phrase extraction | Implemented (mock: first word after the modal verb) |
| PRS-004 | Object/target entity extraction | Implemented (mock: AML verb patterns + entity normalization) |
| PRS-005 | Conditional trigger extraction | Implemented (`if`/`when`/`where`/`unless`/`before`/`after`/`upon` clauses go to `condition.trigger`, even when the type is decided by a modal verb) |
| PRS-006 | Temporal scope extraction into `temporal_scope{deadline, frequency, duration}` | Implemented (mock patterns below) |
| PRS-009 | Confidence scoring | Implemented (mock: heuristic score 0.10–0.95 based on pattern specificity and extraction quality) |

**LLM providers:** Anthropic (default model `claude-sonnet-4-20250514`), OpenAI (default `gpt-4-turbo-preview`), and Mock (regex-based, no network). LLM providers are prompted to return the same JSON shape as the mock parser. There is no retry logic. The REST API selects Anthropic if `ANTHROPIC_API_KEY` is set, else OpenAI if `OPENAI_API_KEY` is set, else Mock.

**Not implemented:** PRS-008 (cross-reference resolution between clauses; `cross_references` is always `[]`).

**Clause Type Taxonomy (mock parser):**

Indicators are matched case-insensitively on word boundaries, so "verify" does not match "if" and "by means of" is not a definition. A sentence that starts with `if`, `when`, `where` or `unless` is `conditional`. Otherwise the first matching row below wins, in this order of precedence:

| Precedence | Type | Modal Indicators |
|-----------|------|------------------|
| 1 | `prohibition` | must not, shall not, may not, should not, is/are prohibited from, prohibited |
| 2 | `definition` | is/are defined as, defined as, refers to / refer to, means (not "by means of") |
| 3 | `exception` | notwithstanding, except, exempt from / exempted from |
| 4 | `permission` | is/are permitted to, may, permitted, can |
| 5 | `obligation` | is/are required to, must, shall, required |
| 6 | `conditional` | if, when, where, unless (anywhere in the sentence) |
| — | `obligation` (default) | no indicator matched (e.g. "should") |

**Temporal scope patterns (mock parser):**

| Field | Extracted from |
|-------|----------------|
| `deadline` | "within N units", "no later than N units" (digits or number words; hours/days/weeks/months/years, optionally business/calendar), "within a reasonable time/period", "by/before the end/close of …" |
| `duration` | "for / at least / up to / a period of / a minimum of N units", e.g. "for at least five years", "for a period of 5 years" |
| `frequency` | annually, semi-annually, monthly, weekly, daily, quarterly, periodically, "on an ongoing/periodic/regular/annual basis" |

`temporal_scope` is `null` when none of these are found.

### 3.3 Mapping Layer (L3)

**Module:** `aegislang/agents/schema_mapping_agent.py`

**Implemented features:**

| ID | Feature | Status |
|----|---------|--------|
| MAP-001 | Match entities (actor, object, condition subject) to schema field paths (`table.field`) | Implemented |
| MAP-002 | Semantic embedding matching | Partial: mock embeddings by default (deterministic SHA-256 pseudo-vectors, not semantic). `SentenceTransformerProvider` / `OpenAIEmbeddingProvider` are used only when injected via `embedding_provider` in library code. The API and CLI always use mock embeddings. |
| MAP-003 | Multiple target schema formats (SQL, API, Object) | Metadata only: `schema_type` is stored, but every schema is matched as tables and fields |
| MAP-004 | Schema Registry with versioning | Partial: `version` is a label. Re-registering a `schema_id` replaces the schema (no history) |
| MAP-005 | Synonym resolution | Implemented (built-in synonym map) |
| MAP-006 | Manual mapping overrides | Implemented (registry `manual_overrides`; library / registry JSON only, no API endpoint) |
| MAP-007 | Confidence scoring | Implemented (threshold 0.7 by default; fixed at 0.7 in the API) |
| MAP-008 | Unmappable entity detection | Implemented (`unmapped_entities` with reason and suggestions) |

Built-in schemas: `kyc_schema`, `org_schema`, `records_schema`. The API merges schemas registered via `POST /schemas` with them; a registered schema with the same ID replaces the built-in one. The mapper never calls an LLM.

### 3.4 Compilation Layer (L4)

**Module:** `aegislang/agents/compiler_agent.py`

**Supported output formats:**

| Format | Status | Available in |
|--------|--------|--------------|
| YAML compliance rules (`control:` blocks) | Implemented | API, CLI, library |
| SQL check constraints + triggers | Implemented: obligation → `ADD CONSTRAINT … CHECK` + PL/pgSQL trigger; prohibition → `CHECK (NOT …)`; other types → comment-only artifact. Table taken from the mapped entity path, else generic `compliance_table` | API, CLI, library |
| Python pytest stubs | Implemented | API, CLI, library |
| Terraform | Experimental (placeholder Sentinel-style policy resource) | CLI/library only (not API) |
| OPA/Rego | Experimental (simple render) | CLI/library only (not API) |
| JSON | Experimental (simple render) | CLI/library only (not API) |

**Templates:** the built-in templates are Python strings in `compiler_agent.py`, rendered in a Jinja2 `SandboxedEnvironment`. The repository's `templates/` directory holds optional overrides and is not loaded by default. Load it with `CompilerAgent(templates_dir=Path("templates"))` or the compiler CLI's `--templates templates`. Files named `<format>/<name>.<ext>.j2` are selected when `<name>` equals the clause type or `default`. In the repository, `yaml/{obligation,prohibition,conditional}.yaml.j2` override by clause type, while the `sql/` and `python/` files are reference templates that are not selected automatically. The REST API always uses the built-in templates.

**Severity:** prohibition = critical, obligation = high, other types = medium.

**Syntax checks:** YAML via `yaml.safe_load`; Python via `ast.parse`; JSON via `json.loads`; SQL via `sqlparse` (lenient: valid if it tokenizes, unknown statement types become warnings); Terraform checks brace/quote balance only; Rego is never marked invalid (warnings only).

### 3.5 Validation Layer (L5)

**Module:** `aegislang/agents/trace_validator_agent.py`

**Implemented features:**

| ID | Feature | Status |
|----|---------|--------|
| VAL-001 | Clause-to-artifact provenance chain validation (`chain_completeness` check) | Implemented |
| VAL-002 | Artifact syntax validation (`syntax_validity`, reusing L4's checks; the SQL check is lenient sqlparse tokenization) | Implemented |
| VAL-003 | Confidence scoring for trace links (average of parse and mapping confidences) | Implemented |
| VAL-005 | Lineage metadata generation: `lineage{document_id, section_id, chunk_id, clause_id, mapping_id, artifact_id}` per result, plus a provenance graph (document/section/chunk/clause/artifact nodes) exportable as JSON or DOT | Implemented |
| VAL-007 | Low-confidence flagging for human review (`review_flags`; status `needs_review`) | Implemented |

**Status rules:** `failed` if chain completeness or syntax fails, or confidence is below 0.50; `needs_review` if any review flag is raised (confidence below `confidence_threshold`, default 0.85, or below `review_threshold`, default 0.70; key terms missing from the artifact); otherwise `passed`.

**Not implemented:** VAL-004 (semantic drift detection; only a key-term presence check, `semantic_alignment`, exists) and VAL-008 (graph database persistence). A Neo4j writer (`store_provenance_graph`) exists but is never called by the API or CLI; the API stores graphs in its own storage backend.

---

## 4. API Specification

Full reference: [`docs/API.md`](docs/API.md).

### 4.1 REST API

**Base URL:** `http://localhost:8080/api/v1`

**Authentication:** API key via `X-API-Key` header. Valid keys come from `AEGISLANG_API_KEYS` (comma-separated). If no keys are set, a random development key is generated at startup and logged. Set `AEGISLANG_DISABLE_AUTH=true` to disable auth. A missing key returns 401 and an invalid key 403. All keys have equal access.

**Rate limiting:** in-memory, per process and per API key. Defaults are 60 requests/minute and 1000/hour (`AEGISLANG_RATE_LIMIT_MINUTE`, `AEGISLANG_RATE_LIMIT_HOUR`). Exceeding them returns 429 with `Retry-After`. `/health` is exempt.

**Errors:** `{"error", "status_code", "request_id", "error_code"?, "details"?}`. Every response carries `X-Request-ID`.

**Docs:** Swagger UI `/api/docs`, ReDoc `/api/redoc`, OpenAPI `/api/openapi.json`.

### 4.2 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check: version, `llm_provider` (no auth required) |
| `POST` | `/ingest` | Upload a document; returns an ingestion job (`ing_…`) |
| `GET` | `/documents` | List all ingested documents |
| `GET` | `/documents/{doc_id}` | Get document details |
| `POST` | `/compile` | Parse, map, compile (`yaml`/`sql`/`python`) and validate; returns a job (`cmp_…`) |
| `GET` | `/jobs/{job_id}` | Check async job status |
| `GET` | `/jobs/{job_id}/stream` | Stream job status (Server-Sent Events) |
| `GET` | `/clauses/{doc_id}` | Get extracted clauses (available after a compile job) |
| `GET` | `/rules/{clause_id}` | Get generated artifacts for a clause |
| `GET` | `/trace/{doc_id}` | Get validation results and provenance graph |
| `POST` | `/schemas` | Register or replace a target schema |
| `GET` | `/schemas` | List schemas registered via the API |
| `GET` | `/schemas/{schema_id}` | Get a registered schema |

### 4.3 Storage

**Default:** in-memory dictionaries (`AEGISLANG_STORAGE_BACKEND=memory`), which are lost on restart. `python -m aegislang.api.server` forces a single worker with this backend.

**Available:** SQLite backend (`AEGISLANG_STORAGE_BACKEND=sqlite`, file `AEGISLANG_SQLITE_PATH`, default `aegislang_data.db`) with tables `jobs`, `documents`, `schemas`, `clauses`, `artifacts` and `traces` (plus key-index tables `clauses_keys` and `artifacts_keys`). Docker Compose uses SQLite on a named volume.

Finished jobs are removed after `AEGISLANG_JOB_TTL_SECONDS` (default 86400). Documents, clauses, artifacts, schemas and traces never expire, and there are no delete endpoints. Data is not encrypted at rest. No PostgreSQL, Redis or Neo4j is used.

---

## 5. Dependencies

### 5.1 Core Runtime (`requirements.txt`, Python 3.11+)

| Package | Purpose |
|---------|---------|
| fastapi, uvicorn, pydantic | REST API framework + data models |
| python-multipart | Form/file upload handling |
| pdfminer.six | PDF text extraction |
| python-docx | DOCX parsing |
| beautifulsoup4 | HTML parsing |
| tiktoken | Token counting for chunking |
| anthropic, openai | LLM clients for clause parsing (unused in mock mode) |
| jinja2, pyyaml, sqlparse | Template rendering + artifact syntax checks |
| structlog | Structured logging |
| sentry-sdk | Error tracking (active only when `SENTRY_DSN` is set) |
| neo4j | Optional Neo4j provenance writer (not used by the API or CLI) |

### 5.2 Development (`make dev-install`)

| Package | Purpose |
|---------|---------|
| pytest, pytest-asyncio, pytest-cov | Test runner + coverage |
| httpx | Required by FastAPI's `TestClient` in the API tests |
| black, ruff, mypy | Formatting, linting, type checking |

### 5.3 Optional ML (`requirements-ml.txt`)

| Package | Purpose |
|---------|---------|
| sentence-transformers | `SentenceTransformerProvider` embeddings for schema mapping (library use only) |
| transformers, torch | ML model runtime |

---

## 6. Testing

### 6.1 Test Suite

252 tests, all passing (`python -m pytest tests/`). Counts are collected test cases, including parametrized cases.

| File | Tests | Covers |
|------|-------|--------|
| `test_api.py` | 46 | REST endpoints, auth, request IDs, SSE stream, mock-mode detection |
| `test_api_contract.py` | 17 | API behaviour as documented in `docs/API.md` (auth, error format, validation, schemas, rules, SQLite backend) |
| `test_cli.py` | 13 | Per-agent CLIs, chained L1 → L5 |
| `test_ingestor.py` | 24 | L1 chunking, Markdown parsing, ingestor |
| `test_parser.py` | 24 | L2 mock parser and data models |
| `test_mapper.py` | 25 | L3 mock embeddings, registry, mapping |
| `test_integration.py` | 12 | Multi-stage pipeline flows |
| `test_system.py` | 20 | End-to-end user journeys through the API |
| `test_spec_conformance.py` | 39 | Behaviour described in this SPEC (taxonomy, temporal scope, hierarchy, templates, lineage, LLM SDK call signatures) |
| `test_regression.py` | 19 | Guards for previously fixed bugs |
| `test_pipeline_regression.py` | 13 | Output regression on the three AML documents (clause counts, type distributions, mapping) |
| **Total** | **252** | **All passing** |

`tests/performance/` contains Locust load tests that are not part of the normal run. The only marker in use is `slow`.

### 6.2 Running Tests

```bash
make dev-install          # pytest and httpx are dev dependencies
python -m pytest tests/ -v
```

---

## 7. Glossary

| Term | Definition |
|------|------------|
| **Artifact** | Generated executable output (YAML rule, SQL check, Python test) |
| **Clause** | A single regulatory statement extracted from policy text |
| **Confidence Score** | Numeric measure (0-1) of system certainty in extraction/mapping |
| **Entity** | A noun phrase representing an actor, object, or concept in policy |
| **Lineage** | The complete traceability chain from source document to artifact |
| **Mapping** | Association between a policy entity and a system schema field |
| **Provenance** | Audit trail documenting the origin and transformation of data |
| **Schema Registry** | Catalog of target system schemas available for entity mapping (built-in plus API-registered) |
| **Trace** | A validation result linking a clause to an artifact, with lineage and checks |

---

*End of Specification*
