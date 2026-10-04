# AegisLang Architecture

This document describes the system architecture, data flow, and component interactions in AegisLang.

## Table of Contents
- [High-Level Architecture](#high-level-architecture)
- [Data Flow Pipeline](#data-flow-pipeline)
- [Component Details](#component-details)
- [Storage](#storage)
- [API Architecture](#api-architecture)
- [Deployment Architecture](#deployment-architecture)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AegisLang System                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────────────────────────────────────────┐   │
│  │ HTTP clients │    │                 REST API Layer                   │   │
│  │ (curl,       │───▶│  FastAPI Server (POST /ingest, POST /compile,    │   │
│  │  scripts)    │    │  GET /rules, GET /trace, ...) - background jobs  │   │
│  └──────────────┘    └──────────────────────────────────────────────────┘   │
│                                              │                              │
│  ┌──────────────┐                            ▼                              │
│  │ Per-agent    │    ┌──────────────────────────────────────────────────┐   │
│  │ CLIs         │───▶│              Agent Pipeline (L1-L5)              │   │
│  │ (python -m   │    │                                                  │   │
│  │  aegislang.  │    │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  │   │
│  │  agents.*)   │    │  │   L1   │─▶│   L2   │─▶│   L3   │─▶│   L4   │  │   │
│  ├──────────────┤    │  │Ingest  │  │ Parse  │  │  Map   │  │Compile │  │   │
│  │ Python       │    │  └────────┘  └────────┘  └────────┘  └────────┘  │   │
│  │ library      │    │                   │                       │      │   │
│  │ (import the  │    │                   │                       ▼      │   │
│  │  agents)     │    │                   │                  ┌────────┐  │   │
│  └──────────────┘    │                   │                  │   L5   │  │   │
│                      │                   │                  │Validate│  │   │
│                      │                   │                  └────────┘  │   │
│                      └───────────────────┼──────────────────────────────┘   │
│                                          ▼                                  │
│                      ┌──────────────────────────────────────────────────┐   │
│                      │  LLM provider (optional): Anthropic or OpenAI    │   │
│                      │  Without an API key, a keyword-based mock parser │   │
│                      │  is used. L3 never calls an LLM.                 │   │
│                      └──────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    Data Layer (REST API server only)                  │  │
│  │                                                                       │  │
│  │  ┌──────────────────────────┐    ┌──────────────────────────────┐     │  │
│  │  │ In-memory (default)      │ or │ SQLite file                  │     │  │
│  │  │ lost on restart,         │    │ AEGISLANG_STORAGE_BACKEND=   │     │  │
│  │  │ single worker            │    │   sqlite                     │     │  │
│  │  └──────────────────────────┘    └──────────────────────────────┘     │  │
│  │  CLIs and the library write JSON/artifact files instead.              │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

There is no web UI and no separate SDK package. The three ways in are the REST API, the per-agent command-line tools (`python -m aegislang.agents.<agent>`), and importing the agent classes from Python (`from aegislang.agents.aegis_ingestor import AegisIngestor`, etc.).

---

## Data Flow Pipeline

### Complete Processing Flow

```
                              AegisLang Data Flow

    ┌─────────────┐
    │   Policy    │
    │  Document   │  (PDF, DOCX, MD, HTML)
    └──────┬──────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  L1: INGESTION LAYER (aegis_ingestor.py)                        │
    │  ─────────────────────────────────────────                      │
    │  • Parse document format (PDF, DOCX, Markdown, HTML)            │
    │  • Extract text content (text-based PDFs only; no OCR)          │
    │  • Build section hierarchy, split sections into token chunks    │
    │  • Extract metadata                                             │
    │  • Compute SHA-256 content hash (metadata.hash)                 │
    │                                                                 │
    │  Output: IngestedDocument                                       │
    │  {doc_id, metadata{..., hash}, sections[{..., text_chunks[]}]}  │
    └──────┬──────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  L2: PARSING LAYER (policy_parser_agent.py)                     │
    │  ──────────────────────────────────────────                     │
    │  • Split chunks into candidate clauses (sentences)              │
    │  • Parse each clause with Anthropic/OpenAI, or with the         │
    │    keyword-based mock parser when no LLM key is configured      │
    │  • Classify clause types (obligation, prohibition, etc.)        │
    │  • Extract semantic components:                                 │
    │    - Actor (who)                                                │
    │    - Action (what verb)                                         │
    │    - Object (what target)                                       │
    │    - Condition (when/if)                                        │
    │    - Temporal scope (deadline, frequency, duration)             │
    │  • Assign confidence scores                                     │
    │                                                                 │
    │  Output: ParsedClause[]                                         │
    │  {clause_id, type, actor, action, object, condition,            │
    │   temporal_scope, confidence}                                   │
    └──────┬──────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  L3: MAPPING LAYER (schema_mapping_agent.py)                    │
    │  ─────────────────────────────────────────                      │
    │  • Load target schema definitions (built-in + registered)       │
    │  • Apply manual overrides (library API)                         │
    │  • Exact match on field names / semantic labels                 │
    │  • Resolve synonyms and aliases                                 │
    │  • Embedding similarity matching (threshold 0.7). Uses mock     │
    │    hash-based embeddings unless an embedding provider is        │
    │    injected in library code; API and CLI always use mock        │
    │  • Flag unmappable entities                                     │
    │                                                                 │
    │  Output: MappedClause[]                                         │
    │  {clause_id, source_clause, mapped_entities[],                  │
    │   unmapped_entities[], mapping_status}                          │
    └──────┬──────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  L4: COMPILATION LAYER (compiler_agent.py)                      │
    │  ────────────────────────────────────────                       │
    │  • Select template (built-in Jinja2 templates, or a custom      │
    │    templates directory passed explicitly)                       │
    │  • Render executable artifacts:                                 │
    │    - YAML compliance rules                                      │
    │    - SQL check constraints & triggers                           │
    │    - Python test stubs                                          │
    │    - Terraform/Rego/JSON (experimental, CLI/library only)       │
    │  • Validate syntax                                              │
    │  • Embed source references                                      │
    │                                                                 │
    │  Output: CompiledArtifact[]                                     │
    │  {artifact_id, format, content, syntax_valid, template_used}    │
    └──────┬──────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  L5: VALIDATION LAYER (trace_validator_agent.py)                │
    │  ──────────────────────────────────────────────                 │
    │  • Verify artifact integrity (syntax validity)                  │
    │  • Check semantic consistency                                   │
    │  • Validate lineage (source → artifact traceability)            │
    │  • Calculate overall confidence                                 │
    │  • Generate review flags for low-confidence items               │
    │  • Build provenance graph (the audit trail)                     │
    │                                                                 │
    │  Output: ValidationResult[] + ProvenanceGraph                   │
    │  {trace_id, validation_status, confidence_score,                │
    │   validation_checks[], lineage, review_flags[]}                 │
    └──────┬──────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────┐
    │   Output    │
    │  Artifacts  │  (.yaml, .sql, .py; .tf, .rego, .json experimental)
    └─────────────┘
```

The CLI tools and the library hand JSON from one layer to the next and write artifact files. The REST API stores each layer's output (see [Storage](#storage)) and returns artifacts as JSON. It does not write artifact files.

### Asynchronous Job Flow (REST API)

```
    POST /api/v1/ingest ───▶ {status: "accepted", job_id: "ing_xxxxxxxx",
          │                   doc_id, status_url}
          │
          └──▶ background task: L1 ingest ──▶ store document

    POST /api/v1/compile ──▶ {status: "accepted", job_id: "cmp_xxxxxxxx",
          │                   doc_id, status_url}
          │
          └──▶ background task: L2 parse ──▶ L3 map ──▶ L4 compile ──▶ L5 validate
                                   │                        │               │
                                   ▼                        ▼               ▼
                                clauses                 artifacts     trace (results +
                                                                      provenance graph)

    Follow a job:   GET /api/v1/jobs/{job_id}          (poll)
                    GET /api/v1/jobs/{job_id}/stream   (Server-Sent Events, 1 event/s)
    Read results:   GET /api/v1/clauses/{doc_id}
                    GET /api/v1/rules/{clause_id}
                    GET /api/v1/trace/{doc_id}
```

Jobs run inside the API process as FastAPI background tasks. There is no message queue, event bus or webhook mechanism. Clients poll the job endpoint or follow its Server-Sent Events stream. Finished jobs are deleted after `AEGISLANG_JOB_TTL_SECONDS` (default 86400), after which the job endpoint returns 404.

---

## Component Details

### Agent Layer Responsibilities

| Layer | Agent | Input | Output | Key Functions |
|-------|-------|-------|--------|---------------|
| L1 | `AegisIngestor` | Raw documents | `IngestedDocument` | Parse, chunk, extract metadata |
| L2 | `PolicyParserAgent` | Text chunks | `ParsedClause[]` | LLM (or mock) extraction, classification |
| L3 | `SchemaMappingAgent` | Parsed clauses | `MappedClause[]` | Override/exact/synonym match, embedding match |
| L4 | `CompilerAgent` | Mapped clauses | `CompiledArtifact[]` | Template rendering, syntax check |
| L5 | `TraceValidatorAgent` | Artifacts + mapped + parsed clauses | `ValidationResult[]`, `ProvenanceGraph` | Integrity, lineage, confidence |

### Supported Clause Types

```
┌────────────────────────────────────────────────────────────┐
│                     Clause Types                           │
├─────────────────┬──────────────────────────────────────────┤
│ obligation      │ Actor MUST perform action                │
│ prohibition     │ Actor MUST NOT perform action            │
│ permission      │ Actor MAY perform action                 │
│ conditional     │ IF condition THEN action                 │
│ definition      │ Term IS defined as meaning               │
│ exception       │ EXCEPT when condition applies            │
└─────────────────┴──────────────────────────────────────────┘
```

### Output Formats

```
┌──────────────────────────────────────────────────────────────────┐
│                      Output Formats                              │
├──────────────┬───────────────────────────────────────────────────┤
│ YAML         │ Compliance rule definitions (control: documents)  │
│ SQL          │ PostgreSQL CHECK constraints and triggers         │
│ Python       │ pytest test stubs                                 │
├──────────────┼───────────────────────────────────────────────────┤
│ Terraform    │ Experimental: placeholder Sentinel-style policy   │
│ Rego         │ Experimental: simple OPA policy render            │
│ JSON         │ Experimental: simple JSON rule render             │
└──────────────┴───────────────────────────────────────────────────┘
```

The REST API accepts `yaml`, `sql` and `python` only and returns 422 for any other value. The compiler CLI (`--formats`) and `CompilerAgent` also accept `terraform`, `rego` and `json`. Their syntax checks are basic.

### Traceability and Audit Trail

L5 links every artifact back through mapping, clause, chunk, section and document (`lineage` on each `ValidationResult`: `document_id`, `section_id`, `chunk_id`, `clause_id`, `mapping_id`, `artifact_id`). It also builds a provenance graph with node types `document`, `section`, `chunk`, `clause` and `artifact`, joined by `CONTAINS_SECTION`, `CONTAINS_CHUNK`, `PARSED_TO` and `COMPILED_TO` edges. Mappings are not graph nodes. They appear only as `lineage.mapping_id`. The API serves both from `GET /api/v1/trace/{doc_id}`. They are persisted only with the SQLite backend. The structured logs (request IDs included) are the only other record. There is no separate audit-log table and no per-user tracking.

---

## Storage

The REST API server keeps all of its state in one storage backend, selected with `AEGISLANG_STORAGE_BACKEND`:

| Backend | Setting | Notes |
|---------|---------|-------|
| In-memory | `memory` (default; any unknown value also falls back to it) | Python dictionaries. Lost on restart. The server forces `WORKERS=1`. |
| SQLite | `sqlite` | File at `AEGISLANG_SQLITE_PATH` (default `aegislang_data.db`; `/app/data/aegislang.db` in Docker Compose). |

The SQLite backend uses simple key-value tables. Every value is a JSON document:

| Table | Key | Value |
|-------|-----|-------|
| `jobs` | job ID (`ing_xxxxxxxx` / `cmp_xxxxxxxx`) | Job status, result, error, timestamps |
| `documents` | `doc_id` | Ingested document (metadata, sections, chunks) |
| `schemas` | `schema_id` | Schema registered via `POST /api/v1/schemas` |
| `clauses` | `doc_id` (one row per clause) | `ParsedClause` |
| `artifacts` | `doc_id` (one row per artifact) | `CompiledArtifact` |
| `traces` | `doc_id` | Validation summary, `ValidationResult[]`, provenance graph |
| `clauses_keys`, `artifacts_keys` | `doc_id` | Which documents have clauses or artifacts stored (including empty lists) |

- Only jobs expire. Documents, clauses, artifacts, traces and schemas are kept until the database file is removed, and there are no DELETE endpoints.
- Data is not encrypted at rest.
- PostgreSQL, Redis and Neo4j are not used. `TraceValidatorAgent` has an optional Neo4j helper (`store_provenance_graph`, configured through `NEO4J_*`), but neither the API nor the CLIs call it.

---

## API Architecture

### REST Endpoints

```
┌────────────────────────────────────────────────────────────────────┐
│                         API v1 Routes                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Document Processing                                               │
│  ───────────────────                                               │
│  POST   /api/v1/ingest              Upload document (async job)    │
│  GET    /api/v1/documents           List all documents             │
│  GET    /api/v1/documents/{doc_id}  Get document details           │
│                                                                    │
│  Clauses & Rules                                                   │
│  ───────────────                                                   │
│  POST   /api/v1/compile             Parse/map/compile/validate     │
│                                     a document (async job)         │
│  GET    /api/v1/clauses/{doc_id}    Get clauses for a document     │
│  GET    /api/v1/rules/{clause_id}   Get all artifacts for a clause │
│  GET    /api/v1/trace/{doc_id}      Validation results and         │
│                                     provenance graph               │
│                                                                    │
│  Schema Management                                                 │
│  ─────────────────                                                 │
│  GET    /api/v1/schemas             List registered schemas        │
│  GET    /api/v1/schemas/{schema_id} Get a specific schema          │
│  POST   /api/v1/schemas             Register or replace a schema   │
│                                                                    │
│  System                                                            │
│  ──────                                                            │
│  GET    /api/v1/health              Health check (no auth)         │
│  GET    /api/v1/jobs/{job_id}       Get async job status           │
│  GET    /api/v1/jobs/{job_id}/stream  Job status as Server-Sent    │
│                                       Events                       │
│                                                                    │
│  Docs (no auth): /api/docs (Swagger), /api/redoc,                  │
│                  /api/openapi.json                                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

Every route except health and the docs pages requires an `X-API-Key` header. Missing keys get 401 and wrong keys get 403. All keys have the same access, with no user accounts or roles. Errors use one body format: `{"error": ..., "status_code": ..., "request_id": ..., "error_code"?: ..., "details"?: ...}`. Every response carries an `X-Request-ID` header.

### Request Flow

```
    Client Request
          │
          ▼
    ┌─────────────┐
    │ Middleware  │  CORS, X-Request-ID
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐     ┌─────────────┐
    │   FastAPI   │────▶│  Auth       │  X-API-Key → 401 / 403
    │   Router    │     │  Rate limit │  per key, in-memory → 429
    └──────┬──────┘     └─────────────┘
           │
           ▼
    ┌─────────────┐     ┌─────────────┐
    │  Request    │────▶│   Pydantic  │  invalid → 422
    │ Validation  │     │   Models    │
    └──────┬──────┘     └─────────────┘
           │
           ▼
    ┌─────────────┐
    │  Endpoint   │
    │  handler    │
    └──────┬──────┘
           │
           ├──────────────────────────┐
           ▼                          ▼
    ┌─────────────┐            ┌─────────────┐
    │ Background  │            │   Storage   │
    │ task: agent │───────────▶│ (memory or  │
    │ pipeline    │            │  SQLite)    │
    └─────────────┘            └─────────────┘
           │
           ▼
    ┌─────────────┐
    │  Response   │
    │   (JSON)    │
    └─────────────┘
```

---

## Deployment Architecture

### Docker Compose Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Docker Compose (docker-compose.yml)             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    aegislang (API)                           │   │
│  │                    Port: 8080, WORKERS=1                     │   │
│  │                                                              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │   │
│  │  │   FastAPI    │  │    Agents    │  │  Background  │        │   │
│  │  │   Server     │  │   (L1-L5)    │  │    tasks     │        │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────────────────────┐                                    │
│  │  SQLite database            │                                    │
│  │  /app/data/aegislang.db     │                                    │
│  │  (named volume              │                                    │
│  │   aegislang-data)           │                                    │
│  └─────────────────────────────┘                                    │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  aegislang-dev (profile "dev"): port 8081, hot reload,       │   │
│  │  source mounted read-only, auth disabled by default,         │   │
│  │  in-memory storage                                           │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

External Access:
  - API:      http://localhost:8080
  - Dev API:  http://localhost:8081 (with --profile dev)

Outbound (optional): Anthropic or OpenAI API (clause parsing), Sentry (SENTRY_DSN)
```

No database, cache or message broker containers are needed. Secrets (`AEGISLANG_API_KEYS`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `SENTRY_DSN`) come from the shell environment or a `.env` file next to `docker-compose.yml`. Docker Compose substitutes them into the service. The Python code does not read `.env` itself.

### Production Deployment

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Production Environment                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────┐                                           │
│  │   Reverse proxy      │  TLS termination (the app serves plain    │
│  │   (nginx, Caddy, …)  │  HTTP only)                               │
│  └──────────┬───────────┘                                           │
│             │                                                       │
│             ▼                                                       │
│  ┌──────────────────────┐        ┌──────────────────────┐           │
│  │  AegisLang           │───────▶│  SQLite file on a    │           │
│  │  (single instance)   │        │  persistent volume   │           │
│  └──────────────────────┘        └──────────────────────┘           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

AegisLang runs as a single instance. Running several instances behind a load balancer is not supported:

- The rate limiter lives in each process's memory.
- Ingest and compile jobs run inside the process that accepted the request.
- SQLite is a local file.
- The in-memory backend cannot be shared at all.

With the SQLite backend, `WORKERS` can be raised on one host, but rate limits then apply per worker process. Docker Compose sets `WORKERS=1`.

---

## Security Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Security Layers                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Layer 1: Network                                                   │
│  ─────────────────                                                  │
│  • CORS configuration (CORS_ORIGINS)                                │
│  • Rate limiting (in-memory, per API key and process)               │
│  • No TLS in the app: terminate TLS at a reverse proxy              │
│                                                                     │
│  Layer 2: Authentication                                            │
│  ───────────────────────                                            │
│  • API key validation (X-API-Key header, constant-time compare)     │
│  • All keys have equal access (no users or roles)                   │
│  • Random dev key generated and logged if no keys are configured    │
│  • Disable auth for development (AEGISLANG_DISABLE_AUTH=true)       │
│                                                                     │
│  Layer 3: Input Validation                                          │
│  ─────────────────────────                                          │
│  • Pydantic model validation                                        │
│  • File extension check only (no content sniffing)                  │
│  • Filename sanitization (path traversal rejected)                  │
│  • Size limit (AEGISLANG_MAX_FILE_SIZE, default 50 MB)              │
│                                                                     │
│  Layer 4: Data Protection                                           │
│  ────────────────────────                                           │
│  • Parameterized SQLite queries                                     │
│  • Sandboxed Jinja2 templates; SQL escaping filters                 │
│  • Uploads in a private temp dir, overwritten and deleted after     │
│    ingestion                                                        │
│  • No encryption at rest                                            │
│                                                                     │
│  Layer 5: Runtime                                                   │
│  ────────────────                                                   │
│  • Non-root container user                                          │
│  • Secret management (env vars)                                     │
│  • Generic error messages when AEGISLANG_ENV=production             │
│  • Sentry error tracking when SENTRY_DSN is set and the server is   │
│    started with python -m aegislang.api.server                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Integration Points

### External Integrations (Future)

NatLangChain and Agent-OS integrations are not currently on the roadmap. See [ROADMAP.md](../ROADMAP.md) for planned features.

AegisLang publishes no events and has no webhooks. To integrate, call the REST API (poll `GET /api/v1/jobs/{job_id}` or follow `/stream`) or run the CLI tools from your own scripts.

---

*Last updated: October 2026 | Version: 0.1.0*
