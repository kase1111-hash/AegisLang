# AegisLang REST API Documentation

The AegisLang API provides a RESTful interface for document ingestion, compilation (clause parsing, schema mapping, artifact generation and validation), and clause-to-artifact traceability.

**Base URL:** `/api/v1`

**API Version:** `0.1.0` (reported by `GET /api/v1/health` and in the OpenAPI document)

---

## Table of Contents

- [Authentication](#authentication)
- [Common Response Formats](#common-response-formats)
  - [Request IDs](#request-ids)
  - [Timestamps](#timestamps)
  - [Identifier Formats](#identifier-formats)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [Document Ingestion](#document-ingestion)
  - [Documents](#documents)
  - [Clauses](#clauses)
  - [Rules](#rules)
  - [Traceability](#traceability)
  - [Compilation](#compilation)
  - [Jobs](#jobs)
  - [Schemas](#schemas)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [CORS](#cors)
- [Configuration](#configuration)
- [OpenAPI Specification](#openapi-specification)
- [Example Workflow](#example-workflow)

---

## Authentication

The API uses API key authentication via the `X-API-Key` header. There are no user accounts or roles: every configured key has the same access.

**Configuration:**
- Set the `AEGISLANG_API_KEYS` environment variable to a comma-separated list of valid keys (e.g., `AEGISLANG_API_KEYS="key1,key2"`).
- If no keys are configured and auth is not disabled, a random development key is generated once at startup and logged as a `no_api_keys_configured` warning (field `development_key`). When the server is started with `python -m aegislang.api.server` (or `make run`), all workers share that key. A new key is generated on every restart.
- Set `AEGISLANG_DISABLE_AUTH=true` to disable authentication (local development only).

**Open endpoints (no key required):** `/api/v1/health`, `/api/docs`, `/api/redoc`, `/api/openapi.json`. All other endpoints require a valid API key.

| Situation | Status | Error body `error` |
|-----------|--------|--------------------|
| `X-API-Key` header missing | 401 | `Missing API key. Provide X-API-Key header.` |
| Key not in `AEGISLANG_API_KEYS` | 403 | `Invalid API key.` |

Authentication is checked before rate limiting.

**Example:**
```bash
curl -H "X-API-Key: your-api-key" http://localhost:8080/api/v1/documents
```

---

## Common Response Formats

### Success Response

Successful responses use status `200` and return the resource directly as a plain JSON object. There is no `{"status": "success", "data": ...}` envelope. `GET /api/v1/documents` returns a JSON array.

The asynchronous endpoints (`POST /ingest`, `POST /compile`) also return `200`, with `"status": "accepted"` in the body and a job ID to poll.

### Error Response

Every error, including unknown routes (404), unsupported methods (405) and request validation failures (422), uses the same shape:

```json
{
  "error": "Document not found",
  "status_code": 404,
  "request_id": "1de8c756-f3d7-4f4b-9383-9a1e69afdedb"
}
```

`error_code` and `details` are added when they apply. See [Error Handling](#error-handling).

### Request IDs

Every response carries an `X-Request-ID` header. If the request sends an `X-Request-ID` header, its value is echoed back. Otherwise the server generates a UUID. The same value appears as `request_id` in error bodies and in the server's structured logs.

### Timestamps

All timestamps are ISO 8601 in UTC, produced by Python's `datetime.isoformat()` with an explicit `+00:00` offset (not `Z`), for example `2026-10-04T12:00:00.931825+00:00`.

### Identifier Formats

| Identifier | Format | Example |
|------------|--------|---------|
| Ingestion job ID | `ing_` + 8 hex chars | `ing_3f9a1c2e` |
| Compilation job ID | `cmp_` + 8 hex chars | `cmp_7b41d0e9` |
| `doc_id` | Uploaded file stem, uppercased, with characters other than letters, digits, `_` and `-` replaced by `_` (max 50 chars), then `_` + 6 random uppercase hex chars. Each upload gets a new `doc_id`, even for the same file. | `policy.md` → `POLICY_66C62B` |
| Section ID | `{NAME}_S{nnn}`, where `NAME` is the file stem with non-alphanumerics replaced by `_` | `POLICY_S002` |
| Chunk ID | `{section_id}_C{nnn}` | `POLICY_S002_C000` |
| `clause_id` | `{doc_id}_{chunk_id}_CL{nnn}` | `POLICY_66C62B_POLICY_S002_C000_CL001` |
| `artifact_id` | `{clause_id}_{format}` | `POLICY_66C62B_POLICY_S002_C000_CL001_yaml` |
| Validation `trace_id` | `TRC_` + 8 uppercase hex chars | `TRC_D1A63B5B` |
| Lineage `mapping_id` | `map_{clause_id}` | `map_POLICY_66C62B_POLICY_S002_C000_CL001` |
| Provenance `graph_id` | `graph:{doc_id}:{8 hex chars}` | `graph:POLICY_66C62B:53c4123b` |

---

## Endpoints

### Health Check

#### GET `/api/v1/health`

Check the API server health status. This endpoint requires no API key and is exempt from rate limiting.

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Server health status (always `"healthy"`) |
| `version` | string | AegisLang version (`0.1.0`) |
| `timestamp` | string | Current server time (ISO 8601, `+00:00`) |
| `llm_provider` | string | Clause parser that `/compile` will use: `"anthropic"` if `ANTHROPIC_API_KEY` is set, else `"openai"` if `OPENAI_API_KEY` is set, else `"mock"` (regex-based parser, no LLM calls) |

**Example Response:**

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2026-10-04T12:00:00.915498+00:00",
  "llm_provider": "mock"
}
```

---

### Document Ingestion

#### POST `/api/v1/ingest`

Upload a policy document for ingestion. The endpoint accepts multipart form data, stores the upload in a private temporary directory and returns immediately. Ingestion (text extraction, section hierarchy, chunking) runs as a background job; poll the returned `status_url` until the job is `completed`. The temporary file is overwritten and deleted after ingestion.

Ingestion does **not** extract clauses. Clauses, artifacts and traces are produced by [`POST /api/v1/compile`](#compilation).

**Request:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Document file (PDF, DOCX, Markdown, HTML) |
| `metadata` | string (JSON) | No | A JSON **object** encoded as a string. Defaults to `{}` |

**Supported File Types:**
- `.pdf` - PDF documents
- `.docx` - Microsoft Word documents
- `.md`, `.markdown` - Markdown files
- `.html`, `.htm` - HTML files

Only the file extension is checked; the content is not inspected before ingestion. Files larger than `AEGISLANG_MAX_FILE_SIZE` bytes (default 52428800, i.e. 50 MB) are rejected with 413.

**Metadata:**

Metadata is free-form. No fields are required or validated. Its keys are merged into the stored document metadata and override generated fields with the same name. Example:

```json
{
  "document_name": "CDD Policy",
  "document_type": "regulation",
  "jurisdiction": "US",
  "effective_date": "2026-01-01"
}
```

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Request status (`"accepted"`) |
| `job_id` | string | Ingestion job ID (`ing_` + 8 hex chars) |
| `doc_id` | string | Document ID assigned to this upload |
| `estimated_completion` | string \| null | Always `null` (reserved) |
| `status_url` | string | Relative URL of the job status endpoint |

**Errors:** 400 (missing or invalid filename, unsupported extension, `metadata` not valid JSON, `metadata` not a JSON object), 401/403, 413 (file too large), 422 (no `file` part), 429.

**Example Request (cURL):**

```bash
curl -X POST "http://localhost:8080/api/v1/ingest" \
  -H "X-API-Key: your-api-key" \
  -F "file=@policy.md" \
  -F 'metadata={"document_name": "CDD Policy", "jurisdiction": "US"}'
```

**Example Response:**

```json
{
  "status": "accepted",
  "job_id": "ing_3f9a1c2e",
  "doc_id": "POLICY_66C62B",
  "estimated_completion": null,
  "status_url": "/api/v1/jobs/ing_3f9a1c2e"
}
```

---

### Documents

#### GET `/api/v1/documents`

List all ingested documents. Returns a JSON array (empty if nothing has been ingested).

**Response item:**

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | string | Document ID |
| `metadata` | object | Document metadata (see below) |
| `section_count` | integer | Number of sections |

**Example Response:**

```json
[
  {
    "doc_id": "POLICY_66C62B",
    "metadata": {
      "source_file": "policy.md",
      "ingestion_timestamp": "2026-10-04T12:00:01.666215+00:00",
      "document_type": "markdown",
      "page_count": null,
      "language": "en",
      "hash": "33ea08b5e6e4a9113c28a9a70b1a30452235e7b4c3ee7c4e0214fea5a77d81fd",
      "document_name": "CDD Policy",
      "jurisdiction": "US"
    },
    "section_count": 4
  }
]
```

#### GET `/api/v1/documents/{doc_id}`

Retrieve document metadata. A document becomes available only after its ingestion job has completed. Until then, and for unknown IDs, the endpoint returns 404.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `doc_id` | string | Document ID returned by `POST /ingest` |

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | string | Document ID |
| `metadata` | object | Document metadata |
| `section_count` | integer | Number of sections |
| `status` | string | Always `"processed"` (documents are stored only after successful ingestion) |

**Metadata fields** (generated by the ingestor, then merged with the metadata supplied at upload):

| Field | Description |
|-------|-------------|
| `source_file` | Uploaded file name |
| `ingestion_timestamp` | When ingestion finished (ISO 8601, `+00:00`) |
| `document_type` | `pdf`, `docx`, `markdown` or `html` |
| `page_count` | Page count for PDFs, otherwise `null` |
| `language` | Always `"en"` (no language detection) |
| `hash` | SHA-256 of the uploaded file |

**Example Response:**

```json
{
  "doc_id": "POLICY_66C62B",
  "metadata": {
    "source_file": "policy.md",
    "ingestion_timestamp": "2026-10-04T12:00:01.666215+00:00",
    "document_type": "markdown",
    "page_count": null,
    "language": "en",
    "hash": "33ea08b5e6e4a9113c28a9a70b1a30452235e7b4c3ee7c4e0214fea5a77d81fd",
    "document_name": "CDD Policy",
    "jurisdiction": "US"
  },
  "section_count": 4,
  "status": "processed"
}
```

---

### Clauses

#### GET `/api/v1/clauses/{doc_id}`

List all parsed clauses for a document. Clauses exist only after a [`POST /compile`](#compilation) job has parsed the document; before that the endpoint returns 404 (`Clauses not found for document`). Each new compile job for the same document replaces its clauses.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `doc_id` | string | Document ID |

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | string | Document ID |
| `clause_count` | integer | Number of clauses |
| `clauses` | array | List of parsed clauses |

**Clause fields:**

| Field | Type | Description |
|-------|------|-------------|
| `clause_id` | string | `{doc_id}_{chunk_id}_CL{nnn}` |
| `source_chunk_id` | string | Chunk the clause was extracted from |
| `source_text` | string | Original sentence |
| `type` | string | `obligation`, `prohibition`, `permission`, `conditional`, `definition` or `exception` |
| `actor` | object | `{entity, qualifiers[]}` |
| `action` | object | `{verb, modifiers[]}` |
| `object` | object \| null | `{entity, qualifiers[]}` |
| `condition` | object \| null | `{trigger, temporal}` |
| `temporal_scope` | object \| null | `{deadline, frequency, duration}` |
| `cross_references` | array | Always `[]` (cross-reference resolution is not implemented) |
| `confidence` | number | Parser confidence, 0–1 |

**Clause Object:**

```json
{
  "clause_id": "POLICY_66C62B_POLICY_S002_C000_CL002",
  "source_chunk_id": "POLICY_S002_C000",
  "source_text": "Covered institutions shall retain identification records for five years after the account is closed.",
  "type": "obligation",
  "actor": {
    "entity": "Covered institutions",
    "qualifiers": []
  },
  "action": {
    "verb": "retain",
    "modifiers": []
  },
  "object": {
    "entity": "identification records",
    "qualifiers": []
  },
  "condition": {
    "trigger": "the account is closed",
    "temporal": null
  },
  "temporal_scope": {
    "deadline": null,
    "frequency": null,
    "duration": "five years"
  },
  "cross_references": [],
  "confidence": 0.93
}
```

---

### Rules

#### GET `/api/v1/rules/{clause_id}`

Retrieve the generated artifacts for a specific clause, one per output format requested in the compile job. Returns 404 (`Rule not found`) if no artifacts exist for the clause.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `clause_id` | string | Clause ID (from `GET /clauses/{doc_id}`) |

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `clause_id` | string | Clause ID |
| `doc_id` | string | Parent document ID |
| `artifacts` | array | Generated artifacts |

**Artifact fields:**

| Field | Type | Description |
|-------|------|-------------|
| `artifact_id` | string | `{clause_id}_{format}` |
| `clause_id` | string | Source clause |
| `format` | string | `yaml`, `sql` or `python` |
| `content` | string | Generated artifact text |
| `file_path` | string | Relative path the artifact would be written to (`{doc_id}/{clause_id}.{yaml\|sql\|py}`). The API does not write files to disk; use `content`. |
| `syntax_valid` | boolean | Result of the syntax check (YAML: `yaml.safe_load`; Python: `ast`/`compile`; SQL: `sqlparse` tokenization, which is lenient) |
| `template_used` | string | Built-in template, e.g. `yaml/obligation`, `sql/default`, `python/default` |
| `compilation_timestamp` | string | ISO 8601, `+00:00` |
| `warnings` | array | Syntax-check warnings |

**Artifact Object** (`content` truncated):

```json
{
  "artifact_id": "POLICY_66C62B_POLICY_S002_C000_CL001_yaml",
  "clause_id": "POLICY_66C62B_POLICY_S002_C000_CL001",
  "format": "yaml",
  "content": "# Source: Financial institutions must verify the identity of each customer before opening ...\n# Clause ID: POLICY_66C62B_POLICY_S002_C000_CL001\n...\ncontrol:\n  id: POLICY_66C62B_POLICY_S002_C000_CL001\n  type: obligation\n  rule: \"verify identity\"\n...",
  "file_path": "POLICY_66C62B/POLICY_66C62B_POLICY_S002_C000_CL001.yaml",
  "syntax_valid": true,
  "template_used": "yaml/obligation",
  "compilation_timestamp": "2026-10-04T12:00:02.820020+00:00",
  "warnings": []
}
```

---

### Traceability

#### GET `/api/v1/trace/{doc_id}`

Retrieve the validation results and provenance graph produced by the most recent compile job for a document. This is the API's audit trail. Returns 404 (`Trace not found for document`) until a compile job has finished.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `doc_id` | string | Document ID |

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | string | Document ID |
| `summary` | object | `{total, passed, failed, needs_review}` counts over all artifacts |
| `validation_timestamp` | string | ISO 8601, `+00:00` |
| `results` | array | One validation result per artifact (see below) |
| `provenance_graph` | object | `{graph_id, doc_id, nodes[], edges[], created_at}` |

**Validation result fields:**

| Field | Description |
|-------|-------------|
| `trace_id` | `TRC_` + 8 hex chars |
| `source_clause` | Clause ID |
| `generated_artifact` | Artifact `file_path` |
| `validation_status` | `passed`, `failed` or `needs_review` |
| `confidence_score` | Average of parser and mapping confidences |
| `validation_checks` | `[{check_name, passed, message, details}]`. Checks: `chain_completeness`, `syntax_validity`, `confidence_threshold`, `semantic_alignment`, `cross_reference_integrity` |
| `lineage` | `{document_id, section_id, chunk_id, clause_id, mapping_id, artifact_id}` |
| `review_flags` | e.g. `low_confidence`, `below_threshold`, `semantic_drift`, `syntax_error`, `incomplete_chain`, `blocked_low_confidence` |
| `validated_at` | ISO 8601, `+00:00` |
| `validated_by` | Validator identifier |

A result is `failed` if the chain-completeness or syntax check fails or confidence is below 0.50. It is `needs_review` if any review flag is raised, for example confidence below the compile request's `confidence_threshold` (default 0.85) or below 0.70. Otherwise it is `passed`.

**Provenance graph:** nodes have `{node_id, node_type, properties, created_at}`, with `node_type` one of `document`, `section`, `chunk`, `clause` or `artifact` (node IDs are prefixed `doc:`, `section:`, `chunk:`, `clause:`, `artifact:`). Edges have `{edge_id, source_id, target_id, relationship, properties}`, with relationships `CONTAINS_SECTION`, `CONTAINS_CHUNK`, `PARSED_TO` and `COMPILED_TO`. The mapping step is recorded in each result's `lineage.mapping_id`.

**Example Response** (abbreviated):

```json
{
  "doc_id": "POLICY_66C62B",
  "summary": {"total": 12, "passed": 12, "failed": 0, "needs_review": 0},
  "validation_timestamp": "2026-10-04T12:00:02.988982+00:00",
  "results": [
    {
      "trace_id": "TRC_D1A63B5B",
      "source_clause": "POLICY_66C62B_POLICY_S002_C000_CL001",
      "generated_artifact": "POLICY_66C62B/POLICY_66C62B_POLICY_S002_C000_CL001.yaml",
      "validation_status": "passed",
      "confidence_score": 0.945,
      "validation_checks": [
        {
          "check_name": "confidence_threshold",
          "passed": true,
          "message": "Confidence 0.94 meets threshold 0.85",
          "details": {"confidence": 0.945, "threshold": 0.85}
        }
      ],
      "lineage": {
        "document_id": "POLICY_66C62B",
        "section_id": "POLICY_S002",
        "chunk_id": "POLICY_S002_C000",
        "clause_id": "POLICY_66C62B_POLICY_S002_C000_CL001",
        "mapping_id": "map_POLICY_66C62B_POLICY_S002_C000_CL001",
        "artifact_id": "POLICY_66C62B_POLICY_S002_C000_CL001_yaml"
      },
      "review_flags": [],
      "validated_at": "2026-10-04T12:00:02.987334+00:00",
      "validated_by": "aegislang-validator-v1.0.0"
    }
  ],
  "provenance_graph": {
    "graph_id": "graph:POLICY_66C62B:53c4123b",
    "doc_id": "POLICY_66C62B",
    "nodes": [
      {
        "node_id": "doc:POLICY_66C62B",
        "node_type": "document",
        "properties": {"doc_id": "POLICY_66C62B"},
        "created_at": "2026-10-04T12:00:02.989578+00:00"
      },
      {
        "node_id": "clause:POLICY_66C62B_POLICY_S002_C000_CL001",
        "node_type": "clause",
        "properties": {"clause_id": "POLICY_66C62B_POLICY_S002_C000_CL001", "confidence": 0.945, "status": "passed"},
        "created_at": "2026-10-04T12:00:02.989578+00:00"
      }
    ],
    "edges": [
      {
        "edge_id": "e:POLICY_66C62B_POLICY_S002_C000_CL001->POLICY_66C62B_POLICY_S002_C000_CL001_yaml",
        "source_id": "clause:POLICY_66C62B_POLICY_S002_C000_CL001",
        "target_id": "artifact:POLICY_66C62B_POLICY_S002_C000_CL001_yaml",
        "relationship": "COMPILED_TO",
        "properties": {"confidence": 0.945, "validated": true}
      }
    ],
    "created_at": "2026-10-04T12:00:02.989578+00:00"
  }
}
```

---

### Compilation

#### POST `/api/v1/compile`

Trigger the full compilation pipeline for an ingested document. The endpoint returns immediately with a job ID; a background job then runs:

1. **Parse**: extracts clauses using the provider reported as `llm_provider` by `/health` (Anthropic, OpenAI, or the regex-based mock parser when no LLM key is set).
2. **Map**: links clause entities to schema fields using exact, synonym and embedding matching. The API always uses deterministic mock embeddings (hash-based, not semantic) with a fixed mapping threshold of 0.7. The mapper never calls an LLM.
3. **Compile**: generates artifacts from the built-in templates in `compiler_agent.py`. The repository's `templates/` directory is not used by the API.
4. **Validate**: checks provenance, syntax and confidence, and builds the provenance graph.

When the job completes, results are available from [`GET /clauses/{doc_id}`](#clauses), [`GET /rules/{clause_id}`](#rules) and [`GET /trace/{doc_id}`](#traceability). Compiling the same document again replaces its clauses, artifacts and trace.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `doc_id` | string | Yes | - | Document ID to compile (must be fully ingested) |
| `output_formats` | array | No | `["yaml", "sql"]` | One or more of `yaml`, `sql`, `python` |
| `target_schema` | string | No | null | Restrict mapping to one schema: a built-in schema (`kyc_schema`, `org_schema`, `records_schema`) or one registered via `POST /schemas`. If omitted, all built-in and registered schemas are searched. |
| `confidence_threshold` | float | No | 0.85 | Validator threshold: artifacts whose confidence is below it are marked `needs_review` |

**Supported Output Formats:**
- `yaml` - YAML compliance rules (`control:` blocks)
- `sql` - SQL constraints and triggers: obligations produce `ALTER TABLE ... ADD CONSTRAINT ... CHECK` plus a PL/pgSQL trigger, prohibitions produce `CHECK (NOT ...)`, and other clause types produce a comment-only artifact
- `python` - Python pytest stubs

Any other value (for example `terraform`, `rego` or `json`, which exist only in the CLI and library as experimental formats) is rejected with 422.

**Built-in target schemas:**

| Schema ID | Tables (fields) |
|-----------|-----------------|
| `kyc_schema` | `customer` (`customer_id`, `identity_verified`, `verification_date`), `transaction` (`transaction_id`, `amount`, `reported`) |
| `org_schema` | `institution` (`institution_id`, `license_status`), `employee` (`employee_id`, `access_level`) |
| `records_schema` | `audit_record` (`record_id`, `retention_period`, `created_at`) |

**Errors:** 404 (`Document not found`, or `Schema not found: <id>` for an unknown `target_schema`), 422 (missing `doc_id`, empty or unsupported `output_formats`), 401/403, 429. Failures inside the pipeline do not produce an HTTP error; they are reported as a `failed` job with an `error` message.

**Example Request:**

```json
{
  "doc_id": "POLICY_66C62B",
  "output_formats": ["yaml", "sql", "python"],
  "target_schema": "kyc_schema",
  "confidence_threshold": 0.85
}
```

**Example Response:**

```json
{
  "status": "accepted",
  "job_id": "cmp_7b41d0e9",
  "doc_id": "POLICY_66C62B",
  "status_url": "/api/v1/jobs/cmp_7b41d0e9"
}
```

---

### Jobs

#### GET `/api/v1/jobs/{job_id}`

Check the status of an async job (ingestion or compilation). Finished jobs (`completed` or `failed`) are deleted `AEGISLANG_JOB_TTL_SECONDS` after completion (default 86400, i.e. 24 hours). After that the endpoint returns 404. Documents, clauses, artifacts and traces do not expire.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Job ID (`ing_…` or `cmp_…`) |

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | Job ID |
| `status` | string | Job status |
| `created_at` | string | Job creation timestamp |
| `completed_at` | string \| null | Completion timestamp (set when `completed` or `failed`) |
| `result` | object \| null | Job result (when `completed`) |
| `error` | string \| null | Error message (when `failed`) |

**Job Status Values:**
- `pending` - Job is queued
- `processing` - Job is running
- `completed` - Job finished successfully
- `failed` - Job encountered an error

**Result shapes:**

| Job type | `result` |
|----------|----------|
| Ingestion (`ing_…`) | `{doc_id, sections}`, where `sections` is the number of sections |
| Compilation (`cmp_…`) | `{doc_id, clauses_parsed, artifacts_generated, validation_summary: {total, passed, failed, needs_review}}` |

**Example Response (Ingestion completed):**

```json
{
  "job_id": "ing_3f9a1c2e",
  "status": "completed",
  "created_at": "2026-10-04T12:00:00.931825+00:00",
  "completed_at": "2026-10-04T12:00:01.666386+00:00",
  "result": {
    "doc_id": "POLICY_66C62B",
    "sections": 4
  },
  "error": null
}
```

**Example Response (Compilation completed):**

```json
{
  "job_id": "cmp_7b41d0e9",
  "status": "completed",
  "created_at": "2026-10-04T12:00:02.719862+00:00",
  "completed_at": "2026-10-04T12:00:02.989604+00:00",
  "result": {
    "doc_id": "POLICY_66C62B",
    "clauses_parsed": 4,
    "artifacts_generated": 12,
    "validation_summary": {
      "total": 12,
      "passed": 12,
      "failed": 0,
      "needs_review": 0
    }
  },
  "error": null
}
```

#### GET `/api/v1/jobs/{job_id}/stream`

Stream job status updates as Server-Sent Events (`Content-Type: text/event-stream`). The server sends one event per second, each a `data:` line holding `{job_id, status, result, error}`. The stream closes after the event that reports `completed` or `failed`. Returns 404 if the job does not exist.

The endpoint requires the `X-API-Key` header like every other endpoint. The browser `EventSource` API cannot send custom headers, so use `fetch()` with a streaming response reader, or disable auth for local development.

**Example:**

```bash
curl -N -H "X-API-Key: your-api-key" \
  "http://localhost:8080/api/v1/jobs/cmp_7b41d0e9/stream"
```

```
data: {"job_id": "cmp_7b41d0e9", "status": "processing", "result": null, "error": null}

data: {"job_id": "cmp_7b41d0e9", "status": "completed", "result": {"doc_id": "POLICY_66C62B", "clauses_parsed": 4, "artifacts_generated": 12, "validation_summary": {"total": 12, "passed": 12, "failed": 0, "needs_review": 0}}, "error": null}
```

---

### Schemas

Schemas are the mapping targets for clause entities. Three built-in schemas (`kyc_schema`, `org_schema`, `records_schema`; see [Compilation](#compilation)) are always available to `POST /compile` but are not listed by these endpoints. Schemas registered here are added to the built-ins for compilation, and a registered schema with the same ID as a built-in one replaces it.

#### POST `/api/v1/schemas`

Register or replace a target schema for entity mapping. Posting an existing `schema_id` again replaces the stored schema entirely. `version` is only a label; no version history is kept. There is no delete endpoint.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `schema_id` | string | Yes | - | Unique schema identifier (non-empty) |
| `schema_type` | string | Yes | - | `sql`, `api` or `object` (stored as metadata; mapping treats all types the same) |
| `version` | string | No | `"1.0.0"` | Version label |
| `tables` | array | Yes | - | Table definitions |

**Table Definition:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `table_name` | string | Yes | Table or object name |
| `description` | string | No | Free-text description |
| `fields` | array | No | Fields: `{field_name, field_type, semantic_labels?, description?}` |

`semantic_labels` (default `[]`) are matched against clause entities. Mapped entities get target paths such as `customer.identity_verified`.

```json
{
  "table_name": "customer",
  "fields": [
    {
      "field_name": "customer_id",
      "field_type": "UUID",
      "semantic_labels": ["customer", "client", "user"]
    },
    {
      "field_name": "identity_verified",
      "field_type": "BOOLEAN",
      "semantic_labels": ["identity", "verification", "kyc"]
    }
  ]
}
```

**Errors:** 422 for an invalid body (for example an unknown `schema_type` or a missing `tables`), with the field errors in `details.errors`.

**Example Response:**

```json
{
  "status": "registered",
  "schema_id": "bank_schema",
  "version": "1.0.0"
}
```

#### GET `/api/v1/schemas`

List the schemas registered via `POST /schemas`. Built-in schemas are not included.

**Response:**

```json
{
  "schemas": [
    {
      "schema_id": "bank_schema",
      "schema_type": "sql",
      "version": "1.0.0",
      "tables": [...],
      "registered_at": "2026-10-04T12:00:03.068026+00:00"
    }
  ],
  "count": 1
}
```

#### GET `/api/v1/schemas/{schema_id}`

Get a schema registered via `POST /schemas`. Returns 404 (`Schema not found`) for unknown IDs and for built-in schemas that have not been re-registered.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `schema_id` | string | Schema ID |

**Response:**

```json
{
  "schema_id": "bank_schema",
  "schema_type": "sql",
  "version": "1.0.0",
  "tables": [...],
  "registered_at": "2026-10-04T12:00:03.068026+00:00"
}
```

---

## Error Handling

The API uses standard HTTP status codes:

| Status Code | Description |
|-------------|-------------|
| 200 | Success (including accepted async jobs) |
| 400 | Bad Request: invalid upload (filename, extension, metadata) |
| 401 | Missing `X-API-Key` header |
| 403 | Invalid API key |
| 404 | Not Found: unknown resource, resource not produced yet, or unknown route |
| 405 | Method Not Allowed |
| 413 | Uploaded file exceeds `AEGISLANG_MAX_FILE_SIZE` |
| 422 | Request validation failed (missing or invalid fields) |
| 429 | Rate limit exceeded (see `Retry-After` header) |
| 500 | Internal Server Error |

**Error Response Format:**

| Field | Type | Always present | Description |
|-------|------|----------------|-------------|
| `error` | string | Yes | Human-readable message |
| `status_code` | integer | Yes | HTTP status code |
| `request_id` | string | Yes | Same value as the `X-Request-ID` response header |
| `error_code` | string | No | `VALIDATION_ERROR` (422) or `INTERNAL_ERROR` (500) |
| `details` | object | No | For 422: `{"errors": [{field, message, type}]}` |

```json
{
  "error": "Missing API key. Provide X-API-Key header.",
  "status_code": 401,
  "request_id": "3fbb7689-33cd-41ba-8c46-a761c919c474"
}
```

**Validation error (422):**

```json
{
  "error": "Request validation failed",
  "status_code": 422,
  "error_code": "VALIDATION_ERROR",
  "request_id": "e5fe6824-4f6f-4988-9399-2b3c9101d95c",
  "details": {
    "errors": [
      {
        "field": "body.output_formats.0",
        "message": "Input should be 'yaml', 'sql' or 'python'",
        "type": "literal_error"
      }
    ]
  }
}
```

Unhandled exceptions return 500 with `error_code: "INTERNAL_ERROR"`. When `AEGISLANG_ENV=production`, the message is replaced by a generic one (for example `An internal error occurred`) so internal details are not exposed.

---

## Rate Limiting

The API includes a built-in in-memory rate limiter (sliding one-minute and one-hour windows). Default limits:

| Window | Limit | Environment Variable |
|--------|-------|---------------------|
| Per minute | 60 requests | `AEGISLANG_RATE_LIMIT_MINUTE` |
| Per hour | 1000 requests | `AEGISLANG_RATE_LIMIT_HOUR` |

- Limits apply per API key. When auth is disabled, all clients share a single `anonymous` bucket.
- `GET /api/v1/health` is exempt.
- When a limit is exceeded the API returns HTTP 429 with a `Retry-After` header (`60` for the per-minute limit, `3600` for the per-hour limit) and a body such as `{"error": "Rate limit exceeded: 60/minute", "status_code": 429, "request_id": "..."}`.
- Counters are kept in memory per server process. They are not shared between workers or instances and reset on restart.

For production deployments, you may also configure rate limiting at the load balancer or API gateway level.

---

## CORS

CORS is enabled through FastAPI's `CORSMiddleware`:

| Setting | Value |
|---------|-------|
| Allowed origins | `CORS_ORIGINS` (comma-separated; default `http://localhost:3000`) |
| Allowed methods | `GET`, `POST`, `OPTIONS` |
| Allowed request headers | `X-API-Key`, `Content-Type`, `X-Request-ID`, `Accept` |
| Credentials | Allowed |

Response headers such as `X-Request-ID` are not exposed to cross-origin JavaScript; read `request_id` from error bodies instead.

---

## Configuration

Environment variables read by the API server. Python does not load `.env` automatically: export the variables, or use `docker compose`, which substitutes values from `.env` into `docker-compose.yml`.

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind address (used by `python -m aegislang.api.server`) |
| `PORT` | `8080` | Listen port |
| `WORKERS` | `4` | Uvicorn workers; forced to 1 with the memory backend or `RELOAD=true` |
| `RELOAD` | `false` | Auto-reload on code changes |
| `AEGISLANG_API_KEYS` | unset | Comma-separated valid API keys |
| `AEGISLANG_DISABLE_AUTH` | `false` | `true` disables API key checks |
| `AEGISLANG_RATE_LIMIT_MINUTE` | `60` | Requests per minute per key |
| `AEGISLANG_RATE_LIMIT_HOUR` | `1000` | Requests per hour per key |
| `AEGISLANG_STORAGE_BACKEND` | `memory` | `memory` (lost on restart) or `sqlite` (persistent); other values fall back to `memory` |
| `AEGISLANG_SQLITE_PATH` | `aegislang_data.db` | SQLite database file (sqlite backend only) |
| `AEGISLANG_JOB_TTL_SECONDS` | `86400` | Seconds a finished job is kept |
| `AEGISLANG_MAX_FILE_SIZE` | `52428800` | Maximum upload size in bytes (50 MB) |
| `AEGISLANG_ENV` | `development` | `production` (or `prod`): generic 500 messages and JSON console logs |
| `AEGISLANG_LOG_LEVEL` / `LOG_LEVEL` | `INFO` | Log level (`AEGISLANG_LOG_LEVEL` takes precedence) |
| `AEGISLANG_LOG_FILE` | unset | Also write JSON logs to this file |
| `SENTRY_DSN` | unset | Enable Sentry error reporting |
| `AEGISLANG_VERSION` | unset | Release name reported to Sentry |
| `CORS_ORIGINS` | `http://localhost:3000` | Allowed CORS origins |
| `ANTHROPIC_API_KEY` | unset | Use Anthropic for clause parsing |
| `OPENAI_API_KEY` | unset | Use OpenAI for clause parsing when no Anthropic key is set |

The logging settings (`AEGISLANG_LOG_LEVEL`, `AEGISLANG_LOG_FILE`, `SENTRY_DSN`, JSON output) are applied when the server is started with `python -m aegislang.api.server` (`make run`). Running `uvicorn aegislang.api.server:app` directly skips that setup.

The application does not use PostgreSQL, Redis or Neo4j, and serves plain HTTP. Put a TLS-terminating reverse proxy in front of it for any non-local deployment.

---

## OpenAPI Specification

The full OpenAPI (Swagger) specification is available at the following paths, none of which require an API key:

- **Swagger UI:** `/api/docs`
- **ReDoc:** `/api/redoc`
- **OpenAPI JSON:** `/api/openapi.json`

---

## Example Workflow

### Complete Document Processing

The steps below use a small sample policy. Start the server first, for example with `AEGISLANG_API_KEYS=my-local-key python -m aegislang.api.server`. IDs and timestamps in your responses will differ from the examples.

```bash
export API=http://localhost:8080/api/v1
export KEY=my-local-key   # one of the values in AEGISLANG_API_KEYS
```

0. **Create a sample document:**

```bash
cat > policy.md <<'EOF'
# Customer Due Diligence Policy

## Customer Identification

Financial institutions must verify the identity of each customer before opening an account.

Covered institutions shall retain identification records for five years after the account is closed.

## Prohibited Activities

Banks must not open anonymous accounts.

## Permitted Reliance

A bank may rely on another financial institution to perform customer identification.
EOF
```

1. **Upload document:**

```bash
curl -X POST "$API/ingest" \
  -H "X-API-Key: $KEY" \
  -F "file=@policy.md" \
  -F 'metadata={"document_name": "CDD Policy"}'
# {"status":"accepted","job_id":"ing_3f9a1c2e","doc_id":"POLICY_66C62B","estimated_completion":null,"status_url":"/api/v1/jobs/ing_3f9a1c2e"}
```

2. **Check ingestion status** (repeat until `status` is `completed`):

```bash
curl -H "X-API-Key: $KEY" "$API/jobs/ing_3f9a1c2e"
# {"job_id":"ing_3f9a1c2e","status":"completed",...,"result":{"doc_id":"POLICY_66C62B","sections":4},"error":null}
```

3. **(Optional) Register a custom target schema.** The built-in `kyc_schema`, `org_schema` and `records_schema` are available without this step.

```bash
curl -X POST "$API/schemas" \
  -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "schema_id": "bank_schema",
    "schema_type": "sql",
    "tables": [
      {
        "table_name": "customer",
        "fields": [
          {"field_name": "customer_id", "field_type": "UUID", "semantic_labels": ["customer", "client"]},
          {"field_name": "identity_verified", "field_type": "BOOLEAN", "semantic_labels": ["identity", "verification"]}
        ]
      }
    ]
  }'
# {"status":"registered","schema_id":"bank_schema","version":"1.0.0"}
```

4. **Compile document** (this is the step that parses clauses):

```bash
curl -X POST "$API/compile" \
  -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "POLICY_66C62B",
    "output_formats": ["yaml", "sql", "python"],
    "target_schema": "kyc_schema"
  }'
# {"status":"accepted","job_id":"cmp_7b41d0e9","doc_id":"POLICY_66C62B","status_url":"/api/v1/jobs/cmp_7b41d0e9"}
```

5. **Check compilation status** (poll, or stream until done):

```bash
curl -H "X-API-Key: $KEY" "$API/jobs/cmp_7b41d0e9"
# or
curl -N -H "X-API-Key: $KEY" "$API/jobs/cmp_7b41d0e9/stream"
```

6. **Retrieve parsed clauses:**

```bash
curl -H "X-API-Key: $KEY" "$API/clauses/POLICY_66C62B"
```

7. **Retrieve generated rules** for one clause:

```bash
curl -H "X-API-Key: $KEY" "$API/rules/POLICY_66C62B_POLICY_S002_C000_CL001"
```

8. **Retrieve the validation trace and provenance graph:**

```bash
curl -H "X-API-Key: $KEY" "$API/trace/POLICY_66C62B"
```

### Scripted Version

The same flow as a script that captures the IDs automatically (requires `jq`):

```bash
#!/usr/bin/env bash
set -euo pipefail
API=${API:-http://localhost:8080/api/v1}
KEY=${KEY:?set KEY to one of the keys in AEGISLANG_API_KEYS}

wait_for_job() {
  local job_id=$1 status
  while true; do
    status=$(curl -sf -H "X-API-Key: $KEY" "$API/jobs/$job_id" | jq -r .status)
    case "$status" in
      completed) return 0 ;;
      failed) curl -s -H "X-API-Key: $KEY" "$API/jobs/$job_id" | jq .; return 1 ;;
    esac
    sleep 1
  done
}

INGEST=$(curl -sf -H "X-API-Key: $KEY" \
  -F "file=@policy.md" -F 'metadata={"document_name": "CDD Policy"}' "$API/ingest")
DOC_ID=$(jq -r .doc_id <<<"$INGEST")
wait_for_job "$(jq -r .job_id <<<"$INGEST")"

COMPILE=$(curl -sf -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d "{\"doc_id\": \"$DOC_ID\", \"output_formats\": [\"yaml\", \"sql\", \"python\"], \"target_schema\": \"kyc_schema\"}" \
  "$API/compile")
wait_for_job "$(jq -r .job_id <<<"$COMPILE")"

curl -sf -H "X-API-Key: $KEY" "$API/clauses/$DOC_ID" | jq '.clauses[] | {clause_id, type, source_text}'
CLAUSE_ID=$(curl -sf -H "X-API-Key: $KEY" "$API/clauses/$DOC_ID" | jq -r '.clauses[0].clause_id')
curl -sf -H "X-API-Key: $KEY" "$API/rules/$CLAUSE_ID" | jq -r '.artifacts[] | select(.format == "yaml") | .content'
curl -sf -H "X-API-Key: $KEY" "$API/trace/$DOC_ID" | jq '.summary'
```
