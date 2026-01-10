# AegisLang REST API Documentation

The AegisLang API provides a RESTful interface for document ingestion, parsing, schema mapping, compilation, and validation operations.

**Base URL:** `/api/v1`

**API Version:** `1.0.0`

---

## Table of Contents

- [Authentication](#authentication)
- [Common Response Formats](#common-response-formats)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [Document Ingestion](#document-ingestion)
  - [Documents](#documents)
  - [Clauses](#clauses)
  - [Rules](#rules)
  - [Compilation](#compilation)
  - [Jobs](#jobs)
  - [Schemas](#schemas)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)

---

## Authentication

Currently, the API does not require authentication for development purposes. In production deployments, JWT-based authentication should be configured via the `JWT_SECRET` environment variable.

---

## Common Response Formats

### Success Response

```json
{
  "status": "success",
  "data": { ... }
}
```

### Error Response

```json
{
  "error": "Error message",
  "status_code": 400
}
```

---

## Endpoints

### Health Check

#### GET `/api/v1/health`

Check the API server health status.

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Server health status ("healthy") |
| `version` | string | API version |
| `timestamp` | string | ISO 8601 timestamp |

**Example Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-10T12:00:00Z"
}
```

---

### Document Ingestion

#### POST `/api/v1/ingest`

Upload and parse a new policy document. This endpoint accepts multipart form data.

**Request:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Document file (PDF, DOCX, MD, HTML) |
| `metadata` | string (JSON) | No | Document metadata as JSON string |

**Supported File Types:**
- `.pdf` - PDF documents
- `.docx` - Microsoft Word documents
- `.md`, `.markdown` - Markdown files
- `.html`, `.htm` - HTML files

**Metadata Schema:**

```json
{
  "document_name": "AML Policy",
  "document_type": "regulation",
  "jurisdiction": "US",
  "effective_date": "2025-01-01"
}
```

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Request status ("accepted") |
| `job_id` | string | Async job ID for tracking |
| `doc_id` | string | Assigned document ID |
| `estimated_completion` | string | Estimated completion time (optional) |
| `webhook_url` | string | URL to check job status |

**Example Request (cURL):**

```bash
curl -X POST "http://localhost:8080/api/v1/ingest" \
  -F "file=@policy.md" \
  -F 'metadata={"document_name": "AML Policy", "jurisdiction": "US"}'
```

**Example Response:**

```json
{
  "status": "accepted",
  "job_id": "ing_a1b2c3d4",
  "doc_id": "AML_POLICY_E5F6G7",
  "webhook_url": "/api/v1/jobs/ing_a1b2c3d4"
}
```

---

### Documents

#### GET `/api/v1/documents/{doc_id}`

Retrieve document metadata and processing status.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `doc_id` | string | Document ID |

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | string | Document ID |
| `metadata` | object | Document metadata |
| `section_count` | integer | Number of sections |
| `status` | string | Processing status |

**Example Response:**

```json
{
  "doc_id": "AML_POLICY_E5F6G7",
  "metadata": {
    "document_name": "AML Policy",
    "jurisdiction": "US"
  },
  "section_count": 15,
  "status": "processed"
}
```

---

### Clauses

#### GET `/api/v1/clauses/{doc_id}`

List all parsed clauses for a document.

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

**Clause Object:**

```json
{
  "clause_id": "CL_001",
  "source_chunk_id": "C_001",
  "source_text": "Financial institutions must verify customer identity.",
  "type": "obligation",
  "actor": {
    "entity": "financial institutions",
    "qualifiers": []
  },
  "action": {
    "verb": "verify",
    "modifiers": []
  },
  "object": {
    "entity": "customer identity",
    "qualifiers": []
  },
  "condition": null,
  "confidence": 0.92
}
```

---

### Rules

#### GET `/api/v1/rules/{clause_id}`

Retrieve generated rule artifacts for a specific clause.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `clause_id` | string | Clause ID |

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `clause_id` | string | Clause ID |
| `doc_id` | string | Parent document ID |
| `artifacts` | array | Generated artifacts |

**Artifact Object:**

```json
{
  "artifact_id": "ART_001",
  "clause_id": "CL_001",
  "format": "yaml",
  "content": "rule_id: RULE_CL_001\ntype: obligation\n...",
  "template_used": "obligation.yaml.j2",
  "syntax_valid": true,
  "generated_at": "2025-01-10T12:00:00Z"
}
```

---

### Compilation

#### POST `/api/v1/compile`

Trigger the full compilation pipeline for a document. This runs parsing, mapping, compilation, and validation.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `doc_id` | string | Yes | - | Document ID to compile |
| `output_formats` | array | No | `["yaml", "sql"]` | Output formats to generate |
| `target_schema` | string | No | null | Target schema ID for mapping |
| `confidence_threshold` | float | No | 0.85 | Minimum confidence threshold |

**Supported Output Formats:**
- `yaml` - YAML compliance rules
- `sql` - SQL constraints and triggers
- `python` - Python test stubs
- `terraform` - Terraform policy rules
- `rego` - Open Policy Agent (Rego)
- `json` - JSON rules

**Example Request:**

```json
{
  "doc_id": "AML_POLICY_E5F6G7",
  "output_formats": ["yaml", "sql", "python"],
  "target_schema": "kyc_schema",
  "confidence_threshold": 0.8
}
```

**Example Response:**

```json
{
  "status": "accepted",
  "job_id": "cmp_h8i9j0k1",
  "doc_id": "AML_POLICY_E5F6G7",
  "webhook_url": "/api/v1/jobs/cmp_h8i9j0k1"
}
```

---

### Jobs

#### GET `/api/v1/jobs/{job_id}`

Check the status of an async job (ingestion or compilation).

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Job ID |

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | Job ID |
| `status` | string | Job status |
| `created_at` | string | Job creation timestamp |
| `completed_at` | string | Completion timestamp (if done) |
| `result` | object | Job result (if completed) |
| `error` | string | Error message (if failed) |

**Job Status Values:**
- `pending` - Job is queued
- `processing` - Job is running
- `completed` - Job finished successfully
- `failed` - Job encountered an error

**Example Response (Completed):**

```json
{
  "job_id": "cmp_h8i9j0k1",
  "status": "completed",
  "created_at": "2025-01-10T12:00:00Z",
  "completed_at": "2025-01-10T12:00:15Z",
  "result": {
    "doc_id": "AML_POLICY_E5F6G7",
    "clauses_parsed": 25,
    "artifacts_generated": 75,
    "validation_summary": {
      "valid_count": 70,
      "needs_review_count": 5,
      "failed_count": 0
    }
  },
  "error": null
}
```

---

### Schemas

#### POST `/api/v1/schemas`

Register or update a target schema for entity mapping.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `schema_id` | string | Yes | - | Unique schema identifier |
| `schema_type` | string | Yes | - | Schema type (sql, api, etc.) |
| `version` | string | No | "1.0.0" | Schema version |
| `tables` | array | Yes | - | Table definitions |

**Table Definition:**

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

**Example Response:**

```json
{
  "status": "registered",
  "schema_id": "kyc_schema",
  "version": "1.0.0"
}
```

#### GET `/api/v1/schemas`

List all registered schemas.

**Response:**

```json
{
  "schemas": [
    {
      "schema_id": "kyc_schema",
      "schema_type": "sql",
      "version": "1.0.0",
      "tables": [...],
      "registered_at": "2025-01-10T12:00:00Z"
    }
  ],
  "count": 1
}
```

#### GET `/api/v1/schemas/{schema_id}`

Get a specific schema by ID.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `schema_id` | string | Schema ID |

**Response:**

```json
{
  "schema_id": "kyc_schema",
  "schema_type": "sql",
  "version": "1.0.0",
  "tables": [...],
  "registered_at": "2025-01-10T12:00:00Z"
}
```

---

## Error Handling

The API uses standard HTTP status codes:

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 404 | Not Found - Resource doesn't exist |
| 422 | Validation Error - Missing required fields |
| 500 | Internal Server Error |

**Error Response Format:**

```json
{
  "error": "Detailed error message",
  "status_code": 400
}
```

---

## Rate Limiting

Rate limiting is not currently implemented. Production deployments should configure rate limiting at the load balancer or API gateway level.

---

## OpenAPI Specification

The full OpenAPI (Swagger) specification is available at:

- **Swagger UI:** `/api/docs`
- **ReDoc:** `/api/redoc`
- **OpenAPI JSON:** `/api/openapi.json`

---

## Example Workflow

### Complete Document Processing

1. **Upload document:**

```bash
curl -X POST "http://localhost:8080/api/v1/ingest" \
  -F "file=@policy.md" \
  -F 'metadata={"document_name": "AML Policy"}'
```

2. **Check ingestion status:**

```bash
curl "http://localhost:8080/api/v1/jobs/ing_a1b2c3d4"
```

3. **Register target schema:**

```bash
curl -X POST "http://localhost:8080/api/v1/schemas" \
  -H "Content-Type: application/json" \
  -d '{
    "schema_id": "kyc_schema",
    "schema_type": "sql",
    "tables": [...]
  }'
```

4. **Compile document:**

```bash
curl -X POST "http://localhost:8080/api/v1/compile" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "AML_POLICY_E5F6G7",
    "output_formats": ["yaml", "sql"],
    "target_schema": "kyc_schema"
  }'
```

5. **Check compilation status:**

```bash
curl "http://localhost:8080/api/v1/jobs/cmp_h8i9j0k1"
```

6. **Retrieve parsed clauses:**

```bash
curl "http://localhost:8080/api/v1/clauses/AML_POLICY_E5F6G7"
```

7. **Retrieve generated rules:**

```bash
curl "http://localhost:8080/api/v1/rules/CL_001"
```
