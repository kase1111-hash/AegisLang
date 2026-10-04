# AegisLang Troubleshooting Guide

This guide helps resolve common issues with AegisLang installation, configuration, and operation.

---

## Table of Contents

- [Installation Issues](#installation-issues)
- [Docker Issues](#docker-issues)
- [Storage Issues](#storage-issues)
- [API Issues](#api-issues)
- [Processing Issues](#processing-issues)
- [Performance Issues](#performance-issues)
- [LLM/AI Issues](#llmai-issues)

---

## Installation Issues

### Python version mismatch

**Symptom:**
```
SyntaxError: invalid syntax
```
or
```
ModuleNotFoundError: No module named 'typing_extensions'
```

**Solution:**
Ensure Python 3.11+ is installed:
```bash
python --version  # Should be 3.11 or higher

# If using pyenv:
pyenv install 3.11
pyenv local 3.11
```

---

### Dependency installation fails

**Symptom:**
```
ERROR: Could not build wheels for torch
```

**Solution:**
torch is not a core dependency. It appears only in `requirements-ml.txt`, which you need only if you pass `SentenceTransformerProvider` to the mapper in library code. The API and CLIs run on `requirements.txt` alone:
```bash
pip install -r requirements.txt
```

If you do need the ML extras:

1. Upgrade pip:
```bash
pip install --upgrade pip wheel setuptools
```

2. Install with pre-built CPU wheels:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-ml.txt
```

---

## Docker Issues

### Container fails to start

**Symptom:**
```
aegislang exited with code 1
```

**Solution:**
1. Check logs:
```bash
docker-compose logs aegislang
```

2. Verify environment variables:
```bash
docker-compose config
```

3. Rebuild the image:
```bash
docker-compose build --no-cache aegislang
```

If the log ends with `sqlite3.OperationalError: unable to open database file`, see [SQLite database cannot be opened](#sqlite-database-cannot-be-opened).

---

### Port already in use

**Symptom:**
```
Error: bind: address already in use
```

**Solution:**
1. Find the process using the port:
```bash
lsof -i :8080
```

2. Kill the process or change the port in `docker-compose.yml`:
```yaml
ports:
  - "8081:8080"  # Use different host port
```

---

### Volume permission issues

**Symptom:**
```
sqlite3.OperationalError: unable to open database file
```
or
```
sqlite3.OperationalError: attempt to write a readonly database
```

**Solution:**
The production image runs as the non-root user `aegislang` and writes its database to `/app/data/aegislang.db`. The default named volume `aegislang-data` gets the right ownership automatically. If you replaced it with a bind mount, make the host directory writable by the container user:
```bash
# Find the container user's UID
docker-compose run --rm aegislang id -u

# Fix ownership of the bind-mounted directory
sudo chown -R <uid>:<uid> ./data
```

---

## Storage Issues

### Data disappears after a restart

**Symptom:**
Documents, clauses and jobs are gone after the server restarts.

**Solution:**
The default storage backend is in-memory. Use SQLite for persistence:
```bash
export AEGISLANG_STORAGE_BACKEND=sqlite
export AEGISLANG_SQLITE_PATH=/var/lib/aegislang/aegislang.db   # default: aegislang_data.db in the working directory
python -m aegislang.api.server
```

Docker Compose already uses SQLite on the `aegislang-data` volume, but the `aegislang-dev` service does not. Any backend value other than `memory` or `sqlite` silently falls back to memory. The startup log then shows the `in_memory_storage_active` warning instead of `sqlite_storage_initialized`.

---

### SQLite database cannot be opened

**Symptom:**
```
sqlite3.OperationalError: unable to open database file
```

**Solution:**
The directory in `AEGISLANG_SQLITE_PATH` must exist and be writable by the server process. SQLite creates the file but not its parent directories:
```bash
mkdir -p /var/lib/aegislang
export AEGISLANG_SQLITE_PATH=/var/lib/aegislang/aegislang.db
```

A relative path such as the default `aegislang_data.db` resolves against the server's working directory.

---

### Job returns 404 after a while

**Symptom:**
`GET /api/v1/jobs/{job_id}` used to work but now returns `{"error": "Job not found", "status_code": 404, ...}`.

**Solution:**
Finished jobs are deleted after `AEGISLANG_JOB_TTL_SECONDS` (default 86400 = 24 hours). Documents, clauses, artifacts and traces do not expire, so read the results from `/documents/{doc_id}`, `/clauses/{doc_id}`, `/rules/{clause_id}` or `/trace/{doc_id}`. With the in-memory backend, every job is also lost on restart.

---

## API Issues

All API errors share one JSON format:
```json
{"error": "<message>", "status_code": 404, "request_id": "<uuid>"}
```
Validation and internal errors add `error_code` and sometimes `details`. Quote the `request_id` (also returned in the `X-Request-ID` header) when searching the logs.

### 404 Not Found on all endpoints

**Symptom:**
```json
{"error": "Not Found", "status_code": 404, "request_id": "..."}
```

**Solution:**
Ensure you're using the correct API path:
```bash
# Correct:
curl http://localhost:8080/api/v1/health

# Incorrect:
curl http://localhost:8080/health
```

---

### 404 for a document, clauses or trace that should exist

**Symptom:**
```json
{"error": "Document not found", "status_code": 404, "request_id": "..."}
```
or `Clauses not found for document` / `Trace not found for document`.

**Solution:**
Ingest and compile run as background jobs. `GET /api/v1/documents/{doc_id}` returns 404 until the ingestion job has completed. `/clauses/{doc_id}`, `/trace/{doc_id}` and `/rules/{clause_id}` return 404 until a compile job for that document has completed. Check the job first:
```bash
curl -H "X-API-Key: your-api-key" http://localhost:8080/api/v1/jobs/ing_xxxxxxxx
```

---

### 401 / 403 authentication errors

**Symptom:**
```json
{"error": "Missing API key. Provide X-API-Key header.", "status_code": 401, "request_id": "..."}
```
or
```json
{"error": "Invalid API key.", "status_code": 403, "request_id": "..."}
```

**Solution:**
1. Send the key in the `X-API-Key` header. Every endpoint except `/api/v1/health` and the docs pages needs it:
```bash
curl -H "X-API-Key: your-api-key" http://localhost:8080/api/v1/documents
```

2. Make sure the key is listed in `AEGISLANG_API_KEYS` (comma-separated). For Docker, set it in `.env` and recreate the container (`docker-compose up -d`).

3. If `AEGISLANG_API_KEYS` is unset, the server generates a random development key once at startup and logs it. Look for the `no_api_keys_configured` warning and its `development_key` field. The key changes on every restart:
```bash
docker-compose logs aegislang | grep no_api_keys_configured
```

4. For local development only, disable auth with `AEGISLANG_DISABLE_AUTH=true`.

---

### 400 Bad Request on upload

**Symptom:**
```json
{"error": "Unsupported file type: .txt. Allowed: {'.pdf', '.docx', '.md', '.markdown', '.html', '.htm'}", "status_code": 400, "request_id": "..."}
```
or
```json
{"error": "metadata must be valid JSON", "status_code": 400, "request_id": "..."}
```

**Solution:**
- Upload a `.pdf`, `.docx`, `.md`, `.markdown`, `.html` or `.htm` file. Only the extension is checked; the content is not inspected.
- The optional `metadata` form field must be a JSON object. A JSON array or other value gives `metadata must be a JSON object`. Single-quote it in the shell:
```bash
curl -X POST http://localhost:8080/api/v1/ingest \
  -H "X-API-Key: your-api-key" \
  -F "file=@document.pdf" \
  -F 'metadata={"document_name": "My Policy"}'
```

---

### 413 File too large

**Symptom:**
```json
{"error": "File too large. Maximum size: 50MB", "status_code": 413, "request_id": "..."}
```

**Solution:**
Raise the limit (in bytes) with `AEGISLANG_MAX_FILE_SIZE`. The whole upload is held in memory while it is checked.

---

### 422 Validation Error

**Symptom:**
```json
{
  "error": "Request validation failed",
  "status_code": 422,
  "error_code": "VALIDATION_ERROR",
  "request_id": "...",
  "details": {"errors": [{"field": "body.file", "message": "Field required", "type": "missing"}]}
}
```

**Solution:**
Check request format. For file upload:
```bash
# Correct:
curl -X POST http://localhost:8080/api/v1/ingest \
  -H "X-API-Key: your-api-key" \
  -F "file=@document.pdf"

# Incorrect (JSON body for file):
curl -X POST http://localhost:8080/api/v1/ingest \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"file": "document.pdf"}'
```

Other common 422 causes:
- An `output_formats` value other than `yaml`, `sql` or `python` (`Input should be 'yaml', 'sql' or 'python'`).
- A schema body that uses `tables_json` instead of `tables`.

---

### Schema not found on compile

**Symptom:**
```json
{"error": "Schema not found: x", "status_code": 404, "request_id": "..."}
```

**Solution:**
`target_schema` must be a built-in schema (`kyc_schema`, `org_schema`, `records_schema`) or one registered with `POST /api/v1/schemas`. `GET /api/v1/schemas` lists only the registered ones. With the in-memory backend, registered schemas are lost on restart.

---

### 429 Too Many Requests

**Symptom:**
```json
{"error": "Rate limit exceeded: 60/minute", "status_code": 429, "request_id": "..."}
```

**Solution:**
Wait for the number of seconds in the `Retry-After` header, or raise `AEGISLANG_RATE_LIMIT_MINUTE` / `AEGISLANG_RATE_LIMIT_HOUR` (defaults 60 and 1000). Limits count per API key. With auth disabled, all clients share one bucket.

---

### 500 Internal Server Error

**Symptom:**
```json
{"error": "...", "status_code": 500, "error_code": "INTERNAL_ERROR", "request_id": "..."}
```
With `AEGISLANG_ENV=production`, the message is generic (e.g. `An internal error occurred`).

**Solution:**
1. Check application logs for the `request_id`:
```bash
docker-compose logs aegislang | grep <request_id>
```

2. Enable debug logging. `docker-compose.yml` sets `LOG_LEVEL=INFO` directly, so change it there (`- LOG_LEVEL=DEBUG`) and recreate the container:
```bash
docker-compose up -d aegislang
```
Locally: `LOG_LEVEL=DEBUG python -m aegislang.api.server` (or `make run-dev`).

3. Check Sentry for detailed error traces. Sentry is active only when `SENTRY_DSN` is set and the server was started with `python -m aegislang.api.server`, as the Docker image does.

---

### CORS errors in browser

**Symptom:**
```
Access to fetch has been blocked by CORS policy
```

**Solution:**
Add your frontend origin to `CORS_ORIGINS`:
```yaml
environment:
  - CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

Browsers' `EventSource` cannot send the `X-API-Key` header. Read `/api/v1/jobs/{job_id}/stream` with a streaming `fetch()` that sets the header instead.

---

## Processing Issues

### Document ingestion fails

**Symptom:**
`GET /api/v1/jobs/{job_id}` returns `"status": "failed"` with the parser's exception in `"error"`.

**Solutions:**

1. **PDF issues**: Ensure the PDF is not encrypted:
```bash
# Check if PDF requires password
pdfinfo document.pdf
```

2. **Scanned documents**: Text is extracted with pdfminer, and there is no OCR. Image-only PDFs produce no text.

3. **Corrupt files**: Validate file integrity before upload. The extension is checked at upload time, but content problems show up only when the job fails.

4. **Chunk sizes**: To tune chunking in library code, pass a `ChunkingConfig`:
```python
from aegislang.agents.aegis_ingestor import AegisIngestor, ChunkingConfig

ingestor = AegisIngestor(
    chunking_config=ChunkingConfig(target_tokens=512, min_tokens=128, max_tokens=768)
)
```
In the CLI, use `--target-tokens`, `--min-tokens` and `--max-tokens`. The API always uses the defaults (768/256/1024).

---

### No clauses extracted

**Symptom:**
The compile job completes with `"clauses_parsed": 0` (or far fewer clauses than expected).

**Solutions:**

1. Check that the document content is text-based (not scanned images) and that ingestion produced sections with text (`section_count` in `GET /api/v1/documents/{doc_id}`).

2. Check which parser the server uses:
```bash
curl http://localhost:8080/api/v1/health
# "llm_provider": "anthropic" | "openai" | "mock"
```

3. If `llm_provider` is `anthropic` or `openai`, look in the logs for `clause_parse_failed` / `skipping_unparseable_clause`. A clause the LLM call fails on is logged and skipped instead of failing the job. An invalid or placeholder key (for example the `sk-ant-xxx...` value from `.env.example`), an unavailable model or no network access therefore yields 0 clauses. Fix the key, or unset it to fall back to the mock parser.

---

### Entity mapping fails

**Symptom:**
Clauses come back with `"mapping_status": "needs_review"` (or `"partial"`) and entries like:
```json
"unmapped_entities": [{"entity": "customer identity", "reason": "No match found above threshold (0.7)", "suggested_matches": [...]}]
```

**Solutions:**

1. Register a schema whose field names or `semantic_labels` match the entity text, and compile against it:
```bash
curl -X POST http://localhost:8080/api/v1/schemas \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"schema_id": "my_schema", "schema_type": "sql", "tables": [{"table_name": "customers", "fields": [{"field_name": "customer_id", "field_type": "UUID", "semantic_labels": ["customer", "users"]}]}]}'

curl -X POST http://localhost:8080/api/v1/compile \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"doc_id": "POLICY_66C62B", "target_schema": "my_schema"}'
```

2. Use real embeddings. The API and CLI always use mock embeddings, which carry no meaning, so only exact, synonym and override matches succeed. Lowering the threshold there (`--threshold` in the CLI) only produces arbitrary matches. In library code, pass a real provider and, if needed, a lower threshold:
```python
from aegislang.agents.schema_mapping_agent import (
    SchemaMappingAgent, SentenceTransformerProvider, create_default_registry,
)

mapper = SchemaMappingAgent(
    registry=create_default_registry(),
    embedding_provider=SentenceTransformerProvider(),  # needs requirements-ml.txt
    confidence_threshold=0.6,
)
```

3. Add manual overrides for specific entities (library code only):
```python
mapper.add_manual_override("Banks", "org_schema:institution.institution_id")
```

---

### Template rendering errors

**Symptom:**
This only happens with custom templates (`--templates` / `CompilerAgent(templates_dir=...)`). An artifact is missing, and the log shows:
```
compilation_failed  clause_id=...  error="'actor' is undefined"  format=yaml
```

**Solution:**
Rendering errors are logged and that artifact is skipped. Verify template syntax and variable names:
```jinja2
{# Correct #}
{{ clause.actor.entity }}

{# Incorrect #}
{{ actor.entity }}
```

Also check the file name. Only `<format>/<clause_type>.<ext>.j2` or `<format>/default.<ext>.j2` is ever selected, and a YAML `default` template never is.

---

## Performance Issues

### Slow document processing

**Symptom:**
Compile jobs take minutes.

**Solutions:**

1. **Expect one LLM request per clause**: With Anthropic or OpenAI, clauses are parsed one at a time and there is no concurrency setting. Chunk size does not change the number of requests. Split very large documents, or iterate with the mock parser (unset the LLM keys) and switch to the LLM for the final run.

2. **Don't block on the request**: `POST /ingest` and `POST /compile` already return immediately with a job ID. There is no `?async=true` option. Poll `GET /api/v1/jobs/{job_id}` or follow `/stream`.

3. **Workers**: `WORKERS` only helps with concurrent requests, and each job still runs in a single process. With the default in-memory backend the server forces `WORKERS=1`. Use `AEGISLANG_STORAGE_BACKEND=sqlite` before raising it. Rate limits then apply per worker.
```yaml
environment:
  - AEGISLANG_STORAGE_BACKEND=sqlite
  - WORKERS=2
```

---

### High memory usage

**Symptom:**
Container OOM killed or system slowdown.

**Solutions:**

1. **Use the SQLite backend**: The in-memory backend keeps every document, clause, artifact and trace in RAM for the life of the process. Only jobs expire.

2. **Limit upload size**: Uploads are read fully into memory. Lower `AEGISLANG_MAX_FILE_SIZE` if needed.

3. **Increase container memory**:
```yaml
deploy:
  resources:
    limits:
      memory: 4G
```

4. **Use CPU-only torch** (smaller footprint). This only applies if you installed `requirements-ml.txt`:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

### API responses slow as data grows

**Symptom:**
`GET /api/v1/documents` or `GET /api/v1/rules/{clause_id}` gets slower over time.

**Solutions:**

1. Both endpoints read through all stored data:
   - `/documents` loads every stored document, sections included.
   - `/rules/{clause_id}` scans the artifacts of every document.
   Data never expires and there are no DELETE endpoints.

2. To start fresh, stop the server and move the SQLite file aside, keeping it as a backup. With the in-memory backend, restart the server.

---

## LLM/AI Issues

### API key invalid

**Symptom:**
`clause_parse_failed` errors in the logs mentioning an authentication error (HTTP 401), and the compile job completes with `"clauses_parsed": 0`.

**Solution:**
1. Verify key is set:
```bash
echo $ANTHROPIC_API_KEY
```
For Docker, check `.env` and `docker-compose config`.

2. Check key format (should start with `sk-ant-` for Anthropic). Make sure it is not the placeholder from `.env.example`.

3. Regenerate key in provider dashboard.

4. The API prefers Anthropic whenever `ANTHROPIC_API_KEY` is set. To use OpenAI, unset it and set only `OPENAI_API_KEY`. `/api/v1/health` shows which provider is active.

---

### Rate limit exceeded

**Symptom:**
`clause_parse_failed` errors in the logs mentioning a rate limit (HTTP 429) from the LLM provider.

**Solutions:**

1. **No retry logic in AegisLang**: Apart from the provider SDK's own defaults, AegisLang does not retry. Clauses that still fail are skipped, so re-run the compile job once the limit resets.

2. **Requests are already sequential**: One clause at a time, with no concurrency setting to lower. Split large documents or raise your provider rate limit.

3. **Use the mock parser** while iterating: unset the LLM keys.

---

### Model not available

**Symptom:**
`clause_parse_failed` errors in the logs mentioning a model-not-found error (HTTP 404).

**Solution:**
The default models are `claude-sonnet-4-20250514` (Anthropic) and `gpt-4-turbo-preview` (OpenAI). The API server has no model setting and always uses these defaults. Choose a model in the CLI or in Python:
```bash
python -m aegislang.agents.policy_parser_agent ingested.json --provider anthropic --model <model-id>
```
```python
from aegislang.agents.policy_parser_agent import PolicyParserAgent

parser = PolicyParserAgent(llm_provider="anthropic", model="<model-id>")
```

Newer Claude models (Claude Opus 4.7 and later, Claude Sonnet 5 and later) reject sampling parameters with HTTP 400. The Anthropic client sends `temperature` (default 0.1) in the request body, so when you choose one of those models also turn it off:
```python
parser = PolicyParserAgent(llm_provider="anthropic", model="claude-opus-5-5")
parser.llm_client.temperature = None  # omit temperature from the request
```

---

### Inconsistent results

**Symptom:**
Same document produces different clauses on re-processing.

**Solutions:**

1. **Lower the temperature**: LLM calls use temperature 0.1 by default. In library code you can set it to 0 (only on models that accept sampling parameters; see [Model not available](#model-not-available)). The mock parser is fully deterministic.
```python
parser = PolicyParserAgent(llm_provider="anthropic")
parser.llm_client.temperature = 0.0
```

2. **Use the content hash** to detect duplicates:
```python
if doc.metadata.hash == cached_hash:
    return cached_result
```
`AegisIngestor` doc IDs come from the file name plus the content hash, so they are stable. The API instead adds a random suffix to each upload's `doc_id`, so compare `metadata.hash`.

---

## Getting Help

### Still stuck?

1. **Check logs** with debug enabled:
```bash
LOG_LEVEL=DEBUG python -m aegislang.api.server
# Docker: set LOG_LEVEL=DEBUG in docker-compose.yml, then
docker-compose up aegislang
```

2. **Search issues**: https://github.com/kase1111-hash/AegisLang/issues

3. **Open new issue** with:
   - AegisLang version (`cat VERSION` or the `version` field of `/api/v1/health`)
   - Full error message and `request_id`
   - Steps to reproduce
   - Relevant logs (sanitized)

---

*Last updated: October 2026 | Version: 0.1.0*
