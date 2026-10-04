# AegisLang FAQ

Frequently Asked Questions about AegisLang.

---

## General

### What is AegisLang?

AegisLang is a natural language programming platform that transforms policy documents (regulations, SOPs, governance rules) into executable compliance artifacts. It bridges the gap between human-written rules and machine enforcement. It is alpha software (version 0.1.0), and its generated artifacts are drafts for human review.

### What problem does AegisLang solve?

- **Manual compliance coding**: Generates draft YAML rules, SQL constraints and pytest stubs instead of hand-coding them
- **Traceability gaps**: Keeps lineage from policy text (document → section → chunk → clause, plus the mapping ID) to each artifact
- **Update lag**: When a policy changes, re-run the pipeline on the new version to regenerate its artifacts
- **Audit complexity**: Stores validation results and a provenance graph per document (`GET /api/v1/trace/{doc_id}`)

### What document formats are supported?

- PDF documents (text-based; scanned PDFs are not OCR'd)
- Microsoft Word (.docx)
- Markdown (.md, .markdown)
- HTML files (.html, .htm)

### What output formats can AegisLang generate?

| Format | Use Case | Status |
|--------|----------|--------|
| YAML | Compliance rule definitions (`control:` documents) | Implemented (API, CLI, library) |
| SQL | PostgreSQL CHECK constraints and triggers | Implemented (API, CLI, library) |
| Python | pytest test stubs | Implemented (API, CLI, library) |
| Terraform | Placeholder Sentinel-style policy resource | Experimental (CLI and library only) |
| Rego | Simple OPA policy render | Experimental (CLI and library only) |
| JSON | Simple JSON rule render | Experimental (CLI and library only) |

The REST API accepts only `yaml`, `sql` and `python` in `output_formats` and returns 422 for anything else.

---

## Installation & Setup

### What are the system requirements?

- Python 3.11 or higher
- Docker and Docker Compose (for containerized deployment)
- No ML models are needed for the core install (`requirements.txt`). The optional `requirements-ml.txt` pulls in sentence-transformers and torch, a large download that needs more RAM. You only need it if you pass `SentenceTransformerProvider` to the mapper in library code.

### How do I install AegisLang?

**Option 1: Docker (Recommended)**
```bash
git clone https://github.com/kase1111-hash/AegisLang.git
cd AegisLang
cp .env.example .env
# Edit .env: set AEGISLANG_API_KEYS=<your-key> (or AEGISLANG_DISABLE_AUTH=true for local use)
# and replace or delete the placeholder ANTHROPIC_API_KEY / OPENAI_API_KEY lines
docker-compose up -d
curl http://localhost:8080/api/v1/health
```

The Compose service stores its data in SQLite on the `aegislang-data` volume. Do not leave the placeholder LLM keys from `.env.example` in place. If you do, the server selects that provider and every clause fails to parse.

**Option 2: Local Python**
```bash
git clone https://github.com/kase1111-hash/AegisLang.git
cd AegisLang
pip install -r requirements.txt
export AEGISLANG_API_KEYS=my-local-key     # or: export AEGISLANG_DISABLE_AUTH=true (development only)
python -m aegislang.api.server             # same as: make run
```

If you set neither variable, the server generates a random development key once at startup. It logs the key as the `development_key` field of the `no_api_keys_configured` warning. The key changes on every restart.

### How do I configure API keys?

The Python code does not read `.env`. Only Docker Compose does, substituting its values into `docker-compose.yml`. For Docker, put the keys in `.env` next to `docker-compose.yml`:
```bash
AEGISLANG_API_KEYS=your-client-key       # keys clients send in the X-API-Key header
ANTHROPIC_API_KEY=your-anthropic-key     # optional
OPENAI_API_KEY=your-openai-key           # optional
```

When running locally, export them in your shell instead:
```bash
export AEGISLANG_API_KEYS=your-client-key
export ANTHROPIC_API_KEY=your-key
```

The API picks the clause-parsing provider from the keys that are set:
- `ANTHROPIC_API_KEY` set: Anthropic, preferred when both keys are set.
- Only `OPENAI_API_KEY` set: OpenAI.
- Neither: the mock parser.

`GET /api/v1/health` reports the choice in `llm_provider`.

### Which LLM providers are supported?

- Anthropic Claude (default model `claude-sonnet-4-20250514`)
- OpenAI (default model `gpt-4-turbo-preview`)
- Mock client: keyword-based parsing with no API calls. It is used automatically when no LLM key is set.

The API always uses the provider's default model and has no model setting. To choose a model, use the parser CLI (`--model`) or `PolicyParserAgent(model=...)` in Python.

---

## Usage

### How do I process a policy document?

**Via API:**
```bash
# 1. Upload. Returns {"job_id": "ing_xxxxxxxx", "doc_id": "POLICY_66C62B", ...}
curl -X POST http://localhost:8080/api/v1/ingest \
  -H "X-API-Key: your-api-key" \
  -F "file=@policy.pdf" \
  -F 'metadata={"document_name": "My Policy", "jurisdiction": "US"}'

# 2. Wait until the ingestion job is "completed"
curl -H "X-API-Key: your-api-key" http://localhost:8080/api/v1/jobs/ing_xxxxxxxx

# 3. Compile (parse, map, compile, validate). Returns {"job_id": "cmp_xxxxxxxx", ...}
curl -X POST http://localhost:8080/api/v1/compile \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"doc_id": "POLICY_66C62B"}'

# 4. After the compile job completes, read the results
curl -H "X-API-Key: your-api-key" http://localhost:8080/api/v1/clauses/POLICY_66C62B
curl -H "X-API-Key: your-api-key" http://localhost:8080/api/v1/rules/<clause_id>
curl -H "X-API-Key: your-api-key" http://localhost:8080/api/v1/trace/POLICY_66C62B
```

Ingestion does not extract clauses. Clauses, artifacts and traces exist only after a compile job has finished.

**Via Python:**
```python
from aegislang.agents.aegis_ingestor import AegisIngestor

ingestor = AegisIngestor()
doc = ingestor.ingest("policy.pdf")
print(doc.doc_id, len(doc.sections), doc.metadata.hash)
```

The full pipeline in-process (the agents exchange plain dicts):
```python
from aegislang.agents.aegis_ingestor import AegisIngestor
from aegislang.agents.policy_parser_agent import PolicyParserAgent
from aegislang.agents.schema_mapping_agent import SchemaMappingAgent, create_default_registry
from aegislang.agents.compiler_agent import CompilerAgent, ArtifactFormat
from aegislang.agents.trace_validator_agent import TraceValidatorAgent

doc = AegisIngestor().ingest("policy.pdf").model_dump(mode="json")
parsed = PolicyParserAgent(use_mock=True).parse_ingested_document(doc).model_dump(mode="json")
mapped = SchemaMappingAgent(registry=create_default_registry()).map_parsed_collection(parsed).model_dump(mode="json")
compiled = CompilerAgent().compile_mapped_collection(mapped, [ArtifactFormat.YAML, ArtifactFormat.SQL])
results = TraceValidatorAgent().validate_compiled_collection(
    compiled.model_dump(mode="json"), mapped, parsed
)
print(results.summary)
```

Drop `use_mock=True` to parse with Anthropic (needs `ANTHROPIC_API_KEY`), or pass `llm_provider="openai"`. Each layer also has a CLI (`python -m aegislang.agents.<agent> --help`).

### How do I compile clauses to specific formats?

```bash
curl -X POST http://localhost:8080/api/v1/compile \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"doc_id": "POLICY_66C62B", "output_formats": ["yaml", "sql", "python"]}'
```

`output_formats` defaults to `["yaml", "sql"]`. The compiler CLI also takes the experimental formats: `python -m aegislang.agents.compiler_agent mapped.json --formats yaml sql python terraform rego json`.

### What clause types does AegisLang recognize?

| Type | Description | Example |
|------|-------------|---------|
| `obligation` | Required action | "Banks must verify customer identity" |
| `prohibition` | Forbidden action | "Employees shall not share passwords" |
| `permission` | Allowed action | "Users may request data deletion" |
| `conditional` | Triggered action | "If transaction > $10k, report to FinCEN" |
| `definition` | Term definition | "PII means personally identifiable information" |
| `exception` | Rule exception | "Except for internal transfers under $1000" |

Without an LLM key, the mock parser picks the type from keywords such as must/shall, must not, may and means. A sentence with no indicator defaults to `obligation`.

### How do I register a custom schema?

```bash
curl -X POST http://localhost:8080/api/v1/schemas \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "schema_id": "my_crm_schema",
    "schema_type": "sql",
    "tables": [
      {
        "table_name": "customers",
        "fields": [
          {"field_name": "customer_id", "field_type": "UUID", "semantic_labels": ["customer", "users"]},
          {"field_name": "kyc_verified", "field_type": "BOOLEAN", "semantic_labels": ["identity", "kyc"]}
        ]
      }
    ]
  }'
```

Then compile against it with `"target_schema": "my_crm_schema"` in the `POST /api/v1/compile` body. An unknown `target_schema` returns 404 (`Schema not found: ...`). Without `target_schema`, the mapper searches the registered schemas together with the built-in ones (`kyc_schema`, `org_schema`, `records_schema`). Posting the same `schema_id` again replaces the schema. `version` is only a label.

---

## Architecture

### What is the L1-L5 pipeline?

AegisLang processes documents through 5 layers:

1. **L1 (Ingestion)**: Parse documents, extract text, chunk content
2. **L2 (Parsing)**: Split chunks into clauses, classify types, extract semantics (LLM or mock parser)
3. **L3 (Mapping)**: Match entities to target schema fields (overrides, exact/synonym matching, embeddings)
4. **L4 (Compilation)**: Generate output artifacts from templates
5. **L5 (Validation)**: Verify integrity, calculate confidence, build the provenance graph (audit trail)

### How does entity mapping work?

The L3 layer maps policy entities (e.g., "customer") to schema fields (e.g., `customer.customer_id`). For each entity it tries, in order:

1. Manual overrides, added in library code with `SchemaMappingAgent.add_manual_override(entity, "schema_id:table.field")`
2. Exact match on a field name or one of its `semantic_labels` (case-insensitive)
3. Synonyms from the registry (e.g., client → customer)
4. Cosine similarity of embeddings, accepted at or above the threshold (default 0.7)

The mapper never calls an LLM. If no embedding provider is passed in, it uses mock embeddings: deterministic SHA-256 pseudo-vectors that carry no meaning. The REST API and the CLI always use them, and the API threshold is fixed at 0.7. In practice, only overrides, exact matches and synonyms succeed there. For real semantic matching, pass `SentenceTransformerProvider` (needs `requirements-ml.txt`) or `OpenAIEmbeddingProvider` as `embedding_provider` in library code.

### What storage does AegisLang use?

| Backend | Purpose | Status |
|---------|---------|--------|
| In-memory | Default storage (data lost on restart; server forced to 1 worker) | Default |
| SQLite | Persistent local storage (`AEGISLANG_SQLITE_PATH`, default `aegislang_data.db`) | Available (`AEGISLANG_STORAGE_BACKEND=sqlite`); used by Docker Compose |

The API server stores everything in one of these two backends: jobs, documents, clauses, artifacts, schemas and traces. PostgreSQL, Redis and Neo4j are not used. The CLI tools and the Python library keep no store; they read and write JSON and artifact files.

---

## API

### What are the main API endpoints?

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/ingest` | POST | Upload a document (starts an ingestion job) |
| `/api/v1/documents` | GET | List all documents |
| `/api/v1/documents/{doc_id}` | GET | Document metadata |
| `/api/v1/compile` | POST | Parse, map, compile and validate a document (starts a compile job) |
| `/api/v1/jobs/{job_id}` | GET | Job status |
| `/api/v1/jobs/{job_id}/stream` | GET | Job status as Server-Sent Events |
| `/api/v1/clauses/{doc_id}` | GET | Parsed clauses for a document |
| `/api/v1/rules/{clause_id}` | GET | All compiled artifacts for a clause (one per format) |
| `/api/v1/trace/{doc_id}` | GET | Validation results and provenance graph |
| `/api/v1/schemas` | GET / POST | List or register schemas |
| `/api/v1/schemas/{schema_id}` | GET | Get a registered schema |
| `/api/v1/health` | GET | Health check (no API key needed) |

Interactive docs are at `/api/docs`.

### How do I check job status for async operations?

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8080/api/v1/jobs/{job_id}
```

Response:
```json
{
  "job_id": "cmp_1a2b3c4d",
  "status": "completed",
  "created_at": "2026-10-04T12:00:00.000000+00:00",
  "completed_at": "2026-10-04T12:00:01.000000+00:00",
  "result": {...},
  "error": null
}
```

Ingestion job IDs look like `ing_xxxxxxxx` and compile job IDs like `cmp_xxxxxxxx`. `status` is one of `pending`, `processing`, `completed` or `failed`. Finished jobs are deleted after `AEGISLANG_JOB_TTL_SECONDS` (default 24 hours). To get updates pushed instead of polling, use `GET /api/v1/jobs/{job_id}/stream`, which also needs the `X-API-Key` header.

### Is there rate limiting?

Yes. The API includes a built-in in-memory rate limiter. The defaults are 60 requests/minute and 1000 requests/hour per API key, counted separately in each server process. When auth is disabled, all clients share one bucket. Configure the limits with the `AEGISLANG_RATE_LIMIT_MINUTE` and `AEGISLANG_RATE_LIMIT_HOUR` environment variables. Over the limit, the API returns 429 with a `Retry-After` header. `/api/v1/health` is exempt. For production, you may also add rate limiting at your reverse proxy.

---

## Templates

### How do I create custom templates?

1. Create a Jinja2 template file in a templates directory, named `<format>/<clause_type>.<ext>.j2` or `<format>/default.<ext>.j2`:
```jinja2
{# my_templates/yaml/permission.yaml.j2 #}
# Custom rule for {{ clause.clause_id }}
rule:
  type: {{ clause.type }}
  actor: {{ clause.actor.entity }}
```

2. Pass the directory explicitly, because custom templates are not loaded automatically:
   - CLI: `python -m aegislang.agents.compiler_agent mapped.json --templates my_templates`
   - Python: `CompilerAgent(templates_dir=Path("my_templates"))`

The REST API always uses the built-in templates, and the repository's `templates/` folder is not loaded unless you point at it.

A file named after a clause type (`obligation`, `prohibition`, `permission`, `conditional`, `definition`, `exception`) is used for that type. `default.<ext>.j2` replaces the format's fallback template. Built-in YAML has a template for every clause type, so a `default.yaml.j2` is never selected. Name YAML templates after a clause type instead. Templates render in a sandboxed Jinja2 environment.

### What variables are available in templates?

| Variable | Description |
|----------|-------------|
| `clause` | Full parsed clause object |
| `clause.clause_id` | Unique clause identifier |
| `clause.type` | obligation, prohibition, etc. |
| `clause.actor` | Actor entity with qualifiers |
| `clause.action` | Action verb with modifiers |
| `clause.object` | Target object (optional) |
| `clause.condition` | Trigger condition (optional) |
| `clause.temporal_scope` | Deadline, frequency, duration (optional) |
| `clause.source_text` | Original clause text |
| `mappings` | `actor_path` and `object_path` from entity mapping (or `None`) |
| `doc_id` | Source document ID |
| `confidence` | Average mapping confidence (0.5 if nothing mapped) |
| `severity` | critical (prohibition), high (obligation), medium (others) |
| `timestamp` | Generation timestamp |
| `version` | AegisLang version |
| `table_name`, `check_condition` | SQL helpers (table from the mapped path, else `compliance_table`) |
| `resource_type` | Terraform helper |

Extra filters: `truncate`, `sqlescape` and `sqlsafe`.

---

## Deployment

### How do I deploy to production?

1. Build the production image:
```bash
docker-compose build aegislang
```

2. Configure the environment:
   - Set `AEGISLANG_API_KEYS`, an LLM key if you want real parsing, and optionally `SENTRY_DSN` in `.env`.
   - Add `AEGISLANG_ENV=production` to the service's `environment:` in `docker-compose.yml`. Compose does not pass it through by default. It enables JSON console logs and generic error messages.

3. Start the service:
```bash
docker-compose up -d aegislang
```

4. Put a reverse proxy in front for TLS, because the app serves plain HTTP. Data lives in SQLite on the `aegislang-data` volume.

### How do I scale horizontally?

You can't. AegisLang runs as a single instance:
- Rate limiting is in each process's memory.
- Jobs run inside the process that accepted them.
- SQLite is a local file.

Do not run replicas behind a load balancer. On a single host with the SQLite backend you can raise `WORKERS`, but the rate limits then apply per worker. With the in-memory backend the server forces `WORKERS=1`.

### How do I backup the database?

With the SQLite backend, back up the database file. To take a consistent copy while the Docker service is running:

```bash
docker-compose exec aegislang python -c "import sqlite3; sqlite3.connect('/app/data/aegislang.db').backup(sqlite3.connect('/app/data/backup.db'))"
docker cp "$(docker-compose ps -q aegislang)":/app/data/backup.db ./aegislang-backup.db
```

For a local server, stop it and copy the file at `AEGISLANG_SQLITE_PATH`. The in-memory backend has nothing to back up, because its data is gone after a restart.

---

## Troubleshooting

### Where can I find logs?

- Docker: `docker-compose logs aegislang`
- Local: stdout of `python -m aegislang.api.server`. Logs are colored text, or JSON when `AEGISLANG_ENV=production`. There is no log directory by default.
- Set `AEGISLANG_LOG_FILE=/path/to/file.log` to also write JSON logs to a file.
- Log entries carry the request ID that is returned in the `X-Request-ID` header.
- The CLI tools log to stderr.

Logging setup runs only when the server is started with `python -m aegislang.api.server` (or `make run`). Starting it with plain `uvicorn aegislang.api.server:app` skips the log level, log file and Sentry settings.

### How do I enable debug mode?

```bash
export LOG_LEVEL=DEBUG          # AEGISLANG_LOG_LEVEL takes precedence if both are set
python -m aegislang.api.server  # or: make run-dev (reload + DEBUG)
```

`docker-compose.yml` sets `LOG_LEVEL=INFO` directly, so a shell variable does not override it. Edit the file instead:
```yaml
environment:
  - LOG_LEVEL=DEBUG
```

### Common issues

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions.

---

## Integration

### How do I integrate with CI/CD?

The simplest route is the per-agent CLIs, which need no server:

```yaml
# .github/workflows/compliance.yml
- name: Compile compliance rules
  run: |
    pip install -r requirements.txt
    python -m aegislang.agents.aegis_ingestor policies/policy.md -o ingested.json
    python -m aegislang.agents.policy_parser_agent ingested.json --provider mock -o parsed.json
    python -m aegislang.agents.schema_mapping_agent parsed.json -o mapped.json
    python -m aegislang.agents.compiler_agent mapped.json -o artifacts --formats yaml
```

Use `--provider anthropic` with an `ANTHROPIC_API_KEY` secret for LLM parsing. Against a running server, use double quotes so the shell expands `$DOC_ID`:

```bash
curl -X POST http://localhost:8080/api/v1/compile \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $AEGISLANG_API_KEY" \
  -d "{\"doc_id\": \"$DOC_ID\", \"output_formats\": [\"yaml\"]}"
```

Then poll the job, list the clauses with `GET /api/v1/clauses/{doc_id}`, and fetch each clause's artifacts with `GET /api/v1/rules/{clause_id}`.

### Does AegisLang support webhooks?

No. There are no webhooks and no event bus. Poll `GET /api/v1/jobs/{job_id}` or follow `GET /api/v1/jobs/{job_id}/stream` (Server-Sent Events) to learn when a job finishes.

### How do I integrate with Slack/Teams?

There is no built-in integration. Have your own script poll the job endpoint and post the result to Slack or Teams.

---

## Security

### Is my data secure?

- Data stays on the host running AegisLang (in memory or in the SQLite file).
- No data is sent to external services, except clause text sent to Anthropic or OpenAI when an LLM key is configured, and error reports sent to Sentry when `SENTRY_DSN` is set.
- Uploaded files are written to a private temp directory, then overwritten and deleted after ingestion.
- Data is not encrypted at rest, and the app has no TLS. Use a reverse proxy for TLS and protect the database file.
- There is no local-model or air-gapped LLM option. Offline, only the mock parser is available.

### Are LLM API calls logged?

Clause text and LLM responses are not logged. Logs contain IDs, counts, clause types and confidence scores, and error messages when a call fails. `LOG_LEVEL=DEBUG` adds per-clause entries (IDs and text length). Use it only in development anyway.

### How do I run without external LLM APIs?

Leave `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` unset. The API then uses the keyword-based mock parser, and `/api/v1/health` shows `"llm_provider": "mock"`. In Python, pass `PolicyParserAgent(use_mock=True)`. In the CLI, pass `--provider mock`. There is no built-in client for local or on-prem LLMs.

---

## Contributing

### How do I contribute?

1. Fork the repository
2. Create a feature branch
3. Install dev dependencies and run tests: `make dev-install && make test`
4. Submit a pull request

### Where do I report bugs?

Open an issue at: https://github.com/kase1111-hash/AegisLang/issues

---

*Last updated: October 2026 | Version: 0.1.0*
