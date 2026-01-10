# AegisLang FAQ

Frequently Asked Questions about AegisLang.

---

## General

### What is AegisLang?

AegisLang is a natural language programming platform that transforms policy documents (regulations, SOPs, governance rules) into executable compliance artifacts. It bridges the gap between human-written rules and machine enforcement.

### What problem does AegisLang solve?

- **Manual compliance coding**: Eliminates hand-coding of compliance rules
- **Traceability gaps**: Maintains full lineage from policy text to code
- **Update lag**: Automates rule updates when policies change
- **Audit complexity**: Provides clear audit trails for regulators

### What document formats are supported?

- PDF documents
- Microsoft Word (.docx)
- Markdown (.md)
- HTML files

### What output formats can AegisLang generate?

| Format | Use Case |
|--------|----------|
| YAML | CI/CD compliance rules, Kubernetes policies |
| SQL | Database CHECK constraints, triggers |
| Python | pytest test stubs, validator classes |
| Terraform | Sentinel policies for infrastructure |
| Rego | OPA policies for authorization |
| JSON | Custom integrations, APIs |

---

## Installation & Setup

### What are the system requirements?

- Python 3.11 or higher
- Docker and Docker Compose (for containerized deployment)
- 4GB RAM minimum (8GB recommended for ML models)
- 10GB disk space

### How do I install AegisLang?

**Option 1: Docker (Recommended)**
```bash
git clone https://github.com/kase1111-hash/AegisLang.git
cd AegisLang
docker-compose up -d
```

**Option 2: Local Python**
```bash
git clone https://github.com/kase1111-hash/AegisLang.git
cd AegisLang
pip install -r requirements.txt
python -m aegislang.api.server
```

### How do I configure API keys?

Create a `.env` file in the project root:
```bash
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key
```

Or set environment variables directly:
```bash
export ANTHROPIC_API_KEY=your-key
```

### Which LLM providers are supported?

- Anthropic Claude (recommended)
- OpenAI GPT-4
- Mock client (for testing without API calls)

---

## Usage

### How do I process a policy document?

**Via API:**
```bash
curl -X POST http://localhost:8080/api/v1/ingest \
  -F "file=@policy.pdf" \
  -F "document_type=regulation"
```

**Via Python:**
```python
from aegislang.agents import AegisIngestor

ingestor = AegisIngestor()
doc = ingestor.ingest_file("policy.pdf")
```

### How do I compile clauses to specific formats?

```bash
curl -X POST http://localhost:8080/api/v1/compile \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "DOC-001", "formats": ["yaml", "sql", "python"]}'
```

### What clause types does AegisLang recognize?

| Type | Description | Example |
|------|-------------|---------|
| `obligation` | Required action | "Banks must verify customer identity" |
| `prohibition` | Forbidden action | "Employees shall not share passwords" |
| `permission` | Allowed action | "Users may request data deletion" |
| `conditional` | Triggered action | "If transaction > $10k, report to FinCEN" |
| `definition` | Term definition | "PII means personally identifiable information" |
| `exception` | Rule exception | "Except for internal transfers under $1000" |

### How do I register a custom schema?

```bash
curl -X POST http://localhost:8080/api/v1/schemas \
  -H "Content-Type: application/json" \
  -d '{
    "schema_id": "my_crm_schema",
    "schema_type": "sql",
    "tables_json": [
      {
        "table_name": "customers",
        "fields": [
          {"field_name": "id", "field_type": "UUID"},
          {"field_name": "kyc_verified", "field_type": "BOOLEAN"}
        ]
      }
    ]
  }'
```

---

## Architecture

### What is the L1-L5 pipeline?

AegisLang processes documents through 5 layers:

1. **L1 (Ingestion)**: Parse documents, extract text, chunk content
2. **L2 (Parsing)**: Identify clauses, classify types, extract semantics
3. **L3 (Mapping)**: Match entities to target schema using embeddings
4. **L4 (Compilation)**: Generate output artifacts from templates
5. **L5 (Validation)**: Verify integrity, calculate confidence, create audit trail

### How does entity mapping work?

The L3 layer uses semantic embeddings to match policy entities (e.g., "customer") to schema entities (e.g., `users.customer_id`). It:

1. Generates embeddings for both source and target entities
2. Calculates cosine similarity
3. Applies configurable threshold (default: 0.7)
4. Allows manual overrides for edge cases

### What databases does AegisLang use?

| Database | Purpose |
|----------|---------|
| PostgreSQL | Document metadata, clauses, artifacts, validation results |
| Redis | Event bus, caching, async job queue |
| Neo4j | Provenance graphs, lineage visualization (optional) |

---

## API

### What are the main API endpoints?

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/ingest` | POST | Upload and process document |
| `/api/v1/documents` | GET | List all documents |
| `/api/v1/clauses` | GET | List extracted clauses |
| `/api/v1/compile` | POST | Compile clauses to artifacts |
| `/api/v1/rules/{id}` | GET | Get compiled rules for clause |
| `/api/v1/health` | GET | Health check |

### How do I check job status for async operations?

```bash
curl http://localhost:8080/api/v1/jobs/{job_id}
```

Response:
```json
{
  "job_id": "JOB-123",
  "status": "completed",
  "result": {...}
}
```

### Is there rate limiting?

By default, no rate limiting is applied. For production, configure your load balancer or add middleware.

---

## Templates

### How do I create custom templates?

1. Create a Jinja2 template file in `templates/{format}/`:
```jinja2
{# templates/yaml/custom.yaml.j2 #}
# Custom rule for {{ clause.clause_id }}
rule:
  type: {{ clause.type }}
  actor: {{ clause.actor.entity }}
```

2. The template is automatically loaded on startup.

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
| `mappings` | Entity mapping results |
| `confidence` | Overall confidence score |
| `timestamp` | Generation timestamp |

---

## Deployment

### How do I deploy to production?

1. Build the production image:
```bash
docker-compose build aegislang
```

2. Configure environment variables for production databases

3. Run with appropriate workers:
```bash
docker-compose up -d aegislang
```

### How do I scale horizontally?

AegisLang is stateless and can be scaled behind a load balancer:

```yaml
# docker-compose.override.yml
services:
  aegislang:
    deploy:
      replicas: 3
```

### How do I backup the database?

```bash
docker-compose exec postgres pg_dump -U aegislang aegislang > backup.sql
```

---

## Troubleshooting

### Where can I find logs?

- Docker: `docker-compose logs aegislang`
- Local: Check `./logs/` directory
- Structured JSON logs when `LOG_LEVEL=INFO`

### How do I enable debug mode?

```bash
export LOG_LEVEL=DEBUG
# or in docker-compose.yml
environment:
  - LOG_LEVEL=DEBUG
```

### Common issues

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions.

---

## Integration

### How do I integrate with CI/CD?

Use the generated YAML artifacts in your pipeline:

```yaml
# .github/workflows/compliance.yml
- name: Check compliance rules
  run: |
    aegislang compile --format yaml --output rules/
    # Use rules in subsequent validation steps
```

### Does AegisLang support webhooks?

Events are published to Redis. Subscribe to:
- `policy.ingested`
- `policy.parsed`
- `policy.compiled`
- `policy.validated`

### How do I integrate with Slack/Teams?

Subscribe to Redis events and forward to your notification service.

---

## Security

### Is my data secure?

- All data stays within your infrastructure
- No data is sent to external services (except LLM API calls)
- Supports air-gapped deployment with local models

### Are LLM API calls logged?

No sensitive content is logged by default. Enable `LOG_LEVEL=DEBUG` only in development.

### How do I run without external LLM APIs?

Use the mock client for testing, or integrate a local LLM.

---

## Contributing

### How do I contribute?

1. Fork the repository
2. Create a feature branch
3. Run tests: `make test`
4. Submit a pull request

### Where do I report bugs?

Open an issue at: https://github.com/kase1111-hash/AegisLang/issues

---

*Last updated: 2024-01-10 | Version: 1.0.0*
