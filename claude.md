# AegisLang

AegisLang is a natural language policy compiler for compliance automation. It transforms regulatory and policy documents (PDF, DOCX, Markdown, HTML) into executable control logic while maintaining complete traceability from source clause to generated code.

## Quick Reference

```bash
# Install dependencies
make dev-install

# Run the API server (port 8080)
make run

# Run tests
make test                 # All tests
make test-cov             # With coverage (70% minimum)
make test-fast            # Skip slow tests

# Code quality
make lint                 # Run Ruff linter
make format               # Format with Ruff/Black
make type-check           # MyPy strict mode
make check-all            # All quality checks

# Docker
make docker-up            # Start all services
make docker-down          # Stop services
```

## Project Structure

```
aegislang/                 # Main source code
├── agents/               # Multi-agent pipeline (L1-L5)
│   ├── aegis_ingestor.py        # L1: Document ingestion
│   ├── policy_parser_agent.py   # L2: Semantic parsing
│   ├── schema_mapping_agent.py  # L3: Entity mapping
│   ├── compiler_agent.py        # L4: Code generation
│   └── trace_validator_agent.py # L5: Validation & lineage
├── api/                  # FastAPI REST server
├── core/                 # Utilities (logging, events, metrics)
└── config/               # Configuration management

templates/                # Jinja2 templates for code generation
├── yaml/                 # YAML compliance rules
├── sql/                  # SQL constraints & triggers
└── python/               # Python validators

tests/                    # Test suite
├── test_*.py             # Unit tests
├── test_integration.py   # Integration tests
└── performance/          # Load tests (Locust)
```

## Architecture

5-layer multi-agent pipeline:

| Layer | Agent | Purpose |
|-------|-------|---------|
| L1 | `aegis_ingestor.py` | Parse documents, chunk text, extract metadata |
| L2 | `policy_parser_agent.py` | Extract semantic clauses (obligations, prohibitions) |
| L3 | `schema_mapping_agent.py` | Map entities to target schemas via embeddings |
| L4 | `compiler_agent.py` | Generate artifacts (YAML, SQL, Python, Terraform, Rego) |
| L5 | `trace_validator_agent.py` | Verify traceability, emit provenance graph |

## Coding Standards

- **Python 3.11+** required
- **Type hints** mandatory for all functions (MyPy strict mode)
- **Docstrings** Google-style for public APIs
- **Line length** 100 characters
- **Formatting** Ruff (primary), Black (backup)

### Patterns

- Use **Pydantic models** for all data schemas with Field descriptions
- Each agent is a discrete processing layer with defined input/output types
- Use **Jinja2 templates** for code generation (in `templates/`)
- All transformations include **confidence scores**
- Use **structlog** for logging with context variables

### Example Model Pattern

```python
class TextChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., min_length=1)
    token_count: int = Field(..., ge=0)
    embedding_vector: list[float] | None = Field(default=None)
```

## Testing

```bash
pytest tests/                    # Run all tests
pytest tests/test_parser.py      # Single test file
pytest -m "not slow"             # Skip slow tests
pytest --cov=aegislang           # With coverage
```

Test markers:
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.security` - Security tests

Coverage target: **70% minimum**

## Environment Variables

- `ANTHROPIC_API_KEY` - Claude API key (optional, for LLM features)
- `OPENAI_API_KEY` - OpenAI API key (optional)
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `NEO4J_URI` - Neo4j connection string

## Services (Docker)

- **aegislang** (8080) - REST API server
- **postgres** (5432) - Schema registry
- **redis** (6379) - Event bus & cache
- **neo4j** (7687) - Provenance graph

## Key Files

- `Makefile` - Build automation commands
- `pyproject.toml` - Python config, tool settings
- `config.yaml` - Application configuration
- `docker-compose.yml` - Service orchestration
- `requirements.txt` - Production dependencies
