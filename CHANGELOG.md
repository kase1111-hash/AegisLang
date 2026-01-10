# Changelog

All notable changes to AegisLang will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-10

### Added

#### Core Pipeline (L1-L5 Agents)
- **L1 Ingestion Layer** (`aegis_ingestor.py`): Document parsing for PDF, DOCX, Markdown, and HTML formats with metadata extraction and content chunking
- **L2 Parsing Layer** (`policy_parser_agent.py`): LLM-powered clause extraction supporting obligations, prohibitions, permissions, and conditionals with structured output
- **L3 Mapping Layer** (`schema_mapping_agent.py`): Semantic entity mapping using embeddings with configurable similarity thresholds and manual override support
- **L4 Compilation Layer** (`compiler_agent.py`): Multi-format code generation with Jinja2 templates for YAML, SQL, Python, Terraform, and Rego
- **L5 Validation Layer** (`trace_validator_agent.py`): Artifact validation with confidence scoring, lineage tracking, and review flag generation

#### Templates
- YAML templates for obligation, prohibition, and conditional rules
- SQL templates for check constraints and audit triggers
- Python templates for test stubs and validator classes

#### API & Infrastructure
- REST API server with FastAPI (`/api/v1/`)
- Endpoints: `/ingest`, `/documents`, `/clauses`, `/rules`, `/compile`, `/trace`, `/health`
- Docker multi-stage build for production deployment
- Docker Compose with PostgreSQL, Redis, and Neo4j services
- CI/CD pipeline with GitHub Actions

#### Developer Experience
- Makefile with 30+ automation targets
- Structured logging with Sentry integration
- Comprehensive unit and integration tests
- API documentation

#### Configuration
- Environment-based configuration (dev/stage/prod)
- Secure secrets management via environment variables
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
- KEYWORDS.md with domain glossary

---

## [Unreleased]

### Planned
- RAG-based policy retrieval
- Rule drift detection
- Neo4j provenance graph visualization
- Audit chain visualizer (NatLangChain integration)
- Performance benchmarking suite
- Terraform and OPA/Rego output format improvements

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2024-01-10 | Initial release with full L1-L5 pipeline |

---

## Contributing

See [README.md](README.md) for contribution guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
