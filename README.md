# AegisLang

**Language in. Compliance out.**

AegisLang is a multi-agent semantic compiler that transforms natural-language policy documents (regulations, SOPs, governance rules) into executable control logic — YAML rules, SQL constraints, and Python checks — with full clause-to-artifact traceability.

## Purpose

- Convert policy documents to machine-enforceable rules automatically
- Maintain traceability from source regulation to generated artifact
- Support multiple output formats (YAML, SQL, Python)

## Core Flow

```
policy_doc → AegisIngestor → PolicyParser → SchemaMapper → Compiler → TraceValidator
```

1. **Ingest** — parse PDF, DOCX, Markdown, or HTML into structured sections
2. **Parse** — extract clauses with type detection (obligation, prohibition, permission, conditional)
3. **Map** — align policy entities to target database schema via synonym + semantic matching
4. **Compile** — emit YAML, SQL, and Python artifacts from Jinja2 templates
5. **Validate** — verify clause-to-artifact traceability and provenance

## Example Output

```yaml
control:
  id: KYC-102
  source: "AML Reg §5.3"
  rule: "Verify customer identity for all accounts > $5,000"
  emit: "identity_check_routine()"
```

## Features

- **Full clause-to-code traceability** — every artifact links back to its source clause
- **Plug-in compiler templates** for YAML, SQL, and Python output
- **LLM-driven schema mapping** (Anthropic / OpenAI) — no hard-coded entity rules
- **Mock mode** for offline development and testing without LLM API keys
- **Configurable template system** via Jinja2

## Current Status

AegisLang is an early-stage prototype. The core pipeline works end-to-end with mock LLM clients. Real LLM integration (Anthropic, OpenAI) is implemented but not yet validated against real regulatory documents.

## Roadmap

- RAG-based policy retrieval
- Continuous rule drift detection
- Persistent storage (SQLite) for the API layer
- Domain-specific prompt tuning (AML/KYC)

## Quick Start

```bash
pip install -r requirements.txt
python -m aegislang.api.server
```

AegisLang can integrate with Agent-OS for event-driven pipeline orchestration.

---

## Setup Guide

### Phase 1: Project Foundation
- [x] Create project directory structure
- [x] Create `requirements.txt` with dependencies
- [x] Create `config.yaml` configuration file
- [x] Set up environment variables (`.env.example`)

### Phase 2: Core Agents Implementation
- [x] Implement L1 Ingestion Layer (`aegis_ingestor.py`)
- [x] Implement L2 Parsing Layer (`policy_parser_agent.py`)
- [x] Implement L3 Mapping Layer (`schema_mapping_agent.py`)
- [x] Implement L4 Compilation Layer (`compiler_agent.py`)
- [x] Implement L5 Validation Layer (`trace_validator_agent.py`)

### Phase 3: Templates & Output Formats
- [x] Create YAML templates (`templates/yaml/`)
- [x] Create SQL templates (`templates/sql/`)
- [x] Create Python test templates (`templates/python/`)

### Phase 4: API & Deployment
- [x] Implement REST API server
- [x] Create Dockerfile
- [x] Create docker-compose.yml
- [x] Set up CI/CD pipeline

### Phase 5: Testing & Documentation
- [x] Write unit tests
- [x] Write integration tests
- [x] Create API documentation
