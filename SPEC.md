# AegisLang Technical Specification

**Version:** 0.1.0
**Status:** Alpha
**Last Updated:** February 2026

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

AegisLang employs a **layered multi-agent architecture**. Each layer operates as a discrete agent, callable independently or as part of the full pipeline.

### 2.2 Layer Specification

| Layer | Purpose | Agent Module | Input | Output |
|-------|---------|------------|-------|--------|
| **L1: Ingestion** | Collect and preprocess policy documents | `aegis_ingestor.py` | Raw documents (PDF, DOCX, MD, HTML) | Normalized JSON sections + text chunks |
| **L2: Parsing** | Extract obligations, conditions, actors | `policy_parser_agent.py` | Text chunks | Semantic clause structures |
| **L3: Mapping** | Link entities to system schemas | `schema_mapping_agent.py` | Clause structures | Entity-to-field mappings |
| **L4: Compilation** | Generate executable artifacts | `compiler_agent.py` | Mapped clauses | YAML, SQL, Python artifacts |
| **L5: Validation** | Verify correctness, emit provenance | `trace_validator_agent.py` | Artifacts + source clauses | Trace graphs + validation metadata |

### 2.3 Data Flow

```
     ┌──────────────┐
     │   Document   │  PDF, DOCX, MD, HTML
     │    Source     │
     └──────┬───────┘
            │
            ▼
┌──────────────────────┐
│   Document Ingestor  │  Tokenization via tiktoken
│       (L1)           │  Section hierarchy extraction
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐     ┌─────────────────┐
│  Policy Parser Agent │────▶│    LLM API      │  Anthropic/OpenAI
│       (L2)           │     │  (or Mock)      │  for clause extraction
└──────────┬───────────┘     └─────────────────┘
           │
           ▼
┌──────────────────────┐     ┌─────────────────┐
│ Schema Mapping Agent │────▶│ Schema Registry │  In-memory registry
│       (L3)           │     │  (Pydantic)     │  with synonym maps
└──────────┬───────────┘     └─────────────────┘
           │
           ▼
┌──────────────────────┐     ┌─────────────────┐
│   Compiler Agent     │────▶│   Templates     │  Jinja2 templates
│       (L4)           │     │   (File System) │  for each format
└──────────┬───────────┘     └─────────────────┘
           │
           ▼
┌──────────────────────┐
│ Trace Validator Agent│  Provenance chain validation
│       (L5)           │  Confidence scoring
└──────────┬───────────┘
           │
           ▼
     ┌─────┴─────┐
     │ Artifacts │  YAML rules, SQL constraints, Python tests
     └───────────┘
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
| ING-006 | Semantic text chunking (tiktoken) | Implemented |
| ING-007 | Document hierarchy preservation | Implemented |
| ING-008 | Standardized JSON output | Implemented |

**Not implemented:** ING-005 (OCR for scanned documents).

### 3.2 Parsing Layer (L2)

**Module:** `aegislang/agents/policy_parser_agent.py`

**Implemented features:**

| ID | Feature | Status |
|----|---------|--------|
| PRS-001 | Clause type detection (obligation, prohibition, permission, conditional, definition, exception) | Implemented |
| PRS-002 | Actor entity extraction | Implemented |
| PRS-003 | Action/verb phrase extraction | Implemented |
| PRS-004 | Object/target entity extraction | Implemented |
| PRS-005 | Conditional trigger extraction | Implemented |
| PRS-006 | Temporal scope extraction | Implemented |
| PRS-009 | Confidence scoring | Implemented |

**LLM providers:** Anthropic (Claude), OpenAI (GPT), Mock (keyword-based).

**Not implemented:** PRS-008 (cross-reference resolution between clauses).

**Clause Type Taxonomy:**

| Type | Modal Indicators |
|------|------------------|
| `obligation` | must, shall, is required to |
| `prohibition` | must not, shall not, is prohibited from |
| `permission` | may, is permitted to, can |
| `conditional` | if, when, where, unless |
| `definition` | means, refers to, is defined as |

### 3.3 Mapping Layer (L3)

**Module:** `aegislang/agents/schema_mapping_agent.py`

**Implemented features:**

| ID | Feature | Status |
|----|---------|--------|
| MAP-001 | Match entities to schema field paths | Implemented |
| MAP-002 | Semantic embedding matching (mock + SentenceTransformer) | Implemented |
| MAP-003 | Multiple target schema formats (SQL, API, Object) | Implemented |
| MAP-004 | Schema Registry with versioning | Implemented |
| MAP-005 | Synonym resolution | Implemented |
| MAP-006 | Manual mapping overrides | Implemented |
| MAP-007 | Confidence scoring | Implemented |
| MAP-008 | Unmappable entity detection | Implemented |

### 3.4 Compilation Layer (L4)

**Module:** `aegislang/agents/compiler_agent.py`

**Supported output formats:**

| Format | Status | Template Directory |
|--------|--------|-------------------|
| YAML compliance rules | Implemented | `templates/yaml/` |
| SQL check constraints + triggers | Implemented | `templates/sql/` |
| Python pytest stubs | Implemented | `templates/python/` |
| Terraform | Not implemented | — |
| OPA/Rego | Not implemented | — |

### 3.5 Validation Layer (L5)

**Module:** `aegislang/agents/trace_validator_agent.py`

**Implemented features:**

| ID | Feature | Status |
|----|---------|--------|
| VAL-001 | Clause-to-artifact provenance chain validation | Implemented |
| VAL-002 | Artifact syntax validation | Implemented |
| VAL-003 | Confidence scoring for trace links | Implemented |
| VAL-005 | Lineage metadata generation | Implemented |
| VAL-007 | Low-confidence flagging for human review | Implemented |

**Not implemented:** VAL-004 (semantic drift detection), VAL-008 (graph database persistence — Neo4j driver present but optional).

---

## 4. API Specification

### 4.1 REST API

**Base URL:** `http://localhost:8080/api/v1`

**Authentication:** API key via `X-API-Key` header. Set `AEGISLANG_DISABLE_AUTH=true` to disable.

### 4.2 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (no auth required) |
| `POST` | `/ingest` | Upload and process a document |
| `GET` | `/documents` | List all ingested documents |
| `GET` | `/documents/{doc_id}` | Get document details |
| `GET` | `/clauses/{doc_id}` | Get extracted clauses for a document |
| `POST` | `/compile` | Trigger compilation for a document |
| `POST` | `/schemas` | Register a target schema |
| `GET` | `/schemas` | List registered schemas |
| `GET` | `/schemas/{schema_id}` | Get a specific schema |
| `GET` | `/jobs/{job_id}` | Check async job status |

### 4.3 Storage

**Default:** In-memory dictionaries (non-persistent). Data is lost on server restart.

**Available:** SQLite backend for persistent storage of jobs, documents, schemas, clauses, and artifacts. Enable with `AEGISLANG_STORAGE_BACKEND=sqlite`.

---

## 5. Dependencies

### 5.1 Core Runtime (`requirements.txt`)

| Package | Purpose |
|---------|---------|
| fastapi, uvicorn, pydantic | REST API framework |
| python-multipart | Form/file upload handling |
| pdfminer.six | PDF text extraction |
| python-docx | DOCX parsing |
| beautifulsoup4 | HTML parsing |
| tiktoken | Token counting for chunking |
| anthropic, openai | LLM integration |
| jinja2, pyyaml, sqlparse | Template rendering + artifact formatting |
| structlog | Structured logging |
| sentry-sdk | Error tracking (optional) |
| neo4j | Graph database for provenance (optional) |

### 5.2 Optional ML (`requirements-ml.txt`)

| Package | Purpose |
|---------|---------|
| sentence-transformers | Semantic embedding for schema mapping |
| transformers, torch | ML model runtime |

---

## 6. Testing

### 6.1 Test Suite

| Level | Count | Status |
|-------|-------|--------|
| Unit tests | ~80 | All passing |
| Integration tests | ~40 | All passing |
| System tests | ~20 | All passing |
| **Total** | **139** | **All passing** |

### 6.2 Running Tests

```bash
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
| **Schema Registry** | Catalog of target system schemas available for entity mapping |
| **Trace** | A validated provenance record linking clause to artifact |

---

*End of Specification*
