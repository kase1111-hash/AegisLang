# AegisLang Architecture

This document describes the system architecture, data flow, and component interactions in AegisLang.

## Table of Contents
- [High-Level Architecture](#high-level-architecture)
- [Data Flow Pipeline](#data-flow-pipeline)
- [Component Details](#component-details)
- [Database Schema](#database-schema)
- [API Architecture](#api-architecture)
- [Deployment Architecture](#deployment-architecture)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AegisLang System                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────────────────────────────────────────┐   │
│  │   Clients    │    │                 REST API Layer                    │   │
│  │              │───▶│  FastAPI Server (POST /ingest, GET /rules, etc.) │   │
│  │ - Web UI     │    └──────────────────────────────────────────────────┘   │
│  │ - CLI        │                            │                              │
│  │ - SDK        │                            ▼                              │
│  └──────────────┘    ┌──────────────────────────────────────────────────┐   │
│                      │              Agent Pipeline (L1-L5)               │   │
│                      │                                                    │   │
│                      │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  │   │
│                      │  │   L1   │─▶│   L2   │─▶│   L3   │─▶│   L4   │  │   │
│                      │  │Ingest  │  │ Parse  │  │  Map   │  │Compile │  │   │
│                      │  └────────┘  └────────┘  └────────┘  └────────┘  │   │
│                      │                                           │       │   │
│                      │                                           ▼       │   │
│                      │                                      ┌────────┐   │   │
│                      │                                      │   L5   │   │   │
│                      │                                      │Validate│   │   │
│                      │                                      └────────┘   │   │
│                      └──────────────────────────────────────────────────┘   │
│                                            │                                 │
│                                            ▼                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         Data Layer                                    │   │
│  │                                                                       │   │
│  │  ┌────────────┐    ┌────────────┐    ┌────────────┐                  │   │
│  │  │ PostgreSQL │    │   Redis    │    │   Neo4j    │                  │   │
│  │  │  (Schema)  │    │  (Events)  │    │ (Lineage)  │                  │   │
│  │  └────────────┘    └────────────┘    └────────────┘                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

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
    │  • Extract text content                                         │
    │  • Chunk into sections                                          │
    │  • Extract metadata                                             │
    │  • Generate content hash                                        │
    │                                                                 │
    │  Output: IngestedDocument                                       │
    │  {doc_id, sections[], metadata, content_hash}                   │
    └──────┬──────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  L2: PARSING LAYER (policy_parser_agent.py)                     │
    │  ──────────────────────────────────────────                     │
    │  • Identify clause boundaries                                   │
    │  • Classify clause types (obligation, prohibition, etc.)        │
    │  • Extract semantic components:                                 │
    │    - Actor (who)                                                │
    │    - Action (what verb)                                         │
    │    - Object (what target)                                       │
    │    - Condition (when/if)                                        │
    │    - Temporal scope (deadline, frequency)                       │
    │  • Calculate confidence scores                                  │
    │                                                                 │
    │  Output: ParsedClause[]                                         │
    │  {clause_id, type, actor, action, object, condition, temporal}  │
    └──────┬──────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  L3: MAPPING LAYER (schema_mapping_agent.py)                    │
    │  ─────────────────────────────────────────                      │
    │  • Load target schema definitions                               │
    │  • Generate embeddings for entities                             │
    │  • Semantic similarity matching                                 │
    │  • Resolve synonyms and aliases                                 │
    │  • Apply manual overrides                                       │
    │  • Flag unmappable entities                                     │
    │                                                                 │
    │  Output: MappedClause[]                                         │
    │  {clause_id, source_clause, mapped_entities[], confidence}      │
    └──────┬──────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  L4: COMPILATION LAYER (compiler_agent.py)                      │
    │  ────────────────────────────────────────                       │
    │  • Select appropriate template (Jinja2)                         │
    │  • Render executable artifacts:                                 │
    │    - YAML compliance rules                                      │
    │    - SQL check constraints & triggers                           │
    │    - Python test stubs & validators                             │
    │    - Terraform/Sentinel policies                                │
    │    - OPA/Rego policies                                          │
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
    │  • Verify artifact integrity                                    │
    │  • Check semantic consistency                                   │
    │  • Validate lineage (source → artifact traceability)            │
    │  • Calculate overall confidence                                 │
    │  • Generate review flags for low-confidence items               │
    │  • Create audit trail                                           │
    │                                                                 │
    │  Output: ValidationTrace                                        │
    │  {trace_id, status, confidence, checks[], review_flags[]}       │
    └──────┬──────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────┐
    │   Output    │
    │  Artifacts  │  (.yaml, .sql, .py, .tf, .rego)
    └─────────────┘
```

### Event Flow

```
    ┌────────────────────────────────────────────────────────────────┐
    │                      Redis Event Bus                            │
    └────────────────────────────────────────────────────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
         ▼                          ▼                          ▼
    ┌─────────┐              ┌─────────────┐            ┌───────────┐
    │ policy  │              │   policy    │            │  policy   │
    │.ingested│              │  .parsed    │            │ .compiled │
    └─────────┘              └─────────────┘            └───────────┘
         │                          │                          │
         │                          │                          │
         ▼                          ▼                          ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Event Subscribers                           │
    │  • Audit Logger                                                 │
    │  • Notification Service                                         │
    │  • Analytics Pipeline                                           │
    │  • External Integrations (Agent-OS, NatLangChain)               │
    └─────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### Agent Layer Responsibilities

| Layer | Agent | Input | Output | Key Functions |
|-------|-------|-------|--------|---------------|
| L1 | `AegisIngestor` | Raw documents | `IngestedDocument` | Parse, chunk, extract metadata |
| L2 | `PolicyParserAgent` | Text sections | `ParsedClause[]` | NLP extraction, classification |
| L3 | `SchemaMappingAgent` | Parsed clauses | `MappedClause[]` | Embedding match, resolve entities |
| L4 | `CompilerAgent` | Mapped clauses | `CompiledArtifact[]` | Template rendering, syntax check |
| L5 | `TraceValidatorAgent` | Artifacts | `ValidationTrace` | Integrity, lineage, confidence |

### Supported Clause Types

```
┌────────────────────────────────────────────────────────────┐
│                     Clause Types                            │
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
│                      Output Formats                               │
├──────────────┬───────────────────────────────────────────────────┤
│ YAML         │ Compliance rule definitions for CI/CD             │
│ SQL          │ CHECK constraints, triggers for databases         │
│ Python       │ pytest stubs, validator classes                   │
│ Terraform    │ Sentinel policies for infrastructure              │
│ Rego         │ OPA policies for authorization                    │
│ JSON         │ Structured rules for custom integrations          │
└──────────────┴───────────────────────────────────────────────────┘
```

---

## Database Schema

### Entity Relationship Diagram

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│    schemas      │       │   documents     │       │    clauses      │
├─────────────────┤       ├─────────────────┤       ├─────────────────┤
│ id (PK)         │       │ id (PK)         │       │ id (PK)         │
│ schema_id       │       │ doc_id          │◄──────│ doc_id (FK)     │
│ schema_type     │       │ source_file     │       │ clause_id       │
│ version         │       │ document_type   │       │ clause_type     │
│ tables_json     │       │ metadata_json   │       │ source_text     │
│ created_at      │       │ content_hash    │       │ parsed_json     │
│ updated_at      │       │ status          │       │ confidence      │
└─────────────────┘       │ created_at      │       │ created_at      │
                          └─────────────────┘       └────────┬────────┘
                                                             │
                          ┌──────────────────────────────────┘
                          │
                          ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   artifacts     │       │validation_results│      │compliance_audit │
├─────────────────┤       ├─────────────────┤       ├─────────────────┤
│ id (PK)         │       │ id (PK)         │       │ id (PK)         │
│ artifact_id     │◄──────│ artifact_id(FK) │       │ clause_id       │
│ clause_id (FK)  │───────│ clause_id (FK)  │       │ violation_type  │
│ format          │       │ trace_id        │       │ table_name      │
│ content         │       │ validation_stat │       │ severity        │
│ file_path       │       │ confidence_score│       │ resolved        │
│ syntax_valid    │       │ checks_json     │       │ occurred_at     │
│ template_used   │       │ review_flags    │       │ created_at      │
│ created_at      │       │ validated_at    │       └─────────────────┘
└─────────────────┘       └─────────────────┘

┌─────────────────┐
│      jobs       │
├─────────────────┤
│ id (PK)         │
│ job_id          │
│ job_type        │
│ status          │
│ result_json     │
│ error_message   │
│ created_at      │
│ started_at      │
│ completed_at    │
└─────────────────┘
```

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
│  POST   /api/v1/ingest           Upload and process document       │
│  GET    /api/v1/documents        List all documents                │
│  GET    /api/v1/documents/{id}   Get document details              │
│                                                                    │
│  Clause Management                                                 │
│  ─────────────────                                                 │
│  GET    /api/v1/clauses          List clauses (filterable)         │
│  GET    /api/v1/clauses/{id}     Get clause details                │
│                                                                    │
│  Rule Compilation                                                  │
│  ────────────────                                                  │
│  GET    /api/v1/rules/{id}       Get compiled rules for clause     │
│  POST   /api/v1/compile          Compile clauses to artifacts      │
│                                                                    │
│  Validation & Tracing                                              │
│  ────────────────────                                              │
│  GET    /api/v1/trace/{id}       Get validation trace              │
│  GET    /api/v1/artifacts        List artifacts                    │
│                                                                    │
│  Schema Management                                                 │
│  ─────────────────                                                 │
│  GET    /api/v1/schemas          List registered schemas           │
│  POST   /api/v1/schemas          Register new schema               │
│                                                                    │
│  System                                                            │
│  ──────                                                            │
│  GET    /api/v1/health           Health check                      │
│  GET    /api/v1/jobs/{id}        Get async job status              │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Request Flow

```
    Client Request
          │
          ▼
    ┌─────────────┐
    │   FastAPI   │
    │   Router    │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐     ┌─────────────┐
    │  Request    │────▶│   Pydantic  │
    │ Validation  │     │   Models    │
    └──────┬──────┘     └─────────────┘
           │
           ▼
    ┌─────────────┐
    │   Service   │
    │    Layer    │
    └──────┬──────┘
           │
           ├────────────────┬────────────────┐
           ▼                ▼                ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │   Agent     │  │  Database   │  │   Redis     │
    │  Pipeline   │  │   (SQL)     │  │  (Events)   │
    └─────────────┘  └─────────────┘  └─────────────┘
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
│                     Docker Compose Network                          │
│                     (aegislang-network)                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    aegislang (API)                           │   │
│  │                    Port: 8080                                │   │
│  │                                                              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │   │
│  │  │   FastAPI    │  │    Agents    │  │   Workers    │       │   │
│  │  │   Server     │  │   (L1-L5)    │  │   (Async)    │       │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│         │                    │                    │                 │
│         │                    │                    │                 │
│         ▼                    ▼                    ▼                 │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐         │
│  │  PostgreSQL │      │    Redis    │      │    Neo4j    │         │
│  │  Port: 5432 │      │  Port: 6379 │      │ Port: 7474  │         │
│  │             │      │             │      │      7687   │         │
│  │  - schemas  │      │  - events   │      │  - lineage  │         │
│  │  - documents│      │  - cache    │      │  - graphs   │         │
│  │  - clauses  │      │  - pub/sub  │      │             │         │
│  │  - artifacts│      │             │      │             │         │
│  └─────────────┘      └─────────────┘      └─────────────┘         │
│         │                    │                    │                 │
│         ▼                    ▼                    ▼                 │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐         │
│  │ postgres-   │      │  redis-     │      │  neo4j-     │         │
│  │   data      │      │   data      │      │   data      │         │
│  │  (volume)   │      │  (volume)   │      │  (volume)   │         │
│  └─────────────┘      └─────────────┘      └─────────────┘         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

External Access:
  - API:      http://localhost:8080
  - Neo4j UI: http://localhost:7474
  - Dev API:  http://localhost:8081 (with --profile dev)
```

### Production Deployment

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Production Environment                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐                                                   │
│  │   Load       │                                                   │
│  │  Balancer    │                                                   │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ├──────────────────┬──────────────────┐                     │
│         ▼                  ▼                  ▼                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │  AegisLang  │    │  AegisLang  │    │  AegisLang  │             │
│  │  Instance 1 │    │  Instance 2 │    │  Instance N │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│         │                  │                  │                     │
│         └──────────────────┼──────────────────┘                     │
│                            │                                        │
│         ┌──────────────────┼──────────────────┐                     │
│         ▼                  ▼                  ▼                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │  PostgreSQL │    │    Redis    │    │    Neo4j    │             │
│  │   (RDS)     │    │  (Cluster)  │    │  (Cluster)  │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Security Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Security Layers                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Layer 1: Network                                                   │
│  ─────────────────                                                  │
│  • Docker network isolation                                         │
│  • CORS configuration                                               │
│  • Rate limiting                                                    │
│                                                                      │
│  Layer 2: Authentication                                            │
│  ───────────────────────                                            │
│  • API key validation                                               │
│  • JWT token support (optional)                                     │
│                                                                      │
│  Layer 3: Input Validation                                          │
│  ─────────────────────────                                          │
│  • Pydantic model validation                                        │
│  • File type verification                                           │
│  • Size limits                                                      │
│                                                                      │
│  Layer 4: Data Protection                                           │
│  ────────────────────────                                           │
│  • Parameterized SQL queries                                        │
│  • Content sanitization                                             │
│  • Secure file handling                                             │
│                                                                      │
│  Layer 5: Runtime                                                   │
│  ────────────────                                                   │
│  • Non-root container user                                          │
│  • Secret management (env vars)                                     │
│  • Sentry error tracking                                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Integration Points

### NatLangChain Ecosystem

```
┌─────────────────────────────────────────────────────────────────────┐
│                   NatLangChain Ecosystem Integration                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │  AegisLang  │◄──▶│  Agent-OS   │◄──▶│ NatLangChain│             │
│  │  (Compiler) │    │ (Orchestr.) │    │ (Blockchain)│             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│        │                  │                   │                     │
│        │                  │                   │                     │
│        ▼                  ▼                   ▼                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Redis Event Bus                           │   │
│  │                                                              │   │
│  │  Topics:                                                     │   │
│  │  • policy.ingested    • policy.parsed                        │   │
│  │  • policy.mapped      • policy.compiled                      │   │
│  │  • policy.validated   • policy.error                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

*Last updated: 2024-01-10 | Version: 1.0.0*
