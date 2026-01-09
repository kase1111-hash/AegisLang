# AegisLang Technical Specification

**Version:** 1.0.0  
**Status:** Draft  
**Last Updated:** January 2026  
**Classification:** Internal Technical Document

---

## Document Control

| Field | Value |
|-------|-------|
| Document ID | AEGIS-SPEC-001 |
| Author | True North Systems |
| Reviewers | — |
| Approval Status | Pending |

---

## 1. Executive Summary

### 1.1 Purpose

AegisLang is a multi-agent semantic compiler that transforms unstructured regulatory and policy text into executable controls, workflows, and audit artifacts. The system maintains complete semantic traceability from source clause to generated code, enabling organizations to automate compliance implementation while preserving human-auditable provenance chains.

### 1.2 Tagline

**Language in. Compliance out.**

### 1.3 Core Value Proposition

- **Eliminates manual translation** of regulatory requirements into technical controls
- **Maintains provenance** from policy clause to executable artifact
- **Reduces compliance drift** through automated policy-to-code synchronization
- **Enables audit transparency** via traceable lineage graphs

### 1.4 Target Domains

| Domain | Example Regulations |
|--------|---------------------|
| Financial Services | AML, KYC, BSA, SOX |
| Data Privacy | GDPR, CCPA, HIPAA |
| Information Security | ISO 27001, NIST CSF, SOC 2 |
| Industry-Specific | PCI-DSS, NERC CIP, FDA 21 CFR Part 11 |

---

## 2. System Architecture

### 2.1 Architectural Pattern

AegisLang employs a **layered multi-agent architecture** coordinated through NatLangChain's semantic routing graph. Each layer operates as a discrete Agent-OS node, enabling horizontal scaling and isolated failure domains.

### 2.2 Layer Specification

| Layer | Purpose | Agent Node | Input | Output |
|-------|---------|------------|-------|--------|
| **L1: Ingestion** | Collect and preprocess policy documents | `aegis_ingestor` | Raw documents (PDF, DOCX, MD, HTML) | Normalized JSON chunks |
| **L2: Parsing** | Extract obligations, conditions, actors | `policy_parser_agent` | JSON chunks | Semantic clause structures |
| **L3: Mapping** | Link entities to system schemas | `schema_mapping_agent` | Clause structures | Entity-to-field mappings |
| **L4: Compilation** | Generate executable artifacts | `compiler_agent` | Mapped clauses | YAML, SQL, Python, Terraform |
| **L5: Validation** | Verify correctness, emit provenance | `trace_validator_agent` | Artifacts + source clauses | Trace graphs + audit metadata |

### 2.3 System Context Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EXTERNAL SYSTEMS                                │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│  Policy Portals │  Document Repos │   CI/CD Pipes   │   Audit Dashboards    │
└────────┬────────┴────────┬────────┴────────┬────────┴───────────┬───────────┘
         │                 │                 │                    │
         ▼                 ▼                 ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AEGISLANG CORE                                  │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  │
│  │ Ingestion │──│  Parsing  │──│  Mapping  │──│ Compiling │──│ Validation│  │
│  │   (L1)    │  │   (L2)    │  │   (L3)    │  │   (L4)    │  │   (L5)    │  │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘  └───────────┘  │
│                                     │                                        │
│                    ┌────────────────┴────────────────┐                       │
│                    │   NatLangChain Orchestration    │                       │
│                    │      + Agent-OS Scheduler       │                       │
│                    └─────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
         │                 │                 │                    │
         ▼                 ▼                 ▼                    ▼
┌─────────────────┬─────────────────┬─────────────────┬───────────────────────┐
│  Vector Store   │  Schema Registry│ Artifact Store  │   Provenance Graph    │
│   (Pinecone)    │   (PostgreSQL)  │  (File System)  │   (Neo4j/ArangoDB)    │
└─────────────────┴─────────────────┴─────────────────┴───────────────────────┘
```

### 2.4 Agent-OS Integration

AegisLang agents operate as first-class Agent-OS nodes, subscribing and publishing to semantic event topics for distributed coordination.

**Event Topic Registry:**

| Topic | Publisher | Subscribers | Payload Type |
|-------|-----------|-------------|--------------|
| `policy.ingested` | L1 Ingestor | L2 Parser | `IngestedDocument` |
| `policy.parsed` | L2 Parser | L3 Mapper | `ParsedClause[]` |
| `policy.mapped` | L3 Mapper | L4 Compiler | `MappedClause[]` |
| `policy.compiled` | L4 Compiler | L5 Validator | `CompiledArtifact[]` |
| `policy.validated` | L5 Validator | External Systems | `ValidationResult` |
| `policy.error` | Any Agent | Error Handler | `AgentError` |

---

## 3. Component Specifications

### 3.1 Ingestion Layer (L1)

#### 3.1.1 Module: `aegis_ingestor.py`

**Purpose:** Intake, clean, and normalize policy text from heterogeneous sources.

**Functional Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| ING-001 | Parse PDF documents with text extraction | Must Have |
| ING-002 | Parse DOCX documents preserving structure | Must Have |
| ING-003 | Parse Markdown files | Must Have |
| ING-004 | Parse HTML from web-based policy portals | Should Have |
| ING-005 | Apply OCR for scanned documents | Should Have |
| ING-006 | Chunk text into semantically coherent sections | Must Have |
| ING-007 | Detect and preserve document hierarchy (sections, subsections) | Must Have |
| ING-008 | Emit standardized JSON for downstream agents | Must Have |

**Chunking Strategy:**

- Target chunk size: 512–1024 tokens
- Chunking method: Semantic embedding similarity with sliding window
- Overlap: 10% token overlap between adjacent chunks
- Hierarchy preservation: Maintain parent section references

**Output Schema:**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "IngestedDocument",
  "type": "object",
  "required": ["doc_id", "metadata", "sections"],
  "properties": {
    "doc_id": {
      "type": "string",
      "pattern": "^[A-Z0-9_]+$",
      "description": "Unique document identifier"
    },
    "metadata": {
      "type": "object",
      "properties": {
        "source_file": { "type": "string" },
        "ingestion_timestamp": { "type": "string", "format": "date-time" },
        "document_type": { "type": "string", "enum": ["pdf", "docx", "markdown", "html"] },
        "page_count": { "type": "integer" },
        "language": { "type": "string" },
        "hash": { "type": "string", "description": "SHA-256 of source document" }
      }
    },
    "sections": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["section_id", "section_title", "text_chunks"],
        "properties": {
          "section_id": { "type": "string" },
          "section_title": { "type": "string" },
          "parent_section": { "type": ["string", "null"] },
          "hierarchy_level": { "type": "integer", "minimum": 1, "maximum": 6 },
          "text_chunks": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "chunk_id": { "type": "string" },
                "text": { "type": "string" },
                "token_count": { "type": "integer" },
                "embedding_vector": { 
                  "type": "array", 
                  "items": { "type": "number" },
                  "description": "Optional pre-computed embedding"
                }
              }
            }
          }
        }
      }
    }
  }
}
```

**Dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| `unstructured` | ≥0.10.0 | Multi-format document parsing |
| `pdfminer.six` | ≥20221105 | PDF text extraction |
| `tika` | ≥2.6.0 | Fallback document parsing |
| `langchain.text_splitter` | ≥0.1.0 | Semantic chunking |
| `tiktoken` | ≥0.5.0 | Token counting |

---

### 3.2 Parsing Layer (L2)

#### 3.2.1 Module: `policy_parser_agent.py`

**Purpose:** Identify clause structure and extract semantic components from policy language.

**Functional Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| PRS-001 | Detect clause type (obligation, prohibition, conditional, definition, exception) | Must Have |
| PRS-002 | Extract actor entity from clause | Must Have |
| PRS-003 | Extract action/verb phrase from clause | Must Have |
| PRS-004 | Extract object/target entity from clause | Must Have |
| PRS-005 | Extract conditional triggers | Must Have |
| PRS-006 | Extract temporal scope (deadlines, frequencies) | Should Have |
| PRS-007 | Represent parsed clauses as semantic triples | Must Have |
| PRS-008 | Handle cross-references between clauses | Should Have |
| PRS-009 | Confidence scoring for each extraction | Must Have |

**Clause Type Taxonomy:**

| Type | Definition | Modal Indicators |
|------|------------|------------------|
| `obligation` | Required action | must, shall, is required to |
| `prohibition` | Forbidden action | must not, shall not, is prohibited from |
| `permission` | Allowed action | may, is permitted to, can |
| `conditional` | Action contingent on condition | if, when, where, unless |
| `definition` | Term definition | means, refers to, is defined as |
| `exception` | Carve-out from rule | except, unless, notwithstanding |

**Output Schema:**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ParsedClause",
  "type": "object",
  "required": ["clause_id", "source_chunk_id", "type", "actor", "action"],
  "properties": {
    "clause_id": {
      "type": "string",
      "description": "Unique clause identifier (doc_id + section + sequence)"
    },
    "source_chunk_id": {
      "type": "string",
      "description": "Reference to originating text chunk"
    },
    "source_text": {
      "type": "string",
      "description": "Original clause text for traceability"
    },
    "type": {
      "type": "string",
      "enum": ["obligation", "prohibition", "permission", "conditional", "definition", "exception"]
    },
    "actor": {
      "type": "object",
      "properties": {
        "entity": { "type": "string" },
        "qualifiers": { "type": "array", "items": { "type": "string" } }
      }
    },
    "action": {
      "type": "object",
      "properties": {
        "verb": { "type": "string" },
        "modifiers": { "type": "array", "items": { "type": "string" } }
      }
    },
    "object": {
      "type": ["object", "null"],
      "properties": {
        "entity": { "type": "string" },
        "qualifiers": { "type": "array", "items": { "type": "string" } }
      }
    },
    "condition": {
      "type": ["object", "null"],
      "properties": {
        "trigger": { "type": "string" },
        "temporal": { "type": ["string", "null"] }
      }
    },
    "temporal_scope": {
      "type": ["object", "null"],
      "properties": {
        "deadline": { "type": ["string", "null"] },
        "frequency": { "type": ["string", "null"] },
        "duration": { "type": ["string", "null"] }
      }
    },
    "cross_references": {
      "type": "array",
      "items": { "type": "string" },
      "description": "References to other clause_ids"
    },
    "confidence": {
      "type": "number",
      "minimum": 0,
      "maximum": 1
    }
  }
}
```

**NatLangChain Prompt Template:**

```
You are a legal language parser specializing in regulatory and policy documents.

Analyze the following clause and extract its semantic structure:

<clause>
{clause_text}
</clause>

Extract and return as JSON:
1. clause_type: obligation | prohibition | permission | conditional | definition | exception
2. actor: the entity responsible for the action (with any qualifiers)
3. action: the verb phrase describing what must/must not/may be done
4. object: the target of the action (if present)
5. condition: any triggering condition or prerequisite
6. temporal_scope: deadlines, frequencies, or durations mentioned
7. confidence: your confidence in this extraction (0.0 to 1.0)

Return ONLY valid JSON matching the ParsedClause schema.
```

**Dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| `spacy` | ≥3.7.0 | NLP pipeline, dependency parsing |
| `transformers` | ≥4.35.0 | Transformer models for extraction |
| `natlangchain` | ≥0.1.0 | Agent node framework |

---

### 3.3 Mapping Layer (L3)

#### 3.3.1 Module: `schema_mapping_agent.py`

**Purpose:** Align natural-language entities from parsed clauses to operational system schemas.

**Functional Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| MAP-001 | Match regulatory entities to schema field paths | Must Have |
| MAP-002 | Use semantic embeddings for fuzzy matching | Must Have |
| MAP-003 | Support multiple target schema formats (SQL, API, Object) | Must Have |
| MAP-004 | Maintain Schema Registry with versioning | Must Have |
| MAP-005 | Handle synonym resolution | Should Have |
| MAP-006 | Support manual mapping overrides | Should Have |
| MAP-007 | Confidence scoring for each mapping | Must Have |
| MAP-008 | Detect unmappable entities and flag for review | Must Have |

**Schema Registry Structure:**

```json
{
  "registry_version": "1.0.0",
  "schemas": [
    {
      "schema_id": "user_schema_v2",
      "schema_type": "sql",
      "tables": [
        {
          "table_name": "user_table",
          "fields": [
            {
              "field_name": "user_id",
              "field_type": "UUID",
              "semantic_labels": ["customer", "account holder", "individual", "person"],
              "embedding": [0.123, 0.456, ...]
            }
          ]
        }
      ]
    }
  ],
  "synonyms": {
    "customer": ["client", "account holder", "consumer", "user"],
    "transaction": ["payment", "transfer", "wire", "remittance"]
  }
}
```

**Output Schema:**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "MappedClause",
  "type": "object",
  "required": ["clause_id", "mapped_entities", "mapping_status"],
  "properties": {
    "clause_id": { "type": "string" },
    "source_clause": { "$ref": "#/definitions/ParsedClause" },
    "mapped_entities": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["source_entity", "target_path", "confidence"],
        "properties": {
          "source_entity": { "type": "string" },
          "source_role": { 
            "type": "string", 
            "enum": ["actor", "object", "condition_subject"] 
          },
          "target_path": { 
            "type": "string",
            "description": "Dot-notation path to schema field (e.g., user_table.user_id)"
          },
          "target_schema": { "type": "string" },
          "confidence": { "type": "number", "minimum": 0, "maximum": 1 },
          "mapping_method": {
            "type": "string",
            "enum": ["exact", "synonym", "semantic", "manual_override"]
          }
        }
      }
    },
    "unmapped_entities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "entity": { "type": "string" },
          "reason": { "type": "string" },
          "suggested_matches": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "target_path": { "type": "string" },
                "confidence": { "type": "number" }
              }
            }
          }
        }
      }
    },
    "mapping_status": {
      "type": "string",
      "enum": ["complete", "partial", "failed", "needs_review"]
    }
  }
}
```

**Dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| `pinecone-client` | ≥2.2.0 | Vector similarity search |
| `weaviate-client` | ≥3.24.0 | Alternative vector store |
| `sqlalchemy` | ≥2.0.0 | Schema introspection |
| `sentence-transformers` | ≥2.2.0 | Embedding generation |

---

### 3.4 Compilation Layer (L4)

#### 3.4.1 Module: `compiler_agent.py`

**Purpose:** Translate parsed and mapped clauses into executable artifacts across multiple target formats.

**Functional Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| CMP-001 | Generate YAML compliance rule definitions | Must Have |
| CMP-002 | Generate SQL check constraints | Must Have |
| CMP-003 | Generate Python test stubs | Should Have |
| CMP-004 | Generate Terraform policy configs | Should Have |
| CMP-005 | Generate OPA/Rego policies | Could Have |
| CMP-006 | Support pluggable Jinja2 template system | Must Have |
| CMP-007 | Validate generated artifacts syntactically | Must Have |
| CMP-008 | Embed source clause references in artifacts | Must Have |

**Supported Output Formats:**

| Format | File Extension | Use Case |
|--------|----------------|----------|
| Compliance YAML | `.yaml` | Policy-as-code engines, GRC platforms |
| SQL Constraints | `.sql` | Database-level enforcement |
| Python Tests | `.py` | Automated compliance testing |
| Terraform | `.tf` | Infrastructure policy |
| OPA/Rego | `.rego` | Runtime policy decisions |
| JSON Rules | `.json` | API-based rule engines |

**Template Structure:**

```
templates/
├── yaml/
│   ├── obligation.yaml.j2
│   ├── prohibition.yaml.j2
│   └── conditional.yaml.j2
├── sql/
│   ├── check_constraint.sql.j2
│   └── trigger.sql.j2
├── python/
│   ├── test_stub.py.j2
│   └── validator_class.py.j2
├── terraform/
│   └── policy.tf.j2
└── rego/
    └── policy.rego.j2
```

**Example Template (YAML Obligation):**

```jinja2
# Source: {{ clause.source_text | truncate(80) }}
# Clause ID: {{ clause.clause_id }}
# Generated: {{ timestamp }}

control:
  id: {{ clause.clause_id }}
  type: {{ clause.type }}
  rule: "{{ clause.action.verb }} {{ clause.object.entity if clause.object else '' }}"
  
  actor:
    entity: "{{ clause.actor.entity }}"
    mapped_to: "{{ mapping.actor_path }}"
  
  action:
    operation: "{{ mapping.action_function }}"
    parameters:
      {% for param in mapping.parameters %}
      - {{ param.name }}: {{ param.value }}
      {% endfor %}
  
  {% if clause.condition %}
  trigger:
    event: "{{ clause.condition.trigger }}"
    {% if clause.condition.temporal %}
    temporal: "{{ clause.condition.temporal }}"
    {% endif %}
  {% endif %}
  
  severity: {{ severity_level | default('medium') }}
  
  metadata:
    source_document: "{{ clause.source_doc_id }}"
    confidence: {{ mapping.confidence }}
    generated_by: "aegislang-compiler-v{{ version }}"
```

**Output Schema:**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "CompiledArtifact",
  "type": "object",
  "required": ["artifact_id", "clause_id", "format", "content", "file_path"],
  "properties": {
    "artifact_id": { "type": "string" },
    "clause_id": { "type": "string" },
    "format": { 
      "type": "string",
      "enum": ["yaml", "sql", "python", "terraform", "rego", "json"]
    },
    "content": { "type": "string" },
    "file_path": { "type": "string" },
    "syntax_valid": { "type": "boolean" },
    "template_used": { "type": "string" },
    "compilation_timestamp": { "type": "string", "format": "date-time" },
    "warnings": {
      "type": "array",
      "items": { "type": "string" }
    }
  }
}
```

**Dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| `jinja2` | ≥3.1.0 | Template rendering |
| `pyyaml` | ≥6.0.0 | YAML validation |
| `sqlparse` | ≥0.4.0 | SQL validation |
| `black` | ≥23.0.0 | Python formatting |

---

### 3.5 Validation Layer (L5)

#### 3.5.1 Module: `trace_validator_agent.py`

**Purpose:** Ensure semantic traceability, validate correctness, and emit provenance metadata.

**Functional Requirements:**

| ID | Requirement | Priority |
|----|-------------|----------|
| VAL-001 | Validate clause→artifact provenance chain completeness | Must Have |
| VAL-002 | Check artifact syntax validity | Must Have |
| VAL-003 | Compute confidence scores for trace links | Must Have |
| VAL-004 | Detect semantic drift between clause and artifact | Should Have |
| VAL-005 | Generate lineage metadata for audit | Must Have |
| VAL-006 | Publish validated artifacts to downstream systems | Must Have |
| VAL-007 | Flag low-confidence traces for human review | Must Have |
| VAL-008 | Maintain provenance graph in graph database | Should Have |

**Validation Checks:**

| Check | Description | Failure Action |
|-------|-------------|----------------|
| Chain Completeness | All pipeline stages have output | Block publication |
| Syntax Validity | Artifact parses without errors | Block publication |
| Confidence Threshold | Trace confidence ≥ configured minimum | Flag for review |
| Semantic Alignment | Artifact semantics match clause intent | Flag for review |
| Cross-Reference Integrity | Referenced clauses exist | Warn |

**Output Schema:**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ValidationResult",
  "type": "object",
  "required": ["trace_id", "source_clause", "generated_artifact", "validation_status"],
  "properties": {
    "trace_id": { "type": "string" },
    "source_clause": { "type": "string" },
    "generated_artifact": { "type": "string" },
    "validation_status": {
      "type": "string",
      "enum": ["passed", "failed", "needs_review"]
    },
    "confidence_score": { 
      "type": "number",
      "minimum": 0,
      "maximum": 1
    },
    "validation_checks": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "check_name": { "type": "string" },
          "passed": { "type": "boolean" },
          "message": { "type": "string" }
        }
      }
    },
    "lineage": {
      "type": "object",
      "properties": {
        "document_id": { "type": "string" },
        "section_id": { "type": "string" },
        "chunk_id": { "type": "string" },
        "clause_id": { "type": "string" },
        "mapping_id": { "type": "string" },
        "artifact_id": { "type": "string" }
      }
    },
    "review_flags": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "validated_at": { "type": "string", "format": "date-time" },
    "validated_by": { "type": "string" }
  }
}
```

**Dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| `neo4j` | ≥5.0.0 | Graph provenance storage |
| `arangodb` | ≥3.11.0 | Alternative graph store |
| `jsonschema` | ≥4.19.0 | Schema validation |

---

## 4. API Specification

### 4.1 REST API

**Base URL:** `https://api.aegislang.io/v1`

**Authentication:** Bearer token (JWT) or API key header

#### 4.1.1 Endpoints

##### POST /ingest

Upload and parse a new policy document.

**Request:**
```http
POST /v1/ingest HTTP/1.1
Content-Type: multipart/form-data
Authorization: Bearer {token}

file: [binary]
metadata: {
  "document_name": "AML Policy 2025",
  "document_type": "regulation",
  "jurisdiction": "US",
  "effective_date": "2025-01-01"
}
```

**Response:**
```json
{
  "status": "accepted",
  "job_id": "ing_7f3a9c2b",
  "doc_id": "AML2025_01",
  "estimated_completion": "2025-01-09T15:30:00Z",
  "webhook_url": "https://api.aegislang.io/v1/jobs/ing_7f3a9c2b"
}
```

##### GET /documents/{doc_id}

Retrieve document metadata and processing status.

##### GET /clauses/{doc_id}

List all parsed clauses for a document.

##### GET /rules/{clause_id}

Retrieve generated rule artifact for a clause.

**Response:**
```json
{
  "clause_id": "AML2025_01_2",
  "artifacts": [
    {
      "format": "yaml",
      "file_path": "/artifacts/AML2025_01_2.yaml",
      "download_url": "https://...",
      "confidence": 0.94
    }
  ],
  "trace": {
    "trace_id": "TRC_9832",
    "validation_status": "passed"
  }
}
```

##### POST /compile

Trigger full compilation pipeline for a document.

**Request:**
```json
{
  "doc_id": "AML2025_01",
  "output_formats": ["yaml", "sql"],
  "target_schema": "banking_schema_v3",
  "confidence_threshold": 0.85
}
```

##### GET /trace/{trace_id}

View trace and provenance graph.

##### GET /jobs/{job_id}

Check async job status.

##### POST /schemas

Register or update a target schema.

##### GET /health

Health check endpoint.

### 4.2 Response Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Resource created |
| 202 | Accepted (async processing) |
| 400 | Bad request |
| 401 | Unauthorized |
| 404 | Resource not found |
| 422 | Validation error |
| 429 | Rate limited |
| 500 | Server error |

### 4.3 Rate Limits

| Tier | Requests/min | Documents/day | Artifacts/day |
|------|--------------|---------------|---------------|
| Free | 10 | 5 | 50 |
| Pro | 100 | 100 | 2,000 |
| Enterprise | 1,000 | Unlimited | Unlimited |

---

## 5. Data Flow Pipeline

### 5.1 Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AEGISLANG PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌──────────────┐
     │   Document   │  PDF, DOCX, MD, HTML
     │    Source    │
     └──────┬───────┘
            │
            ▼
┌──────────────────────┐     ┌─────────────────┐
│   Document Ingestor  │────▶│  Vector Store   │  Embeddings for
│       (L1)           │     │   (Pinecone)    │  semantic chunking
└──────────┬───────────┘     └─────────────────┘
           │
           │ policy.ingested
           ▼
┌──────────────────────┐     ┌─────────────────┐
│  Policy Parser Agent │────▶│    LLM API      │  Claude/GPT for
│       (L2)           │     │  (Anthropic)    │  clause extraction
└──────────┬───────────┘     └─────────────────┘
           │
           │ policy.parsed
           ▼
┌──────────────────────┐     ┌─────────────────┐
│ Schema Mapping Agent │────▶│ Schema Registry │  Entity-to-field
│       (L3)           │     │  (PostgreSQL)   │  mappings
└──────────┬───────────┘     └─────────────────┘
           │
           │ policy.mapped
           ▼
┌──────────────────────┐     ┌─────────────────┐
│   Compiler Agent     │────▶│   Templates     │  Jinja2 templates
│       (L4)           │     │   (File System) │  for each format
└──────────┬───────────┘     └─────────────────┘
           │
           │ policy.compiled
           ▼
┌──────────────────────┐     ┌─────────────────┐
│ Trace Validator Agent│────▶│ Provenance Graph│  Lineage tracking
│       (L5)           │     │     (Neo4j)     │
└──────────┬───────────┘     └─────────────────┘
           │
           │ policy.validated
           ▼
     ┌─────┴─────┐
     │           │
     ▼           ▼
┌─────────┐ ┌─────────┐
│Artifact │ │  Audit  │
│  Repo   │ │Dashboard│
└─────────┘ └─────────┘
```

### 5.2 NatLangChain Orchestration

Each pipeline transition is managed by NatLangChain's semantic routing graph, which:

1. Maintains semantic context between agents
2. Routes based on clause type and complexity
3. Handles branching for multi-format output
4. Manages retry logic and error recovery
5. Provides observability hooks

---

## 6. Configuration Specification

### 6.1 Configuration File: `config.yaml`

```yaml
# AegisLang Configuration
# Version: 1.0.0

# ─────────────────────────────────────────────────────────────────────────────
# LLM Configuration
# ─────────────────────────────────────────────────────────────────────────────
llm:
  provider: "anthropic"                    # anthropic | openai | local
  model: "claude-sonnet-4-5-20250929"      # Model identifier
  api_key_env: "ANTHROPIC_API_KEY"         # Environment variable for API key
  temperature: 0.1                         # Low temp for consistent extraction
  max_tokens: 4096                         # Maximum response tokens
  timeout_seconds: 60                      # Request timeout
  retry_attempts: 3                        # Retry on transient failures

# ─────────────────────────────────────────────────────────────────────────────
# Embedding Configuration
# ─────────────────────────────────────────────────────────────────────────────
embeddings:
  model: "text-embedding-3-large"          # Embedding model
  dimensions: 3072                         # Vector dimensions
  batch_size: 100                          # Batch size for bulk embedding

# ─────────────────────────────────────────────────────────────────────────────
# Vector Store Configuration
# ─────────────────────────────────────────────────────────────────────────────
vector_store:
  provider: "pinecone"                     # pinecone | weaviate | qdrant
  index_name: "aegislang-policies"
  environment: "us-east-1"
  api_key_env: "PINECONE_API_KEY"

# ─────────────────────────────────────────────────────────────────────────────
# Database Configuration
# ─────────────────────────────────────────────────────────────────────────────
database:
  url_env: "DATABASE_URL"                  # PostgreSQL connection string
  pool_size: 10
  max_overflow: 20

# ─────────────────────────────────────────────────────────────────────────────
# Graph Database Configuration (Provenance)
# ─────────────────────────────────────────────────────────────────────────────
graph_store:
  provider: "neo4j"                        # neo4j | arangodb
  uri_env: "NEO4J_URI"
  user_env: "NEO4J_USER"
  password_env: "NEO4J_PASSWORD"

# ─────────────────────────────────────────────────────────────────────────────
# Artifact Output Configuration
# ─────────────────────────────────────────────────────────────────────────────
artifacts:
  output_dir: "./artifacts/"
  formats:
    - yaml
    - sql
    - python
  naming_pattern: "{doc_id}_{clause_id}.{format}"

# ─────────────────────────────────────────────────────────────────────────────
# Validation Configuration
# ─────────────────────────────────────────────────────────────────────────────
validation:
  confidence_threshold: 0.85               # Minimum confidence for auto-approval
  review_threshold: 0.70                   # Below this, flag for human review
  block_threshold: 0.50                    # Below this, block artifact

# ─────────────────────────────────────────────────────────────────────────────
# Chunking Configuration
# ─────────────────────────────────────────────────────────────────────────────
chunking:
  target_tokens: 768                       # Target chunk size
  min_tokens: 256                          # Minimum chunk size
  max_tokens: 1024                         # Maximum chunk size
  overlap_tokens: 64                       # Overlap between chunks

# ─────────────────────────────────────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────────────────────────────────────
logging:
  level: "INFO"                            # DEBUG | INFO | WARNING | ERROR
  format: "json"                           # json | text
  output: "stdout"                         # stdout | file
  file_path: "./logs/aegislang.log"

# ─────────────────────────────────────────────────────────────────────────────
# Agent-OS Integration
# ─────────────────────────────────────────────────────────────────────────────
agent_os:
  enabled: true
  event_bus: "redis"                       # redis | kafka | rabbitmq
  redis_url_env: "REDIS_URL"
  node_prefix: "aegislang"

# ─────────────────────────────────────────────────────────────────────────────
# API Server Configuration
# ─────────────────────────────────────────────────────────────────────────────
server:
  host: "0.0.0.0"
  port: 8080
  workers: 4
  cors_origins:
    - "https://app.aegislang.io"
    - "http://localhost:3000"
```

### 6.2 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for LLM |
| `OPENAI_API_KEY` | Conditional | OpenAI API key (if using OpenAI) |
| `PINECONE_API_KEY` | Conditional | Pinecone API key |
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `NEO4J_URI` | Conditional | Neo4j connection URI |
| `NEO4J_USER` | Conditional | Neo4j username |
| `NEO4J_PASSWORD` | Conditional | Neo4j password |
| `REDIS_URL` | Conditional | Redis URL for event bus |
| `JWT_SECRET` | Yes | Secret for JWT signing |
| `LOG_LEVEL` | No | Override log level |

---

## 7. Performance Requirements

### 7.1 Throughput Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Document ingestion | < 30 seconds | Time from upload to `policy.ingested` |
| Clause parsing | < 5 seconds/clause | Average LLM inference + processing |
| Full pipeline (50-page doc) | < 10 minutes | End-to-end for typical regulation |
| Artifact generation | < 1 second/artifact | Template rendering time |
| API response (GET) | < 200ms | P95 latency |
| API response (POST) | < 500ms | P95 latency (sync), immediate (async) |

### 7.2 Scalability Targets

| Metric | Minimum | Target |
|--------|---------|--------|
| Concurrent document processing | 10 | 100 |
| Clauses stored | 100,000 | 1,000,000 |
| Artifacts stored | 500,000 | 5,000,000 |
| Daily ingestion volume | 100 documents | 1,000 documents |

### 7.3 Availability

| Metric | Target |
|--------|--------|
| Uptime | 99.9% |
| RTO (Recovery Time Objective) | < 1 hour |
| RPO (Recovery Point Objective) | < 15 minutes |

---

## 8. Security Specification

### 8.1 Authentication & Authorization

| Method | Use Case |
|--------|----------|
| JWT Bearer Token | API access, session management |
| API Key | Service-to-service, CI/CD |
| OAuth 2.0 / OIDC | Enterprise SSO integration |

**Role-Based Access Control (RBAC):**

| Role | Permissions |
|------|-------------|
| `viewer` | Read documents, clauses, artifacts |
| `operator` | Above + trigger compilations |
| `editor` | Above + upload documents, modify schemas |
| `admin` | Above + manage users, configure system |

### 8.2 Data Security

| Control | Implementation |
|---------|----------------|
| Encryption at rest | AES-256 for database, artifact storage |
| Encryption in transit | TLS 1.3 for all connections |
| Secret management | HashiCorp Vault or AWS Secrets Manager |
| PII handling | Configurable redaction in logs, optional tokenization |

### 8.3 Audit Logging

All operations emit structured audit logs:

```json
{
  "timestamp": "2025-01-09T15:30:00Z",
  "event_type": "document.ingested",
  "user_id": "usr_abc123",
  "resource_id": "AML2025_01",
  "action": "create",
  "ip_address": "192.168.1.1",
  "user_agent": "AegisLang-CLI/1.0",
  "outcome": "success"
}
```

### 8.4 Compliance Considerations

| Standard | Relevance |
|----------|-----------|
| SOC 2 Type II | Service organization controls |
| GDPR | If processing EU policy documents |
| HIPAA | If processing healthcare regulations |

---

## 9. Deployment Specification

### 9.1 Deployment Options

#### 9.1.1 Containerized (Docker)

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8080"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  aegislang:
    build: .
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgresql://aegislang:password@db/aegis
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    volumes:
      - pgdata:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=aegislang
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=aegis

  redis:
    image: redis:7-alpine

  neo4j:
    image: neo4j:5
    ports:
      - "7474:7474"
      - "7687:7687"

volumes:
  pgdata:
```

#### 9.1.2 Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aegislang
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aegislang
  template:
    metadata:
      labels:
        app: aegislang
    spec:
      containers:
        - name: aegislang
          image: aegislang/core:latest
          ports:
            - containerPort: 8080
          envFrom:
            - secretRef:
                name: aegislang-secrets
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "2000m"
```

#### 9.1.3 Serverless (AWS)

- **Lambda Functions:** Individual agents as Lambda handlers
- **SQS:** Event bus for `policy.*` topics
- **S3:** Document and artifact storage
- **RDS:** PostgreSQL for schema registry
- **Neptune:** Graph database for provenance

### 9.2 CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: AegisLang CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install ruff black mypy
      - run: ruff check .
      - run: black --check .
      - run: mypy .

  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -r requirements-dev.txt
      - run: pytest --cov=aegislang --cov-report=xml
      - uses: codecov/codecov-action@v4

  build:
    needs: [lint, test]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@v5
        with:
          push: ${{ github.ref == 'refs/heads/main' }}
          tags: aegislang/core:${{ github.sha }}

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/aegislang \
            aegislang=aegislang/core:${{ github.sha }}
```

### 9.3 Human-in-the-Loop Approval

For production rule deployment:

1. Artifacts with confidence < 0.85 require manual approval
2. GitHub Actions workflow pauses at `approval` environment
3. Compliance officer reviews flagged artifacts
4. Approval triggers final deployment to policy engine

---

## 10. Testing Specification

### 10.1 Test Strategy

| Level | Coverage Target | Tools |
|-------|-----------------|-------|
| Unit Tests | 90% | pytest, pytest-cov |
| Integration Tests | 80% | pytest, testcontainers |
| E2E Tests | Critical paths | pytest, playwright |
| Contract Tests | API endpoints | schemathesis |
| Performance Tests | Key flows | locust |

### 10.2 Test Data

**Sample Policy Documents:**

| Document | Domain | Complexity | Clauses |
|----------|--------|------------|---------|
| `test_aml_basic.pdf` | AML | Low | 15 |
| `test_gdpr_full.pdf` | Privacy | High | 150 |
| `test_sox_404.docx` | Financial | Medium | 50 |
| `test_pci_dss.md` | Security | Medium | 80 |

### 10.3 Evaluation Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Clause Accuracy** | % of correctly extracted clause structures | ≥ 95% |
| **Mapping Precision** | % of entity-to-schema matches verified correct | ≥ 90% |
| **Code Validity** | % of emitted artifacts that execute/parse successfully | 100% |
| **Trace Integrity** | % of generated trace links with confidence ≥ 0.8 | ≥ 95% |
| **Pipeline Success** | % of documents completing full pipeline | ≥ 98% |

### 10.4 Test Examples

```python
# tests/test_policy_parser.py

import pytest
from aegislang.agents.policy_parser_agent import PolicyParserAgent

class TestPolicyParser:
    
    @pytest.fixture
    def parser(self):
        return PolicyParserAgent()
    
    def test_parse_obligation_clause(self, parser):
        clause = "Financial institutions must verify customer identity before account creation."
        
        result = parser.parse(clause)
        
        assert result.type == "obligation"
        assert result.actor.entity == "financial institutions"
        assert result.action.verb == "verify"
        assert result.object.entity == "customer identity"
        assert result.condition.trigger == "before account creation"
        assert result.confidence >= 0.85
    
    def test_parse_prohibition_clause(self, parser):
        clause = "Employees shall not access customer data without authorization."
        
        result = parser.parse(clause)
        
        assert result.type == "prohibition"
        assert result.actor.entity == "employees"
    
    @pytest.mark.parametrize("clause,expected_type", [
        ("Banks must report suspicious transactions.", "obligation"),
        ("Trading on insider information is prohibited.", "prohibition"),
        ("Customers may request data deletion.", "permission"),
    ])
    def test_clause_type_detection(self, parser, clause, expected_type):
        result = parser.parse(clause)
        assert result.type == expected_type
```

---

## 11. Dependency Matrix

### 11.1 Python Dependencies

```
# requirements.txt

# Core Framework
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0

# Document Processing
unstructured>=0.10.0
pdfminer.six>=20221105
python-docx>=1.0.0
beautifulsoup4>=4.12.0
tiktoken>=0.5.0

# NLP & ML
spacy>=3.7.0
transformers>=4.35.0
sentence-transformers>=2.2.0
torch>=2.1.0

# LLM Integration
anthropic>=0.8.0
openai>=1.3.0

# Vector Stores
pinecone-client>=2.2.0
weaviate-client>=3.24.0

# Databases
sqlalchemy>=2.0.0
asyncpg>=0.29.0
neo4j>=5.0.0
redis>=5.0.0

# Template & Compilation
jinja2>=3.1.0
pyyaml>=6.0.0
sqlparse>=0.4.0

# Utilities
httpx>=0.25.0
tenacity>=8.2.0
structlog>=23.2.0

# NatLangChain / Agent-OS
natlangchain>=0.1.0
agent-os-sdk>=0.1.0
```

### 11.2 Infrastructure Dependencies

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Relational DB | PostgreSQL | 15+ | Schema registry, metadata |
| Graph DB | Neo4j | 5+ | Provenance graphs |
| Vector Store | Pinecone | — | Semantic search |
| Cache/Events | Redis | 7+ | Event bus, caching |
| Object Storage | S3/MinIO | — | Document & artifact storage |

---

## 12. Roadmap

### 12.1 Current Version (1.0)

- [x] Core pipeline (Ingest → Parse → Map → Compile → Validate)
- [x] PDF, DOCX, Markdown ingestion
- [x] YAML, SQL, Python artifact generation
- [x] Basic provenance tracking
- [x] REST API

### 12.2 Version 1.1 (Q2 2025)

- [ ] **Rule Drift Monitor:** Detect policy updates and auto-diff against existing artifacts
- [ ] **Batch Processing:** Parallel ingestion of document sets
- [ ] OPA/Rego output format
- [ ] Terraform output format

### 12.3 Version 1.2 (Q3 2025)

- [ ] **Interactive Audit Explorer:** Web UI for clause-rule lineage visualization
- [ ] **RAG Integration:** Retrieval from external regulation databases (OECD, EU Lex)
- [ ] Webhook notifications for pipeline events

### 12.4 Version 2.0 (Q4 2025)

- [ ] **Multilingual Compliance:** EU directives, ISO standards in multiple languages
- [ ] **Semantic Diff Engine:** Compare regulation versions and propagate changes
- [ ] **Compliance Simulation:** Test artifact behavior before deployment
- [ ] Self-hosted LLM support (Llama, Mistral)

---

## 13. Glossary

| Term | Definition |
|------|------------|
| **Artifact** | Generated executable output (YAML rule, SQL check, Python test) |
| **Clause** | A single regulatory statement extracted from policy text |
| **Confidence Score** | Numeric measure (0-1) of system certainty in extraction/mapping |
| **Entity** | A noun phrase representing an actor, object, or concept in policy |
| **Lineage** | The complete traceability chain from source document to artifact |
| **Mapping** | Association between a policy entity and a system schema field |
| **NatLangChain** | Semantic routing framework for multi-agent orchestration |
| **Provenance** | Audit trail documenting the origin and transformation of data |
| **Schema Registry** | Catalog of target system schemas available for entity mapping |
| **Trace** | A validated provenance record linking clause to artifact |

---

## 14. Appendices

### Appendix A: Sample Pipeline Execution

**Input Document:** `AML_Policy_2025.pdf`

**Stage 1: Ingestion**
```json
{
  "doc_id": "AML2025_01",
  "sections": [
    {
      "section_id": "AML2025_01_S3",
      "section_title": "Customer Due Diligence",
      "text_chunks": [
        {
          "chunk_id": "AML2025_01_S3_C1",
          "text": "Financial institutions must verify customer identity before account creation. Records must be maintained for at least 5 years."
        }
      ]
    }
  ]
}
```

**Stage 2: Parsing**
```json
[
  {
    "clause_id": "AML2025_01_S3_C1_1",
    "type": "obligation",
    "actor": { "entity": "financial institutions" },
    "action": { "verb": "verify" },
    "object": { "entity": "customer identity" },
    "condition": { "trigger": "before account creation" },
    "confidence": 0.94
  },
  {
    "clause_id": "AML2025_01_S3_C1_2",
    "type": "obligation",
    "actor": { "entity": "financial institutions" },
    "action": { "verb": "maintain" },
    "object": { "entity": "records" },
    "temporal_scope": { "duration": "at least 5 years" },
    "confidence": 0.91
  }
]
```

**Stage 3: Mapping**
```json
{
  "clause_id": "AML2025_01_S3_C1_1",
  "mapped_entities": [
    {
      "source_entity": "customer identity",
      "target_path": "kyc.check_identity",
      "confidence": 0.92
    },
    {
      "source_entity": "financial institutions",
      "target_path": "org.institution_id",
      "confidence": 0.88
    }
  ]
}
```

**Stage 4: Compilation (YAML)**
```yaml
# Source: Financial institutions must verify customer identity before accoun...
# Clause ID: AML2025_01_S3_C1_1

control:
  id: AML2025_01_S3_C1_1
  type: obligation
  rule: "Verify customer identity"
  
  actor:
    entity: "financial institutions"
    mapped_to: "org.institution_id"
  
  action:
    operation: "kyc.check_identity"
    parameters:
      - user_id: "user_table.user_id"
  
  trigger:
    event: "on_account_open"
  
  severity: critical
  
  metadata:
    source_document: "AML2025_01"
    confidence: 0.92
    generated_by: "aegislang-compiler-v1.0.0"
```

**Stage 5: Validation**
```json
{
  "trace_id": "TRC_9832",
  "source_clause": "AML2025_01_S3_C1_1",
  "generated_artifact": "artifacts/AML2025_01_S3_C1_1.yaml",
  "validation_status": "passed",
  "confidence_score": 0.92,
  "validation_checks": [
    { "check_name": "chain_completeness", "passed": true },
    { "check_name": "syntax_validity", "passed": true },
    { "check_name": "confidence_threshold", "passed": true }
  ]
}
```

---

### Appendix B: Error Codes

| Code | Message | Resolution |
|------|---------|------------|
| `AEGIS-001` | Document parsing failed | Check file format, retry with alternate parser |
| `AEGIS-002` | LLM rate limit exceeded | Wait and retry, or increase quota |
| `AEGIS-003` | Schema not found | Register target schema before compilation |
| `AEGIS-004` | Confidence below threshold | Review flagged clause, adjust manually |
| `AEGIS-005` | Artifact syntax invalid | Check template, review mapped entities |
| `AEGIS-006` | Provenance chain broken | Reprocess from last valid stage |

---

*End of Specification*
