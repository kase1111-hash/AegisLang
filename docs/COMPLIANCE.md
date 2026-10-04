# AegisLang Compliance Review

This document reviews AegisLang against major regulatory frameworks including GDPR, HIPAA, SOX, and PCI-DSS. It describes what the software (v0.1.0, Alpha) actually provides and what the deploying organization has to supply itself.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [GDPR Compliance](#gdpr-compliance)
- [HIPAA Compliance](#hipaa-compliance)
- [SOX Compliance](#sox-compliance)
- [PCI-DSS Compliance](#pci-dss-compliance)
- [Security Controls](#security-controls)
- [Data Handling Practices](#data-handling-practices)
- [Recommendations](#recommendations)

---

## Executive Summary

AegisLang is designed as a **self-hosted solution** that processes policy documents to generate compliance artifacts. Running it on your own infrastructure provides some privacy benefits, but several controls that compliance frameworks expect (TLS, encryption at rest, per-user access control, audit logging, retention) are **not built in** and must be provided by the deployment:

| Aspect | Status | Notes |
|--------|--------|-------|
| Data Residency | ✅ Self-hosted | Data stays on infrastructure you run, except LLM calls (below) |
| Data Processing | ✅ Compliant | No data sent to AegisLang servers (there are none) |
| Third-Party Sharing | ⚠️ Conditional | If `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is set, clause text is sent to that provider; with neither set, the local mock parser is used |
| Audit Logging | ⚠️ Partial | Per-document traceability via `GET /api/v1/trace/{doc_id}` plus structured logs with request IDs; no user-level audit log |
| Access Control | ⚠️ Basic | Shared API keys (`X-API-Key`); no user accounts or roles |
| Encryption | ❌ Not built in | No TLS or encryption at rest in the application; use a TLS reverse proxy and encrypted storage |

---

## GDPR Compliance

### General Data Protection Regulation (EU)

#### Article 5 - Principles

| Principle | Implementation | Status |
|-----------|----------------|--------|
| **Lawfulness, fairness, transparency** | Processing only occurs when explicitly initiated by a user (API call or CLI run) | ✅ |
| **Purpose limitation** | Data used only for policy compilation | ✅ |
| **Data minimization** | The full text of every uploaded document is stored (sections and chunks); nothing is redacted | ⚠️ |
| **Accuracy** | SHA-256 content hash recorded in document metadata (`metadata.hash`) | ✅ |
| **Storage limitation** | Only jobs expire (`AEGISLANG_JOB_TTL_SECONDS`, default 24 h). Documents, clauses, artifacts and traces are kept until deleted manually (memory backend: until restart) | ⚠️ |
| **Integrity & confidentiality** | No TLS or encryption at rest in the application; provide a TLS reverse proxy and disk/volume encryption | ⚠️ Deployer |

#### Article 17 - Right to Erasure

There is **no DELETE endpoint** in the API. Data must be removed directly from storage:

- **Memory backend** (`AEGISLANG_STORAGE_BACKEND=memory`, the default): data exists only in the server process; restarting the server erases it.
- **SQLite backend** (`AEGISLANG_STORAGE_BACKEND=sqlite`, used by `docker compose`): delete the rows from the database file at `AEGISLANG_SQLITE_PATH` (`/app/data/aegislang.db` on the `aegislang-data` volume in Docker Compose), or delete the whole file / volume (`make docker-down-clean` removes the volume). The tables are simple key/value tables keyed by `doc_id`:

```sql
-- sqlite3 /app/data/aegislang.db   (preferably while the server is stopped)
DELETE FROM documents      WHERE key = 'POLICY_66C62B';
DELETE FROM clauses        WHERE key = 'POLICY_66C62B';
DELETE FROM clauses_keys   WHERE key = 'POLICY_66C62B';
DELETE FROM artifacts      WHERE key = 'POLICY_66C62B';
DELETE FROM artifacts_keys WHERE key = 'POLICY_66C62B';
DELETE FROM traces         WHERE key = 'POLICY_66C62B';
-- Job records reference the doc_id inside their JSON value; they also expire after the job TTL
DELETE FROM jobs WHERE value LIKE '%"POLICY_66C62B"%';
VACUUM;
```

Also remove any artifacts written to disk by the CLI tools, and any copies in your logs or log platform (document IDs appear in log events and request paths). Uploaded files themselves are written to a private temporary directory, overwritten with random bytes and deleted after ingestion.

**Status:** ⚠️ Manual - erasure is possible but requires direct storage access

#### Article 25 - Data Protection by Design

| Measure | Implementation |
|---------|----------------|
| Pseudonymization | Not applied. API document IDs are derived from the uploaded file name (sanitized, uppercased stem plus a random 6-hex suffix, e.g. `policy.md` -> `POLICY_66C62B`), and section/chunk/clause IDs build on it. If file names contain personal data, so do the IDs and logs; rename files before upload. |
| Encryption | Not provided by the application: terminate TLS at a reverse proxy and store the SQLite file on an encrypted disk or volume |
| Access logging | Uvicorn access log (method, path, status, client address) and structured application logs with request IDs; no user identity, because API keys are shared |
| Minimal data | Only policy text is processed, but it is stored in full |

**Status:** ⚠️ Partial - depends on deployment measures

#### Article 30 - Records of Processing

AegisLang does not maintain a record-of-processing register or an audit-log table. What it records is per-document **traceability**:

- `GET /api/v1/documents/{doc_id}` - document metadata (`source_file`, `ingestion_timestamp`, `document_type`, `hash`, plus any user-supplied metadata)
- `GET /api/v1/trace/{doc_id}` - validation results (each with a lineage of document, section, chunk, clause, mapping and artifact IDs) and the provenance graph (document -> section -> chunk -> clause -> artifact nodes, linked by `CONTAINS_SECTION`, `CONTAINS_CHUNK`, `PARSED_TO` and `COMPILED_TO` edges) for the last compile of that document; persisted when the SQLite backend is used
- Structured log events such as `ingestion_completed` and `compilation_completed` (with `job_id`, `doc_id` and the request ID)

Example lineage entry from a `/trace` validation result:

```json
"lineage": {
  "document_id": "FINCEN_CDD_RULE_09FC62",
  "section_id": "FINCEN_CDD_RULE_S009",
  "chunk_id": "FINCEN_CDD_RULE_S009_C000",
  "clause_id": "FINCEN_CDD_RULE_09FC62_FINCEN_CDD_RULE_S009_C000_CL001",
  "mapping_id": "map_FINCEN_CDD_RULE_09FC62_FINCEN_CDD_RULE_S009_C000_CL001",
  "artifact_id": "FINCEN_CDD_RULE_09FC62_FINCEN_CDD_RULE_S009_C000_CL001_yaml"
}
```

**Status:** ⚠️ Partial - artifact lineage is recorded; the controller must keep its own Article 30 register

#### Article 32 - Security of Processing

| Control | Status |
|---------|--------|
| Encryption of data | ❌ Not built in (deployer: TLS proxy, disk encryption) |
| Pseudonymization | ⚠️ Not applied (IDs derive from file names) |
| Confidentiality | ⚠️ Shared API keys; no per-user or per-document access control |
| Integrity | ✅ Content hashing (SHA-256) |
| Availability | ⚠️ Single instance only (in-memory rate limiter, local SQLite file); no HA. Back up the SQLite volume |
| Regular testing | ⚠️ Requires customer implementation |

---

## HIPAA Compliance

### Health Insurance Portability and Accountability Act (US)

#### Administrative Safeguards (§164.308)

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| Security Management | No roles; every API key has the same access | ❌ |
| Workforce Security | API key authentication (on unless `AEGISLANG_DISABLE_AUTH=true`) | ⚠️ Shared keys |
| Information Access | No per-document or per-schema access control | ❌ |
| Security Awareness | Documentation provided | ✅ |
| Contingency Plan | Back up the `aegislang-data` volume (SQLite file); no built-in backup | ⚠️ |
| Evaluation | Structured logs and `/trace` provenance available for review | ⚠️ |

#### Technical Safeguards (§164.312)

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| Access Control | API key authentication (X-API-Key header); no unique user identification | ⚠️ |
| Audit Controls | Structured logs with request IDs and Uvicorn access log; no user attribution | ⚠️ |
| Integrity Controls | Content hashing | ✅ |
| Transmission Security | No TLS in the application; terminate TLS at a reverse proxy | ⚠️ Deployer |

#### Physical Safeguards (§164.310)

| Requirement | Notes |
|-------------|-------|
| Facility Access | Customer responsibility (self-hosted) |
| Workstation Use | Customer responsibility |
| Device Controls | Docker isolation provides separation |

**PHI Handling Note:**
AegisLang does not detect or redact PHI. Whatever is in an uploaded document is stored as-is (document text, clauses, artifacts). If policy documents contain PHI:
1. Run without LLM API keys (the mock parser runs locally), or use a cloud LLM under a BAA - AegisLang has no on-premises LLM integration
2. Store the SQLite file on an encrypted volume
3. Delete documents manually when no longer needed (there is no automatic retention)
4. Implement BAA with LLM provider if using cloud APIs
5. Keep PHI out of file names (document IDs are derived from them)

---

## SOX Compliance

### Sarbanes-Oxley Act (US)

#### Section 302 - Corporate Responsibility

| Requirement | Implementation |
|-------------|----------------|
| Internal controls documentation | YAML/SQL artifacts provide reviewable rules |
| Control effectiveness | Validation layer (L5) checks provenance, syntax and confidence of artifacts |
| Disclosure controls | Lineage from policy clause to artifact (`GET /api/v1/trace/{doc_id}`) |

**Status:** ✅ AegisLang can support SOX work by generating draft control artifacts with traceability; generated artifacts require human review before use

#### Section 404 - Internal Control Assessment

| Control Area | AegisLang Support |
|--------------|-------------------|
| Control documentation | Generated from policy documents (review required) |
| Testing evidence | Generated pytest stubs (Python) that check placeholder fields |
| Change management | Not built in - commit generated artifacts to version control yourself |
| Audit trail | Clause-to-artifact lineage via `/trace` |

**Artifact Example** (generated by `python examples/run_aml_pipeline.py`, from `examples/output/fincen_cdd_rule/`; blank lines removed):
```yaml
# Source: Financial institutions must maintain records of customer identification informat...
# Clause ID: FINCEN_CDD_RULE_09FC62_FINCEN_CDD_RULE_S009_C000_CL001
# Generated: 2026-10-04T20:17:09.752576+00:00
# Confidence: 0.5
control:
  id: FINCEN_CDD_RULE_09FC62_FINCEN_CDD_RULE_S009_C000_CL001
  type: obligation
  rule: "maintain records"
  actor:
    entity: "Financial institutions"
  action:
    operation: "maintain"
  object:
    entity: "records"
  trigger:
    event: "account closure"
  temporal_scope:
    duration: "five years"
  severity: high
  metadata:
    source_document: "FINCEN_CDD_RULE_09FC62"
    confidence: 0.5
    generated_by: "aegislang-compiler-v0.1.0"
```

The YAML itself carries the clause ID and source document; the full lineage (section, chunk, mapping, artifact) is in the `/trace` validation result shown above.

---

## PCI-DSS Compliance

### Payment Card Industry Data Security Standard

#### Requirement 1: Network Security

| Sub-requirement | Status | Notes |
|-----------------|--------|-------|
| 1.1 Firewall configuration | N/A | Customer infrastructure |
| 1.2 Router configuration | N/A | Customer infrastructure |
| 1.3 DMZ implementation | N/A | Customer infrastructure (Docker Compose publishes port 8080 on the host; place it behind a proxy in a segmented network) |

#### Requirement 3: Protect Stored Data

| Sub-requirement | Implementation |
|-----------------|----------------|
| 3.1 Minimize data storage | Only policy text stored |
| 3.2 No sensitive auth data | Not applicable |
| 3.4 Render PAN unreadable | Not applicable (no PAN storage) |

#### Requirement 6: Secure Development

| Sub-requirement | Status |
|-----------------|--------|
| 6.1 Vulnerability identification | ⚠️ Trivy filesystem scan in CI on pushes to `main`/`develop` (not on pull requests, non-blocking); `make security-check` (Bandit + Safety) run manually |
| 6.2 Security patches | ⚠️ No automated dependency updates (no Dependabot); base image `python:3.11-slim` refreshed on rebuild |
| 6.3 Secure development | ⚠️ Pull-request review (enforcement depends on repository settings) |
| 6.4 Change control | ✅ Git-based versioning |
| 6.5 Common vulnerabilities | ✅ Pydantic input validation, parameterized SQLite queries, sandboxed Jinja2 templates, upload name/extension/size checks |

#### Requirement 10: Track Access

| Sub-requirement | Implementation |
|-----------------|----------------|
| 10.1 Audit trail | ⚠️ No audit-log table; document provenance via `GET /api/v1/trace/{doc_id}` and structured logs |
| 10.2 Automated audit | ✅ Structured logging (structlog) with request IDs; JSON output when `AEGISLANG_ENV=production` |
| 10.3 Audit entry details | ⚠️ Timestamp, request ID, event, job/document IDs; no user ID (API keys are shared) |
| 10.5 Secure audit trails | ❌ Not provided - logs go to stdout and optionally `AEGISLANG_LOG_FILE`; ship them to a tamper-evident store |

---

## Security Controls

### OWASP Top 10 Mitigation

| Vulnerability | Mitigation | Status |
|---------------|------------|--------|
| A01: Broken Access Control | API key authentication; all keys have equal access (no RBAC) | ⚠️ |
| A02: Cryptographic Failures | No TLS or encryption at rest in the app; provided by the deployment | ⚠️ |
| A03: Injection | Parameterized SQL, Pydantic input validation, sandboxed Jinja2 templates | ✅ |
| A04: Insecure Design | Path-traversal checks on upload names, upload size limit, private temp directory with overwrite-on-delete | ✅ |
| A05: Security Misconfiguration | Auth on by default (a random development key is generated and logged if `AEGISLANG_API_KEYS` is unset - set real keys in production); generic error messages with `AEGISLANG_ENV=production`; CORS limited to `CORS_ORIGINS` | ⚠️ |
| A06: Vulnerable Components | Trivy filesystem scan in CI (non-PR, non-blocking); `make security-check` / `make vuln-scan` locally; no Dependabot | ⚠️ |
| A07: Auth Failures | API key validation; per-key rate limiting (in-memory) | ✅ |
| A08: Software Integrity | Content hashing of ingested documents; container images are not signed | ⚠️ |
| A09: Logging Failures | Structured logging with request IDs, optional Sentry (`SENTRY_DSN`); no audit log | ⚠️ |
| A10: SSRF | No user-controlled URL fetching (outbound calls only to the configured LLM API) | ✅ |

### Security Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Security Layers                       │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: Network          │ Docker isolation, CORS          │
│ Layer 2: Transport        │ TLS at your reverse proxy       │
│ Layer 3: Authentication   │ API keys (X-API-Key header)     │
│ Layer 4: Authorization    │ None (all API keys are equal)   │
│ Layer 5: Input Validation │ Pydantic models, upload checks  │
│ Layer 6: Data Protection  │ Parameterized queries           │
│ Layer 7: Audit            │ Structured logs, optional Sentry│
└─────────────────────────────────────────────────────────────┘
```

---

## Data Handling Practices

### Data Flow

```
User Document → AegisLang (Self-hosted) → Generated Artifacts
                      │
                      ▼ (Only if ANTHROPIC_API_KEY or OPENAI_API_KEY is set)
                 LLM API Call
                 (Anthropic/OpenAI)
```

### Data Categories

| Category | Stored | Transmitted Externally | Retention |
|----------|--------|------------------------|-----------|
| Policy text | Yes (memory or SQLite) | Only to LLM (if configured) | Until deleted manually (memory: until restart) |
| Parsed clauses | Yes | No | Until deleted manually |
| Generated artifacts | Yes | No | Until deleted manually |
| Validation results / provenance graph | Yes | No | Until deleted manually |
| Jobs | Yes | No | `AEGISLANG_JOB_TTL_SECONDS` after completion (default 24 h) |
| Application logs | stdout, optional `AEGISLANG_LOG_FILE` | Error events to Sentry only if `SENTRY_DSN` is set | Managed by your log platform |
| User credentials | No (API keys only in environment variables) | No | N/A |

### Third-Party Data Sharing

| Service | Data Shared | Purpose | Configurable |
|---------|-------------|---------|--------------|
| Anthropic/OpenAI | Clause text | Clause parsing | Yes - unset both keys to use the local mock parser (no on-premises LLM integration) |
| Sentry | Error events and breadcrumbs (`send_default_pii=False`) | Error monitoring | Yes - only enabled when `SENTRY_DSN` is set |

No PostgreSQL, Redis or Neo4j is used by the application. Persistent data lives in a local SQLite file when `AEGISLANG_STORAGE_BACKEND=sqlite`.

---

## Recommendations

### For GDPR Compliance

1. **Data Processing Agreement**: Establish DPA with LLM provider (if an LLM key is configured)
2. **Privacy Policy**: Document AegisLang usage in privacy policy
3. **Retention Policy**: There is no automatic deletion of documents or artifacts; schedule a scripted cleanup of the SQLite rows. Job retention is set with `AEGISLANG_JOB_TTL_SECONDS`
4. **Access Logs**: Retain Uvicorn access logs and structured application logs (`AEGISLANG_LOG_FILE` or a log shipper)

### For HIPAA Compliance

1. **BAA**: Obtain Business Associate Agreement with LLM provider
2. **Avoid PHI transmission**: Run without LLM API keys (mock parser) - AegisLang cannot use an on-premises LLM
3. **Encryption**: Place the SQLite volume on encrypted storage
4. **Access Control**: AegisLang only has shared API keys; add per-user authentication in a reverse proxy or API gateway

### For SOX Compliance

1. **Change Management**: Commit generated artifacts to git and tag versions
2. **Testing**: Run generated test stubs regularly (they test placeholder fields and need adapting)
3. **Documentation**: Keep the `/trace` output alongside the artifacts to preserve artifact-to-policy lineage

### For PCI-DSS Compliance

1. **Network Segmentation**: Deploy in isolated network segment
2. **Access Logging**: Retain access and application logs in a central, tamper-evident store
3. **Vulnerability Scanning**: Run `make security-check` (Bandit + Safety) and `make vuln-scan` regularly from a development checkout (the production image does not include the Makefile)

---

## Compliance Checklist

### Pre-Deployment

- [ ] Review data classification of input documents (and their file names, which become document IDs)
- [ ] Set `AEGISLANG_API_KEYS` (do not rely on the generated development key) and `AEGISLANG_ENV=production`
- [ ] Enable TLS for all endpoints at a reverse proxy
- [ ] Put the SQLite volume on encrypted storage and set up backups
- [ ] Define a manual deletion procedure for documents and artifacts
- [ ] Set up audit log retention
- [ ] Document data flows for privacy impact assessment

### Operational

- [ ] Regular security patch updates
- [ ] Periodic API key review and rotation
- [ ] Audit log review
- [ ] Backup verification
- [ ] Incident response plan

### Annual

- [ ] Penetration testing
- [ ] Compliance audit
- [ ] Policy document review
- [ ] Third-party vendor assessment

---

## Attestation

This compliance review was performed on: **February 2026** (revised **October 2026** to match the code)

Review covers: AegisLang v0.1.0 (Alpha)

Next review due: **before the next release**

---

*This document is for informational purposes. Organizations should conduct their own compliance assessments with qualified legal and security professionals.*
