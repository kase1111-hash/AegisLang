# Vibe-Code Detection Audit v2.0 — AegisLang

**Repository:** `kase1111-hash/AegisLang`
**Audit Date:** 2026-02-26
**Framework:** Vibe-Code Detection Audit v2.0
**Auditor:** Claude Opus 4.6

---

## Executive Summary

| Metric | Value |
|---|---|
| **Weighted Authenticity** | **71.9%** |
| **Vibe-Code Confidence** | **28.1%** |
| **Classification** | **AI-Assisted (16-35%)** |

AegisLang is a well-structured multi-agent semantic compiler with a genuine 5-stage
pipeline that actually functions end-to-end. However, the repository's *provenance*
is overwhelmingly AI-generated: 100% of source code commits originate from
`noreply@anthropic.com` (Claude Code), with the human contributor limited to
specification documents and merging pull requests.

The code quality is moderate-to-strong — error handling is real, security measures
are genuine, and the pipeline is complete — but the surface-level uniformity
(222 section dividers, zero parametrized tests, formulaic naming) and total
absence of human code authorship are unmistakable markers of AI generation.

---

## Domain Scores

| Domain | Weight | Raw Score | Percentage | Weighted |
|---|---|---|---|---|
| **A: Surface Provenance** | 20% | 1.43/3 | 47.6% | 9.5% |
| **B: Behavioral Integrity** | 50% | 2.29/3 | 76.2% | 38.1% |
| **C: Interface Authenticity** | 30% | 2.43/3 | 81.0% | 24.3% |
| **Total** | | | **Weighted Authenticity** | **71.9%** |

**Vibe-Code Confidence = 100% - 71.9% = 28.1%**

---

## Domain A: Surface Provenance (20%)

### A1. Commit History Patterns — Score: 1/3 (Weak)

**Evidence:**
- **43 of 69 commits** authored by `noreply@anthropic.com` (Claude Code)
- **26 human commits** from `kase1111@gmail.com`, but **ALL are merge commits** except 5
- The 5 non-merge human commits: `Initial commit`, `Update README.md`, `Create SPEC.md`, `Create KEYWORDS.md`, `Create Step-by-step.md` — **zero source code**
- Every branch follows `claude/` prefix naming convention
- Commit messages are formulaic: "Add X", "Fix X", "Phase N: Description"
- **Zero reverts**, zero `WIP`, zero frustration markers, zero course corrections
- No organic cadence variation — commits arrive in rapid-fire batches within single sessions

```
$ git log --format="%ae" | sort | uniq -c | sort -rn
     43 noreply@anthropic.com
     26 kase1111@gmail.com
```

**Remediation:** This cannot be remediated retroactively. The commit history is a permanent record.

### A2. Comment Archaeology — Score: 1/3 (Weak)

**Evidence:**
- **222 section dividers** (`# ====` / `# ----`) across 17 files — extreme uniformity
- Every module follows identical structure: module docstring → section divider → class → section divider → next section
- Docstrings are tutorial-style: "Provides X for Y", "Args:", "Returns:"
- **Only 3 TODO/FIXME markers** in entire 12,067-line codebase:
  - `aegis_ingestor.py:695`: `# TODO: Add language detection`
  - `compiler_agent.py:324`: `# TODO: Implement condition testing`
  - `compiler_agent.py:331`: `# TODO: Implement deadline testing`
- **Zero WHY comments** — no comments explaining design rationale or tradeoffs
- Comments uniformly describe *what* code does, never *why*

**Remediation:** Add WHY comments where design decisions were non-obvious. Reduce formulaic section dividers.

### A3. Test Quality Signals — Score: 1/3 (Weak)

**Evidence:**
- 139 tests, all passing — but quantity over depth
- **Zero `@pytest.mark.parametrize`** usage across all test files
- Tests are predominantly happy-path with shallow assertions:
  - `assert response.status_code == 200` (repeated pattern)
  - `assert len(result.sections) > 0` (existence checks, not value checks)
  - `assert "job_id" in data` (key existence, not content validation)
- Test naming is suspiciously uniform: every test follows `test_verb_noun` pattern
- All test files use identical section divider pattern (`# ====`)
- Mock LLM mode means **real LLM integration is completely untested**
- No property-based testing (hypothesis), no fuzzing, no mutation testing
- No flaky test handling or retry mechanisms
- Error-path testing is minimal (only 404 not-found paths)

**Remediation:**
- Add parametrized tests for edge cases
- Add error-path tests (malformed input, timeout scenarios, concurrent access)
- Add property-based tests for parsers and chunkers
- Test with real LLM providers in CI (even with small test documents)

### A4. Import & Dependency Hygiene — Score: 2/3 (Moderate)

**Evidence:**
- All declared dependencies in `requirements.txt` are actually imported
- No wildcard imports found
- No phantom dependencies
- Clean separation: `requirements.txt` (core) vs `requirements-ml.txt` (optional)
- Optional imports properly guarded with try/except: `neo4j`, `sentry-sdk`, `anthropic`, `openai`
- Import ordering follows standard convention (stdlib → third-party → local)

**Minor issue:** `python-multipart` is in requirements but never explicitly imported (it's a FastAPI transitive dependency for form data).

### A5. Naming Consistency — Score: 1/3 (Weak)

**Evidence:**
- **Suspiciously uniform** naming across all 17 Python source files
- Every class uses perfect PascalCase with no abbreviations
- Every function uses perfect snake_case with descriptive names
- Every variable is descriptive — no abbreviated names, no shortcuts, no jargon
- Zero organic variation between files that would suggest different authoring sessions
- No domain-specific shorthand (e.g., `cdd` for Customer Due Diligence, `sar` for Suspicious Activity Report) that would suggest compliance domain expertise
- All files follow identical structural patterns: imports → constants → classes → functions → main

**Contrast with human patterns:** Human code typically shows naming drift over time — earlier files may use shorter names, later files longer ones. Different modules show different naming preferences. Domain experts use abbreviations. None of this variation is present.

### A6. Documentation vs Reality — Score: 2/3 (Moderate)

**Evidence:**
- README pipeline claims are accurate — all 5 stages are implemented
- API endpoints documented in `docs/API.md` match actual server.py endpoints
- Supported file formats (PDF, DOCX, MD, HTML) are all implemented
- **BUT:** Version inconsistency — `VERSION` file says `0.1.0`, `server.py` says `1.0.0`, `CHANGELOG.md` says `1.0.0`
- **BUT:** Terraform and Rego output formats are declared in `ArtifactFormat` enum but have no Jinja2 templates
- **BUT:** 17+ documentation markdown files for a 12,067-line alpha project is excessive — documentation-to-code ratio suggests padding
- **BUT:** Several docs are self-referential audit/evaluation reports generated by prior Claude sessions

### A7. Dependency Utilization — Score: 2/3 (Moderate)

**Evidence:**
- Every dependency serves a clear purpose:
  - `fastapi`/`uvicorn`/`pydantic`: web framework (core to API)
  - `pdfminer.six`/`python-docx`/`beautifulsoup4`: document parsing (core to L1)
  - `tiktoken`: token counting for chunking (core to L1)
  - `anthropic`/`openai`: LLM clients (core to L2-L3)
  - `jinja2`/`pyyaml`/`sqlparse`: compilation (core to L4)
  - `structlog`/`sentry-sdk`: logging (core to infrastructure)
  - `neo4j`: graph database (optional, for L5)
- **BUT:** Several dependencies are behind try/except with mock fallbacks, meaning the default runtime uses ~60% of declared dependencies

### Domain A Summary: 10/21 = 47.6%

---

## Domain B: Behavioral Integrity (50%)

### B1. Error Handling Authenticity — Score: 3/3 (Strong)

**Evidence:**
- Custom exception hierarchy with 8 domain-specific exceptions (`aegislang/core/errors.py:49-157`)
- Each exception carries HTTP status code, machine-readable error code, and structured details
- **Zero bare `except:` clauses** across the entire codebase
- Minimal broad `except Exception:` (4 instances, all justified — Sentry client and last-resort cleanup)
- Critical paths fail closed: background tasks catch exceptions and mark jobs as `FAILED`
- Production error sanitization prevents information leakage (`errors.py:171-193`)
- Global exception handlers registered for all FastAPI routes (`server.py:862`)
- `ErrorHandlingContext` async context manager for consistent error handling (`errors.py:299-358`)

### B2. Configuration Actually Used — Score: 2/3 (Moderate)

**Evidence:**
- `config.yaml` (127 lines) is explicitly documented as "NOT automatically loaded" — it is purely reference documentation
- **6 phantom environment variables** in `.env.example` that are never read by code:
  - `REDIS_URL`, `DATABASE_URL`, `PINECONE_API_KEY`, `JWT_SECRET` — never imported
  - `SERVER_PORT` (code uses `PORT`), `ENVIRONMENT` (code uses `AEGISLANG_ENV`), `LOG_LEVEL` (code uses `AEGISLANG_LOG_LEVEL`)
- **12+ environment variables** actually used by code but missing from `.env.example`:
  - `AEGISLANG_API_KEYS`, `AEGISLANG_DISABLE_AUTH`, `AEGISLANG_STORAGE_BACKEND`, `AEGISLANG_SQLITE_PATH`, `AEGISLANG_MAX_FILE_SIZE`, `AEGISLANG_RATE_LIMIT_MINUTE`, `AEGISLANG_RATE_LIMIT_HOUR`, `AEGISLANG_JOB_TTL_SECONDS`, `AEGISLANG_ENV`, `AEGISLANG_LOG_LEVEL`, `CORS_ORIGINS`, `HOST`, `PORT`, `WORKERS`, `RELOAD`
- Variables that ARE used do produce observable behavior changes

**Remediation:** Synchronize `.env.example` with actual code. Remove phantom entries, add missing entries.

### B3. Call Chain Completeness — Score: 2/3 (Moderate)

**Evidence:**
- **Full pipeline is connected** in `process_compilation()` (`server.py:482-562`):
  `Ingest → Parse → Map → Compile → Validate` — all stages run and pass data forward
- All API endpoints trigger real functionality, not stubs
- **CRITICAL:** Production API **hardcodes `use_mock=True`** for both PolicyParserAgent (`server.py:503`) and SchemaMappingAgent (`server.py:518`), meaning:
  - Parser uses regex-based pattern matching instead of LLM API calls
  - Mapper uses hash-based deterministic vectors instead of real embeddings
  - These are labeled "for testing" in their docstrings
- CompilerAgent and TraceValidatorAgent run without mocks — fully real
- Each agent has `publish_*_event()` functions that call `aegislang.core.events.publish_event` — but this module **does not exist** (dead-end stubs)

**Remediation:** Make `use_mock` configurable via environment variable. Default to real providers when API keys are available.

### B4. Async Correctness — Score: 2/3 (Moderate)

**Evidence:**
- `process_ingestion()` and `process_compilation()` are declared `async def` but contain **exclusively synchronous operations** — file I/O, regex processing, template rendering
- When FastAPI's `BackgroundTasks` runs an `async def` function, it executes on the event loop, **blocking it** during execution
- No `await`, `asyncio.to_thread()`, or `run_in_executor()` used for sync operations
- API endpoints themselves are properly lightweight async handlers
- No `asyncio.get_event_loop()` or `loop.run_until_complete()` anti-patterns
- Mitigating factor: multiple uvicorn workers (default 4) limit blast radius

**Remediation:** Convert background tasks to regular `def` (not `async def`) so FastAPI runs them in a thread pool automatically.

### B5. State Management Coherence — Score: 2/3 (Moderate)

**Evidence:**
- In-memory `Storage` class uses plain `dict` objects with **no locks for CRUD operations**
- `threading.Lock` exists (`server.py:281`) but is ONLY used for `_cleanup_expired_jobs()`
- Direct dict mutations (`storage.documents[key] = val`, `storage.clauses[doc_id] = [...]`) are unprotected
- With `workers=4` (default), each worker gets its own `Storage` — **no cross-worker state sharing**
- A document ingested via worker 1 is invisible to worker 2
- `SqliteStorage` alternative IS properly thread-safe (all operations locked)
- Pipeline stages run sequentially within `process_compilation()`, preventing inter-stage races

**Remediation:** Add locking to in-memory Storage, or enforce single-worker mode when using memory backend.

### B6. Security Implementation Depth — Score: 3/3 (Strong)

**Evidence:**
- API key auth with **constant-time comparison** via `hmac.compare_digest` (`server.py:64-66`)
- Loop iterates ALL keys without early break for true constant-time behavior (`server.py:88`)
- Rate limiter with dual-window enforcement (per-minute + per-hour) (`server.py:104-151`)
- File upload security:
  - Extension allowlist with path traversal prevention (`server.py:579-616`)
  - Double-extension attack prevention (`server.py:601`)
  - UUID-only temp filenames — no user input in filesystem paths (`server.py:659`)
  - Restrictive temp directory permissions `mode=0o700` (`server.py:656`)
  - **Secure file deletion** with random data overwrite before unlink (`server.py:400-436`)
- SQLite uses parameterized queries throughout (`sqlite_storage.py`)
- Jinja2 autoescape enabled for HTML/XML templates (`compiler_agent.py:551`)
- **Vulnerability noted:** Cypher injection potential at `trace_validator_agent.py:948,963` — `node.node_type` and `edge.relationship` interpolated via f-string into MERGE queries. Currently safe (values are hardcoded internally), but no allowlist validation as defense-in-depth.

### B7. Resource Management — Score: 2/3 (Moderate)

**Evidence:**
- File handles use context managers (`with open(...)`) for PDF, hash computation, secure delete
- Neo4j `Neo4jConnectionPool` properly manages connection lifecycle (`trace_validator_agent.py:39-164`)
- Neo4j sessions used as context managers (`trace_validator_agent.py:943`)
- Temp file cleanup in `finally` block ensures cleanup on failure (`server.py:477-479`)
- **BUT:** DOCX `Document()` not wrapped in context manager (`aegis_ingestor.py:417`)
- **BUT:** Rate limiter dict entries grow unbounded by client ID — empty entries never pruned
- **BUT:** No max-size limit on in-memory Storage dictionaries
- **BUT:** Full file read into memory before size check (`server.py:638`) — 50MB allocation before validation

### Domain B Summary: 16/21 = 76.2%

---

## Domain C: Interface Authenticity (30%)

### C1. API Design Consistency — Score: 3/3 (Strong)

**Evidence:**
- Consistent `snake_case` parameter naming across all endpoints
- Versioned URL prefix `/api/v1/` with resource-oriented paths
- Proper REST verbs: GET for retrieval, POST for creation
- Semantic HTTP status codes: 200, 400, 401, 403, 404, 413, 422, 429
- Tags for OpenAPI grouping (Health, Ingestion, Documents, Clauses, Rules, Compilation, Jobs, Schemas)
- Centralized `ErrorResponse` model for uniform error structure
- **Minor:** Async operations return 200 instead of 202 Accepted despite spawning background tasks

### C2. UI Implementation Depth — Score: 3/3 (Strong)

**Evidence:**
- Backend-only project — no frontend to evaluate
- **All 5 agent modules have fully functional CLI entry points** with `argparse`:
  - Each supports `-o/--output`, configurable parameters, error handling with exit codes
  - Not stubs — tested and functional
- OpenAPI docs at `/api/docs`, ReDoc at `/api/redoc`, OpenAPI JSON at `/api/openapi.json`
- All doc endpoints verified by tests (`test_api.py:306-328`)

### C3. State Management (API-side) — Score: 2/3 (Moderate)

**Evidence:**
- Job tracking with 4-state lifecycle: `PENDING → PROCESSING → COMPLETED/FAILED`
- Background cleanup thread with configurable TTL for expired jobs
- Storage backend switchable via environment variable (memory → SQLite)
- **BUT:** Request ID middleware is built but **never wired in** — `set_request_context()` exists but is never called by the API server
- **BUT:** No pipeline stage visibility — clients can only see "processing", not which of 4 stages is running
- **BUT:** No job cancellation mechanism

### C4. Security Infrastructure — Score: 2/3 (Moderate)

**Evidence:**
- API key authentication is real and robust (see B6)
- CORS configured via environment variable, defaults to `localhost:3000`
- **BUT:** `allow_methods=["*"]` and `allow_headers=["*"]` is overly permissive
- **BUT:** No security headers middleware (X-Content-Type-Options, X-Frame-Options, HSTS)
- **BUT:** No HTTPS enforcement
- **BUT:** No request body size limit middleware (only file upload size checked)

### C5. WebSocket Implementation — Score: 1/3 (Weak)

**Evidence:**
- **Zero WebSocket endpoints** — no matches for "websocket" in entire repository
- **No Server-Sent Events (SSE)**
- **No webhook callbacks** — despite `webhook_url` field in responses (misleading name; it's a polling URL)
- Only mechanism: poll `GET /api/v1/jobs/{job_id}` repeatedly
- For a multi-stage pipeline, this provides poor user experience

**Remediation:** Add WebSocket or SSE for real-time job status. Rename `webhook_url` to `status_url`.

### C6. Error UX — Score: 3/3 (Strong)

**Evidence:**
- 8 custom exception types with machine-readable error codes
- Structured `ErrorResponse` with `error`, `status_code`, `error_code`, `request_id`, `details`
- Production/development error distinction — sensitive details hidden in production
- User-facing messages are helpful: "Provide X-API-Key header", "Unsupported file type: .xyz. Allowed: {set}"
- Global exception handlers catch all unhandled errors and return sanitized responses
- **Minor:** Custom exceptions defined but not consistently used — server.py endpoints use `HTTPException` directly instead

### C7. Logging & Observability — Score: 3/3 (Strong)

**Evidence:**
- `structlog` with 8-processor chain including timestamps, log levels, context vars, Sentry integration
- JSON output mode for ELK/production, colored console for development
- `ContextVar`-based request ID and user ID propagation
- Sentry integration with breadcrumbs, exception capture, message capture
- `@log_exception` and `@log_async_exception` decorators
- Proper log level usage: DEBUG for starts, INFO for completions, WARNING for degraded operation, ERROR for failures
- Noisy library suppression (httpx, httpcore, urllib3)
- **BUT:** Request ID middleware not wired in (infrastructure exists but unused)
- **BUT:** No metrics endpoint (/metrics for Prometheus)
- **BUT:** No OpenTelemetry tracing

### Domain C Summary: 17/21 = 81.0%

---

## Final Calculation

```
Weighted Authenticity = (A% × 0.20) + (B% × 0.50) + (C% × 0.30)
                      = (47.6% × 0.20) + (76.2% × 0.50) + (81.0% × 0.30)
                      = 9.5% + 38.1% + 24.3%
                      = 71.9%

Vibe-Code Confidence  = 100% - 71.9% = 28.1%
```

### Classification Scale

| Range | Classification | This Repo |
|---|---|---|
| 0-15% | Human-Authored | |
| 16-35% | **AI-Assisted** | **28.1%** |
| 36-60% | Substantially Vibe-Coded | |
| 61-85% | Predominantly Vibe-Coded | |
| 86-100% | Almost Certainly AI-Generated | |

---

## Honest Assessment

AegisLang scores as **AI-Assisted** because Domains B and C demonstrate that the code
*actually works* — this is not a hollow scaffold. The pipeline genuinely processes documents,
extracts clauses, maps schemas, compiles artifacts, and validates provenance. The security
measures are real (constant-time auth, path traversal prevention, secure deletion). The
error handling is genuine. The logging infrastructure is production-grade.

However, **Domain A is devastating** for authenticity claims:

1. **100% of source code** was written by AI — the human never committed a single line of Python
2. **222 section dividers** across 17 files show machine-like structural uniformity
3. **Zero parametrized tests** despite 139 tests — breadth without depth
4. **Zero human frustration markers** in 69 commits — no reverts, no WIP, no "this is broken"
5. **Formulaic naming** with zero organic variation between files

The repository represents a **well-specified project** (the human wrote good specs in SPEC.md
and Step-by-step.md) that was **faithfully implemented by AI** through 21 pull requests, each
created and merged from Claude Code branches. The human's role was architect/reviewer, not coder.

---

## Top 10 Remediation Priorities

| # | Priority | Domain | Issue | File:Line |
|---|---|---|---|---|
| 1 | **Critical** | B3 | Hardcoded `use_mock=True` in production API | `server.py:503,518` |
| 2 | **Critical** | B4 | `async def` background tasks block event loop | `server.py:439,482` |
| 3 | **High** | B5 | In-memory Storage has no locking for CRUD | `server.py:253-376` |
| 4 | **High** | B2 | 6 phantom env vars + 12 missing from .env.example | `.env.example` |
| 5 | **High** | B6 | Cypher injection potential (defense-in-depth) | `trace_validator_agent.py:948,963` |
| 6 | **Medium** | C5 | No real-time job status (WebSocket/SSE) | `server.py` |
| 7 | **Medium** | C4 | Overly permissive CORS (wildcard methods/headers) | `server.py:179-180` |
| 8 | **Medium** | C3 | Request ID middleware built but never wired in | `logging.py:459` / `server.py` |
| 9 | **Low** | A3 | Zero parametrized tests — add edge case coverage | `tests/` |
| 10 | **Low** | B7 | DOCX Document() not in context manager | `aegis_ingestor.py:417` |

---

## Provenance Fingerprint

```
Commits by Author:
  noreply@anthropic.com   43 (62.3%)  ← Claude Code
  kase1111@gmail.com      26 (37.7%)  ← Human (merges + specs only)

Human Source Code Commits: 0
AI Source Code Commits:    43
Human Spec/Doc Commits:    5
Human Merge Commits:       21

Section Dividers:          222 across 17 files
Parametrized Tests:        0 of 139 tests
TODO/FIXME Markers:        3 in 12,067 LOC
Bare Except Clauses:       0
WHY Comments:              0
```

---

*Generated using the Vibe-Code Detection Audit v2.0 framework.*
*Audit methodology: https://github.com/kase1111-hash/Claude-prompts/blob/main/vibe-checkV2.md*
