# AegisLang Remediation Plan

**Based on:** Vibe-Code Detection Audit v2.0 (VIBE_CHECK_AUDIT.md)
**Created:** 2026-02-26
**Vibe-Code Score:** 28.1% (AI-Assisted)
**Goal:** Reduce to <15% (Human-Authored) by fixing real defects and improving test depth

---

## Phase 1: Critical Fixes (Week 1)

These two issues cause incorrect runtime behavior and must be fixed before any deployment.

---

### 1.1 Remove hardcoded `use_mock=True` from production API

**Audit ref:** B3 — Call Chain Completeness
**File:** `aegislang/api/server.py`
**Lines:** 503, 518
**Severity:** Critical — production API never uses real LLM providers

**Current code:**

```python
# server.py:503
parser = PolicyParserAgent(use_mock=True)

# server.py:518-519
mapper = SchemaMappingAgent(
    registry=create_default_registry(),
    use_mock=True,
)
```

**Problem:** The compile endpoint always uses pattern-matching mock instead of Claude/GPT.
Every API consumer gets regex-quality clause extraction regardless of available API keys.

**Fix:** Auto-detect based on whether LLM API keys are present in the environment.

```python
# server.py — near top, after imports
def _should_use_mock() -> bool:
    """Use real LLM providers when API keys are available."""
    return not (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )

# server.py:503 — replace the hardcoded True
parser = PolicyParserAgent(use_mock=_should_use_mock())

# server.py:518-519 — replace the hardcoded True
mapper = SchemaMappingAgent(
    registry=create_default_registry(),
    use_mock=_should_use_mock(),
)
```

**Test:** Add to `test_api.py`:
```python
def test_compile_uses_real_provider_when_api_key_set(monkeypatch, client, ...):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    # Verify PolicyParserAgent is instantiated with use_mock=False
```

**Risk:** Low. Mock fallback is preserved when no keys are set. Existing tests run without keys and will continue to use mock mode.

---

### 1.2 Fix async background tasks blocking the event loop

**Audit ref:** B4 — Async Correctness
**File:** `aegislang/api/server.py`
**Lines:** 439, 482
**Severity:** Critical — synchronous file I/O and computation blocks the event loop

**Current code:**

```python
# server.py:439
async def process_ingestion(job_id, file_path, metadata, storage, doc_id=None):
    ...
    result = ingestor.ingest(file_path)  # Blocking file I/O
    ...

# server.py:482
async def process_compilation(job_id, doc_id, output_formats, ...):
    ...
    parsed = parser.parse_ingested_document(doc_data)       # Blocking
    mapped = mapper.map_parsed_collection(parsed_data, ...)  # Blocking
    compiled = compiler.compile_mapped_collection(...)        # Blocking
    validated = validator.validate_compiled_collection(...)   # Blocking
    ...
```

**Problem:** FastAPI's `BackgroundTasks` runs `async def` functions on the event loop.
These functions contain zero `await` calls — they're purely synchronous (file reads,
regex, Jinja2 rendering). This blocks the event loop for the entire duration of pipeline
execution, stalling all concurrent HTTP requests.

**Fix:** Change both functions from `async def` to plain `def`. FastAPI will automatically
run them in a thread pool.

```python
# server.py:439 — change async def to def
def process_ingestion(job_id, file_path, metadata, storage, doc_id=None):
    ...  # Body stays identical

# server.py:482 — change async def to def
def process_compilation(job_id, doc_id, output_formats, ...):
    ...  # Body stays identical
```

That's it. Two words deleted. FastAPI's `BackgroundTasks.add_task()` accepts both sync
and async functions. When given a sync `def`, it runs it in the default thread pool executor,
which is exactly what we want for CPU/IO-bound work.

**Test:** Add to `test_api.py`:
```python
import inspect

def test_background_tasks_are_sync():
    """Background tasks must be sync to avoid blocking the event loop."""
    from aegislang.api.server import process_ingestion, process_compilation
    assert not inspect.iscoroutinefunction(process_ingestion)
    assert not inspect.iscoroutinefunction(process_compilation)
```

**Risk:** None. This is a strict improvement with no behavioral change.

---

## Phase 2: High Priority Fixes (Week 2)

These issues cause data loss, configuration confusion, or security weaknesses.

---

### 2.1 Add locking to in-memory Storage

**Audit ref:** B5 — State Management Coherence
**File:** `aegislang/api/server.py`
**Lines:** 253-376
**Severity:** High — concurrent requests can corrupt shared state

**Problem:** The `Storage` class uses plain `dict` objects. Only `_cleanup_expired_jobs()`
is protected by `self._cleanup_lock`. All other mutations (create_job, update_job, and
direct dict assignments in process_ingestion/process_compilation) are unprotected.

Additionally, with `workers=4` (default uvicorn config at line 875), each worker gets its
own `Storage` instance — a document ingested via worker 1 is invisible to worker 2.

**Fix (option A — recommended): Enforce single worker for memory backend:**

```python
# server.py:869-884 — in main()
def main() -> None:
    """Run the API server."""
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    workers = int(os.environ.get("WORKERS", "4"))
    reload = os.environ.get("RELOAD", "false").lower() == "true"
    backend = os.environ.get("AEGISLANG_STORAGE_BACKEND", "memory").lower()

    # In-memory storage cannot be shared across workers
    if backend == "memory" and workers > 1:
        logger.warning(
            "forcing_single_worker",
            message="In-memory storage requires workers=1. "
                    "Set AEGISLANG_STORAGE_BACKEND=sqlite for multi-worker.",
        )
        workers = 1

    uvicorn.run(
        "aegislang.api.server:app",
        host=host,
        port=port,
        workers=1 if reload else workers,
        reload=reload,
    )
```

**Fix (option B — belt and suspenders): Also add a lock for CRUD operations:**

```python
# server.py — in Storage.__init__
self._write_lock = threading.Lock()

# Wrap all mutation methods with the lock:
def create_job(self, job_type: str) -> str:
    with self._write_lock:
        ...  # existing body

def update_job(self, job_id, status, result=None, error=None):
    with self._write_lock:
        ...  # existing body
```

And wrap the direct dict mutations in `process_ingestion` and `process_compilation`:

```python
# server.py:463 — in process_ingestion
with storage._write_lock:  # or use a storage.store_document() method
    storage.documents[storage_key] = doc_data

# server.py:508,521,532 — in process_compilation
with storage._write_lock:
    storage.clauses[doc_id] = parsed_data.get("clauses", [])
# etc.
```

**Recommended:** Implement both A and B. A prevents cross-worker data loss; B prevents
intra-worker race conditions between the background thread and request handlers.

**Risk:** Low. Option A may reduce throughput for CPU-bound workloads. Option B adds
minimal lock contention (mutations are fast dict operations).

---

### 2.2 Synchronize .env.example with actual code

**Audit ref:** B2 — Configuration Actually Used
**File:** `.env.example`
**Severity:** High — developers configure wrong variables, miss required ones

**Phantom variables (in .env.example but never read by code):**

| Variable | Status |
|---|---|
| `DATABASE_URL` | Never imported — code uses SQLite or in-memory |
| `PINECONE_API_KEY` | Never imported — no vector store implemented |
| `REDIS_URL` | Never imported — no Redis integration |
| `JWT_SECRET` | Never imported — auth uses API keys, not JWT |
| `LOG_LEVEL` | Wrong name — code reads `AEGISLANG_LOG_LEVEL` |
| `SERVER_PORT` | Wrong name — code reads `PORT` |
| `ENVIRONMENT` | Wrong name — code reads `AEGISLANG_ENV` |

**Missing variables (read by code but not in .env.example):**

| Variable | Used in | Purpose |
|---|---|---|
| `AEGISLANG_API_KEYS` | `server.py:48` | API key auth (comma-separated) |
| `AEGISLANG_DISABLE_AUTH` | `server.py:51` | Disable auth for development |
| `AEGISLANG_STORAGE_BACKEND` | `server.py:381` | `memory` or `sqlite` |
| `AEGISLANG_SQLITE_PATH` | `sqlite_storage.py` | SQLite file path |
| `AEGISLANG_MAX_FILE_SIZE` | `server.py:637` | Max upload size in bytes |
| `AEGISLANG_RATE_LIMIT_MINUTE` | `server.py:149` | Requests per minute |
| `AEGISLANG_RATE_LIMIT_HOUR` | `server.py:150` | Requests per hour |
| `AEGISLANG_JOB_TTL_SECONDS` | `server.py:277` | Job expiry TTL |
| `AEGISLANG_ENV` | `errors.py:167` | Environment name |
| `AEGISLANG_LOG_LEVEL` | `logging.py:268` | Log level override |
| `AEGISLANG_LOG_FILE` | `logging.py:317` | Log file path |
| `AEGISLANG_VERSION` | `logging.py:112` | Version for Sentry |
| `SENTRY_DSN` | `logging.py:271` | Sentry DSN |
| `CORS_ORIGINS` | `server.py:177` | Allowed CORS origins |
| `HOST` | `server.py:873` | Server bind host |
| `PORT` | `server.py:874` | Server bind port |
| `WORKERS` | `server.py:875` | Uvicorn worker count |
| `RELOAD` | `server.py:876` | Hot reload mode |

**Fix:** Rewrite `.env.example` to match reality. Group by category. Remove all phantom
entries. Add all missing entries with sensible defaults and documentation comments.

**Risk:** None. This is a documentation-only change.

---

### 2.3 Fix Cypher injection in trace validator

**Audit ref:** B6 — Security Implementation Depth
**File:** `aegislang/agents/trace_validator_agent.py`
**Lines:** 948, 963
**Severity:** High — defense-in-depth (currently safe because values are internal)

**Current code:**

```python
# Line 948 — node_type interpolated via f-string
MERGE (n:{node.node_type} {{node_id: $node_id}})

# Line 963 — relationship interpolated via f-string
MERGE (a)-[r:{edge.relationship}]->(b)
```

**Problem:** `node.node_type` and `edge.relationship` are inserted into Cypher queries
via f-string. Currently safe because values are hardcoded internally (`document`,
`section`, `chunk`, `clause`, `artifact` for types; `CONTAINS_SECTION`, `CONTAINS_CHUNK`,
`PARSED_TO`, `COMPILED_TO` for relationships). But if any upstream code ever passes
user-controlled values, this becomes Cypher injection.

**Fix:** Add an allowlist check before query construction.

```python
# trace_validator_agent.py — add near top of class or as module constant
ALLOWED_NODE_TYPES = frozenset({
    "document", "section", "chunk", "clause", "mapping", "artifact"
})
ALLOWED_RELATIONSHIPS = frozenset({
    "CONTAINS_SECTION", "CONTAINS_CHUNK", "PARSED_TO",
    "MAPPED_TO", "COMPILED_TO"
})

# trace_validator_agent.py:944-955 — add validation before MERGE
for node in graph.nodes:
    if node.node_type not in ALLOWED_NODE_TYPES:
        logger.error("invalid_node_type", node_type=node.node_type)
        continue
    session.run(
        f"""
        MERGE (n:{node.node_type} {{node_id: $node_id}})
        ...

# trace_validator_agent.py:958-969 — add validation before MERGE
for edge in graph.edges:
    if edge.relationship not in ALLOWED_RELATIONSHIPS:
        logger.error("invalid_relationship", relationship=edge.relationship)
        continue
    session.run(
        f"""
        MATCH (a {{node_id: $source_id}})
        ...
```

**Risk:** None. All current values are in the allowlist.

---

## Phase 3: Medium Priority Fixes (Week 3)

These improve operational quality and developer experience.

---

### 3.1 Wire in request ID middleware

**Audit ref:** C3 — State Management / C7 — Logging & Observability
**Files:** `aegislang/api/server.py`, `aegislang/core/logging.py`
**Severity:** Medium — infrastructure exists but is never activated

**Problem:** `logging.py:459-476` defines `set_request_context()` and
`clear_request_context()`. The `add_context_processor` at line 211 reads
`request_id_var`. Error responses include a `request_id` field. But **no middleware
ever calls `set_request_context()`** — the request ID is always `None`.

**Fix:** Add a FastAPI middleware to `server.py` after the CORS middleware:

```python
# server.py — after CORS middleware (line 181)
from aegislang.core.logging import set_request_context, clear_request_context

@app.middleware("http")
async def request_id_middleware(request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    set_request_context(request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        clear_request_context()
```

**Test:** Add to `test_api.py`:
```python
def test_request_id_returned_in_header(self, client):
    response = client.get("/api/v1/health")
    assert "X-Request-ID" in response.headers

def test_custom_request_id_propagated(self, client):
    response = client.get(
        "/api/v1/health",
        headers={"X-Request-ID": "test-123"}
    )
    assert response.headers["X-Request-ID"] == "test-123"
```

**Risk:** None. Strictly additive.

---

### 3.2 Restrict CORS configuration

**Audit ref:** C4 — Security Infrastructure
**File:** `aegislang/api/server.py`
**Lines:** 175-181
**Severity:** Medium — overly permissive headers

**Current code:**

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[...from env...],
    allow_credentials=True,
    allow_methods=["*"],      # ← Too permissive
    allow_headers=["*"],      # ← Too permissive
)
```

**Fix:** Restrict to methods and headers the API actually uses:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:3000"
    ).split(",")],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-API-Key", "Content-Type", "X-Request-ID", "Accept"],
)
```

**Risk:** Low. If a future endpoint uses PUT/DELETE/PATCH, the methods list needs updating.

---

### 3.3 Add SSE endpoint for real-time job status

**Audit ref:** C5 — WebSocket Implementation
**File:** `aegislang/api/server.py`
**Severity:** Medium — polling is the only mechanism for long-running pipelines

**Problem:** `webhook_url` in `IngestResponse` is misleadingly named — it's a polling URL.
Clients must repeatedly call `GET /api/v1/jobs/{job_id}` to check status.

**Fix (minimal):** Rename the field and add an SSE endpoint.

```python
# 1. Rename webhook_url to status_url in IngestResponse
class IngestResponse(BaseModel):
    status: str = Field(...)
    job_id: str = Field(...)
    doc_id: str = Field(...)
    estimated_completion: str | None = Field(default=None)
    status_url: str = Field(..., description="URL to poll for job status")

# 2. Add SSE endpoint
from fastapi.responses import StreamingResponse
import asyncio

@app.get("/api/v1/jobs/{job_id}/stream", tags=["Jobs"])
async def stream_job_status(
    job_id: str,
    storage: Storage = Depends(get_storage),
    api_key: str = Depends(check_rate_limit),
):
    """Stream job status updates via Server-Sent Events."""
    if job_id not in storage.jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        while True:
            if job_id not in storage.jobs:
                break
            job = storage.jobs[job_id]
            data = json.dumps({"status": job["status"], "result": job.get("result")})
            yield f"data: {data}\n\n"
            if job["status"] in (JobStatus.COMPLETED.value, JobStatus.FAILED.value):
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

**Risk:** Low. Additive endpoint. Polling continues to work.

---

## Phase 4: Low Priority Fixes (Week 4)

These improve code quality and test confidence.

---

### 4.1 Add parametrized tests for edge cases

**Audit ref:** A3 — Test Quality Signals
**Files:** `tests/test_ingestor.py`, `tests/test_api.py`, `tests/test_parser.py`
**Severity:** Low — tests pass but lack depth

**Problem:** 139 tests, zero use of `@pytest.mark.parametrize`. Tests are all
single-case happy-path. No edge case coverage.

**Fix:** Add parametrized tests to the three most impactful areas:

**4.1a — Chunker edge cases** (`tests/test_ingestor.py`):

```python
@pytest.mark.parametrize("text,expected_min_chunks", [
    ("", 0),                                    # empty
    ("   \n\n   ", 0),                          # whitespace only
    ("Hello.", 1),                              # single word
    ("Word " * 500, 3),                         # exceeds max tokens
    ("Line\n" * 100, 1),                        # many short lines
    ("A" * 10000, 1),                           # no word boundaries
    ("## Heading\n\nBody text\n\n## Heading 2\n\nMore text", 1),  # markdown
])
def test_chunk_text_edge_cases(self, chunker, text, expected_min_chunks):
    chunks = chunker.chunk_text(text, "EDGE_S001")
    assert len(chunks) >= expected_min_chunks
    for chunk in chunks:
        assert chunk.token_count >= 0
```

**4.1b — File upload validation** (`tests/test_api.py`):

```python
@pytest.mark.parametrize("filename,expected_status", [
    ("policy.md", 200),
    ("policy.pdf", 200),
    ("policy.html", 200),
    ("policy.docx", 200),
    ("policy.exe", 400),
    ("policy.md.exe", 400),
    ("../../../etc/passwd", 400),
    ("policy.MD", 200),           # case insensitive
    (".hidden.md", 200),
    ("", 400),                    # empty filename
])
def test_file_extension_validation(self, client, tmp_path, filename, expected_status):
    test_file = tmp_path / (filename or "empty")
    test_file.write_text("# Test")
    with open(test_file, "rb") as f:
        response = client.post(
            "/api/v1/ingest",
            files={"file": (filename, f, "application/octet-stream")},
        )
    assert response.status_code == expected_status
```

**4.1c — Clause type detection** (`tests/test_parser.py`):

```python
@pytest.mark.parametrize("text,expected_type", [
    ("Institutions must verify identity", "obligation"),
    ("Banks shall report transactions", "obligation"),
    ("Staff must not share information", "prohibition"),
    ("Employees shall not process without auth", "prohibition"),
    ("Institutions may request extensions", "permission"),
    ("If the amount exceeds $10000, then report", "conditional"),
])
def test_clause_type_detection(self, text, expected_type):
    from aegislang.agents.policy_parser_agent import PolicyParserAgent
    parser = PolicyParserAgent(use_mock=True)
    # Test the mock pattern matching directly
    ...
```

**Risk:** None. Additive tests only.

---

### 4.2 Fix DOCX Document not in context manager

**Audit ref:** B7 — Resource Management
**File:** `aegislang/agents/aegis_ingestor.py`
**Line:** 417
**Severity:** Low — file handle may not be closed on exception

**Current code:**

```python
doc = Document(str(file_path))  # Line 417
...
for para in doc.paragraphs:     # Line 427
```

**Problem:** `python-docx`'s `Document()` opens a file handle internally. If an
exception occurs during paragraph iteration, the handle leaks.

**Fix:**

```python
# Note: python-docx Document doesn't support context manager protocol,
# but we can ensure the underlying file is closed
from docx import Document
from contextlib import closing

# Wrap to ensure cleanup
doc = Document(str(file_path))
try:
    ...  # existing paragraph processing
finally:
    # python-docx stores the package internally
    if hasattr(doc, 'part') and hasattr(doc.part, 'package'):
        try:
            doc.part.package.close()
        except Exception:
            pass
```

**Alternative (simpler):** Since `python-docx` reads the entire file into memory on
construction, the file handle is typically released quickly. If this is deemed low-risk,
add a comment explaining the decision:

```python
# python-docx reads entire file into memory on construction;
# no explicit close needed as handle is released after init
doc = Document(str(file_path))
```

**Risk:** None.

---

## Phase Summary

| Phase | Issues | Effort | Impact on Score |
|---|---|---|---|
| **Phase 1** (Week 1) | 1.1, 1.2 | ~2 hours | B3: 2→3, B4: 2→3 |
| **Phase 2** (Week 2) | 2.1, 2.2, 2.3 | ~4 hours | B2: 2→3, B5: 2→3, B6: 3→3 |
| **Phase 3** (Week 3) | 3.1, 3.2, 3.3 | ~4 hours | C3: 2→3, C4: 2→3, C5: 1→2 |
| **Phase 4** (Week 4) | 4.1, 4.2 | ~3 hours | A3: 1→2, B7: 2→3 |

**Projected post-remediation score:**

```
Domain A: 52.4% × 0.20 = 10.5%  (A3 improved)
Domain B: 90.5% × 0.50 = 45.2%  (B2-B5 improved)
Domain C: 90.5% × 0.30 = 27.1%  (C3-C5 improved)

New Weighted Authenticity = 82.8%
New Vibe-Code Confidence  = 17.2%  (down from 28.1%)
```

Domain A (Surface Provenance) will remain the weakest area because commit history and
code uniformity are permanent artifacts. The most impactful long-term improvement is
**human code contribution** — manual modifications, domain-specific naming, and organic
refactoring that break the uniform AI fingerprint.
