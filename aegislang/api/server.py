"""
AegisLang REST API Server

Provides REST API endpoints for document ingestion, parsing, mapping,
compilation, and validation operations.

Base URL: /api/v1
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# =============================================================================
# Security: API Key Authentication
# =============================================================================

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# API keys can be set via environment variable (comma-separated)
# Example: AEGISLANG_API_KEYS="key1,key2,key3"
def get_valid_api_keys() -> set[str]:
    """Get valid API keys from environment."""
    keys_env = os.environ.get("AEGISLANG_API_KEYS", "")
    if not keys_env:
        # If no keys configured, check if auth is disabled
        if os.environ.get("AEGISLANG_DISABLE_AUTH", "").lower() == "true":
            return set()
        # Generate a random key for development and log it
        dev_key = secrets.token_urlsafe(32)
        logger.warning(
            "no_api_keys_configured",
            message="No API keys configured. Set AEGISLANG_API_KEYS or AEGISLANG_DISABLE_AUTH=true",
            development_key=dev_key,
        )
        return {dev_key}
    return {k.strip() for k in keys_env.split(",") if k.strip()}


def _constant_time_compare(val1: str, val2: str) -> bool:
    """Perform constant-time string comparison to prevent timing attacks."""
    return hmac.compare_digest(val1.encode("utf-8"), val2.encode("utf-8"))


async def verify_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> str:
    """Verify API key and return it if valid (uses constant-time comparison)."""
    valid_keys = get_valid_api_keys()

    # If auth is disabled (empty keys set), allow all requests
    if not valid_keys:
        return "anonymous"

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide X-API-Key header.",
        )

    # Use constant-time comparison to prevent timing attacks
    key_valid = False
    for valid_key in valid_keys:
        if _constant_time_compare(api_key, valid_key):
            key_valid = True
            # Don't break early - continue checking all keys for constant time

    if not key_valid:
        logger.warning("invalid_api_key_attempt", key_prefix=api_key[:8] if api_key else "none")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key.",
        )

    return api_key


# =============================================================================
# Security: Rate Limiting
# =============================================================================

class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self._minute_counts: dict[str, list[float]] = defaultdict(list)
        self._hour_counts: dict[str, list[float]] = defaultdict(list)

    def _cleanup_old_entries(self, entries: list[float], max_age: float) -> list[float]:
        """Remove entries older than max_age seconds."""
        cutoff = time.time() - max_age
        return [t for t in entries if t > cutoff]

    def is_allowed(self, client_id: str) -> tuple[bool, str | None]:
        """Check if request is allowed for client."""
        now = time.time()

        # Clean up and check minute limit
        self._minute_counts[client_id] = self._cleanup_old_entries(
            self._minute_counts[client_id], 60
        )
        if len(self._minute_counts[client_id]) >= self.requests_per_minute:
            return False, f"Rate limit exceeded: {self.requests_per_minute}/minute"

        # Clean up and check hour limit
        self._hour_counts[client_id] = self._cleanup_old_entries(
            self._hour_counts[client_id], 3600
        )
        if len(self._hour_counts[client_id]) >= self.requests_per_hour:
            return False, f"Rate limit exceeded: {self.requests_per_hour}/hour"

        # Record request
        self._minute_counts[client_id].append(now)
        self._hour_counts[client_id].append(now)

        return True, None


# Global rate limiter instance
_rate_limiter = RateLimiter(
    requests_per_minute=int(os.environ.get("AEGISLANG_RATE_LIMIT_MINUTE", "60")),
    requests_per_hour=int(os.environ.get("AEGISLANG_RATE_LIMIT_HOUR", "1000")),
)


async def check_rate_limit(api_key: str = Depends(verify_api_key)) -> str:
    """Check rate limit for the API key."""
    allowed, error = _rate_limiter.is_allowed(api_key)
    if not allowed:
        raise HTTPException(status_code=429, detail=error)
    return api_key

# =============================================================================
# Application Setup
# =============================================================================

app = FastAPI(
    title="AegisLang API",
    description="Multi-agent semantic compiler for regulatory compliance",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request/Response Models
# =============================================================================

class JobStatus(str, Enum):
    """Status of an async job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentMetadataRequest(BaseModel):
    """Metadata for document ingestion."""
    document_name: str = Field(..., description="Name of the document")
    document_type: str = Field(default="regulation", description="Type of document")
    jurisdiction: str | None = Field(default=None, description="Jurisdiction")
    effective_date: str | None = Field(default=None, description="Effective date")


class IngestResponse(BaseModel):
    """Response from document ingestion."""
    status: str = Field(..., description="Request status")
    job_id: str = Field(..., description="Async job ID")
    doc_id: str = Field(..., description="Document ID")
    estimated_completion: str | None = Field(default=None, description="Estimated completion time")
    webhook_url: str = Field(..., description="URL to check job status")


class CompileRequest(BaseModel):
    """Request for document compilation."""
    doc_id: str = Field(..., description="Document ID to compile")
    output_formats: list[str] = Field(
        default=["yaml", "sql"],
        description="Output formats to generate"
    )
    target_schema: str | None = Field(default=None, description="Target schema ID")
    confidence_threshold: float = Field(default=0.85, description="Minimum confidence")


class JobResponse(BaseModel):
    """Response with job status."""
    job_id: str
    status: JobStatus
    created_at: str
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str


class SchemaRegistryRequest(BaseModel):
    """Request to register a schema."""
    schema_id: str
    schema_type: str
    version: str = "1.0.0"
    tables: list[dict[str, Any]]


# =============================================================================
# In-Memory Storage (replace with database in production)
# =============================================================================

class Storage:
    """
    Simple in-memory storage for jobs and documents.

    WARNING: This storage is ephemeral. All data is lost on server restart.
    For production use, implement a persistent storage backend (PostgreSQL, etc.)
    """

    _warned: bool = False

    # Default job TTL: 24 hours (in seconds)
    DEFAULT_JOB_TTL = 24 * 60 * 60
    # Cleanup interval: 1 hour (in seconds)
    CLEANUP_INTERVAL = 60 * 60

    def __init__(self, job_ttl_seconds: int | None = None):
        self.jobs: dict[str, dict[str, Any]] = {}
        self.documents: dict[str, dict[str, Any]] = {}
        self.clauses: dict[str, list[dict[str, Any]]] = {}
        self.artifacts: dict[str, list[dict[str, Any]]] = {}
        self.schemas: dict[str, dict[str, Any]] = {}

        # Job TTL configuration (can be overridden via env var)
        self.job_ttl = job_ttl_seconds or int(
            os.environ.get("AEGISLANG_JOB_TTL_SECONDS", str(self.DEFAULT_JOB_TTL))
        )

        # Lock for thread-safe cleanup
        self._cleanup_lock = threading.Lock()
        self._last_cleanup = time.time()

        # Log warning once per process
        if not Storage._warned:
            logger.warning(
                "in_memory_storage_active",
                message="Using in-memory storage. Data will be lost on restart. "
                        "Set AEGISLANG_STORAGE_BACKEND for production use.",
            )
            Storage._warned = True

        # Start background cleanup
        self._start_cleanup_thread()

    def _start_cleanup_thread(self) -> None:
        """Start background thread for periodic job cleanup."""
        def cleanup_loop():
            while True:
                time.sleep(self.CLEANUP_INTERVAL)
                self._cleanup_expired_jobs()

        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
        logger.info("job_cleanup_thread_started", ttl_seconds=self.job_ttl)

    def _cleanup_expired_jobs(self) -> None:
        """Remove jobs that have exceeded their TTL."""
        with self._cleanup_lock:
            now = datetime.now(timezone.utc)
            expired_jobs = []

            for job_id, job in self.jobs.items():
                # Only clean up completed or failed jobs
                if job["status"] not in (JobStatus.COMPLETED, JobStatus.FAILED):
                    continue

                completed_at = job.get("completed_at")
                if not completed_at:
                    continue

                # Parse completion time and check TTL
                try:
                    completed_time = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
                    age_seconds = (now - completed_time).total_seconds()

                    if age_seconds > self.job_ttl:
                        expired_jobs.append(job_id)
                except (ValueError, TypeError):
                    # If we can't parse the timestamp, skip this job
                    continue

            # Remove expired jobs
            for job_id in expired_jobs:
                del self.jobs[job_id]

            if expired_jobs:
                logger.info(
                    "expired_jobs_cleaned",
                    count=len(expired_jobs),
                    job_ids=expired_jobs[:10],  # Log first 10 for brevity
                )

    def create_job(self, job_type: str) -> str:
        """Create a new job and return its ID."""
        # Trigger cleanup if it's been a while (opportunistic cleanup)
        if time.time() - self._last_cleanup > self.CLEANUP_INTERVAL:
            self._cleanup_expired_jobs()
            self._last_cleanup = time.time()

        job_id = f"{job_type}_{uuid.uuid4().hex[:8]}"
        self.jobs[job_id] = {
            "job_id": job_id,
            "job_type": job_type,
            "status": JobStatus.PENDING,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "result": None,
            "error": None,
        }
        return job_id

    def update_job(
        self,
        job_id: str,
        status: JobStatus,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Update job status."""
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = status
            self.jobs[job_id]["result"] = result
            self.jobs[job_id]["error"] = error
            if status in (JobStatus.COMPLETED, JobStatus.FAILED):
                self.jobs[job_id]["completed_at"] = datetime.now(timezone.utc).isoformat()


def _create_storage() -> Storage:
    """Create storage backend based on AEGISLANG_STORAGE_BACKEND env var."""
    backend = os.environ.get("AEGISLANG_STORAGE_BACKEND", "memory").lower()
    if backend == "sqlite":
        from aegislang.api.sqlite_storage import SqliteStorage
        return SqliteStorage()  # type: ignore[return-value]
    return Storage()


storage = _create_storage()


def get_storage() -> Storage:
    """Dependency to get storage instance."""
    return storage


# =============================================================================
# Background Tasks
# =============================================================================

def secure_delete_file(file_path: Path) -> None:
    """
    Securely delete a file by overwriting with random data before unlinking.

    For sensitive policy documents, this provides basic protection against
    simple file recovery techniques.
    """
    try:
        if not file_path.exists():
            return

        # Get file size
        file_size = file_path.stat().st_size

        # Overwrite with random data (single pass)
        with open(file_path, "wb") as f:
            # Write in chunks to handle large files
            chunk_size = 8192
            remaining = file_size
            while remaining > 0:
                write_size = min(chunk_size, remaining)
                f.write(os.urandom(write_size))
                remaining -= write_size
            f.flush()
            os.fsync(f.fileno())

        # Now unlink the file
        file_path.unlink()

    except Exception as e:
        logger.warning("secure_delete_failed", path=str(file_path), error=str(e))
        # Fall back to simple deletion
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass


def _should_use_mock() -> bool:
    """Use mock providers when no LLM API keys are available."""
    return not (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )


def process_ingestion(
    job_id: str,
    file_path: Path,
    metadata: dict[str, Any],
    storage: Storage,
    doc_id: str | None = None,
) -> None:
    """Background task for document ingestion."""
    try:
        storage.update_job(job_id, JobStatus.PROCESSING)

        # Import and run ingestor
        from aegislang.agents.aegis_ingestor import AegisIngestor

        ingestor = AegisIngestor()
        result = ingestor.ingest(file_path)

        # Use the endpoint-provided doc_id for storage consistency
        storage_key = doc_id or result.doc_id

        # Store document
        doc_data = result.model_dump()
        doc_data["metadata"].update(metadata)
        doc_data["doc_id"] = storage_key
        storage.documents[storage_key] = doc_data

        storage.update_job(
            job_id,
            JobStatus.COMPLETED,
            result={"doc_id": storage_key, "sections": len(result.sections)},
        )

        logger.info("ingestion_completed", job_id=job_id, doc_id=storage_key)

    except Exception as e:
        logger.error("ingestion_failed", job_id=job_id, error=str(e))
        storage.update_job(job_id, JobStatus.FAILED, error=str(e))

    finally:
        # Securely cleanup temp file
        secure_delete_file(file_path)


def process_compilation(
    job_id: str,
    doc_id: str,
    output_formats: list[str],
    target_schema: str | None,
    confidence_threshold: float,
    storage: Storage,
) -> None:
    """Background task for full pipeline compilation."""
    try:
        storage.update_job(job_id, JobStatus.PROCESSING)

        # Get document
        if doc_id not in storage.documents:
            raise ValueError(f"Document not found: {doc_id}")

        doc_data = storage.documents[doc_id]

        # Run parser
        from aegislang.agents.policy_parser_agent import PolicyParserAgent

        parser = PolicyParserAgent(use_mock=_should_use_mock())
        parsed = parser.parse_ingested_document(doc_data)
        parsed_data = parsed.model_dump()

        # Store clauses
        storage.clauses[doc_id] = parsed_data.get("clauses", [])

        # Run mapper
        from aegislang.agents.schema_mapping_agent import (
            SchemaMappingAgent,
            create_default_registry,
        )

        mapper = SchemaMappingAgent(
            registry=create_default_registry(),
            use_mock=_should_use_mock(),
        )
        mapped = mapper.map_parsed_collection(parsed_data, target_schema)
        mapped_data = mapped.model_dump()

        # Run compiler
        from aegislang.agents.compiler_agent import CompilerAgent, ArtifactFormat

        compiler = CompilerAgent()
        formats = [ArtifactFormat(f) for f in output_formats]
        compiled = compiler.compile_mapped_collection(mapped_data, formats)
        compiled_data = compiled.model_dump()

        # Store artifacts
        storage.artifacts[doc_id] = compiled_data.get("artifacts", [])

        # Run validator
        from aegislang.agents.trace_validator_agent import (
            TraceValidatorAgent,
            ValidationConfig,
        )

        validator = TraceValidatorAgent(
            config=ValidationConfig(confidence_threshold=confidence_threshold)
        )
        validated = validator.validate_compiled_collection(
            compiled_data, mapped_data, parsed_data
        )

        storage.update_job(
            job_id,
            JobStatus.COMPLETED,
            result={
                "doc_id": doc_id,
                "clauses_parsed": len(parsed_data.get("clauses", [])),
                "artifacts_generated": len(compiled_data.get("artifacts", [])),
                "validation_summary": validated.summary,
            },
        )

        logger.info("compilation_completed", job_id=job_id, doc_id=doc_id)

    except Exception as e:
        logger.error("compilation_failed", job_id=job_id, error=str(e))
        storage.update_job(job_id, JobStatus.FAILED, error=str(e))


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def validate_file_extension(filename: str | None) -> str:
    """
    Validate and extract file extension securely.

    Prevents path traversal and double extension attacks.
    Returns the validated extension or raises HTTPException.
    """
    allowed_extensions = {".pdf", ".docx", ".md", ".markdown", ".html", ".htm"}

    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # Sanitize filename - remove any path components
    safe_filename = Path(filename).name

    # Check for path traversal attempts
    if ".." in safe_filename or "/" in filename or "\\" in filename:
        logger.warning("path_traversal_attempt", filename=filename)
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Extract extension (only the last one, lowercased)
    # This prevents double extension attacks like "file.pdf.exe"
    ext = Path(safe_filename).suffix.lower()

    # Validate extension exists and starts with a dot
    if not ext or not ext.startswith("."):
        raise HTTPException(
            status_code=400,
            detail=f"File must have an extension. Allowed: {allowed_extensions}",
        )

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {allowed_extensions}",
        )

    return ext


@app.post("/api/v1/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: str = Form(default="{}"),
    storage: Storage = Depends(get_storage),
    api_key: str = Depends(check_rate_limit),
) -> IngestResponse:
    """
    Upload and parse a new policy document.

    Accepts PDF, DOCX, Markdown, or HTML files.
    Requires X-API-Key header for authentication.
    """
    # Validate file type with security checks
    file_ext = validate_file_extension(file.filename)

    # Validate file size (max 50MB)
    max_size = int(os.environ.get("AEGISLANG_MAX_FILE_SIZE", str(50 * 1024 * 1024)))
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {max_size // (1024*1024)}MB",
        )

    # Parse metadata
    try:
        meta_dict = json.loads(metadata)
        # Validate metadata is a dict (prevent injection)
        if not isinstance(meta_dict, dict):
            meta_dict = {}
    except json.JSONDecodeError:
        meta_dict = {}

    # Save file to temp location with secure naming
    temp_dir = Path(tempfile.gettempdir()) / "aegislang"
    temp_dir.mkdir(exist_ok=True, mode=0o700)  # Restrictive permissions

    # Use only UUID for temp filename - no user input
    temp_path = temp_dir / f"{uuid.uuid4().hex}{file_ext}"
    temp_path.write_bytes(content)

    # Create job
    job_id = storage.create_job("ing")

    # Sanitize document ID from filename
    safe_stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(file.filename or "document").stem)
    safe_stem = safe_stem[:50].upper()  # Limit length
    doc_id = f"{safe_stem}_{uuid.uuid4().hex[:6].upper()}"

    # Schedule background task
    background_tasks.add_task(
        process_ingestion,
        job_id,
        temp_path,
        meta_dict,
        storage,
        doc_id,
    )

    return IngestResponse(
        status="accepted",
        job_id=job_id,
        doc_id=doc_id,
        webhook_url=f"/api/v1/jobs/{job_id}",
    )


@app.get("/api/v1/documents", tags=["Documents"])
async def list_documents(
    storage: Storage = Depends(get_storage),
    api_key: str = Depends(check_rate_limit),
) -> list[dict[str, Any]]:
    """List all ingested documents. Requires X-API-Key header."""
    return [
        {
            "doc_id": doc_id,
            "metadata": doc.get("metadata", {}),
            "section_count": len(doc.get("sections", [])),
        }
        for doc_id, doc in storage.documents.items()
    ]


@app.get("/api/v1/documents/{doc_id}", tags=["Documents"])
async def get_document(
    doc_id: str,
    storage: Storage = Depends(get_storage),
    api_key: str = Depends(check_rate_limit),
) -> dict[str, Any]:
    """Retrieve document metadata and processing status. Requires X-API-Key header."""
    if doc_id not in storage.documents:
        raise HTTPException(status_code=404, detail="Document not found")

    doc = storage.documents[doc_id]
    return {
        "doc_id": doc_id,
        "metadata": doc.get("metadata", {}),
        "section_count": len(doc.get("sections", [])),
        "status": "processed",
    }


@app.get("/api/v1/clauses/{doc_id}", tags=["Clauses"])
async def get_clauses(
    doc_id: str,
    storage: Storage = Depends(get_storage),
    api_key: str = Depends(check_rate_limit),
) -> dict[str, Any]:
    """List all parsed clauses for a document. Requires X-API-Key header."""
    if doc_id not in storage.clauses:
        raise HTTPException(status_code=404, detail="Clauses not found for document")

    clauses = storage.clauses[doc_id]
    return {
        "doc_id": doc_id,
        "clause_count": len(clauses),
        "clauses": clauses,
    }


@app.get("/api/v1/rules/{clause_id}", tags=["Rules"])
async def get_rule(
    clause_id: str,
    storage: Storage = Depends(get_storage),
    api_key: str = Depends(check_rate_limit),
) -> dict[str, Any]:
    """Retrieve generated rule artifact for a clause. Requires X-API-Key header."""
    # Search through all artifacts
    for doc_id, artifacts in storage.artifacts.items():
        for artifact in artifacts:
            if artifact.get("clause_id") == clause_id:
                return {
                    "clause_id": clause_id,
                    "artifacts": [artifact],
                    "doc_id": doc_id,
                }

    raise HTTPException(status_code=404, detail="Rule not found")


@app.post("/api/v1/compile", tags=["Compilation"])
async def compile_document(
    request: CompileRequest,
    background_tasks: BackgroundTasks,
    storage: Storage = Depends(get_storage),
    api_key: str = Depends(check_rate_limit),
) -> dict[str, Any]:
    """Trigger full compilation pipeline for a document. Requires X-API-Key header."""
    if request.doc_id not in storage.documents:
        raise HTTPException(status_code=404, detail="Document not found")

    # Create job
    job_id = storage.create_job("cmp")

    # Schedule background task
    background_tasks.add_task(
        process_compilation,
        job_id,
        request.doc_id,
        request.output_formats,
        request.target_schema,
        request.confidence_threshold,
        storage,
    )

    return {
        "status": "accepted",
        "job_id": job_id,
        "doc_id": request.doc_id,
        "webhook_url": f"/api/v1/jobs/{job_id}",
    }


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse, tags=["Jobs"])
async def get_job_status(
    job_id: str,
    storage: Storage = Depends(get_storage),
    api_key: str = Depends(check_rate_limit),
) -> JobResponse:
    """Check async job status. Requires X-API-Key header."""
    if job_id not in storage.jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = storage.jobs[job_id]
    return JobResponse(**job)


@app.post("/api/v1/schemas", tags=["Schemas"])
async def register_schema(
    request: SchemaRegistryRequest,
    storage: Storage = Depends(get_storage),
    api_key: str = Depends(check_rate_limit),
) -> dict[str, Any]:
    """Register or update a target schema. Requires X-API-Key header."""
    storage.schemas[request.schema_id] = {
        "schema_id": request.schema_id,
        "schema_type": request.schema_type,
        "version": request.version,
        "tables": request.tables,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "status": "registered",
        "schema_id": request.schema_id,
        "version": request.version,
    }


@app.get("/api/v1/schemas", tags=["Schemas"])
async def list_schemas(
    storage: Storage = Depends(get_storage),
    api_key: str = Depends(check_rate_limit),
) -> dict[str, Any]:
    """List all registered schemas. Requires X-API-Key header."""
    return {
        "schemas": list(storage.schemas.values()),
        "count": len(storage.schemas),
    }


@app.get("/api/v1/schemas/{schema_id}", tags=["Schemas"])
async def get_schema(
    schema_id: str,
    storage: Storage = Depends(get_storage),
    api_key: str = Depends(check_rate_limit),
) -> dict[str, Any]:
    """Get a specific schema. Requires X-API-Key header."""
    if schema_id not in storage.schemas:
        raise HTTPException(status_code=404, detail="Schema not found")

    return storage.schemas[schema_id]


# =============================================================================
# Error Handlers
# =============================================================================

# Register centralized error handlers from core module
from aegislang.core.errors import register_error_handlers, create_error_response

register_error_handlers(app)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> None:
    """Run the API server."""
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    workers = int(os.environ.get("WORKERS", "4"))
    reload = os.environ.get("RELOAD", "false").lower() == "true"

    uvicorn.run(
        "aegislang.api.server:app",
        host=host,
        port=port,
        workers=1 if reload else workers,
        reload=reload,
    )


if __name__ == "__main__":
    main()
