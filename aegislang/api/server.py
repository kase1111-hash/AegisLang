"""
AegisLang REST API Server

Provides REST API endpoints for document ingestion, parsing, mapping,
compilation, and validation operations.

Base URL: /api/v1
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

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
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
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
    """Simple in-memory storage for jobs and documents."""

    def __init__(self):
        self.jobs: dict[str, dict[str, Any]] = {}
        self.documents: dict[str, dict[str, Any]] = {}
        self.clauses: dict[str, list[dict[str, Any]]] = {}
        self.artifacts: dict[str, list[dict[str, Any]]] = {}
        self.schemas: dict[str, dict[str, Any]] = {}

    def create_job(self, job_type: str) -> str:
        """Create a new job and return its ID."""
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


storage = Storage()


def get_storage() -> Storage:
    """Dependency to get storage instance."""
    return storage


# =============================================================================
# Background Tasks
# =============================================================================

async def process_ingestion(
    job_id: str,
    file_path: Path,
    metadata: dict[str, Any],
    storage: Storage,
) -> None:
    """Background task for document ingestion."""
    try:
        storage.update_job(job_id, JobStatus.PROCESSING)

        # Import and run ingestor
        from aegislang.agents.aegis_ingestor import AegisIngestor

        ingestor = AegisIngestor()
        result = ingestor.ingest(file_path)

        # Store document
        doc_data = result.model_dump()
        doc_data["metadata"].update(metadata)
        storage.documents[result.doc_id] = doc_data

        storage.update_job(
            job_id,
            JobStatus.COMPLETED,
            result={"doc_id": result.doc_id, "sections": len(result.sections)},
        )

        logger.info("ingestion_completed", job_id=job_id, doc_id=result.doc_id)

    except Exception as e:
        logger.error("ingestion_failed", job_id=job_id, error=str(e))
        storage.update_job(job_id, JobStatus.FAILED, error=str(e))

    finally:
        # Cleanup temp file
        if file_path.exists():
            file_path.unlink()


async def process_compilation(
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

        parser = PolicyParserAgent(use_mock=True)
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
            use_mock=True,
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


@app.post("/api/v1/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: str = Form(default="{}"),
    storage: Storage = Depends(get_storage),
) -> IngestResponse:
    """
    Upload and parse a new policy document.

    Accepts PDF, DOCX, Markdown, or HTML files.
    """
    # Validate file type
    allowed_extensions = {".pdf", ".docx", ".md", ".markdown", ".html", ".htm"}
    file_ext = Path(file.filename or "").suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}",
        )

    # Parse metadata
    try:
        meta_dict = json.loads(metadata)
    except json.JSONDecodeError:
        meta_dict = {}

    # Save file to temp location
    temp_dir = Path(tempfile.gettempdir()) / "aegislang"
    temp_dir.mkdir(exist_ok=True)

    temp_path = temp_dir / f"{uuid.uuid4().hex}{file_ext}"
    content = await file.read()
    temp_path.write_bytes(content)

    # Create job
    job_id = storage.create_job("ing")
    doc_id = Path(file.filename or "document").stem.upper().replace(" ", "_")
    doc_id = f"{doc_id}_{uuid.uuid4().hex[:6].upper()}"

    # Schedule background task
    background_tasks.add_task(
        process_ingestion,
        job_id,
        temp_path,
        meta_dict,
        storage,
    )

    return IngestResponse(
        status="accepted",
        job_id=job_id,
        doc_id=doc_id,
        webhook_url=f"/api/v1/jobs/{job_id}",
    )


@app.get("/api/v1/documents/{doc_id}", tags=["Documents"])
async def get_document(
    doc_id: str,
    storage: Storage = Depends(get_storage),
) -> dict[str, Any]:
    """Retrieve document metadata and processing status."""
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
) -> dict[str, Any]:
    """List all parsed clauses for a document."""
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
) -> dict[str, Any]:
    """Retrieve generated rule artifact for a clause."""
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
) -> dict[str, Any]:
    """Trigger full compilation pipeline for a document."""
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
) -> JobResponse:
    """Check async job status."""
    if job_id not in storage.jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = storage.jobs[job_id]
    return JobResponse(**job)


@app.post("/api/v1/schemas", tags=["Schemas"])
async def register_schema(
    request: SchemaRegistryRequest,
    storage: Storage = Depends(get_storage),
) -> dict[str, Any]:
    """Register or update a target schema."""
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
) -> dict[str, Any]:
    """List all registered schemas."""
    return {
        "schemas": list(storage.schemas.values()),
        "count": len(storage.schemas),
    }


@app.get("/api/v1/schemas/{schema_id}", tags=["Schemas"])
async def get_schema(
    schema_id: str,
    storage: Storage = Depends(get_storage),
) -> dict[str, Any]:
    """Get a specific schema."""
    if schema_id not in storage.schemas:
        raise HTTPException(status_code=404, detail="Schema not found")

    return storage.schemas[schema_id]


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error("unhandled_exception", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
        },
    )


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
