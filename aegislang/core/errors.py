"""
AegisLang Error Handling Module

Provides consistent error handling utilities and middleware for the API.
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Callable

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# -----------------------------------------------------------------------------
# Error Response Models
# -----------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    error_code: str | None = Field(default=None, description="Application-specific error code")
    request_id: str | None = Field(default=None, description="Request ID for tracking")
    details: dict[str, Any] | None = Field(default=None, description="Additional error details")


class ValidationErrorDetail(BaseModel):
    """Detail for validation errors."""

    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    type: str = Field(..., description="Error type")


# -----------------------------------------------------------------------------
# Custom Exceptions
# -----------------------------------------------------------------------------


class AegisLangError(Exception):
    """Base exception for AegisLang errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details


class DocumentNotFoundError(AegisLangError):
    """Raised when a document is not found."""

    def __init__(self, doc_id: str):
        super().__init__(
            message=f"Document not found: {doc_id}",
            status_code=404,
            error_code="DOCUMENT_NOT_FOUND",
            details={"doc_id": doc_id},
        )


class JobNotFoundError(AegisLangError):
    """Raised when a job is not found."""

    def __init__(self, job_id: str):
        super().__init__(
            message=f"Job not found: {job_id}",
            status_code=404,
            error_code="JOB_NOT_FOUND",
            details={"job_id": job_id},
        )


class SchemaNotFoundError(AegisLangError):
    """Raised when a schema is not found."""

    def __init__(self, schema_id: str):
        super().__init__(
            message=f"Schema not found: {schema_id}",
            status_code=404,
            error_code="SCHEMA_NOT_FOUND",
            details={"schema_id": schema_id},
        )


class ValidationError(AegisLangError):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: str | None = None):
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details={"field": field} if field else None,
        )


class ProcessingError(AegisLangError):
    """Raised when document processing fails."""

    def __init__(self, message: str, stage: str | None = None):
        super().__init__(
            message=message,
            status_code=500,
            error_code="PROCESSING_ERROR",
            details={"stage": stage} if stage else None,
        )


class RateLimitError(AegisLangError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(
            message=message,
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
            details={"retry_after": retry_after} if retry_after else None,
        )


class AuthenticationError(AegisLangError):
    """Raised for authentication failures."""

    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTHENTICATION_ERROR",
        )


class AuthorizationError(AegisLangError):
    """Raised for authorization failures."""

    def __init__(self, message: str = "Access denied"):
        super().__init__(
            message=message,
            status_code=403,
            error_code="AUTHORIZATION_ERROR",
        )


# -----------------------------------------------------------------------------
# Error Handler Utilities
# -----------------------------------------------------------------------------


def is_production() -> bool:
    """Check if running in production mode."""
    env = os.environ.get("AEGISLANG_ENV", "development").lower()
    return env in ("production", "prod")


def sanitize_error_message(error: Exception, include_details: bool = False) -> str:
    """
    Sanitize error message for external consumption.

    In production, returns generic messages to avoid leaking internal details.
    In development, returns full error messages for debugging.
    """
    if is_production() and not include_details:
        # Map exception types to user-friendly messages
        if isinstance(error, (FileNotFoundError, KeyError)):
            return "The requested resource was not found"
        elif isinstance(error, (ValueError, TypeError)):
            return "Invalid input provided"
        elif isinstance(error, PermissionError):
            return "Access denied"
        elif isinstance(error, TimeoutError):
            return "The operation timed out"
        elif isinstance(error, ConnectionError):
            return "Service temporarily unavailable"
        else:
            return "An internal error occurred"

    return str(error)


def create_error_response(
    error: str | Exception,
    status_code: int,
    error_code: str | None = None,
    request_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """Create a standardized error response."""
    if isinstance(error, Exception):
        message = sanitize_error_message(error)
    else:
        message = error

    # Don't include details in production unless explicitly set
    if is_production() and details is None:
        response_details = None
    else:
        response_details = details

    content = ErrorResponse(
        error=message,
        status_code=status_code,
        error_code=error_code,
        request_id=request_id,
        details=response_details,
    ).model_dump(exclude_none=True)

    return JSONResponse(status_code=status_code, content=content)


# -----------------------------------------------------------------------------
# FastAPI Error Handler Registration
# -----------------------------------------------------------------------------


def register_error_handlers(app: FastAPI) -> None:
    """
    Register all error handlers with the FastAPI application.

    Args:
        app: The FastAPI application instance
    """

    @app.exception_handler(AegisLangError)
    async def aegislang_error_handler(request: Request, exc: AegisLangError) -> JSONResponse:
        """Handle AegisLang custom exceptions."""
        request_id = getattr(request.state, "request_id", None)

        logger.warning(
            "aegislang_error",
            error_code=exc.error_code,
            status_code=exc.status_code,
            message=exc.message,
            request_id=request_id,
        )

        return create_error_response(
            error=exc.message,
            status_code=exc.status_code,
            error_code=exc.error_code,
            request_id=request_id,
            details=exc.details if not is_production() else None,
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        """Handle FastAPI HTTP exceptions."""
        request_id = getattr(request.state, "request_id", None)

        return create_error_response(
            error=exc.detail,
            status_code=exc.status_code,
            request_id=request_id,
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle all other exceptions."""
        request_id = getattr(request.state, "request_id", None)

        # Log the full error internally
        logger.error(
            "unhandled_exception",
            error=str(exc),
            error_type=type(exc).__name__,
            request_id=request_id,
            traceback=traceback.format_exc() if not is_production() else None,
        )

        # Return sanitized error to client
        return create_error_response(
            error=exc,
            status_code=500,
            error_code="INTERNAL_ERROR",
            request_id=request_id,
        )


# -----------------------------------------------------------------------------
# Context Manager for Error Handling
# -----------------------------------------------------------------------------


class ErrorHandlingContext:
    """
    Context manager for consistent error handling in async operations.

    Usage:
        async with ErrorHandlingContext("processing document", logger) as ctx:
            result = await process_document(doc_id)
            ctx.set_result(result)
    """

    def __init__(
        self,
        operation: str,
        logger: Any = None,
        reraise: bool = True,
        default_result: Any = None,
    ):
        self.operation = operation
        self.logger = logger or structlog.get_logger()
        self.reraise = reraise
        self.default_result = default_result
        self._result = default_result
        self._error: Exception | None = None

    async def __aenter__(self) -> "ErrorHandlingContext":
        self.logger.debug(f"starting_{self.operation.replace(' ', '_')}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_val is not None:
            self._error = exc_val
            self.logger.error(
                f"{self.operation.replace(' ', '_')}_failed",
                error=str(exc_val),
                error_type=exc_type.__name__ if exc_type else "Unknown",
            )
            if not self.reraise:
                return True  # Suppress the exception
        else:
            self.logger.debug(f"{self.operation.replace(' ', '_')}_completed")
        return False

    def set_result(self, result: Any) -> None:
        """Set the result of the operation."""
        self._result = result

    @property
    def result(self) -> Any:
        """Get the result (or default if error occurred)."""
        return self._result

    @property
    def error(self) -> Exception | None:
        """Get the error if one occurred."""
        return self._error

    @property
    def success(self) -> bool:
        """Check if operation completed successfully."""
        return self._error is None
