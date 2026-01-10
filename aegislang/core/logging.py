"""
AegisLang Core Logging Module

Provides structured logging with optional Sentry integration for error tracking.
Supports multiple output formats and integrates with ELK stack via JSON logging.

Usage:
    from aegislang.core.logging import setup_logging, get_logger

    # Initialize logging (call once at startup)
    setup_logging(
        level="INFO",
        sentry_dsn="https://...@sentry.io/...",  # Optional
        json_output=True,  # For ELK integration
    )

    # Get a logger instance
    logger = get_logger(__name__)
    logger.info("Processing document", doc_id="DOC-001")
    logger.error("Failed to parse", error=str(e), clause_id="CLS-001")
"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, TypeVar

import structlog
from pydantic import BaseModel, Field

# Context variables for request tracing
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


class LogLevel(str, Enum):
    """Supported log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ErrorContext(BaseModel):
    """Structured error context for tracking."""

    error_id: str = Field(..., description="Unique error identifier")
    error_type: str = Field(..., description="Exception class name")
    error_message: str = Field(..., description="Error message")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    module: str = Field(..., description="Module where error occurred")
    function: str = Field(..., description="Function where error occurred")
    line_number: int | None = Field(None, description="Line number")
    stack_trace: str | None = Field(None, description="Full stack trace")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context")
    request_id: str | None = Field(None, description="Request ID if available")
    user_id: str | None = Field(None, description="User ID if available")


class SentryIntegration:
    """Sentry SDK integration for error tracking."""

    _initialized: bool = False
    _sentry_sdk: Any = None

    @classmethod
    def initialize(
        cls,
        dsn: str,
        environment: str = "development",
        release: str | None = None,
        sample_rate: float = 1.0,
        traces_sample_rate: float = 0.1,
    ) -> bool:
        """
        Initialize Sentry SDK.

        Args:
            dsn: Sentry DSN
            environment: Environment name (development, staging, production)
            release: Release version
            sample_rate: Error sampling rate (0.0 to 1.0)
            traces_sample_rate: Performance tracing sample rate

        Returns:
            True if initialization successful
        """
        try:
            import sentry_sdk
            from sentry_sdk.integrations.logging import LoggingIntegration

            sentry_logging = LoggingIntegration(
                level=logging.INFO,
                event_level=logging.ERROR,
            )

            sentry_sdk.init(
                dsn=dsn,
                environment=environment,
                release=release or os.environ.get("AEGISLANG_VERSION", "1.0.0"),
                sample_rate=sample_rate,
                traces_sample_rate=traces_sample_rate,
                integrations=[sentry_logging],
                send_default_pii=False,
            )

            cls._sentry_sdk = sentry_sdk
            cls._initialized = True
            return True

        except ImportError:
            logging.warning("sentry-sdk not installed. Sentry integration disabled.")
            return False
        except Exception as e:
            logging.warning(f"Failed to initialize Sentry: {e}")
            return False

    @classmethod
    def capture_exception(
        cls,
        exception: Exception,
        context: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> str | None:
        """
        Capture an exception to Sentry.

        Returns:
            Sentry event ID if captured, None otherwise
        """
        if not cls._initialized or cls._sentry_sdk is None:
            return None

        try:
            with cls._sentry_sdk.push_scope() as scope:
                if context:
                    for key, value in context.items():
                        scope.set_extra(key, value)
                if tags:
                    for key, value in tags.items():
                        scope.set_tag(key, value)

                # Add request context
                if request_id := request_id_var.get():
                    scope.set_tag("request_id", request_id)
                if user_id := user_id_var.get():
                    scope.set_user({"id": user_id})

                return cls._sentry_sdk.capture_exception(exception)

        except Exception:
            return None

    @classmethod
    def capture_message(
        cls,
        message: str,
        level: str = "info",
        context: dict[str, Any] | None = None,
    ) -> str | None:
        """Capture a message to Sentry."""
        if not cls._initialized or cls._sentry_sdk is None:
            return None

        try:
            with cls._sentry_sdk.push_scope() as scope:
                if context:
                    for key, value in context.items():
                        scope.set_extra(key, value)

                return cls._sentry_sdk.capture_message(message, level=level)

        except Exception:
            return None

    @classmethod
    def add_breadcrumb(
        cls,
        message: str,
        category: str = "log",
        level: str = "info",
        data: dict[str, Any] | None = None,
    ) -> None:
        """Add a breadcrumb for debugging context."""
        if not cls._initialized or cls._sentry_sdk is None:
            return

        try:
            cls._sentry_sdk.add_breadcrumb(
                message=message,
                category=category,
                level=level,
                data=data or {},
            )
        except Exception:
            pass


def add_context_processor(
    logger: structlog.typing.WrappedLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Add request/user context to log events."""
    if request_id := request_id_var.get():
        event_dict["request_id"] = request_id
    if user_id := user_id_var.get():
        event_dict["user_id"] = user_id
    return event_dict


def add_error_context_processor(
    logger: structlog.typing.WrappedLogger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Capture errors to Sentry when logging at error level or above."""
    if method_name in ("error", "critical", "exception"):
        exc_info = event_dict.get("exc_info")

        if exc_info and isinstance(exc_info, tuple) and exc_info[1]:
            exception = exc_info[1]
            context = {k: v for k, v in event_dict.items() if k != "exc_info"}

            # Capture to Sentry
            event_id = SentryIntegration.capture_exception(
                exception,
                context=context,
                tags={"module": event_dict.get("module", "unknown")},
            )

            if event_id:
                event_dict["sentry_event_id"] = event_id

    return event_dict


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    sentry_dsn: str | None = None,
    sentry_environment: str = "development",
    log_file: str | None = None,
) -> None:
    """
    Configure structured logging for AegisLang.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: Output JSON format (for ELK integration)
        sentry_dsn: Sentry DSN for error tracking
        sentry_environment: Sentry environment name
        log_file: Optional file path for logging
    """
    # Get level from environment or parameter
    log_level = os.environ.get("AEGISLANG_LOG_LEVEL", level).upper()

    # Initialize Sentry if DSN provided
    if sentry_dsn or os.environ.get("SENTRY_DSN"):
        SentryIntegration.initialize(
            dsn=sentry_dsn or os.environ.get("SENTRY_DSN", ""),
            environment=os.environ.get("AEGISLANG_ENV", sentry_environment),
        )

    # Configure processors
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        add_context_processor,
        add_error_context_processor,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        # JSON output for ELK/production
        shared_processors.append(structlog.processors.format_exc_info)
        renderer = structlog.processors.JSONRenderer()
    else:
        # Console output for development
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=shared_processors,
        )
    )

    # Add file handler if specified
    handlers = [handler]
    if log_file or os.environ.get("AEGISLANG_LOG_FILE"):
        file_path = log_file or os.environ.get("AEGISLANG_LOG_FILE")
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
                foreign_pre_chain=shared_processors,
            )
        )
        handlers.append(file_handler)

    root_logger = logging.getLogger()
    root_logger.handlers = handlers
    root_logger.setLevel(getattr(logging, log_level))

    # Set levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Bound structlog logger
    """
    return structlog.get_logger(name)


def log_error(
    logger: structlog.stdlib.BoundLogger,
    error: Exception,
    message: str | None = None,
    **context: Any,
) -> ErrorContext:
    """
    Log an error with full context and Sentry capture.

    Args:
        logger: Logger instance
        error: Exception to log
        message: Optional custom message
        **context: Additional context

    Returns:
        ErrorContext with error details
    """
    import uuid

    # Extract error details
    tb = traceback.extract_tb(error.__traceback__)
    last_frame = tb[-1] if tb else None

    error_context = ErrorContext(
        error_id=str(uuid.uuid4()),
        error_type=type(error).__name__,
        error_message=str(error),
        timestamp=datetime.now(timezone.utc).isoformat(),
        module=last_frame.filename if last_frame else "unknown",
        function=last_frame.name if last_frame else "unknown",
        line_number=last_frame.lineno if last_frame else None,
        stack_trace=traceback.format_exc(),
        context=context,
        request_id=request_id_var.get(),
        user_id=user_id_var.get(),
    )

    # Log the error
    logger.error(
        message or error_context.error_message,
        error_id=error_context.error_id,
        error_type=error_context.error_type,
        exc_info=(type(error), error, error.__traceback__),
        **context,
    )

    return error_context


def log_exception(func: F) -> F:
    """
    Decorator to automatically log exceptions from functions.

    Usage:
        @log_exception
        def my_function():
            ...
    """
    logger = get_logger(func.__module__)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_error(
                logger,
                e,
                f"Exception in {func.__name__}",
                function=func.__name__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys()),
            )
            raise

    return wrapper  # type: ignore


def log_async_exception(func: F) -> F:
    """
    Decorator to automatically log exceptions from async functions.

    Usage:
        @log_async_exception
        async def my_async_function():
            ...
    """
    logger = get_logger(func.__module__)

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            log_error(
                logger,
                e,
                f"Exception in {func.__name__}",
                function=func.__name__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys()),
            )
            raise

    return wrapper  # type: ignore


def set_request_context(request_id: str, user_id: str | None = None) -> None:
    """Set request context for logging."""
    request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)

    # Add Sentry breadcrumb
    SentryIntegration.add_breadcrumb(
        message="Request started",
        category="request",
        data={"request_id": request_id, "user_id": user_id},
    )


def clear_request_context() -> None:
    """Clear request context."""
    request_id_var.set(None)
    user_id_var.set(None)


# Export convenience functions
__all__ = [
    "setup_logging",
    "get_logger",
    "log_error",
    "log_exception",
    "log_async_exception",
    "set_request_context",
    "clear_request_context",
    "SentryIntegration",
    "ErrorContext",
    "LogLevel",
]
