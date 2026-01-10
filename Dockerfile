# AegisLang Dockerfile
# Multi-stage build for optimized production image

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# =============================================================================
# Stage 2: Production
# =============================================================================
FROM python:3.11-slim as production

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash aegis

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=aegis:aegis aegislang/ ./aegislang/
COPY --chown=aegis:aegis templates/ ./templates/
COPY --chown=aegis:aegis config.yaml ./

# Create directories for runtime data
RUN mkdir -p /app/artifacts /app/data /app/logs && \
    chown -R aegis:aegis /app

# Switch to non-root user
USER aegis

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AEGISLANG_ENV=production \
    AEGISLANG_LOG_LEVEL=INFO \
    AEGISLANG_TEMPLATES_DIR=/app/templates \
    AEGISLANG_ARTIFACTS_DIR=/app/artifacts

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health')" || exit 1

# Default command: run the API server
CMD ["uvicorn", "aegislang.api:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Stage 3: Development
# =============================================================================
FROM production as development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install dev Python packages
RUN pip install --no-cache-dir \
    pytest>=7.4.0 \
    pytest-asyncio>=0.21.0 \
    pytest-cov>=4.1.0 \
    black>=23.0.0 \
    ruff>=0.1.0 \
    mypy>=1.7.0 \
    ipython>=8.0.0

USER aegis

# Override command for development
CMD ["uvicorn", "aegislang.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
