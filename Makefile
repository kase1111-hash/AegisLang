# =============================================================================
# AegisLang Makefile
# Automation for development, testing, and deployment tasks
# =============================================================================

.PHONY: help install dev-install test lint format type-check security-check \
        security-scan vuln-scan pre-commit-install pre-commit-run check-all \
        build run clean docker-build docker-up docker-down docker-logs \
        db-init db-migrate docs

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
PYTEST := pytest
DOCKER_COMPOSE := docker-compose
APP_NAME := aegislang
VERSION := $(shell cat VERSION 2>/dev/null || echo "1.0.0")

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# =============================================================================
# Help
# =============================================================================

help: ## Show this help message
	@echo "$(BLUE)AegisLang - Build Automation$(NC)"
	@echo ""
	@echo "$(GREEN)Usage:$(NC) make [target]"
	@echo ""
	@echo "$(YELLOW)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# =============================================================================
# Installation
# =============================================================================

install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)Installation complete!$(NC)"

dev-install: install ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install pytest pytest-asyncio pytest-cov black ruff mypy httpx
	@echo "$(GREEN)Development installation complete!$(NC)"

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTEST) tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTEST) tests/ -v --cov=$(APP_NAME) --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTEST) tests/ -v --ignore=tests/test_integration.py

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTEST) tests/test_integration.py -v

test-fast: ## Run tests without slow markers
	@echo "$(BLUE)Running fast tests...$(NC)"
	$(PYTEST) tests/ -v -m "not slow"

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run linter (ruff)
	@echo "$(BLUE)Running linter...$(NC)"
	ruff check $(APP_NAME)/ tests/
	@echo "$(GREEN)Linting complete!$(NC)"

lint-fix: ## Run linter and fix issues
	@echo "$(BLUE)Running linter with auto-fix...$(NC)"
	ruff check --fix $(APP_NAME)/ tests/
	@echo "$(GREEN)Linting complete!$(NC)"

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	black $(APP_NAME)/ tests/
	@echo "$(GREEN)Formatting complete!$(NC)"

format-check: ## Check code formatting
	@echo "$(BLUE)Checking code format...$(NC)"
	black --check $(APP_NAME)/ tests/

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checker...$(NC)"
	mypy $(APP_NAME)/ --ignore-missing-imports
	@echo "$(GREEN)Type checking complete!$(NC)"

security-check: ## Run security checks (bandit + safety)
	@echo "$(BLUE)Running security checks...$(NC)"
	$(PIP) install bandit safety 2>/dev/null || true
	bandit -r $(APP_NAME)/ -c .bandit.yaml -f txt
	safety check -r requirements.txt --short-report || true
	@echo "$(GREEN)Security checks complete!$(NC)"

security-scan: ## Run comprehensive security scan
	@echo "$(BLUE)Running comprehensive security scan...$(NC)"
	$(PIP) install bandit safety pip-audit 2>/dev/null || true
	@echo "\n$(YELLOW)=== Bandit Security Scan ===$(NC)"
	bandit -r $(APP_NAME)/ -c .bandit.yaml -f txt || true
	@echo "\n$(YELLOW)=== Safety Dependency Check ===$(NC)"
	safety check -r requirements.txt --full-report || true
	@echo "\n$(YELLOW)=== pip-audit CVE Scan ===$(NC)"
	pip-audit -r requirements.txt || true
	@echo "$(GREEN)Security scan complete!$(NC)"

vuln-scan: ## Scan dependencies for vulnerabilities
	@echo "$(BLUE)Scanning dependencies for vulnerabilities...$(NC)"
	$(PIP) install pip-audit safety 2>/dev/null || true
	pip-audit -r requirements.txt
	safety check -r requirements.txt
	@echo "$(GREEN)Vulnerability scan complete!$(NC)"

pre-commit-install: ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	$(PIP) install pre-commit 2>/dev/null || true
	pre-commit install
	@echo "$(GREEN)Pre-commit hooks installed!$(NC)"

pre-commit-run: ## Run pre-commit on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files
	@echo "$(GREEN)Pre-commit complete!$(NC)"

check-all: lint type-check security-check test ## Run all checks (lint, type, security, test)
	@echo "$(GREEN)All checks passed!$(NC)"

quality: lint format-check type-check ## Run all code quality checks

# =============================================================================
# Build & Run
# =============================================================================

build: ## Build the application
	@echo "$(BLUE)Building $(APP_NAME)...$(NC)"
	$(PYTHON) -m py_compile $(APP_NAME)/**/*.py
	@echo "$(GREEN)Build complete!$(NC)"

run: ## Run the API server locally
	@echo "$(BLUE)Starting $(APP_NAME) API server...$(NC)"
	$(PYTHON) -m $(APP_NAME).api.server

run-dev: ## Run the API server in development mode
	@echo "$(BLUE)Starting $(APP_NAME) in development mode...$(NC)"
	RELOAD=true LOG_LEVEL=DEBUG $(PYTHON) -m $(APP_NAME).api.server

# =============================================================================
# Docker
# =============================================================================

docker-build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	$(DOCKER_COMPOSE) build
	@echo "$(GREEN)Docker build complete!$(NC)"

docker-up: ## Start all Docker services
	@echo "$(BLUE)Starting Docker services...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)Services started! API available at http://localhost:8080$(NC)"

docker-up-dev: ## Start Docker services in development mode
	@echo "$(BLUE)Starting Docker services (dev mode)...$(NC)"
	$(DOCKER_COMPOSE) --profile dev up -d
	@echo "$(GREEN)Dev services started! API available at http://localhost:8081$(NC)"

docker-down: ## Stop all Docker services
	@echo "$(BLUE)Stopping Docker services...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)Services stopped!$(NC)"

docker-down-clean: ## Stop services and remove volumes
	@echo "$(RED)Stopping services and removing volumes...$(NC)"
	$(DOCKER_COMPOSE) down -v
	@echo "$(GREEN)Services stopped and volumes removed!$(NC)"

docker-logs: ## View Docker logs
	$(DOCKER_COMPOSE) logs -f

docker-logs-api: ## View API server logs
	$(DOCKER_COMPOSE) logs -f aegislang

docker-ps: ## Show running containers
	$(DOCKER_COMPOSE) ps

docker-shell: ## Open shell in API container
	$(DOCKER_COMPOSE) exec aegislang /bin/bash

# =============================================================================
# Database
# =============================================================================

db-init: ## Initialize the database
	@echo "$(BLUE)Initializing database...$(NC)"
	$(DOCKER_COMPOSE) exec postgres psql -U aegislang -d aegislang -f /docker-entrypoint-initdb.d/init.sql
	@echo "$(GREEN)Database initialized!$(NC)"

db-shell: ## Open database shell
	$(DOCKER_COMPOSE) exec postgres psql -U aegislang -d aegislang

db-reset: ## Reset database (destructive!)
	@echo "$(RED)Resetting database...$(NC)"
	$(DOCKER_COMPOSE) exec postgres dropdb -U aegislang aegislang || true
	$(DOCKER_COMPOSE) exec postgres createdb -U aegislang aegislang
	$(DOCKER_COMPOSE) exec postgres psql -U aegislang -d aegislang -f /docker-entrypoint-initdb.d/init.sql
	@echo "$(GREEN)Database reset complete!$(NC)"

# =============================================================================
# Documentation
# =============================================================================

docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@echo "API documentation available at: docs/API.md"
	@echo "$(GREEN)Documentation complete!$(NC)"

# =============================================================================
# Cleanup
# =============================================================================

clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete!$(NC)"

clean-docker: ## Clean Docker resources
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	docker system prune -f
	@echo "$(GREEN)Docker cleanup complete!$(NC)"

clean-all: clean clean-docker ## Clean everything

# =============================================================================
# CI/CD Helpers
# =============================================================================

ci: lint type-check test ## Run CI pipeline locally
	@echo "$(GREEN)CI pipeline complete!$(NC)"

pre-commit: format lint type-check test-fast ## Run pre-commit checks
	@echo "$(GREEN)Pre-commit checks passed!$(NC)"

# =============================================================================
# Release
# =============================================================================

version: ## Show current version
	@echo "$(APP_NAME) version: $(VERSION)"

tag: ## Create a git tag for current version
	@echo "$(BLUE)Creating tag v$(VERSION)...$(NC)"
	git tag -a v$(VERSION) -m "Release v$(VERSION)"
	@echo "$(GREEN)Tag created! Push with: git push origin v$(VERSION)$(NC)"
