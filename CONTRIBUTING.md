# Contributing to AegisLang

Thank you for your interest in contributing to AegisLang! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/AegisLang.git
   cd AegisLang
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/kase1111-hash/AegisLang.git
   ```

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose (for running services)
- Git

### Environment Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   make dev-install
   # Or manually:
   pip install -r requirements.txt
   pip install pytest pytest-cov pytest-asyncio httpx ruff mypy black
   ```

3. Copy the environment template and configure:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Install pre-commit hooks:
   ```bash
   make pre-commit-install
   # Or manually:
   pip install pre-commit
   pre-commit install
   ```

5. Start local services (PostgreSQL, Redis, Neo4j):
   ```bash
   make docker-up
   ```

6. Initialize the database:
   ```bash
   make db-init
   ```

## Code Style

We use automated tools to maintain consistent code style:

### Linting and Formatting

- **Ruff**: Primary linter and formatter
- **Black**: Backup formatter (100 char line length)
- **MyPy**: Static type checking

Run all checks:
```bash
make check-all
```

Or individually:
```bash
make lint        # Run Ruff linter
make format      # Format code with Ruff/Black
make type-check  # Run MyPy
```

### Style Guidelines

- Use type hints for all function parameters and return values
- Write docstrings for public functions and classes
- Follow PEP 8 conventions (enforced by Ruff)
- Maximum line length: 100 characters
- Use double quotes for strings

### Pre-commit Hooks

Pre-commit hooks run automatically on each commit to ensure code quality. The hooks include:

- Ruff (linting and formatting)
- MyPy (type checking)
- Bandit (security scanning)
- Various file checks (trailing whitespace, YAML validation, etc.)

If a hook fails, fix the issues and re-commit.

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test types
make test-unit
make test-integration

# Run tests matching a pattern
pytest tests/ -k "test_ingest"
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Use pytest fixtures for common setup
- Mark tests appropriately:
  - `@pytest.mark.unit` for unit tests
  - `@pytest.mark.integration` for integration tests
  - `@pytest.mark.slow` for slow-running tests
  - `@pytest.mark.security` for security tests

### Coverage Requirements

- Minimum coverage threshold: 70%
- New code should include appropriate tests
- Run `make test-cov` to check coverage

## Submitting Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-template` for new features
- `fix/parser-memory-leak` for bug fixes
- `docs/update-api-guide` for documentation
- `refactor/simplify-mapper` for refactoring

### Commit Messages

Write clear, concise commit messages:
- Use present tense ("Add feature" not "Added feature")
- First line: Brief summary (50 chars or less)
- Blank line, then detailed description if needed
- Reference issues: "Fixes #123" or "Relates to #456"

Example:
```
Add YAML template for conditional rules

Implement new Jinja2 template for generating conditional
rule structures. Supports nested conditions and multiple
action types.

Fixes #42
```

### Before Submitting

1. Ensure all tests pass: `make test`
2. Run all quality checks: `make check-all`
3. Update documentation if needed
4. Rebase on latest upstream main:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

## Pull Request Process

1. **Create a Pull Request** from your feature branch to `main`

2. **Fill out the PR template** completely:
   - Describe the changes
   - Link related issues
   - Include test plan

3. **Ensure CI passes**: All automated checks must pass

4. **Address review feedback**: Make requested changes and push updates

5. **Merge**: Once approved, a maintainer will merge your PR

### PR Guidelines

- Keep PRs focused and reasonably sized
- One logical change per PR
- Include tests for new functionality
- Update documentation as needed
- Respond to review comments promptly

## Reporting Issues

### Bug Reports

When reporting bugs, include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages

### Feature Requests

For feature requests, describe:
- The problem you're trying to solve
- Your proposed solution
- Alternative approaches considered
- Potential impact on existing functionality

## Questions?

- Check the [FAQ](docs/FAQ.md)
- Review the [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
- Open a [Discussion](https://github.com/kase1111-hash/AegisLang/discussions) for general questions

Thank you for contributing to AegisLang!
