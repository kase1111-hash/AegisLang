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
- Git
- Docker and Docker Compose (optional, only to run the containerized API)

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

3. Optionally copy the environment template and configure:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```
   `.env` is read by Docker Compose only; the Python code does not load it, so `export` variables in your shell (or use a tool such as direnv) when running locally. No LLM key is needed for development - without `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` the API uses the mock parser.

4. Install pre-commit hooks (optional - see the caveats under [Pre-commit Hooks](#pre-commit-hooks)):
   ```bash
   make pre-commit-install
   # Or manually:
   pip install pre-commit
   pre-commit install
   ```

5. Run the API server locally:
   ```bash
   make run        # http://localhost:8080 (in-memory storage)
   make run-dev    # auto-reload, debug logging
   ```
   No external services are required: the API keeps data in memory by default, or in a local SQLite file with `AEGISLANG_STORAGE_BACKEND=sqlite`. `make docker-up` starts the same API in Docker (with SQLite on a named volume); there is no database to initialize. Unless you set `AEGISLANG_API_KEYS` or `AEGISLANG_DISABLE_AUTH=true`, the server generates a development API key at startup and logs it (`no_api_keys_configured`).

## Code Style

We use automated tools to maintain consistent code style:

### Linting and Formatting

- **Ruff**: Linter
- **Black**: Formatter (100 char line length)
- **MyPy**: Static type checking

Run all checks:
```bash
make check-all   # lint, type-check, security-check, test
```

Or individually:
```bash
make lint          # Run Ruff linter (aegislang/ and tests/)
make format        # Format code with Black
make format-check  # Check formatting with Black
make type-check    # Run MyPy
make security-check  # Bandit + Safety
```

**Known lint debt:** the codebase does not pass these checks yet. `ruff check .` reports about 367 findings, `black --check .` would reformat about two dozen files, and MyPy reports about 58 errors, so `make lint`, `make format-check`, `make type-check` and `make check-all` currently fail on code you did not touch. Please make sure your change does not add new findings (for example, run `ruff check` and `black --check` on the files you changed), and avoid mixing large reformatting with functional changes. Bandit (`make security-check`) and the test suite pass and should stay that way.

### Style Guidelines

- Use type hints for all function parameters and return values
- Write docstrings for public functions and classes
- Follow PEP 8 conventions (checked by Ruff)
- Maximum line length: 100 characters
- Use double quotes for strings

### Pre-commit Hooks

Once installed, pre-commit hooks run on each commit. The hooks include:

- Ruff (linting with `--fix`, and `ruff-format`)
- MyPy (type checking)
- Bandit (security scanning)
- detect-secrets and detect-private-key
- A local Safety dependency check (for changes to `requirements*.txt`)
- Various file checks (trailing whitespace, YAML validation, etc.)

Caveats with the current configuration:

- The `detect-secrets` hook expects a `.secrets.baseline` file, which is not committed. Create it first with `pip install detect-secrets && detect-secrets scan > .secrets.baseline`, or skip the hook (`SKIP=detect-secrets git commit ...`).
- The `mypy` hook fails because of the existing type errors (see Known lint debt above); use `SKIP=mypy` if it blocks an unrelated change.
- The `safety-check` hook runs the `safety` command from your environment, so install it (`pip install safety`) or skip it.

If a hook fails on code you changed, fix the issues and re-commit.

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test types
make test-unit         # everything except tests/test_integration.py
make test-integration  # tests/test_integration.py
make test-fast         # skip tests marked slow

# Run tests matching a pattern
pytest tests/ -k "test_ingest"
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Use pytest fixtures for common setup
- Mark slow-running tests with `@pytest.mark.slow` (excluded by `make test-fast`). `pyproject.toml` also declares `integration`, `unit` and `security` markers, but the suite does not currently use them; test selection is by file (see the Makefile targets above).
- Load tests live in `tests/performance/` (Locust) and are not part of the normal test run.

### Coverage Requirements

- Minimum coverage threshold: 70% (enforced by `fail_under` in `pyproject.toml`; current total is about 73%)
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
2. Check that your change adds no new lint, formatting or type findings (see Known lint debt above) and that `make security-check` passes
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

3. **Check CI**: The CI workflow runs the lint job (`ruff check .`, `black --check .`, and MyPy as non-blocking) and the test job (`pytest` with coverage). The test job must pass. The Ruff and Black steps currently fail on `main` because of the pre-existing lint debt; make sure your change does not add new findings

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
