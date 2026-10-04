# AegisLang Security Scan Report

This document describes the static analysis, type checking, and vulnerability scanning setup for AegisLang, and the current results.

---

## Table of Contents

- [Overview](#overview)
- [Tools Configuration](#tools-configuration)
- [Running Scans](#running-scans)
- [Security Findings](#security-findings)
- [Type Checking Results](#type-checking-results)
- [Linting Results](#linting-results)
- [Remediation Guidelines](#remediation-guidelines)

---

## Overview

AegisLang uses the following static analysis and scanning tools:

| Tool | Purpose | Configuration |
|------|---------|---------------|
| **Ruff** | Python linter | `pyproject.toml` |
| **Black** | Code formatter (`make format`) | `pyproject.toml` |
| **Bandit** | Security vulnerability scanner | `.bandit.yaml` |
| **MyPy** | Static type checker | `pyproject.toml` |
| **Safety** | Dependency vulnerability scanner | CLI (`[tool.safety]` in `pyproject.toml`) |
| **pip-audit** | CVE scanner for dependencies | CLI |
| **Trivy** | Filesystem vulnerability scan in CI | `.github/workflows/ci.yml` |

**Current status in short:** Bandit passes. Ruff, Black and MyPy do **not** pass yet - there is pre-existing lint/type debt (see [Linting Results](#linting-results) and [Type Checking Results](#type-checking-results)).

---

## Tools Configuration

### Ruff (Linter)

Ruff is configured in `pyproject.toml` (`[tool.ruff.lint]`, line length 100) with the following rule sets, among others:

```toml
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # Pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "S",      # flake8-bandit (security)
    "UP",     # pyupgrade
    "PL",     # Pylint
    # ... and more
]
```

**Security-specific rules enabled:**
- `S`: flake8-bandit security checks, except **S101** (use of `assert`), which is in the global `ignore` list
- `B`: bug detection and best practices (B008 is ignored)

### Bandit (Security Scanner)

Bandit configuration in `.bandit.yaml` (abridged):

```yaml
tests: [B101, B102, ..., B703]   # explicit list of enabled tests
exclude_dirs:
  - tests
  - .venv
  - docs
  # ... plus venv, .git, caches, build, dist
severity: low      # Report all severity levels
confidence: low    # Report all confidence levels
```

**Tests included:**
- B101-B112: Assertions, `exec`, file permissions, bind-all-interfaces, hardcoded passwords, temp files, `try/except/pass|continue`
- B201: Flask debug mode
- B301-B324: Dangerous imports and functions
- B401-B413: Import security
- B501-B509: SSL/TLS security
- B601-B611: Injection vulnerabilities
- B701-B703: Template security

### MyPy (Type Checker)

`pyproject.toml` enables a set of strict-leaning flags (it does **not** set `strict = true`), for example:

```toml
[tool.mypy]
strict_equality = true
check_untyped_defs = true
disallow_untyped_defs = true
disallow_any_generics = true
```

The code does not currently pass these settings (see [Type Checking Results](#type-checking-results)).

---

## Running Scans

### Quick Security Scan

```bash
make security-check   # Bandit (fails on findings) + Safety (report only; exit code ignored)
```

### Individual Tools

```bash
# Linting with Ruff (aegislang/ and tests/)
make lint

# Formatting check with Black
make format-check

# Type checking with MyPy
make type-check

# Bandit + Safety + pip-audit, all report-only (never fails)
make security-scan

# Dependency vulnerability scan (pip-audit + Safety; fails on findings)
make vuln-scan

# All checks (lint, type-check, security-check, test)
make check-all   # currently stops at `lint` because of the existing lint debt
```

### Docker-based Scans

The production image contains only the application (no Makefile, tests or scanning tools), so do not run `make` inside it. To scan without a local Python setup, use a throwaway container with the source mounted:

```bash
docker run --rm -v "$PWD":/src -w /src python:3.11-slim \
  sh -c "pip install -q bandit safety && bandit -r aegislang/ -c .bandit.yaml -f txt && safety check -r requirements.txt --short-report"
```

### CI/CD Integration

Bandit and Safety are **not** run in CI. The GitHub Actions workflow (`.github/workflows/ci.yml`) runs:

- **Lint job** (pushes and pull requests): `ruff check . --output-format=github`, `black --check .`, and `mypy aegislang/ --ignore-missing-imports` (MyPy is `continue-on-error`). The Ruff and Black steps currently fail because of the pre-existing lint debt.
- **Security job** (pushes to `main`/`develop` only - skipped for pull requests, runs after the Docker build): a Trivy **filesystem** scan of the repository, with SARIF upload to GitHub code scanning. Both steps are `continue-on-error`, so findings never fail the pipeline.

```yaml
# .github/workflows/ci.yml (security job)
- name: Run Trivy vulnerability scanner
  uses: aquasecurity/trivy-action@master
  with:
    scan-type: 'fs'
    scan-ref: '.'
    format: 'sarif'
    output: 'trivy-results.sarif'
  continue-on-error: true
```

---

## Security Findings

### Scan Summary

| Category | Status | Count | Severity |
|----------|--------|-------|----------|
| SQL Injection | ✅ Mitigated | 0 | N/A |
| Command Injection | ✅ Not present (no subprocess or shell calls) | 0 | N/A |
| XSS | N/A (JSON API; no HTML is rendered) | 0 | N/A |
| Hardcoded Secrets | ✅ Clean (Bandit B105-B107) | 0 | N/A |
| Insecure Crypto | ✅ Clean | 0 | N/A |
| Dependency CVEs | ⚠️ Not tracked continuously - run `make vuln-scan` | - | - |

Bandit reports **no issues** on `aegislang/`. The findings below are known false positives that are suppressed inline with a justification.

### Detailed Findings

#### B608: Hardcoded SQL Expressions

**Status:** Mitigated (suppressed with `# nosec B608`)

`aegislang/api/sqlite_storage.py` builds statements with an f-string for the **table name only**. Table names are constants chosen in code (`jobs`, `documents`, `schemas`, `traces`, `clauses`, `artifacts` and their `*_keys` tables), never user input; all values are bound parameters:

```python
# aegislang/api/sqlite_storage.py
row = self._conn.execute(
    f"SELECT value FROM {self._table} WHERE key = ?", (key,)  # noqa: S608  # nosec B608
).fetchone()
```

#### B104: Binding to All Interfaces

**Status:** Accepted (suppressed with `# nosec B104`)

The server's default `HOST` is `0.0.0.0` for container deployment (`aegislang/api/server.py`). Set `HOST=127.0.0.1` to bind locally.

#### B110: try/except/pass

**Status:** Accepted (suppressed with `# nosec B110`)

Adding a Sentry breadcrumb in `aegislang/core/logging.py` is best effort and must never fail the caller; the `except Exception: pass` is commented and annotated.

#### B602: Subprocess with Shell=True

**Status:** Not present

The package does not use `subprocess`, `os.system` or shell execution.

#### B506: Unsafe YAML Load

**Status:** Mitigated

YAML is only parsed when the compiler checks generated YAML artifacts, using `yaml.safe_load()`:

```python
# aegislang/agents/compiler_agent.py (YAML syntax check)
yaml.safe_load(content)
```

#### B105-B107: Hardcoded Passwords

**Status:** Clean

Credentials are read from environment variables:

```python
# aegislang/agents/policy_parser_agent.py
self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

# aegislang/api/server.py
keys_env = os.environ.get("AEGISLANG_API_KEYS", "")
```

---

## Type Checking Results

### Current Status

`mypy aegislang/ --ignore-missing-imports` (the `make type-check` and CI command) currently reports **58 errors in 9 of 14 source files**. CI runs MyPy with `continue-on-error: true`, so these errors do not block merges.

| Module | Errors |
|--------|--------|
| aegislang/api/server.py | 20 |
| aegislang/agents/compiler_agent.py | 9 |
| aegislang/agents/policy_parser_agent.py | 9 |
| aegislang/core/logging.py | 7 |
| aegislang/agents/aegis_ingestor.py | 6 |
| aegislang/agents/trace_validator_agent.py | 6 |
| aegislang/core/errors.py | 5 |
| aegislang/api/sqlite_storage.py | 3 |
| aegislang/agents/schema_mapping_agent.py | 2 |

Most common error codes: `untyped-decorator` (19, mainly FastAPI route decorators), `no-any-return` (17), `no-untyped-def` (9), `no-untyped-call` (5), `arg-type` (4).

### Type Annotations

Most public functions and methods include type annotations, for example:

```python
# aegislang/agents/policy_parser_agent.py
class PolicyParserAgent:
    def parse_clause(
        self,
        clause_text: str,
        clause_id: str,
        source_chunk_id: str,
    ) -> ParsedClause:
        ...
```

### Known Gaps

- ⚠️ Some functions (notably FastAPI handlers and middleware) lack annotations
- ⚠️ Several functions return `Any` from untyped library calls
- ⚠️ Not all generic types are parameterized

These are tracked as lint debt; they are not security findings.

---

## Linting Results

### Current Status (known lint debt)

| Check | Command | Result |
|-------|---------|--------|
| Ruff | `ruff check .` | ❌ 367 findings (99 auto-fixable) |
| Black | `black --check .` | ❌ 21 files would be reformatted |
| MyPy | `mypy aegislang/ --ignore-missing-imports` | ❌ 58 errors (non-blocking in CI) |

Most frequent Ruff findings: `PLC0415` import-outside-top-level (150), `I001` unsorted imports (36), `PTH123` builtin `open()` (33), `UP017` `datetime.UTC` alias (19), `ARG001` unused function argument (15). There are also 5 unused imports (`F401`).

### Style Conventions

These are the configured targets; the code does not fully meet them yet:

- Double quotes for strings
- 100-character line limit
- Imports sorted (isort rules via Ruff)
- Formatting by Black (`make format`)

---

## Remediation Guidelines

### If Bandit Finds Issues

1. **High Severity**: Fix immediately before merge
2. **Medium Severity**: Fix within the same PR if possible
3. **Low Severity**: Track in issue, fix in next release

### Common Fixes

#### SQL Injection (B608)
```python
# Bad
query = f"SELECT * FROM users WHERE id = {user_id}"

# Good (sqlite3 placeholder)
query = "SELECT * FROM users WHERE id = ?"
cursor.execute(query, (user_id,))
```

#### Command Injection (B602)
```python
# Bad
subprocess.run(f"echo {user_input}", shell=True)

# Good
subprocess.run(["echo", user_input], shell=False)
```

#### Hardcoded Secrets (B105)
```python
# Bad
api_key = "sk-ant-12345"

# Good
api_key = os.environ["API_KEY"]
```

### False Positive Handling

Add an inline comment **on the flagged line**, with a short justification nearby:

```python
# Table name is a constant chosen in code; values are bound parameters
row = conn.execute(f"SELECT value FROM {table} WHERE key = ?", (key,))  # nosec B608
```

Or add to `.bandit.yaml`:

```yaml
skips:
  - B101  # Skip all assert checks
```

---

## Dependency Vulnerabilities

### Scanning Process

```bash
# Using Safety
safety check -r requirements.txt --json

# Using pip-audit
pip-audit --requirement requirements.txt
```

### Current Status

Dependency CVEs are not tracked continuously: `make security-check` runs Safety but ignores its exit code, CI does not run Safety or pip-audit, and the Trivy filesystem scan in CI is non-blocking and skipped on pull requests. `requirements.txt` specifies minimum versions only (for example `fastapi>=0.104.0`, `pydantic>=2.5.0`, `anthropic>=0.8.0`), so results depend on the versions resolved when you install. Run `make vuln-scan` to get the current status.

### Update Policy (target)

- **Critical CVEs**: Patch within 24 hours
- **High CVEs**: Patch within 1 week
- **Medium/Low CVEs**: Patch in next release cycle

---

## Continuous Monitoring

### GitHub Security Features

- **Dependabot**: Not configured (there is no `.github/dependabot.yml`)
- **Code Scanning**: Trivy SARIF results are uploaded to GitHub code scanning on pushes to `main`/`develop` (requires GitHub Advanced Security for private repositories)
- **Secret Scanning**: A GitHub repository setting; not configured in this repository. Locally, the pre-commit `detect-private-key` and `detect-secrets` hooks cover this

### Pre-commit Hooks

Install pre-commit hooks for local scanning:

```bash
pip install pre-commit
pre-commit install
```

`.pre-commit-config.yaml` (abridged):
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.7
    hooks:
      - id: bandit
        args: ["-c", ".bandit.yaml", "-r", "aegislang/"]
  # also: mypy, pre-commit-hooks (whitespace, YAML/JSON/TOML checks,
  # detect-private-key, ...), detect-secrets, validate-pyproject,
  # and a local safety-check hook
```

**Caveats:**
- The `detect-secrets` hook expects a `.secrets.baseline` file, which is not committed. Create one with `detect-secrets scan > .secrets.baseline` before running the hooks.
- The `mypy` and `ruff` hooks fail on the existing lint/type debt when run against all files.
- The local `safety-check` hook uses `language: system`, so `safety` must be installed in your environment.

---

## Attestation

| Check | Last Run | Result |
|-------|----------|--------|
| Ruff Lint | October 2026 | ❌ 367 findings (pre-existing debt) |
| Black Format Check | October 2026 | ❌ 21 files would be reformatted |
| MyPy Type Check | October 2026 | ❌ 58 errors (non-blocking in CI) |
| Bandit Security | October 2026 | ✅ Pass (no issues; false positives suppressed with `# nosec`) |
| Safety CVE Scan | Not recorded | Report-only in `make security-check` |

---

*Generated: February 2026, revised October 2026 | AegisLang v0.1.0*
