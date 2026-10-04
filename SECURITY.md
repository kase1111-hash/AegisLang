# Security Policy

## Supported Versions

The following versions of AegisLang are currently supported with security updates:

| Version       | Supported          |
| ------------- | ------------------ |
| 0.1.x (Alpha) | :white_check_mark: |

AegisLang is alpha software; fixes are made on the latest 0.1.x code only.

## Reporting a Vulnerability

We take the security of AegisLang seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: **security@aegislang.io**

Include the following information in your report:
- Type of vulnerability (e.g., SQL injection, XSS, authentication bypass)
- Full path of the affected source file(s)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact assessment of the vulnerability
- Any potential mitigations you've identified

### What to Expect

1. **Acknowledgment**: We will acknowledge receipt of your report within 48 hours.

2. **Assessment**: Our security team will assess the vulnerability and determine its severity and impact.

3. **Updates**: We will keep you informed of our progress toward a fix. You can expect updates at least every 7 days.

4. **Resolution**: Once a fix is ready, we will:
   - Prepare a security patch
   - Coordinate disclosure timing with you
   - Credit you in the security advisory (unless you prefer anonymity)

5. **Public Disclosure**: We aim to resolve critical vulnerabilities within 90 days of the initial report.

### Severity Classification

We use the following severity levels:

| Severity | Response Time | Examples |
|----------|---------------|----------|
| Critical | 24-48 hours | Remote code execution, authentication bypass |
| High | 7 days | SQL injection, sensitive data exposure |
| Medium | 30 days | XSS, CSRF, privilege escalation |
| Low | 90 days | Information disclosure, minor issues |

## Security Best Practices

When using AegisLang, follow these security recommendations:

### Environment Configuration

- Never commit `.env` files or credentials to version control
- Set strong, unique API keys in `AEGISLANG_API_KEYS` (comma-separated). If it is unset, the server generates a random development key at startup and writes it to the log - do not rely on that in production
- Never set `AEGISLANG_DISABLE_AUTH=true` outside local development
- Set `AEGISLANG_ENV=production` so internal error details are not returned to clients
- Rotate API keys and LLM provider keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) regularly
- Use environment-specific configurations for development, staging, and production

### Deployment

- Run AegisLang behind a reverse proxy (nginx, Traefik)
- Enable TLS/HTTPS in production at the proxy (the application itself serves plain HTTP)
- Restrict file-system access to the SQLite database (`AEGISLANG_SQLITE_PATH`) and store it on encrypted storage; the application does not encrypt data at rest
- Keep all dependencies up to date
- Monitor logs for suspicious activity (for example `invalid_api_key_attempt` events)

### API Security

- Tune the built-in per-key rate limits (`AEGISLANG_RATE_LIMIT_MINUTE`, `AEGISLANG_RATE_LIMIT_HOUR`); they are kept in memory per process, so add proxy-level limits if you need more
- Restrict browser origins with `CORS_ORIGINS`
- Use parameterized queries (handled by default in AegisLang)
- All API keys have the same access (there are no roles); add per-user authentication at a gateway if you need it, and retain request logs there

### LLM Integration

- Validate LLM outputs before execution
- Set appropriate token limits
- Monitor API usage for anomalies
- Review generated code before deployment

## Security Features

AegisLang includes several built-in security features:

- **Input Validation**: Request bodies are validated with Pydantic; uploads are checked for file name, extension and size (`AEGISLANG_MAX_FILE_SIZE`, default 50 MB). Only the extension is checked - file contents are not sniffed
- **Authentication and Rate Limiting**: `X-API-Key` authentication and per-key rate limits (429 with `Retry-After`)
- **Parameterized Queries**: SQLite storage uses parameterized queries to prevent SQL injection
- **Logging and Traceability**: Structured logs carry a request ID (`X-Request-ID`), and each compiled document has validation results and a provenance graph (`GET /api/v1/trace/{doc_id}`). There is no user-level audit log
- **Confidence Scoring**: Generated artifacts include confidence scores for review
- **Template Sandboxing**: Jinja2 templates run in a sandboxed environment (`SandboxedEnvironment`)
- **Temporary Upload Handling**: Uploaded files are written to a private (0700) temp directory, overwritten with random bytes and deleted after ingestion

## Security Tools

We use the following tools to maintain security:

- **Bandit**: Static security analysis for Python (run locally; currently reports no issues)
- **Safety**: Dependency vulnerability scanning (run locally; report-only in `make security-check`)
- **Trivy**: Filesystem vulnerability scan of the repository in CI (pushes to `main`/`develop` only, non-blocking)
- **Pre-commit hooks**: Optional local checks, including Bandit and secret detection (see [docs/SECURITY_SCAN.md](docs/SECURITY_SCAN.md) for caveats)

Run security checks locally:
```bash
make security-check   # Bandit + Safety
```

See [docs/SECURITY_SCAN.md](docs/SECURITY_SCAN.md) for current scan results.

## Acknowledgments

We appreciate the security research community's efforts in helping keep AegisLang secure. Contributors who responsibly disclose vulnerabilities will be acknowledged in our security advisories (with their permission).

## Contact

For security-related inquiries: **security@aegislang.io**

For general questions: [GitHub Issues](https://github.com/kase1111-hash/AegisLang/issues)
