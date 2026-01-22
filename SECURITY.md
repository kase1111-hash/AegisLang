# Security Policy

## Supported Versions

The following versions of AegisLang are currently supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

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
- Use strong, unique values for `API_SECRET_KEY` and `JWT_SECRET`
- Rotate API keys and secrets regularly
- Use environment-specific configurations for development, staging, and production

### Deployment

- Run AegisLang behind a reverse proxy (nginx, Traefik)
- Enable TLS/HTTPS in production
- Use the principle of least privilege for database users
- Keep all dependencies up to date
- Monitor logs for suspicious activity

### API Security

- Implement rate limiting for API endpoints
- Validate and sanitize all input data
- Use parameterized queries (handled by default in AegisLang)
- Enable audit logging for sensitive operations

### LLM Integration

- Validate LLM outputs before execution
- Set appropriate token limits
- Monitor API usage for anomalies
- Review generated code before deployment

## Security Features

AegisLang includes several built-in security features:

- **Input Validation**: All policy documents are validated before processing
- **Parameterized Queries**: Database operations use parameterized queries to prevent SQL injection
- **Audit Logging**: All operations are logged with full traceability
- **Confidence Scoring**: Generated artifacts include confidence scores for review
- **Template Sandboxing**: Jinja2 templates run in a sandboxed environment

## Security Tools

We use the following tools to maintain security:

- **Bandit**: Static security analysis for Python
- **Safety**: Dependency vulnerability scanning
- **Trivy**: Container security scanning
- **Pre-commit hooks**: Automated security checks on every commit

Run security checks locally:
```bash
make security-check
```

## Acknowledgments

We appreciate the security research community's efforts in helping keep AegisLang secure. Contributors who responsibly disclose vulnerabilities will be acknowledged in our security advisories (with their permission).

## Contact

For security-related inquiries: **security@aegislang.io**

For general questions: [GitHub Issues](https://github.com/kase1111-hash/AegisLang/issues)
