# AegisLang Troubleshooting Guide

This guide helps resolve common issues with AegisLang installation, configuration, and operation.

---

## Table of Contents

- [Installation Issues](#installation-issues)
- [Docker Issues](#docker-issues)
- [Database Issues](#database-issues)
- [API Issues](#api-issues)
- [Processing Issues](#processing-issues)
- [Performance Issues](#performance-issues)
- [LLM/AI Issues](#llmai-issues)

---

## Installation Issues

### Python version mismatch

**Symptom:**
```
SyntaxError: invalid syntax
```
or
```
ModuleNotFoundError: No module named 'typing_extensions'
```

**Solution:**
Ensure Python 3.11+ is installed:
```bash
python --version  # Should be 3.11 or higher

# If using pyenv:
pyenv install 3.11
pyenv local 3.11
```

---

### Dependency installation fails

**Symptom:**
```
ERROR: Could not build wheels for torch
```

**Solution:**
1. Upgrade pip:
```bash
pip install --upgrade pip wheel setuptools
```

2. Install with pre-built wheels:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

### spaCy model not found

**Symptom:**
```
OSError: [E050] Can't find model 'en_core_web_sm'
```

**Solution:**
```bash
python -m spacy download en_core_web_sm
```

---

## Docker Issues

### Container fails to start

**Symptom:**
```
aegislang exited with code 1
```

**Solution:**
1. Check logs:
```bash
docker-compose logs aegislang
```

2. Verify environment variables:
```bash
docker-compose config
```

3. Rebuild the image:
```bash
docker-compose build --no-cache aegislang
```

---

### Port already in use

**Symptom:**
```
Error: bind: address already in use
```

**Solution:**
1. Find the process using the port:
```bash
lsof -i :8080
```

2. Kill the process or change the port in `docker-compose.yml`:
```yaml
ports:
  - "8081:8080"  # Use different host port
```

---

### Database connection refused

**Symptom:**
```
connection refused: postgres:5432
```

**Solution:**
1. Wait for PostgreSQL to be ready:
```bash
docker-compose up -d postgres
docker-compose logs postgres  # Check if it says "ready to accept connections"
```

2. Restart the aegislang service:
```bash
docker-compose restart aegislang
```

3. Check network connectivity:
```bash
docker-compose exec aegislang ping postgres
```

---

### Volume permission issues

**Symptom:**
```
PermissionError: [Errno 13] Permission denied: '/app/artifacts'
```

**Solution:**
```bash
# Fix ownership
sudo chown -R 1000:1000 ./artifacts ./logs

# Or in docker-compose.yml, run as root (not recommended for production):
user: root
```

---

## Database Issues

### PostgreSQL initialization fails

**Symptom:**
```
ERROR: relation "documents" does not exist
```

**Solution:**
1. Reinitialize the database:
```bash
docker-compose down -v  # Remove volumes
docker-compose up -d postgres
docker-compose logs postgres  # Verify init.sql ran
```

2. Manually run init script:
```bash
docker-compose exec postgres psql -U aegislang -d aegislang -f /docker-entrypoint-initdb.d/init.sql
```

---

### Redis connection issues

**Symptom:**
```
ConnectionError: Error connecting to redis://redis:6379
```

**Solution:**
1. Check Redis is running:
```bash
docker-compose ps redis
docker-compose logs redis
```

2. Test Redis connectivity:
```bash
docker-compose exec redis redis-cli ping
# Should return: PONG
```

---

### Neo4j authentication failed

**Symptom:**
```
AuthError: Invalid credentials
```

**Solution:**
Verify credentials match in `docker-compose.yml`:
```yaml
environment:
  - NEO4J_AUTH=neo4j/aegislang123
```

And in application config:
```yaml
NEO4J_PASSWORD=aegislang123
```

---

## API Issues

### 404 Not Found on all endpoints

**Symptom:**
```
{"detail": "Not Found"}
```

**Solution:**
Ensure you're using the correct API path:
```bash
# Correct:
curl http://localhost:8080/api/v1/health

# Incorrect:
curl http://localhost:8080/health
```

---

### 422 Validation Error

**Symptom:**
```json
{"detail": [{"loc": ["body", "file"], "msg": "field required"}]}
```

**Solution:**
Check request format. For file upload:
```bash
# Correct:
curl -X POST http://localhost:8080/api/v1/ingest \
  -F "file=@document.pdf"

# Incorrect (JSON body for file):
curl -X POST http://localhost:8080/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"file": "document.pdf"}'
```

---

### 500 Internal Server Error

**Symptom:**
```json
{"detail": "Internal server error"}
```

**Solution:**
1. Check application logs:
```bash
docker-compose logs aegislang | tail -50
```

2. Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
docker-compose up -d aegislang
```

3. Check Sentry (if configured) for detailed error traces.

---

### CORS errors in browser

**Symptom:**
```
Access to fetch has been blocked by CORS policy
```

**Solution:**
Add your frontend origin to `CORS_ORIGINS`:
```yaml
environment:
  - CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

---

## Processing Issues

### Document ingestion fails

**Symptom:**
```
Error: Failed to parse document
```

**Solutions:**

1. **PDF issues**: Ensure the PDF is not encrypted:
```bash
# Check if PDF requires password
pdfinfo document.pdf
```

2. **Large files**: Increase timeout:
```python
ingestor = AegisIngestor(timeout=300)
```

3. **Corrupt files**: Validate file integrity before upload.

---

### No clauses extracted

**Symptom:**
Document processes but returns 0 clauses.

**Solutions:**

1. Check document content is text-based (not scanned images)

2. Verify LLM API is configured and working:
```bash
curl http://localhost:8080/api/v1/health
# Check llm_status in response
```

3. Lower confidence threshold:
```python
parser = PolicyParserAgent(min_confidence=0.3)
```

---

### Entity mapping fails

**Symptom:**
```
Warning: No mappings found for entity 'customer'
```

**Solutions:**

1. Register a schema first:
```bash
curl -X POST http://localhost:8080/api/v1/schemas \
  -H "Content-Type: application/json" \
  -d '{"schema_id": "my_schema", ...}'
```

2. Lower similarity threshold:
```python
mapper = SchemaMappingAgent(similarity_threshold=0.5)
```

3. Add manual overrides for specific entities.

---

### Template rendering errors

**Symptom:**
```
jinja2.exceptions.UndefinedError: 'clause' is undefined
```

**Solution:**
Verify template syntax and variable names:
```jinja2
{# Correct #}
{{ clause.actor.entity }}

{# Incorrect #}
{{ actor.entity }}
```

---

## Performance Issues

### Slow document processing

**Symptom:**
Documents take >5 minutes to process.

**Solutions:**

1. **Reduce chunk size**:
```python
ingestor = AegisIngestor(chunk_size=500)
```

2. **Use async processing**:
```bash
curl -X POST http://localhost:8080/api/v1/ingest?async=true
```

3. **Scale workers**:
```yaml
environment:
  - WORKERS=8
```

---

### High memory usage

**Symptom:**
Container OOM killed or system slowdown.

**Solutions:**

1. **Limit batch size**:
```python
compiler = CompilerAgent(batch_size=10)
```

2. **Increase container memory**:
```yaml
deploy:
  resources:
    limits:
      memory: 4G
```

3. **Use CPU-only torch** (smaller footprint):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

### Database queries slow

**Symptom:**
API responses take >5 seconds.

**Solutions:**

1. Check indexes exist:
```sql
\d+ documents  -- In psql
```

2. Analyze query performance:
```sql
EXPLAIN ANALYZE SELECT * FROM clauses WHERE doc_id = 'DOC-001';
```

3. Add connection pooling:
```python
engine = create_engine(url, pool_size=20, max_overflow=30)
```

---

## LLM/AI Issues

### API key invalid

**Symptom:**
```
AuthenticationError: Invalid API key
```

**Solution:**
1. Verify key is set:
```bash
echo $ANTHROPIC_API_KEY
```

2. Check key format (should start with `sk-ant-` for Anthropic).

3. Regenerate key in provider dashboard.

---

### Rate limit exceeded

**Symptom:**
```
RateLimitError: Rate limit exceeded
```

**Solutions:**

1. **Add retry logic** (already built-in with tenacity)

2. **Reduce parallel requests**:
```python
parser = PolicyParserAgent(max_concurrent=2)
```

3. **Use caching** for repeated queries.

---

### Model not available

**Symptom:**
```
ModelNotFoundError: claude-3-opus not available
```

**Solution:**
Use an available model:
```python
client = AnthropicClient(model="claude-3-sonnet-20240229")
```

---

### Inconsistent results

**Symptom:**
Same document produces different clauses on re-processing.

**Solutions:**

1. **Set temperature to 0** for deterministic output:
```python
client = AnthropicClient(temperature=0)
```

2. **Use content hash** to detect duplicates:
```python
if doc.content_hash == cached_hash:
    return cached_result
```

---

## Getting Help

### Still stuck?

1. **Check logs** with debug enabled:
```bash
LOG_LEVEL=DEBUG docker-compose up aegislang
```

2. **Search issues**: https://github.com/kase1111-hash/AegisLang/issues

3. **Open new issue** with:
   - AegisLang version (`cat VERSION`)
   - Full error message
   - Steps to reproduce
   - Relevant logs (sanitized)

---

*Last updated: 2024-01-10 | Version: 1.0.0*
