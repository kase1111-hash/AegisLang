# AegisLang Roadmap

**Current Version:** 0.1.0 (Alpha)

Features are listed by priority. No timeline commitments — each item ships when it's ready and validated.

---

## Near-term (next)

- [ ] **Real LLM validation** — Run pipeline with Anthropic/OpenAI on AML/KYC docs, measure extraction accuracy
- [ ] **LLM prompt tuning** — Improve `CLAUSE_PARSER_SYSTEM_PROMPT` for AML domain vocabulary
- [x] **SQLite persistence** — Opt-in SQLite backend (`AEGISLANG_STORAGE_BACKEND=sqlite`) for jobs, documents, schemas, clauses, artifacts and traces; used by `docker compose`. In-memory storage remains the default for `make run`
- [x] **Output regression tests** — `tests/test_pipeline_regression.py` re-runs the AML/KYC documents and checks clause counts, clause-type distributions and mapping success against the baseline

## Medium-term

- [ ] **Additional output formats** — Terraform and OPA/Rego exist as experimental CLI/library formats (Terraform emits a placeholder Sentinel-style policy resource; Rego is a simple render; syntax checks are basic). Remaining: production-quality output and exposure via the API (which accepts only yaml/sql/python)
- [ ] **Cross-reference resolution** — Handle "per Section 2.1 above" references between clauses (`cross_references` is currently always empty)
- [x] **Schema mapping quality metrics** — `MappingQualityReport` tracks % mapped, average confidence, method and status distribution
- [ ] **Batch document processing** — Parallel ingestion of document sets via API
- [ ] **OCR support** — Scanned document ingestion via OCR preprocessing

## Longer-term

- [ ] **Additional domains** — GDPR, HIPAA, NIST CSF evaluation and prompt tuning
- [ ] **Rule drift detection** — Detect policy updates and diff against existing artifacts
- [ ] **Audit chain visualizer** — Web UI for clause-to-artifact lineage exploration
- [ ] **RAG integration** — Retrieval from external regulation databases
- [ ] **Semantic diff engine** — Compare regulation versions and propagate changes
- [ ] **Multilingual support** — EU directives, ISO standards in multiple languages

## Not planned

These were in the original spec but are not on the current roadmap:

- NatLangChain orchestration (build the product first, integrate later)
- Pinecone/Weaviate vector store (mock embeddings work; add when real embedding quality matters)
- PostgreSQL / Redis / Neo4j services (SQLite and in-memory storage are sufficient for current scale; the unused services were removed from Docker Compose, and the validator's Neo4j export helper is not called by the API or CLI)
- Kubernetes deployment (Docker Compose is sufficient)
- RBAC / OAuth / JWT auth (API key auth is sufficient for alpha)
- Agent-OS event bus (deleted `events.py`; re-add when there's an actual consumer)
