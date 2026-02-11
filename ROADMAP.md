# AegisLang Roadmap

**Current Version:** 0.1.0 (Alpha)

Features are listed by priority. No timeline commitments — each item ships when it's ready and validated.

---

## Near-term (next)

- [ ] **Real LLM validation** — Run pipeline with Anthropic/OpenAI on AML/KYC docs, measure extraction accuracy
- [ ] **LLM prompt tuning** — Improve `CLAUSE_PARSER_SYSTEM_PROMPT` for AML domain vocabulary
- [ ] **SQLite persistence** — Replace in-memory storage in `server.py` with SQLite for jobs, documents, artifacts
- [ ] **Output regression tests** — Snapshot expected outputs from Phase 3 documents as regression tests

## Medium-term

- [ ] **Additional output formats** — Terraform (`policy.tf`) and OPA/Rego (`policy.rego`) compilation targets
- [ ] **Cross-reference resolution** — Handle "per Section 2.1 above" references between clauses
- [ ] **Schema mapping quality metrics** — Track % mapped, average confidence, method distribution
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
- PostgreSQL schema registry (in-memory is sufficient for current scale)
- Kubernetes deployment (Docker Compose is sufficient)
- RBAC / OAuth / JWT auth (API key auth is sufficient for alpha)
- Agent-OS event bus (deleted `events.py`; re-add when there's an actual consumer)
