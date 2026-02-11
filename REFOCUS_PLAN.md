# AegisLang Refocus Plan

**Date:** February 2026
**Based on:** EVALUATION_REPORT.md findings
**Goal:** Transform AegisLang from a broad-but-shallow prototype into a focused, honest, working product.

---

## Guiding Principle

**Do fewer things. Make them actually work. Prove it with real documents.**

---

## Phase 1: Stop the Bleeding (Test Suite to Green)

**Objective:** Get from 96 passed / 6 failed / 39 errors to 100% green.
**Estimated scope:** ~2 hours of focused work.

### 1.1 Fix the missing dependency (unblocks 43 tests)

**Root cause:** `aegislang/api/server.py:609` uses `Form()` which requires `python-multipart`. Not listed in `requirements.txt`.

**Action:** Add `python-multipart>=0.0.6` to `requirements.txt` under Core Framework.

**Acceptance:** All 19 `test_api.py` tests and all 20 `test_system.py` tests pass.

### 1.2 Fix 2 regression tests calling nonexistent method

**Root cause:** `tests/test_regression.py:214,243` call `mapper._compute_similarity()` which doesn't exist in `SchemaMappingAgent`. The agent has `_cosine_similarity()` (line 655) and `_find_semantic_matches()` (line 634) but no `_compute_similarity`.

**Action:** Delete both `test_empty_entity_mapping` and `test_special_characters_in_entity_names` from `tests/test_regression.py`. These test a method that was never implemented -- they are phantom regression tests for a fix (v0.9.5) that never shipped.

**Acceptance:** 0 failures in `test_regression.py`.

### 1.3 Fix 4 regression tests hitting python-multipart (same root cause as 1.1)

**Root cause:** Same missing `python-multipart` dependency.

**Action:** Resolved by 1.1.

**Acceptance:** `test_concurrent_uploads`, `test_large_request_handling`, `test_api_v1_compatibility`, `test_rapid_sequential_requests` all pass.

### 1.4 Fix performance test collection warnings

**Root cause:** `tests/performance/stress_test.py` defines `TestConfig` and `TestResults` as `@dataclass` classes. Pytest tries to collect them as test classes but fails because they have `__init__` constructors.

**Action:** Rename to `StressConfig` and `StressResults` in `tests/performance/stress_test.py`.

**Root cause 2:** `tests/performance/locustfile.py` imports `locust` which is not in `requirements.txt`.

**Action:** Add `conftest.py` to `tests/performance/` that marks the directory as requiring optional deps, OR add `collect_ignore` in `conftest.py` at tests root. Simplest: add `# pragma: no pytest` or just exclude via `pyproject.toml` testpaths.

**Acceptance:** `python -m pytest tests/ -v` runs clean with 0 errors, 0 failures, 0 warnings about collection.

### Phase 1 Exit Criteria

```bash
python -m pytest tests/ -v --ignore=tests/performance
# Result: ALL PASSED, 0 failures, 0 errors
```

---

## Phase 2: Cut the Dead Weight

**Objective:** Remove unused code, unused dependencies, and dishonest marketing.
**Estimated scope:** ~3 hours.

### 2.1 Delete KEYWORDS.md

**Why:** It's an explicit "LLM-SEO optimization for AI recommendation surfaces" strategy doc. It has nothing to do with the software product and damages credibility for any reviewer.

**Action:** `git rm KEYWORDS.md`

### 2.2 Remove unused dependencies from requirements.txt

Remove these 8 dependencies that have zero imports in the `aegislang/` source tree:

| Dependency | Size Impact | Reason for Removal |
|---|---|---|
| `unstructured>=0.10.0` | Heavy | Zero imports anywhere |
| `spacy>=3.7.0` | ~500MB | Zero imports anywhere |
| `torch>=2.1.0` | ~2GB | Only transitive dep of sentence-transformers; not directly used |
| `pinecone-client>=2.2.0` | Medium | Zero imports anywhere |
| `weaviate-client>=3.24.0` | Medium | Zero imports anywhere |
| `sqlalchemy>=2.0.0` | Medium | Zero imports anywhere |
| `asyncpg>=0.29.0` | Medium | Zero imports anywhere |
| `tenacity>=8.2.0` | Small | Zero imports anywhere |

**Note on torch:** sentence-transformers pulls it in transitively. If sentence-transformers is kept, torch comes along. But sentence-transformers itself has only 1 import (in schema_mapping_agent.py, behind a try/except). Move to optional extras.

**Action:** Create two requirement tiers:
- `requirements.txt` — core runtime (what you actually need to run the tested code path)
- `requirements-ml.txt` — optional ML/embedding dependencies (sentence-transformers, torch, transformers)

### 2.3 Delete dead core modules

| File | LOC | Used By | Action |
|---|---|---|---|
| `aegislang/core/events.py` | 224 | Nothing (only docstring mentions) | Delete |
| `aegislang/core/metrics.py` | 618 | Nothing (zero imports) | Delete |

These are fully implemented modules with zero callers. They represent aspirational architecture. If Agent-OS event bus integration is needed later, re-implement when there's an actual consumer.

**Action:** Delete both files. Remove their exports from `aegislang/core/__init__.py`.

### 2.4 Strip README.md buzzwords

Remove or rewrite every phrase that has zero implementation:

| Phrase | Location | Replacement |
|---|---|---|
| "semantic blockchain technology" | Line 27 | Delete entirely |
| "auditable prose transactions" | Line 27 | Delete entirely |
| "constitutional AI design principles" | Line 39 | "configurable template system" |
| "process legibility" | Line 38 | "clause-to-artifact traceability" |
| "human authorship verification" | Line 38 | Delete |
| "cognitive work value extraction" | Line 40 | "automated semantic extraction" |
| "digital sovereignty" | Line 41 | Delete |
| "owned AI infrastructure" | Line 41 | "self-hostable" |
| "Proof of human work verification layer" | Line 50 | Delete from roadmap |
| "AI learning contracts" | Line 49 | Delete from roadmap |

### 2.5 Remove or minimize ecosystem cross-promotion

The "Part of the NatLangChain Ecosystem" table (README.md lines 54-70) lists 10 repos. Most are equally early-stage or speculative.

**Action:** Replace the 10-repo table with a single sentence: "AegisLang can integrate with Agent-OS for event-driven pipeline orchestration." Remove all references to NatLangChain, synth-mind, value-ledger, boundary-daemon, etc.

### Phase 2 Exit Criteria

- `KEYWORDS.md` does not exist
- `requirements.txt` contains only actively imported dependencies
- `aegislang/core/events.py` and `aegislang/core/metrics.py` do not exist
- README contains zero unimplemented buzzwords
- README ecosystem table is gone
- All tests still pass

---

## Phase 3: Pick One Domain and Prove It

**Objective:** Validate the full pipeline with real regulatory documents through real LLM calls for ONE specific domain.
**Estimated scope:** 1-2 weeks.

### 3.1 Choose the domain: AML/KYC

**Why AML/KYC:**
- The existing test fixture (`tests/test_integration.py:17-57`) already uses an AML policy
- The README example uses KYC-102
- AML regulations (BSA, CDD Rule, FinCEN guidance) are publicly available
- Clause structures are relatively well-defined (obligations, prohibitions, reporting thresholds)
- Easier to validate output than abstract frameworks like NIST CSF

### 3.2 Obtain 3-5 real regulatory source documents

Candidates (all publicly available):
1. **FinCEN CDD Rule** (31 CFR 1010.230) — Customer Due Diligence requirements
2. **BSA/AML Examination Manual** (FFIEC) — Section on Customer Identification Program
3. **FATF Recommendation 10** — Customer Due Diligence
4. **OCC Bulletin 2021-23** — BSA/AML Compliance Program
5. **EU 6th Anti-Money Laundering Directive** (6AMLD) — Key articles

Place in `examples/regulations/` as source documents.

### 3.3 Run full pipeline with real LLM calls

For each document:
1. Ingest with `AegisIngestor`
2. Parse with `PolicyParserAgent` using real Anthropic/OpenAI client (NOT mock)
3. Map with `SchemaMappingAgent` against a realistic AML database schema
4. Compile to YAML, SQL, and Python artifacts
5. Validate with `TraceValidatorAgent`

### 3.4 Build a real target schema for AML

Create a `SchemaRegistry` with actual financial services tables:
- `customers` (id, name, risk_level, identity_verified, verification_date)
- `accounts` (id, customer_id, balance, opened_date, status)
- `transactions` (id, account_id, amount, type, timestamp, suspicious_flag)
- `sar_reports` (id, transaction_id, filed_date, fiu_reference)
- `audit_log` (id, entity_type, entity_id, action, timestamp, actor)

### 3.5 Evaluate output quality

For each generated artifact, answer:
- Would a compliance officer recognize this as a valid interpretation of the source clause?
- Would a DBA deploy this SQL constraint on a production database?
- Does the YAML rule capture the semantic intent of the regulation?
- Are the provenance links accurate and useful?

Document results in `examples/aml_evaluation.md`.

### Phase 3 Exit Criteria

- `examples/regulations/` contains 3+ real regulatory documents
- `examples/output/` contains generated artifacts from real LLM pipeline runs
- `examples/aml_evaluation.md` documents accuracy assessment per artifact
- At least 70% of generated artifacts are judged "semantically correct" by manual review

---

## Phase 4: Honest Versioning and Documentation

**Objective:** Make the project's claims match its capabilities.

### 4.1 Version to 0.1.0

The current `1.0.0` version is not earned. A project that has never processed a real document with real LLM calls is pre-alpha.

**Action:** Change version to `0.1.0` in:
- `pyproject.toml:10`
- `VERSION`
- `aegislang/agents/compiler_agent.py:33`

Update `pyproject.toml` classifier from `"Development Status :: 4 - Beta"` to `"Development Status :: 3 - Alpha"`.

### 4.2 Rewrite README as honest product description

New README structure:
1. **What it does** (2 sentences, no buzzwords)
2. **How it works** (pipeline diagram)
3. **Current status** (what works today, what doesn't)
4. **Quick start** (install, run on example doc)
5. **Supported domain** (AML/KYC, with results from Phase 3)
6. **Architecture** (brief, link to ARCHITECTURE.md)
7. **Contributing**

### 4.3 Update SPEC.md to match reality

The 54KB spec describes systems that don't exist. Trim to document what's actually implemented. Move aspirational sections to a separate `ROADMAP.md` with honest status markers.

### Phase 4 Exit Criteria

- Version is `0.1.0` everywhere
- README accurately describes current capabilities
- No claims without backing implementation
- SPEC.md documents implemented features only

---

## Phase 5: Strengthen What Works

**Objective:** Harden the core pipeline based on Phase 3 learnings.

### 5.1 Improve LLM prompt engineering

Based on Phase 3 results, iterate on:
- `CLAUSE_PARSER_SYSTEM_PROMPT` in `policy_parser_agent.py` — tune for AML domain
- Extraction accuracy for multi-clause sentences
- Handling of cross-references ("per Section 2.1 above")
- Conditional clause nesting

### 5.2 Add output regression tests

For each real document processed in Phase 3, snapshot the expected output and create regression tests that verify:
- Same document produces same clause count
- Clause types are stable across runs
- Critical semantic triples don't drift

### 5.3 Add schema mapping quality metrics

Track and report:
- % of entities mapped vs unmapped
- Average mapping confidence
- Distribution of mapping methods (exact, synonym, semantic)

### 5.4 Implement persistent storage for API

Replace in-memory dicts in `server.py` with SQLite for:
- Job status tracking
- Document metadata
- Generated artifacts

This is the minimum viable persistence for a compliance tool.

### Phase 5 Exit Criteria

- LLM prompts tuned for AML/KYC with documented accuracy
- Regression tests for real document outputs
- API persistence via SQLite
- Schema mapping quality dashboard

---

## Summary: What Gets Cut, What Gets Built

### CUT
- [x] `KEYWORDS.md`
- [x] `aegislang/core/events.py` (224 LOC)
- [x] `aegislang/core/metrics.py` (618 LOC)
- [x] 8 unused dependencies (~3GB install weight)
- [x] README buzzwords (14 unimplemented claims)
- [x] Ecosystem cross-promotion table
- [x] Phantom regression tests for unimplemented methods
- [x] Version 1.0.0 claim

### BUILT
- [x] Green test suite (Phase 1)
- [x] Honest `requirements.txt` (Phase 2)
- [x] Real AML regulatory document processing (Phase 3)
- [x] Quality evaluation with real documents (Phase 3)
- [x] Honest README and versioning (Phase 4)
- [x] Tuned LLM prompts for AML domain (Phase 5)
- [x] Output regression tests (Phase 5)
- [x] SQLite persistence for API (Phase 5)

### NOT BUILT (and that's OK)
- Multi-domain support (GDPR, HIPAA, NIST) — earn it after AML works
- Neo4j persistence — SQLite first, graph DB later
- Agent-OS integration — build the product first, integrate later
- RAG-based retrieval — solve basic extraction first
- Continuous drift detection — manual validation first
