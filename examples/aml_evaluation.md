# AML/KYC Pipeline Evaluation

**Date:** October 2026 (re-run after the parser, document ID and lineage fixes; first run February 2026)
**Pipeline mode:** Mock LLM (no API keys)
**Documents processed:** 3
**Domain:** Anti-Money Laundering / Know Your Customer
**Reproduce:** `python examples/run_aml_pipeline.py` (writes `examples/output/`; results are deterministic apart from timestamps)

---

## Documents

| Document | Source | Sections | Chunks | Clauses | Clause Types |
|----------|--------|----------|--------|---------|--------------|
| FinCEN CDD Rule | 31 CFR 1010.230 | 9 | 5 | 14 | 13 obligation, 1 prohibition |
| FFIEC CIP Manual | BSA/AML Examination Manual | 9 | 6 | 17 | 13 obligation, 3 permission, 1 prohibition |
| FATF Rec. 10 | FATF International Standards | 9 | 6 | 18 | 14 obligation, 2 permission, 2 prohibition |

**Total:** 49 clauses extracted from 3 documents (40 obligation, 5 permission, 4 prohibition; no conditional, definition or exception clauses).

---

## Pipeline Stage Results

### Stage 1: Ingestion

**Result: Strong.** The Markdown parser correctly identifies section hierarchy (H1/H2 headings), preserves section boundaries, and produces stable document IDs: each ID is the normalized file name plus the first 6 hex characters of the content SHA-256 (`FATF_RECOMMENDATION_10_8D5616`, `FFIEC_CIP_MANUAL_CA1E58`, `FINCEN_CDD_RULE_09FC62`), so re-running the pipeline on unchanged documents gives the same IDs. Text chunking with tiktoken fallback works reliably when the tokenizer endpoint is unavailable (offline mode). Each document produced 5-6 text chunks across 9 sections.

**Issues:** None. The ingestor handles these documents cleanly.

### Stage 2: Parsing (Mock LLM)

**Result: Mixed.** The mock LLM client applies keyword-based clause type detection. Indicators are matched on word boundaries and the first match wins, in the order prohibition, definition, exception, permission, obligation, conditional; a sentence that *starts* with "if"/"when"/"where"/"unless" is always conditional.

| Clause Type | Count | Detection Method |
|-------------|-------|-----------------|
| obligation | 40 | "must", "shall", "required", "is/are required to"; also the default when no indicator matches (e.g. "should") |
| permission | 5 | "may", "can", "permitted", "is/are permitted to" |
| prohibition | 4 | "must not", "shall not", "may not", "should not", "prohibited", "is/are prohibited from" |
| conditional | 0 | Sentence starting with "if", "when", "where", "unless" (or those words anywhere when no other indicator matched) |
| definition / exception | 0 | "means", "refers to", "defined as" / "except", "notwithstanding", "exempt from" |

None of the FATF "when ..." sentences is classified as conditional: they start with the actor and contain a modal verb ("Financial institutions should be required to undertake CDD measures when establishing business relationships"), so the modal decides the type and the "when" clause is extracted as the condition trigger.

**Strengths:**
- Correctly identifies obligation clauses (the dominant type in regulatory text)
- Properly detects prohibitions ("Banks must not destroy records", "Staff must not open accounts", "should not open the account", "Simplified measures shall not apply")
- Extracts actor entities ("Financial institutions", "Banks", "Institutions", "Staff")
- Extracts conditions into `condition.trigger` (e.g. "establishing business relationships", "account closure", "account opening")
- Extracts temporal scope on 5 clauses: three "five years" retention durations, one deadline ("within a reasonable time") and one frequency ("on a periodic basis")
- Assigns confidence scores between 0.58 and 0.95

**Weaknesses (mock-specific):**
- Action extraction is simplistic: it takes the first verb after the modal, which sometimes is an auxiliary ("should be required to undertake CDD measures" becomes the rule "be CDD measures")
- Object extraction keeps a short noun phrase and drops qualifiers ("identity" rather than "identity of each customer")
- Some descriptive "may" sentences are classified as permissions ("Non-documentary methods may include contacting the customer ...")
- Some condition triggers are sentence fragments ("for identity verification", "or at the time of account opening")
- No cross-reference resolution ("per Section 2.1 above"; `cross_references` is always empty)
- Conditional nesting is not captured beyond the top-level trigger

**Assessment:** The parsing architecture is sound. With a real LLM, the structured extraction prompts in `CLAUSE_PARSER_SYSTEM_PROMPT` should produce much richer semantic triples (this has not been measured yet). The mock client proves the pipeline plumbing works but does not validate extraction quality.

### Stage 3: Schema Mapping

**Result: Weak (expected in mock mode).**

| Status | Count | Percentage |
|--------|-------|-----------|
| Fully mapped | 1 | 2.0% |
| Partially mapped | 8 | 16.3% |
| Unmapped | 40 | 81.6% |

18.4% of clauses (9 of 49) have at least one mapped entity. At the entity level, 10 of 90 extracted entities were mapped (8 by exact match, 2 by synonym). "Unmapped" clauses have mapping status `needs_review`.

**Why:** The mock LLM extracts generic actors ("Financial institutions") and short action/object phrases rather than specific database entities ("customer", "identity_verified"). The schema mapper tries exact match, synonym match, and embedding similarity - but the runner uses deterministic mock embeddings (hash-based, not semantic), so in practice only exact and synonym matches succeed, and most extracted entities don't align with table/column names in the AML schema.

The mappings that did succeed:
- "identity" -> `customers.identity_verified` (one clause in each document)
- "account" -> `accounts.id`, "risk level" -> `customers.risk_level`, "beneficial owner" -> `customers.beneficial_owner`
- "Staff" -> `audit_log.actor` and "suspicious activity" -> `transactions.suspicious_flag` (synonym); the one fully mapped clause (FinCEN) maps both its actor and object this way

**Assessment:** The schema registry and mapping logic work correctly. The bottleneck is entity extraction quality from the parser (and the lack of real embeddings in the runner). With a real LLM extracting entities like "customer identity", "account", "transaction amount", mapping rates should increase, since the AML schema has rich semantic labels.

### Stage 4: Compilation

**Result: Strong structurally, weak semantically.**

| Format | Artifacts | Structure Valid |
|--------|-----------|----------------|
| YAML | 49 | Yes (parseable YAML with traceability headers) |
| SQL | 49 | Yes (passes the lenient sqlparse check; 44 constraints, 5 comment-only) |
| Python | 49 | Yes (valid pytest classes; running them gives 88 passed, 16 skipped) |

**Total artifacts:** 147, all syntax-valid

**YAML artifacts:**
- Structured as `control:` with id, type, rule, actor, action, object, trigger (when a condition was found), temporal_scope (when found), severity
- `metadata` carries the source document ID, confidence and generator version
- Severity follows the clause type: prohibition = critical (4), obligation = high (40), others = medium (5)
- The `Confidence` header is the average mapping confidence, so it is 0.5 for the 40 clauses with no mapping
- Every artifact traces back to its source clause ID

**SQL artifacts:**
- Obligations: PostgreSQL `ALTER TABLE ... ADD CONSTRAINT ... CHECK` plus a PL/pgSQL enforcement trigger; prohibitions: `CHECK (NOT (...))`; permissions: comment-only artifacts (no constraint)
- The table comes from the mapped entity path, otherwise the generic `compliance_table`:

| Table | Artifacts |
|-------|-----------|
| `compliance_table` | 35 |
| `customers` | 5 |
| `audit_log` | 2 |
| `accounts` | 1 |
| `transactions` | 1 |
| comment-only (permission clauses) | 5 |

- Check conditions use placeholder columns derived from the action (e.g. `maintain_status = TRUE`), not real schema columns

**Python artifacts:**
- Valid pytest classes with compliant/non-compliant fixtures
- Assertion logic mirrors the obligation/prohibition type
- Runnable as-is with `pytest` (88 passed, 16 skipped), though they test generic placeholder fields rather than domain-specific logic

### Stage 5: Validation

| Status | Count | Percentage |
|--------|-------|-----------|
| Passed | 123 | 83.7% |
| Failed | 0 | 0% |
| Needs review | 24 | 16.3% |

All 147 artifacts pass the chain-completeness and syntax checks. The 24 "needs review" results are the three artifacts of each of 8 clauses whose trace confidence is below the 0.85 threshold: 18 are flagged `below_threshold` (confidence 0.70-0.84) and 6 `low_confidence` (below 0.70). By document: FATF 9, FFIEC 15, FinCEN 0.

---

## Quality Assessment

### Would a compliance officer recognize these as valid interpretations?

**YAML artifacts: Partially.** The YAML rules capture the correct clause type (obligation vs prohibition vs permission), conditions and retention periods, and preserve the source text. A compliance officer would recognize the source regulation. However, the action/object fields are truncated extracts rather than semantic interpretations.

**SQL artifacts: No.** Most SQL references a generic `compliance_table` and placeholder columns. A DBA would not deploy these constraints as-is. With successful schema mapping (real LLM), the SQL would reference `customers.identity_verified` etc.

**Python tests: Partially.** The test structure is sound (compliant vs non-compliant fixtures, obligation assertions). The field names are derived from the mock extraction and would need refinement.

### Are provenance links accurate?

**Yes.** Every validation result's lineage links the artifact back to:
- Source document ID
- Section ID (the real source section, e.g. `FINCEN_CDD_RULE_S009`)
- Chunk ID
- Source clause ID
- Mapping ID
- Artifact ID

Each artifact also carries its clause ID, source document, confidence score and generation timestamp in its header. The traceability chain is the strongest aspect of the pipeline.

---

## Summary

| Criterion | Mock LLM | Expected with Real LLM (estimated) |
|-----------|----------|----------------------|
| Clause extraction | 49 clauses | Similar |
| Type detection accuracy | Not measured (no labelled ground truth); spot checks show descriptive "may" sentences labelled as permissions | Higher (LLM-based) |
| Entity extraction quality | Low (truncated phrases) | High (semantic entities) |
| Schema mapping rate | 18.4% of clauses (10 of 90 entities) | 60-80% |
| Artifact structural validity | 100% (all 147 pass syntax checks) | 100% |
| Artifact semantic accuracy | Low (generic tables and placeholder columns) | Domain-specific |
| Provenance accuracy | 100% | 100% |

### Key Takeaways

1. **The pipeline architecture works end-to-end.** All 5 stages execute correctly and artifacts are generated with full traceability.
2. **The bottleneck is LLM quality, not pipeline plumbing.** Mock extraction is deliberately simplistic; real LLM calls should substantially improve entity extraction and schema mapping.
3. **The AML schema registry is well-designed.** Rich semantic labels and synonyms provide good matching surface for when entity extraction improves.
4. **Traceability is the pipeline's strongest feature.** Every artifact chains back to its source clause, section and document - this is the core value proposition for compliance tooling.
5. **SQL output needs schema-aware mapping to be useful.** Without successful entity-to-table mapping, SQL artifacts reference generic tables.

### Next Steps

- Run pipeline with real Anthropic/OpenAI API keys to validate LLM extraction quality
- Tune `CLAUSE_PARSER_SYSTEM_PROMPT` for AML domain vocabulary
- Add entity extraction examples to few-shot prompts
- ~~Build regression tests from these baseline outputs~~ Done: `tests/test_pipeline_regression.py` checks clause counts, type distributions and mapping success for these documents
