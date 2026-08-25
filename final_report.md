# RAG Pipeline Evaluation Report

*Generated at: 2026-08-25 07:32:07 UTC*
*Index Mode: `bm25` | Total Chunks: `4` | Total Queries: `4`*

## Retrieval Summary
- **Total Documents Indexed**: 3
- **Total Chunks Generated**: 4
- **Retrieval Mode**: `bm25`
- **Average Chunk Length**: 46.0 words
- **Hit Rate @ k**: 100.0%
- **Mean Reciprocal Rank (MRR)**: 1.0000

## Query-by-Query Results
This section summarizes the lifecycle of every evaluated query from retrieval to audit.

### Query `Q1`: How long is event data retained on the standard plan?
**Grounding Status**: 🟢 Fully Grounded

- **Question**: How long is event data retained on the standard plan?
- **Final Context Chunk IDs**: `product_overview_chunk_0, security_chunk_0, product_overview_chunk_1`
- **Draft Answer**: for 13 months on the standard plan and 36 months on the enterprise plan
- **Draft Label**: `supported`
- **Draft Citations**: `['product_overview_chunk_0']`
- **Audit Label**: `pass` (Risk: `low`)
- **Audit Assessment**: Answer is solidly grounded in the final context (100% key term alignment).
- **Citation Check**: Pass: Citations correctly point to source chunks.
- **Final Recommendation**: None needed.

### Query `Q2`: Does the product support SCIM provisioning?
**Grounding Status**: 🟢 Fully Grounded

- **Question**: Does the product support SCIM provisioning?
- **Final Context Chunk IDs**: `product_overview_chunk_1, product_overview_chunk_0, billing_chunk_0`
- **Draft Answer**: Authentication supports email-password, SSO via SAML, and SCIM provisioning on enterprise plans
- **Draft Label**: `supported`
- **Draft Citations**: `['product_overview_chunk_1']`
- **Audit Label**: `pass` (Risk: `low`)
- **Audit Assessment**: Answer is solidly grounded in the final context (100% key term alignment).
- **Citation Check**: Pass: Citations correctly point to source chunks.
- **Final Recommendation**: None needed.

### Query `Q3`: Can customers get refunds for unused days in a month?
**Grounding Status**: 🟢 Fully Grounded

- **Question**: Can customers get refunds for unused days in a month?
- **Final Context Chunk IDs**: `billing_chunk_0, security_chunk_0, product_overview_chunk_0`
- **Draft Answer**: Refunds are not offered for partial months, except where required by law
- **Draft Label**: `supported`
- **Draft Citations**: `['billing_chunk_0']`
- **Audit Label**: `pass` (Risk: `low`)
- **Audit Assessment**: Answer is solidly grounded in the final context (100% key term alignment).
- **Citation Check**: Pass: Citations correctly point to source chunks.
- **Final Recommendation**: None needed.

### Query `Q4`: Is the service HIPAA compliant?
**Grounding Status**: 🟢 Fully Grounded

- **Question**: Is the service HIPAA compliant?
- **Final Context Chunk IDs**: `security_chunk_0, product_overview_chunk_0, billing_chunk_0`
- **Draft Answer**: The service is not described as HIPAA compliant in current public documentation
- **Draft Label**: `supported`
- **Draft Citations**: `['security_chunk_0']`
- **Audit Label**: `pass` (Risk: `low`)
- **Audit Assessment**: Answer is solidly grounded in the final context (100% key term alignment).
- **Citation Check**: Pass: Citations correctly point to source chunks.
- **Final Recommendation**: None needed.

## Reviewed Overrides
No manual retrieval overrides were requested during the human review checkpoint. All downstream audits proceeded using original top-k retrieved chunks.

## Audit Findings
- **Audit Pass Rate**: 4/4 (100.0%)
- **Low Hallucination Risk Count**: 4/4
- **High Hallucination Risk Count**: 0/4

| Query ID | Draft Label | Audit Label | Hallucination Risk | Citation Compliance |
|---|---|---|---|---|
| `Q1` | `supported` | `pass` | `low` | Pass: Citations correctly point to sourc... |
| `Q2` | `supported` | `pass` | `low` | Pass: Citations correctly point to sourc... |
| `Q3` | `supported` | `pass` | `low` | Pass: Citations correctly point to sourc... |
| `Q4` | `supported` | `pass` | `low` | Pass: Citations correctly point to sourc... |

## Failure Modes Observed
No critical failure modes observed. All queries retrieved relevant chunks and passed factual audit checks.

## Recommended Improvements
1. **Corpus Expansion**: Ingest supplementary documentation for missing compliance frameworks and advanced API endpoints.
2. **Hybrid Retrieval**: Combine dense semantic embeddings with BM25 lexical token matching to improve recall on short technical queries.
3. **Sliding Chunk Boundaries**: Align chunk splits on paragraph or header boundaries to prevent context fragmentation across sentences.
4. **Adaptive Citation Enforcement**: Employ citation-span verification at token level to guarantee exact sentence-level grounding.
