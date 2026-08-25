# Replayable Mini RAG Pipeline

A deterministic, staged, and observable mini Retrieval-Augmented Generation (RAG) pipeline with grounded answer generation, human-in-the-loop review checkpoints, two-stage auditing, and automated evaluation reporting.

---

## Architecture & Lifecycle Stages

The pipeline strictly enforces monotonic stage progression through an internal state machine ([src/state.py](file:///Users/kandula/Projects/RAG%20Project/src/state.py)):

```text
INIT
 └──> INPUTS_LOADED
       └──> DOCUMENTS_CHUNKED
             └──> INDEX_BUILT
                   └──> RETRIEVAL_COMPLETE
                         └──> DRAFT_ANSWERS_GENERATED
                               └──> HUMAN_REVIEW_COMPLETE
                                     └──> ANSWERS_AUDITED
                                           └──> FINAL_REPORT_GENERATED
                                                 └──> VALIDATION_COMPLETE
                                                       └──> RESULTS_FINALISED
```

The final evaluation report cannot be generated until chunking, retrieval, answer generation, human review, and answer audit have strictly completed in sequence.

---

## Key Features

1. **Deterministic Document Chunking & Indexing**
   - Pure code-based chunking with configurable character size and overlap from `policy.json`.
   - Index builder supporting **BM25**, **Vector (TF-IDF Cosine Similarity)**, and **Hybrid** retrieval modes.
   - Emits structured `chunks.json` and `index_metadata.json`.

2. **Observable Retrieval & Deterministic Metrics**
   - Retrieves top-$k$ chunks per query with deterministic tie-breaking.
   - Automatically computes `Hit@k` and `MRR@k` in `retrieval_metrics.json` when ground-truth annotations are present.

3. **Stage 1 Grounded Draft Generation**
   - Generates answers strictly grounded in retrieved chunks.
   - Strictly enforces policy constraints (`allowed_labels`, `max_citations_per_answer`, `forbidden_behaviours`).
   - Logs every call with SHA-256 prompt hashing to `llm_calls.jsonl`.

4. **Stage 4 Human Review Checkpoint**
   - Displays query retrieval results and draft labels in the terminal.
   - Interactive prompt allowing human reviewers to force/override chunk context for one or more queries before audit.
   - Full CLI override support via `--override "Q1: chunk_id1, chunk_id2"`.
   - Persists decisions to `review_overrides.json` and guarantees that downstream audits use the overridden final context.

5. **Stage 2 Answer Audit & Revision**
   - Evaluates each draft answer sequentially against the post-review final context.
   - Flags missing citations, forbidden claims, and assigns hallucination risk (`low`, `medium`, `high`).
   - Automatically produces conservative revised answers in `revised_answers.json` for any query flagged as `fail` or `high` risk.

6. **Retrieval Error Analysis & Evaluation Report**
   - Classifies failures in `retrieval_error_analysis.json` (`ranking`, `chunking`, `ambiguity`, `corpus_gap`).
   - Generates comprehensive markdown report `final_report.md` with all 6 required sections.

---

## Project Structure

```text
.
├── documents/                     # Input document corpus (.txt files)
│   ├── billing.txt
│   ├── product_overview.txt
│   └── security.txt
├── queries.json                   # Input evaluation queries
├── policy.json                    # Retrieval & answer generation policy
├── pipeline.py                    # Main pipeline orchestrator
├── main.py                        # Entrypoint alias
├── validate.py                    # Comprehensive 27-check validation script
├── Makefile                       # Execution shortcuts
├── pyproject.toml                 # Project metadata & test configuration
├── src/
│   ├── state.py                   # State machine & transition validator
│   ├── chunker.py                 # Deterministic text & directory chunker
│   ├── indexer.py                 # BM25 & Vector retrieval index builder
│   ├── retriever.py               # Top-k retriever & metrics calculator
│   ├── generator.py               # Stage 1 Draft Answer Generator
│   ├── human_review.py            # Stage 4 Human Review & Overrides checkpoint
│   ├── auditor.py                 # Stage 2 Answer Auditor
│   ├── reviser.py                 # Stage 3 Conservative Answer Reviser
│   ├── error_analyzer.py          # Observable Retrieval Error Classifier
│   ├── report_generator.py        # Final Markdown Report Generator
│   └── llm_logger.py              # llm_calls.jsonl structured logger
└── tests/
    ├── test_pipeline.py           # Unit & integration test suite
    └── test_dynamic_fixtures.py   # Dynamic fixture replacement test suite
```

---

## Artifacts Generated

Running the pipeline produces the following artifacts on disk:

| Artifact | Description |
| :--- | :--- |
| `chunks.json` | Deterministic document chunk records with character offsets. |
| `index_metadata.json` | Index statistics, vocabulary size, and active retrieval mode. |
| `retrieval_results.json` | Top-$k$ retrieved chunks with ranking scores for each query. |
| `draft_answers.json` | Stage 1 answers, labels (`supported`, `insufficient_support`, `not_in_corpus`), and citations. |
| `review_overrides.json` | Record of reviewer overrides and final context chunk IDs per query. |
| `answer_audit.json` | Stage 2 audit results, hallucination risk scores, and recommended fixes. |
| `final_report.md` | Final evaluation report with query lifecycles and audit findings. |
| `llm_calls.jsonl` | Append-only structured log of all generation, audit, and revision calls. |
| `retrieval_metrics.json` | *(Optional)* Deterministic `Hit@k` and `MRR@k` evaluation metrics. |
| `revised_answers.json` | *(Optional)* Conservative regenerated answers for failed/high-risk queries. |
| `retrieval_error_analysis.json` | *(Optional)* Observable failure classification. |

---

## Quickstart & Usage

### Prerequisites
- Python 3.9+

### Installation
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (for testing)
pip install pytest
```

---

### Running the Pipeline

#### 1. Interactive Mode (Default)
Prompts the user at Stage 4 to optionally override retrieved chunks for any query:
```bash
make run
# or: python pipeline.py
```

#### 2. Non-Interactive Mode
Runs the entire pipeline end-to-end without pausing:
```bash
make run-non-interactive
# or: python pipeline.py --non-interactive
```

#### 3. CLI Overrides
Supply overrides directly via CLI flags:
```bash
python pipeline.py --non-interactive --override "Q1: billing_chunk_0"
```

#### 4. Configurable Retrieval Mode
Select between BM25, TF-IDF vector, or hybrid retrieval:
```bash
python pipeline.py --mode vector
# or: python pipeline.py --mode hybrid
```

---

## Validation & Testing

### Running Validation Checks
The validation script verifies that all stages were executed, schemas are valid, citations reference only retrieved chunks, downstream audits respect overrides, and timestamps are chronologically ordered:
```bash
make validate
# or: python validate.py
```

### Running Test Suite
Runs unit and integration tests including dynamic fixture replacement tests:
```bash
make test
# or: pytest tests/ -v
```

### Cleaning Artifacts
Remove all generated output files before a fresh run:
```bash
make clean
```

---

## Dynamic Fixture Compatibility

The pipeline is completely decoupled from specific document names, wording, or hardcoded answers. Evaluators can swap out `documents/`, `queries.json`, and `policy.json` with arbitrary fixtures, and the pipeline will chunk, retrieve, answer, audit, and report dynamically.
