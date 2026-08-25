PYTHON ?= $(shell if [ -f .venv/bin/python ]; then echo .venv/bin/python; elif command -v python3 >/dev/null 2>&1; then echo python3; else echo python; fi)
PYTEST ?= $(shell if [ -f .venv/bin/pytest ]; then echo .venv/bin/pytest; elif command -v pytest >/dev/null 2>&1; then echo pytest; else echo $(PYTHON) -m pytest; fi)

.PHONY: run run-non-interactive validate test clean help

help:
	@echo "Available commands:"
	@echo "  make run                   - Run interactive RAG pipeline"
	@echo "  make run-non-interactive   - Run pipeline non-interactively"
	@echo "  make validate              - Run pipeline validation checks"
	@echo "  make test                  - Run pytest test suite"
	@echo "  make clean                 - Remove all generated artifacts"

run:
	$(PYTHON) pipeline.py

run-non-interactive:
	$(PYTHON) pipeline.py --non-interactive

validate:
	$(PYTHON) validate.py

test:
	$(PYTEST) tests/ -v

clean:
	rm -f chunks.json index_metadata.json retrieval_results.json draft_answers.json review_overrides.json answer_audit.json final_report.md retrieval_metrics.json revised_answers.json retrieval_error_analysis.json llm_calls.jsonl
