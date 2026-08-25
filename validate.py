"""
Validation script for the Replayable Mini RAG Pipeline.
Verifies all constraints, stages, artifacts, schemas, citation disciplines, audit logs, and reports.
Run with: python validate.py
"""
from datetime import datetime
import json
import os
import sys
from typing import Any, Dict, List, Tuple


class PipelineValidator:
    def __init__(
        self,
        docs_dir: str = "documents",
        queries_file: str = "queries.json",
        policy_file: str = "policy.json",
        chunks_file: str = "chunks.json",
        index_metadata_file: str = "index_metadata.json",
        retrieval_file: str = "retrieval_results.json",
        draft_answers_file: str = "draft_answers.json",
        overrides_file: str = "review_overrides.json",
        audit_file: str = "answer_audit.json",
        report_file: str = "final_report.md",
        log_file: str = "llm_calls.jsonl",
    ):
        self.docs_dir = docs_dir
        self.queries_file = queries_file
        self.policy_file = policy_file
        self.chunks_file = chunks_file
        self.index_metadata_file = index_metadata_file
        self.retrieval_file = retrieval_file
        self.draft_answers_file = draft_answers_file
        self.overrides_file = overrides_file
        self.audit_file = audit_file
        self.report_file = report_file
        self.log_file = log_file

        self.failures: List[str] = []
        self.passes: List[str] = []

    def check(self, condition: bool, success_msg: str, fail_msg: str) -> bool:
        if condition:
            self.passes.append(success_msg)
            print(f"  [PASS] {success_msg}")
            return True
        else:
            self.failures.append(fail_msg)
            print(f"  [FAIL] {fail_msg}")
            return False

    def validate_all(self) -> bool:
        print("\n" + "=" * 70)
        print("           RUNNING RAG PIPELINE VALIDATION CHECKS")
        print("=" * 70)

        # 1. Check Artifacts Existence
        print("\n--- 1. Checking Artifact Files Existence ---")
        for fpath in [
            self.docs_dir,
            self.queries_file,
            self.policy_file,
            self.chunks_file,
            self.index_metadata_file,
            self.retrieval_file,
            self.draft_answers_file,
            self.overrides_file,
            self.audit_file,
            self.report_file,
            self.log_file,
        ]:
            self.check(os.path.exists(fpath), f"Artifact exists: '{fpath}'", f"Required artifact missing: '{fpath}'")

        if self.failures:
            print("\nCritical files missing, cannot proceed with content validation.")
            return False

        # Load data
        with open(self.queries_file, "r", encoding="utf-8") as f:
            queries_data = json.load(f)
        with open(self.policy_file, "r", encoding="utf-8") as f:
            policy_data = json.load(f)
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        with open(self.index_metadata_file, "r", encoding="utf-8") as f:
            index_metadata = json.load(f)
        with open(self.retrieval_file, "r", encoding="utf-8") as f:
            retrieval_results = json.load(f)
        with open(self.draft_answers_file, "r", encoding="utf-8") as f:
            draft_answers = json.load(f)
        with open(self.overrides_file, "r", encoding="utf-8") as f:
            review_overrides = json.load(f)
        with open(self.audit_file, "r", encoding="utf-8") as f:
            audit_results = json.load(f)
        with open(self.report_file, "r", encoding="utf-8") as f:
            report_text = f.read()

        # 2. Validate Chunks Schema & Determinism
        print("\n--- 2. Validating Chunks Schema & Structure ---")
        self.check(isinstance(chunks_data, list) and len(chunks_data) > 0, "chunks.json is a non-empty list", "chunks.json must be a non-empty list")
        all_cids = set()
        chunk_schema_valid = True
        for c in chunks_data:
            required_keys = {"chunk_id", "document_name", "start_char", "end_char", "text"}
            if not required_keys.issubset(c.keys()):
                chunk_schema_valid = False
                break
            all_cids.add(c["chunk_id"])
        self.check(chunk_schema_valid, "All chunk records match schema (chunk_id, document_name, start_char, end_char, text)", "Some chunk records fail schema")

        # 3. Validate Index Metadata
        print("\n--- 3. Validating Index Metadata ---")
        mode = index_metadata.get("retrieval_mode")
        self.check(mode in ["bm25", "vector", "hybrid"], f"Index metadata specifies valid mode: '{mode}'", "Invalid or missing retrieval_mode in index_metadata.json")
        self.check(index_metadata.get("total_chunks") == len(chunks_data), "Index total_chunks matches chunk count", "Index total_chunks mismatch")

        # 4. Validate Retrieval Results
        print("\n--- 4. Validating Retrieval Results Coverage ---")
        query_ids = [q["query_id"] for q in queries_data.get("queries", [])]
        retrieval_qids = [r["query_id"] for r in retrieval_results]
        self.check(set(query_ids) == set(retrieval_qids), f"All {len(query_ids)} queries have retrieval results", "Missing queries in retrieval_results.json")

        all_queries_have_chunks = True
        for r in retrieval_results:
            if not r.get("retrieved_chunks") or len(r["retrieved_chunks"]) == 0:
                all_queries_have_chunks = False
                break
        self.check(all_queries_have_chunks, "Every query has at least one retrieved chunk", "Some query has empty retrieved_chunks")

        # 5. Validate Draft Answers & Policy Constraints
        print("\n--- 5. Validating Draft Answers & Policy Compliance ---")
        allowed_labels = set(policy_data.get("answer_policy", {}).get("allowed_labels", []))
        max_citations = policy_data.get("answer_policy", {}).get("max_citations_per_answer", 3)

        retrieval_map = {r["query_id"]: {c["chunk_id"] for c in r["retrieved_chunks"]} for r in retrieval_results}
        labels_valid = True
        citations_valid = True

        for draft in draft_answers:
            qid = draft["query_id"]
            label = draft.get("label")
            citations = draft.get("citations", [])

            if label not in allowed_labels:
                labels_valid = False
            valid_retrieved_cids = retrieval_map.get(qid, set())
            for cid in citations:
                if cid not in valid_retrieved_cids:
                    citations_valid = False
            if len(citations) > max_citations:
                citations_valid = False

        self.check(labels_valid, f"All draft answer labels use allowed labels ({allowed_labels})", f"Invalid labels found in draft_answers.json")
        self.check(citations_valid, "All citations reference ONLY retrieved chunk IDs for that query and respect max limit", "Citation violation found in draft_answers.json")

        # 6. Validate Human Review Overrides & Downstream Audit Context
        print("\n--- 6. Validating Human Review Overrides & Audit Inputs ---")
        overrides_list = review_overrides.get("overrides", [])
        self.check(len(overrides_list) == len(query_ids), "review_overrides.json contains records for all queries", "Missing queries in review_overrides.json")

        final_context_map = {o["query_id"]: o["final_context_chunk_ids"] for o in overrides_list}
        audit_qids = [a["query_id"] for a in audit_results]
        self.check(set(query_ids) == set(audit_qids), "Every query has an audit result in answer_audit.json", "Missing queries in answer_audit.json")

        # 7. Validate LLM Call Logs
        print("\n--- 7. Validating LLM Call Logging (llm_calls.jsonl) ---")
        log_records: List[Dict[str, Any]] = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    log_records.append(json.loads(line))

        self.check(len(log_records) > 0, f"llm_calls.jsonl contains {len(log_records)} call records", "llm_calls.jsonl is empty")

        stage1_calls = [r for r in log_records if r.get("stage") == "STAGE_1_DRAFT_GENERATION"]
        stage2_calls = [r for r in log_records if r.get("stage") == "STAGE_2_ANSWER_AUDIT"]

        self.check(len(stage1_calls) == len(query_ids), f"Stage 1 draft calls recorded for all {len(query_ids)} queries", "Missing Stage 1 call records")
        self.check(len(stage2_calls) == len(query_ids), f"Stage 2 audit calls recorded for all {len(query_ids)} queries", "Missing Stage 2 audit call records")

        # Verify call schema
        log_schema_valid = True
        for rec in log_records:
            req = {"stage", "query_id", "timestamp", "provider", "model", "prompt_hash", "input_artifacts", "output_artifact"}
            if not req.issubset(rec.keys()):
                log_schema_valid = False
                break
        self.check(log_schema_valid, "All LLM log records conform to required schema", "LLM log record schema mismatch")

        # Verify temporal ordering (Stage 1 before Stage 2)
        stage1_times = [datetime.fromisoformat(r["timestamp"]) for r in stage1_calls]
        stage2_times = [datetime.fromisoformat(r["timestamp"]) for r in stage2_calls]
        if stage1_times and stage2_times:
            self.check(min(stage2_times) >= min(stage1_times), "Stage 2 audit calls executed chronologically after Stage 1 draft generation", "Chronological order violation in LLM logs")

        # 8. Validate Final Evaluation Report
        print("\n--- 8. Validating Final Evaluation Report (final_report.md) ---")
        required_sections = [
            "Retrieval Summary",
            "Query-by-Query Results",
            "Reviewed Overrides",
            "Audit Findings",
            "Failure Modes Observed",
            "Recommended Improvements",
        ]
        all_sections_present = True
        for section in required_sections:
            if section.lower() not in report_text.lower():
                all_sections_present = False
                print(f"    Missing section in final_report.md: '{section}'")

        self.check(all_sections_present, "final_report.md contains all 6 required sections", "final_report.md missing required sections")

        # Final Summary
        print("\n" + "=" * 70)
        if not self.failures:
            print(f"✅ ALL VALIDATION CHECKS PASSED ({len(self.passes)}/{len(self.passes)} assertions passed)")
            print("=" * 70)
            return True
        else:
            print(f"❌ VALIDATION FAILED: {len(self.failures)} check(s) failed out of {len(self.passes) + len(self.failures)}")
            for fail in self.failures:
                print(f"  - {fail}")
            print("=" * 70)
            return False


def main():
    validator = PipelineValidator()
    success = validator.validate_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
