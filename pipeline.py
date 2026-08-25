"""
Main RAG Pipeline Orchestrator.
Enforces the mandatory stage sequence:
INIT
 -> INPUTS_LOADED
 -> DOCUMENTS_CHUNKED
 -> INDEX_BUILT
 -> RETRIEVAL_COMPLETE
 -> DRAFT_ANSWERS_GENERATED
 -> HUMAN_REVIEW_COMPLETE
 -> ANSWERS_AUDITED
 -> FINAL_REPORT_GENERATED
 -> VALIDATION_COMPLETE
 -> RESULTS_FINALISED
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from src.state import PipelineStage, PipelineStateMachine
from src.chunker import chunk_documents_directory
from src.indexer import build_index_and_save_metadata
from src.retriever import execute_retrieval, compute_retrieval_metrics
from src.llm_logger import LLMLogger
from src.generator import generate_draft_answers
from src.human_review import conduct_human_review
from src.auditor import run_answer_audit
from src.reviser import generate_revised_answers
from src.error_analyzer import analyze_retrieval_errors
from src.report_generator import generate_final_evaluation_report


class RAGPipeline:
    def __init__(
        self,
        docs_dir: str = "documents",
        queries_file: str = "queries.json",
        policy_file: str = "policy.json",
        log_file: str = "llm_calls.jsonl",
    ):
        self.docs_dir = docs_dir
        self.queries_file = queries_file
        self.policy_file = policy_file
        self.log_file = log_file

        self.sm = PipelineStateMachine()
        self.logger = LLMLogger(log_filepath=log_file, clear_existing=True)

        self.policy_data: Dict[str, Any] = {}
        self.queries_data: Dict[str, Any] = {}
        self.chunks_data: List[Dict[str, Any]] = []
        self.index_metadata: Dict[str, Any] = {}
        self.retrieval_results: List[Dict[str, Any]] = []
        self.draft_answers: List[Dict[str, Any]] = []
        self.review_overrides: Dict[str, Any] = {}
        self.audit_results: List[Dict[str, Any]] = []
        self.revised_answers: List[Dict[str, Any]] = []
        self.error_analysis: List[Dict[str, Any]] = []
        self.retrieval_metrics: Optional[Dict[str, Any]] = None

    def run(
        self,
        interactive: bool = True,
        override_args: Optional[List[str]] = None,
        retrieval_mode_override: Optional[str] = None,
    ) -> bool:
        print(f"[Pipeline] Starting at stage {self.sm.current_stage.value}...")

        # Stage 1: INPUTS_LOADED
        if not os.path.exists(self.docs_dir):
            raise FileNotFoundError(f"Documents directory '{self.docs_dir}' does not exist.")
        if not os.path.exists(self.queries_file):
            raise FileNotFoundError(f"Queries file '{self.queries_file}' does not exist.")
        if not os.path.exists(self.policy_file):
            raise FileNotFoundError(f"Policy file '{self.policy_file}' does not exist.")

        with open(self.queries_file, "r", encoding="utf-8") as f:
            self.queries_data = json.load(f)
        with open(self.policy_file, "r", encoding="utf-8") as f:
            self.policy_data = json.load(f)

        self.sm.transition_to(PipelineStage.INPUTS_LOADED)
        print(f"[Pipeline -> {self.sm.current_stage.value}] Loaded documents dir, {len(self.queries_data.get('queries', []))} queries, and policy.")

        # Stage 2: DOCUMENTS_CHUNKED
        retrieval_cfg = self.policy_data.get("retrieval", {})
        chunk_size = retrieval_cfg.get("chunk_size_chars", 350)
        chunk_overlap = retrieval_cfg.get("chunk_overlap_chars", 50)
        mode = retrieval_mode_override or retrieval_cfg.get("mode", "bm25")

        self.chunks_data = chunk_documents_directory(
            documents_dir=self.docs_dir,
            chunk_size_chars=chunk_size,
            chunk_overlap_chars=chunk_overlap,
            output_filepath="chunks.json",
        )
        self.sm.transition_to(PipelineStage.DOCUMENTS_CHUNKED)
        print(f"[Pipeline -> {self.sm.current_stage.value}] Produced {len(self.chunks_data)} chunks in chunks.json.")

        # Stage 3: INDEX_BUILT
        search_index, self.index_metadata = build_index_and_save_metadata(
            chunks=self.chunks_data,
            mode=mode,
            output_metadata_path="index_metadata.json",
        )
        self.sm.transition_to(PipelineStage.INDEX_BUILT)
        print(f"[Pipeline -> {self.sm.current_stage.value}] Index built with mode '{mode}' -> index_metadata.json.")

        # Stage 4: RETRIEVAL_COMPLETE
        top_k = retrieval_cfg.get("top_k", 3)
        self.retrieval_results = execute_retrieval(
            index=search_index,
            queries_data=self.queries_data,
            top_k=top_k,
            output_filepath="retrieval_results.json",
        )
        self.retrieval_metrics = compute_retrieval_metrics(
            queries_data=self.queries_data,
            retrieval_results=self.retrieval_results,
            top_k=top_k,
            output_filepath="retrieval_metrics.json",
        )
        self.sm.transition_to(PipelineStage.RETRIEVAL_COMPLETE)
        print(f"[Pipeline -> {self.sm.current_stage.value}] Retrieved top-{top_k} chunks for {len(self.retrieval_results)} queries -> retrieval_results.json.")

        # Stage 5: DRAFT_ANSWERS_GENERATED
        self.draft_answers = generate_draft_answers(
            queries_data=self.queries_data,
            retrieval_results=self.retrieval_results,
            chunks_data=self.chunks_data,
            policy=self.policy_data,
            logger=self.logger,
            output_filepath="draft_answers.json",
        )
        self.sm.transition_to(PipelineStage.DRAFT_ANSWERS_GENERATED)
        print(f"[Pipeline -> {self.sm.current_stage.value}] Generated {len(self.draft_answers)} draft answers -> draft_answers.json.")

        # Stage 6: HUMAN_REVIEW_COMPLETE
        self.review_overrides = conduct_human_review(
            queries_data=self.queries_data,
            retrieval_results=self.retrieval_results,
            draft_answers=self.draft_answers,
            chunks_data=self.chunks_data,
            interactive=interactive,
            override_args=override_args,
            output_filepath="review_overrides.json",
        )
        self.sm.transition_to(PipelineStage.HUMAN_REVIEW_COMPLETE)
        print(f"[Pipeline -> {self.sm.current_stage.value}] Human review complete -> review_overrides.json.")

        # Stage 7: ANSWERS_AUDITED
        self.audit_results = run_answer_audit(
            queries_data=self.queries_data,
            draft_answers=self.draft_answers,
            review_overrides=self.review_overrides,
            chunks_data=self.chunks_data,
            policy=self.policy_data,
            logger=self.logger,
            output_filepath="answer_audit.json",
        )
        self.revised_answers = generate_revised_answers(
            queries_data=self.queries_data,
            audit_results=self.audit_results,
            draft_answers=self.draft_answers,
            review_overrides=self.review_overrides,
            chunks_data=self.chunks_data,
            policy=self.policy_data,
            logger=self.logger,
            output_filepath="revised_answers.json",
        )
        self.error_analysis = analyze_retrieval_errors(
            queries_data=self.queries_data,
            retrieval_results=self.retrieval_results,
            draft_answers=self.draft_answers,
            audit_results=self.audit_results,
            review_overrides=self.review_overrides,
            chunks_data=self.chunks_data,
            output_filepath="retrieval_error_analysis.json",
        )
        self.sm.transition_to(PipelineStage.ANSWERS_AUDITED)
        print(f"[Pipeline -> {self.sm.current_stage.value}] Audited {len(self.audit_results)} answers -> answer_audit.json.")

        # Stage 8: FINAL_REPORT_GENERATED
        generate_final_evaluation_report(
            queries_data=self.queries_data,
            chunks_data=self.chunks_data,
            index_metadata=self.index_metadata,
            retrieval_results=self.retrieval_results,
            draft_answers=self.draft_answers,
            review_overrides=self.review_overrides,
            audit_results=self.audit_results,
            revised_answers=self.revised_answers,
            error_analysis=self.error_analysis,
            metrics_data=self.retrieval_metrics,
            output_filepath="final_report.md",
        )
        self.sm.transition_to(PipelineStage.FINAL_REPORT_GENERATED)
        print(f"[Pipeline -> {self.sm.current_stage.value}] Evaluation report generated -> final_report.md.")

        # Stage 9: VALIDATION_COMPLETE
        # Internal consistency check
        assert os.path.exists("chunks.json")
        assert os.path.exists("index_metadata.json")
        assert os.path.exists("retrieval_results.json")
        assert os.path.exists("draft_answers.json")
        assert os.path.exists("review_overrides.json")
        assert os.path.exists("answer_audit.json")
        assert os.path.exists("final_report.md")
        assert os.path.exists("llm_calls.jsonl")

        self.sm.transition_to(PipelineStage.VALIDATION_COMPLETE)
        print(f"[Pipeline -> {self.sm.current_stage.value}] Internal sanity checks passed.")

        # Stage 10: RESULTS_FINALISED
        self.sm.transition_to(PipelineStage.RESULTS_FINALISED)
        print(f"\n✨ [Pipeline -> {self.sm.current_stage.value}] Successfully completed all stages!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Replayable Mini RAG Pipeline")
    parser.add_argument("--docs-dir", default="documents", help="Path to documents/ directory")
    parser.add_argument("--queries", default="queries.json", help="Path to queries.json")
    parser.add_argument("--policy", default="policy.json", help="Path to policy.json")
    parser.add_argument("--mode", default=None, choices=["bm25", "vector", "hybrid"], help="Retrieval mode override")
    parser.add_argument("--non-interactive", action="store_true", help="Run without pausing for interactive review")
    parser.add_argument("--override", action="append", help="Retrieval override format 'Q1: chunk_id1, chunk_id2'")

    args = parser.parse_args()

    pipeline = RAGPipeline(
        docs_dir=args.docs_dir,
        queries_file=args.queries,
        policy_file=args.policy,
    )
    pipeline.run(
        interactive=not args.non_interactive,
        override_args=args.override,
        retrieval_mode_override=args.mode,
    )


if __name__ == "__main__":
    main()
