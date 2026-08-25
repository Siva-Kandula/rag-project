"""
Unit and Integration Test Suite for Replayable Mini RAG Pipeline.
Run with: pytest tests/ -v
"""
import json
import os
import shutil
import tempfile
import pytest

from src.state import PipelineStage, PipelineStateMachine
from src.chunker import chunk_text_deterministic, chunk_documents_directory
from src.indexer import SearchIndex, build_index_and_save_metadata, tokenize
from src.retriever import retrieve_query, execute_retrieval, compute_retrieval_metrics
from src.llm_logger import LLMLogger
from src.generator import deterministic_grounded_answer, generate_draft_answers
from src.human_review import conduct_human_review, parse_override_input
from src.auditor import audit_single_answer, run_answer_audit
from src.reviser import generate_revised_answers
from src.error_analyzer import analyze_retrieval_errors
from src.report_generator import generate_final_evaluation_report
from pipeline import RAGPipeline
from validate import PipelineValidator


def test_state_machine():
    sm = PipelineStateMachine()
    assert sm.current_stage == PipelineStage.INIT

    sm.transition_to(PipelineStage.INPUTS_LOADED)
    assert sm.current_stage == PipelineStage.INPUTS_LOADED

    # Test invalid transition (skipping stage) raises RuntimeError
    with pytest.raises(RuntimeError):
        sm.transition_to(PipelineStage.RETRIEVAL_COMPLETE)


def test_chunker_deterministic():
    text = "Short test document for chunking validation."
    chunks = chunk_text_deterministic(text, doc_name="test.txt", chunk_size_chars=50, chunk_overlap_chars=10)
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "test_chunk_0"
    assert chunks[0]["start_char"] == 0
    assert chunks[0]["end_char"] == len(text)
    assert chunks[0]["text"] == text

    long_text = "A" * 120
    chunks_long = chunk_text_deterministic(long_text, doc_name="long.txt", chunk_size_chars=50, chunk_overlap_chars=10)
    assert len(chunks_long) > 1
    for c in chunks_long:
        assert len(c["text"]) <= 50
        assert long_text[c["start_char"]:c["end_char"]] == c["text"]


def test_bm25_and_vector_retrieval():
    chunks = [
        {"chunk_id": "c1", "document_name": "doc1.txt", "text": "InsightBoard platform analytics dashboards and reports."},
        {"chunk_id": "c2", "document_name": "doc2.txt", "text": "Security encryption TLS 1.2 AES-256 compliance."},
        {"chunk_id": "c3", "document_name": "doc3.txt", "text": "Billing monthly discount annual subscriptions and refunds."},
    ]
    index_bm25 = SearchIndex(chunks, mode="bm25")
    res_bm25 = retrieve_query(index_bm25, "analytics dashboards", top_k=2, mode="bm25")
    assert len(res_bm25) == 2
    assert res_bm25[0]["chunk_id"] == "c1"

    index_vec = SearchIndex(chunks, mode="vector")
    res_vec = retrieve_query(index_vec, "encryption security", top_k=2, mode="vector")
    assert len(res_vec) == 2
    assert res_vec[0]["chunk_id"] == "c2"


def test_grounded_answer_and_audit():
    policy = {
        "answer_policy": {
            "allowed_labels": ["supported", "insufficient_support", "not_in_corpus"],
            "max_citations_per_answer": 2,
            "forbidden_behaviours": ["claiming compliance not stated"],
        }
    }
    retrieved = [
        {"chunk_id": "c1", "text": "The platform retains event data for 13 months on the standard plan."},
    ]

    # Grounded query
    ans = deterministic_grounded_answer(
        question="How long is event data retained on the standard plan?",
        retrieved_chunks=retrieved,
        policy=policy,
    )
    assert ans["label"] == "supported"
    assert "c1" in ans["citations"]

    # Audit check on supported answer
    audit = audit_single_answer(
        query_id="Q1",
        question="How long is event data retained on the standard plan?",
        draft_answer=ans,
        final_chunks=retrieved,
        policy=policy,
    )
    assert audit["audit_label"] == "pass"
    assert audit["hallucination_risk"] == "low"

    # Audit check on unsupported answer / missing citations
    audit_fail = audit_single_answer(
        query_id="Q1",
        question="How long is event data retained on the standard plan?",
        draft_answer=ans,
        final_chunks=[],  # context missing!
        policy=policy,
    )
    assert audit_fail["audit_label"] == "fail"
    assert audit_fail["hallucination_risk"] == "high"


def test_parse_override_input():
    valid = {"chunk_0", "chunk_1", "chunk_2"}
    qid, cids, err = parse_override_input("Q1: chunk_0, chunk_1", valid)
    assert qid == "Q1"
    assert cids == ["chunk_0", "chunk_1"]
    assert err is None

    qid_bad, _, err_bad = parse_override_input("Q1: chunk_999", valid)
    assert err_bad is not None


def test_end_to_end_pipeline_execution(tmp_path):
    # Run full pipeline non-interactively
    pipeline = RAGPipeline(
        docs_dir="documents",
        queries_file="queries.json",
        policy_file="policy.json",
        log_file="llm_calls.jsonl",
    )
    success = pipeline.run(interactive=False)
    assert success is True

    # Validate output artifacts
    validator = PipelineValidator()
    val_success = validator.validate_all()
    assert val_success is True
