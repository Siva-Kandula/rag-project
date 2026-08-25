"""
Test with replaced custom fixtures (simulating evaluator replacement).
"""
import json
import os
import shutil
import tempfile
from pipeline import RAGPipeline
from validate import PipelineValidator


def test_custom_replaced_fixtures():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create custom documents
        docs_dir = os.path.join(tmpdir, "documents")
        os.makedirs(docs_dir, exist_ok=True)
        with open(os.path.join(docs_dir, "doc_alpha.txt"), "w") as f:
            f.write("AlphaCorp was founded in 2018. It operates across 14 global regions.")
        with open(os.path.join(docs_dir, "doc_beta.txt"), "w") as f:
            f.write("BetaService maintains 99.99% uptime SLA on high availability cluster setups.")

        # Create custom queries
        queries_file = os.path.join(tmpdir, "queries.json")
        with open(queries_file, "w") as f:
            json.dump({
                "queries": [
                    {"query_id": "QA1", "question": "When was AlphaCorp founded?"},
                    {"query_id": "QA2", "question": "What is the uptime SLA for BetaService?"},
                    {"query_id": "QA3", "question": "Does GammaApp support offline sync?"}
                ]
            }, f)

        # Create custom policy
        policy_file = os.path.join(tmpdir, "policy.json")
        with open(policy_file, "w") as f:
            json.dump({
                "retrieval": {"top_k": 2, "chunk_size_chars": 200, "chunk_overlap_chars": 20, "mode": "vector"},
                "answer_policy": {
                    "allowed_labels": ["supported", "insufficient_support", "not_in_corpus"],
                    "require_citations": True,
                    "max_citations_per_answer": 2,
                    "forbidden_behaviours": ["hallucination"]
                }
            }, f)

        cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            pipeline = RAGPipeline(
                docs_dir="documents",
                queries_file="queries.json",
                policy_file="policy.json",
                log_file="llm_calls.jsonl"
            )
            success = pipeline.run(interactive=False)
            assert success is True

            validator = PipelineValidator()
            val_success = validator.validate_all()
            assert val_success is True

            # Verify QA3 is recognized as not_in_corpus or insufficient
            with open("draft_answers.json") as f:
                drafts = json.load(f)
            qa3 = next(d for d in drafts if d["query_id"] == "QA3")
            assert qa3["label"] in ["not_in_corpus", "insufficient_support"]

        finally:
            os.chdir(cwd)
