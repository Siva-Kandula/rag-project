"""
Stage 3 (Optional / Should Attempt): Regenerated Answer After Audit Failure.
For any query with hallucination_risk = high or audit_label = fail,
generates a conservative revised answer using audited final context.
Emits revised_answers.json and logs to llm_calls.jsonl.
"""
import json
from typing import Any, Dict, List
from src.generator import deterministic_grounded_answer
from src.llm_logger import LLMLogger


def generate_revised_answers(
    queries_data: Dict[str, Any],
    audit_results: List[Dict[str, Any]],
    draft_answers: List[Dict[str, Any]],
    review_overrides: Dict[str, Any],
    chunks_data: List[Dict[str, Any]],
    policy: Dict[str, Any],
    logger: LLMLogger,
    output_filepath: str = "revised_answers.json",
) -> List[Dict[str, Any]]:
    """
    Generates revised answers for queries that failed audit or were flagged with high hallucination risk.
    """
    chunk_map = {c["chunk_id"]: c for c in chunks_data}
    query_map = {q["query_id"]: q for q in queries_data.get("queries", [])}
    draft_map = {d["query_id"]: d for d in draft_answers}
    overrides_list = review_overrides.get("overrides", [])
    final_context_map = {o["query_id"]: o["final_context_chunk_ids"] for o in overrides_list}

    revised_answers: List[Dict[str, Any]] = []

    for audit in audit_results:
        qid = audit["query_id"]
        is_fail = audit.get("audit_label") == "fail"
        is_high_risk = audit.get("hallucination_risk") == "high"

        if is_fail or is_high_risk:
            q_item = query_map.get(qid, {"question": ""})
            draft = draft_map.get(qid, {})
            final_cids = final_context_map.get(qid, [])
            final_chunks = [chunk_map[cid] for cid in final_cids if cid in chunk_map]

            prompt_payload = {
                "stage": "STAGE_3_REVISED_ANSWER_GENERATION",
                "query_id": qid,
                "question": q_item["question"],
                "failed_draft": draft,
                "audit_feedback": audit,
                "final_context_chunks": final_chunks,
                "policy": policy.get("answer_policy", {}),
            }

            # Generate conservative answer strictly from final context
            revised_raw = deterministic_grounded_answer(
                question=q_item["question"],
                retrieved_chunks=final_chunks,
                policy=policy,
            )

            record = {
                "query_id": qid,
                "original_draft_answer": draft.get("answer", ""),
                "revised_answer": revised_raw.get("answer", ""),
                "label": revised_raw.get("label", "insufficient_support"),
                "citations": revised_raw.get("citations", []),
                "revision_reasoning": f"Revised following audit feedback: {audit.get('recommended_fix', 'Re-grounded in final context.')}",
            }
            revised_answers.append(record)

            # Log call
            logger.log_call(
                stage="STAGE_3_REVISED_ANSWER_GENERATION",
                query_id=qid,
                provider="deterministic_reviser_engine",
                model="conservative_reviser_v1",
                prompt_content=prompt_payload,
                input_artifacts=["answer_audit.json", "review_overrides.json", "chunks.json"],
                output_artifact=output_filepath,
            )

    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(revised_answers, f, indent=2, ensure_ascii=False)

    return revised_answers
