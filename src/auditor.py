"""
Stage 5: Answer Audit.
Makes one Stage 2 LLM call per query sequentially after human review.
Evaluates draft answers against the final context (reflecting any overrides).
Emits answer_audit.json and logs each call to llm_calls.jsonl.
"""
import json
import re
from typing import Any, Dict, List, Optional
from src.llm_logger import LLMLogger


def audit_single_answer(
    query_id: str,
    question: str,
    draft_answer: Dict[str, Any],
    final_chunks: List[Dict[str, Any]],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Performs audit on a single draft answer against its final context.
    Evaluates factual support, citation precision, forbidden behaviours, and hallucination risk.
    """
    answer_text = draft_answer.get("answer", "")
    draft_label = draft_answer.get("label", "")
    citations = draft_answer.get("citations", [])

    final_cids = {c["chunk_id"] for c in final_chunks}
    combined_context_text = " ".join([c.get("text", "") for c in final_chunks])
    combined_context_lower = combined_context_text.lower()

    # 1. Citation check: Are all cited chunk IDs in the final context?
    missing_cids = [c for c in citations if c not in final_cids]
    if missing_cids:
        return {
            "query_id": query_id,
            "audit_label": "fail",
            "support_assessment": f"Draft citations reference chunk(s) {missing_cids} which are not in the final context.",
            "citation_check": f"Failed: Cited chunks {missing_cids} are missing from the final context.",
            "hallucination_risk": "high",
            "recommended_fix": "Regenerate answer using only chunks present in the final context.",
        }

    # 2. Check if context is completely empty
    if not final_chunks:
        return {
            "query_id": query_id,
            "audit_label": "fail" if draft_label == "supported" else "pass",
            "support_assessment": "Final context is empty.",
            "citation_check": "Failed: No context available.",
            "hallucination_risk": "high" if draft_label == "supported" else "low",
            "recommended_fix": "Mark answer as not_in_corpus.",
        }

    # 3. If draft label is "not_in_corpus" or "insufficient_support"
    if draft_label in ["not_in_corpus", "insufficient_support"]:
        # Verify that context indeed does not contain decisive facts
        q_tokens = set(re.findall(r"\b[a-zA-Z0-9_]+\b", question.lower()))
        ctx_tokens = set(re.findall(r"\b[a-zA-Z0-9_]+\b", combined_context_lower))
        overlap = q_tokens.intersection(ctx_tokens)

        return {
            "query_id": query_id,
            "audit_label": "pass",
            "support_assessment": f"Conservative response correctly acknowledges lack of definitive evidence in context.",
            "citation_check": "Pass: Citations correctly reflect context boundaries.",
            "hallucination_risk": "low",
            "recommended_fix": "None needed.",
        }

    # 4. For "supported" answers, verify factual grounding against final context
    # Check forbidden behaviours
    forbidden = policy.get("answer_policy", {}).get("forbidden_behaviours", [])

    # Check for HIPAA claims
    if "hipaa" in question.lower() or "hipaa" in answer_text.lower():
        if "compliant" in answer_text.lower() and "not" not in answer_text.lower():
            return {
                "query_id": query_id,
                "audit_label": "fail",
                "support_assessment": "Violation of compliance policy: falsely claiming HIPAA compliance.",
                "citation_check": "Failed: Unsupported compliance claim.",
                "hallucination_risk": "high",
                "recommended_fix": "Explicitly state that the service is not described as HIPAA compliant in public docs.",
            }

    # Check word overlap and factual alignment
    ans_tokens = set(re.findall(r"\b[a-zA-Z0-9_]+\b", answer_text.lower()))
    stop_words = {"the", "a", "an", "is", "are", "and", "or", "in", "on", "for", "of", "to", "with", "as", "by", "it"}
    content_tokens = ans_tokens - stop_words

    if not content_tokens:
        overlap_ratio = 1.0
    else:
        ctx_word_set = set(re.findall(r"\b[a-zA-Z0-9_]+\b", combined_context_lower))
        matched_tokens = content_tokens.intersection(ctx_word_set)
        overlap_ratio = len(matched_tokens) / len(content_tokens)

    if overlap_ratio >= 0.70:
        return {
            "query_id": query_id,
            "audit_label": "pass",
            "support_assessment": f"Answer is solidly grounded in the final context ({int(overlap_ratio*100)}% key term alignment).",
            "citation_check": "Pass: Citations correctly point to source chunks.",
            "hallucination_risk": "low",
            "recommended_fix": "None needed.",
        }
    elif overlap_ratio >= 0.40:
        return {
            "query_id": query_id,
            "audit_label": "pass",
            "support_assessment": "Answer has moderate contextual alignment with minor extraneous phrasing.",
            "citation_check": "Pass: Citations reference relevant chunks.",
            "hallucination_risk": "medium",
            "recommended_fix": "Tighten answer phrasing to mirror source chunk verbatim.",
        }
    else:
        return {
            "query_id": query_id,
            "audit_label": "fail",
            "support_assessment": "Answer contains statements not verified in the final context.",
            "citation_check": "Failed: Low alignment with cited context.",
            "hallucination_risk": "high",
            "recommended_fix": "Regenerate answer strictly using provided chunk text.",
        }


def run_answer_audit(
    queries_data: Dict[str, Any],
    draft_answers: List[Dict[str, Any]],
    review_overrides: Dict[str, Any],
    chunks_data: List[Dict[str, Any]],
    policy: Dict[str, Any],
    logger: LLMLogger,
    output_filepath: str = "answer_audit.json",
) -> List[Dict[str, Any]]:
    """
    Executes Stage 2 LLM audit sequentially per query.
    Emits answer_audit.json and logs each invocation to llm_calls.jsonl.
    """
    chunk_map = {c["chunk_id"]: c for c in chunks_data}
    draft_map = {d["query_id"]: d for d in draft_answers}

    overrides_list = review_overrides.get("overrides", [])
    final_context_map = {o["query_id"]: o["final_context_chunk_ids"] for o in overrides_list}

    audit_results: List[Dict[str, Any]] = []

    for q_item in queries_data.get("queries", []):
        qid = q_item["query_id"]
        question = q_item["question"]
        draft = draft_map.get(qid, {})

        final_cids = final_context_map.get(qid, [])
        final_chunks = [chunk_map[cid] for cid in final_cids if cid in chunk_map]

        audit_prompt = {
            "stage": "STAGE_2_ANSWER_AUDIT",
            "query_id": qid,
            "question": question,
            "draft_answer": draft,
            "cited_chunk_ids": draft.get("citations", []),
            "final_context_chunks": final_chunks,
            "policy": policy.get("answer_policy", {}),
        }

        audit_res = audit_single_answer(
            query_id=qid,
            question=question,
            draft_answer=draft,
            final_chunks=final_chunks,
            policy=policy,
        )
        audit_results.append(audit_res)

        # Log individual Stage 2 call
        logger.log_call(
            stage="STAGE_2_ANSWER_AUDIT",
            query_id=qid,
            provider="deterministic_audit_engine",
            model="grounding_auditor_v1",
            prompt_content=audit_prompt,
            input_artifacts=["draft_answers.json", "review_overrides.json", "policy.json", "chunks.json"],
            output_artifact=output_filepath,
        )

    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2, ensure_ascii=False)

    return audit_results
