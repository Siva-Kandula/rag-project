"""
Stretch Goal 9: Retrieval Error Analysis.
Classifies retrieval and grounding failure causes (ranking | chunking | ambiguity | corpus_gap).
Emits retrieval_error_analysis.json based on observable evidence.
"""
import json
import re
from typing import Any, Dict, List


def analyze_retrieval_errors(
    queries_data: Dict[str, Any],
    retrieval_results: List[Dict[str, Any]],
    draft_answers: List[Dict[str, Any]],
    audit_results: List[Dict[str, Any]],
    review_overrides: Dict[str, Any],
    chunks_data: List[Dict[str, Any]],
    output_filepath: str = "retrieval_error_analysis.json",
) -> List[Dict[str, Any]]:
    """
    Analyzes retrieval and generation failure modes systematically.
    """
    retrieval_map = {r["query_id"]: r["retrieved_chunks"] for r in retrieval_results}
    draft_map = {d["query_id"]: d for d in draft_answers}
    audit_map = {a["query_id"]: a for a in audit_results}
    overrides_list = review_overrides.get("overrides", [])
    override_map = {o["query_id"]: o for o in overrides_list}

    all_chunk_text = " ".join([c.get("text", "").lower() for c in chunks_data])
    analysis_entries: List[Dict[str, Any]] = []

    for q_item in queries_data.get("queries", []):
        qid = q_item["query_id"]
        question = q_item["question"]
        retrieved = retrieval_map.get(qid, [])
        draft = draft_map.get(qid, {})
        audit = audit_map.get(qid, {})
        override = override_map.get(qid, {})

        q_tokens = [t for t in re.findall(r"\b[a-zA-Z0-9_]+\b", question.lower()) if len(t) > 3]

        # Check if human override occurred
        if override.get("is_overridden"):
            analysis_entries.append({
                "query_id": qid,
                "failure_type": "ranking",
                "description": f"Human review overrode retrieval for {qid}. The initial top-k ranking failed to place optimal context in top positions.",
            })
            continue

        # Check if audit failed or flagged high risk
        if audit.get("audit_label") == "fail" or audit.get("hallucination_risk") == "high":
            analysis_entries.append({
                "query_id": qid,
                "failure_type": "ranking",
                "description": f"Audit identified grounding discrepancy for {qid}. Retrieved context was insufficient to support the claim.",
            })
            continue

        # Check corpus gap (no key terms in entire corpus)
        matched_in_corpus = [t for t in q_tokens if t in all_chunk_text]
        if not matched_in_corpus and q_tokens:
            analysis_entries.append({
                "query_id": qid,
                "failure_type": "corpus_gap",
                "description": f"Core query entities ({', '.join(q_tokens)}) are absent from the entire document corpus.",
            })
            continue

        # Check if label is not_in_corpus / insufficient_support
        if draft.get("label") == "not_in_corpus":
            analysis_entries.append({
                "query_id": qid,
                "failure_type": "corpus_gap",
                "description": "Requested information is not present in the indexed documents.",
            })
        elif draft.get("label") == "insufficient_support":
            # Check if partial tokens match
            analysis_entries.append({
                "query_id": qid,
                "failure_type": "ambiguity",
                "description": "Retrieved context partially overlaps with query terminology but lacks definitive facts to form a complete answer.",
            })

    # Save to disk
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(analysis_entries, f, indent=2, ensure_ascii=False)

    return analysis_entries
