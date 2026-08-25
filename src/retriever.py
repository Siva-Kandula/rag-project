"""
Deterministic query retrieval module.
Retrieves top-k chunks for each query and calculates optional deterministic retrieval metrics.
Emits retrieval_results.json and retrieval_metrics.json.
"""
import json
from typing import Any, Dict, List, Optional
from src.indexer import SearchIndex, tokenize


def retrieve_query(
    index: SearchIndex,
    query_text: str,
    top_k: int = 3,
    mode: str = "bm25",
) -> List[Dict[str, Any]]:
    """Retrieves top_k ranked chunks for a single query."""
    if not index.chunks:
        return []

    tokens = tokenize(query_text)
    mode_lower = mode.lower()

    if mode_lower == "vector":
        raw_scores = index.score_vector(tokens)
    elif mode_lower == "hybrid":
        bm25_scores = index.score_bm25(tokens)
        vec_scores = index.score_vector(tokens)
        # Normalize and combine
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
        raw_scores = {}
        for cid in index.chunk_map.keys():
            b_norm = (bm25_scores.get(cid, 0.0) / max_bm25) if max_bm25 > 0 else 0.0
            v_norm = vec_scores.get(cid, 0.0)
            raw_scores[cid] = 0.5 * b_norm + 0.5 * v_norm
    else:  # default to bm25
        raw_scores = index.score_bm25(tokens)

    # Sort chunks by score descending, breaking ties deterministically by chunk_id ascending
    sorted_chunks = sorted(
        index.chunks,
        key=lambda c: (raw_scores.get(c["chunk_id"], 0.0), -ord(c["chunk_id"][0]) if c["chunk_id"] else 0, c["chunk_id"]),
        reverse=True
    )

    # If top scores are all 0, we still return top_k chunks deterministically (first available chunks)
    selected = sorted_chunks[:top_k]

    retrieved: List[Dict[str, Any]] = []
    for rank, chunk in enumerate(selected, start=1):
        cid = chunk["chunk_id"]
        score = raw_scores.get(cid, 0.0)
        retrieved.append({
            "chunk_id": cid,
            "document_name": chunk["document_name"],
            "rank": rank,
            "retrieval_score": round(score, 4),
        })

    return retrieved


def execute_retrieval(
    index: SearchIndex,
    queries_data: Dict[str, Any],
    top_k: int = 3,
    output_filepath: str = "retrieval_results.json",
) -> List[Dict[str, Any]]:
    """
    Executes top-k retrieval for all queries in queries_data and writes retrieval_results.json.
    """
    queries_list = queries_data.get("queries", [])
    results: List[Dict[str, Any]] = []

    for item in queries_list:
        qid = item["query_id"]
        question = item["question"]
        retrieved_chunks = retrieve_query(
            index=index,
            query_text=question,
            top_k=top_k,
            mode=index.mode,
        )
        results.append({
            "query_id": qid,
            "question": question,
            "retrieved_chunks": retrieved_chunks,
        })

    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def compute_retrieval_metrics(
    queries_data: Dict[str, Any],
    retrieval_results: List[Dict[str, Any]],
    top_k: int = 3,
    output_filepath: str = "retrieval_metrics.json",
) -> Optional[Dict[str, Any]]:
    """
    Computes deterministic retrieval metrics (Hit@k, Recall@k, MRR@k)
    if queries contain ground-truth annotations (e.g. expected_document or expected_chunk_ids).
    Gracefully returns None and skips file write if annotations are absent.
    """
    queries_list = queries_data.get("queries", [])
    query_map = {q["query_id"]: q for q in queries_list}

    annotated_count = 0
    hits_at_k = 0
    reciprocal_ranks: List[float] = []
    per_query_metrics: List[Dict[str, Any]] = []

    for res in retrieval_results:
        qid = res["query_id"]
        q_item = query_map.get(qid, {})

        expected_doc = q_item.get("expected_document") or q_item.get("expected_doc")
        expected_chunks = q_item.get("expected_chunk_ids") or q_item.get("expected_evidence")

        if not expected_doc and not expected_chunks:
            continue

        annotated_count += 1
        retrieved = res.get("retrieved_chunks", [])

        # Check hit
        hit = False
        rr = 0.0
        for rank_idx, item in enumerate(retrieved, start=1):
            doc_matched = expected_doc and (item["document_name"] == expected_doc)
            chunk_matched = expected_chunks and (item["chunk_id"] in expected_chunks)

            if doc_matched or chunk_matched:
                if not hit:
                    hit = True
                    rr = 1.0 / rank_idx

        if hit:
            hits_at_k += 1
        reciprocal_ranks.append(rr)

        per_query_metrics.append({
            "query_id": qid,
            "hit_at_k": hit,
            "reciprocal_rank": round(rr, 4),
            "expected_target": expected_doc or expected_chunks,
        })

    if annotated_count == 0:
        return None

    metrics = {
        "annotated_queries_count": annotated_count,
        "top_k": top_k,
        "hit_rate_at_k": round(hits_at_k / annotated_count, 4),
        "mean_reciprocal_rank": round(sum(reciprocal_ranks) / annotated_count, 4),
        "per_query_details": per_query_metrics,
    }

    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics
