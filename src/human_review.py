"""
Stage 4: Human Review Checkpoint for Retrieval Overrides.
Displays retrieval results and draft answers in terminal, prompts for human overrides,
and writes review_overrides.json.
"""
import json
import sys
from typing import Any, Dict, List, Optional, Tuple


def display_review_summary(
    queries_data: Dict[str, Any],
    retrieval_results: List[Dict[str, Any]],
    draft_answers: List[Dict[str, Any]],
) -> None:
    """Prints a clear review summary for each query to stdout."""
    retrieval_map = {r["query_id"]: r["retrieved_chunks"] for r in retrieval_results}
    draft_map = {d["query_id"]: d for d in draft_answers}

    print("\n" + "=" * 70)
    print("           STAGE 4: HUMAN REVIEW CHECKPOINT")
    print("=" * 70)

    for q_item in queries_data.get("queries", []):
        qid = q_item["query_id"]
        question = q_item["question"]
        retrieved = retrieval_map.get(qid, [])
        draft = draft_map.get(qid, {})

        print(f"\n[{qid}] Question: {question}")
        print(f"  Draft Label: {draft.get('label', 'N/A')}")
        print(f"  Draft Answer: {draft.get('answer', 'N/A')[:120]}...")
        print(f"  Retrieved Chunks ({len(retrieved)}):")
        for chunk in retrieved:
            print(f"    - Rank {chunk['rank']}: {chunk['chunk_id']} (Doc: {chunk['document_name']}, Score: {chunk['retrieval_score']})")

    print("\n" + "-" * 70)


def parse_override_input(user_input: str, valid_chunk_ids: set) -> Tuple[Optional[str], List[str], Optional[str]]:
    """
    Parses an override line such as:
    'Q1: product_overview_chunk_0, product_overview_chunk_1' or 'Q1 product_overview_chunk_0'
    Returns (query_id, list_of_valid_chunk_ids, error_message).
    """
    user_input = user_input.strip()
    if not user_input:
        return None, [], None

    # Support 'Q1: chunk1, chunk2' or 'Q1 chunk1, chunk2' or 'Q1=chunk1, chunk2'
    if ":" in user_input:
        parts = user_input.split(":", 1)
    elif "=" in user_input:
        parts = user_input.split("=", 1)
    else:
        parts = user_input.split(None, 1)

    if len(parts) != 2:
        return None, [], f"Invalid format. Expected 'query_id: chunk_id1, chunk_id2'. Got: '{user_input}'"

    qid = parts[0].strip()
    raw_chunks = [c.strip() for c in parts[1].replace(";", ",").split(",") if c.strip()]

    invalid_chunks = [c for c in raw_chunks if c not in valid_chunk_ids]
    if invalid_chunks:
        return qid, [], f"Unknown chunk ID(s): {', '.join(invalid_chunks)}. Must exist in chunks.json."

    if not raw_chunks:
        return qid, [], "No chunk IDs provided."

    return qid, raw_chunks, None


def conduct_human_review(
    queries_data: Dict[str, Any],
    retrieval_results: List[Dict[str, Any]],
    draft_answers: List[Dict[str, Any]],
    chunks_data: List[Dict[str, Any]],
    interactive: bool = True,
    override_args: Optional[List[str]] = None,
    output_filepath: str = "review_overrides.json",
) -> Dict[str, Any]:
    """
    Runs the human review checkpoint, allowing overrides for retrieved chunks.
    Saves results to review_overrides.json.
    """
    valid_chunk_ids = {c["chunk_id"] for c in chunks_data}
    retrieval_map = {r["query_id"]: [c["chunk_id"] for c in r["retrieved_chunks"]] for r in retrieval_results}
    overrides_applied: Dict[str, List[str]] = {}

    display_review_summary(queries_data, retrieval_results, draft_answers)

    # 1. Apply any CLI-specified overrides first
    if override_args:
        for arg in override_args:
            qid, chunk_ids, err = parse_override_input(arg, valid_chunk_ids)
            if err:
                print(f"[Warning] CLI Override error: {err}")
            elif qid:
                overrides_applied[qid] = chunk_ids
                print(f"[Override Applied via CLI] Query {qid} -> {chunk_ids}")

    # 2. Interactive prompt if in interactive mode
    prompt_text = (
        "\nDo you want to override retrieved chunks for any query before audit?\n"
        "Enter query_id and comma-separated chunk_ids to force as final context, or press Enter to continue: "
    )

    if interactive and sys.stdin.isatty():
        while True:
            try:
                user_input = input(prompt_text).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nContinuing without further overrides...")
                break

            if not user_input:
                break

            qid, chunk_ids, err = parse_override_input(user_input, valid_chunk_ids)
            if err:
                print(f"Error: {err}")
                continue

            if qid:
                overrides_applied[qid] = chunk_ids
                print(f"Successfully recorded override for {qid}: {chunk_ids}")

    # 3. Assemble review_overrides.json
    overrides_list: List[Dict[str, Any]] = []
    final_context_map: Dict[str, List[str]] = {}

    for q_item in queries_data.get("queries", []):
        qid = q_item["query_id"]
        original_cids = retrieval_map.get(qid, [])
        is_overridden = qid in overrides_applied
        forced_cids = overrides_applied.get(qid, [])
        final_cids = forced_cids if is_overridden else original_cids

        final_context_map[qid] = final_cids

        overrides_list.append({
            "query_id": qid,
            "original_retrieved_chunk_ids": original_cids,
            "is_overridden": is_overridden,
            "overridden_chunk_ids": forced_cids,
            "final_context_chunk_ids": final_cids,
        })

    result_payload = {
        "overrides_count": len(overrides_applied),
        "overrides": overrides_list,
    }

    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2, ensure_ascii=False)

    print(f"\n[Human Review Complete] Saved review state to {output_filepath}")
    return result_payload
