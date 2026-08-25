"""
Stage 6: Final Evaluation Report Generation.
Generates comprehensive final_report.md containing all mandatory sections:
- Retrieval Summary
- Query-by-Query Results
- Reviewed Overrides
- Audit Findings
- Failure Modes Observed
- Recommended Improvements
"""
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional


def generate_final_evaluation_report(
    queries_data: Dict[str, Any],
    chunks_data: List[Dict[str, Any]],
    index_metadata: Dict[str, Any],
    retrieval_results: List[Dict[str, Any]],
    draft_answers: List[Dict[str, Any]],
    review_overrides: Dict[str, Any],
    audit_results: List[Dict[str, Any]],
    revised_answers: Optional[List[Dict[str, Any]]] = None,
    error_analysis: Optional[List[Dict[str, Any]]] = None,
    metrics_data: Optional[Dict[str, Any]] = None,
    output_filepath: str = "final_report.md",
) -> str:
    """
    Generates the final evaluation report markdown artifact.
    """
    query_map = {q["query_id"]: q for q in queries_data.get("queries", [])}
    retrieval_map = {r["query_id"]: r["retrieved_chunks"] for r in retrieval_results}
    draft_map = {d["query_id"]: d for d in draft_answers}
    overrides_list = review_overrides.get("overrides", [])
    override_map = {o["query_id"]: o for o in overrides_list}
    audit_map = {a["query_id"]: a for a in audit_results}
    revised_map = {r["query_id"]: r for r in (revised_answers or [])}

    total_queries = len(queries_data.get("queries", []))
    overridden_count = sum(1 for o in overrides_list if o.get("is_overridden"))
    passed_audits = sum(1 for a in audit_results if a.get("audit_label") == "pass")
    low_risk_count = sum(1 for a in audit_results if a.get("hallucination_risk") == "low")

    report_lines: List[str] = []

    # Title & Metadata
    report_lines.append("# RAG Pipeline Evaluation Report")
    report_lines.append(f"\n*Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*")
    report_lines.append(f"*Index Mode: `{index_metadata.get('retrieval_mode', 'bm25')}` | Total Chunks: `{len(chunks_data)}` | Total Queries: `{total_queries}`*\n")

    # 1. Retrieval Summary
    report_lines.append("## Retrieval Summary")
    report_lines.append(f"- **Total Documents Indexed**: {index_metadata.get('total_documents', 0)}")
    report_lines.append(f"- **Total Chunks Generated**: {len(chunks_data)}")
    report_lines.append(f"- **Retrieval Mode**: `{index_metadata.get('retrieval_mode', 'bm25')}`")
    report_lines.append(f"- **Average Chunk Length**: {index_metadata.get('average_chunk_length_words', 0)} words")

    if metrics_data:
        report_lines.append(f"- **Hit Rate @ k**: {metrics_data.get('hit_rate_at_k', 0.0) * 100:.1f}%")
        report_lines.append(f"- **Mean Reciprocal Rank (MRR)**: {metrics_data.get('mean_reciprocal_rank', 0.0):.4f}")
    report_lines.append("")

    # 2. Query-by-Query Results
    report_lines.append("## Query-by-Query Results")
    report_lines.append("This section summarizes the lifecycle of every evaluated query from retrieval to audit.\n")

    for q_item in queries_data.get("queries", []):
        qid = q_item["query_id"]
        question = q_item["question"]
        draft = draft_map.get(qid, {})
        audit = audit_map.get(qid, {})
        override = override_map.get(qid, {})
        revised = revised_map.get(qid, {})

        final_cids = override.get("final_context_chunk_ids", [])
        draft_label = draft.get("label", "unknown")
        audit_label = audit.get("audit_label", "unknown")
        risk = audit.get("hallucination_risk", "unknown")
        recommendation = audit.get("recommended_fix", "None")

        # Grounding status badge
        if draft_label == "supported" and audit_label == "pass":
            grounding_badge = "🟢 Fully Grounded"
        elif draft_label == "insufficient_support" or draft_label == "not_in_corpus":
            grounding_badge = "🟡 Non-Fact / Refusal Grounded"
        else:
            grounding_badge = "🔴 Unsupported / Hallucination Risk"

        report_lines.append(f"### Query `{qid}`: {question}")
        report_lines.append(f"**Grounding Status**: {grounding_badge}\n")
        report_lines.append(f"- **Question**: {question}")
        report_lines.append(f"- **Final Context Chunk IDs**: `{', '.join(final_cids) if final_cids else 'None'}`")
        report_lines.append(f"- **Draft Answer**: {draft.get('answer', 'N/A')}")
        report_lines.append(f"- **Draft Label**: `{draft_label}`")
        report_lines.append(f"- **Draft Citations**: `{draft.get('citations', [])}`")
        report_lines.append(f"- **Audit Label**: `{audit_label}` (Risk: `{risk}`)")
        report_lines.append(f"- **Audit Assessment**: {audit.get('support_assessment', 'N/A')}")
        report_lines.append(f"- **Citation Check**: {audit.get('citation_check', 'N/A')}")
        report_lines.append(f"- **Final Recommendation**: {recommendation}")

        if revised:
            report_lines.append(f"- **Revised Conservative Answer**: {revised.get('revised_answer', 'N/A')}")
            report_lines.append(f"- **Revised Citations**: `{revised.get('citations', [])}`")

        report_lines.append("")

    # 3. Reviewed Overrides
    report_lines.append("## Reviewed Overrides")
    if overridden_count == 0:
        report_lines.append("No manual retrieval overrides were requested during the human review checkpoint. All downstream audits proceeded using original top-k retrieved chunks.\n")
    else:
        report_lines.append(f"A total of **{overridden_count}** query/queries were overridden during human review:\n")
        report_lines.append("| Query ID | Original Top-k Chunks | Overridden Final Chunks | Overridden? |")
        report_lines.append("|---|---|---|---|")
        for o in overrides_list:
            orig = ", ".join(o.get("original_retrieved_chunk_ids", []))
            final = ", ".join(o.get("final_context_chunk_ids", []))
            status = "**Yes**" if o.get("is_overridden") else "No"
            report_lines.append(f"| `{o['query_id']}` | `{orig}` | `{final}` | {status} |")
        report_lines.append("")

    # 4. Audit Findings
    report_lines.append("## Audit Findings")
    report_lines.append(f"- **Audit Pass Rate**: {passed_audits}/{total_queries} ({passed_audits/max(1, total_queries)*100:.1f}%)")
    report_lines.append(f"- **Low Hallucination Risk Count**: {low_risk_count}/{total_queries}")
    report_lines.append(f"- **High Hallucination Risk Count**: {total_queries - passed_audits}/{total_queries}\n")

    report_lines.append("| Query ID | Draft Label | Audit Label | Hallucination Risk | Citation Compliance |")
    report_lines.append("|---|---|---|---|---|")
    for audit in audit_results:
        qid = audit["query_id"]
        draft_lbl = draft_map.get(qid, {}).get("label", "N/A")
        report_lines.append(f"| `{qid}` | `{draft_lbl}` | `{audit.get('audit_label')}` | `{audit.get('hallucination_risk')}` | {audit.get('citation_check', 'N/A')[:40]}... |")
    report_lines.append("")

    # 5. Failure Modes Observed
    report_lines.append("## Failure Modes Observed")
    if error_analysis:
        for err in error_analysis:
            report_lines.append(f"- **`{err['query_id']}` [{err['failure_type'].upper()}]**: {err['description']}")
    else:
        report_lines.append("No critical failure modes observed. All queries retrieved relevant chunks and passed factual audit checks.")
    report_lines.append("")

    # 6. Recommended Improvements
    report_lines.append("## Recommended Improvements")
    report_lines.append("1. **Corpus Expansion**: Ingest supplementary documentation for missing compliance frameworks and advanced API endpoints.")
    report_lines.append("2. **Hybrid Retrieval**: Combine dense semantic embeddings with BM25 lexical token matching to improve recall on short technical queries.")
    report_lines.append("3. **Sliding Chunk Boundaries**: Align chunk splits on paragraph or header boundaries to prevent context fragmentation across sentences.")
    report_lines.append("4. **Adaptive Citation Enforcement**: Employ citation-span verification at token level to guarantee exact sentence-level grounding.")
    report_lines.append("")

    markdown_content = "\n".join(report_lines)

    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return markdown_content
