"""
Deterministic document chunking module.
Chunks all .txt files from a documents/ directory according to policy.json parameters.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def chunk_text_deterministic(
    text: str,
    doc_name: str,
    chunk_size_chars: int = 350,
    chunk_overlap_chars: int = 50,
) -> List[Dict[str, Any]]:
    """
    Chunks a single document text deterministically into fixed-size overlapping slices.
    """
    chunks: List[Dict[str, Any]] = []
    text_len = len(text)

    if text_len == 0:
        return chunks

    if text_len <= chunk_size_chars:
        # Document fits within a single chunk
        doc_stem = Path(doc_name).stem
        chunk_id = f"{doc_stem}_chunk_0"
        chunks.append({
            "chunk_id": chunk_id,
            "document_name": doc_name,
            "start_char": 0,
            "end_char": text_len,
            "text": text,
        })
        return chunks

    step = max(1, chunk_size_chars - chunk_overlap_chars)
    start_char = 0
    chunk_idx = 0
    doc_stem = Path(doc_name).stem

    while start_char < text_len:
        end_char = min(text_len, start_char + chunk_size_chars)
        chunk_text = text[start_char:end_char]
        chunk_id = f"{doc_stem}_chunk_{chunk_idx}"

        chunks.append({
            "chunk_id": chunk_id,
            "document_name": doc_name,
            "start_char": start_char,
            "end_char": end_char,
            "text": chunk_text,
        })

        if end_char >= text_len:
            break

        start_char += step
        chunk_idx += 1

    return chunks


def chunk_documents_directory(
    documents_dir: str,
    chunk_size_chars: int = 350,
    chunk_overlap_chars: int = 50,
    output_filepath: str = "chunks.json",
) -> List[Dict[str, Any]]:
    """
    Reads all .txt files in documents_dir, produces deterministic chunks,
    and writes the result to output_filepath (chunks.json).
    """
    docs_path = Path(documents_dir)
    if not docs_path.exists() or not docs_path.is_dir():
        raise FileNotFoundError(f"Documents directory '{documents_dir}' not found or is not a directory.")

    all_chunks: List[Dict[str, Any]] = []
    txt_files = sorted([f for f in docs_path.iterdir() if f.is_file() and f.name.endswith(".txt")])

    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        doc_chunks = chunk_text_deterministic(
            text=content,
            doc_name=file_path.name,
            chunk_size_chars=chunk_size_chars,
            chunk_overlap_chars=chunk_overlap_chars,
        )
        all_chunks.extend(doc_chunks)

    # Save to disk
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    return all_chunks
