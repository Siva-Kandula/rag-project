"""
Retrieval index builder supporting deterministic BM25, TF-IDF Vector, and Hybrid modes.
Emits index_metadata.json.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import re
from typing import Any, Dict, List, Set, Tuple


def tokenize(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric tokens."""
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


class SearchIndex:
    """In-memory index supporting BM25 and Vector (TF-IDF Cosine Similarity) retrieval."""

    def __init__(self, chunks: List[Dict[str, Any]], mode: str = "bm25"):
        self.chunks = chunks
        self.mode = mode.lower()
        self.chunk_map = {c["chunk_id"]: c for c in chunks}
        self.num_chunks = len(chunks)
        self.unique_documents: Set[str] = {c["document_name"] for c in chunks}

        # Tokenized representation
        self.chunk_tokens: Dict[str, List[str]] = {}
        self.chunk_tf: Dict[str, Counter] = {}
        self.chunk_lengths: Dict[str, int] = {}
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.inverted_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.vocabulary: Set[str] = set()

        total_length = 0
        for chunk in chunks:
            cid = chunk["chunk_id"]
            tokens = tokenize(chunk["text"])
            self.chunk_tokens[cid] = tokens
            self.chunk_lengths[cid] = len(tokens)
            total_length += len(tokens)

            tf = Counter(tokens)
            self.chunk_tf[cid] = tf

            for term, count in tf.items():
                self.doc_freq[term] += 1
                self.inverted_index[term].append((cid, count))
                self.vocabulary.add(term)

        self.avg_chunk_length = total_length / max(1, self.num_chunks)

        # Precompute IDF and TF-IDF vectors
        self.idf: Dict[str, float] = {}
        self.vector_idf: Dict[str, float] = {}
        self._compute_idfs()
        self._compute_vectors()

    def _compute_idfs(self) -> None:
        N = self.num_chunks
        for term, df in self.doc_freq.items():
            # Okapi BM25 IDF
            self.idf[term] = math.log(((N - df + 0.5) / (df + 0.5)) + 1.0)
            # Standard Vector TF-IDF IDF
            self.vector_idf[term] = math.log(1.0 + (N / (df + 0.5)))

    def _compute_vectors(self) -> None:
        """Precomputes TF-IDF vector representations and L2 norms."""
        self.chunk_vectors: Dict[str, Dict[str, float]] = {}
        self.chunk_norms: Dict[str, float] = {}

        for cid, tf in self.chunk_tf.items():
            vec: Dict[str, float] = {}
            norm_sq = 0.0
            for term, count in tf.items():
                tfidf_val = (1.0 + math.log(count)) * self.vector_idf.get(term, 1.0)
                vec[term] = tfidf_val
                norm_sq += tfidf_val * tfidf_val

            self.chunk_vectors[cid] = vec
            self.chunk_norms[cid] = math.sqrt(norm_sq) if norm_sq > 0 else 1.0

    def score_bm25(self, query_tokens: List[str], k1: float = 1.5, b: float = 0.75) -> Dict[str, float]:
        """Calculates Okapi BM25 scores for all chunks against query tokens."""
        scores: Dict[str, float] = defaultdict(float)
        query_tf = Counter(query_tokens)

        for term, q_count in query_tf.items():
            if term not in self.inverted_index:
                continue
            idf = self.idf.get(term, 0.0)
            for cid, count in self.inverted_index[term]:
                doc_len = self.chunk_lengths.get(cid, 0)
                len_norm = (1.0 - b) + b * (doc_len / max(1e-6, self.avg_chunk_length))
                term_score = idf * ((count * (k1 + 1.0)) / (count + k1 * len_norm))
                scores[cid] += term_score

        return scores

    def score_vector(self, query_tokens: List[str]) -> Dict[str, float]:
        """Calculates Cosine Similarity scores for all chunks against query TF-IDF vector."""
        scores: Dict[str, float] = defaultdict(float)
        q_tf = Counter(query_tokens)
        q_vec: Dict[str, float] = {}
        q_norm_sq = 0.0

        for term, count in q_tf.items():
            if term in self.vector_idf:
                val = (1.0 + math.log(count)) * self.vector_idf[term]
                q_vec[term] = val
                q_norm_sq += val * val

        q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 1.0

        for cid, doc_vec in self.chunk_vectors.items():
            dot_product = sum(weight * doc_vec.get(term, 0.0) for term, weight in q_vec.items())
            doc_norm = self.chunk_norms.get(cid, 1.0)
            scores[cid] = dot_product / (q_norm * doc_norm) if (q_norm * doc_norm) > 0 else 0.0

        return scores

    def get_metadata(self) -> Dict[str, Any]:
        """Returns structured index metadata."""
        return {
            "retrieval_mode": self.mode,
            "total_chunks": self.num_chunks,
            "total_documents": len(self.unique_documents),
            "document_names": sorted(list(self.unique_documents)),
            "vocabulary_size": len(self.vocabulary),
            "average_chunk_length_words": round(self.avg_chunk_length, 2),
            "supported_modes": ["bm25", "vector", "hybrid"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


def build_index_and_save_metadata(
    chunks: List[Dict[str, Any]],
    mode: str = "bm25",
    output_metadata_path: str = "index_metadata.json",
) -> Tuple[SearchIndex, Dict[str, Any]]:
    """Builds search index and writes index_metadata.json to disk."""
    index = SearchIndex(chunks=chunks, mode=mode)
    metadata = index.get_metadata()

    with open(output_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return index, metadata
