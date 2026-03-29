from dataclasses import dataclass
from typing import Dict, List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from data_loader import FAQEntry


@dataclass(frozen=True)
class RetrievalHit:
    entry: FAQEntry
    semantic_score: float
    keyword_score: float
    final_score: float


class HybridRetriever:
    """Hybrid semantic + keyword retriever over fixed FAQ entries."""

    def __init__(
        self,
        entries: List[FAQEntry],
        model_name: str = "all-MiniLM-L6-v2",
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        local_files_only: bool = True,
    ) -> None:
        if not entries:
            raise ValueError("Cannot build retriever with empty entries.")

        self.entries = entries
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.exact_lookup = {entry.normalized_question: entry for entry in entries}

        self.model = SentenceTransformer(model_name, local_files_only=local_files_only)
        self.embeddings = self._build_embeddings()
        self.faiss_index = self._build_faiss_index(self.embeddings)

        corpus_tokens = [entry.question_tokens for entry in self.entries]
        self.bm25 = BM25Okapi(corpus_tokens)

    def _build_embeddings(self) -> np.ndarray:
        questions = [entry.normalized_question for entry in self.entries]
        vectors = self.model.encode(
            questions,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    @staticmethod
    def _build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index

    @staticmethod
    def _minmax(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        v_min = float(np.min(values))
        v_max = float(np.max(values))
        if v_max - v_min <= 1e-12:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - v_min) / (v_max - v_min)).astype(np.float32)

    def retrieve(self, sub_query: str, top_k: int = 5) -> List[RetrievalHit]:
        if sub_query in self.exact_lookup:
            return [
                RetrievalHit(
                    entry=self.exact_lookup[sub_query],
                    semantic_score=1.0,
                    keyword_score=1.0,
                    final_score=1.0,
                )
            ]

        query_tokens = sub_query.split()
        query_vec = self.model.encode(
            [sub_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        safe_top_k = max(1, min(top_k, len(self.entries)))

        # Semantic retrieval (cosine via normalized vectors + inner product).
        sem_scores, sem_indices = self.faiss_index.search(query_vec, safe_top_k)
        sem_scores = sem_scores[0]
        sem_indices = sem_indices[0]

        semantic_map: Dict[int, float] = {}
        for idx, raw_score in zip(sem_indices, sem_scores):
            # Convert cosine score range [-1, 1] to [0, 1].
            semantic_map[int(idx)] = float(np.clip((raw_score + 1.0) / 2.0, 0.0, 1.0))

        # Keyword retrieval over all docs, normalized per query.
        bm25_raw = np.array(self.bm25.get_scores(query_tokens), dtype=np.float32)
        bm25_norm = self._minmax(bm25_raw)

        bm25_top_indices = np.argsort(-bm25_raw)[:safe_top_k]
        keyword_map: Dict[int, float] = {
            int(idx): float(np.clip(bm25_norm[idx], 0.0, 1.0)) for idx in bm25_top_indices
        }

        candidate_indices = sorted(set(semantic_map.keys()) | set(keyword_map.keys()))

        hits: List[RetrievalHit] = []
        for idx in candidate_indices:
            semantic_score = semantic_map.get(idx, 0.0)
            keyword_score = keyword_map.get(idx, 0.0)
            final_score = (
                self.semantic_weight * semantic_score
                + self.keyword_weight * keyword_score
            )
            hits.append(
                RetrievalHit(
                    entry=self.entries[idx],
                    semantic_score=semantic_score,
                    keyword_score=keyword_score,
                    final_score=float(final_score),
                )
            )

        hits.sort(
            key=lambda item: (
                item.final_score,
                item.semantic_score,
                item.keyword_score,
                item.entry.normalized_question,
            ),
            reverse=True,
        )
        return hits[:safe_top_k]

