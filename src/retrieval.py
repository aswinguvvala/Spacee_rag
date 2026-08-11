"""Dense + BM25 hybrid retrieval over the persisted FAISS index.

Fusion strategy
---------------
Dense (cosine similarity via FAISS) and sparse (BM25) retrieval are combined
with Reciprocal Rank Fusion (RRF) rather than a weighted sum of raw scores.
RRF combines two rankings using only each document's *rank* in each list
(``1 / (k_rrf + rank)``), so it needs no score normalization or tuned mixing
weight -- cosine similarities and BM25 scores live on different, incomparable
scales, and RRF sidesteps that problem entirely. This matters here because
dense retrieval alone (see the Phase 1 smoke test) can miss an exact keyword
match that BM25 finds trivially, e.g. a question that quotes a name or number
verbatim.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.ingestion import EMBEDDING_MODEL_NAME, Chunk, load_index
from src.utils import INDEX_DIR, get_logger

logger = get_logger(__name__)

RRF_K = 60
DEFAULT_TOP_K = 5
CANDIDATE_POOL_MULTIPLIER = 8


class RetrievalError(Exception):
    """Raised when the retriever can't be initialized or a search fails."""


@dataclass(frozen=True)
class RetrievalResult:
    """A single retrieved chunk with its fused rank and score.

    Attributes:
        chunk: The retrieved :class:`~src.ingestion.Chunk`.
        score: Fusion score (RRF) if hybrid, else raw cosine similarity.
        rank: 0-indexed rank in the final result list.
    """

    chunk: Chunk
    score: float
    rank: int


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenizer used for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever:
    """Loads a persisted index once and serves dense/hybrid search over it."""

    def __init__(
        self,
        index_dir=INDEX_DIR,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        use_hybrid: bool = True,
    ) -> None:
        """Load the FAISS index, chunk metadata, embedding model, and BM25 index.

        Args:
            index_dir: Directory containing the persisted FAISS index and chunk metadata.
            embedding_model_name: Must match the model used at ingestion time.
            use_hybrid: If True, fuse dense results with BM25 via RRF.

        Raises:
            RetrievalError: If the index/model can't be loaded.
        """
        try:
            self.index, self.chunks = load_index(index_dir)
            self.model = SentenceTransformer(embedding_model_name)
        except Exception as exc:  # ingestion.IngestionError or model load errors
            raise RetrievalError(f"Failed to initialize retriever: {exc}") from exc

        self.use_hybrid = use_hybrid
        self.bm25: BM25Okapi | None = None
        if use_hybrid:
            tokenized_corpus = [_tokenize(c.text) for c in self.chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)

        self._chunk_id_to_pos = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        logger.info(
            "Retriever ready: %d chunks, hybrid=%s", len(self.chunks), use_hybrid
        )

    def _dense_ranked_positions(self, query: str, pool_size: int) -> list[int]:
        query_vec = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        _, ids = self.index.search(query_vec, min(pool_size, len(self.chunks)))
        return [int(i) for i in ids[0] if i != -1]

    def _bm25_ranked_positions(self, query: str, pool_size: int) -> list[int]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:pool_size]

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> list[RetrievalResult]:
        """Retrieve the top-``k`` chunks for ``query``.

        Args:
            query: Natural-language query.
            k: Number of results to return.

        Returns:
            List of :class:`RetrievalResult`, ranked best-first.

        Raises:
            RetrievalError: If ``query`` is empty or ``k`` is non-positive.
        """
        if not query or not query.strip():
            raise RetrievalError("Query must be a non-empty string")
        if k <= 0:
            raise RetrievalError(f"k must be positive, got {k}")

        pool_size = min(len(self.chunks), max(k * CANDIDATE_POOL_MULTIPLIER, 50))

        if not self.use_hybrid:
            dense_positions = self._dense_ranked_positions(query, k)
            return [
                RetrievalResult(chunk=self.chunks[pos], score=1.0 / (rank + 1), rank=rank)
                for rank, pos in enumerate(dense_positions[:k])
            ]

        dense_positions = self._dense_ranked_positions(query, pool_size)
        bm25_positions = self._bm25_ranked_positions(query, pool_size)

        rrf_scores: dict[int, float] = {}
        for rank, pos in enumerate(dense_positions):
            rrf_scores[pos] = rrf_scores.get(pos, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, pos in enumerate(bm25_positions):
            rrf_scores[pos] = rrf_scores.get(pos, 0.0) + 1.0 / (RRF_K + rank + 1)

        fused_ranking = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            RetrievalResult(chunk=self.chunks[pos], score=score, rank=rank)
            for rank, (pos, score) in enumerate(fused_ranking)
        ]

    def get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        """Look up a chunk by its id, or ``None`` if it doesn't exist in this index.

        Used by the verification layer to detect hallucinated citation ids: a
        citation that fails this lookup did not come from the index at all.
        """
        pos = self._chunk_id_to_pos.get(chunk_id)
        return self.chunks[pos] if pos is not None else None


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "Who commanded Apollo 13?"
    retriever = Retriever()
    for result in retriever.retrieve(query, k=5):
        print(f"[{result.rank}] {result.chunk.chunk_id}  score={result.score:.4f}")
        print(" ", result.chunk.text[:150].replace("\n", " "))
