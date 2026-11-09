"""Hybrid retrieval (bonus 2): BM25 results reranked by meaning.

BM25 finds the candidates and the vector index only reorders them, so the
vectors can move a relevant chunk up but never bring in a chunk with none
of the query words.
"""
from .base_retriever import Retriever
from .bm25_index import BM25Index
from .exceptions import VectorIndexError
from .models import SearchHit
from .vector_index import VectorIndex


class HybridRetriever(Retriever):
    """Combine the BM25 and vector scores into a single ranking.

    Has the same ``search(query, k)`` as ``BM25Index``, so the commands
    use either one without knowing which.

    Attributes:
        CANDIDATES: How many BM25 results are reranked.
        ALPHA: Weight of the vector similarity, from 0 (BM25 order) to 1
            (similarity only). ``CANDIDATES`` and ``ALPHA`` were chosen
            with a sweep on the public datasets (README, *Semantic
            embeddings: tuning*): the centre of the stable zone.
    """

    CANDIDATES = 30
    ALPHA = 0.5

    def __init__(self, bm25: BM25Index, vector: VectorIndex) -> None:
        """Keep both indexes after checking they belong together.

        Args:
            bm25: The lexical index.
            vector: The vector index built from the same chunks, in the
                same order (chunk ids are shared).

        Raises:
            VectorIndexError: The indexes have a different number of
                chunks (one was rebuilt without the other).
        """
        if vector.chunk_count != bm25.chunk_count:
            raise VectorIndexError("the vector index does not match the BM25 "
                                   "index. Run 'index' again")
        self._bm25 = bm25
        self._vector = vector

    def search(self, query: str, k: int,
               sources: frozenset[str] | None = None) -> list[SearchHit]:
        """Return the ``k`` best chunks for the query.

        BM25 gives ``CANDIDATES`` hits (or ``k`` when it is larger, so a
        large ``k`` still gets ``k`` results), and each one gets
        ``(1 - ALPHA) * score / best score + ALPHA * similarity``: BM25 is
        divided by the best one so both parts go up to 1. Equal scores
        keep the BM25 order (the sort is stable).

        Args:
            query: Free text.
            k: Maximum number of results.
            sources: Names of the files to search, without folder or
                extension; ``None`` searches them all.

        Returns:
            Up to ``k`` hits, best first, with the combined score. Empty
            when BM25 finds nothing (the model is not used then).

        Raises:
            RagError: The embedding model cannot be loaded.
        """
        hits = self._bm25.search(query, max(self.CANDIDATES, k), sources)
        if not hits:
            return []
        best = self._vector.chunk_score(query)
        top = hits[0].score

        scored_hits: list[SearchHit] = []
        for hit in hits:
            bm25_norm = hit.score / top
            semantic = best[hit.chunk_id]
            combined = (1 - self.ALPHA) * bm25_norm + self.ALPHA * semantic
            scored_hits.append(
                hit.model_copy(update={"score": float(combined)})
            )
        scored_hits.sort(key=lambda h: h.score, reverse=True)
        return scored_hits[:k]
