"""Tests for ``src.hybrid_retriever.HybridRetriever``.

Two kinds of index: real ``BM25Index`` and ``VectorIndex`` built on small
chunks (with ``FakeEncoder``, see ``conftest.py``), and stand-ins with
fixed scores to check the combination formula exactly.
"""
import numpy as np
from numpy.typing import NDArray
import pytest

from src.bm25_index import BM25Index
from src.exceptions import RagError, VectorIndexError
from src.hybrid_retriever import HybridRetriever
from src.models import Chunk, SearchHit
from src.vector_index import VectorIndex

TEXTS = ["sampling temperature top_p", "install pip cuda",
         "engine step scheduler", "temperature of the gpu",
         "sampling params class", "pip install from source",
         "scheduler policy fcfs", "top_k sampling", "cuda graphs",
         "engine args"]
QUERIES = ["sampling temperature", "pip install cuda",
           "engine scheduler step", "cuda"]


def make_chunks(*texts: str) -> list[Chunk]:
    """One chunk per text, each in its own fake file."""
    return [Chunk(file_path=f"data/raw/{i}.md", first_character_index=0,
                  last_character_index=len(text), text=text)
            for i, text in enumerate(texts)]


class FixedBM25:
    """BM25 stand-in that always returns the same hits."""

    def __init__(self, hits: list[SearchHit], chunk_count: int) -> None:
        """Keep the hits and the reported number of chunks."""
        self.hits = hits
        self.chunk_count = chunk_count
        self.asked: list[int] = []

    def search(self, query: str, k: int) -> list[SearchHit]:
        """The first ``k`` hits; records ``k``."""
        self.asked.append(k)
        return self.hits[:k]


class FixedVectors:
    """VectorIndex stand-in with fixed chunk scores."""

    def __init__(self, scores: list[float]) -> None:
        """Keep the scores; one per chunk."""
        self.scores = np.array(scores, dtype=np.float64)
        self.chunk_count = len(scores)
        self.queries = 0

    def chunk_score(self, query: str) -> NDArray[np.float64]:
        """The fixed scores; counts the calls."""
        self.queries += 1
        return self.scores


def plan_example() -> tuple[FixedBM25, FixedVectors]:
    """The plan example: BM25 ranks 7, 2, 9."""
    chunks = make_chunks(*["x"] * 10)
    hits = [SearchHit(chunk=chunks[7], score=12.0, chunk_id=7),
            SearchHit(chunk=chunks[2], score=10.8, chunk_id=2),
            SearchHit(chunk=chunks[9], score=6.0, chunk_id=9)]
    scores = [0.0] * 10
    scores[7], scores[2], scores[9] = 0.30, 0.62, 0.70
    return FixedBM25(hits, 10), FixedVectors(scores)


@pytest.fixture
def indexes() -> tuple[BM25Index, VectorIndex]:
    """Real indexes built from the same chunks, in the same order."""
    chunks = make_chunks(*TEXTS)
    return BM25Index.build(chunks), VectorIndex.build(chunks)


class TestCombination:
    """The combined score ``(1 - α) BM25 / best BM25 + α similarity``."""

    def test_plan_example(self) -> None:
        """7, 2, 9 becomes 2, 7, 9 with 0.760, 0.650, 0.600."""
        bm25, vectors = plan_example()
        hits = HybridRetriever(bm25, vectors).search("q", 10)  # type: ignore
        assert [h.chunk_id for h in hits] == [2, 7, 9]
        assert [h.score for h in hits] == pytest.approx([0.76, 0.65, 0.60])

    def test_k_cuts_after_reranking(self) -> None:
        """``k`` keeps the best combined hits, not the best BM25 ones."""
        bm25, vectors = plan_example()
        hits = HybridRetriever(bm25, vectors).search("q", 2)  # type: ignore
        assert [h.chunk_id for h in hits] == [2, 7]

    def test_asks_bm25_for_candidates(self) -> None:
        """BM25 is asked for ``CANDIDATES`` hits when ``k`` is smaller."""
        bm25, vectors = plan_example()
        HybridRetriever(bm25, vectors).search("q", 3)  # type: ignore
        assert bm25.asked == [HybridRetriever.CANDIDATES]

    def test_large_k_gets_k_results(self,
                                    indexes: tuple[BM25Index, VectorIndex],
                                    monkeypatch: pytest.MonkeyPatch) -> None:
        """``k`` above ``CANDIDATES``: BM25 is asked for ``k`` hits, so
        the search still returns ``k`` results (it stopped at 30)."""
        bm25, vectors = indexes
        monkeypatch.setattr(HybridRetriever, "CANDIDATES", 2)
        hits = HybridRetriever(bm25, vectors).search("sampling cuda pip", 5)
        assert len(hits) == len(bm25.search("sampling cuda pip", 5)) == 5

    def test_bm25_hits_untouched(self) -> None:
        """The BM25 hits keep their own scores (copies are returned)."""
        bm25, vectors = plan_example()
        HybridRetriever(bm25, vectors).search("q", 3)  # type: ignore
        assert [h.score for h in bm25.hits] == [12.0, 10.8, 6.0]

    def test_ties_keep_bm25_order(self) -> None:
        """Equal combined scores keep the BM25 order (stable sort)."""
        chunks = make_chunks("a", "b", "c")
        hits = [SearchHit(chunk=chunks[i], score=1.0, chunk_id=i)
                for i in (2, 0, 1)]
        bm25, vectors = FixedBM25(hits, 3), FixedVectors([0.5, 0.5, 0.5])
        retriever = HybridRetriever(bm25, vectors)  # type: ignore
        assert [h.chunk_id for h in retriever.search("q", 3)] == [2, 0, 1]


class TestAlpha:
    """The weight between BM25 and the vectors."""

    @pytest.mark.parametrize("query", QUERIES)
    def test_alpha_zero_is_bm25(self, indexes: tuple[BM25Index, VectorIndex],
                                monkeypatch: pytest.MonkeyPatch,
                                query: str) -> None:
        """``ALPHA = 0``: exactly the BM25 order."""
        bm25, vectors = indexes
        monkeypatch.setattr(HybridRetriever, "ALPHA", 0.0)
        hybrid = HybridRetriever(bm25, vectors).search(query, 10)
        assert ([h.chunk_id for h in hybrid]
                == [h.chunk_id for h in bm25.search(query, 10)])

    @pytest.mark.parametrize("query", QUERIES)
    def test_alpha_one_is_similarity(self,
                                     indexes: tuple[BM25Index, VectorIndex],
                                     monkeypatch: pytest.MonkeyPatch,
                                     query: str) -> None:
        """``ALPHA = 1``: the BM25 candidates, ordered by similarity."""
        bm25, vectors = indexes
        monkeypatch.setattr(HybridRetriever, "ALPHA", 1.0)
        hybrid = HybridRetriever(bm25, vectors).search(query, 10)
        similarity = vectors.chunk_score(query)
        assert ({h.chunk_id for h in hybrid}
                == {h.chunk_id for h in bm25.search(query, 10)})
        assert [h.score for h in hybrid] == pytest.approx(
            sorted((similarity[h.chunk_id] for h in hybrid), reverse=True))

    def test_never_adds_chunks(self,
                               indexes: tuple[BM25Index, VectorIndex]) -> None:
        """Only BM25 candidates come out: reranking, not new results."""
        bm25, vectors = indexes
        for query in QUERIES:
            candidates = {h.chunk_id for h in
                          bm25.search(query, HybridRetriever.CANDIDATES)}
            hybrid = HybridRetriever(bm25, vectors).search(query, 10)
            assert {h.chunk_id for h in hybrid} <= candidates


class TestEdgeCases:
    """No result, mismatched indexes, the chunk ids."""

    def test_no_bm25_hit(self) -> None:
        """Nothing matches: empty list, and no vector query is made."""
        vectors = FixedVectors([0.1, 0.2])
        retriever = HybridRetriever(FixedBM25([], 2), vectors)  # type: ignore
        assert retriever.search("zzzqqq", 5) == []
        assert vectors.queries == 0

    def test_mismatched_indexes(self) -> None:
        """Indexes of different sizes: ``VectorIndexError``."""
        bm25 = BM25Index.build(make_chunks(*TEXTS))
        vectors = VectorIndex.build(make_chunks(*TEXTS[:3]))
        with pytest.raises(VectorIndexError, match="Run 'index' again"):
            HybridRetriever(bm25, vectors)

    def test_error_is_rag_error(self) -> None:
        """The mismatch error is caught with every other ``RagError``."""
        assert issubclass(VectorIndexError, RagError)

    def test_hits_keep_chunk_and_id(self,
                                    indexes: tuple[BM25Index, VectorIndex]
                                    ) -> None:
        """Each hit's chunk is the one its ``chunk_id`` points to."""
        bm25, vectors = indexes
        for hit in HybridRetriever(bm25, vectors).search("sampling", 10):
            assert hit.chunk.text == TEXTS[hit.chunk_id]
