"""GeminiRetriever and QuotaGuard, with a fake embedder: never the API."""
import json
from pathlib import Path
from typing import Any

import pytest

from src.base_retriever import Retriever
from src.bm25_index import BM25Index
from src.exceptions import RagError
from src.gemini_retriever import GeminiRetriever, QuotaGuard
from src.models import Chunk, SearchHit

VECTORS = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]}


def chunk(name: str, text: str) -> Chunk:
    """A chunk of the file ``name``."""
    return Chunk(file_path=f"data/raw/{name}.txt", first_character_index=0,
                 last_character_index=len(text), text=text)


class Fallback(Retriever):
    """Records that the local search was used."""

    def __init__(self) -> None:
        """No call yet."""
        self.calls = 0

    def search(self, query: str, k: int,
               sources: frozenset[str] | None = None) -> list[SearchHit]:
        """Return nothing, counting the call."""
        self.calls += 1
        return []


class Embedder:
    """Counts the calls; a question containing ``b`` is near ``b``."""

    def __init__(self, error: bool = False) -> None:
        """Choose whether every call fails."""
        self.calls = 0
        self.error = error

    def __call__(self, text: str) -> list[float]:
        """Give a fixed vector."""
        self.calls += 1
        if self.error:
            raise RagError("HTTP 429")
        return [0.0, 1.0] if "bbb" in text else [1.0, 0.0]


def save(folder: Path, chunks: list[Chunk]) -> None:
    """Write a vector for each chunk, by its text."""
    with (folder / GeminiRetriever.VECTORS_NAME).open("w") as file:
        for item in chunks:
            key = GeminiRetriever.key(GeminiRetriever.document_text(item))
            file.write(json.dumps({"key": key,
                                   "v": VECTORS[item.text]}) + "\n")


CHUNKS = [chunk("tic", "a"), chunk("tic", "b"), chunk("et", "c")]


def build(folder: Path, chunks: list[Chunk] | None = None,
          saved: list[Chunk] | None = None, **kwargs: Any
          ) -> tuple[GeminiRetriever, Fallback]:
    """A retriever over the chunks, with the given saved vectors."""
    chunks = CHUNKS if chunks is None else chunks
    save(folder, chunks if saved is None else saved)
    fallback = Fallback()
    return (GeminiRetriever(BM25Index.build(chunks), fallback, folder,
                            **kwargs), fallback)


def lines(folder: Path) -> int:
    """Calls written in the ledger."""
    path = folder / GeminiRetriever.LEDGER_NAME
    return len(path.read_text().splitlines()) if path.exists() else 0


class TestSearch:
    """Ranking, filter and the saved vectors."""

    def test_best_first(self, tmp_path: Path) -> None:
        """The most similar chunk comes first."""
        retriever, fallback = build(tmp_path, embed=Embedder())
        hits = retriever.search("bbb", 3)
        assert [hit.chunk.text for hit in hits] == ["b", "c", "a"]
        assert [hit.chunk_id for hit in hits][0] == 1
        assert fallback.calls == 0

    def test_sources(self, tmp_path: Path) -> None:
        """Only the chosen files are searched."""
        retriever, _ = build(tmp_path, embed=Embedder())
        hits = retriever.search("bbb", 3, frozenset({"et"}))
        assert [hit.chunk.text for hit in hits] == ["c"]

    def test_one_call_per_question(self, tmp_path: Path) -> None:
        """Agreement and Statute searches share one vector."""
        embed = Embedder()
        retriever, _ = build(tmp_path, embed=embed)
        retriever.search("q", 1, frozenset({"tic"}))
        retriever.search("q", 1, frozenset({"et"}))
        assert embed.calls == 1
        assert lines(tmp_path) == 1

    def test_saved_question_costs_nothing(self, tmp_path: Path) -> None:
        """Another process reads the vector from the file."""
        retriever, _ = build(tmp_path, embed=Embedder())
        retriever.search("q", 1)
        other = Embedder()
        again = GeminiRetriever(BM25Index.build(CHUNKS), Fallback(),
                                tmp_path, embed=other)
        again.search("q", 1)
        assert other.calls == 0


class TestFallback:
    """When Gemini cannot answer, the local search does, without calls."""

    def test_not_embedded_file(self, tmp_path: Path) -> None:
        """A file with a missing vector is never searched by Gemini."""
        embed = Embedder()
        retriever, fallback = build(tmp_path, saved=CHUNKS[:2], embed=embed)
        retriever.search("q", 1, frozenset({"et"}))
        assert fallback.calls == 1
        assert embed.calls == 0
        assert lines(tmp_path) == 0

    def test_no_vectors_at_all(self, tmp_path: Path) -> None:
        """A corpus without vectors behaves as before."""
        embed = Embedder()
        fallback = Fallback()
        GeminiRetriever(BM25Index.build(CHUNKS), fallback, tmp_path,
                        embed=embed).search("q", 1)
        assert (fallback.calls, embed.calls) == (1, 0)

    def test_error_stops_the_calls(self, tmp_path: Path) -> None:
        """No retry: after a failure, no more calls in this process."""
        embed = Embedder(error=True)
        retriever, fallback = build(tmp_path, embed=embed)
        retriever.search("one", 1)
        retriever.search("two", 1)
        assert embed.calls == 1
        assert fallback.calls == 2
        assert lines(tmp_path) == 1

    def test_no_key_no_network(self, tmp_path: Path,
                               monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the key nothing is sent or written."""
        monkeypatch.delenv(GeminiRetriever.API_KEY_ENV, raising=False)

        def forbidden(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("network used")

        monkeypatch.setattr("urllib.request.urlopen", forbidden)
        retriever, fallback = build(tmp_path)
        retriever.search("q", 1)
        assert fallback.calls == 1
        assert lines(tmp_path) == 0

    def test_quota_reached(self, tmp_path: Path) -> None:
        """At the limit the embedder is not called."""
        ledger = tmp_path / GeminiRetriever.LEDGER_NAME
        guard = QuotaGuard(ledger, clock=lambda: 1_000_000.0)
        for _ in range(QuotaGuard.MINUTE_REQUESTS):
            assert guard.reserve(1)
        embed = Embedder()
        retriever, fallback = build(tmp_path, embed=embed, guard=guard)
        retriever.search("q", 1)
        assert (embed.calls, fallback.calls) == (0, 1)


class TestQuotaGuard:
    """Every limit is checked before writing."""

    def make(self, tmp_path: Path) -> tuple[QuotaGuard, list[float]]:
        """A guard with a clock the test moves."""
        now = [1_000_000.0]
        return QuotaGuard(tmp_path / "l.jsonl", clock=lambda: now[0]), now

    def test_minute_requests(self, tmp_path: Path) -> None:
        """The minute limit frees up when the window passes."""
        guard, now = self.make(tmp_path)
        for _ in range(QuotaGuard.MINUTE_REQUESTS):
            assert guard.reserve(1)
        assert not guard.reserve(1)
        now[0] += QuotaGuard.WINDOW
        assert guard.reserve(1)

    def test_minute_tokens(self, tmp_path: Path) -> None:
        """A call that would go over the tokens is refused."""
        guard, _ = self.make(tmp_path)
        assert guard.reserve(QuotaGuard.MINUTE_TOKENS)
        assert not guard.reserve(1)

    def test_day_limit(self, tmp_path: Path) -> None:
        """At the daily limit, nothing more that day; the next day, yes."""
        guard, now = self.make(tmp_path)
        for _ in range(QuotaGuard.DAY_LIMIT):
            assert guard.reserve(1)
            now[0] += 2
        now[0] += QuotaGuard.WINDOW
        assert not guard.reserve(1)
        now[0] += 86400
        assert guard.reserve(1)

    def test_broken_ledger_refuses(self, tmp_path: Path) -> None:
        """If the spending cannot be read, nothing is spent."""
        guard, _ = self.make(tmp_path)
        (tmp_path / "l.jsonl").write_text("not json\n")
        assert not guard.reserve(1)
