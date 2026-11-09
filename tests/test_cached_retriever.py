"""Tests for ``src.cached_retriever.CachedRetriever`` and its CLI use.

A stand-in retriever counts how often it is built and searched, so the
tests can tell a cache hit (nothing loaded, nothing searched) from a miss.
The CLI tests build tiny real indexes in ``tmp_path`` (with
``FakeEncoder``, see ``conftest.py``), never in ``data/processed``.
"""
import os
from pathlib import Path

import pytest

from src.base_retriever import Retriever
from src.bm25_index import BM25Index
from src.cached_retriever import CachedRetriever
from src.commands import RagCLI
from src.exceptions import IndexNotFoundError
from src.hybrid_retriever import HybridRetriever
from src.models import Chunk, QueryCache, SearchHit
from src.reader import CorpusReader

TEXTS = ["Load a LoRA adapter at runtime with the API.",
         "Configure the OpenAI compatible server.",
         "LoRA adapters can be served per request."]


def make_hits(query: str, k: int) -> list[SearchHit]:
    """``k`` hits whose text names the query, so results can be told
    apart."""
    return [SearchHit(chunk=Chunk(file_path=f"data/raw/{i}.md",
                                  first_character_index=0,
                                  last_character_index=10,
                                  text=f"{query} {i}"),
                      score=float(k - i), chunk_id=i)
            for i in range(k)]


class Counting(Retriever):
    """Retriever stand-in that records every search."""

    def __init__(self) -> None:
        """Start with no search."""
        self.searches: list[tuple[str, int]] = []

    def search(self, query: str, k: int,
               sources: frozenset[str] | None = None
               ) -> list[SearchHit]:
        """``make_hits(query, k)``; records the call."""
        self.searches.append((query, k))
        return make_hits(query, k)


class Loader:
    """The ``load_inner`` function: builds a ``Counting`` and counts the
    builds."""

    def __init__(self) -> None:
        """No retriever built yet."""
        self.loads = 0
        self.inner = Counting()

    def __call__(self) -> Retriever:
        """Build the retriever (always the same one, to read its
        searches)."""
        self.loads += 1
        return self.inner


@pytest.fixture
def loader() -> Loader:
    """A fresh loader for each test."""
    return Loader()


@pytest.fixture
def path(tmp_path: Path) -> Path:
    """Where the cache file goes; it does not exist yet."""
    return tmp_path / "query_cache_test.json"


class TestSearch:
    """Searches in memory, within one process."""

    def test_nothing_loaded_on_creation(self, loader: Loader,
                                        path: Path) -> None:
        """Creating the cache does not build the wrapped retriever."""
        CachedRetriever(loader, "ns", path)
        assert loader.loads == 0

    def test_miss_then_hit(self, loader: Loader, path: Path) -> None:
        """The first search goes to the retriever, the repeated one
        does not."""
        cache = CachedRetriever(loader, "ns", path)
        first = cache.search("lora", 5)
        second = cache.search("lora", 5)
        assert first == second == make_hits("lora", 5)
        assert loader.inner.searches == [("lora", 5)]

    def test_loaded_once(self, loader: Loader, path: Path) -> None:
        """Several misses build the retriever only once."""
        cache = CachedRetriever(loader, "ns", path)
        for query in ("lora", "cuda", "pip"):
            cache.search(query, 3)
        assert loader.loads == 1
        assert len(loader.inner.searches) == 3

    def test_other_k_is_other_entry(self, loader: Loader,
                                    path: Path) -> None:
        """The same query with another ``k`` is searched again."""
        cache = CachedRetriever(loader, "ns", path)
        assert len(cache.search("lora", 5)) == 5
        assert len(cache.search("lora", 10)) == 10
        assert loader.inner.searches == [("lora", 5), ("lora", 10)]

    def test_returns_copies(self, loader: Loader, path: Path) -> None:
        """Changing the returned list does not change the cache."""
        cache = CachedRetriever(loader, "ns", path)
        cache.search("lora", 3).clear()
        assert len(cache.search("lora", 3)) == 3

    def test_load_error_reaches_caller(self, path: Path) -> None:
        """A failure to load (no index) is not swallowed by the cache."""
        def failing() -> Retriever:
            """Like loading a missing index."""
            raise IndexNotFoundError("idx")
        cache = CachedRetriever(failing, "ns", path)
        with pytest.raises(IndexNotFoundError):
            cache.search("lora", 3)

    def test_is_a_retriever(self, loader: Loader, path: Path) -> None:
        """The commands can use it like any other retriever."""
        assert isinstance(CachedRetriever(loader, "ns", path), Retriever)


class TestFlush:
    """Saving the cache to disk and reading it back."""

    def test_new_instance_reads_saved_results(self, loader: Loader,
                                              path: Path) -> None:
        """Another instance (another command) reuses the results without
        loading the retriever."""
        cache = CachedRetriever(loader, "ns", path)
        cache.search("lora", 5)
        cache.flush()
        other = Loader()
        assert (CachedRetriever(other, "ns", path).search("lora", 5)
                == make_hits("lora", 5))
        assert other.loads == 0

    def test_other_namespace_starts_empty(self, loader: Loader,
                                          path: Path) -> None:
        """Results saved with another namespace (another index or
        retriever) are not reused."""
        cache = CachedRetriever(loader, "old", path)
        cache.search("lora", 5)
        cache.flush()
        other = Loader()
        CachedRetriever(other, "new", path).search("lora", 5)
        assert other.loads == 1

    def test_nothing_new_nothing_written(self, loader: Loader,
                                         path: Path) -> None:
        """No file when nothing was searched."""
        CachedRetriever(loader, "ns", path).flush()
        assert not path.exists()

    def test_only_hits_not_rewritten(self, loader: Loader,
                                     path: Path) -> None:
        """A command whose questions were all cached leaves the file
        untouched."""
        cache = CachedRetriever(loader, "ns", path)
        cache.search("lora", 5)
        cache.flush()
        os.utime(path, ns=(0, 0))
        again = CachedRetriever(loader, "ns", path)
        again.search("lora", 5)
        again.flush()
        assert path.stat().st_mtime_ns == 0

    def test_second_flush_not_rewritten(self, loader: Loader,
                                        path: Path) -> None:
        """After a successful flush, another flush with nothing new does
        not write."""
        cache = CachedRetriever(loader, "ns", path)
        cache.search("lora", 5)
        cache.flush()
        os.utime(path, ns=(0, 0))
        cache.flush()
        assert path.stat().st_mtime_ns == 0

    def test_saved_as_query_cache(self, loader: Loader,
                                  path: Path) -> None:
        """The file is a valid ``QueryCache`` with the namespace and the
        ``"<k>:<query>"`` keys."""
        cache = CachedRetriever(loader, "ns", path)
        cache.search("lora", 2)
        cache.flush()
        data = QueryCache.model_validate_json(path.read_text("utf-8"))
        assert data.namespace == "ns"
        assert list(data.entries) == ["2:lora"]

    def test_no_tmp_left(self, loader: Loader, path: Path) -> None:
        """The temporary file is moved over the cache, not left behind."""
        cache = CachedRetriever(loader, "ns", path)
        cache.search("lora", 2)
        cache.flush()
        assert not path.with_suffix(".tmp").exists()

    def test_unwritable_folder_warns(self, loader: Loader, tmp_path: Path,
                                     capsys: pytest.CaptureFixture[str]
                                     ) -> None:
        """A cache that cannot be written prints a warning, no error."""
        cache = CachedRetriever(loader, "ns", tmp_path / "no" / "c.json")
        cache.search("lora", 2)
        cache.flush()
        err = capsys.readouterr().err
        assert "WARNING: cannot save the query cache" in err

    def test_failed_replace_keeps_old_file(self, loader: Loader, path: Path,
                                           monkeypatch: pytest.MonkeyPatch
                                           ) -> None:
        """If the final move fails, the previous cache is still whole."""
        cache = CachedRetriever(loader, "ns", path)
        cache.search("lora", 2)
        cache.flush()
        before = path.read_bytes()

        def broken(src: object, dst: object) -> None:
            """Like a move interrupted by the system."""
            raise OSError("interrupted")
        monkeypatch.setattr(os, "replace", broken)
        cache.search("cuda", 2)
        cache.flush()
        assert path.read_bytes() == before


class TestBrokenFile:
    """A bad cache file means an empty cache, never an error."""

    @pytest.mark.parametrize("content", [
        b"garbage {",
        b"\xff\xfe not utf-8",
        b'{"namespace": "ns"}',
        b'{"namespace": "ns", "entries": {"5:lora": "not hits"}}',
        b"",
    ])
    def test_starts_empty(self, loader: Loader, path: Path,
                          content: bytes) -> None:
        """Broken JSON, not UTF-8, wrong shape or empty: searched again."""
        path.write_bytes(content)
        CachedRetriever(loader, "ns", path).search("lora", 5)
        assert loader.loads == 1

    def test_folder_instead_of_file(self, loader: Loader,
                                    path: Path) -> None:
        """A folder where the file should be: empty cache."""
        path.mkdir()
        CachedRetriever(loader, "ns", path).search("lora", 5)
        assert loader.loads == 1


class TestFilePath:
    """Each retriever keeps its own cache file."""

    def test_one_file_per_retriever(self, tmp_path: Path) -> None:
        """``bm25`` and ``hybrid`` never share (and overwrite) a file."""
        bm25 = CachedRetriever.file_path(tmp_path, "bm25")
        hybrid = CachedRetriever.file_path(tmp_path, "hybrid")
        assert bm25 != hybrid
        assert bm25.parent == hybrid.parent == tmp_path
        assert bm25.name == "query_cache_bm25.json"


class TestNamespace:
    """``RagCLI._cache_namespace``: changes whenever results may change."""

    @pytest.fixture
    def index_dir(self, tmp_path: Path) -> Path:
        """A folder with an index file and a vectors file (content does
        not matter, only their metadata is read)."""
        (tmp_path / "index.json").write_text("{}", encoding="utf-8")
        (tmp_path / "vectors.npy").write_bytes(b"x")
        return tmp_path

    def test_retrievers_differ(self, index_dir: Path) -> None:
        """``bm25`` and ``hybrid`` give different namespaces."""
        assert (RagCLI._cache_namespace(str(index_dir), "bm25")
                != RagCLI._cache_namespace(str(index_dir), "hybrid"))

    def test_stable(self, index_dir: Path) -> None:
        """Nothing changed: the same namespace."""
        assert (RagCLI._cache_namespace(str(index_dir), "hybrid")
                == RagCLI._cache_namespace(str(index_dir), "hybrid"))

    @pytest.mark.parametrize("name", ["index.json", "vectors.npy"])
    def test_rebuilt_index_changes(self, index_dir: Path,
                                   name: str) -> None:
        """A new modification time of an index file (``index`` run
        again) changes the hybrid namespace."""
        before = RagCLI._cache_namespace(str(index_dir), "hybrid")
        os.utime(index_dir / name, ns=(1, 1))
        assert RagCLI._cache_namespace(str(index_dir), "hybrid") != before

    def test_parameter_changes(self, index_dir: Path,
                               monkeypatch: pytest.MonkeyPatch) -> None:
        """Another ``ALPHA`` (results change) changes the namespace."""
        before = RagCLI._cache_namespace(str(index_dir), "hybrid")
        monkeypatch.setattr(HybridRetriever, "ALPHA", 0.9)
        assert RagCLI._cache_namespace(str(index_dir), "hybrid") != before

    def test_missing_file_no_error(self, tmp_path: Path) -> None:
        """No index: no error here (loading reports it later)."""
        namespace = RagCLI._cache_namespace(str(tmp_path / "none"), "bm25")
        assert "index.json:missing" in namespace


class TestCli:
    """The commands with ``--cache`` (on by default)."""

    @pytest.fixture
    def index_dir(self, tmp_path: Path,
                  capsys: pytest.CaptureFixture[str]) -> Path:
        """A small saved BM25 index (its progress bar is discarded)."""
        chunks = [Chunk(file_path=f"data/raw/{i}.md",
                        first_character_index=0,
                        last_character_index=len(text), text=text)
                  for i, text in enumerate(TEXTS)]
        BM25Index.build(chunks).save(tmp_path)
        capsys.readouterr()
        return tmp_path

    @pytest.fixture
    def loads(self, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
        """Every ``BM25Index.load``, recorded."""
        calls: list[Path] = []
        original = BM25Index.load

        def counting(index_dir: Path) -> BM25Index:
            """Record and load as usual."""
            calls.append(index_dir)
            return original(index_dir)
        monkeypatch.setattr(BM25Index, "load", staticmethod(counting))
        return calls

    def search(self, index_dir: Path, cache: bool = True) -> None:
        """Run the ``search`` command with ``bm25`` (the fixture has no
        vector index)."""
        RagCLI().search(query="lora adapter", k=2, index_dir=str(index_dir),
                        retriever="bm25", cache=cache)

    def test_repeated_search_does_not_load(self, index_dir: Path,
                                           loads: list[Path]) -> None:
        """The second ``search`` of the same question loads no index."""
        self.search(index_dir)
        self.search(index_dir)
        assert len(loads) == 1
        assert CachedRetriever.file_path(index_dir, "bm25").is_file()

    def test_same_output(self, index_dir: Path,
                         capsys: pytest.CaptureFixture[str]) -> None:
        """Cached and uncached searches print the same sources."""
        self.search(index_dir)
        first = capsys.readouterr().out
        self.search(index_dir)
        second = capsys.readouterr().out
        drop_id = [line for line in first.splitlines()
                   if "question_id" not in line]
        assert drop_id == [line for line in second.splitlines()
                           if "question_id" not in line]

    def test_cache_false(self, index_dir: Path, loads: list[Path]) -> None:
        """``--cache=False`` loads every time and writes no cache."""
        self.search(index_dir, cache=False)
        self.search(index_dir, cache=False)
        assert len(loads) == 2
        assert not CachedRetriever.file_path(index_dir, "bm25").exists()

    def test_index_removes_caches(self, monkeypatch: pytest.MonkeyPatch,
                                  tmp_path: Path,
                                  capsys: pytest.CaptureFixture[str]
                                  ) -> None:
        """``index`` removes the cache of every retriever."""
        monkeypatch.setattr(CorpusReader, "PROJECT_ROOT", tmp_path)
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.md").write_text("load a lora adapter", encoding="utf-8")
        index_dir = tmp_path / "idx"
        index_dir.mkdir()
        caches = [CachedRetriever.file_path(index_dir, name)
                  for name in ("bm25", "hybrid")]
        for cache in caches:
            cache.write_text("{}", encoding="utf-8")
        RagCLI().index(raw_dir=str(corpus), index_dir=str(index_dir))
        capsys.readouterr()
        assert not any(cache.exists() for cache in caches)
