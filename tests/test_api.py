"""Tests for ``src.api.RagApi`` (bonus 5), through FastAPI's TestClient.

No test loads PyTorch or the real model: ``/answer`` gets a fake
generator, and the tests of a failed load replace ``src.generator`` in
``sys.modules`` before the endpoint imports it.
"""
import os
import sys
import threading
import time
from pathlib import Path
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

from src.answer_cache import AnswerCache
from src.api import RagApi, RetrieverLoader
from src.base_retriever import Retriever
from src.bm25_index import BM25Index
from src.cached_retriever import CachedRetriever
from src.commands import RagCLI
from src.reader import CorpusReader
from src.exceptions import IndexNotFoundError, RagError
from src.incremental_indexer import IncrementalIndexer
from src.models import (NO_SOURCES, Chunk, RetrieverMode, SearchHit,
                        StudentSearchResults, StudentSearchResultsAndAnswer)

TEXTS = ["Load a LoRA adapter at runtime with the API.",
         "Configure the OpenAI compatible server.",
         "LoRA adapters can be served per request."]


class FakeGenerator:
    """Stand-in for ``Generator``: records the calls, answers at once."""

    def __init__(self) -> None:
        """No call yet."""
        self.calls: list[tuple[str, list[Chunk]]] = []

    def generate(self, question: str, chunks: list[Chunk]) -> str:
        """Record the call; the answer names the number of chunks."""
        self.calls.append((question, chunks))
        return f"Answer from {len(chunks)} chunks."


class Broken(Retriever):
    """Retriever whose index is missing, as before ``index``."""

    def search(self, query: str, k: int,
               sources: frozenset[str] | None = None
               ) -> list[SearchHit]:
        """Always fails with the CLI's error."""
        raise IndexNotFoundError("data/processed")


class Empty(Retriever):
    """Retriever that never finds anything."""

    def search(self, query: str, k: int,
               sources: frozenset[str] | None = None
               ) -> list[SearchHit]:
        """No hits."""
        return []


@pytest.fixture
def bm25(capsys: pytest.CaptureFixture[str]) -> BM25Index:
    """A small BM25 index in memory (its progress bar is discarded)."""
    chunks = [Chunk(file_path=f"data/raw/{i}.md",
                    first_character_index=0,
                    last_character_index=len(text), text=text)
              for i, text in enumerate(TEXTS)]
    index = BM25Index.build(chunks)
    capsys.readouterr()
    return index


@pytest.fixture
def api(bm25: BM25Index) -> RagApi:
    """The API over the small index, with a fake model already loaded."""
    rag = RagApi(bm25)
    rag._generator = FakeGenerator()  # type: ignore[assignment]
    return rag


@pytest.fixture
def client(api: RagApi) -> TestClient:
    """HTTP client for ``api``."""
    return TestClient(api.app)


def fake_generator_module(monkeypatch: pytest.MonkeyPatch,
                          generator_class: type | None) -> None:
    """Make ``from .generator import Generator`` give ``generator_class``
    (``None``: the import fails, as without PyTorch)."""
    if generator_class is None:
        monkeypatch.setitem(sys.modules, "src.generator", None)
        return
    module = ModuleType("src.generator")
    module.Generator = generator_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.generator", module)


class TestHealth:
    """``/health``."""

    def test_ok(self, client: TestClient) -> None:
        """Always ``{"status": "ok"}``."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestSearch:
    """``/search``."""

    def test_valid_results(self, client: TestClient) -> None:
        """A ``StudentSearchResults`` with the question and its sources."""
        response = client.get("/search",
                              params={"query": "lora adapter", "k": 2})
        assert response.status_code == 200
        results = StudentSearchResults.model_validate(response.json())
        assert results.k == 2
        [found] = results.search_results
        assert found.question == "lora adapter"
        assert len(found.retrieved_sources) == 2
        assert found.retrieved_sources[0].file_path in (
            "data/raw/0.md", "data/raw/2.md")

    def test_default_k(self, client: TestClient) -> None:
        """Without ``k``, 10 (the CLI default)."""
        response = client.get("/search", params={"query": "lora"})
        assert response.json()["k"] == 10

    def test_no_text_in_output(self, client: TestClient) -> None:
        """Only locations: no chunk text and no score."""
        response = client.get("/search", params={"query": "lora"})
        source = response.json()["search_results"][0]["retrieved_sources"][0]
        assert set(source) == {"file_path", "first_character_index",
                               "last_character_index"}

    def test_new_id_each_request(self, client: TestClient) -> None:
        """Every request is a new question, with its own id."""
        ids = {client.get("/search", params={"query": "lora"}).json()
               ["search_results"][0]["question_id"] for _ in range(2)}
        assert len(ids) == 2


class TestValidation:
    """Invalid parameters give 422, on both endpoints."""

    @pytest.mark.parametrize("endpoint", ["/search", "/answer"])
    @pytest.mark.parametrize("params", [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": "\t\n"},
        {"query": "lora", "k": 0},
        {"query": "lora", "k": -1},
        {"query": "lora", "k": "abc"},
        {"query": "lora", "k": 1.5},
    ])
    def test_rejected(self, client: TestClient, api: RagApi, endpoint: str,
                      params: dict[str, str | int | float]) -> None:
        """422 and the model is never called."""
        response = client.get(endpoint, params=params)
        assert response.status_code == 422
        assert api._generator.calls == []  # type: ignore[union-attr]

    def test_spaces_stripped(self, client: TestClient) -> None:
        """Spaces around the question are removed, as in the CLI."""
        response = client.get("/search", params={"query": "  lora  "})
        assert response.status_code == 200
        assert response.json()["search_results"][0]["question"] == "lora"

    def test_unknown_path(self, client: TestClient) -> None:
        """404 for an endpoint that does not exist."""
        assert client.get("/nope").status_code == 404


class TestAnswer:
    """``/answer`` with the fake model."""

    def test_valid_answer(self, client: TestClient, api: RagApi) -> None:
        """A ``StudentSearchResultsAndAnswer`` with the generated text."""
        response = client.get("/answer",
                              params={"query": "lora adapter", "k": 2})
        assert response.status_code == 200
        results = StudentSearchResultsAndAnswer.model_validate(
            response.json())
        assert results.k == 2
        [answered] = results.search_results
        assert answered.question == "lora adapter"
        assert answered.answer == "Answer from 2 chunks."
        assert len(answered.retrieved_sources) == 2

    def test_model_gets_chunk_texts(self, client: TestClient,
                                    api: RagApi) -> None:
        """The model receives the question and the full chunks, in the
        same order as the sources."""
        response = client.get("/answer",
                              params={"query": "lora adapter", "k": 2})
        [(question, chunks)] = api._generator.calls  # type: ignore
        assert question == "lora adapter"
        assert all(chunk.text in TEXTS for chunk in chunks)
        sources = response.json()["search_results"][0]["retrieved_sources"]
        assert [c.file_path for c in chunks] == [
            s["file_path"] for s in sources]

    def test_no_sources(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No hit: the fixed message, and the model is not loaded."""
        fake_generator_module(monkeypatch, None)
        rag = RagApi(Empty())
        response = TestClient(rag.app).get("/answer",
                                           params={"query": "lora"})
        assert response.status_code == 200
        answered = response.json()["search_results"][0]
        assert answered["answer"] == NO_SOURCES
        assert answered["retrieved_sources"] == []
        assert rag._generator is None


class TestModelLoading:
    """The model: loaded once, on the first ``/answer``."""

    def test_not_loaded_at_start(self, bm25: BM25Index) -> None:
        """Creating the API and searching never load it."""
        rag = RagApi(bm25)
        TestClient(rag.app).get("/search", params={"query": "lora"})
        assert rag._generator is None

    def test_loaded_once_with_dtype(self, bm25: BM25Index,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
        """Several answers, one load, with the server's dtype."""
        dtypes: list[str] = []

        class Counting(FakeGenerator):
            """Records the dtype of each load."""

            MODEL_NAME = "fake"

            def __init__(self, dtype: str) -> None:
                """Record the load."""
                super().__init__()
                dtypes.append(dtype)

        fake_generator_module(monkeypatch, Counting)
        client = TestClient(RagApi(bm25, "bfloat16").app)
        for _ in range(3):
            assert client.get("/answer",
                              params={"query": "lora"}).status_code == 200
        assert dtypes == ["bfloat16"]

    def test_no_pytorch_503(self, bm25: BM25Index,
                            monkeypatch: pytest.MonkeyPatch) -> None:
        """PyTorch cannot be imported: 503 with the reason."""
        fake_generator_module(monkeypatch, None)
        response = TestClient(RagApi(bm25).app).get(
            "/answer", params={"query": "lora"})
        assert response.status_code == 503
        assert "Cannot load PyTorch" in response.json()["detail"]

    def test_not_enough_memory_503(self, bm25: BM25Index,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
        """The model refuses to load (RagError): 503 with its message,
        and the server keeps answering ``/search``."""

        class NoMemory:
            """Fails like ``Generator`` without enough memory."""

            MODEL_NAME = "fake"

            def __init__(self, dtype: str) -> None:
                """Always refuse."""
                raise RagError("Not enough memory to load the model.")

        fake_generator_module(monkeypatch, NoMemory)
        rag = RagApi(bm25)
        client = TestClient(rag.app)
        response = client.get("/answer", params={"query": "lora"})
        assert response.status_code == 503
        assert response.json() == {
            "detail": "Not enough memory to load the model."}
        assert rag._generator is None
        assert client.get("/search",
                          params={"query": "lora"}).status_code == 200


class TestRagErrors:
    """A ``RagError`` is a 503 with its message, never a 500."""

    @pytest.mark.parametrize("endpoint", ["/search", "/answer"])
    def test_index_missing_503(self, endpoint: str) -> None:
        """The retriever fails: 503 with the CLI's message."""
        response = TestClient(RagApi(Broken()).app).get(
            endpoint, params={"query": "lora"})
        assert response.status_code == 503
        assert response.json() == {
            "detail": "No index found in 'data/processed'. "
                      "Run 'index' first."}

    def test_server_keeps_running(self) -> None:
        """After a 503 the server still answers."""
        client = TestClient(RagApi(Broken()).app)
        client.get("/search", params={"query": "lora"})
        assert client.get("/health").status_code == 200

    def test_index_built_later(self, tmp_path: Path, bm25: BM25Index,
                               capsys: pytest.CaptureFixture[str]) -> None:
        """With the cache (the ``serve`` default) the index is loaded by
        the first search: a 503 before ``index``, results after it,
        without restarting the server."""
        retriever = RagCLI._load_retriever(str(tmp_path), "bm25", True)
        client = TestClient(RagApi(retriever).app)
        params = {"query": "lora adapter", "k": 2}
        assert client.get("/search", params=params).status_code == 503
        bm25.save(tmp_path)
        capsys.readouterr()
        response = client.get("/search", params=params)
        assert response.status_code == 200
        assert len(response.json()["search_results"][0]
                   ["retrieved_sources"]) == 2


class TestConcurrency:
    """The locks: requests at the same time do not run together."""

    def test_one_search_at_a_time(self) -> None:
        """Two searches in parallel never overlap inside the retriever."""

        class Slow(Retriever):
            """Records how many searches run at the same time."""

            def __init__(self) -> None:
                """Nothing running yet."""
                self.running = 0
                self.most = 0

            def search(self, query: str, k: int,
                       sources: frozenset[str] | None = None
                       ) -> list[SearchHit]:
                """Wait a little while counted as running."""
                self.running += 1
                self.most = max(self.most, self.running)
                time.sleep(0.05)
                self.running -= 1
                return []

        retriever = Slow()
        client = TestClient(RagApi(retriever).app)
        threads = [threading.Thread(
            target=client.get, args=("/search",),
            kwargs={"params": {"query": "lora"}}) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert retriever.most == 1


class TestCache:
    """With a ``CachedRetriever``, every request saves the cache."""

    def test_saved_after_request(self, bm25: BM25Index,
                                 tmp_path: Path) -> None:
        """A new question is in the cache file after ``/search``."""
        path = tmp_path / "query_cache_bm25.json"
        cached = CachedRetriever(lambda: bm25, "test", path)
        TestClient(RagApi(cached).app).get(
            "/search", params={"query": "lora", "k": 2})
        assert "2:lora" in path.read_text(encoding="utf-8")


class TestAnswerCache:
    """``/answer`` with an answer cache (``serve`` with ``--cache``)."""

    PARAMS = {"query": "lora adapter", "k": 2}

    @pytest.fixture
    def answers(self, tmp_path: Path) -> AnswerCache:
        """An empty answer cache in ``tmp_path``."""
        return AnswerCache(AnswerCache.file_path(tmp_path))

    @staticmethod
    def answer_text(client: TestClient) -> str:
        """The answer of one ``/answer`` request."""
        response = client.get("/answer", params=TestAnswerCache.PARAMS)
        assert response.status_code == 200
        return str(response.json()["search_results"][0]["answer"])

    def test_second_request_from_cache(self, bm25: BM25Index,
                                       answers: AnswerCache) -> None:
        """The same question twice: generated once, same answer."""
        api = RagApi(bm25, answers=answers)
        fake = FakeGenerator()
        api._generator = fake  # type: ignore[assignment]
        client = TestClient(api.app)
        first = self.answer_text(client)
        assert self.answer_text(client) == first
        assert len(fake.calls) == 1

    def test_saved_after_request(self, bm25: BM25Index, tmp_path: Path,
                                 answers: AnswerCache) -> None:
        """A new answer is in the file after the request, so a server
        started later reuses it."""
        api = RagApi(bm25, answers=answers)
        api._generator = FakeGenerator()  # type: ignore[assignment]
        first = self.answer_text(TestClient(api.app))
        later = RagApi(bm25, answers=AnswerCache(
            AnswerCache.file_path(tmp_path)))
        fake = FakeGenerator()
        later._generator = fake  # type: ignore[assignment]
        assert self.answer_text(TestClient(later.app)) == first
        assert fake.calls == []

    def test_hit_never_loads_the_model(self, bm25: BM25Index,
                                       answers: AnswerCache,
                                       monkeypatch: pytest.MonkeyPatch
                                       ) -> None:
        """A cached answer is returned without loading the model, even
        with PyTorch missing."""
        loaded = RagApi(bm25, answers=answers)
        loaded._generator = FakeGenerator()  # type: ignore[assignment]
        self.answer_text(TestClient(loaded.app))

        fake_generator_module(monkeypatch, None)
        cold = RagApi(bm25, answers=answers)
        assert self.answer_text(TestClient(cold.app))
        assert cold._generator is None

    def test_no_cache_always_generates(self, api: RagApi) -> None:
        """Without a cache (``--cache=False``), every request generates."""
        client = TestClient(api.app)
        self.answer_text(client)
        self.answer_text(client)
        assert len(api._generator.calls) == 2  # type: ignore[union-attr]

    def test_same_question_in_parallel_generated_once(
            self, bm25: BM25Index, answers: AnswerCache) -> None:
        """Four identical requests at once: the ones that waited for the
        model find the answer in the cache, so it is generated once."""

        class Slow(FakeGenerator):
            """Takes a while, like a real generation."""

            def generate(self, question: str, chunks: list[Chunk]) -> str:
                """Wait, then answer."""
                time.sleep(0.1)
                return super().generate(question, chunks)

        api = RagApi(bm25, answers=answers)
        slow = Slow()
        api._generator = slow  # type: ignore[assignment]
        client = TestClient(api.app)
        threads = [threading.Thread(target=client.get, args=("/answer",),
                                    kwargs={"params": self.PARAMS})
                   for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(slow.calls) == 1

    def test_search_does_not_wait_for_generation(
            self, bm25: BM25Index, answers: AnswerCache) -> None:
        """``/search`` answers while an answer is being generated."""
        started = threading.Event()
        release = threading.Event()

        class Blocking(FakeGenerator):
            """Generates only when the test allows it."""

            def generate(self, question: str, chunks: list[Chunk]) -> str:
                """Signal the start, then wait for the release."""
                started.set()
                release.wait(5)
                return super().generate(question, chunks)

        api = RagApi(bm25, answers=answers)
        api._generator = Blocking()  # type: ignore[assignment]
        client = TestClient(api.app)
        answering = threading.Thread(target=client.get, args=("/answer",),
                                     kwargs={"params": self.PARAMS})
        answering.start()
        assert started.wait(5)
        response = client.get("/search", params={"query": "lora"})
        release.set()
        answering.join()
        assert response.status_code == 200

    def test_serve_passes_the_cache(self, bm25: BM25Index, tmp_path: Path,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
        """``serve`` gives ``RagApi`` an answer cache in ``index_dir``,
        and none with ``--cache=False``."""
        bm25.save(tmp_path)
        made: list[RagApi] = []

        def run(app: object, host: str, port: int) -> None:
            """Do not start a server."""

        real_init = RagApi.__init__

        def init(self: RagApi, *args: object, **kwargs: object) -> None:
            """Record the new API."""
            real_init(self, *args, **kwargs)  # type: ignore[arg-type]
            made.append(self)

        monkeypatch.setattr("uvicorn.run", run)
        monkeypatch.setattr(RagApi, "__init__", init)
        RagCLI().serve(index_dir=str(tmp_path), retriever="bm25")
        RagCLI().serve(index_dir=str(tmp_path), retriever="bm25",
                       cache=False)
        assert made[0]._answers is not None
        assert made[0]._answers._path == AnswerCache.file_path(tmp_path)
        assert made[1]._answers is None
        assert made[0]._index_dir == str(tmp_path)
        assert made[0]._raw_dir == "data/raw"


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
           capsys: pytest.CaptureFixture[str]) -> Path:
    """A corpus in a project root at ``tmp_path``, already indexed into
    ``tmp_path/index`` (progress bars discarded)."""
    monkeypatch.setattr(CorpusReader, "PROJECT_ROOT", tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.md").write_text("Load a LoRA adapter at runtime.\n",
                              encoding="utf-8")
    (raw / "b.md").write_text("Configure the server port.\n",
                              encoding="utf-8")
    IncrementalIndexer(str(raw), str(tmp_path / "index"), 2000).run()
    capsys.readouterr()
    return tmp_path


class TestIndex:
    """``POST /index``: re-index the changed files and reload."""

    @staticmethod
    def make_api(corpus: Path, cache: bool = True,
                 mode: RetrieverMode = "bm25") -> RagApi:
        """The API as ``serve`` builds it."""
        index_dir = str(corpus / "index")
        answers = (AnswerCache(AnswerCache.file_path(Path(index_dir)))
                   if cache else None)
        loader = RetrieverLoader(index_dir, cache)
        return RagApi(loader.load(mode), answers=answers,
                      index_dir=index_dir, raw_dir=str(corpus / "raw"),
                      loader=loader, mode=mode)

    @staticmethod
    def edit(corpus: Path, text: str) -> None:
        """Rewrite ``a.md`` with a newer modification time."""
        path = corpus / "raw" / "a.md"
        stat = path.stat()
        path.write_text(text, encoding="utf-8")
        later = stat.st_mtime_ns + 10**9
        os.utime(path, ns=(later, later))

    def test_disabled_without_settings(self, api: RagApi) -> None:
        """An API built without an index folder cannot re-index: 503."""
        response = TestClient(api.app).post("/index")
        assert response.status_code == 503
        assert response.json() == {"detail": "Re-indexing is not enabled"}

    def test_up_to_date(self, corpus: Path) -> None:
        """Nothing changed: nothing written, the retriever is kept."""
        api = self.make_api(corpus)
        before = api._retrievers["bm25"]
        response = TestClient(api.app).post("/index")
        assert response.status_code == 200
        assert response.json()["up_to_date"] is True
        assert response.json()["fresh_count"] == 0
        assert api._retrievers["bm25"] is before

    def test_changed_file_searched_at_once(self, corpus: Path) -> None:
        """A modified file is re-indexed and the next search finds its
        new text, without restarting the server."""
        api = self.make_api(corpus)
        client = TestClient(api.app)
        assert client.get("/search", params={"query": "quantization"}
                          ).json()["search_results"][0][
                              "retrieved_sources"] == []
        self.edit(corpus, "Quantization with bitsandbytes.\n")
        summary = client.post("/index").json()
        assert summary["up_to_date"] is False
        assert summary["full"] is False
        assert summary["modified"] == ["raw/a.md"]
        sources = client.get("/search", params={"query": "quantization"}
                             ).json()["search_results"][0][
                                 "retrieved_sources"]
        assert [s["file_path"] for s in sources] == ["raw/a.md"]

    def test_full(self, corpus: Path) -> None:
        """``full=true`` re-indexes everything even with no change."""
        response = TestClient(self.make_api(corpus).app).post(
            "/index", params={"full": "true"})
        assert response.json()["full"] is True
        assert response.json()["new"] == ["raw/a.md", "raw/b.md"]

    def test_answer_cache_reset(self, corpus: Path) -> None:
        """A changed index removes the answer cache file and the server
        starts from an empty one; an index up to date keeps it."""
        api = self.make_api(corpus)
        assert api._answers is not None
        api._answers.put("k", "Old answer.")
        api._answers.flush()
        client = TestClient(api.app)
        client.post("/index")
        assert api._answers.get("k") == "Old answer."
        self.edit(corpus, "Something else.\n")
        client.post("/index")
        assert api._answers.get("k") is None
        assert not AnswerCache.file_path(corpus / "index").exists()

    def test_no_answer_cache(self, corpus: Path) -> None:
        """With ``--cache=False`` there is no answer cache to reset."""
        api = self.make_api(corpus, cache=False)
        self.edit(corpus, "Something else.\n")
        assert TestClient(api.app).post("/index").status_code == 200
        assert api._answers is None

    @pytest.mark.parametrize("params", [{"max_chunk_size": "0"},
                                        {"max_chunk_size": "2001"},
                                        {"full": "maybe"}])
    def test_invalid_parameters(self, corpus: Path,
                                params: dict[str, str]) -> None:
        """Invalid options: 422, nothing indexed."""
        response = TestClient(self.make_api(corpus).app).post(
            "/index", params=params)
        assert response.status_code == 422

    def test_get_not_allowed(self, corpus: Path) -> None:
        """``/index`` changes the index, so it is a POST only."""
        response = TestClient(self.make_api(corpus).app).get("/index")
        assert response.status_code == 405

    def test_corpus_error_is_503(self, corpus: Path) -> None:
        """A corpus folder that is gone: 503 with the CLI's message, and
        the server keeps the index it had."""
        api = self.make_api(corpus)
        before = api._retrievers["bm25"]
        for path in (corpus / "raw").iterdir():
            path.unlink()
        (corpus / "raw").rmdir()
        response = TestClient(api.app).post("/index")
        assert response.status_code == 503
        assert "Corpus folder not found" in response.json()["detail"]
        assert api._retrievers["bm25"] is before

    def test_index_reloads_both(self, corpus: Path,
                                monkeypatch: pytest.MonkeyPatch) -> None:
        """After a change, the retrievers start again from the new files:
        the default one at once, the other one when asked for."""
        api = self.make_api(corpus, mode="hybrid")
        client = TestClient(api.app)
        client.get("/search", params={"query": "lora",
                                      "retriever": "bm25"})
        assert set(api._retrievers) == {"hybrid", "bm25"}
        self.edit(corpus, "Quantization with bitsandbytes.\n")
        client.post("/index")
        assert set(api._retrievers) == {"hybrid"}
        sources = client.get("/search", params={
            "query": "quantization", "retriever": "bm25"}).json()[
                "search_results"][0]["retrieved_sources"]
        assert [s["file_path"] for s in sources] == ["raw/a.md"]


class TestRetrieverChoice:
    """``?retriever=`` on ``/search`` and ``/answer``."""

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

    @staticmethod
    def sources(client: TestClient, endpoint: str,
                params: dict[str, str]) -> list[str]:
        """The source paths of one request."""
        response = client.get(endpoint, params=params)
        assert response.status_code == 200
        return [source["file_path"] for source in
                response.json()["search_results"][0]["retrieved_sources"]]

    def test_default_only_loaded(self, corpus: Path) -> None:
        """No choice: the server's retriever, and only that one."""
        api = TestIndex.make_api(corpus, mode="hybrid")
        self.sources(TestClient(api.app), "/search", {"query": "lora"})
        assert set(api._retrievers) == {"hybrid"}

    def test_other_loaded_on_demand(self, corpus: Path) -> None:
        """Asking for the other retriever loads it once and keeps it."""
        api = TestIndex.make_api(corpus, mode="hybrid")
        client = TestClient(api.app)
        self.sources(client, "/search", {"query": "lora",
                                         "retriever": "bm25"})
        first = api._retrievers["bm25"]
        self.sources(client, "/search", {"query": "server",
                                         "retriever": "bm25"})
        assert api._retrievers["bm25"] is first

    def test_one_bm25_index_shared(self, corpus: Path,
                                   loads: list[Path]) -> None:
        """``bm25`` and ``hybrid`` share one BM25 index in memory."""
        api = TestIndex.make_api(corpus, mode="hybrid")
        client = TestClient(api.app)
        self.sources(client, "/search", {"query": "lora"})
        self.sources(client, "/search", {"query": "lora",
                                         "retriever": "bm25"})
        assert len(loads) == 1

    def test_same_results_as_cli(self, corpus: Path) -> None:
        """Each choice gives what that retriever gives in the CLI."""
        api = TestIndex.make_api(corpus, cache=False, mode="hybrid")
        client = TestClient(api.app)
        index_dir = str(corpus / "index")
        for mode in ("bm25", "hybrid"):
            expected = RagCLI._build_retriever(index_dir, mode).search(
                "lora adapter", 2)
            assert self.sources(client, "/search", {
                "query": "lora adapter", "k": "2", "retriever": mode}) == [
                    hit.chunk.file_path for hit in expected]

    def test_answer_uses_the_choice(self, corpus: Path) -> None:
        """``/answer`` takes the same parameter."""
        api = TestIndex.make_api(corpus, mode="hybrid")
        api._generator = FakeGenerator()  # type: ignore[assignment]
        self.sources(TestClient(api.app), "/answer",
                     {"query": "lora", "retriever": "bm25"})
        assert set(api._retrievers) == {"hybrid", "bm25"}

    def test_invalid_retriever(self, corpus: Path) -> None:
        """Another value: 422."""
        response = TestClient(TestIndex.make_api(corpus).app).get(
            "/search", params={"query": "lora", "retriever": "vector"})
        assert response.status_code == 422

    def test_without_loader_only_default(self, api: RagApi) -> None:
        """An API without a loader only has its retriever: asking for
        another one is a 503, asking for it by name works."""
        client = TestClient(api.app)
        params = {"query": "lora", "retriever": "bm25"}
        response = client.get("/search", params=params)
        assert response.status_code == 503
        assert "Only the 'hybrid' retriever" in response.json()["detail"]
        assert client.get("/search", params={
            "query": "lora", "retriever": "hybrid"}).status_code == 200
