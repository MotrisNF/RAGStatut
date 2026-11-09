"""Local HTTP API (bonus 5): the search, answer and index commands over
HTTP.

``RagApi`` keeps one retriever and, after the first answer, one model
loaded between requests, so every request after the first is fast. With
the answer cache, a question already answered with the same context is
answered without the model. ``/index`` updates the index on disk and
reloads it, so the server never needs a restart. The responses are the
same pydantic models the CLI prints.
"""
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import StringConstraints
from tqdm import tqdm

from .answer_cache import AnswerCache
from .base_retriever import Retriever
from .bm25_index import BM25Index
from .cached_retriever import CachedRetriever
from .commands import RagCLI
from .convenio_search import ConvenioSearch
from .exceptions import RagError, UsageError
from .gemini_retriever import GeminiRetriever
from .hybrid_retriever import HybridRetriever
from .incremental_indexer import IncrementalIndexer
from .models import (NO_SOURCES, Chunk, DtypeMode, IndexSummary,
                     MinimalAnswer, MinimalSource, RetrieverMode,
                     StudentSearchResults, StudentSearchResultsAndAnswer,
                     UnansweredQuestion)
from .vector_index import VectorIndex

if TYPE_CHECKING:
    from .generator import Generator

# A question with text: surrounding spaces are removed first, so a query
# of only spaces is rejected (422), as the CLI does.
QueryText = Annotated[str, StringConstraints(strip_whitespace=True,
                                             min_length=1), Query()]
TopK = Annotated[int, Query(gt=0)]
ChunkSize = Annotated[int, Query(gt=0, le=2000)]
ConvenioName = Annotated[str | None, Query(min_length=1)]


class RetrieverLoader:
    """Loads the retrievers of one index folder for the server.

    ``bm25`` and ``hybrid`` share one ``BM25Index`` (the largest object
    in memory), so a server that uses both does not load it twice. With
    the cache, each retriever is a ``CachedRetriever`` with its own file,
    as in the CLI, and the indexes are only loaded by a search that is
    not cached.
    """

    def __init__(self, index_dir: str, cache: bool) -> None:
        """Remember where the indexes are; nothing is loaded yet.

        Args:
            index_dir: Folder where the index command saved the indexes.
            cache: Wrap each retriever in a ``CachedRetriever``.
        """
        self._index_dir = index_dir
        self._cache = cache
        self._bm25: BM25Index | None = None

    def _shared_bm25(self) -> BM25Index:
        """The BM25 index, loaded the first time one is needed."""
        if self._bm25 is None:
            tqdm.write("Loading the index...", file=sys.stderr)
            self._bm25 = BM25Index.load(Path(self._index_dir))
        return self._bm25

    def _build(self, mode: RetrieverMode) -> Retriever:
        """Load what ``mode`` needs, reusing the shared BM25 index.

        Raises:
            RagError: An index file is missing or broken (as
                ``RagCLI._build_retriever``).
        """
        bm25 = self._shared_bm25()
        if mode == "bm25":
            return bm25
        hybrid = HybridRetriever(bm25, VectorIndex.load(Path(self._index_dir)))
        if mode == "hybrid":
            return hybrid
        return GeminiRetriever(bm25, hybrid,
                               RagCLI.embeddings_dir(self._index_dir))

    def load(self, mode: RetrieverMode) -> Retriever:
        """Give the retriever of ``mode``, cached or not.

        Args:
            mode: ``bm25`` or ``hybrid``.

        Returns:
            The retriever, or a ``CachedRetriever`` around it that loads
            it on the first search not cached.

        Raises:
            RagError: Without the cache, an index file is missing or
                broken.
        """
        if not self._cache or mode == "gemini":
            return self._build(mode)
        return CachedRetriever(
            lambda: self._build(mode),
            RagCLI._cache_namespace(self._index_dir, mode),
            CachedRetriever.file_path(Path(self._index_dir), mode))

    def reset(self) -> None:
        """Forget the loaded BM25 index, after ``index`` replaced it."""
        self._bm25 = None


class RagApi:
    """The FastAPI application and the state shared by its requests.

    FastAPI runs each request of a plain ``def`` endpoint in a thread of
    its own, so the shared objects are guarded by three locks (mutexes):
    one for the retriever and its query cache, one for the model, and one
    for the answer cache. The answer cache lock is only taken alone or
    inside one of the other two, never the other way round, and those two
    are never held together, so they cannot deadlock.

    Every ``RagError`` raised by a request (no index, a broken index,
    the model not fitting in memory...) becomes a 503 with its message:
    never a 500 with a traceback, as the CLI never prints one.

    Attributes:
        DESCRIPTION: Shown at the top of the ``/docs`` page.
        app: The FastAPI application, to run with uvicorn.
    """

    DESCRIPTION = (
        "Question answering over the vLLM codebase. `/search` returns the "
        "best source locations for a question; `/answer` also generates "
        "an answer from them with Qwen/Qwen3-0.6B. The first `/answer` "
        "loads the model (tens of seconds); the next ones reuse it, and "
        "a question already answered with the same sources is read from "
        "the cache. `POST /index` re-indexes the files changed on disk "
        "(or everything, with `full`) and reloads the index."
    )

    def __init__(self, retriever: Retriever,
                 dtype: DtypeMode = "auto",
                 answers: AnswerCache | None = None,
                 index_dir: str | None = None,
                 raw_dir: str = "data/raw",
                 loader: RetrieverLoader | None = None,
                 mode: RetrieverMode = "hybrid"
                 ) -> None:
        """Create the application; the model is not loaded yet.

        Args:
            retriever: The retriever used when a request does not choose
                one (see ``RagCLI._load_retriever``).
            dtype: How the model is loaded by the first ``/answer``:
                ``auto`` chooses from the available memory, ``float32``
                or ``bfloat16`` force it. Set once for the server: the
                model is loaded only once.
            answers: The answer cache, or ``None`` to generate every
                answer (``serve --cache=False``).
            index_dir: Folder of the index, for ``/index``; ``None``
                disables re-indexing (503).
            raw_dir: Corpus folder ``/index`` reads.
            loader: Loads the other retriever when a request asks for it,
                and both again after ``/index`` changed the index.
                ``None``: only ``retriever`` is available, and
                re-indexing is disabled.
            mode: Which retriever ``retriever`` is.
        """
        self._mode = mode
        # Every retriever loaded so far, by mode; the default one first.
        self._retrievers: dict[RetrieverMode, Retriever] = {mode: retriever}
        self._dtype = dtype
        self._answers = answers
        self._index_dir = index_dir
        self._raw_dir = raw_dir
        self._loader = loader
        # The answer cache is a dictionary shared by every request: one
        # read or write (with its flush) at a time.
        self._answers_lock = threading.Lock()
        # Loaded on the first /answer, then kept for the next requests.
        self._generator: "Generator | None" = None
        # FastAPI runs each request in its own thread: one lock so the
        # model is loaded only once and answers one question at a time.
        self._generator_lock = threading.Lock()
        # The retrievers (and their caches) are shared by every request:
        # one search, with its flush, at a time. Separate from the
        # generator lock, so /search does not wait for an answer being
        # generated.
        self._retriever_lock = threading.Lock()
        self.app = FastAPI(title="RAG against the machine",
                           description=self.DESCRIPTION)
        self.app.exception_handler(RagError)(self._rag_error)
        self.app.get("/health")(self.health)
        self.app.get("/search")(self.search)
        self.app.get("/answer")(self.answer)
        self.app.post("/index")(self.index)

    @staticmethod
    def _rag_error(request: Request, error: Exception) -> JSONResponse:
        """Turn a ``RagError`` into a 503 with its message.

        503 (Service Unavailable): the request was valid, but the server
        cannot serve it until something is fixed on its side (run
        ``index``, free memory...). The next request tries again, so
        the server does not need a restart.

        Args:
            request: The request that failed (not used).
            error: The ``RagError``.

        Returns:
            ``{"detail": "<message>"}``, the same shape FastAPI uses for
            its own errors.
        """
        return JSONResponse(status_code=503, content={"detail": str(error)})

    def health(self) -> dict[str, str]:
        """Check that the server is up (it never loads anything)."""
        return {"status": "ok"}

    def _get_retriever(self, mode: RetrieverMode | None) -> Retriever:
        """Return the retriever of ``mode``, loading it the first time.

        Must be called with ``_retriever_lock`` held.

        Args:
            mode: ``bm25``, ``hybrid``, or ``None`` for the server's
                default (``serve --retriever``).

        Returns:
            The retriever.

        Raises:
            HTTPException: 503 when another retriever is asked for and
                this server cannot load it (no loader).
            RagError: Without the query cache, an index file is missing
                or broken (503, through ``_rag_error``).
        """
        mode = mode or self._mode
        if mode not in self._retrievers:
            if self._loader is None:
                raise HTTPException(
                    status_code=503,
                    detail=f"Only the '{self._mode}' retriever is "
                           "available")
            self._retrievers[mode] = self._loader.load(mode)
        return self._retrievers[mode]

    def _check_convenio(self, convenio: str | None) -> None:
        """Answer 400 to an agreement that is not in the corpus.

        Args:
            convenio: The value of the request, or ``None``.

        Raises:
            HTTPException: 400 with the list of valid values.
        """
        if self._index_dir is None:
            return
        try:
            ConvenioSearch.check(convenio, self._index_dir)
        except UsageError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    def search(self, query: QueryText, k: TopK = 10,
               retriever: RetrieverMode | None = None,
               convenio: ConvenioName = None
               ) -> StudentSearchResults:
        """Return the ``k`` best source locations for a question.

        Same JSON as the ``search`` command, without saving a file.
        ``retriever`` chooses ``bm25`` or ``hybrid`` for this request
        (default: the server's, ``serve --retriever``). ``convenio``
        searches that agreement and the Statute, or only the Statute
        with ``estatuto`` (default: the whole corpus).
        """
        self._check_convenio(convenio)
        question = UnansweredQuestion(question=query)
        with self._retriever_lock:
            chosen = self._get_retriever(retriever)
            results = StudentSearchResults(
                search_results=[RagCLI._search_question(
                    chosen,
                    question,
                    k,
                    convenio
                    )],
                k=k
            )
            chosen.flush()
        return results

    def _get_generator(self) -> "Generator":
        """Return the generator, loading it on the first call.

        Must be called with ``_generator_lock`` held.

        Raises:
            HTTPException: 503 when PyTorch cannot be imported.
            RagError: The model does not fit in memory (503, through
                ``_rag_error``). Either way the next request tries again.
        """
        if self._generator is None:
            try:
                print("Loading PyTorch...", file=sys.stderr)
                from .generator import Generator
            except (ImportError, MemoryError, OSError) as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Cannot load PyTorch ({type(e).__name__}: {e})."
                ) from e
            print(f"Loading {Generator.MODEL_NAME}...", file=sys.stderr)
            self._generator = Generator(self._dtype)
        return self._generator

    def _cached(self, key: str) -> str | None:
        """Return the cached answer of a key, if there is a cache.

        Args:
            key: From ``AnswerCache.key``.

        Returns:
            The answer, or ``None`` when it is not cached or there is no
            cache.
        """
        if self._answers is None:
            return None
        with self._answers_lock:
            return self._answers.get(key)

    def _remember(self, key: str, answer: str) -> None:
        """Cache a generated answer and save the cache file at once.

        Args:
            key: From ``AnswerCache.key``.
            answer: The generated answer.
        """
        if self._answers is None:
            return
        with self._answers_lock:
            self._answers.put(key, answer)
            self._answers.flush()

    def _generate(self, query: str, chunks: list[Chunk]) -> str:
        """Answer from the cache, or generate and cache the answer.

        The cache is checked again once the model lock is held: another
        request may have generated the same answer while this one was
        waiting, and it must not be generated twice.

        Args:
            query: The question.
            chunks: The context chunks, in order.

        Returns:
            The answer.

        Raises:
            HTTPException, RagError: As ``_get_generator``, only on a miss.
        """
        key = AnswerCache.key(query, chunks)
        answer = self._cached(key)
        if answer is not None:
            return answer
        with self._generator_lock:
            answer = self._cached(key)
            if answer is None:
                generator = self._get_generator()
                print("Generating an answer...", file=sys.stderr)
                answer = generator.generate(query, chunks)
                self._remember(key, answer)
        return answer

    def answer(self, query: QueryText, k: TopK = 10,
               retriever: RetrieverMode | None = None,
               convenio: ConvenioName = None
               ) -> StudentSearchResultsAndAnswer:
        """Answer a question from its ``k`` best sources.

        Same JSON as the ``answer`` command, without saving a file. When
        no source is found, the answer is a fixed message and the model
        is not loaded. A question already answered with the same sources
        is answered from the cache, without the model. ``retriever``
        chooses how the sources are found and ``convenio`` where, as in
        ``/search``.
        """
        self._check_convenio(convenio)
        question = UnansweredQuestion(question=query)
        with self._retriever_lock:
            chosen = self._get_retriever(retriever)
            hits = ConvenioSearch.search(chosen, query, k, convenio)
            chosen.flush()
        # No source found: nothing to ground an answer on, so the model
        # is not loaded.
        if hits:
            answer = self._generate(query, [hit.chunk for hit in hits])
        else:
            answer = NO_SOURCES
        answered = MinimalAnswer(
            question_id=question.question_id,
            question=query,
            retrieved_sources=[MinimalSource(
                file_path=hit.chunk.file_path,
                first_character_index=hit.chunk.first_character_index,
                last_character_index=hit.chunk.last_character_index
                ) for hit in hits],
            answer=answer
        )
        return StudentSearchResultsAndAnswer(search_results=[answered], k=k)

    def index(self, full: bool = False,
              max_chunk_size: ChunkSize = 2000) -> IndexSummary:
        """Re-index the corpus files that changed, then reload the index.

        The same work as the ``index`` command: only new and modified
        files are processed (everything with ``full``). If the index
        changed, the retrievers are loaded again (the one not in use,
        when a request asks for it) and the answer cache starts empty
        (``index`` removed the cache files). Searches wait until it ends;
        an answer being generated finishes first.
        """
        if self._index_dir is None or self._loader is None:
            raise HTTPException(status_code=503,
                                detail="Re-indexing is not enabled")
        with self._retriever_lock:
            report = IncrementalIndexer(self._raw_dir, self._index_dir,
                                        max_chunk_size).run(full)
            if report.read is not None:
                self._loader.reset()
                self._retrievers = {
                    self._mode: self._loader.load(self._mode)}
                if self._answers is not None:
                    with self._answers_lock:
                        self._answers = AnswerCache(AnswerCache.file_path(
                            Path(self._index_dir)))
        return IndexSummary.from_report(report)
