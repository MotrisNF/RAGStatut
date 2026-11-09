"""Command-line interface of the RAG system, exposed through Python Fire.

Every public method of ``RagCLI`` is a command:
``uv run python -m src <command> [options]``.
"""
import functools
import inspect
import sys
from types import SimpleNamespace
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import (TYPE_CHECKING, Annotated, Any, Callable, Literal,
                    TypeVar, cast, get_args, get_origin)
import uuid

from pydantic import (BaseModel, ConfigDict, Field, FilePath, ValidationError,
                      validate_call)
from tqdm import tqdm

from .answer_cache import AnswerCache
from .base_retriever import Retriever
from .bm25_index import BM25Index
from .cached_retriever import CachedRetriever
from .convenio_search import ConvenioSearch
from .exceptions import (ArgError, DatasetError, DatasetEvaluationError,
                         PathError, RagError, ResultsEvaluationError)
from .evaluator import Evaluator
from .gemini_retriever import GeminiRetriever
from .models import (Chunk, DtypeMode, MinimalSearchResults, MinimalSource,
                     StudentSearchResults, RagDataset, UnansweredQuestion,
                     EvaluationInform, MinimalAnswer,
                     StudentSearchResultsAndAnswer, NO_SOURCES, RetrieverMode,)
from .reader import CorpusReader
from .vector_index import VectorIndex
from .hybrid_retriever import HybridRetriever
from .incremental_indexer import IncrementalIndexer

if TYPE_CHECKING:
    # Only for the type hints: importing it would load PyTorch.
    from .generator import Generator

# Defaults of the dataset commands, so that search_dataset, answer_dataset
# and evaluate run one after the other without options: the results of
# search_dataset are where answer_dataset and evaluate look for them.
DEFAULT_DATASET = "dataset_docs_public.json"
DATASET_QUESTIONS = (Path("data/datasets/UnansweredQuestions")
                     / DEFAULT_DATASET)
DATASET_ANSWERS = Path("data/datasets/AnsweredQuestions") / DEFAULT_DATASET
SEARCH_RESULTS = (Path("data/output/search_results/UnansweredQuestions")
                  / DEFAULT_DATASET)

F = TypeVar("F", bound=Callable[..., Any])
M = TypeVar("M", bound=BaseModel)


class CliCommand:
    """Decorator that turns a method into a validated CLI command."""

    # validate_default: defaults are checked like given values, so a
    # default FilePath arrives as a Path and a missing default file is a
    # clear argument error, not a crash inside the command.
    CONFIG = ConfigDict(
        coerce_numbers_to_str=True,
        str_strip_whitespace=True,
        validate_default=True
    )

    @staticmethod
    def validated(func: F) -> F:
        """Validate the arguments with pydantic and show plain types to Fire.

        1. ``validate_call`` checks every argument against its annotation
           (including ``Annotated`` constraints).
        2. If an ARGUMENT is invalid, its ``ValidationError`` is turned into
           an ``ArgError`` that names each option (``--k '0': ...``).
           Validation errors raised inside the command body (e.g. a broken
           dataset) are not arguments and are left untouched: they are
           told apart by their title, which is the method name only for
           argument errors.
        3. Fire reads the parameter types from ``__signature__``. A copy of
           the signature without the ``Annotated`` wrapper is stored there,
           so ``--help`` shows ``int`` instead of ``Annotated``. pydantic
           keeps using the original annotations.

        Args:
            func: The method to decorate.

        Returns:
            The validating command, with the same type as ``func``.
        """
        validator = validate_call(config=CliCommand.CONFIG)(func)
        signature = inspect.signature(func)
        names = list(signature.parameters)

        @functools.wraps(func)
        def command(*args: Any, **kwargs: Any) -> Any:
            """Run ``func`` after validating its arguments."""
            try:
                return validator(*args, **kwargs)
            except ValidationError as e:
                if e.title != func.__qualname__:
                    raise
                raise ArgError(e, names) from e

        params = [
            param.replace(
                annotation=CliCommand._help_type(param.annotation))
            for param in signature.parameters.values()
        ]
        setattr(command, "__signature__",
                signature.replace(parameters=params))
        return cast(F, command)

    @staticmethod
    def _help_type(annotation: Any) -> Any:
        """The type Fire shows in ``--help`` for one parameter.

        Fire prints ``annotation.__qualname__``: ``Annotated`` is unwrapped
        to its real type, and a ``Literal`` (whose name is only
        ``Literal``) becomes an object named after its values
        (``auto | float32 | bfloat16``).

        Args:
            annotation: The parameter's annotation in the method.

        Returns:
            The annotation to show; only the help uses it, pydantic keeps
            validating with the original one.
        """
        if get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]
        if get_origin(annotation) is Literal:
            values = " | ".join(str(v) for v in get_args(annotation))
            return SimpleNamespace(__qualname__=values)
        return annotation


class RagCLI:
    """RAG over the vLLM codebase: index, search, answer and evaluate.

    Arguments are validated by pydantic (``CliCommand.validated``) before
    each command runs, so invalid values raise ``ValidationError``.

    Attributes:
        MAX_IDS_SHOWN: How many question ids ``evaluate`` lists before
            summarising the rest.
        FLUSH_EVERY: ``answer_dataset`` saves the answer cache after this
            many generated answers (~15 s each), and at the end, even
            when interrupted.
        SEPARATOR: Line between the sections of the ``evaluate`` report.
    """

    MAX_IDS_SHOWN: int = 5
    FLUSH_EVERY: int = 10
    SEPARATOR: str = "=" * 35

    @CliCommand.validated
    def index(
            self,
            max_chunk_size: Annotated[int, Field(gt=0, le=2000)] = 2000,
            raw_dir: Annotated[str, Field(min_length=1)] = "data/raw",
            index_dir: Annotated[str, Field(min_length=1)] = "data/processed",
            full: bool = False
            ) -> None:
        """Read the corpus in raw_dir, build the search indexes, save them.

        Steps: read the files (CorpusReader), split them into chunks
        (PyChunker for Python, ArticleChunker for Markdown and text), build the
        BM25 index and the vector index from the same chunks, and save
        both in index_dir, where the search commands load them from.

        Incremental (bonus 3): when index_dir already holds an index of
        the same raw_dir and max_chunk_size, only the new and modified
        files are read and indexed, and the chunks of modified and
        deleted files are removed (``IncrementalIndexer``).

        Args:
            max_chunk_size: Maximum length of a chunk, in characters (the
                moulinette rejects sources longer than 2000).
            raw_dir: Folder with the corpus to index. Relative paths are
                resolved from the working directory.
            index_dir: Folder where the index is saved; created if needed.
                When the index changes, its query cache is removed.
            full: Index the whole corpus again, even if nothing changed
                (``--full``).
        """
        report = IncrementalIndexer(raw_dir, index_dir,
                                    max_chunk_size).run(full)
        if report.read is None:
            print(f"Index up to date: no file changed in {raw_dir} "
                  f"({report.chunk_count} chunks in {index_dir}/)")
            return

        result = report.read
        characters = sum(len(doc.text) for doc in result.documents)
        extensions = Counter(
            PurePosixPath(doc.file_path).suffix for doc in result.documents
        )
        print(f"Read {len(result.documents)} of {result.found} files "
              f"({characters:,} characters) from {raw_dir}")
        if result.documents:
            print("  " + " | ".join(
                f"{ext}: {count}" for ext, count in sorted(extensions.items())
            ))
        if result.skipped:
            print(f"Skipped {len(result.skipped)} unreadable files:")
            for path in result.skipped:
                print(f"  {path}")

        if not report.full:
            changes = report.changes
            print(f"Incremental: {len(changes.new)} new, "
                  f"{len(changes.modified)} modified, "
                  f"{len(changes.deleted)} deleted")
        print(f"{report.chunk_count} chunks (max {max_chunk_size} "
              f"characters), {report.fresh_count} indexed now")
        print(f"Ingestion complete! Index saved under {index_dir}/")

    @CliCommand.validated
    def search(
            self,
            query: Annotated[str, Field(min_length=1)],
            k: Annotated[int, Field(gt=0)] = 10,
            index_dir: Annotated[str, Field(min_length=1)] = "data/processed",
            retriever: RetrieverMode = "hybrid",
            cache: bool = True,
            save_directory: Annotated[
                str,
                Field(min_length=1)
                ] = "data/output/search_results/SingleQuestions",
            convenio: str | None = None
            ) -> None:
        """Print the top-k sources for a single question, and save them.

        The output is a StudentSearchResults with one entry, the same
        format search_dataset writes, so both commands behave alike. The
        JSON goes to stdout (so it can be piped) and to a file named after
        the question id; where it was saved goes to stderr.

        Args:
            query: The question to search for.
            k: Number of sources to return.
            index_dir: Folder where the index command saved the index.
            retriever: ``hybrid`` reranks the BM25 results with the vector
                index (better recall); ``bm25`` uses the words only.
            cache: Reuse the results of questions already searched with
                the same index and retriever (``--cache=False`` to search
                again).
            save_directory: Folder where the StudentSearchResults file is
                written, as ``<question_id>.json``.
            convenio: Search this agreement and the Statute (``tic``), only
                the Statute (``estatuto``) or, by default, the whole
                corpus.
        """
        ConvenioSearch.check(convenio, index_dir)
        indexer = self._load_retriever(index_dir, retriever, cache)
        question = UnansweredQuestion(question=query)
        results = StudentSearchResults(
            search_results=[self._search_question(indexer, question, k,
                                                  convenio)],
            k=k,
        )
        indexer.flush()
        print(results.model_dump_json(indent=2))
        path = self._save_json(save_directory,
                               f"{question.question_id}.json", results)
        print(f"File saved in '{path}'", file=sys.stderr)

    @staticmethod
    def _build_retriever(index_dir: str,
                         retriever: RetrieverMode
                         ) -> Retriever:
        """Load the saved indexes the chosen retriever needs.

        Both retrievers have the same ``search(query, k)``, so the
        commands use either one without knowing which.

        Args:
            index_dir: Folder where the index command saved the indexes.
            retriever: ``bm25`` or ``hybrid``.

        Returns:
            The BM25 index, or a HybridRetriever over both indexes.

        Raises:
            IndexNotFoundError: An index file is missing.
            PathError: An index file cannot be read.
            DatasetError: The BM25 index file is not valid.
            VectorIndexError: The vector index is not valid or does not
                match the BM25 index.
        """
        # With the query cache this runs inside the progress bar loop
        # (first uncached question): tqdm.write keeps it on its own line.
        tqdm.write("Loading the index...", file=sys.stderr)
        bm25 = BM25Index.load(Path(index_dir))
        if retriever == "bm25":
            return bm25
        hybrid = HybridRetriever(bm25, VectorIndex.load(Path(index_dir)))
        if retriever == "hybrid":
            return hybrid
        return GeminiRetriever(bm25, hybrid,
                               RagCLI._embeddings_dir(index_dir))

    @staticmethod
    def _embeddings_dir(index_dir: str) -> Path:
        """Folder of the Gemini vectors, next to the index folder."""
        return Path(index_dir).parent / "embeddings"

    @staticmethod
    def _load_retriever(index_dir: str,
                        retriever: RetrieverMode,
                        cache: bool
                        ) -> Retriever:
        """Give the commands the chosen retriever, cached or not.

        With the cache, nothing is loaded here: the indexes are only
        loaded by the first question that is not already cached, so a
        command whose questions are all cached never loads them.

        Args:
            index_dir: Folder where the index command saved the indexes.
            retriever: ``bm25`` or ``hybrid``.
            cache: Wrap the retriever in a ``CachedRetriever`` that reads
                and writes ``index_dir/query_cache_<retriever>.json``.

        Returns:
            The retriever, or a ``CachedRetriever`` around it.

        Raises:
            IndexNotFoundError, PathError, DatasetError, VectorIndexError:
                Without the cache, as in ``_build_retriever``. With it,
                the same errors come from the first uncached search.
        """
        # gemini keeps its question vectors on disk, and a local answer
        # given during an outage must not be remembered as its result.
        if not cache or retriever == "gemini":
            return RagCLI._build_retriever(index_dir, retriever)
        return CachedRetriever(
            lambda: RagCLI._build_retriever(index_dir, retriever),
            RagCLI._cache_namespace(index_dir, retriever),
            CachedRetriever.file_path(Path(index_dir), retriever)
        )

    @staticmethod
    def _save_json(save_directory: str, file_name: str,
                   output: BaseModel) -> Path:
        """Write a command's output as JSON, creating the folder if needed.

        Shared by the four commands that save results, so the write and
        its error are handled in one place.

        Args:
            save_directory: Folder where the file is written.
            file_name: Name of the file inside that folder.
            output: The model to save (StudentSearchResults or
                StudentSearchResultsAndAnswer).

        Returns:
            The path of the written file.

        Raises:
            PathError: The folder cannot be created or the file written.
        """
        try:
            Path(save_directory).mkdir(parents=True, exist_ok=True)
            path = Path(save_directory) / file_name
            path.write_text(output.model_dump_json(indent=2),
                            encoding="utf-8")
        except OSError as e:
            raise PathError(save_directory, e) from e
        return path

    @staticmethod
    def _open_file_save(path: Path, model: type[M]) -> M:
        """Read a JSON file and validate it with a pydantic model.

        Shared by every command that reads a JSON input (datasets, search
        results), so the two possible failures are handled in one place.

        Args:
            path: The JSON file to read.
            model: The model the content must match (e.g. ``RagDataset``).

        Returns:
            The validated content, as an instance of ``model``.

        Raises:
            PathError: The file cannot be read, or is not UTF-8 text.
            DatasetError: The content is not valid JSON for ``model``.
        """
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise PathError(path, e) from e
        try:
            return model.model_validate_json(text)
        except ValidationError as e:
            raise DatasetError(str(path), e) from e

    @staticmethod
    def _search_question(indexer: Retriever,
                         question: UnansweredQuestion,
                         k: int,
                         convenio: str | None = None) -> MinimalSearchResults:
        """Search one question and keep only the source locations.

        Each hit becomes a MinimalSource (file path and character range):
        the chunk text and the score are not part of the output format.

        Args:
            indexer: A loaded retriever (see ``_load_retriever``).
            question: The question, with its id.
            k: Number of sources to retrieve.
            convenio: The legal texts to search (``ConvenioSearch.search``).

        Returns:
            The question with its ranked sources, best first.
        """
        sources = [
            MinimalSource(
                file_path=hit.chunk.file_path,
                first_character_index=hit.chunk.first_character_index,
                last_character_index=hit.chunk.last_character_index,
            )
            for hit in ConvenioSearch.search(indexer, question.question, k,
                                             convenio)
        ]
        return MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=sources,
        )

    @staticmethod
    def _load_chunks(sources: list[MinimalSource],
                     files: dict[str, str]) -> list[Chunk]:
        """Recover the text of each source, as the chunk the index stored.

        The search results only keep where each chunk is, not its text,
        and the generator needs the text. Each file is read once (like
        CorpusReader, as UTF-8, from the project root its paths are
        relative to) and kept in ``files``, so the questions
        that share a file do not read it again; the chunk is the slice
        ``[first_character_index:last_character_index]``, the same one
        the chunker cut.

        Args:
            sources: The retrieved sources of one question, best first.
            files: Cache ``file_path → text`` shared by every question;
                filled with the files read here.

        Returns:
            One Chunk per source, in the same order.

        Raises:
            CorpusError: A source path is outside the project root (an
                absolute path or ``..`` in the JSON).
            PathError: A source file cannot be read, or is not UTF-8 text.
        """
        chunks: list[Chunk] = []
        for source in sources:
            if source.file_path not in files:
                path = CorpusReader.project_path(source.file_path)
                try:
                    files[source.file_path] = path.read_text(
                        encoding="utf-8")
                except (OSError, UnicodeDecodeError) as e:
                    raise PathError(source.file_path, e) from e
            text = files[source.file_path]
            chunks.append(Chunk(
                file_path=source.file_path,
                first_character_index=source.first_character_index,
                last_character_index=source.last_character_index,
                text=text[source.first_character_index:
                          source.last_character_index],
            ))
        return chunks

    @CliCommand.validated
    def search_dataset(
            self,
            dataset_path: FilePath = DATASET_QUESTIONS,
            k: Annotated[int, Field(gt=0)] = 10,
            save_directory: Annotated[
                str,
                Field(min_length=1)
                ] = "data/output/search_results/UnansweredQuestions",
            index_dir: Annotated[str, Field(min_length=1)] = "data/processed",
            retriever: RetrieverMode = "hybrid",
            cache: bool = True
            ) -> None:
        """Search every question of a dataset and save the results as JSON.

        Args:
            dataset_path: JSON file with the questions (RagDataset).
                Default: the public docs dataset,
                ``data/datasets/UnansweredQuestions/dataset_docs_public.json``.
            k: Number of sources to return per question.
            save_directory: Folder where the StudentSearchResults file is
                written, with the same file name as the dataset.
            index_dir: Folder where the index command saved the index.
            retriever: ``hybrid`` reranks the BM25 results with the vector
                index (better recall); ``bm25`` uses the words only.
            cache: Reuse the results of questions already searched with
                the same index and retriever (``--cache=False`` to search
                again).
        """
        indexer = self._load_retriever(index_dir, retriever, cache)
        dataset = self._open_file_save(dataset_path, RagDataset)

        results_list: list[MinimalSearchResults] = []
        for question in tqdm(
                        dataset.rag_questions,
                        desc="Searching",
                        unit="question"
                        ):
            results_list.append(self._search_question(indexer, question, k))
        indexer.flush()
        output = StudentSearchResults(search_results=results_list, k=k)
        path = self._save_json(save_directory, dataset_path.name, output)
        print(f"File saved in '{path}'")

    @CliCommand.validated
    def answer(
            self,
            query: Annotated[str, Field(min_length=1)],
            k: Annotated[int, Field(gt=0)] = 10,
            index_dir: Annotated[str, Field(min_length=1)] = "data/processed",
            retriever: RetrieverMode = "hybrid",
            dtype: DtypeMode = "auto",
            cache: bool = True,
            save_directory: Annotated[
                str,
                Field(min_length=1)
                ] = "data/output/search_results_and_answer/SingleQuestions",
            convenio: str | None = None
            ) -> None:
        """Answer a single question using the retrieved sources as context.

        The StudentSearchResultsAndAnswer JSON goes to stdout (so it can be
        piped) and to a file named after the question id; where it was
        saved goes to stderr.

        Args:
            query: The question to answer.
            k: Number of sources to retrieve as context.
            index_dir: Folder where the index command saved the index.
            retriever: ``hybrid`` reranks the BM25 results with the vector
                index (better recall); ``bm25`` uses the words only.
            dtype: How to load the model. ``auto`` chooses float32 or
                bfloat16 from the available memory; ``float32`` or
                ``bfloat16`` force it, without the memory check (for
                testing).
            cache: Reuse the results of questions already searched with
                the same index and retriever, and the answers already
                generated with the same context, from
                ``index_dir/answer_cache.json`` (``--cache=False`` to
                search and generate again).
            save_directory: Folder where the StudentSearchResultsAndAnswer
                file is written, as ``<question_id>.json``.
            convenio: Search this agreement and the Statute (``tic``), only
                the Statute (``estatuto``) or, by default, the whole
                corpus.
        """

        # The index first: its errors show up before PyTorch is imported.
        ConvenioSearch.check(convenio, index_dir)
        indexer = self._load_retriever(index_dir, retriever, cache)
        hits = ConvenioSearch.search(indexer, query, k, convenio)
        indexer.flush()
        # No source found: nothing to ground an answer on, so neither
        # PyTorch nor the model is loaded (and nothing is cached).
        if hits:
            chunks = [hit.chunk for hit in hits]
            answers = (AnswerCache(AnswerCache.file_path(Path(index_dir)))
                       if cache else None)
            key = AnswerCache.key(query, chunks)
            cached = answers.get(key) if answers is not None else None
            if cached is not None:
                print("Answer read from the cache", file=sys.stderr)
                answer = cached
            else:
                generator = self._load_generator(dtype)
                # One model call: no progress bar, so say what is running.
                print("Generating the answer...", file=sys.stderr)
                answer = generator.generate(query, chunks)
                if answers is not None:
                    # Saved now: an error writing the output below must
                    # not lose an answer that took seconds to generate.
                    answers.put(key, answer)
                    answers.flush()
        else:
            answer = NO_SOURCES
        answered: MinimalAnswer = MinimalAnswer(
            question_id=str(uuid.uuid4()),
            question=query,
            retrieved_sources=[MinimalSource(
                file_path=hit.chunk.file_path,
                first_character_index=hit.chunk.first_character_index,
                last_character_index=hit.chunk.last_character_index
                ) for hit in hits],
            answer=answer
        )
        output = StudentSearchResultsAndAnswer(search_results=[answered],
                                               k=k)
        print(output.model_dump_json(indent=2))
        path = self._save_json(save_directory,
                               f"{answered.question_id}.json", output)
        print(f"File saved in '{path}'", file=sys.stderr)

    @staticmethod
    def _load_generator(dtype: DtypeMode) -> "Generator":
        """Import PyTorch and load the model, with a message for each.

        The import is inside the method, so the commands that never
        generate (and an answer read from the cache) never load PyTorch.

        Args:
            dtype: How to load the model, as in ``answer``.

        Returns:
            The loaded generator.

        Raises:
            RagError: PyTorch cannot be imported (not enough memory, or a
                broken installation), or the model cannot be loaded.
        """
        try:
            print("Loading PyTorch...", file=sys.stderr)
            from .generator import Generator
        except (ImportError, MemoryError, OSError) as e:
            raise RagError(
                f"Cannot load PyTorch ({type(e).__name__}: {e}). "
                "Not enough memory, or the installation is broken: "
                "run 'uv sync'.") from e
        print(f"Loading {Generator.MODEL_NAME}...", file=sys.stderr)
        return Generator(dtype)

    @CliCommand.validated
    def answer_dataset(
            self,
            student_search_results_path: FilePath = SEARCH_RESULTS,
            save_directory: Annotated[
                str,
                Field(min_length=1)
                ] = ("data/output/search_results_and_answer/"
                     "UnansweredQuestions"),
            dtype: DtypeMode = "auto",
            index_dir: Annotated[str, Field(min_length=1)] = "data/processed",
            cache: bool = True
            ) -> None:
        """Generate answers for the output of search_dataset.

        With the answer cache, the questions already answered with the
        same context are not generated again, and the model is only
        loaded if at least one question is not cached.

        Args:
            student_search_results_path: JSON file written by search_dataset
                (StudentSearchResults). Default: what search_dataset writes
                with its defaults, ``data/output/search_results/``
                ``UnansweredQuestions/dataset_docs_public.json``.
            save_directory: Folder where the StudentSearchResultsAndAnswer
                file is written, with the same file name as the input.
            dtype: How to load the model, as in ``answer``.
            index_dir: Folder of the answer cache
                (``index_dir/answer_cache.json``), the one shared with
                ``answer`` and ``serve``. Nothing else is read from it.
            cache: Reuse the answers already generated with the same
                question and context (``--cache=False`` to generate them
                all again).
        """

        # The input first: a broken file, or a source path outside the
        # project, fails before PyTorch is imported.
        results = self._open_file_save(student_search_results_path,
                                       StudentSearchResults)
        for entry in results.search_results:
            for source in entry.retrieved_sources:
                CorpusReader.project_path(source.file_path)

        results_list: list[MinimalAnswer] = []
        if results.search_results:
            answers = (AnswerCache(AnswerCache.file_path(Path(index_dir)))
                       if cache else None)
            results_list = self._answer_all(results, dtype, answers)
        output = StudentSearchResultsAndAnswer(
            search_results=results_list,
            k=results.k)
        path = self._save_json(save_directory,
                               student_search_results_path.name, output)
        print(f"File saved in '{path}'")

    def _answer_all(self, results: StudentSearchResults,
                    dtype: DtypeMode,
                    answers: AnswerCache | None) -> list[MinimalAnswer]:
        """Answer every question of ``results``, generating only what is
        not cached.

        Every context is rebuilt first, so the cached answers are found
        before anything is loaded: PyTorch and the model are loaded once,
        and only if a question with sources is not cached. A question
        without sources gets ``NO_SOURCES``, as in ``answer``. The cache
        is saved every ``FLUSH_EVERY`` generated answers and when the
        loop ends, even on Ctrl+C or an error, so an interrupted run
        keeps its work (a killed process keeps up to the last save).

        Args:
            results: The search results, with valid source paths.
            dtype: How to load the model, as in ``answer``.
            answers: The answer cache, or ``None`` to generate them all.

        Returns:
            One MinimalAnswer per question, in the same order.

        Raises:
            RagError: PyTorch or the model cannot be loaded.
        """
        files: dict[str, str] = {}
        questions = results.search_results
        contexts = [self._load_chunks(question.retrieved_sources, files)
                    for question in questions]
        keys = [AnswerCache.key(question.question, chunks)
                for question, chunks in zip(questions, contexts)]
        texts: list[str | None] = [
            NO_SOURCES if not chunks
            else answers.get(key) if answers is not None
            else None
            for chunks, key in zip(contexts, keys)]
        missing = [i for i, text in enumerate(texts) if text is None]
        if answers is not None:
            print(f"{len(questions) - len(missing)} of {len(questions)} "
                  "answers read from the cache", file=sys.stderr)

        if missing:
            generator = self._load_generator(dtype)
            try:
                for done, i in enumerate(tqdm(missing,
                                              desc="Generating answers",
                                              unit="question"), start=1):
                    answer = generator.generate(questions[i].question,
                                                contexts[i])
                    texts[i] = answer
                    if answers is not None:
                        answers.put(keys[i], answer)
                        if done % self.FLUSH_EVERY == 0:
                            answers.flush()
            finally:
                # Also on Ctrl+C or an error: every answer generated so
                # far is kept for the next run.
                if answers is not None:
                    answers.flush()

        # Every None was generated above: all the texts are set now.
        return [MinimalAnswer(question_id=question.question_id,
                              question=question.question,
                              retrieved_sources=question.retrieved_sources,
                              answer=cast(str, text))
                for question, text in zip(questions, texts)]

    @CliCommand.validated
    def evaluate(
            self,
            student_search_results_path: FilePath = SEARCH_RESULTS,
            dataset_path: FilePath = DATASET_ANSWERS,
            k: Annotated[int, Field(gt=0)] = 10,
            show_failed: bool = False
            ) -> None:
        """Print recall@k of search results against a ground-truth dataset.

        Same metric as the moulinette's evaluate_student_search_results;
        with show_failed it also lists, like its list_valid_questions, the
        questions whose sources were not found.

        Args:
            student_search_results_path: JSON file written by search_dataset.
                Default: what search_dataset writes with its defaults,
                ``data/output/search_results/UnansweredQuestions/``
                ``dataset_docs_public.json``.
            dataset_path: JSON file with the correct sources
                (AnsweredQuestions). Default: the public docs dataset,
                ``data/datasets/AnsweredQuestions/dataset_docs_public.json``.

            k: Largest k for which recall is reported.
            show_failed: List every question not found within the first k
                results, with its text.
        """

        dataset = self._open_file_save(
            dataset_path,
            RagDataset)
        retrived = self._open_file_save(
            student_search_results_path,
            StudentSearchResults)
        try:
            evaluation: EvaluationInform = Evaluator().evaluate(
                retrived, dataset, k
            )
        except DatasetEvaluationError as e:
            raise DatasetEvaluationError(
                f"dataset '{dataset_path}': {e.problem}"
            ) from e
        except ResultsEvaluationError as e:
            raise ResultsEvaluationError(
                f"results '{student_search_results_path}': {e.problem}"
            ) from e
        print("Student data is valid: True")
        print("Evaluation Results")
        print(self.SEPARATOR)
        for cutoff, value in evaluation.recall.items():
            print(f"Recall@{cutoff}: {value:.3f} ", end="")
        print()
        self._print_ids(
            f"Your results contain {len(evaluation.unknown)} questions "
            "that are not in the dataset (ignored):",
            evaluation.unknown,
        )
        self._print_ids(
            f"Your results are missing {len(evaluation.missing)} "
            "questions (counted as 0):",
            evaluation.missing,
        )
        total = len(dataset.rag_questions)
        print(self.SEPARATOR)
        print(f"Found within k={min(k, retrived.k)}: "
              f"{total - len(evaluation.failed)}/{total} questions")
        if show_failed:
            self._print_failed(evaluation.failed, dataset)
        elif evaluation.failed:
            print("Run with --show_failed to list the questions not found.")
        print(self.SEPARATOR)

    @classmethod
    def _print_failed(cls, failed: list[str], dataset: RagDataset) -> None:
        """Print every question that was not found, with its text.

        Args:
            failed: Ids of the questions not found, in dataset order.
            dataset: The dataset, to look up the text of each question.
        """
        if not failed:
            return
        texts = {q.question_id: q.question for q in dataset.rag_questions}
        print(cls.SEPARATOR)
        print(f"Questions not found ({len(failed)}):")
        for question_id in failed:
            print(f"  [FAILED] {question_id}  \"{texts[question_id]}\"")

    @classmethod
    def _print_ids(cls, title: str, ids: list[str]) -> None:
        """Print a section with some question ids, if there are any.

        At most ``MAX_IDS_SHOWN`` ids are listed; the rest are counted.

        Args:
            title: First line of the section.
            ids: Question ids to show; nothing is printed when empty.
        """
        if not ids:
            return
        print(cls.SEPARATOR)
        print(title)
        for question_id in ids[:cls.MAX_IDS_SHOWN]:
            print(f"  {question_id}")
        hidden = len(ids) - cls.MAX_IDS_SHOWN
        if hidden > 0:
            print(f"  ... and {hidden} more")

    @staticmethod
    def _cache_namespace(index_dir: str, retriever: RetrieverMode) -> str:
        """Describe what produces the search results, for the query cache.

        Cached results are only reused when this text is the same: the
        retriever, its parameters, and the size and modification time of
        the index files it reads (rebuilding the index changes them). The
        files are not read, only their metadata, so it is instant.

        Args:
            index_dir: Folder where the index command saved the indexes.
            retriever: ``bm25`` or ``hybrid``.

        Returns:
            The namespace, e.g. ``bm25|k1=1.9|b=0.5|index.json:...``. A
            missing file is written as ``<name>:missing`` instead of
            failing: loading the index reports it ("Run 'index' first").
        """
        parts = [retriever, f"k1={BM25Index.K1}", f"b={BM25Index.B}"]
        files = [BM25Index.FILE_NAME]
        if retriever == "hybrid":
            parts += [f"candidates={HybridRetriever.CANDIDATES}",
                      f"alpha={HybridRetriever.ALPHA}",
                      f"model={VectorIndex.MODEL_NAME}"]
            files.append(VectorIndex.VECTOR_NAME)

        for name in files:
            try:
                stat = (Path(index_dir) / name).stat()
            except OSError:
                parts.append(f"{name}:missing")
                continue
            parts.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
        return "|".join(parts)

    @CliCommand.validated
    def serve(self,
              host: Annotated[str, Field(min_length=1)] = "127.0.0.1",
              port: Annotated[int, Field(ge=1, le=65535)] = 8000,
              index_dir: Annotated[
                  str,
                  Field(min_length=1)] = "data/processed",
              retriever: RetrieverMode = "hybrid",
              dtype: DtypeMode = "auto",
              cache: bool = True,
              raw_dir: Annotated[str, Field(min_length=1)] = "data/raw"
              ) -> None:
        """Start the local HTTP API (bonus 5) until Ctrl+C.

        Endpoints: ``/health``, ``/search`` and ``/answer``, with the same
        JSON as the commands of the same name (nothing is saved to
        ``data/output``), and ``POST /index``, which re-indexes the
        changed files of raw_dir and reloads the index; ``/docs`` is an
        interactive page to try them. The index is loaded now; the model
        on the first ``/answer``, then kept.

        Args:
            host: Address to listen on. The default only accepts
                connections from this machine.
            port: Port to listen on.
            index_dir: Folder where the index command saved the index.
            retriever: The default of every request: ``hybrid`` reranks
                the BM25 results with the vector index (better recall);
                ``bm25`` uses the words only. A request can choose the
                other one with ``?retriever=``; both share one BM25
                index in memory.
            dtype: How to load the model. ``auto`` chooses float32 or
                bfloat16 from the available memory; ``float32`` or
                ``bfloat16`` force it, without the memory check (for
                testing).
            cache: Reuse the results of questions already searched with
                the same index and retriever, and the answers already
                generated with the same context, from
                ``index_dir/answer_cache.json`` (``--cache=False`` to
                search and generate again).
            raw_dir: Corpus folder that ``POST /index`` reads.
        """
        import uvicorn
        from .api import RagApi, RetrieverLoader

        answers = (AnswerCache(AnswerCache.file_path(Path(index_dir)))
                   if cache else None)
        loader = RetrieverLoader(index_dir, cache)
        api = RagApi(loader.load(retriever), dtype, answers, index_dir,
                     raw_dir, loader, retriever)
        print(f"http://{host}:{port}/docs", file=sys.stderr)
        uvicorn.run(api.app, host=host, port=port)
