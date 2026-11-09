"""Query cache (bonus 4): repeated searches without searching again.

``CachedRetriever`` wraps another retriever (BM25 or hybrid) without
changing it. Results are kept by question and ``k``, saved to a JSON file
between commands, and the wrapped retriever is only loaded when a question
is not cached, so a fully cached command never loads the indexes.
"""
import os
import sys
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from .base_retriever import Retriever
from .models import SearchHit, QueryCache


class CachedRetriever(Retriever):
    """A retriever that remembers the results of another one.

    The cache is a help, never a requirement: a missing, broken or foreign
    cache file means an empty cache, and a file that cannot be written
    only prints a warning.

    Attributes:
        FILE_NAME: Name of the cache file inside the index folder;
            ``{name}`` is the retriever, so each one keeps its own file
            (one shared file would be overwritten at every switch).
    """

    FILE_NAME = "query_cache_{name}.json"

    @classmethod
    def file_path(cls, index_dir: Path, name: str) -> Path:
        """Where the cache of one retriever is saved.

        Args:
            index_dir: Folder of the index the results come from.
            name: The retriever (``bm25`` or ``hybrid``).

        Returns:
            ``index_dir/query_cache_<name>.json``.
        """
        return index_dir / cls.FILE_NAME.format(name=name)

    def __init__(self, load_inner: Callable[[], Retriever],
                 namespace: str, cache_path: Path) -> None:
        """Read the saved cache; the wrapped retriever is not loaded yet.

        Args:
            load_inner: Builds the wrapped retriever. Called once, by the
                first search that is not cached.
            namespace: What produces the results (retriever, parameters,
                index files). Saved results with another namespace are
                ignored.
            cache_path: The JSON file the cache is read from and written
                to.
        """
        self._load_inner = load_inner
        self._inner: Retriever | None = None
        self._cache_path = cache_path
        self._namespace = namespace
        self._entries: dict[str, list[SearchHit]] = self._read()
        self._changed = False

    def _read(self) -> dict[str, list[SearchHit]]:
        """Load the saved results, if they can be reused.

        Returns:
            The saved entries, or an empty dict when the file is missing,
            unreadable, not valid, or from another namespace.
        """
        try:
            text = self._cache_path.read_text(encoding="utf-8")
            data = QueryCache.model_validate_json(text)
        except (OSError, ValidationError, UnicodeDecodeError):
            return {}
        if data.namespace != self._namespace:
            return {}
        return data.entries

    def _retriever(self) -> Retriever:
        """Return the wrapped retriever, loading it the first time.

        Returns:
            The retriever built by ``load_inner``.
        """
        if self._inner is None:
            self._inner = self._load_inner()
        return self._inner

    @staticmethod
    def _key(query: str, k: int,
             sources: frozenset[str] | None = None) -> str:
        """Key of a search in the cache: the same query with another
        ``k`` or other ``sources`` is another entry.

        Args:
            query: The question, as given.
            k: Number of results requested.
            sources: The files searched; ``None`` for all of them.

        Returns:
            ``"<k>:<query>"``, or ``"<k>:<sources>:<query>"`` with the
            sources sorted and joined by commas.
        """
        if sources is None:
            return f"{k}:{query}"
        return f"{k}:{','.join(sorted(sources))}:{query}"

    def search(self, query: str, k: int,
               sources: frozenset[str] | None = None) -> list[SearchHit]:
        """Return the cached hits, or search and remember them.

        Args:
            query: Free text.
            k: Maximum number of results.
            sources: Names of the files to search, without folder or
                extension; ``None`` searches them all.

        Returns:
            Up to ``k`` hits, best first. A new list each time, so the
            caller cannot change what is cached.

        Raises:
            RagError: The wrapped retriever cannot be loaded (for example,
                "Run 'index' first"), on the first uncached search.
        """
        key = self._key(query, k, sources)
        if key not in self._entries:
            self._entries[key] = self._retriever().search(query, k, sources)
            self._changed = True
        return list(self._entries[key])

    def flush(self) -> None:
        """Save the cache if a search added something.

        The file is written to a ``.tmp`` file first and then moved over
        the old one (``os.replace``), so an interrupted write never leaves
        a half-written cache. If it cannot be written, a warning is
        printed and the command goes on.
        """
        if not self._changed:
            return
        data = QueryCache(namespace=self._namespace,
                          entries=self._entries)
        tmp = self._cache_path.with_suffix(".tmp")

        try:
            tmp.write_text(data.model_dump_json(), encoding="utf-8")
            os.replace(tmp, self._cache_path)
            self._changed = False
        except OSError as e:
            print("WARNING: cannot save the query cache in "
                  f"'{self._cache_path}': {e}", file=sys.stderr)
            return
