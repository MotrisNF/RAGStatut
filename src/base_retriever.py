"""The common interface of every retriever: ``search`` and ``flush``."""
from abc import ABC, abstractmethod
from .models import SearchHit


class Retriever(ABC):
    """Anything that returns the best chunks for a question."""

    @abstractmethod
    def search(self, query: str, k: int,
               sources: frozenset[str] | None = None) -> list[SearchHit]:
        """Return up to ``k`` hits, best first.

        Args:
            query: Free text.
            k: Maximum number of results.
            sources: Names of the files to search, without folder or
                extension (``convenio_tic``); ``None`` searches them all.

        Returns:
            The hits, only from ``sources`` when it is given.
        """

    def flush(self) -> None:
        """Write any pending state to disk.

        Nothing to do by default; ``CachedRetriever`` saves its query
        cache. Callers can always call it without knowing which retriever
        they have.
        """
