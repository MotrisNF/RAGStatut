"""Search inside one collective agreement and the Workers' Statute.

Every corpus file is one legal text, so the name of the file tells which
one a chunk belongs to (``convenio_tic.txt`` is the agreement ``tic``).
The agreement and the Statute are searched apart and each one gets its own
share of the results, so an answer can always show both.
"""
from math import ceil
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from .base_retriever import Retriever
from .exceptions import UsageError
from .models import IndexManifest, SearchHit


class ConvenioSearch:
    """Choose the legal texts a question is searched in.

    Attributes:
        STATUTE: Value of ``convenio`` that searches only the Statute.
        STATUTE_SOURCE: Name, without extension, of the Statute's file.
        PREFIX: Start of the name of every agreement's file.
        MANIFEST_NAME: File of the index folder that lists the corpus.
        CONVENIO_SHARE: Part of the ``k`` results kept for the agreement;
            the rest are for the Statute.
    """

    STATUTE: str = "estatuto"
    STATUTE_SOURCE: str = "estatuto_trabajadores"
    PREFIX: str = "convenio_"
    MANIFEST_NAME: str = "manifest.json"
    CONVENIO_SHARE: float = 0.7

    @classmethod
    def available(cls, index_dir: str) -> list[str] | None:
        """The values ``convenio`` can take for an index.

        Args:
            index_dir: Folder where the index command saved the index.

        Returns:
            The agreements of the corpus and ``STATUTE``, sorted, or
            ``None`` when the index has no readable manifest (searching
            will then report the missing index).
        """
        try:
            text = (Path(index_dir) / cls.MANIFEST_NAME).read_text(
                encoding="utf-8")
            manifest = IndexManifest.model_validate_json(text)
        except (OSError, UnicodeDecodeError, ValidationError):
            return None
        stems = [PurePosixPath(path).stem for path in manifest.files]
        names = [stem.removeprefix(cls.PREFIX) for stem in stems
                 if stem.startswith(cls.PREFIX)]
        if cls.STATUTE_SOURCE in stems:
            names.append(cls.STATUTE)
        return sorted(names)

    @classmethod
    def check(cls, convenio: str | None, index_dir: str) -> None:
        """Refuse a ``convenio`` that the index does not contain.

        Args:
            convenio: The value given, or ``None`` for no filter.
            index_dir: Folder where the index command saved the index.

        Raises:
            UsageError: The agreement is not in the corpus.
        """
        names = cls.available(index_dir)
        if convenio is None or names is None or convenio in names:
            return
        raise UsageError(f"Unknown convenio '{convenio}'. "
                         f"Available: {', '.join(names)}")

    @classmethod
    def search(cls, retriever: Retriever, query: str, k: int,
               convenio: str | None) -> list[SearchHit]:
        """Return the best chunks of the chosen legal texts.

        The agreement and the Statute are searched separately, because
        their scores cannot be compared. The agreement gets
        ``CONVENIO_SHARE`` of ``k`` and the Statute the rest; when one
        has too few results, the other fills the free places.

        Args:
            retriever: A loaded retriever.
            query: The question.
            k: Maximum number of results.
            convenio: An agreement, ``STATUTE`` for only the Statute, or
                ``None`` to search the whole corpus.

        Returns:
            Up to ``k`` hits: the agreement's first, then the Statute's,
            each group best first.
        """
        if convenio is None:
            return retriever.search(query, k)
        statute = frozenset({cls.STATUTE_SOURCE})
        if convenio == cls.STATUTE:
            return retriever.search(query, k, statute)

        own = retriever.search(query, k,
                               frozenset({cls.PREFIX + convenio}))
        law = retriever.search(query, k, statute)
        own_count = min(ceil(k * cls.CONVENIO_SHARE), len(own))
        law_count = min(k - own_count, len(law))
        own_count = min(k - law_count, len(own))
        return own[:own_count] + law[:law_count]
