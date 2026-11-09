"""BM25 search index over the corpus chunks.

The index is built once by the ``index`` command, saved to disk and loaded
by the search commands, so the corpus is never re-read to answer a query.
"""
from collections import Counter
import heapq
from math import log
from pathlib import Path, PurePosixPath

from pydantic import ValidationError
from tqdm import tqdm

from .base_retriever import Retriever
from .exceptions import (DatasetError, IndexBuildError, IndexNotFoundError,
                         PathError)
from .models import Chunk, IndexData, IndexPlan, SearchHit
from .tokenizer_bm import Tokenizer


class BM25Index(Retriever):
    """Lexical search over chunks with the BM25 ranking function.

    Always created complete: from chunks with ``build`` or from disk with
    ``load``; both end in ``__init__`` with the data already computed.

    Attributes:
        K1: Term-frequency saturation. Higher values let repeated terms
            count for more.
        B: Length normalisation, from 0 (ignore chunk length) to 1 (fully
            relative to the average length).
        FILE_NAME: Name of the index file inside the index folder.
    """

    K1: float = 1.9
    B: float = 0.5
    FILE_NAME: str = "index.json"

    def __init__(self,
                 chunks: list[Chunk],
                 postings: dict[str, dict[int, int]],
                 lengths: list[int]
                 ) -> None:
        """Store an already computed index.

        Use ``build`` or ``load`` instead of calling this directly.

        Args:
            chunks: The indexed chunks; a chunk id is its position here.
            postings: Inverted index, ``term → {chunk id: occurrences}``.
            lengths: Number of tokens of each chunk, by chunk id. Their sum
                must be positive (``build`` guarantees it).
        """
        self._tokenizer = Tokenizer()
        self._chunks = chunks
        self._postings = postings
        self._lengths = lengths
        self._avg_length: float = sum(self._lengths) / len(self._lengths)
        self._n_chunks: int = len(self._chunks)
        # File of each chunk, without folder or extension, by chunk id.
        self._sources: list[str] = [PurePosixPath(chunk.file_path).stem
                                    for chunk in self._chunks]

    @property
    def chunk_count(self) -> int:
        """Number of indexed chunks (chunk ids go from 0 to this - 1)."""
        return self._n_chunks

    @property
    def chunks(self) -> list[Chunk]:
        """The indexed chunks, by chunk id (a copy: changing it does not
        change the index)."""
        return list(self._chunks)

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "BM25Index":
        """Build the index by tokenizing every chunk once.

        Each chunk's tokens are counted, added to the postings and then
        discarded: only the counts and the lengths are kept.

        Args:
            chunks: The chunks to index, in their final order.

        Returns:
            A complete index, ready to search or save.

        Raises:
            IndexBuildError: There is nothing to index (no chunks, or no
                chunk with a single token).
        """
        tokenizer = Tokenizer()
        posting: dict[str, dict[int, int]] = {}
        lengths: list[int] = []

        progress = tqdm(chunks, desc="Tokenizing chunks", unit="chunk")
        for chunk_id, chunk in enumerate(progress):
            tokens = cls._chunk_tokens(tokenizer, chunk)
            lengths.append(len(tokens))
            for term, count in Counter(tokens).items():
                posting.setdefault(term, {})[chunk_id] = count

        if sum(lengths) == 0:
            raise IndexBuildError(
                "Nothing to index: the corpus has no searchable text "
                f"({len(chunks)} chunks, 0 tokens)"
            )

        return cls(chunks, posting, lengths)

    def update(self, plan: IndexPlan) -> "BM25Index":
        """Build the index of ``plan`` reusing this one (bonus 3).

        Kept chunks move their lengths and postings to their new id
        without tokenizing again; removed chunks leave, and so does a term
        left without chunks. Only ``plan.fresh_ids`` are tokenized. IDF
        and the average length are recomputed by ``__init__``, so the
        result equals ``build(plan.chunks)``.

        Args:
            plan: The chunks of the updated index and ``{old id: new id}``
                of the kept ones (``IncrementalIndexer._plan``).

        Returns:
            A new index; this one is not changed.

        Raises:
            IndexBuildError: There is nothing to index, as in ``build``.
        """
        lengths = [0] * len(plan.chunks)
        for old_id, new_id in plan.remap.items():
            lengths[new_id] = self._lengths[old_id]

        postings: dict[str, dict[int, int]] = {}
        for term, counts in self._postings.items():
            moved = {plan.remap[cid]: n
                     for cid, n in counts.items()
                     if cid in plan.remap}

            if moved:
                postings[term] = moved

        # No bar when only deletions changed the index: nothing to tokenize.
        for chunk_id in tqdm(plan.fresh_ids,
                             desc="Tokenizing chunks",
                             unit="chunk",
                             disable=not plan.fresh_ids):
            tokens = self._chunk_tokens(self._tokenizer,
                                        plan.chunks[chunk_id])
            lengths[chunk_id] = len(tokens)
            for term, count in Counter(tokens).items():
                postings.setdefault(term, {})[chunk_id] = count

        if sum(lengths) == 0:
            raise IndexBuildError(
                "Nothing to index: the corpus has no searchable text "
                f"({len(plan.chunks)} chunks, 0 tokens)"
            )

        return BM25Index(plan.chunks, postings, lengths)

    @staticmethod
    def _chunk_tokens(tokenizer: Tokenizer, chunk: Chunk) -> list[str]:
        """The terms a chunk is indexed with: text, file path and title.

        Shared by ``build`` and ``update``, so a chunk gets the same terms
        in a full and in an incremental index.

        Args:
            tokenizer: The tokenizer of the index.
            chunk: The chunk to tokenize.

        Returns:
            The terms of the text, the path and the title, with
            repeats (``Counter`` turns them into occurrences).
        """
        return (tokenizer.tokenize(chunk.text)
                + tokenizer.tokenize(chunk.file_path)
                + tokenizer.tokenize(chunk.title))

    def _idf(self, term: str) -> float:
        """Rarity of a term: ``ln((N - df + 0.5) / (df + 0.5) + 1)``.

        ``N`` is the number of chunks and ``df`` the number of chunks that
        contain the term. Rare terms get high values; a term present in
        every chunk gets almost 0, never a negative value.

        Args:
            term: A term that IS in the postings (``search`` checks it).

        Returns:
            The inverse document frequency of the term.
        """
        df: int = len(self._postings[term])
        return log((self._n_chunks - df + 0.5) / (df + 0.5) + 1)

    def _term_score(self, idf: float, tf: int, chunk_id: int) -> float:
        """BM25 contribution of one query term to one chunk.

        ``idf × tf × (K1 + 1) / (tf + K1 × (1 − B + B × length / avg))``

        - ``idf`` rewards rare terms.
        - The ``tf`` fraction grows with each occurrence but saturates:
          it never exceeds ``K1 + 1`` however often the term repeats.
        - ``length / avg`` lowers the score of long chunks, where a term
          is more likely to appear by chance (``B`` sets how much).

        Args:
            idf: Rarity of the term, from ``_idf``.
            tf: Occurrences of the term in the chunk.
            chunk_id: Position of the chunk in ``self._chunks``.

        Returns:
            The amount to add to the chunk's score.
        """
        length_ratio = self._lengths[chunk_id] / self._avg_length
        normaliser = self.K1 * (1 - self.B + self.B * length_ratio)
        return idf * tf * (self.K1 + 1) / (tf + normaliser)

    def search(self, query: str, k: int,
               sources: frozenset[str] | None = None) -> list[SearchHit]:
        """Return the ``k`` chunks that best match the query.

        Each distinct query term adds its ``_term_score`` to every chunk
        that contains it; chunks with no query term are never scored.
        Ties are broken by chunk id, so results never depend on the order
        in which the query terms are visited. Chunks outside ``sources``
        are skipped before choosing the best ``k``, so all the places go
        to the chosen files.

        Args:
            query: Free text; it goes through the same tokenizer as the
                chunks.
            k: Maximum number of results.
            sources: Names of the files to search, without folder or
                extension; ``None`` searches them all.

        Returns:
            Up to ``k`` hits, best first, each with its ``chunk_id``.
            Empty when no term matches.
        """
        tokenized_query = set(self._tokenizer.tokenize_query(query))

        scores: dict[int, float] = {}
        for term in tokenized_query:
            if term not in self._postings:
                continue
            idf = self._idf(term)

            for chunk_id, tf in self._postings[term].items():
                if (sources is not None
                        and self._sources[chunk_id] not in sources):
                    continue
                scores[chunk_id] = (scores.get(chunk_id, 0.0)
                                    + self._term_score(idf, tf, chunk_id))

        ranked = heapq.nsmallest(
            k, scores.items(), key=lambda item: (-item[1], item[0])
        )

        hits: list[SearchHit] = []

        for chunk_id, score in ranked:
            hits.append(SearchHit(
                chunk=self._chunks[chunk_id],
                score=score,
                chunk_id=chunk_id))

        return hits

    def save(self, index_dir: Path) -> None:
        """Write the index as JSON into ``index_dir / FILE_NAME``.

        The folder is created if needed and a previous index is replaced.

        Args:
            index_dir: Folder where the index is saved.

        Raises:
            PathError: The folder or the file cannot be written.
        """
        try:
            index_dir.mkdir(parents=True, exist_ok=True)

            index = IndexData(
                chunks=self._chunks,
                postings=self._postings,
                lengths=self._lengths)
            path = index_dir / self.FILE_NAME
            path.write_text(index.model_dump_json(), encoding="utf-8")
        except OSError as e:
            raise PathError(index_dir, e) from e

    @classmethod
    def load(cls, index_dir: Path) -> "BM25Index":
        """Read an index saved by ``save``, without recomputing anything.

        Args:
            index_dir: Folder given to ``save``.

        Returns:
            The same index that was saved, ready to search.

        Raises:
            IndexNotFoundError: There is no index file (``index`` was
                never run for this folder).
            PathError: The file exists but cannot be read.
            DatasetError: The file is not a valid index.
        """
        path = index_dir / cls.FILE_NAME
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as e:
            raise IndexNotFoundError(index_dir) from e
        except OSError as e:
            raise PathError(path, e) from e
        try:
            data = IndexData.model_validate_json(text)
        except ValidationError as e:
            raise DatasetError(str(path), e) from e

        return cls(data.chunks, data.postings, data.lengths)
