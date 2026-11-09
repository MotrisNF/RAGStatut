"""Semantic index over the corpus chunks (bonus 1).

Each chunk is split into windows of about ``MAX_WINDOW_SIZE`` characters
and every window becomes a vector with a small static embedding model.
A chunk scores as its best window, so a long chunk is not diluted by the
average of unrelated parts. The index is built by the ``index`` command
next to the BM25 index and saved as two ``.npy`` files, plus the model
itself (``StaticEncoder``), so a search needs nothing else.
"""
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from .exceptions import (IndexNotFoundError, PathError,
                         VectorIndexError)
from .models import Chunk, IndexPlan, WindowChunk
from .static_encoder import StaticEncoder
from .tokenizer_bm import Tokenizer


class VectorIndex:
    """Scores every chunk by meaning, with one vector per window.

    Always created complete: from chunks with ``build`` or from disk with
    ``load``; both end in ``__init__`` with the data already computed.

    Attributes:
        MAX_WINDOW_SIZE: Characters per window; a window is closed when
            the next line would go over it (see the README tuning table).
        MODEL_NAME: Embedding model, the same for chunks and questions.
        BATCH_SIZE: Windows embedded at a time by ``build`` (one step of
            the progress bar).
        VECTOR_NAME: File with the window vectors inside the index folder.
        OWNERS_NAME: File with the chunk id of each window.
    """

    MAX_WINDOW_SIZE = 400
    MODEL_NAME = "minishlab/potion-base-8M"
    BATCH_SIZE = 512

    VECTOR_NAME = "vectors.npy"
    OWNERS_NAME = "owners.npy"

    def __init__(self, vectors: NDArray[np.float32],
                 owners: NDArray[np.intp]) -> None:
        """Store an already computed index.

        Use ``build`` or ``load`` instead of calling this directly.

        Args:
            vectors: One normalised vector per window, shape
                ``(windows, dimensions)``.
            owners: Chunk id of each window, shape ``(windows,)``; a chunk
                id is the chunk's position in the list given to ``build``.
        """
        self._vectors = vectors
        self._owners = owners
        self._model: StaticEncoder | None = None

    @property
    def chunk_count(self) -> int:
        """Number of chunks the windows belong to (highest owner + 1)."""
        return int(self._owners.max()) + 1

    @classmethod
    def _load_model(cls) -> StaticEncoder:
        """Load the embedding model from the Hugging Face cache.

        Only ``build`` needs it: ``load`` reads the copy saved with the
        index. The tests replace this method with a fake model.

        Returns:
            The model (see ``StaticEncoder.from_pretrained``).

        Raises:
            RagError: The model is not cached and cannot be downloaded.
        """
        return StaticEncoder.from_pretrained(cls.MODEL_NAME)

    def _encoder(self) -> StaticEncoder:
        """The embedding model, loaded the first time it is needed.

        ``build`` and ``load`` already set it; this is only a fallback for
        an index created directly with ``__init__``.

        Returns:
            The loaded model.
        """
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @classmethod
    def _get_window(cls, text: str) -> list[str]:
        """Split a text into windows made of whole lines.

        Lines are added to the current window until the next one would
        make it longer than ``MAX_WINDOW_SIZE``. A single line longer than
        that becomes a window of its own, so joining the windows gives
        back the original text.

        Args:
            text: The text of one chunk.

        Returns:
            The windows in order; empty for an empty text.
        """
        window_list: list[str] = []
        window_text: str = ""
        for line in text.splitlines(keepends=True):
            if (window_text
                    and len(window_text) + len(line) > cls.MAX_WINDOW_SIZE):
                window_list.append(window_text)
                window_text = line
                continue
            window_text += line
        if window_text != "":
            window_list.append(window_text)
        return window_list

    @classmethod
    def _window_chunk(cls, chunks: list[Chunk]) -> list[WindowChunk]:
        """Build the texts to embed: the windows of every chunk.

        Each window starts with the words of its file path (tokenized like
        BM25 does), which helps code questions that name a module, and
        with the title of its article, if it has one.

        Args:
            chunks: The chunks to index.

        Returns:
            One ``WindowChunk`` per chunk, in the same order, so its
            position is the chunk id.
        """
        tokenizer = Tokenizer()
        window_chunk_list: list[WindowChunk] = []
        for chunk in chunks:
            prefix = " ".join(tokenizer.tokenize(chunk.file_path)) + "\n"
            if chunk.title:
                prefix += chunk.title + "\n"
            window_text = [prefix + window
                           for window in cls._get_window(chunk.text)]
            window_chunk_list.append(
                WindowChunk(
                    windows=window_text,
                    windows_count=len(window_text)
                )
            )
        return window_chunk_list

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "VectorIndex":
        """Build the index by embedding every window once.

        Windows are embedded ``BATCH_SIZE`` at a time, with a progress bar.
        The model stays loaded in the new index, so a search right after
        ``build`` does not load it again, and ``save`` writes it too.

        Args:
            chunks: The chunks to index, in the same order as the BM25
                index (chunk ids must match).

        Returns:
            The new index, ready to search.

        Raises:
            RagError: The embedding model cannot be loaded.
        """
        model = cls._load_model()
        vectors, owners = cls._embed(model, chunks, list(range(len(chunks))))
        index = cls(vectors, owners)
        index._model = model
        return index

    def update(self, plan: IndexPlan) -> "VectorIndex":
        """Build the index of ``plan`` reusing this one (bonus 3).

        The windows of kept chunks keep their vectors and take the new id
        of their chunk; those of removed chunks leave. Only the windows of
        ``plan.fresh_ids`` are embedded. The rows are then sorted by chunk
        id with a stable sort, which keeps the windows of a chunk in order,
        so the result equals ``build(plan.chunks)``.

        Args:
            plan: The chunks of the updated index and ``{old id: new id}``
                of the kept ones (``IncrementalIndexer._plan``).

        Returns:
            A new index with this index's model; this one is not changed.

        Raises:
            RagError: The embedding model cannot be loaded (only if this
                index has none and there are fresh chunks).
        """
        moved = np.array([plan.remap.get(int(owner), -1)
                          for owner in self._owners], dtype=np.intp)
        keep = moved >= 0
        vectors = self._vectors[keep]
        owners = moved[keep]

        if plan.fresh_ids:
            fresh = [plan.chunks[index] for index in plan.fresh_ids]
            new_vectors, new_owners = self._embed(
                self._encoder(),
                fresh,
                plan.fresh_ids)
            vectors = np.concatenate([vectors, new_vectors])
            owners = np.concatenate([owners, new_owners])

        order = np.argsort(owners, kind="stable")
        index = VectorIndex(vectors[order], owners[order])
        index._model = self._model

        return index

    @classmethod
    def _embed(cls, model: StaticEncoder, chunks: list[Chunk],
               ids: list[int]) -> tuple[NDArray[np.float32], NDArray[np.intp]]:
        """Embed the windows of some chunks, ``BATCH_SIZE`` at a time.

        Shared by ``build`` and ``update``, so a chunk gets the same
        windows and vectors in a full and in an incremental index.

        Args:
            model: The embedding model.
            chunks: The chunks to embed.
            ids: The chunk id of each chunk (``ids[i]`` for ``chunks[i]``):
                ``0..n-1`` in ``build``, the fresh ids in ``update``.

        Returns:
            One vector per window and the chunk id of each window, with
            the windows of each chunk together and in order.
        """
        window_chunks = cls._window_chunk(chunks)
        counts = [item.windows_count for item in window_chunks]
        owners = np.repeat(np.array(ids), counts)
        texts = [window for item in window_chunks for window in item.windows]
        batches = [texts[i:i + cls.BATCH_SIZE]
                   for i in range(0, len(texts), cls.BATCH_SIZE)]
        vectors = np.concatenate(
            [model.encode(batch) for batch in tqdm(
                batches, desc="Embedding windows", unit="batch")]
            or [np.zeros((0, 0), dtype=np.float32)])
        return vectors, owners

    @classmethod
    def load(cls, files_dir: Path) -> "VectorIndex":
        """Read an index saved by ``save``, without recomputing anything.

        ``.npy`` files keep their shape and type, so there is no pydantic
        validation; the shapes are checked by hand instead.

        Args:
            files_dir: Folder given to ``save``.

        Returns:
            The same index that was saved, with the model saved next to
            it (no Hugging Face cache, no network, no PyTorch).

        Raises:
            IndexNotFoundError: A vector or model file is missing
                (``index`` was never run for this folder, or with an older
                version).
            PathError: A file exists but cannot be read.
            VectorIndexError: A file is not valid or the arrays do not
                fit together.
        """
        try:
            vectors = np.load(files_dir / cls.VECTOR_NAME)
            owners = np.load(files_dir / cls.OWNERS_NAME)
        except FileNotFoundError as e:
            raise IndexNotFoundError(files_dir) from e
        except OSError as e:
            raise PathError(files_dir, e) from e
        except ValueError as e:
            raise VectorIndexError("cannot read the vector files. Run 'index' "
                                   "again") from e

        if (owners.ndim != 1
            or vectors.ndim != 2
                or len(vectors) != len(owners)):
            raise VectorIndexError("the data are corrupted. Run 'index' "
                                   "again")

        index = cls(vectors, owners)
        index._model = StaticEncoder.load(files_dir)
        return index

    def save(self, files_dir: Path) -> None:
        """Write the index as ``VECTOR_NAME`` and ``OWNERS_NAME``, and the
        model next to them (``StaticEncoder.save``).

        The folder is created if needed and a previous index is replaced.

        Args:
            files_dir: Folder where the index is saved (the same as the
                BM25 index).

        Raises:
            PathError: The folder or a file cannot be written.
            RagError: The tokenizer cannot be written.
        """
        try:
            files_dir.mkdir(parents=True, exist_ok=True)

            vectors_path = files_dir / self.VECTOR_NAME
            owners_path = files_dir / self.OWNERS_NAME

            np.save(vectors_path, self._vectors)
            np.save(owners_path, self._owners)
            self._encoder().save(files_dir)
        except OSError as e:
            raise PathError(files_dir, e) from e

    def chunk_score(self, query: str) -> NDArray[np.float64]:
        """Score every chunk by its similarity to the query.

        The query is embedded as it is (removing stopwords made results
        worse). Vectors are normalised, so the dot product is the cosine
        similarity; each chunk keeps the score of its best window.

        Args:
            query: Free text.

        Returns:
            One score per chunk id, shape ``(chunk_count,)``, from -1 to 1;
            higher is better.

        Raises:
            RagError: The embedding model cannot be loaded.
        """
        query_vector = self._encoder().encode([query])[0]
        scores = self._vectors @ query_vector
        best = np.full(self.chunk_count, -1.0)
        np.maximum.at(best, self._owners, scores)
        return best
