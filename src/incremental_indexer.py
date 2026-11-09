"""Incremental indexing (bonus 3): re-index only the files that changed.

The saved ``manifest.json`` records the stamp (modification time and
size) of every corpus file. The next ``index`` compares it with the
current files and only reads, chunks, tokenizes and embeds the new and
modified ones; the chunks of modified and deleted files leave both
indexes. Without a usable manifest or index, or with ``--full``, the
whole corpus is indexed again.
"""
import sys
from pathlib import Path
from typing import get_args

from pydantic import ValidationError

from .answer_cache import AnswerCache
from .bm25_index import BM25Index
from .cached_retriever import CachedRetriever
from .chunker import ArticleChunker, PyChunker
from .exceptions import PathError, RagError
from .models import (Chunk, CorpusChanges, Document, FileStamp,
                     GeneratorMode, IndexManifest, IndexPlan, IndexReport,
                     RetrieverMode)
from .reader import CorpusReader
from .vector_index import VectorIndex


class IncrementalIndexer:
    """Bridge between the corpus reader and the two indexes.

    Decides what changed since the saved index and updates the indexes
    with only those files. A full index is the same path with no saved
    index: every file is new.

    Attributes:
        MANIFEST_NAME: File with the corpus stamps inside the index
            folder.
    """

    MANIFEST_NAME = "manifest.json"

    def __init__(self, raw_dir: str, index_dir: str,
                 max_chunk_size: int) -> None:
        """Prepare an ``index`` run; nothing is read yet.

        Args:
            raw_dir: Folder with the corpus, as given to ``index``; it is
                saved in the manifest and compared as written.
            index_dir: Folder where the indexes and the manifest are.
            max_chunk_size: Maximum length of a chunk, in characters.
        """
        self._raw_dir = raw_dir
        self._index_dir = Path(index_dir)
        self._max_chunk_size = max_chunk_size
        self._reader = CorpusReader(raw_dir)

    @property
    def _manifest_path(self) -> Path:
        """Where the manifest is saved: ``index_dir/manifest.json``."""
        return self._index_dir / self.MANIFEST_NAME

    def _load_manifest(self) -> IndexManifest | None:
        """The saved manifest, if the saved index can be updated.

        Like the query cache, a missing or broken manifest is not an
        error: it only means a full index.

        Returns:
            The manifest, or ``None`` when it is missing, unreadable, not
            valid, or saved with another ``raw_dir`` or
            ``max_chunk_size`` (no saved chunk would be valid).
        """
        try:
            text = self._manifest_path.read_text(encoding="utf-8")
            manifest = IndexManifest.model_validate_json(text)
        except (OSError, ValidationError, UnicodeDecodeError):
            return None
        if (manifest.raw_dir != self._raw_dir
                or manifest.max_chunk_size != self._max_chunk_size):
            return None
        return manifest

    def _load_indexers(self) -> tuple[BM25Index, VectorIndex] | None:
        """The saved indexes, if they can be updated.

        Returns:
            Both indexes, or ``None`` when one cannot be loaded (any
            ``RagError``: missing, unreadable or broken) or they do not
            have the same number of chunks.
        """
        try:
            bm25 = BM25Index.load(self._index_dir)
            vectors = VectorIndex.load(self._index_dir)
        except RagError:
            return None
        if bm25.chunk_count != vectors.chunk_count:
            return None
        return bm25, vectors

    def _chunk(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into chunks: ``PyChunker`` for Python,
        ``ArticleChunker`` for Markdown and text.

        Args:
            documents: The documents read.

        Returns:
            Their chunks, not sorted (``_plan`` sorts them).
        """
        chunkerpy = PyChunker(self._max_chunk_size)
        chunkermd = ArticleChunker(self._max_chunk_size)
        chunkspy = chunkerpy.chunk_documents(
            [doc for doc in documents if doc.file_path.endswith(".py")])
        chunksmd = chunkermd.chunk_documents(
            [doc for doc in documents
             if doc.file_path.endswith((".md", ".txt"))])
        return chunkspy + chunksmd

    def _remove_manifest(self) -> None:
        """Remove the manifest before the indexes are written.

        If ``index`` stops halfway, no manifest describes indexes that
        were only partly replaced, so the next run is a full index.

        Raises:
            PathError: The file exists but cannot be removed, or the
                index folder is not a folder.
        """
        try:
            self._manifest_path.unlink(missing_ok=True)
        except OSError as e:
            raise PathError(self._index_dir, e) from e

    def _save(self, bm25: BM25Index, vectors: VectorIndex,
              stamps: dict[str, FileStamp]) -> None:
        """Write both indexes, remove the query and answer caches, and
        write the manifest last.

        Args:
            bm25: The BM25 index to save.
            vectors: The vector index to save.
            stamps: The corpus stamps the indexes were built from.

        Raises:
            PathError: A file cannot be written or removed.
            RagError: The tokenizer of the model cannot be written.
        """
        print("Saving the index...", file=sys.stderr)
        bm25.save(self._index_dir)
        vectors.save(self._index_dir)
        # Results cached with the old index are never reused (the namespace
        # changes), so remove the files instead of leaving them on disk.
        # The answers too: they are only valid for the chunks they were
        # generated from.
        cache_paths = [CachedRetriever.file_path(self._index_dir, name)
                       for name in get_args(RetrieverMode)]
        cache_paths += [AnswerCache.file_path(self._index_dir, name)
                        for name in get_args(GeneratorMode)]
        for cache_path in cache_paths:
            try:
                cache_path.unlink(missing_ok=True)
            except OSError as e:
                raise PathError(cache_path, e) from e
        manifest = IndexManifest(max_chunk_size=self._max_chunk_size,
                                 raw_dir=self._raw_dir, files=stamps)
        try:
            self._manifest_path.write_text(manifest.model_dump_json(),
                                           encoding="utf-8")
        except OSError as e:
            raise PathError(self._index_dir, e) from e

    def run(self, full: bool = False) -> IndexReport:
        """Bring the saved index up to date with the corpus.

        Incremental when the manifest and both indexes can be reused:
        only new and modified files are read, and only their chunks are
        tokenized and embedded. Otherwise, or with ``full``, every file
        is. With no change, nothing is written.

        Args:
            full: Index the whole corpus even if the saved index could be
                updated (for example, after changing the chunker code,
                which the stamps cannot detect).

        Returns:
            What was done, for ``index`` to print.

        Raises:
            CorpusError: The corpus folder is not valid or, indexing in
                full, no file is readable.
            IndexBuildError: There is nothing to index.
            PathError: A file of the index cannot be written.
            RagError: The embedding model cannot be loaded.
        """
        stamps = self._reader.stamps()
        manifest = None if full else self._load_manifest()
        old = None
        if manifest is not None:
            print("Loading the saved index...", file=sys.stderr)
            old = self._load_indexers()

        if manifest is None or old is None:
            changes = self._diff({}, stamps)
        else:
            changes = self._diff(manifest.files, stamps)
            if changes.is_empty:
                return IndexReport(full=False, changes=changes, read=None,
                                   chunk_count=old[0].chunk_count,
                                   fresh_count=0)

        self._remove_manifest()
        read = self._reader.read(
            only=None if old is None else changes.to_read)
        new_chunks = self._chunk(read.documents)

        if old is None:
            plan = self._plan([], [], new_chunks)
            bm25 = BM25Index.build(plan.chunks)
            # Same chunks in the same order: chunk ids must match BM25.
            vectors = VectorIndex.build(plan.chunks)
        else:
            plan = self._plan(old[0].chunks, changes.to_remove, new_chunks)
            bm25 = old[0].update(plan)
            vectors = old[1].update(plan)

        self._save(bm25, vectors, stamps)
        return IndexReport(full=old is None, changes=changes, read=read,
                           chunk_count=len(plan.chunks),
                           fresh_count=len(plan.fresh_ids))

    @staticmethod
    def _diff(old: dict[str, FileStamp],
              new: dict[str, FileStamp]) -> CorpusChanges:
        """Compare the saved stamps with the current ones.

        Static: it only needs its two arguments, so it is tested without
        files or an index.

        Args:
            old: Stamps from the saved manifest (empty when there is
                none: then every file is new).
            new: Stamps of the corpus now (``CorpusReader.stamps``).

        Returns:
            New paths (only in ``new``), deleted paths (only in ``old``)
            and modified paths (in both, with another stamp), each list
            sorted. Unchanged files are in no list.
        """
        news = new.keys() - old.keys()
        deleted = old.keys() - new.keys()
        common = old.keys() & new.keys()
        modified = [file for file in common if old[file] != new[file]]
        return CorpusChanges(
            new=sorted(news),
            deleted=sorted(deleted),
            modified=sorted(modified)
        )

    @staticmethod
    def _plan(old_chunks: list[Chunk],
              to_remove: list[str],
              new_chunks: list[Chunk]) -> IndexPlan:
        """Decide the chunks of the updated index and renumber the kept
        ones, without touching the indexes.

        ``order_key`` identifies a chunk: a kept chunk and a new one never
        share it, because new chunks only come from new or modified files
        and the old chunks of those are removed.

        Args:
            old_chunks: The chunks of the saved index, by chunk id.
            to_remove: Files whose old chunks leave the index
                (``CorpusChanges.to_remove``).
            new_chunks: The chunks of the files read again
                (``CorpusChanges.to_read``), in any order.

        Returns:
            The kept and new chunks sorted as a full index would sort
            them, and ``{old id: new id}`` of the kept ones.
        """
        removed = set(to_remove)
        old_ids = {chunk.order_key(): old_id
                   for old_id, chunk in enumerate(old_chunks)
                   if chunk.file_path not in removed}
        kept = [chunk for chunk in old_chunks
                if chunk.file_path not in removed]
        chunks = sorted(
            new_chunks + kept,
            key=Chunk.order_key
        )
        remap = {old_ids[chunk.order_key()]: new_id
                 for new_id, chunk in enumerate(chunks)
                 if chunk.order_key() in old_ids}
        return IndexPlan(chunks=chunks, remap=remap)
