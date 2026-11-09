"""Tests for the incremental indexing (bonus 3).

The models that describe the corpus (``FileStamp``, ``IndexManifest``,
``CorpusChanges``) and ``IncrementalIndexer._diff``, which compares the
saved manifest with the files found now; ``IndexPlan`` and
``IncrementalIndexer._plan``, which renumber the chunks: there the
"files" are dictionaries of stamps and the chunks are built in memory.
``TestRun`` runs ``IncrementalIndexer.run`` on real files in
``tmp_path``, with the fake embedding model of ``conftest.py``.
"""
from collections.abc import Collection
import os
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from src.answer_cache import AnswerCache
from src.bm25_index import BM25Index
from src.cached_retriever import CachedRetriever
from src.exceptions import PathError
from src.incremental_indexer import IncrementalIndexer
from src.models import (Chunk, CorpusChanges, CorpusReadResult, FileStamp,
                        IndexManifest, IndexPlan, IndexReport)
from src.reader import CorpusReader
from src.vector_index import VectorIndex

OLD = FileStamp(mtime_ns=1_000, size=10)
NEWER = FileStamp(mtime_ns=2_000, size=10)
BIGGER = FileStamp(mtime_ns=1_000, size=11)


def no_changes() -> CorpusChanges:
    """Three empty lists."""
    return CorpusChanges(new=[], modified=[], deleted=[])


def chunk(path: str, start: int, text: str = "old") -> Chunk:
    """A one-character chunk of ``path`` starting at ``start``."""
    return Chunk(file_path=path, first_character_index=start,
                 last_character_index=start + 1, text=text)


OLD_CHUNKS = [chunk("a.md", 0), chunk("a.md", 50), chunk("b.md", 0),
              chunk("c.py", 0)]


class TestFileStamp:
    """When a file last changed."""

    def test_equal(self) -> None:
        """Same time and size: the same stamp (the file did not change)."""
        assert FileStamp(mtime_ns=1_000, size=10) == OLD

    @pytest.mark.parametrize("other", [NEWER, BIGGER])
    def test_different(self, other: FileStamp) -> None:
        """Another time or another size: another stamp."""
        assert other != OLD

    def test_nanoseconds_kept(self) -> None:
        """``st_mtime_ns`` values (about 1.8e18) are kept exactly: two
        writes 1 ns apart are told apart."""
        stamp = FileStamp(mtime_ns=1_791_369_129_831_588_221, size=1)
        twin = FileStamp(mtime_ns=1_791_369_129_831_588_222, size=1)
        assert stamp != twin
        assert FileStamp.model_validate_json(
            stamp.model_dump_json()) == stamp

    @pytest.mark.parametrize("missing", ["mtime_ns", "size"])
    def test_field_required(self, missing: str) -> None:
        """Both fields are required."""
        data = {"mtime_ns": 1, "size": 1}
        del data[missing]
        with pytest.raises(ValidationError):
            FileStamp.model_validate(data)


class TestIndexManifest:
    """What ``manifest.json`` holds."""

    @pytest.fixture
    def manifest(self) -> IndexManifest:
        """A manifest with two files."""
        return IndexManifest(max_chunk_size=2000, raw_dir="data/raw",
                             files={"data/raw/a.py": OLD,
                                    "data/raw/b.md": NEWER})

    def test_round_trip(self, manifest: IndexManifest) -> None:
        """Saved as JSON and read back, it is the same manifest."""
        text = manifest.model_dump_json()
        assert IndexManifest.model_validate_json(text) == manifest

    def test_no_files(self) -> None:
        """An empty corpus is a valid manifest."""
        manifest = IndexManifest(max_chunk_size=500, raw_dir="x", files={})
        assert manifest.files == {}

    @pytest.mark.parametrize("size", [0, -1, 2001])
    def test_chunk_size_out_of_range(self, size: int) -> None:
        """``max_chunk_size`` has the limits of ``index`` (1 to 2000)."""
        with pytest.raises(ValidationError):
            IndexManifest(max_chunk_size=size, raw_dir="data/raw",
                          files={})

    @pytest.mark.parametrize("text", [
        "",
        "not json",
        "[]",
        '{"raw_dir": "data/raw", "files": {}}',
        '{"max_chunk_size": 2000, "files": {}}',
        '{"max_chunk_size": 2000, "raw_dir": "data/raw"}',
        '{"max_chunk_size": 2000, "raw_dir": "data/raw",'
        ' "files": {"a.py": {"size": 1}}}',
        '{"max_chunk_size": 2000, "raw_dir": "data/raw",'
        ' "files": {"a.py": {"mtime_ns": "soon", "size": 1}}}',
        '{"max_chunk_size": 2000, "raw_dir": "data/raw", "files": []}',
    ])
    def test_broken_rejected(self, text: str) -> None:
        """A broken or incomplete manifest is a ``ValidationError``
        (``index`` will treat it as no manifest: a full rebuild)."""
        with pytest.raises(ValidationError):
            IndexManifest.model_validate_json(text)


class TestCorpusChanges:
    """The three lists and the questions they answer."""

    @pytest.fixture
    def changes(self) -> CorpusChanges:
        """One file of each kind."""
        return CorpusChanges(new=["w.md"], modified=["y.py"],
                             deleted=["z.md"])

    def test_to_remove(self, changes: CorpusChanges) -> None:
        """Modified and deleted files lose their old chunks."""
        assert changes.to_remove == ["y.py", "z.md"]

    def test_to_read(self, changes: CorpusChanges) -> None:
        """New and modified files are read and chunked."""
        assert changes.to_read == ["w.md", "y.py"]

    def test_modified_in_both(self, changes: CorpusChanges) -> None:
        """A modified file is removed and read again."""
        assert "y.py" in changes.to_remove
        assert "y.py" in changes.to_read

    def test_sorted(self) -> None:
        """Both lists are sorted, whatever list a path comes from."""
        changes = CorpusChanges(new=["b", "d"], modified=["c", "a"],
                                deleted=["e", "0"])
        assert changes.to_read == ["a", "b", "c", "d"]
        assert changes.to_remove == ["0", "a", "c", "e"]

    def test_empty(self) -> None:
        """No list with anything: nothing changed."""
        assert no_changes().is_empty
        assert no_changes().to_read == []
        assert no_changes().to_remove == []

    @pytest.mark.parametrize("field", ["new", "modified", "deleted"])
    def test_one_change_not_empty(self, field: str) -> None:
        """A single file in any list is a change."""
        changes = no_changes().model_copy(update={field: ["a.py"]})
        assert not changes.is_empty


class TestDiff:
    """``IncrementalIndexer._diff``: saved stamps against current ones."""

    def test_nothing_changed(self) -> None:
        """Same files, same stamps: no change."""
        files = {"a.py": OLD, "b.md": NEWER}
        assert IncrementalIndexer._diff(files, dict(files)).is_empty

    def test_both_empty(self) -> None:
        """No files before nor now: no change."""
        assert IncrementalIndexer._diff({}, {}).is_empty

    def test_new_file(self) -> None:
        """A path only in the current files is new."""
        changes = IncrementalIndexer._diff({"a.py": OLD},
                                           {"a.py": OLD, "b.py": OLD})
        assert changes == CorpusChanges(new=["b.py"], modified=[],
                                        deleted=[])

    def test_deleted_file(self) -> None:
        """A path only in the saved files was deleted."""
        changes = IncrementalIndexer._diff({"a.py": OLD, "b.py": OLD},
                                           {"a.py": OLD})
        assert changes == CorpusChanges(new=[], modified=[],
                                        deleted=["b.py"])

    @pytest.mark.parametrize("now", [NEWER, BIGGER])
    def test_modified_file(self, now: FileStamp) -> None:
        """Same path with another time or size: modified."""
        changes = IncrementalIndexer._diff({"a.py": OLD}, {"a.py": now})
        assert changes == CorpusChanges(new=[], modified=["a.py"],
                                        deleted=[])

    def test_unchanged_not_listed(self) -> None:
        """Files with the same stamp are in no list."""
        changes = IncrementalIndexer._diff({"a.py": OLD, "b.py": OLD},
                                           {"a.py": OLD, "b.py": NEWER})
        assert "a.py" not in (changes.new + changes.modified
                              + changes.deleted)

    def test_all_at_once(self) -> None:
        """New, modified, deleted and unchanged files together."""
        old = {"keep.py": OLD, "edit.py": OLD, "gone.md": OLD}
        new = {"keep.py": OLD, "edit.py": NEWER, "added.md": OLD}
        changes = IncrementalIndexer._diff(old, new)
        assert changes == CorpusChanges(new=["added.md"],
                                        modified=["edit.py"],
                                        deleted=["gone.md"])
        assert changes.to_remove == ["edit.py", "gone.md"]
        assert changes.to_read == ["added.md", "edit.py"]

    def test_no_manifest_all_new(self) -> None:
        """No saved files (first ``index``): every file is new, so a full
        index is the incremental case with an empty past."""
        new = {"b.py": OLD, "a.md": OLD, "c.txt": OLD}
        changes = IncrementalIndexer._diff({}, new)
        assert changes == CorpusChanges(new=["a.md", "b.py", "c.txt"],
                                        modified=[], deleted=[])

    def test_corpus_emptied(self) -> None:
        """Every saved file gone: every file is deleted."""
        changes = IncrementalIndexer._diff({"a.py": OLD, "b.md": OLD}, {})
        assert changes.deleted == ["a.py", "b.md"]
        assert changes.to_read == []

    def test_lists_sorted(self) -> None:
        """Each list is sorted, whatever the order of the dictionaries
        (sets have no fixed order)."""
        paths = [f"data/raw/{name}.py" for name in "qwertyuiop"]
        changes = IncrementalIndexer._diff(
            {}, {path: OLD for path in paths})
        assert changes.new == sorted(paths)

    def test_inputs_unchanged(self) -> None:
        """The two dictionaries are only read."""
        old = {"a.py": OLD, "b.py": OLD}
        new = {"a.py": NEWER, "c.py": OLD}
        old_copy, new_copy = dict(old), dict(new)
        IncrementalIndexer._diff(old, new)
        assert old == old_copy
        assert new == new_copy


class TestIndexPlan:
    """The chunks of the updated index and their renumbering."""

    def test_fresh_ids_from_values(self) -> None:
        """Fresh ids are the new ids missing from the remap values, not
        from its keys: here the old id 2 moved to 3, so 2 is fresh."""
        plan = IndexPlan(chunks=[chunk("a.md", i) for i in range(4)],
                         remap={0: 0, 1: 1, 2: 3})
        assert plan.fresh_ids == [2]

    def test_all_fresh(self) -> None:
        """Nothing reused: every id is fresh, in order."""
        plan = IndexPlan(chunks=[chunk("a.md", i) for i in range(3)],
                         remap={})
        assert plan.fresh_ids == [0, 1, 2]

    def test_none_fresh(self) -> None:
        """Every chunk reused: nothing to process."""
        plan = IndexPlan(chunks=[chunk("a.md", i) for i in range(2)],
                         remap={5: 0, 7: 1})
        assert plan.fresh_ids == []


class TestPlan:
    """``_plan``: the kept and new chunks, sorted, and ``{old: new}``."""

    @staticmethod
    def full_order(kept: list[Chunk], new: list[Chunk]) -> list[Chunk]:
        """What a full ``index`` would build from the same chunks."""
        return sorted(kept + new, key=Chunk.order_key)

    def test_nothing_changed(self) -> None:
        """Same chunks, same ids, nothing to process."""
        plan = IncrementalIndexer._plan(OLD_CHUNKS, [], [])
        assert plan.chunks == OLD_CHUNKS
        assert plan.remap == {0: 0, 1: 1, 2: 2, 3: 3}
        assert plan.fresh_ids == []

    def test_new_file_shifts_the_rest(self) -> None:
        """A new file sorted in the middle pushes the later chunks one
        id up: the case where a wrong remap would go unnoticed."""
        new = [chunk("aa.md", 0, "new")]
        plan = IncrementalIndexer._plan(OLD_CHUNKS, [], new)
        assert plan.chunks == self.full_order(OLD_CHUNKS, new)
        assert plan.remap == {0: 0, 1: 1, 2: 3, 3: 4}
        assert plan.fresh_ids == [2]
        assert plan.chunks[2].text == "new"

    def test_deleted_file(self) -> None:
        """A deleted file leaves; the later chunks move one id down."""
        plan = IncrementalIndexer._plan(OLD_CHUNKS, ["b.md"], [])
        assert [c.file_path for c in plan.chunks] == ["a.md", "a.md",
                                                      "c.py"]
        assert plan.remap == {0: 0, 1: 1, 3: 2}
        assert plan.fresh_ids == []

    def test_modified_same_place(self) -> None:
        """A modified file is replaced where it was: its old chunk is
        not reused even with the same position in the file."""
        new = [chunk("b.md", 0, "new")]
        plan = IncrementalIndexer._plan(OLD_CHUNKS, ["b.md"], new)
        assert plan.remap == {0: 0, 1: 1, 3: 3}
        assert plan.fresh_ids == [2]
        assert plan.chunks[2].text == "new"

    def test_modified_more_chunks(self) -> None:
        """A modified file with more chunks than before shifts the rest."""
        new = [chunk("a.md", start, "new") for start in (0, 30, 60)]
        plan = IncrementalIndexer._plan(OLD_CHUNKS, ["a.md"], new)
        kept = [c for c in OLD_CHUNKS if c.file_path != "a.md"]
        assert plan.chunks == self.full_order(kept, new)
        assert plan.remap == {2: 3, 3: 4}
        assert plan.fresh_ids == [0, 1, 2]

    def test_modified_fewer_chunks(self) -> None:
        """A modified file with fewer chunks: the rest move down."""
        new = [chunk("a.md", 0, "new")]
        plan = IncrementalIndexer._plan(OLD_CHUNKS, ["a.md"], new)
        assert plan.remap == {2: 1, 3: 2}
        assert plan.fresh_ids == [0]

    def test_no_saved_index(self) -> None:
        """No saved index: everything is new and sorted (``new_chunks``
        may come in any order)."""
        new = [chunk("b.md", 0, "new"), chunk("a.md", 0, "new")]
        plan = IncrementalIndexer._plan([], [], new)
        assert [c.file_path for c in plan.chunks] == ["a.md", "b.md"]
        assert plan.remap == {}
        assert plan.fresh_ids == [0, 1]

    def test_everything_removed(self) -> None:
        """Every file removed and nothing new: an empty plan."""
        plan = IncrementalIndexer._plan(
            OLD_CHUNKS, ["a.md", "b.md", "c.py"], [])
        assert plan == IndexPlan(chunks=[], remap={})

    def test_all_at_once(self) -> None:
        """New, modified and deleted together give the same chunks as a
        full index, and every kept chunk lands on itself."""
        new = [chunk("aa.md", 0, "new"), chunk("c.py", 0, "new"),
               chunk("c.py", 99, "new")]
        plan = IncrementalIndexer._plan(OLD_CHUNKS, ["b.md", "c.py"], new)
        kept = OLD_CHUNKS[:2]
        assert plan.chunks == self.full_order(kept, new)
        for old_id, new_id in plan.remap.items():
            assert plan.chunks[new_id] == OLD_CHUNKS[old_id]
        assert plan.remap == {0: 0, 1: 1}
        assert plan.fresh_ids == [2, 3, 4]

    def test_remove_unknown_file(self) -> None:
        """Removing a file that has no chunks (for example, an empty
        file) changes nothing."""
        plan = IncrementalIndexer._plan(OLD_CHUNKS, ["empty.md"], [])
        assert plan.remap == {0: 0, 1: 1, 2: 2, 3: 3}

    def test_inputs_unchanged(self) -> None:
        """The lists given are only read."""
        old, new = list(OLD_CHUNKS), [chunk("aa.md", 0, "new")]
        IncrementalIndexer._plan(old, ["b.md"], new)
        assert old == OLD_CHUNKS
        assert new == [chunk("aa.md", 0, "new")]


class Corpus:
    """A corpus folder and an index folder under ``tmp_path``.

    Every write sets an explicit modification time one second after the
    previous one, so a change is always seen even on a file system with
    coarse timestamps.
    """

    def __init__(self, root: Path) -> None:
        """Create ``root/raw`` with three files and the index paths."""
        self.raw = root / "raw"
        self.index_dir = root / "index"
        self.raw.mkdir()
        self._clock = 1_000_000_000
        self.write("a.md", "# Title\nLoad a LoRA adapter.\n")
        self.write("b.py", "def load():\n    return 1\n")
        self.write("c.txt", "Plain text about the server.\n")

    def write(self, name: str, text: str) -> None:
        """Write a corpus file with a new modification time."""
        path = self.raw / name
        path.write_text(text, encoding="utf-8")
        self._clock += 1
        os.utime(path, ns=(self._clock * 10**9, self._clock * 10**9))

    def indexer(self, index_dir: Path | None = None,
                max_chunk_size: int = 2000) -> IncrementalIndexer:
        """An indexer of this corpus into ``index_dir``."""
        return IncrementalIndexer(str(self.raw),
                                  str(index_dir or self.index_dir),
                                  max_chunk_size)

    def run(self, full: bool = False) -> IndexReport:
        """``run`` into the main index folder."""
        return self.indexer().run(full)

    def path(self, name: str) -> str:
        """The path of a corpus file as stored in chunks and manifest."""
        return f"raw/{name}"


class TestRun:
    """``run``: full, up to date or incremental, on real files."""

    @pytest.fixture
    def corpus(self, tmp_path: Path,
               monkeypatch: pytest.MonkeyPatch) -> Corpus:
        """A three-file corpus inside a project root at ``tmp_path``."""
        monkeypatch.setattr(CorpusReader, "PROJECT_ROOT", tmp_path)
        return Corpus(tmp_path)

    @staticmethod
    def assert_same_index(left: Path, right: Path) -> None:
        """Two saved index folders hold the same indexes."""
        bm25_left, bm25_right = BM25Index.load(left), BM25Index.load(right)
        assert bm25_left.chunks == bm25_right.chunks
        assert bm25_left._postings == bm25_right._postings
        assert bm25_left._lengths == bm25_right._lengths
        vec_left, vec_right = VectorIndex.load(left), VectorIndex.load(right)
        assert np.array_equal(vec_left._owners, vec_right._owners)
        assert np.array_equal(vec_left._vectors, vec_right._vectors)

    def test_first_run_full(self, corpus: Corpus) -> None:
        """No manifest: every file is new, and the manifest is written
        with their stamps."""
        report = corpus.run()
        assert report.full
        assert report.changes.new == [corpus.path("a.md"),
                                      corpus.path("b.py"),
                                      corpus.path("c.txt")]
        assert report.fresh_count == report.chunk_count > 0
        manifest = IndexManifest.model_validate_json(
            (corpus.index_dir / IncrementalIndexer.MANIFEST_NAME)
            .read_text(encoding="utf-8"))
        assert manifest.files == CorpusReader(corpus.raw).stamps()
        assert manifest.raw_dir == str(corpus.raw)

    def test_up_to_date_writes_nothing(self, corpus: Corpus) -> None:
        """Nothing changed: nothing is read or written."""
        corpus.run()
        before = {path.name: path.stat().st_mtime_ns
                  for path in corpus.index_dir.iterdir()}
        report = corpus.run()
        assert report.read is None
        assert not report.full
        assert report.changes.is_empty
        assert report.fresh_count == 0
        assert report.chunk_count == BM25Index.load(
            corpus.index_dir).chunk_count
        assert before == {path.name: path.stat().st_mtime_ns
                          for path in corpus.index_dir.iterdir()}

    def test_incremental_equals_full(self, corpus: Corpus,
                                     tmp_path: Path) -> None:
        """New, modified and deleted files at once: the updated index is
        the same as a full index of the corpus now. The key test."""
        corpus.run()
        corpus.write("b.py", "def load():\n    return 2\n\n"
                     "def save():\n    pass\n")
        corpus.write("aa.md", "# New\nA new page in the middle.\n")
        (corpus.raw / "c.txt").unlink()

        report = corpus.run()
        assert not report.full
        assert report.changes == CorpusChanges(
            new=[corpus.path("aa.md")], modified=[corpus.path("b.py")],
            deleted=[corpus.path("c.txt")])
        assert report.fresh_count < report.chunk_count

        full_dir = tmp_path / "full"
        corpus.indexer(full_dir).run(full=True)
        self.assert_same_index(corpus.index_dir, full_dir)

    def test_only_changed_files_read(self, corpus: Corpus,
                                     monkeypatch: pytest.MonkeyPatch
                                     ) -> None:
        """Only new and modified files are read: the reader gets them as
        ``only``."""
        corpus.run()
        corpus.write("a.md", "# Title\nChanged.\n")
        corpus.write("d.md", "Another page.\n")
        asked: list[list[str] | None] = []
        real_read = CorpusReader.read

        def spy(reader: CorpusReader,
                only: Collection[str] | None = None) -> CorpusReadResult:
            """Record what is asked, then read for real."""
            asked.append(None if only is None else sorted(only))
            return real_read(reader, only)

        monkeypatch.setattr(CorpusReader, "read", spy)
        corpus.run()
        assert asked == [[corpus.path("a.md"), corpus.path("d.md")]]

    def test_only_deleted(self, corpus: Corpus, tmp_path: Path) -> None:
        """Only a deletion: nothing is read or embedded, the chunks of
        the file leave."""
        corpus.run()
        (corpus.raw / "b.py").unlink()
        report = corpus.run()
        assert report.read is not None
        assert report.read.documents == []
        assert report.fresh_count == 0
        assert corpus.path("b.py") not in {
            chunk.file_path
            for chunk in BM25Index.load(corpus.index_dir).chunks}
        corpus.indexer(tmp_path / "full").run(full=True)
        self.assert_same_index(corpus.index_dir, tmp_path / "full")

    def test_full_flag(self, corpus: Corpus) -> None:
        """``full`` indexes everything even with nothing changed."""
        corpus.run()
        report = corpus.run(full=True)
        assert report.full
        assert report.fresh_count == report.chunk_count

    def test_manifest_missing(self, corpus: Corpus) -> None:
        """No manifest next to an index: full index."""
        corpus.run()
        (corpus.index_dir / IncrementalIndexer.MANIFEST_NAME).unlink()
        assert corpus.run().full

    @pytest.mark.parametrize("content", ["", "not json", "{}",
                                         '{"max_chunk_size": 0, '
                                         '"raw_dir": "raw", "files": {}}'])
    def test_manifest_broken(self, corpus: Corpus, content: str) -> None:
        """A broken or invalid manifest is not an error: full index."""
        corpus.run()
        (corpus.index_dir / IncrementalIndexer.MANIFEST_NAME).write_text(
            content, encoding="utf-8")
        assert corpus.run().full

    def test_other_chunk_size(self, corpus: Corpus) -> None:
        """Another ``max_chunk_size``: no saved chunk is valid, full
        index."""
        corpus.run()
        assert corpus.indexer(max_chunk_size=500).run().full
        assert not corpus.indexer(max_chunk_size=500).run().full

    @pytest.mark.parametrize("name", [BM25Index.FILE_NAME,
                                      VectorIndex.VECTOR_NAME])
    def test_index_file_missing(self, corpus: Corpus, name: str) -> None:
        """A manifest without its index: full index, not an error."""
        corpus.run()
        (corpus.index_dir / name).unlink()
        report = corpus.run()
        assert report.full
        assert (corpus.index_dir / name).is_file()

    def test_answer_cache_removed(self, corpus: Corpus) -> None:
        """The answer cache follows the same rule: kept while the index
        is up to date, removed when it changes or with ``full``."""
        corpus.run()
        answers = AnswerCache.file_path(corpus.index_dir)
        answers.write_text("{}", encoding="utf-8")
        corpus.run()
        assert answers.is_file()
        corpus.write("a.md", "# Title\nChanged.\n")
        corpus.run()
        assert not answers.exists()
        answers.write_text("{}", encoding="utf-8")
        corpus.run(full=True)
        assert not answers.exists()

    def test_query_caches_removed(self, corpus: Corpus) -> None:
        """A changed index removes the query caches; an index up to date
        keeps them (their namespace is still valid)."""
        corpus.run()
        caches = [CachedRetriever.file_path(corpus.index_dir, name)
                  for name in ("bm25", "hybrid")]
        for cache in caches:
            cache.write_text("{}", encoding="utf-8")
        corpus.run()
        assert all(cache.is_file() for cache in caches)
        corpus.write("a.md", "# Title\nChanged.\n")
        corpus.run()
        assert not any(cache.is_file() for cache in caches)

    def test_no_manifest_after_failed_save(self, corpus: Corpus,
                                           monkeypatch: pytest.MonkeyPatch
                                           ) -> None:
        """If writing the indexes fails, no manifest is left: the next
        run is a full index instead of trusting a half-written one."""
        corpus.run()
        corpus.write("a.md", "# Title\nChanged.\n")

        def fail(index: VectorIndex, files_dir: Path) -> None:
            """Fail like a full disk."""
            raise PathError(files_dir, OSError("No space left on device"))

        monkeypatch.setattr(VectorIndex, "save", fail)
        with pytest.raises(PathError):
            corpus.run()
        assert not (corpus.index_dir
                    / IncrementalIndexer.MANIFEST_NAME).exists()
