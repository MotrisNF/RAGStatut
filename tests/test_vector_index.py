"""Tests for ``src.vector_index.VectorIndex``.

The embedding model is ``FakeEncoder`` (see ``conftest.py``): the tests
check windows, owners, scores and files, not the quality of the model.
"""
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest

from src.exceptions import IndexNotFoundError, PathError, VectorIndexError
from src.incremental_indexer import IncrementalIndexer
from src.models import Chunk
from src.tokenizer_bm import Tokenizer
from src.vector_index import VectorIndex
from conftest import FakeEncoder


def make_chunks(*texts: str, path: str = "data/raw/{i}.md") -> list[Chunk]:
    """One chunk per text, each in its own fake file."""
    return [Chunk(file_path=path.format(i=i), first_character_index=0,
                  last_character_index=len(text), text=text)
            for i, text in enumerate(texts)]


class OneHotEncoder:
    """Encoder whose query vector is ``e0``, for exact window scores."""

    def encode(self, texts: list[str], normalize_embeddings: bool = False,
               **kwargs: object) -> NDArray[np.float32]:
        """Always the unit vector along the first axis."""
        vector = np.zeros((len(texts), 2), dtype=np.float32)
        vector[:, 0] = 1.0
        return vector


def index_with_scores(scores: list[float], owners: list[int]) -> VectorIndex:
    """An index whose windows score exactly ``scores`` for any query.

    Each window is a unit vector ``(s, sqrt(1 - s²))``; its dot product
    with ``e0`` (``OneHotEncoder``) is ``s``.
    """
    s = np.array(scores, dtype=np.float32)
    vectors = np.stack([s, np.sqrt(1 - s ** 2)], axis=1)
    index = VectorIndex(vectors, np.array(owners))
    index._model = OneHotEncoder()  # type: ignore[assignment]
    return index


class TestGetWindow:
    """``_get_window``: whole lines, closed past ``MAX_WINDOW_SIZE``."""

    def test_lines_fill_windows(self) -> None:
        """25 lines of 80 characters → 5 windows of 400."""
        text = ("x" * 79 + "\n") * 25
        windows = VectorIndex._get_window(text)
        assert [len(w) for w in windows] == [400] * 5

    def test_long_line_alone(self) -> None:
        """A line longer than the limit is a window of its own."""
        text = "a" * 500 + "\n" + "b" * 10 + "\n"
        assert [len(w) for w in VectorIndex._get_window(text)] == [501, 11]

    def test_empty_text(self) -> None:
        """No text, no window."""
        assert VectorIndex._get_window("") == []

    def test_joined_back(self) -> None:
        """Joining the windows gives back the original text."""
        text = "".join(f"line {i} " * (i % 7 + 1) + "\n" for i in range(200))
        assert "".join(VectorIndex._get_window(text)) == text

    def test_no_window_over_limit_unless_one_line(self) -> None:
        """A window only goes over the limit when it is a single line."""
        text = "".join("w" * (i * 37 % 300) + "\n" for i in range(100))
        for window in VectorIndex._get_window(text):
            assert (len(window) <= VectorIndex.MAX_WINDOW_SIZE
                    or window.count("\n") == 1)


class TestWindowChunk:
    """``_window_chunk``: path prefix and chunk order."""

    def test_path_words_in_front(self) -> None:
        """Every window starts with the tokenized file path."""
        [chunk] = make_chunks("x = 1\n", path="vllm/lora/worker_manager.py")
        [item] = VectorIndex._window_chunk([chunk])
        prefix = " ".join(Tokenizer().tokenize(chunk.file_path))
        assert prefix.startswith("vllm lora worker manager py")
        assert item.windows == [prefix + "\nx = 1\n"]

    def test_one_item_per_chunk_in_order(self) -> None:
        """Same order as the chunks, with the right window count."""
        chunks = make_chunks("a\n", ("b" * 300 + "\n") * 3, "")
        items = VectorIndex._window_chunk(chunks)
        assert [item.windows_count for item in items] == [1, 3, 0]
        assert all(item.windows_count == len(item.windows)
                   for item in items)


class TestBuild:
    """``build``: one vector per window, owners by chunk id."""

    def test_owners_follow_chunks(self) -> None:
        """Owners repeat each chunk id once per window, in order."""
        chunks = make_chunks("a\n", ("b" * 300 + "\n") * 3, "c\n")
        index = VectorIndex.build(chunks)
        assert index._owners.tolist() == [0, 1, 1, 1, 2]
        assert index._vectors.shape == (5, FakeEncoder.DIMENSIONS)
        assert index.chunk_count == 3

    def test_vectors_normalised(self) -> None:
        """Every window vector has length 1."""
        index = VectorIndex.build(make_chunks("load a lora adapter\n",
                                              "configure the server\n"))
        assert np.allclose(np.linalg.norm(index._vectors, axis=1), 1.0)

    def test_model_loaded_once(self, fake_encoder: FakeEncoder,
                               monkeypatch: pytest.MonkeyPatch) -> None:
        """``build`` keeps its model: the first query does not reload."""
        loads: list[str] = []

        def load(cls: type[VectorIndex]) -> FakeEncoder:
            """Count the loads, then return the shared fake model."""
            loads.append(cls.MODEL_NAME)
            return fake_encoder

        monkeypatch.setattr(VectorIndex, "_load_model", classmethod(load))
        index = VectorIndex.build(make_chunks("a b\n"))
        index.chunk_score("a")
        assert len(loads) == 1
        assert len(fake_encoder.calls) == 2

    def test_embeds_the_prefixed_windows(self,
                                         fake_encoder: FakeEncoder) -> None:
        """The texts sent to the model are the prefixed windows."""
        chunks = make_chunks("hello\n", path="docs/{i}.md")
        VectorIndex.build(chunks)
        assert fake_encoder.calls[0] == [
            item.windows[0] for item in VectorIndex._window_chunk(chunks)]
        assert fake_encoder.calls[0][0].endswith("md\nhello\n")


def chunk_at(path: str, start: int, text: str) -> Chunk:
    """A chunk of ``path`` starting at ``start``."""
    return Chunk(file_path=path, first_character_index=start,
                 last_character_index=start + len(text), text=text)


LONG = "".join(f"line {i} about loading models\n" for i in range(30))


class TestUpdate:
    """``update`` (bonus 3): the same rows as ``build`` over the plan's
    chunks, embedding only the fresh ones. ``LONG`` has several windows,
    so the windows of one chunk must stay together and in order."""

    OLD = [chunk_at("a.md", 0, "load the index\n"),
           chunk_at("a.md", 50, LONG),
           chunk_at("b.md", 0, "old words here\n"),
           chunk_at("c.py", 0, LONG)]

    def check(self, to_remove: list[str], new: list[Chunk]) -> VectorIndex:
        """Update the index of ``OLD`` and compare with a full build."""
        plan = IncrementalIndexer._plan(self.OLD, to_remove, new)
        updated = VectorIndex.build(self.OLD).update(plan)
        full = VectorIndex.build(plan.chunks)
        assert updated._owners.tolist() == full._owners.tolist()
        assert np.array_equal(updated._vectors, full._vectors)
        assert np.array_equal(updated.chunk_score("load a model"),
                              full.chunk_score("load a model"))
        return updated

    def test_nothing_changed(self) -> None:
        """An empty plan gives the same rows."""
        self.check([], [])

    def test_new_file_shifts_ids(self) -> None:
        """A new file in the middle: later owners move up by one."""
        updated = self.check([], [chunk_at("aa.md", 0, LONG)])
        assert updated.chunk_count == 5

    def test_modified_file(self) -> None:
        """A modified file is replaced where it was."""
        self.check(["b.md"], [chunk_at("b.md", 0, "new search text\n")])

    def test_deleted_file(self) -> None:
        """A deleted file: its rows leave, later owners move down."""
        updated = self.check(["b.md"], [])
        assert updated.chunk_count == 3

    def test_all_at_once(self) -> None:
        """New, modified and deleted together."""
        self.check(["a.md", "c.py"],
                   [chunk_at("a.md", 0, LONG), chunk_at("a.md", 900, "x\n"),
                    chunk_at("aa.md", 0, LONG)])

    def test_only_fresh_embedded(self, fake_encoder: FakeEncoder) -> None:
        """Only the windows of fresh chunks reach the model: that is the
        speed-up."""
        index = VectorIndex.build(self.OLD)
        fake_encoder.calls.clear()
        new = [chunk_at("aa.md", 0, "build index\n"),
               chunk_at("b.md", 0, "new text\n")]
        index.update(IncrementalIndexer._plan(self.OLD, ["b.md"], new))
        sent = [text for call in fake_encoder.calls for text in call]
        assert sent == [item.windows[0]
                        for item in VectorIndex._window_chunk(new)]

    def test_nothing_fresh_no_model_call(self,
                                         fake_encoder: FakeEncoder) -> None:
        """Only removals: nothing is embedded."""
        index = VectorIndex.build(self.OLD)
        fake_encoder.calls.clear()
        index.update(IncrementalIndexer._plan(self.OLD, ["b.md"], []))
        assert fake_encoder.calls == []

    def test_model_kept(self) -> None:
        """The new index keeps the loaded model, so it is not loaded
        again for the first query."""
        index = VectorIndex.build(self.OLD)
        plan = IncrementalIndexer._plan(self.OLD, [], [])
        assert index.update(plan)._model is index._model

    def test_self_unchanged(self) -> None:
        """The old index is not changed: ``update`` returns a new one."""
        index = VectorIndex.build(self.OLD)
        vectors, owners = index._vectors.copy(), index._owners.copy()
        plan = IncrementalIndexer._plan(
            self.OLD, ["a.md"], [chunk_at("aa.md", 0, LONG)])
        assert index.update(plan) is not index
        assert np.array_equal(index._vectors, vectors)
        assert np.array_equal(index._owners, owners)


class TestChunkScore:
    """``chunk_score``: one score per chunk, the best of its windows."""

    def test_best_window_per_chunk(self) -> None:
        """The plan example: windows 0.20 0.71 | 0.35 | 0.10 0.48."""
        index = index_with_scores([0.20, 0.71, 0.35, 0.10, 0.48],
                                  [0, 0, 1, 2, 2])
        assert np.allclose(index.chunk_score("q"), [0.71, 0.35, 0.48])

    def test_one_score_per_chunk_in_range(self) -> None:
        """Shape ``(chunk_count,)`` and cosine values in [-1, 1]."""
        index = VectorIndex.build(make_chunks(
            "load a lora adapter\n", "configure the server\n",
            ("lora " * 100 + "\n") * 3))
        scores = index.chunk_score("lora adapter")
        assert scores.shape == (3,)
        assert ((scores >= -1) & (scores <= 1)).all()

    def test_related_text_scores_higher(self) -> None:
        """A chunk sharing the query words beats an unrelated one."""
        index = VectorIndex.build(make_chunks("load a lora adapter\n",
                                              "tokenizer vocabulary\n"))
        scores = index.chunk_score("lora adapter")
        assert scores[0] > scores[1]

    def test_query_normalised_as_is(self, fake_encoder: FakeEncoder) -> None:
        """The query is sent unchanged (stopwords kept)."""
        index = VectorIndex.build(make_chunks("a\n"))
        index.chunk_score("How do I load the adapter?")
        assert fake_encoder.calls[-1] == ["How do I load the adapter?"]


class TestSaveLoad:
    """``save`` and ``load``: two ``.npy`` files, checked on load."""

    @pytest.fixture
    def index(self) -> VectorIndex:
        """A small built index."""
        return VectorIndex.build(make_chunks(
            "load a lora adapter\n", ("serve " * 90 + "\n") * 3, "x\n"))

    def test_round_trip(self, index: VectorIndex, tmp_path: Path) -> None:
        """What is loaded is exactly what was saved."""
        index.save(tmp_path)
        loaded = VectorIndex.load(tmp_path)
        assert np.array_equal(loaded._vectors, index._vectors)
        assert np.array_equal(loaded._owners, index._owners)
        assert loaded._vectors.dtype == index._vectors.dtype

    def test_same_scores_after_load(self, index: VectorIndex,
                                    tmp_path: Path) -> None:
        """A loaded index scores like the original one."""
        index.save(tmp_path)
        loaded = VectorIndex.load(tmp_path)
        assert np.allclose(loaded.chunk_score("lora"),
                           index.chunk_score("lora"))

    def test_creates_folder(self, index: VectorIndex,
                            tmp_path: Path) -> None:
        """A missing folder is created."""
        folder = tmp_path / "new" / "index"
        index.save(folder)
        assert (folder / VectorIndex.VECTOR_NAME).is_file()
        assert (folder / VectorIndex.OWNERS_NAME).is_file()

    @pytest.mark.parametrize("missing", [VectorIndex.VECTOR_NAME,
                                         VectorIndex.OWNERS_NAME])
    def test_missing_file(self, index: VectorIndex, tmp_path: Path,
                          missing: str) -> None:
        """Either file missing: ``IndexNotFoundError``."""
        index.save(tmp_path)
        (tmp_path / missing).unlink()
        with pytest.raises(IndexNotFoundError, match="Run 'index' first"):
            VectorIndex.load(tmp_path)

    def test_garbage_file(self, index: VectorIndex, tmp_path: Path) -> None:
        """A file that is not ``.npy``: ``VectorIndexError``."""
        index.save(tmp_path)
        (tmp_path / VectorIndex.VECTOR_NAME).write_text("garbage")
        with pytest.raises(VectorIndexError, match="cannot read"):
            VectorIndex.load(tmp_path)

    @pytest.mark.parametrize("vectors, owners", [
        (np.zeros((3, 4)), np.array([0, 1])),          # different lengths
        (np.zeros(3), np.array([0, 1, 2])),            # vectors not 2D
        (np.zeros((3, 4)), np.zeros((3, 1), int)),     # owners not 1D
        (np.zeros((3, 4)), np.array(5)),               # owners 0D
    ])
    def test_inconsistent_shapes(self, tmp_path: Path,
                                 vectors: NDArray[np.float64],
                                 owners: NDArray[np.int64]) -> None:
        """Arrays that do not fit together: ``VectorIndexError``."""
        np.save(tmp_path / VectorIndex.VECTOR_NAME, vectors)
        np.save(tmp_path / VectorIndex.OWNERS_NAME, owners)
        with pytest.raises(VectorIndexError, match="Run 'index' again"):
            VectorIndex.load(tmp_path)

    def test_unwritable_folder(self, index: VectorIndex,
                               tmp_path: Path) -> None:
        """A file where the folder should be: ``PathError``."""
        blocker = tmp_path / "file"
        blocker.write_text("x")
        with pytest.raises(PathError):
            index.save(blocker)
