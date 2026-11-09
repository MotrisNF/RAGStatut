"""Tests for ``src.static_encoder.StaticEncoder`` and ``Retriever``.

A tiny real model is built by hand: a word-level tokenizer with five
words and a table with one known vector per word, so every expected
vector can be computed in the test.
"""
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from safetensors.numpy import save_file
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

import src.static_encoder
from src.base_retriever import Retriever
from src.bm25_index import BM25Index
from src.exceptions import (IndexNotFoundError, PathError, RagError,
                            VectorIndexError)
from src.hybrid_retriever import HybridRetriever
from src.static_encoder import StaticEncoder

VOCABULARY = {"[UNK]": 0, "load": 1, "lora": 2, "adapter": 3, "server": 4}


def tiny_tokenizer() -> HFTokenizer:
    """Word-level tokenizer over ``VOCABULARY`` (unknown words → 0)."""
    tokenizer = HFTokenizer(WordLevel(VOCABULARY, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def tiny_table() -> NDArray[np.float32]:
    """One known 3-dimensional vector per word id."""
    return np.array([[0, 0, 1],      # [UNK]
                     [1, 0, 0],      # load
                     [0, 1, 0],      # lora
                     [1, 1, 0],      # adapter
                     [3, 4, 0]],     # server
                    dtype=np.float32)


@pytest.fixture
def encoder() -> StaticEncoder:
    """The tiny model."""
    return StaticEncoder(tiny_tokenizer(), tiny_table())


class TestEncode:
    """``encode``: mean of the token rows, normalised."""

    def test_mean_then_normalised(self, encoder: StaticEncoder) -> None:
        """``load lora`` → mean (0.5, 0.5, 0) → length 1."""
        [vector] = encoder.encode(["load lora"])
        assert vector == pytest.approx([2 ** -0.5, 2 ** -0.5, 0])

    def test_single_word_normalised(self, encoder: StaticEncoder) -> None:
        """``server`` → (3, 4, 0) / 5."""
        [vector] = encoder.encode(["server"])
        assert vector == pytest.approx([0.6, 0.8, 0])

    def test_repeated_word_counts(self, encoder: StaticEncoder) -> None:
        """Repeating a word weighs the mean towards it."""
        [once, twice] = encoder.encode(["load lora", "load load lora"])
        assert twice[0] > once[0]

    def test_unknown_word(self, encoder: StaticEncoder) -> None:
        """A word outside the vocabulary uses the ``[UNK]`` row."""
        [vector] = encoder.encode(["zzz"])
        assert vector == pytest.approx([0, 0, 1])

    @pytest.mark.parametrize("text", ["", "   ", "\n"])
    def test_no_token_is_zero(self, encoder: StaticEncoder,
                              text: str) -> None:
        """A text with no token gives a zero vector, not NaN."""
        [vector] = encoder.encode([text])
        assert not np.isnan(vector).any()
        assert (vector == 0).all()

    def test_shape_and_type(self, encoder: StaticEncoder) -> None:
        """One float32 row per text, in order."""
        vectors = encoder.encode(["load", "lora", "server", ""])
        assert vectors.shape == (4, 3)
        assert vectors.dtype == np.float32
        assert vectors[1] == pytest.approx([0, 1, 0])

    def test_no_text(self, encoder: StaticEncoder) -> None:
        """No text, no row."""
        assert encoder.encode([]).shape == (0, 3)


class TestSaveLoad:
    """``save`` and ``load``: the table and the tokenizer."""

    def test_round_trip(self, encoder: StaticEncoder,
                        tmp_path: Path) -> None:
        """The loaded model embeds exactly like the saved one."""
        encoder.save(tmp_path)
        loaded = StaticEncoder.load(tmp_path)
        texts = ["load lora adapter", "server", "zzz", ""]
        assert np.array_equal(loaded.encode(texts), encoder.encode(texts))

    def test_files_and_folder(self, encoder: StaticEncoder,
                              tmp_path: Path) -> None:
        """A missing folder is created; both files are written."""
        folder = tmp_path / "new" / "index"
        encoder.save(folder)
        assert (folder / StaticEncoder.TABLE_NAME).is_file()
        assert (folder / StaticEncoder.TOKENIZER_NAME).is_file()

    @pytest.mark.parametrize("missing", [StaticEncoder.TABLE_NAME,
                                         StaticEncoder.TOKENIZER_NAME])
    def test_missing_file(self, encoder: StaticEncoder, tmp_path: Path,
                          missing: str) -> None:
        """Either file missing: ``IndexNotFoundError``."""
        encoder.save(tmp_path)
        (tmp_path / missing).unlink()
        with pytest.raises(IndexNotFoundError, match="Run 'index' first"):
            StaticEncoder.load(tmp_path)

    @pytest.mark.parametrize("name, message", [
        (StaticEncoder.TABLE_NAME, "cannot read the embedding table"),
        (StaticEncoder.TOKENIZER_NAME, "cannot read the tokenizer"),
    ])
    def test_garbage_file(self, encoder: StaticEncoder, tmp_path: Path,
                          name: str, message: str) -> None:
        """A broken file: ``VectorIndexError``, never a Rust traceback."""
        encoder.save(tmp_path)
        (tmp_path / name).write_text("garbage", encoding="utf-8")
        with pytest.raises(VectorIndexError, match=message):
            StaticEncoder.load(tmp_path)

    def test_table_not_2d(self, encoder: StaticEncoder,
                          tmp_path: Path) -> None:
        """A valid ``.npy`` with the wrong shape: ``VectorIndexError``."""
        encoder.save(tmp_path)
        np.save(tmp_path / StaticEncoder.TABLE_NAME, np.zeros(5))
        with pytest.raises(VectorIndexError, match="corrupted"):
            StaticEncoder.load(tmp_path)

    def test_unwritable_folder(self, encoder: StaticEncoder,
                               tmp_path: Path) -> None:
        """A file where the folder should be: ``PathError``."""
        blocker = tmp_path / "file"
        blocker.write_text("x", encoding="utf-8")
        with pytest.raises(PathError):
            encoder.save(blocker)


class TestFromPretrained:
    """``from_pretrained``: cache first, download once, clean error."""

    @pytest.fixture
    def model_folder(self, tmp_path: Path) -> Path:
        """A folder laid out like a Hugging Face snapshot."""
        tiny_tokenizer().save(str(tmp_path / "tokenizer.json"))
        save_file({"embeddings": tiny_table()},
                  str(tmp_path / "model.safetensors"))
        return tmp_path

    def test_reads_the_snapshot(self, model_folder: Path,
                                monkeypatch: pytest.MonkeyPatch) -> None:
        """Tokenizer and ``embeddings`` table come from the snapshot."""
        asked: list[bool] = []

        def download(name: str, local_files_only: bool) -> str:
            """The cached snapshot; records how it was asked for."""
            asked.append(local_files_only)
            return str(model_folder)

        monkeypatch.setattr(src.static_encoder, "snapshot_download",
                            download)
        model = StaticEncoder.from_pretrained("some/model")
        assert asked == [True]
        assert model.encode(["server"])[0] == pytest.approx([0.6, 0.8, 0])

    def test_downloads_when_not_cached(self, model_folder: Path,
                                       monkeypatch: pytest.MonkeyPatch
                                       ) -> None:
        """Not in the cache: a second call is allowed to download."""
        asked: list[bool] = []

        def download(name: str, local_files_only: bool) -> str:
            """Fails from the cache, works with the download."""
            asked.append(local_files_only)
            if local_files_only:
                raise OSError("not cached")
            return str(model_folder)

        monkeypatch.setattr(src.static_encoder, "snapshot_download",
                            download)
        StaticEncoder.from_pretrained("some/model")
        assert asked == [True, False]

    def test_no_cache_no_network(self,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
        """Neither works: a ``RagError`` that says what to check."""
        def download(name: str, local_files_only: bool) -> str:
            """Always fails."""
            raise OSError("no network")

        monkeypatch.setattr(src.static_encoder, "snapshot_download",
                            download)
        with pytest.raises(RagError, match="Check the internet"):
            StaticEncoder.from_pretrained("some/model")


class TestRetriever:
    """``Retriever``: the abstract base of every retriever."""

    def test_cannot_instantiate(self) -> None:
        """``search`` is abstract: no plain ``Retriever``."""
        with pytest.raises(TypeError):
            Retriever()  # type: ignore[abstract]

    @pytest.mark.parametrize("cls", [BM25Index, HybridRetriever])
    def test_retrievers_are_subclasses(self, cls: type) -> None:
        """Both retrievers can be used wherever a ``Retriever`` is."""
        assert issubclass(cls, Retriever)
