"""Shared test setup: no test loads the real embedding model, and no
test writes into the project's ``data/output``.

``VectorIndex._load_model`` is replaced for every test by ``FakeEncoder``,
so ``index`` and the hybrid search run in milliseconds and without
the Hugging Face cache. Outputs saved to a relative folder (the defaults
of ``search`` and ``answer``) go under the test's ``tmp_path`` instead.

``RealData`` tells the tests about the real data, which is not in the
repository: tests that need it are skipped when it is missing.
"""
import functools
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pytest
from pydantic import BaseModel

from src.commands import RagCLI
from src.static_encoder import StaticEncoder
from src.vector_index import VectorIndex


class RealData:
    """The data a fresh clone does not have: the vLLM corpus and the
    question datasets.

    Their folders always exist (``.gitkeep`` files keep the layout the
    subject asks for), so the tests check what is inside them. The
    datasets are found by extension, never by name: the evaluation may
    bring other datasets than the public ones.

    Attributes:
        DATA: The project's ``data`` folder.
        CORPUS: The vLLM corpus, ``data/raw/vllm-0.10.1``.
        UNANSWERED: Folder of the datasets without answers.
        ANSWERED: Folder of the datasets with answers and sources.
    """

    DATA = Path(__file__).resolve().parent.parent / "data"
    CORPUS = DATA / "raw" / "vllm-0.10.1"
    UNANSWERED = DATA / "datasets" / "UnansweredQuestions"
    ANSWERED = DATA / "datasets" / "AnsweredQuestions"

    @classmethod
    def has_corpus(cls) -> bool:
        """True when the vLLM corpus is in ``data/raw``."""
        return cls.CORPUS.is_dir()

    @classmethod
    def unanswered_files(cls) -> list[Path]:
        """Every ``.json`` dataset without answers, sorted."""
        return sorted(cls.UNANSWERED.glob("*.json"))

    @classmethod
    def answered_files(cls) -> list[Path]:
        """Every ``.json`` dataset with answers, sorted."""
        return sorted(cls.ANSWERED.glob("*.json"))

    @classmethod
    def dataset_names(cls) -> list[str]:
        """Names of the datasets present in both folders, sorted."""
        return sorted({path.name for path in cls.unanswered_files()}
                      & {path.name for path in cls.answered_files()})


needs_corpus = pytest.mark.skipif(
    not RealData.has_corpus(),
    reason="vLLM corpus not present (data/raw/vllm-0.10.1)")
needs_answered = pytest.mark.skipif(
    not RealData.answered_files(),
    reason="no answered dataset in data/datasets/AnsweredQuestions")
needs_unanswered = pytest.mark.skipif(
    not RealData.unanswered_files(),
    reason="no dataset in data/datasets/UnansweredQuestions")


class FakeEncoder:
    """Stand-in for ``StaticEncoder`` with the same ``encode`` and ``save``.

    A text becomes a bag of its lowercase words, each hashed (with a
    fixed hash, not Python's per-run ``hash``) into one of ``DIMENSIONS``
    slots: texts that share words get similar vectors, which is enough
    to test the ranking logic.

    Attributes:
        DIMENSIONS: Length of every vector.
        WORD: What counts as a word.
        MARKER: File ``save`` writes, so the patched ``StaticEncoder.load``
            knows the folder holds the fake model.
        calls: Texts received by each ``encode`` call, in order.
    """

    DIMENSIONS = 64
    MARKER = "fake_encoder.txt"
    WORD = re.compile(r"[a-z0-9]+")

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.calls: list[list[str]] = []

    @staticmethod
    @functools.cache
    def _slot(word: str) -> int:
        """A stable slot for a word (weighted sum of its character codes).

        Cached: the corpus repeats the same words, and ``index`` on the
        real corpus sends ~60 000 windows here.
        """
        return (sum(ord(c) * (i + 1) for i, c in enumerate(word))
                % FakeEncoder.DIMENSIONS)

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        """One normalised float32 vector per text, like ``StaticEncoder``.

        Args:
            texts: The texts to embed.

        Returns:
            Array of shape ``(len(texts), DIMENSIONS)``; zeros for a text
            with no word.
        """
        self.calls.append(list(texts))
        rows: list[int] = []
        slots: list[int] = []
        for row, text in enumerate(texts):
            words = self.WORD.findall(text.lower())
            rows.extend([row] * len(words))
            slots.extend(map(self._slot, words))
        vectors = np.zeros((len(texts), self.DIMENSIONS), dtype=np.float32)
        np.add.at(vectors, (rows, slots), 1.0)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors /= np.where(norms == 0, 1.0, norms)
        return vectors

    def save(self, files_dir: Path) -> None:
        """Leave a marker where ``StaticEncoder`` would save its files.

        Args:
            files_dir: The index folder.
        """
        files_dir.mkdir(parents=True, exist_ok=True)
        (files_dir / self.MARKER).write_text("fake", encoding="utf-8")


@pytest.fixture(autouse=True)
def fake_encoder(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeEncoder]:
    """Make every ``VectorIndex`` use a fresh ``FakeEncoder``.

    ``build`` gets it instead of downloading the model, and
    ``StaticEncoder.load`` returns it for a folder the fake saved into;
    any other folder (the ``StaticEncoder`` tests) is read for real.
    """
    encoder = FakeEncoder()
    real_load = StaticEncoder.load

    def load(cls: type[StaticEncoder],
             files_dir: Path) -> StaticEncoder | FakeEncoder:
        """The fake model where it was saved, the real loader elsewhere."""
        if (files_dir / FakeEncoder.MARKER).is_file():
            return encoder
        return real_load(files_dir)

    monkeypatch.setattr(VectorIndex, "_load_model",
                        classmethod(lambda cls: encoder))
    monkeypatch.setattr(StaticEncoder, "load", classmethod(load))
    yield encoder


@pytest.fixture(autouse=True)
def outputs_in_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Send outputs saved to a relative folder under ``tmp_path``.

    ``search`` and ``answer`` save by default to ``data/output/...``;
    without this, every test that runs them would leave a file in the
    project. Absolute folders (``tmp_path`` given by a test) are kept.
    """
    real_save = RagCLI._save_json

    def save(save_directory: str, file_name: str,
             output: BaseModel) -> Path:
        """Save as usual, moving a relative folder into ``tmp_path``."""
        if not Path(save_directory).is_absolute():
            save_directory = str(tmp_path / save_directory)
        return real_save(save_directory, file_name, output)

    monkeypatch.setattr(RagCLI, "_save_json", staticmethod(save))
