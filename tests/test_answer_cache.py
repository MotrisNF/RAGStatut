"""Tests for ``src.answer_cache.AnswerCache`` and its CLI use.

The key (question and context), the namespace (the text of
``generator.py``), the round trip to disk and the files that mean an
empty cache. Nothing here loads the model: checking the cache must never
import ``src.generator``. The CLI tests put a fake ``src.generator`` in
``sys.modules`` (as ``test_api.py`` does), so PyTorch is never imported.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from src.answer_cache import AnswerCache
from src.bm25_index import BM25Index
from src.commands import RagCLI
from src.exceptions import RagError
from src.models import (NO_SOURCES, AnswerCacheData, Chunk,
                        MinimalSearchResults, MinimalSource,
                        StudentSearchResults)
from src.reader import CorpusReader


def chunk(text: str = "hello", path: str = "data/raw/a.md",
          start: int = 0) -> Chunk:
    """A chunk of ``path`` starting at ``start``."""
    return Chunk(file_path=path, first_character_index=start,
                 last_character_index=start + len(text), text=text)


A = chunk("Load a LoRA adapter.", "data/raw/a.md")
B = chunk("Start the server.", "data/raw/b.md")


class TestKey:
    """``key``: changes with anything that goes into the prompt."""

    def test_hexadecimal_sha256(self) -> None:
        """64 hexadecimal characters."""
        key = AnswerCache.key("What?", [A])
        assert len(key) == 64
        assert set(key) <= set("0123456789abcdef")

    def test_same_input_same_key(self) -> None:
        """Repeated calls give the same key (no randomness, no order of a
        set or dict)."""
        assert AnswerCache.key("What?", [A, B]) == AnswerCache.key(
            "What?", [chunk(A.text, A.file_path), B])

    def test_question_changes_it(self) -> None:
        """Another question, another key."""
        assert AnswerCache.key("What?", [A]) != AnswerCache.key("Why?", [A])

    def test_chunk_order_changes_it(self) -> None:
        """The order of the chunks in the prompt changes the answer, so
        it changes the key."""
        assert AnswerCache.key("What?", [A, B]) != AnswerCache.key(
            "What?", [B, A])

    def test_text_changes_it(self) -> None:
        """Same location, edited text: another key, never a stale
        answer."""
        edited = chunk("Load a LoRA adapter now.", A.file_path)
        assert AnswerCache.key("What?", [A]) != AnswerCache.key(
            "What?", [edited])

    @pytest.mark.parametrize("other", [
        chunk("Load a LoRA adapter.", "data/raw/other.md"),
        chunk("Load a LoRA adapter.", "data/raw/a.md", start=7),
    ])
    def test_location_changes_it(self, other: Chunk) -> None:
        """Same text in another file or at another range: another key."""
        assert AnswerCache.key("What?", [A]) != AnswerCache.key(
            "What?", [other])

    def test_no_chunks(self) -> None:
        """An empty context still gives a valid key."""
        assert len(AnswerCache.key("What?", [])) == 64

    def test_unicode(self) -> None:
        """Non-ASCII text is fine and is told apart."""
        assert AnswerCache.key("¿Qué?", [A]) != AnswerCache.key("Que?", [A])


class TestNamespace:
    """``namespace``: the hash of the text of ``generator.py``."""

    def test_stable(self) -> None:
        """The same file gives the same namespace."""
        assert AnswerCache.namespace() == AnswerCache.namespace()
        assert len(AnswerCache.namespace()) == 64

    def test_changes_with_generator(self, tmp_path: Path,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
        """Editing the generator (a prompt, a constant) gives another
        namespace."""
        source = tmp_path / "generator.py"
        monkeypatch.setattr(AnswerCache, "GENERATOR_FILE", source)
        source.write_text("PROMPT = 'a'\n", encoding="utf-8")
        before = AnswerCache.namespace()
        source.write_text("PROMPT = 'b'\n", encoding="utf-8")
        assert AnswerCache.namespace() != before

    def test_points_to_the_real_generator(self) -> None:
        """``GENERATOR_FILE`` is the project's generator."""
        assert AnswerCache.GENERATOR_FILE.name == "generator.py"
        assert AnswerCache.GENERATOR_FILE.is_file()

    def test_never_imports_the_generator(self) -> None:
        """Building a cache, computing a key and reading an answer never
        import ``src.generator`` or PyTorch (in a fresh interpreter, so
        other tests cannot hide it)."""
        code = (
            "import sys, tempfile, pathlib\n"
            "from src.answer_cache import AnswerCache\n"
            "from src.models import Chunk\n"
            "c = Chunk(file_path='a.md', first_character_index=0,"
            " last_character_index=1, text='x')\n"
            "path = pathlib.Path(tempfile.mkdtemp()) / 'answer_cache.json'\n"
            "cache = AnswerCache(path)\n"
            "cache.get(AnswerCache.key('q', [c]))\n"
            "print('src.generator' in sys.modules, 'torch' in sys.modules)\n"
        )
        result = subprocess.run([sys.executable, "-c", code],
                                capture_output=True, text=True, check=True,
                                cwd=CorpusReader.PROJECT_ROOT)
        assert result.stdout.strip() == "False False"


class TestStorage:
    """``get``, ``put``, ``flush`` and the files that mean an empty
    cache."""

    @pytest.fixture
    def path(self, tmp_path: Path) -> Path:
        """Where the cache file goes."""
        return AnswerCache.file_path(tmp_path)

    def test_file_path(self, tmp_path: Path) -> None:
        """``answer_cache.json`` inside the index folder."""
        assert AnswerCache.file_path(tmp_path) == (
            tmp_path / "answer_cache.json")

    def test_missing_file_empty(self, path: Path) -> None:
        """No file yet: an empty cache, no error."""
        assert AnswerCache(path).get("k") is None

    def test_put_get(self, path: Path) -> None:
        """An answer put is returned by ``get``."""
        cache = AnswerCache(path)
        cache.put("k", "An answer.")
        assert cache.get("k") == "An answer."
        assert cache.get("other") is None

    def test_round_trip(self, path: Path) -> None:
        """Saved by ``flush``, read by a new cache (the next command)."""
        cache = AnswerCache(path)
        cache.put("k", "Respuesta con tilde: sí.")
        cache.flush()
        assert AnswerCache(path).get("k") == "Respuesta con tilde: sí."

    def test_nothing_saved_without_flush(self, path: Path) -> None:
        """``put`` alone does not write."""
        AnswerCache(path).put("k", "An answer.")
        assert not path.exists()

    def test_flush_without_changes_writes_nothing(self, path: Path) -> None:
        """Nothing new: the file is not written again."""
        cache = AnswerCache(path)
        cache.put("k", "An answer.")
        cache.flush()
        os.utime(path, ns=(1, 1))
        AnswerCache(path).flush()
        cache.flush()
        assert path.stat().st_mtime_ns == 1

    def test_keeps_earlier_answers(self, path: Path) -> None:
        """A second command adds to the answers of the first."""
        first = AnswerCache(path)
        first.put("k1", "One.")
        first.flush()
        second = AnswerCache(path)
        second.put("k2", "Two.")
        second.flush()
        third = AnswerCache(path)
        assert (third.get("k1"), third.get("k2")) == ("One.", "Two.")

    @pytest.mark.parametrize("content", [
        "", "not json", "{}", '{"namespace": "x"}',
        '{"namespace": "x", "entries": {"k": 1}}'])
    def test_broken_file_empty(self, path: Path, content: str) -> None:
        """A broken or invalid file is an empty cache, not an error."""
        path.write_text(content, encoding="utf-8")
        assert AnswerCache(path).get("k") is None

    def test_not_utf8_empty(self, path: Path) -> None:
        """A file that is not UTF-8 is an empty cache."""
        path.write_bytes(b"\xff\xfe\x00")
        assert AnswerCache(path).get("k") is None

    def test_other_namespace_empty(self, path: Path) -> None:
        """Answers of another generator are ignored."""
        path.write_text(AnswerCacheData(
            namespace="another generator",
            entries={"k": "Old answer."}).model_dump_json(),
            encoding="utf-8")
        assert AnswerCache(path).get("k") is None

    def test_other_namespace_replaced(self, path: Path) -> None:
        """Saving over a foreign file keeps only the current answers,
        with the current namespace."""
        path.write_text(AnswerCacheData(
            namespace="another generator",
            entries={"old": "Old answer."}).model_dump_json(),
            encoding="utf-8")
        cache = AnswerCache(path)
        cache.put("k", "New.")
        cache.flush()
        data = AnswerCacheData.model_validate_json(
            path.read_text(encoding="utf-8"))
        assert data.namespace == AnswerCache.namespace()
        assert data.entries == {"k": "New."}

    def test_no_tmp_left(self, path: Path) -> None:
        """The temporary file is moved over the cache, not left behind."""
        cache = AnswerCache(path)
        cache.put("k", "An answer.")
        cache.flush()
        assert not path.with_suffix(".tmp").exists()

    def test_unwritable_folder_warns(self, tmp_path: Path,
                                     capsys: pytest.CaptureFixture[str]
                                     ) -> None:
        """A cache that cannot be written prints a warning, no error."""
        cache = AnswerCache(tmp_path / "no" / "answer_cache.json")
        cache.put("k", "An answer.")
        cache.flush()
        assert "WARNING: cannot save the answer cache" in (
            capsys.readouterr().err)

    def test_failed_replace_keeps_old_file(self, path: Path,
                                           monkeypatch: pytest.MonkeyPatch
                                           ) -> None:
        """If the final move fails, the previous cache is still whole."""
        cache = AnswerCache(path)
        cache.put("k", "An answer.")
        cache.flush()
        before = path.read_bytes()

        def broken(src: object, dst: object) -> None:
            """Like a move interrupted by the system."""
            raise OSError("interrupted")

        monkeypatch.setattr(os, "replace", broken)
        cache.put("k2", "Another.")
        cache.flush()
        assert path.read_bytes() == before


class FakeGenerator:
    """Stand-in for ``Generator``: counts loads and generations.

    Attributes:
        MODEL_NAME: Printed by ``_load_generator``.
        loads: ``dtype`` of every load, shared by all instances.
        calls: Every ``generate`` call, shared by all instances.
    """

    MODEL_NAME = "fake-model"
    loads: list[str] = []
    calls: list[tuple[str, list[Chunk]]] = []

    def __init__(self, dtype: str) -> None:
        """Record the load."""
        FakeGenerator.loads.append(dtype)

    def generate(self, question: str, chunks: list[Chunk]) -> str:
        """Record the call; the answer names the question."""
        FakeGenerator.calls.append((question, chunks))
        return f"Answer to {question}"


class TestAnswerCommand:
    """``answer`` with ``--cache`` (on by default)."""

    TEXTS = ["Load a LoRA adapter at runtime with the API.",
             "Configure the OpenAI compatible server."]

    @pytest.fixture(autouse=True)
    def generator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A fake ``src.generator`` module with fresh counters."""
        module = ModuleType("src.generator")
        module.Generator = FakeGenerator  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "src.generator", module)
        monkeypatch.setattr(FakeGenerator, "loads", [])
        monkeypatch.setattr(FakeGenerator, "calls", [])

    @pytest.fixture
    def index_dir(self, tmp_path: Path,
                  capsys: pytest.CaptureFixture[str]) -> Path:
        """A small saved BM25 index (its progress bar is discarded)."""
        chunks = [chunk(text, f"data/raw/{i}.md")
                  for i, text in enumerate(self.TEXTS)]
        index_dir = tmp_path / "index"
        BM25Index.build(chunks).save(index_dir)
        capsys.readouterr()
        return index_dir

    @staticmethod
    def answer(index_dir: Path, query: str = "lora adapter",
               cache: bool = True) -> None:
        """Run ``answer`` with ``bm25`` (no vector index here)."""
        RagCLI().answer(query=query, k=2, index_dir=str(index_dir),
                        retriever="bm25", cache=cache,
                        save_directory=str(index_dir.parent / "out"))

    @staticmethod
    def printed_answer(capsys: pytest.CaptureFixture[str]) -> str:
        """The answer in the JSON the command printed."""
        out = json.loads(capsys.readouterr().out)
        return str(out["search_results"][0]["answer"])

    def test_second_answer_from_cache(self, index_dir: Path,
                                      capsys: pytest.CaptureFixture[str]
                                      ) -> None:
        """The same question twice: the model is loaded and called once,
        and the second run says it read the cache."""
        self.answer(index_dir)
        first = self.printed_answer(capsys)
        self.answer(index_dir)
        captured = capsys.readouterr()
        assert len(FakeGenerator.loads) == 1
        assert len(FakeGenerator.calls) == 1
        assert "Answer read from the cache" in captured.err
        assert "Loading PyTorch" not in captured.err
        assert json.loads(captured.out)["search_results"][0][
            "answer"] == first
        assert AnswerCache.file_path(index_dir).is_file()

    def test_cache_false_always_generates(self, index_dir: Path) -> None:
        """``--cache=False``: generated every time, no answer file."""
        self.answer(index_dir, cache=False)
        self.answer(index_dir, cache=False)
        assert len(FakeGenerator.calls) == 2
        assert not AnswerCache.file_path(index_dir).exists()

    def test_cache_false_ignores_saved_answer(self, index_dir: Path) -> None:
        """A saved answer is not used with ``--cache=False``."""
        self.answer(index_dir)
        self.answer(index_dir, cache=False)
        assert len(FakeGenerator.calls) == 2

    def test_other_question_generates(self, index_dir: Path) -> None:
        """Another question is another key."""
        self.answer(index_dir, query="lora adapter")
        self.answer(index_dir, query="lora adapter at runtime")
        assert len(FakeGenerator.calls) == 2

    def test_generator_change_ignores_answers(
            self, index_dir: Path, tmp_path: Path,
            monkeypatch: pytest.MonkeyPatch) -> None:
        """Editing ``generator.py`` makes the saved answers stale."""
        source = tmp_path / "generator.py"
        monkeypatch.setattr(AnswerCache, "GENERATOR_FILE", source)
        source.write_text("PROMPT = 'a'\n", encoding="utf-8")
        self.answer(index_dir)
        source.write_text("PROMPT = 'b'\n", encoding="utf-8")
        self.answer(index_dir)
        assert len(FakeGenerator.calls) == 2

    def test_no_sources_not_cached(self, index_dir: Path,
                                   capsys: pytest.CaptureFixture[str]
                                   ) -> None:
        """No hit: no model and no cache file, as before."""
        self.answer(index_dir, query="zzzz qqqq")
        assert self.printed_answer(capsys) == NO_SOURCES
        assert FakeGenerator.loads == []
        assert not AnswerCache.file_path(index_dir).exists()

    def test_saved_before_output(self, index_dir: Path,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
        """The answer is cached even if writing the output fails, so a
        retry does not generate again."""
        def broken(save_directory: str, file_name: str,
                   output: object) -> Path:
            """Fail like a full disk."""
            raise RagError("cannot write the output")

        monkeypatch.setattr(RagCLI, "_save_json", staticmethod(broken))
        with pytest.raises(RagError):
            self.answer(index_dir)
        assert AnswerCache.file_path(index_dir).is_file()

    def test_pytorch_missing_not_needed_on_hit(
            self, index_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cached answer works even if PyTorch cannot be imported."""
        self.answer(index_dir)
        monkeypatch.setitem(sys.modules, "src.generator", None)
        self.answer(index_dir)
        assert len(FakeGenerator.calls) == 1

    def test_pytorch_missing_on_miss(self, index_dir: Path,
                                     monkeypatch: pytest.MonkeyPatch
                                     ) -> None:
        """Not cached and no PyTorch: the usual clear error."""
        monkeypatch.setitem(sys.modules, "src.generator", None)
        with pytest.raises(RagError, match="Cannot load PyTorch"):
            self.answer(index_dir)


class TestAnswerDatasetCommand:
    """``answer_dataset`` with ``--cache``: only the questions not cached
    are generated, and the model is loaded only if one is missing."""

    QUESTIONS = ["How do I load a LoRA adapter?", "How do I start it?",
                 "Which port?", "Nothing found?"]

    @pytest.fixture(autouse=True)
    def generator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A fake ``src.generator`` module with fresh counters."""
        module = ModuleType("src.generator")
        module.Generator = FakeGenerator  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "src.generator", module)
        monkeypatch.setattr(FakeGenerator, "loads", [])
        monkeypatch.setattr(FakeGenerator, "calls", [])

    @pytest.fixture
    def results_path(self, tmp_path: Path,
                     monkeypatch: pytest.MonkeyPatch) -> Path:
        """A ``search_dataset`` output over two corpus files in a project
        root at ``tmp_path``; the last question has no source. The index
        folder, where the answer cache goes, exists (as after ``index``).
        """
        monkeypatch.setattr(CorpusReader, "PROJECT_ROOT", tmp_path)
        (tmp_path / "index").mkdir()
        (tmp_path / "raw").mkdir()
        (tmp_path / "raw" / "a.md").write_text(
            "Load a LoRA adapter with the API.", encoding="utf-8")
        (tmp_path / "raw" / "b.md").write_text(
            "Start the server on port 8000.", encoding="utf-8")
        a = MinimalSource(file_path="raw/a.md", first_character_index=0,
                          last_character_index=33)
        b = MinimalSource(file_path="raw/b.md", first_character_index=0,
                          last_character_index=30)
        sources = [[a, b], [b], [b, a], []]
        results = StudentSearchResults(k=2, search_results=[
            MinimalSearchResults(question_id=f"q{i}", question=question,
                                 retrieved_sources=found)
            for i, (question, found) in enumerate(
                zip(self.QUESTIONS, sources))])
        path = tmp_path / "results.json"
        path.write_text(results.model_dump_json(), encoding="utf-8")
        return path

    @staticmethod
    def run(results_path: Path, cache: bool = True) -> list[str]:
        """Run ``answer_dataset`` and return the answers, in order."""
        out = results_path.parent / "out"
        RagCLI().answer_dataset(
            student_search_results_path=results_path,
            save_directory=str(out), cache=cache,
            index_dir=str(results_path.parent / "index"))
        data = json.loads((out / results_path.name).read_text(
            encoding="utf-8"))
        return [entry["answer"] for entry in data["search_results"]]

    @staticmethod
    def cache_path(results_path: Path) -> Path:
        """Where the runs save the answer cache."""
        return AnswerCache.file_path(results_path.parent / "index")

    def test_first_run_generates_with_sources(self,
                                              results_path: Path) -> None:
        """The first run generates every question with sources, loads
        the model once, and saves the cache; no source: ``NO_SOURCES``."""
        answers = self.run(results_path)
        assert len(FakeGenerator.loads) == 1
        assert [q for q, _ in FakeGenerator.calls] == self.QUESTIONS[:3]
        assert answers[3] == NO_SOURCES
        assert self.cache_path(results_path).is_file()

    def test_second_run_never_loads(self, results_path: Path,
                                    capsys: pytest.CaptureFixture[str]
                                    ) -> None:
        """Everything cached: same answers, no model, no PyTorch."""
        first = self.run(results_path)
        capsys.readouterr()
        FakeGenerator.loads.clear()
        assert self.run(results_path) == first
        err = capsys.readouterr().err
        assert FakeGenerator.loads == []
        assert "Loading PyTorch" not in err
        assert "4 of 4 answers read from the cache" in err

    def test_hit_works_without_pytorch(self, results_path: Path,
                                       monkeypatch: pytest.MonkeyPatch
                                       ) -> None:
        """All cached: it works even if PyTorch cannot be imported."""
        first = self.run(results_path)
        monkeypatch.setitem(sys.modules, "src.generator", None)
        assert self.run(results_path) == first

    def test_only_missing_generated(self, results_path: Path) -> None:
        """Half cached: only the missing questions reach the model, and
        the output keeps the order of the input."""
        cache = AnswerCache(self.cache_path(results_path))
        chunk_a = Chunk(file_path="raw/a.md", first_character_index=0,
                        last_character_index=33,
                        text="Load a LoRA adapter with the API.")
        chunk_b = Chunk(file_path="raw/b.md", first_character_index=0,
                        last_character_index=30,
                        text="Start the server on port 8000.")
        cache.put(AnswerCache.key(self.QUESTIONS[0], [chunk_a, chunk_b]),
                  "Cached answer.")
        cache.flush()

        answers = self.run(results_path)
        assert [q for q, _ in FakeGenerator.calls] == self.QUESTIONS[1:3]
        assert answers == ["Cached answer.",
                           f"Answer to {self.QUESTIONS[1]}",
                           f"Answer to {self.QUESTIONS[2]}", NO_SOURCES]

    def test_cache_false_generates_all(self, results_path: Path) -> None:
        """``--cache=False``: everything is generated, nothing saved."""
        self.run(results_path)
        FakeGenerator.calls.clear()
        self.run(results_path, cache=False)
        assert len(FakeGenerator.calls) == 3

    def test_cache_false_writes_nothing(self, results_path: Path) -> None:
        """``--cache=False`` does not create the cache file."""
        self.run(results_path, cache=False)
        assert not self.cache_path(results_path).exists()

    def test_edited_file_generates_again(self, results_path: Path) -> None:
        """A source file edited after the search: its chunk text changes,
        so the answers that used it are generated again."""
        self.run(results_path)
        FakeGenerator.calls.clear()
        (results_path.parent / "raw" / "a.md").write_text(
            "Load a LoRA adapter with the CLI.", encoding="utf-8")
        self.run(results_path)
        regenerated = [q for q, _ in FakeGenerator.calls]
        assert regenerated == [self.QUESTIONS[0], self.QUESTIONS[2]]

    def test_interrupted_run_keeps_answers(
            self, results_path: Path,
            monkeypatch: pytest.MonkeyPatch) -> None:
        """Ctrl+C after the first answer (fewer than ``FLUSH_EVERY``):
        that answer is saved, and the next run only generates the rest.
        """
        calls = 0
        real_generate = FakeGenerator.generate

        def interrupted(generator: FakeGenerator, question: str,
                        chunks: list[Chunk]) -> str:
            """Answer once, then act as if Ctrl+C was pressed."""
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt
            return real_generate(generator, question, chunks)

        monkeypatch.setattr(FakeGenerator, "generate", interrupted)
        with pytest.raises(KeyboardInterrupt):
            self.run(results_path)
        assert self.cache_path(results_path).is_file()

        monkeypatch.setattr(FakeGenerator, "generate", real_generate)
        FakeGenerator.calls.clear()
        self.run(results_path)
        assert [q for q, _ in FakeGenerator.calls] == self.QUESTIONS[1:3]

    def test_saved_every_few_answers(self, results_path: Path,
                                     monkeypatch: pytest.MonkeyPatch
                                     ) -> None:
        """With ``FLUSH_EVERY`` answers generated, the cache is saved
        before the end: an interrupted run keeps them."""
        monkeypatch.setattr(RagCLI, "FLUSH_EVERY", 2)
        saved: list[int] = []
        real_flush = AnswerCache.flush

        def counting(cache: AnswerCache) -> None:
            """Record how many answers each save holds."""
            saved.append(len(cache._entries))
            real_flush(cache)

        monkeypatch.setattr(AnswerCache, "flush", counting)
        self.run(results_path)
        assert saved == [2, 3]
