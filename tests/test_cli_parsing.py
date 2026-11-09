"""Tests for the command-line argument parsing of ``RagCLI``.

Each test calls Fire exactly as the terminal would, passing the arguments
as a list. They only check parsing and validation, not what each command
does. They use the mini corpus and datasets in ``tests/data``, which are
in the repository, so they never depend on the real data.
"""
import json
from pathlib import Path

import fire
import pytest
from src.bm25_index import BM25Index
from src.commands import (DATASET_ANSWERS, DATASET_QUESTIONS, SEARCH_RESULTS,
                          RagCLI)
from src.exceptions import ArgError
from src.models import Chunk

MINI_RAW = "tests/data/raw"
DOCS_DATASET = "tests/data/datasets/UnansweredQuestions/dataset_mini.json"
DOCS_ANSWERED = "tests/data/datasets/AnsweredQuestions/dataset_mini.json"
MISSING_FILE = "does_not_exist.json"
A_DIRECTORY = "data"


class TestValidArguments:
    """Arguments that must be accepted without raising."""

    @pytest.mark.parametrize("command", [
        ["index"],
        ["index", "--max_chunk_size", "500"],
        ["index", "--raw_dir", MINI_RAW],
        ["search", "How to use LoRA?"],
        ["search", "How to use LoRA?", "--k", "5"],
        ["search", "How to use LoRA?", "--k=5"],
        ["search", "123"],
        ["search_dataset", "--dataset_path", DOCS_DATASET],
        ["search_dataset", "--dataset_path", DOCS_DATASET, "--k", "10",
         "--save_directory", "folder/that/does/not/exist/yet"],
        ["answer", "How to use LoRA?", "--k", "3"],
        ["answer_dataset", "--student_search_results_path",
         "folder/results.json"],
        ["answer_dataset", "--student_search_results_path",
         "folder/results.json",
         "--save_directory", "folder/that/does/not/exist/yet"],
        ["evaluate", "--student_search_results_path", "folder/results.json",
         "--dataset_path", DOCS_ANSWERED],
        ["evaluate", "--student_search_results_path", "folder/results.json",
         "--dataset_path", DOCS_ANSWERED, "--k", "5"],
    ])
    def test_accepted(self, command: list[str], tmp_path: Path) -> None:
        """Fire runs the command and pydantic accepts every value.

        Commands not written yet raise ``NotImplementedError`` from their
        body: reaching the body means validation already passed. ``index``
        reads the mini corpus (unless the command names a folder) and
        saves into a temporary folder, never into ``data/processed``, and
        ``search_dataset`` writes its results into a temporary folder too.
        ``evaluate`` and ``answer_dataset`` get a small valid results file
        (no questions, so ``answer_dataset`` never loads the model), and
        ``answer`` an index where the question matches nothing (no
        sources, so the model is not loaded either). The searches get
        only a BM25 index, so they use ``--retriever bm25``.
        """
        if command[0] in ("evaluate", "answer_dataset"):
            (tmp_path / "folder").mkdir()
            (tmp_path / "folder" / "results.json").write_text(
                '{"search_results": [], "k": 10}', encoding="utf-8")
        command = [str(tmp_path / arg) if arg.startswith("folder/") else arg
                   for arg in command]
        if command[0] == "index":
            if "--raw_dir" not in command:
                command = [*command, "--raw_dir", MINI_RAW]
            command = [*command, "--index_dir", str(tmp_path)]
        if command[0] == "search_dataset":
            if "--save_directory" not in command:
                command = [*command, "--save_directory", str(tmp_path)]
        if command[0] == "answer_dataset":
            if "--save_directory" not in command:
                command = [*command, "--save_directory", str(tmp_path)]
        if command[0] == "answer":
            chunk = Chunk(file_path="data/raw/a.md", first_character_index=0,
                          last_character_index=12, text="Nothing here")
            BM25Index.build([chunk]).save(tmp_path)
            command = [*command, "--index_dir", str(tmp_path),
                       "--retriever", "bm25"]
        if command[0] in ("search", "search_dataset"):
            chunk = Chunk(file_path="data/raw/a.md", first_character_index=0,
                          last_character_index=15, text="How to use LoRA")
            BM25Index.build([chunk]).save(tmp_path)
            command = [*command, "--index_dir", str(tmp_path),
                       "--retriever", "bm25"]
        try:
            fire.Fire(RagCLI, command=command)
        except NotImplementedError:
            pass


class TestInvalidValues:
    """Values of the right argument but wrong content.

    pydantic rejects them and the command raises ``ArgError``. Any other
    exception (for example ``TypeError``) makes the test fail.
    """

    @pytest.mark.parametrize("command", [
        # Empty or blank queries
        ["search", ""],
        ["search", "   "],
        ["answer", ""],
        # Invalid k
        ["search", "hola", "--k", "0"],
        ["search", "hola", "--k", "-3"],
        ["search", "hola", "--k", "abc"],
        ["search", "hola", "--k", "3.7"],
        ["answer", "hola", "--k", "0"],
        ["search_dataset", "--dataset_path", DOCS_DATASET, "--k", "0"],
        ["evaluate", "--student_search_results_path", DOCS_DATASET,
         "--dataset_path", DOCS_ANSWERED, "--k", "0"],
        # Invalid max_chunk_size
        ["index", "--max_chunk_size", "0"],
        ["index", "--max_chunk_size", "-100"],
        ["index", "--max_chunk_size", "big"],
        # Input files that do not exist or are directories
        ["search_dataset", "--dataset_path", MISSING_FILE],
        ["search_dataset", "--dataset_path", A_DIRECTORY],
        ["answer_dataset", "--student_search_results_path", MISSING_FILE],
        ["evaluate", "--student_search_results_path", MISSING_FILE,
         "--dataset_path", DOCS_ANSWERED],
        ["evaluate", "--student_search_results_path", DOCS_DATASET,
         "--dataset_path", MISSING_FILE],
        # Empty directories
        ["search_dataset", "--dataset_path", DOCS_DATASET,
         "--save_directory", ""],
        ["index", "--raw_dir", ""],
        ["index", "--raw_dir", "   "],
    ])
    def test_rejected(self, command: list[str]) -> None:
        """The command raises ``ArgError``."""
        with pytest.raises(ArgError):
            fire.Fire(RagCLI, command=command)


class TestWrongUsage:
    """Missing or unknown arguments.

    Fire prints the usage and exits with code 2 (``FireExit``, a
    ``SystemExit``) before pydantic is reached.
    """

    @pytest.mark.parametrize("command", [
        ["search"],
        ["answer"],
        ["not_a_command"],
    ])
    def test_usage_error(self, command: list[str]) -> None:
        """Fire exits with a usage error."""
        with pytest.raises(SystemExit):
            fire.Fire(RagCLI, command=command)


class TestDefaultPaths:
    """The dataset commands run without options: their input files have
    defaults (the public docs dataset and what ``search_dataset`` writes
    from it).

    The tests run from an empty folder, where the default files do not
    exist: the command must stop with an ``ArgError`` naming the default
    path, before anything is loaded (``answer_dataset`` never reaches the
    model).
    """

    @pytest.mark.parametrize("command, option, default", [
        (["search_dataset"], "--dataset_path", str(DATASET_QUESTIONS)),
        (["answer_dataset"], "--student_search_results_path",
         str(SEARCH_RESULTS)),
        (["evaluate", "--dataset_path", str(Path.cwd() / DOCS_ANSWERED)],
         "--student_search_results_path", str(SEARCH_RESULTS)),
        (["evaluate", "--student_search_results_path",
          str(Path.cwd() / DOCS_DATASET)], "--dataset_path",
         str(DATASET_ANSWERS)),
    ])
    def test_missing_default_file(self, command: list[str], option: str,
                                  default: str, tmp_path: Path,
                                  monkeypatch: pytest.MonkeyPatch) -> None:
        """A default that does not exist is an argument error that names
        the option and the default path, not a crash."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ArgError) as error:
            fire.Fire(RagCLI, command=command)
        message = str(error.value)
        assert option in message
        assert default in message

    def test_default_used(self, tmp_path: Path,
                          monkeypatch: pytest.MonkeyPatch,
                          capsys: pytest.CaptureFixture[str]) -> None:
        """``evaluate`` with no option reads the two default files (here
        the mini dataset, copied to the default paths)."""
        root = Path.cwd()
        monkeypatch.chdir(tmp_path)
        for source, default in ((DOCS_DATASET, DATASET_QUESTIONS),
                                (DOCS_ANSWERED, DATASET_ANSWERS)):
            target = tmp_path / default
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / source).read_bytes())
        results = tmp_path / SEARCH_RESULTS
        results.parent.mkdir(parents=True)
        dataset = json.loads((root / DOCS_DATASET).read_text("utf-8"))
        results.write_text(json.dumps({"k": 10, "search_results": [
            {"question_id": q["question_id"], "question": q["question"],
             "retrieved_sources": []}
            for q in dataset["rag_questions"]]}), encoding="utf-8")
        fire.Fire(RagCLI, command=["evaluate"])
        assert "Recall@1: 0.000" in capsys.readouterr().out


class TestUnknownFlag:
    """Known Fire behaviour: an unknown flag is reported only AFTER the
    command has run (Fire calls the method with the arguments it
    understood, then fails on the leftover ones).
    """

    def test_command_runs_before_usage_error(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The method is called once, then Fire exits with code 2."""
        calls: list[tuple[str, int]] = []

        def spy(self: RagCLI, query: str, k: int = 10) -> None:
            """Record the call instead of searching."""
            calls.append((query, k))

        monkeypatch.setattr(RagCLI, "search", spy)
        with pytest.raises(SystemExit) as info:
            fire.Fire(RagCLI, command=["search", "hola", "--unknown", "5"])
        assert info.value.code == 2
        assert calls == [("hola", 10)]
