"""Tests for the ``CliCommand.validated`` decorator in ``src.commands``."""
import inspect
from pathlib import Path
from typing import Annotated, get_origin

import pytest
from pydantic import Field, FilePath, ValidationError

from src.commands import CliCommand, RagCLI
from src.exceptions import ArgError, CorpusError, DatasetError, PathError
from src.models import MinimalSource, RagDataset, StudentSearchResults

COMMANDS = ["index", "search", "search_dataset", "answer",
            "answer_dataset", "evaluate"]


class Sample:
    """A small class decorated like ``RagCLI``."""

    @CliCommand.validated
    def run(self, query: Annotated[str, Field(min_length=1)],
            k: Annotated[int, Field(gt=0)] = 10,
            path: FilePath | None = None) -> str:
        """Return the received values.

        Args:
            query: Text.
            k: Number.
            path: Optional file.
        """
        return f"{query}|{k}|{path}"


class TestSignature:
    """What Fire sees when it inspects a command."""

    def test_annotated_removed(self) -> None:
        """``Annotated[int, ...]`` is shown as plain ``int``."""
        params = inspect.signature(Sample.run).parameters
        assert params["query"].annotation is str
        assert params["k"].annotation is int

    def test_other_annotations_kept(self) -> None:
        """Annotations without ``Annotated`` are left as they are."""
        params = inspect.signature(Sample.run).parameters
        assert params["path"].annotation == FilePath | None

    def test_defaults_kept(self) -> None:
        """Default values survive the decoration."""
        params = inspect.signature(Sample.run).parameters
        assert params["k"].default == 10
        assert params["query"].default is inspect.Parameter.empty

    def test_name_and_docstring_kept(self) -> None:
        """Fire uses them for the help text."""
        assert Sample.run.__name__ == "run"
        assert (Sample.run.__doc__ or "").startswith("Return the received")

    def test_raw_function_removed(self) -> None:
        """``validate_call``'s helper attribute is not listed by Fire."""
        assert not hasattr(Sample.run, "raw_function")

    @pytest.mark.parametrize("command", COMMANDS)
    def test_rag_cli_signatures_clean(self, command: str) -> None:
        """No command of ``RagCLI`` exposes ``Annotated`` to Fire."""
        params = inspect.signature(getattr(RagCLI, command)).parameters
        assert all(get_origin(p.annotation) is not Annotated
                   for p in params.values())


class TestValidation:
    """pydantic still validates with the full annotations."""

    def test_valid(self) -> None:
        """Correct values reach the method."""
        assert Sample().run("hi", 3) == "hi|3|None"

    def test_constraints_enforced(self) -> None:
        """``Field`` constraints still apply."""
        with pytest.raises(ArgError):
            Sample().run("hi", 0)
        with pytest.raises(ArgError):
            Sample().run("")

    def test_numbers_to_str(self) -> None:
        """``coerce_numbers_to_str``: a number becomes text."""
        assert Sample().run(123) == "123|10|None"  # type: ignore[arg-type]

    def test_strip_whitespace(self) -> None:
        """``str_strip_whitespace``: blank text is empty, hence invalid."""
        assert Sample().run("  hi  ") == "hi|10|None"
        with pytest.raises(ArgError):
            Sample().run("   ")

    def test_str_to_int(self) -> None:
        """Numeric text is converted to ``int``."""
        assert Sample().run("hi", "7") == "hi|7|None"  # type: ignore[arg-type]

    def test_file_path(self, tmp_path: Path) -> None:
        """``FilePath`` accepts an existing file and rejects others."""
        file = tmp_path / "a.json"
        file.write_text("{}", encoding="utf-8")
        assert Sample().run("hi", path=file).endswith(str(file))
        with pytest.raises(ArgError):
            Sample().run("hi", path=tmp_path / "missing.json")
        with pytest.raises(ArgError):
            Sample().run("hi", path=tmp_path)


class TestErrors:
    """What the decorated command raises."""

    def test_names_the_option(self) -> None:
        """``ArgError`` names each invalid option and its value."""
        with pytest.raises(ArgError) as info:
            Sample().run("hi", 0)
        assert "--k '0': Input should be greater than 0" in str(info.value)

    def test_keeps_cause(self) -> None:
        """The original ``ValidationError`` is chained."""
        with pytest.raises(ArgError) as info:
            Sample().run("")
        assert isinstance(info.value.__cause__, ValidationError)

    def test_body_errors_untouched(self) -> None:
        """A ``ValidationError`` raised inside the command is not an
        argument error: it goes through unchanged.
        """
        class Broken:
            """A command whose body raises a ValidationError of its own."""

            @CliCommand.validated
            def run(self, k: int = 1) -> None:
                """Validate the dataset inside the body (always fails)."""
                RagDataset.model_validate({})

        with pytest.raises(ValidationError) as info:
            Broken().run(3)
        assert info.value.title == "RagDataset"


class TestConfig:
    """The shared configuration."""

    def test_config_values(self) -> None:
        """The two coercion options are enabled."""
        assert CliCommand.CONFIG.get("coerce_numbers_to_str") is True
        assert CliCommand.CONFIG.get("str_strip_whitespace") is True


class TestOpenFileSave:
    """``RagCLI._open_file_save``: read + validate a JSON input."""

    VALID = '{"rag_questions": [{"question_id": "q1", "question": "hi"}]}'

    def test_returns_model_instance(self, tmp_path: Path) -> None:
        """Valid content gives an object of the requested model."""
        path = tmp_path / "d.json"
        path.write_text(self.VALID, encoding="utf-8")
        dataset = RagCLI._open_file_save(path, RagDataset)
        assert isinstance(dataset, RagDataset)
        assert dataset.rag_questions[0].question_id == "q1"

    def test_other_model(self, tmp_path: Path) -> None:
        """Works with any model, e.g. search results."""
        path = tmp_path / "r.json"
        path.write_text('{"search_results": [], "k": 5}', encoding="utf-8")
        results = RagCLI._open_file_save(path, StudentSearchResults)
        assert results.k == 5

    def test_missing_file(self, tmp_path: Path) -> None:
        """A file that does not exist: ``PathError``."""
        with pytest.raises(PathError, match="No such file"):
            RagCLI._open_file_save(tmp_path / "nope.json", RagDataset)

    def test_directory(self, tmp_path: Path) -> None:
        """A folder instead of a file: ``PathError``."""
        with pytest.raises(PathError):
            RagCLI._open_file_save(tmp_path, RagDataset)

    def test_not_utf8(self, tmp_path: Path) -> None:
        """Bytes that are not UTF-8: ``PathError`` saying so."""
        path = tmp_path / "bad.json"
        path.write_bytes(b"\xff\xfe\x00 bad")
        with pytest.raises(PathError, match="not valid UTF-8"):
            RagCLI._open_file_save(path, RagDataset)

    @pytest.mark.parametrize("content", ["", "{", '{"rag_questions": [,]}',
                                         '{"other": []}', "[1, 2]"])
    def test_invalid_content(self, tmp_path: Path, content: str) -> None:
        """Broken JSON or wrong shape: ``DatasetError`` naming the file."""
        path = tmp_path / "d.json"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(DatasetError, match="d.json"):
            RagCLI._open_file_save(path, RagDataset)

    def test_wrong_model(self, tmp_path: Path) -> None:
        """A dataset read as search results is rejected."""
        path = tmp_path / "d.json"
        path.write_text(self.VALID, encoding="utf-8")
        with pytest.raises(DatasetError):
            RagCLI._open_file_save(path, StudentSearchResults)


class TestLoadChunks:
    """``answer_dataset`` re-reads the text of each retrieved source."""

    @staticmethod
    def source(path: str, start: int = 0, end: int = 5) -> MinimalSource:
        """A retrieved source."""
        return MinimalSource(file_path=path, first_character_index=start,
                             last_character_index=end)

    def test_reads_slice(self) -> None:
        """The chunk text is the slice of the real corpus file."""
        path = "README.md"
        text = (Path(__file__).resolve().parent.parent / path).read_text(
            encoding="utf-8")
        chunks = RagCLI._load_chunks([self.source(path, 3, 20)], {})
        assert chunks[0].text == text[3:20]

    def test_file_read_once(self) -> None:
        """Sources of the same file share the cache."""
        files: dict[str, str] = {"cached.md": "0123456789"}
        chunks = RagCLI._load_chunks(
            [self.source("cached.md", 0, 3), self.source("cached.md", 5, 9)],
            files)
        assert [c.text for c in chunks] == ["012", "5678"]

    @pytest.mark.parametrize("path", ["/etc/passwd", "../../.ssh/id_rsa"])
    def test_outside_project_rejected(self, path: str) -> None:
        """A JSON cannot make the generator read files outside the
        project: absolute paths and ``..`` are refused before reading.
        """
        with pytest.raises(CorpusError, match="outside the project root"):
            RagCLI._load_chunks([self.source(path)], {})

    def test_missing_file(self) -> None:
        """A source that no longer exists is a clean PathError."""
        with pytest.raises(PathError):
            RagCLI._load_chunks([self.source("data/raw/nope.md")], {})
