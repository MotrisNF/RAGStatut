"""Tests for the exception hierarchy in ``src.exceptions``."""
import fire
import pytest
from pydantic import ValidationError

from src.commands import RagCLI
from src.exceptions import ArgError, CorpusError, DatasetError, RagError
from src.models import RagDataset, UnansweredQuestion


class ErrorFactory:
    """Build real pydantic ``ValidationError`` objects for the tests."""

    @staticmethod
    def from_model(data: dict[str, object]) -> ValidationError:
        """Validate ``data`` as an UnansweredQuestion and return the error."""
        with pytest.raises(ValidationError) as info:
            UnansweredQuestion.model_validate(data)
        return info.value

    @staticmethod
    def from_json(text: str) -> ValidationError:
        """Validate ``text`` as a RagDataset and return the error."""
        with pytest.raises(ValidationError) as info:
            RagDataset.model_validate_json(text)
        return info.value

    @staticmethod
    def from_cli(command: list[str]) -> ArgError:
        """Run ``command`` through Fire and return the argument error."""
        with pytest.raises(ArgError) as info:
            fire.Fire(RagCLI, command=command)
        return info.value


class TestRagError:
    """The base class of the hierarchy."""

    def test_default_message(self) -> None:
        """Without arguments it uses the default message."""
        assert str(RagError()) == "Unknown RagError"

    def test_custom_message(self) -> None:
        """The message given is the one shown."""
        assert str(RagError("Index not built")) == "Index not built"

    def test_is_exception(self) -> None:
        """It can be caught as any other exception."""
        with pytest.raises(Exception):
            raise RagError("boom")


class TestDatasetError:
    """Formatting of dataset validation errors."""

    def test_header_has_path(self) -> None:
        """The first line names the file."""
        error = ErrorFactory.from_model({})
        lines = str(DatasetError("x.json", error)).splitlines()
        assert lines[0] == "Invalid file 'x.json':"

    def test_one_line_per_error(self) -> None:
        """Every pydantic error gets its own ``field: message`` line."""
        error = ErrorFactory.from_model({"question_id": 5})
        lines = str(DatasetError("x.json", error)).splitlines()
        assert len(lines) == 1 + error.error_count()
        assert "  question_id: Input should be a valid string" in lines
        assert "  question: Field required" in lines

    def test_nested_location(self) -> None:
        """Nested fields are joined with dots, list positions included."""
        error = ErrorFactory.from_json('{"rag_questions": [{}]}')
        message = str(DatasetError("x.json", error))
        assert "rag_questions.0" in message

    def test_invalid_json_uses_root(self) -> None:
        """Broken JSON has no field, so ``<root>`` is shown."""
        error = ErrorFactory.from_json('{"rag_questions": [,]}')
        message = str(DatasetError("x.json", error))
        assert "  <root>: Invalid JSON" in message

    def test_caught_as_rag_error(self) -> None:
        """``except RagError`` catches it."""
        error = ErrorFactory.from_model({})
        with pytest.raises(RagError):
            raise DatasetError("x.json", error) from error


class TestArgError:
    """Formatting of command-line argument errors."""

    NAMES = ["self", "query", "k"]

    def test_header(self) -> None:
        """The first line says the arguments are invalid."""
        error = ErrorFactory.from_cli(["search", "hola", "--k", "0"])
        assert str(error).splitlines()[0] == "Invalid arguments:"

    def test_names_option_value_and_reason(self) -> None:
        """Each line: ``--option 'value': reason``."""
        error = ErrorFactory.from_cli(["search", "hola", "--k", "0"])
        assert "  --k '0': Input should be greater than 0" in str(error)

    def test_missing_file(self) -> None:
        """A missing input file is reported with its option and path."""
        error = ErrorFactory.from_cli(
            ["search_dataset", "--dataset_path", "nope.json"])
        assert "--dataset_path 'nope.json'" in str(error)

    def test_several_errors(self) -> None:
        """All invalid arguments are listed, one per line."""
        error = ErrorFactory.from_cli(["search", "", "--k", "abc"])
        lines = str(error).splitlines()
        assert len(lines) == 3
        assert lines[1].startswith("  --query ''")
        assert lines[2].startswith("  --k 'abc'")

    @pytest.mark.parametrize("value", ["", " ", "   "])
    def test_blank_text(self, value: str) -> None:
        """Empty or blank text gets a clear reason (pydantic strips it
        before checking, but reports the original value).
        """
        error = ErrorFactory.from_cli(["search", value])
        assert f"--query '{value}': must not be empty or blank" in str(error)

    def test_describe_positional(self) -> None:
        """A value written without its option says so."""
        error = ErrorFactory.from_cli(["index", "patata"])
        assert error.describe({"max_chunk_size"}) == (
            "Invalid arguments:\n"
            "  'patata' (positional, taken as --max_chunk_size): Input "
            "should be a valid integer, unable to parse string as an integer"
        )

    def test_describe_without_positional_is_str(self) -> None:
        """With no positional information it is the default message."""
        error = ErrorFactory.from_cli(["search", "", "--k", "abc"])
        assert error.describe() == str(error)

    def test_describe_only_marks_given_params(self) -> None:
        """Each line is worded on its own."""
        error = ErrorFactory.from_cli(["search", "", "--k", "abc"])
        lines = error.describe({"query"}).splitlines()
        assert lines[1].startswith("  '' (positional, taken as --query):")
        assert lines[2].startswith("  --k 'abc':")

    def test_option_by_position(self) -> None:
        """A numeric ``loc`` is turned into the parameter name."""
        assert ArgError._option((2,), self.NAMES) == "--k"

    def test_option_by_name(self) -> None:
        """A keyword ``loc`` is used as it is."""
        assert ArgError._option(("query",), self.NAMES) == "--query"

    @pytest.mark.parametrize("loc", [(), (9,), (-1,)])
    def test_option_unknown(self, loc: tuple[int, ...]) -> None:
        """Without a usable position the text is generic."""
        assert ArgError._option(loc, self.NAMES) == "argument"

    def test_caught_as_rag_error(self) -> None:
        """``except RagError`` catches it."""
        with pytest.raises(RagError):
            fire.Fire(RagCLI, command=["search", "hola", "--k", "0"])


class TestCorpusError:
    """Errors of the corpus reader."""

    def test_default_message(self) -> None:
        """Without arguments it has a generic message."""
        assert str(CorpusError()) != ""

    def test_custom_message(self) -> None:
        """The message given is the one shown."""
        assert str(CorpusError("Corpus folder not found")) == \
            "Corpus folder not found"

    def test_caught_as_rag_error(self) -> None:
        """``except RagError`` catches it."""
        with pytest.raises(RagError):
            raise CorpusError("boom")
