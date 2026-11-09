"""Tests for ``src.argv_guard.ArgvGuard``."""
import pytest

from src.argv_guard import ArgvGuard
from src.commands import RagCLI
from src.exceptions import RagError, UsageError


class Sample:
    """A small component: two commands and a private method."""

    def search(self, query: str, k: int = 10) -> None:
        """Two parameters, the second with a default."""

    def index(self, max_chunk_size: int = 2000, raw_dir: str = "r") -> None:
        """Only options with defaults."""

    def kinds(self, key: str = "", kind: str = "") -> None:
        """Two parameters with the same initial."""

    def flags(self, verbose: bool = False, name: str = "") -> None:
        """A boolean flag next to a normal option."""

    LIMIT: int = 3

    def _private(self) -> None:
        """Not a command."""


@pytest.fixture
def guard() -> ArgvGuard:
    """Guard for the ``Sample`` component."""
    return ArgvGuard(Sample)


class TestAccepted:
    """Every form Fire accepts is let through unchanged."""

    @pytest.mark.parametrize("argv", [
        ["search", "hola"],
        ["search", "hola", "5"],
        ["search", "hola", "--k", "5"],
        ["search", "--k", "5", "hola"],
        ["search", "--query", "hola", "--k", "5"],
        ["search", "hola", "--k=5"],
        ["search", "hola", "-k", "5"],
        ["search", "hola", "-k=5"],
        ["search", "hola", "--k", "-3"],
        ["search", "hola", "--k=-3"],
        ["search", "hola", "--k", "2.5"],
        ["search", "-5"],
        ["index"],
        ["index", "--max_chunk_size", "500"],
        ["index", "--max-chunk-size", "500"],
        ["index", "-max_chunk_size", "500"],
        ["index", "-m", "500"],
        ["index", "-r", "data/raw"],
        ["index", "500", "data/raw"],
        ["kinds", "--key", "a", "--kind", "b"],
    ])
    def test_unchanged(self, guard: ArgvGuard, argv: list[str]) -> None:
        """The command line is returned as it came."""
        assert guard.check(argv) == argv

    @pytest.mark.parametrize("argv", [
        [],
        ["--help"],
        ["-h"],
    ])
    def test_left_to_fire(self, guard: ArgvGuard, argv: list[str]) -> None:
        """No command, or Fire's own flags: Fire shows the help."""
        assert guard.check(argv) == argv


class TestUnknownCommand:
    """Only public methods are commands. Fire would also run private
    methods and print class attributes when named explicitly.
    """

    @pytest.mark.parametrize("name", [
        "not_a_command", "_private", "__init__", "__class__", "LIMIT",
        "Search", "search_", "",
    ])
    def test_rejected(self, guard: ArgvGuard, name: str) -> None:
        """``UsageError`` naming the word and the real commands."""
        with pytest.raises(UsageError) as info:
            guard.check([name, "x"])
        assert f"Unknown command '{name}'" in str(info.value)
        assert "flags, index, kinds, search" in str(info.value)

    @pytest.mark.parametrize("name", [
        "_search_question", "_open_file_save", "_print_ids", "SEPARATOR",
        "MAX_IDS_SHOWN", "__dict__",
    ])
    def test_rag_cli_internals_rejected(self, name: str) -> None:
        """Helpers and attributes of ``RagCLI`` are not reachable."""
        with pytest.raises(UsageError, match="Unknown command"):
            ArgvGuard(RagCLI).check([name])


class TestHelp:
    """``--help`` never runs the command."""

    @pytest.mark.parametrize("argv", [
        ["search", "--help"],
        ["search", "hola", "--help"],
        ["search", "hola", "-h"],
        ["search", "hola", "-help"],
        ["index", "--max_chunk_size", "5", "--help"],
        ["index", "--help", "--max_chunk_size", "5"],
        ["index", "--unknown", "--help"],
    ])
    def test_rewritten(self, guard: ArgvGuard, argv: list[str]) -> None:
        """It becomes ``<command> -- --help``, Fire's help-only form."""
        assert guard.check(argv) == [argv[0], "--", "--help"]

    def test_help_after_separator_untouched(self, guard: ArgvGuard) -> None:
        """``command -- --help`` is already the right form."""
        argv = ["search", "--", "--help"]
        assert guard.check(argv) == argv


class TestUnknownOptions:
    """Options that match no parameter."""

    @pytest.mark.parametrize("argv", [
        ["index", "--chunk_size", "5"],
        ["index", "--max_chunk", "5"],
        ["index", "--max_chunk_size", "5", "--raw", "x"],
        ["index", "--chunk-size=5"],
        ["search", "hola", "--kk", "5"],
        ["search", "hola", "-x", "5"],
        ["search", "hola", "--nok"],
        ["kinds", "-k", "a"],
    ])
    def test_rejected(self, guard: ArgvGuard, argv: list[str]) -> None:
        """``UsageError`` naming the option."""
        with pytest.raises(UsageError, match="Unknown option"):
            guard.check(argv)

    def test_message_lists_valid_options(self, guard: ArgvGuard) -> None:
        """The message says which options exist."""
        with pytest.raises(UsageError) as info:
            guard.check(["index", "--chunk_size", "5"])
        message = str(info.value)
        assert "'--chunk_size'" in message
        assert "'index'" in message
        assert "--max_chunk_size, --raw_dir" in message

    def test_ambiguous_initial(self, guard: ArgvGuard) -> None:
        """``-k`` is ambiguous when two parameters start with ``k``."""
        with pytest.raises(UsageError):
            guard.check(["kinds", "-k", "a"])


class TestTooManyArguments:
    """More values than the command has parameters."""

    @pytest.mark.parametrize("argv", [
        ["search", "hola", "3", "4"],
        ["search", "How", "does", "it", "work"],
        ["search", "hola", "--k", "3", "extra"],
        ["search", "--query", "a", "--k", "3", "extra"],
        ["index", "1", "2", "3"],
    ])
    def test_rejected(self, guard: ArgvGuard, argv: list[str]) -> None:
        """``UsageError`` saying how many it takes."""
        with pytest.raises(UsageError, match="Too many arguments"):
            guard.check(argv)

    def test_message(self, guard: ArgvGuard) -> None:
        """The message gives the limit and the options."""
        with pytest.raises(UsageError) as info:
            guard.check(["search", "a", "b", "c"])
        assert "at most 2 (--query, --k)" in str(info.value)


class TestMissingValue:
    """An option written without its value."""

    @pytest.mark.parametrize("argv", [
        ["index", "--raw_dir"],
        ["index", "--max_chunk_size", "10", "--raw_dir"],
        ["index", "--raw_dir", "--max_chunk_size", "10"],
        ["index", "-m"],
        ["index", "--max-chunk-size"],
        ["search", "hola", "--k"],
        ["index", "--raw_dir", "--", "--trace"],
        ["flags", "--name"],
    ])
    def test_rejected(self, guard: ArgvGuard, argv: list[str]) -> None:
        """``UsageError`` instead of Fire turning it into ``True``."""
        with pytest.raises(UsageError, match="needs a value"):
            guard.check(argv)

    def test_message(self, guard: ArgvGuard) -> None:
        """The message names the option as written and the command."""
        with pytest.raises(UsageError) as info:
            guard.check(["index", "-m"])
        assert str(info.value) == "Option '-m' of 'index' needs a value"

    @pytest.mark.parametrize("argv", [
        ["flags", "--verbose"],
        ["flags", "--verbose", "--name", "x"],
        ["flags", "--name", "x", "--verbose"],
        ["flags", "--verbose", "False"],
        ["flags", "-v"],
    ])
    def test_bool_flag_alone_ok(self, guard: ArgvGuard,
                                argv: list[str]) -> None:
        """A ``bool`` parameter may appear without a value."""
        assert guard.check(argv) == argv

    @pytest.mark.parametrize("argv", [
        ["index", "--raw_dir="],
        ["search", "hola", "--k", "-3"],
        ["search", "hola", "--k=-3"],
    ])
    def test_value_present(self, guard: ArgvGuard, argv: list[str]) -> None:
        """An empty ``=`` value or a negative number is a value (pydantic
        judges it later).
        """
        assert guard.check(argv) == argv

    def test_help_wins(self, guard: ArgvGuard) -> None:
        """Asking for help is honoured even if a value is missing."""
        argv = ["index", "--raw_dir", "--help"]
        assert guard.check(argv) == ["index", "--", "--help"]


class TestCommandAsValue:
    """A command name that Fire would take as a positional value.

    Fire gives positional arguments, in order, to the parameters not given
    as options, so ``index 500 search`` would run ``index`` with
    ``raw_dir='search'``.
    """

    @pytest.mark.parametrize("argv, param", [
        (["index", "search"], "max_chunk_size"),
        (["index", "500", "search"], "raw_dir"),
        (["index", "--max_chunk_size", "500", "search"], "raw_dir"),
        (["index", "-m", "500", "search"], "raw_dir"),
        (["index", "--raw_dir", "x", "index"], "max_chunk_size"),
        (["search", "hola", "index"], "k"),
        (["search", "--query", "hola", "index"], "k"),
        (["kinds", "--kind", "b", "flags"], "key"),
    ])
    def test_rejected(self, guard: ArgvGuard, argv: list[str],
                      param: str) -> None:
        """``UsageError`` naming the command and the parameter it hit."""
        value = argv[-1]
        with pytest.raises(UsageError) as info:
            guard.check(argv)
        assert str(info.value) == (
            f"'{value}' is a command, not a value for --{param} of "
            f"'{argv[0]}'. Run one command at a time"
        )

    @pytest.mark.parametrize("argv", [
        ["search", "index"],
        ["search", "search"],
        ["search", "--k", "5", "index"],
        ["search", "index", "5"],
        ["search", "--query", "index"],
        ["index", "--raw_dir", "search"],
        ["index", "-r", "search"],
        ["index", "--raw_dir=search"],
        ["kinds", "--key", "index"],
    ])
    def test_accepted(self, guard: ArgvGuard, argv: list[str]) -> None:
        """A command name as free text (``query``) or as the explicit
        value of an option is what the user asked for.
        """
        assert guard.check(argv) == argv

    def test_unknown_option_reported_first(self, guard: ArgvGuard) -> None:
        """A typo in an option is the first thing to fix."""
        with pytest.raises(UsageError, match="Unknown option '--nope'"):
            guard.check(["index", "--nope", "1", "search"])

    def test_too_many_reported_first(self, guard: ArgvGuard) -> None:
        """Too many arguments wins over the command-name check."""
        with pytest.raises(UsageError, match="Too many arguments"):
            guard.check(["search", "hola", "5", "index"])


class TestSeparators:
    """Text after ``-`` or ``--``, which Fire handles after running."""

    @pytest.mark.parametrize("argv", [
        ["search", "hola", "-"],
        ["search", "hola", "--"],
        ["search", "--", "--trace"],
        ["search", "hola", "--", "--trace", "--verbose"],
        ["search", "hola", "--", "-t", "-v"],
        ["search", "hola", "--", "--interactive"],
        ["search", "hola", "--", "--separator", "+"],
        ["search", "hola", "--", "--separator=+"],
        ["search", "hola", "--", "--completion"],
        ["search", "hola", "--", "--completion", "fish"],
        ["search", "hola", "--", "--completion", "--trace"],
    ])
    def test_accepted(self, guard: ArgvGuard, argv: list[str]) -> None:
        """A lone separator or Fire's own flags after ``--``."""
        assert guard.check(argv) == argv

    @pytest.mark.parametrize("argv, extra", [
        (["search", "hola", "-", "upper"], "upper"),
        (["index", "-", "search"], "search"),
        (["index", "-m", "500", "-", "search", "hola"], "search hola"),
        (["index", "-", "--help"], "--help"),
    ])
    def test_chain_rejected(self, guard: ArgvGuard, argv: list[str],
                            extra: str) -> None:
        """Nothing can follow ``-``: the commands return nothing."""
        with pytest.raises(UsageError) as info:
            guard.check(argv)
        assert str(info.value) == (
            f"Unexpected arguments after '-' for '{argv[0]}': {extra}. "
            "Commands cannot be chained"
        )

    @pytest.mark.parametrize("argv, unknown", [
        (["index", "--", "--una_flag", "search"], "--una_flag"),
        (["index", "--", "search"], "search"),
        (["search", "hola", "--", "--trace", "extra"], "extra"),
        (["search", "hola", "--", "--trace", "-", "x"], "-"),
        (["search", "hola", "--", "--k", "5"], "--k"),
    ])
    def test_unknown_fire_flag_rejected(self, guard: ArgvGuard,
                                        argv: list[str],
                                        unknown: str) -> None:
        """Fire would silently ignore it, so it is an error here."""
        with pytest.raises(UsageError) as info:
            guard.check(argv)
        message = str(info.value)
        assert message.startswith(
            f"Unknown Fire flag '{unknown}' after '--' for '{argv[0]}'."
        )
        assert "--trace" in message

    def test_help_before_separator_wins(self, guard: ArgvGuard) -> None:
        """Help never runs the command, so what follows does not matter."""
        argv = ["index", "--help", "-", "search"]
        assert guard.check(argv) == ["index", "--", "--help"]

    def test_options_checked_before_separator(self,
                                              guard: ArgvGuard) -> None:
        """An error before the separator is reported first."""
        with pytest.raises(UsageError, match="Unknown option '--nope'"):
            guard.check(["index", "--nope", "1", "-", "search"])


class TestPositionalParams:
    """``positional``: the parameters Fire fills without an option."""

    @pytest.mark.parametrize("argv, expected", [
        (["index"], set()),
        (["index", "patata"], {"max_chunk_size"}),
        (["index", "500", "x"], {"max_chunk_size", "raw_dir"}),
        (["index", "--max_chunk_size", "5"], set()),
        (["index", "-m", "5", "x"], {"raw_dir"}),
        (["index", "x", "--max_chunk_size", "5"], {"raw_dir"}),
        (["index", "--raw_dir", "x", "5"], {"max_chunk_size"}),
        (["search", "hola", "--k", "5"], {"query"}),
        (["search", "--query", "hola", "5"], {"k"}),
        (["index", "--help"], set()),
        (["-h"], set()),
    ])
    def test_after_check(self, guard: ArgvGuard, argv: list[str],
                         expected: set[str]) -> None:
        """Positional values fill, in order, the parameters left free."""
        guard.check(argv)
        assert guard.positional == expected

    def test_reset_between_checks(self, guard: ArgvGuard) -> None:
        """A later check does not keep the previous result."""
        guard.check(["index", "patata"])
        guard.check(["index", "--max_chunk_size", "5"])
        assert guard.positional == set()

    def test_reset_on_error(self, guard: ArgvGuard) -> None:
        """A rejected command line leaves nothing behind."""
        guard.check(["index", "patata"])
        with pytest.raises(UsageError):
            guard.check(["index", "--nope", "1"])
        assert guard.positional == set()


class TestErrorType:
    """How ``main`` handles it."""

    def test_is_rag_error(self, guard: ArgvGuard) -> None:
        """``UsageError`` is a ``RagError``."""
        with pytest.raises(RagError):
            guard.check(["index", "--nope", "1"])


class TestRagCLI:
    """The guard against the real CLI."""

    def test_commands(self) -> None:
        """The six subject commands and ``serve`` (bonus 5) are detected."""
        assert ArgvGuard(RagCLI)._commands() == {
            "index", "search", "search_dataset", "answer",
            "answer_dataset", "evaluate", "serve",
        }

    def test_uses_clean_signatures(self) -> None:
        """Parameters come from the decorated methods."""
        params = ArgvGuard(RagCLI)._params("index")
        assert params == ["max_chunk_size", "raw_dir", "index_dir", "full"]

    @pytest.mark.parametrize("argv", [
        ["index", "--chunk_size", "5"],
        ["search_dataset", "--dataset", "x.json"],
        ["answer_dataset", "--student_search_result_path", "x.json"],
        ["evaluate", "--datase_path", "x.json"],
    ])
    def test_typos_rejected(self, argv: list[str]) -> None:
        """Typical typos in the subject's option names are caught."""
        with pytest.raises(UsageError):
            ArgvGuard(RagCLI).check(argv)

    @pytest.mark.parametrize("argv, message", [
        (["index", "--max_chunk_size", "500", "search"],
         "'search' is a command, not a value for --raw_dir of 'index'"),
        (["index", "search"],
         "'search' is a command, not a value for --max_chunk_size of "
         "'index'"),
        (["search", "lora", "--k", "5", "index"],
         "'index' is a command, not a value for --index_dir of 'search'"),
        (["search_dataset", "--k", "5", "index"],
         "'index' is a command, not a value for --dataset_path of "
         "'search_dataset'"),
        (["answer", "lora", "search"],
         "'search' is a command, not a value for --k of 'answer'"),
        (["index", "-", "search"],
         "Unexpected arguments after '-' for 'index': search"),
        (["index", "--", "--una_flag", "search"],
         "Unknown Fire flag '--una_flag' after '--' for 'index'"),
    ])
    def test_second_command_rejected(self, argv: list[str],
                                     message: str) -> None:
        """A second command is refused before anything runs."""
        with pytest.raises(UsageError, match=message):
            ArgvGuard(RagCLI).check(argv)

    @pytest.mark.parametrize("argv", [
        ["search", "index"],
        ["search", "--k", "5", "search"],
        ["answer", "evaluate"],
        ["index", "--index_dir", "search"],
    ])
    def test_command_words_as_values_accepted(self,
                                              argv: list[str]) -> None:
        """Searching for the word "index" or naming a folder "search"."""
        assert ArgvGuard(RagCLI).check(argv) == argv
