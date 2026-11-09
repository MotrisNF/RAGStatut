"""Exception hierarchy for the RAG project."""
from collections.abc import Collection
from pathlib import Path

from pydantic import ValidationError
from pydantic_core import ErrorDetails


class RagError(Exception):
    """Base class for every error raised by the RAG project.

    Catching ``RagError`` catches all of its subclasses too.
    """

    def __init__(self, msg: str = "Unknown RagError") -> None:
        """Store a human-readable message.

        Args:
            msg: Message shown to the user.
        """
        super().__init__(msg)


class DatasetError(RagError):
    """A JSON file could not be parsed or does not match its model."""

    def __init__(self, path: str, error: ValidationError) -> None:
        """Build the message as one ``field: message`` line per error.

        Args:
            path: Path of the file that failed to validate.
            error: The original pydantic error.
        """
        lines = [f"Invalid file '{path}':"]
        for err in error.errors():
            where = ".".join(str(p) for p in err["loc"]) or "<root>"
            lines.append(f"  {where}: {err['msg']}")
        super().__init__("\n".join(lines))


class ArgError(RagError):
    """A command-line argument was rejected by validation.

    One line per invalid argument: ``--option 'value': reason``, or
    ``'value' (positional, taken as --option): reason`` when the value
    was written without its option (see ``describe``).

    Attributes:
        problems: ``(option, value, reason)`` of each invalid argument.
    """

    def __init__(self, error: ValidationError, names: list[str]) -> None:
        """Build the message from a pydantic validation error.

        Args:
            error: The original pydantic error.
            names: Parameter names of the command, in order and including
                ``self``. pydantic reports each argument by its position
                (``loc``), and Fire always passes them by position, so the
                position is turned back into the option name here.
        """
        self.problems: list[tuple[str, object, str]] = [
            (self._option(err["loc"], names), err["input"],
             self._reason(err))
            for err in error.errors()
        ]
        super().__init__(self.describe())

    def describe(self, positional: Collection[str] = ()) -> str:
        """The message, telling apart the values written without option.

        Fire passes every argument by position, so only the command line
        (``ArgvGuard``) knows which values had no option. Saying
        ``--max_chunk_size 'patata'`` for ``index patata`` would claim the
        user wrote an option they did not.

        Args:
            positional: Parameter names that received a value without
                its option.

        Returns:
            ``Invalid arguments:`` and one line per invalid argument.
        """
        lines = ["Invalid arguments:"]
        for option, value, reason in self.problems:
            if option.removeprefix("--") in positional:
                lines.append(
                    f"  '{value}' (positional, taken as {option}): {reason}"
                )
            else:
                lines.append(f"  {option} '{value}': {reason}")
        return "\n".join(lines)

    @staticmethod
    def _option(loc: tuple[int | str, ...], names: list[str]) -> str:
        """``--name`` of the argument at ``loc`` (position or name)."""
        where = loc[0] if loc else None
        if isinstance(where, int) and 0 <= where < len(names):
            return f"--{names[where]}"
        if isinstance(where, str):
            return f"--{where}"
        return "argument"

    @staticmethod
    def _reason(err: ErrorDetails) -> str:
        """pydantic's message, clearer for blank text.

        Text is stripped before checking its length, but pydantic reports
        the original value, so ``' '`` would read "should have at least 1
        character".
        """
        value = err["input"]
        if (err["type"] == "string_too_short" and isinstance(value, str)
                and not value.strip()):
            return "must not be empty or blank"
        return err["msg"]


class CorpusError(RagError):
    """The corpus folder is missing, outside the project or unreadable."""

    def __init__(
            self, msg: str = "Something failed reading the corpus."
            ) -> None:
        """Store the message.

        Args:
            msg: Message shown to the user.
        """
        super().__init__(msg)


class UsageError(RagError):
    """The command line is malformed (unknown option, too many arguments).

    ``main`` exits with code 2 for it, like Fire's own usage errors.
    """


class IndexBuildError(RagError):
    """The search index cannot be built from the given chunks.

    Raised when there is nothing to index: no chunks at all, or chunks
    without a single token (e.g. only symbols). Without it the average
    chunk length would be 0 and BM25 would divide by zero.
    """


class PathError(RagError):
    """A file or folder cannot be created, read or written.

    Wraps the ``OSError`` raised by the operating system (missing
    permissions, a file where a folder was expected, disk full...) into
    a short message for the user. A ``UnicodeDecodeError`` (a file that
    is not UTF-8 text) is accepted too.
    """

    def __init__(self, path: str | Path,
                 error: OSError | UnicodeDecodeError) -> None:
        """Build the message from the path and the system error.

        Args:
            path: The file or folder that failed.
            error: The original ``OSError`` or ``UnicodeDecodeError``.
        """
        if isinstance(error, UnicodeDecodeError):
            reason = "not valid UTF-8 text"
        else:
            reason = error.strerror or type(error).__name__
        super().__init__(f"Cannot use '{path}': {reason}")


class IndexNotFoundError(RagError):
    """There is no saved index to load.

    The usual cause is running a search command before ``index``.
    """

    def __init__(self, index_dir: str | Path) -> None:
        """Tell the user where the index was expected and what to do.

        Args:
            index_dir: Folder where the index should be.
        """
        super().__init__(
            f"No index found in '{index_dir}'. Run 'index' first."
        )


class EvaluationError(RagError):
    """The inputs of ``evaluate`` cannot be used to compute recall.

    Attributes:
        problem: The message without the ``Evaluation error:`` prefix, so
            the command can wrap it with the path of the faulty file.
    """

    def __init__(self, error: str) -> None:
        """Store the problem and build the message.

        Args:
            error: What is wrong.
        """
        self.problem = error
        super().__init__(f"Evaluation error: {error}")


class DatasetEvaluationError(EvaluationError):
    """The ground-truth dataset cannot be used (no questions, or a
    question without sources)."""


class ResultsEvaluationError(EvaluationError):
    """The search results are not valid (repeated question, too many
    sources, impossible or too long range)."""


class VectorIndexError(RagError):
    """The saved vector index cannot be used: an unreadable file, arrays
    that do not fit together, a broken embedding model, or a vector index
    that does not match the BM25 index."""

    def __init__(self, error: str) -> None:
        """Prefix the problem so the user knows which index failed.

        Args:
            error: What is wrong, ending with what to do.
        """
        super().__init__(f"Vector index error: {error}")
