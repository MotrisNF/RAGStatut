"""Entry point for ``uv run python -m src <command>``."""
import sys

import fire

from .argv_guard import ArgvGuard
from .commands import RagCLI
from .exceptions import ArgError, RagError, UsageError


def main() -> None:
    """Run the command-line interface.

    Every error ends here and is printed without a traceback:

    - ``UsageError`` (ArgvGuard): malformed command line → exit 2.
    - ``ArgError``: invalid values, worded with what ArgvGuard saw (a
      value written without its option is shown as positional) → exit 1.
    - ``RagError`` (CorpusError, DatasetError...): its message → exit 1.
    - Ctrl+C → exit 130.
    - Anything else is a bug → ``Unexpected error`` → exit 1.
    """
    guard = ArgvGuard(RagCLI)
    try:
        argv = guard.check(sys.argv[1:])
        fire.Fire(RagCLI, command=argv, name="src")
    except UsageError as e:
        print(e, file=sys.stderr)
        sys.exit(2)
    except ArgError as e:
        print(e.describe(guard.positional), file=sys.stderr)
        sys.exit(1)
    except RagError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
