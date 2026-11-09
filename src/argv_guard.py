"""Check a command line against a Fire component before Fire runs it.

Fire calls the command with the arguments it understands and only then
complains about the rest, so a typo in an option (``--chunk_size``), too
many arguments, a second command written as a value (``index -m 500
search``), text after a separator (``index - search``) or a ``--help``
after other arguments would run the whole command first. ``ArgvGuard``
catches those cases up front.
"""
import inspect
from typing import Any

from .exceptions import UsageError


class ArgvGuard:
    """Validate ``argv`` against the public methods of a Fire component.

    Attributes:
        SEPARATORS: ``-`` chains a call on the result of the command and
            ``--`` starts Fire's own flags.
        HELP_FLAGS: Ways of asking for help.
        FIRE_FLAGS: Fire's own flags, the only ones allowed after ``--``.
        FIRE_VALUE_FLAGS: Fire flags that may take a value.
        FREE_TEXT_PARAMS: Parameters whose value is free text, so a
            command name there is a real value (``search index``).
        positional: Parameters that receive a value written without its
            option, filled by the last ``check`` (``main`` passes it to
            ``ArgError.describe``).
    """

    SEPARATORS: frozenset[str] = frozenset({"--", "-"})
    HELP_FLAGS: frozenset[str] = frozenset({"--help", "-help", "-h"})
    FIRE_FLAGS: frozenset[str] = frozenset({
        "--help", "-h", "--interactive", "-i", "--trace", "-t",
        "--verbose", "-v", "--completion", "--separator",
    })
    FIRE_VALUE_FLAGS: frozenset[str] = frozenset({"--completion",
                                                  "--separator"})
    FREE_TEXT_PARAMS: frozenset[str] = frozenset({"query"})

    def __init__(self, component: type[Any]) -> None:
        """Keep the class whose methods are the commands.

        Args:
            component: The class given to ``fire.Fire``.
        """
        self._component = component
        self.positional: set[str] = set()

    def check(self, argv: list[str]) -> list[str]:
        """Return the command line Fire should receive.

        Args:
            argv: Arguments after the program name.

        Returns:
            ``argv`` unchanged, or ``[command, "--", "--help"]`` when help
            was asked for, so the command is never run.

        Raises:
            UsageError: Unknown command (Fire would also run private
                methods and show class attributes), unknown option, too
                many arguments, a command name given as a value, or text
                after a separator that Fire would only reject after
                running the command.
        """
        self.positional = set()
        if not argv:
            return argv
        commands = self._commands()
        if not argv[0].startswith("-") and argv[0] not in commands:
            raise UsageError(
                f"Unknown command '{argv[0]}'. Available commands: "
                f"{', '.join(sorted(commands))}"
            )
        if argv[0] not in commands:
            return argv
        command, args = argv[0], self._before_separator(argv[1:])
        if any(arg in self.HELP_FLAGS for arg in args):
            return [command, "--", "--help"]
        params = self._params(command)
        named, positional = self._split(command, args, params)
        if len(named) + len(positional) > len(params):
            raise UsageError(
                f"Too many arguments for '{command}': it takes at most "
                f"{len(params)} ({self._options(params)})"
            )
        self._check_positional(command, named, positional, params)
        self._check_after_separator(command, argv[1 + len(args):])
        self.positional = self._positional_params(named, positional, params)
        return argv

    @staticmethod
    def _positional_params(named: set[str], positional: list[str],
                           params: list[str]) -> set[str]:
        """Parameters Fire fills with the positional arguments.

        Fire gives them, in order, to the parameters not given as
        options, wherever they are written on the command line.
        """
        free = [param for param in params if param not in named]
        return set(free[:len(positional)])

    def _check_positional(self, command: str, named: set[str],
                          positional: list[str], params: list[str]) -> None:
        """Reject a command name that Fire would take as a value.

        Fire gives the positional arguments, in order, to the parameters
        not given as options, wherever they are written: in ``index -m
        500 search``, ``search`` becomes ``raw_dir``. A command name there
        is almost surely a second command, except in free-text parameters.

        Raises:
            UsageError: A command name lands in a non free-text parameter.
        """
        free = [param for param in params if param not in named]
        for value, param in zip(positional, free):
            if value in self._commands() and \
                    param not in self.FREE_TEXT_PARAMS:
                raise UsageError(
                    f"'{value}' is a command, not a value for --{param} "
                    f"of '{command}'. Run one command at a time"
                )

    def _check_after_separator(self, command: str, rest: list[str]) -> None:
        """Reject what Fire would only refuse after running the command.

        - After ``-`` Fire calls the next words on the command's result;
          the commands return nothing, so it fails once the work is done.
        - After ``--`` Fire silently ignores anything that is not one of
          its own flags.

        Args:
            command: The command being run.
            rest: The separator and everything after it (may be empty).

        Raises:
            UsageError: Text after ``-``, or a non-Fire flag after ``--``.
        """
        if not rest:
            return
        separator, extra = rest[0], rest[1:]
        if separator == "-":
            if extra:
                raise UsageError(
                    f"Unexpected arguments after '-' for '{command}': "
                    f"{' '.join(extra)}. Commands cannot be chained"
                )
            return
        unknown = self._first_unknown_fire_flag(extra)
        if unknown is not None:
            raise UsageError(
                f"Unknown Fire flag '{unknown}' after '--' for "
                f"'{command}'. Valid flags: "
                f"{', '.join(sorted(self.FIRE_FLAGS))}"
            )

    def _first_unknown_fire_flag(self, args: list[str]) -> str | None:
        """First argument after ``--`` that is not a Fire flag or value.

        ``--separator`` and ``--completion`` may take a value, written
        with ``=`` or as the next argument.
        """
        position = 0
        while position < len(args):
            arg = args[position]
            name, equals, _ = arg.partition("=")
            if name not in self.FIRE_FLAGS:
                return arg
            if (name in self.FIRE_VALUE_FLAGS and not equals
                    and position + 1 < len(args)
                    and not args[position + 1].startswith("-")):
                position += 1
            position += 1
        return None

    def _commands(self) -> set[str]:
        """Names of the public methods (the available commands)."""
        return {
            name for name, _ in inspect.getmembers(self._component,
                                                   inspect.isfunction)
            if not name.startswith("_")
        }

    def _params(self, command: str) -> list[str]:
        """Parameter names of a command, without ``self``."""
        signature = inspect.signature(getattr(self._component, command))
        return [name for name in signature.parameters if name != "self"]

    def _before_separator(self, args: list[str]) -> list[str]:
        """Arguments up to the first separator (the rest is for Fire)."""
        for position, arg in enumerate(args):
            if arg in self.SEPARATORS:
                return args[:position]
        return args

    def _split(self, command: str, args: list[str],
               params: list[str]) -> tuple[set[str], list[str]]:
        """Resolve the options and collect the positional arguments.

        Returns:
            The parameters given as options and the positional arguments,
            in order.

        Raises:
            UsageError: An option does not match any parameter, or a
                non-boolean option has no value.
        """
        named: set[str] = set()
        positional: list[str] = []
        position = 0
        while position < len(args):
            arg = args[position]
            if self._is_option(arg):
                name, has_value = self._option_name(arg)
                param = self._resolve(command, arg, name, params)
                named.add(param)
                if not has_value:
                    if self._value_follows(args, position):
                        position += 1
                    elif not self._is_bool(command, param):
                        raise UsageError(
                            f"Option '{arg}' of '{command}' needs a value"
                        )
            else:
                positional.append(arg)
            position += 1
        return named, positional

    def _value_follows(self, args: list[str], position: int) -> bool:
        """Whether the argument after ``position`` can be its value."""
        return (position + 1 < len(args)
                and not self._is_option(args[position + 1]))

    def _is_bool(self, command: str, param: str) -> bool:
        """Whether a parameter is a flag Fire may set with no value.

        ``--verbose`` alone means ``True`` for a ``bool`` parameter.
        """
        signature = inspect.signature(getattr(self._component, command))
        return signature.parameters[param].annotation is bool

    def _is_option(self, arg: str) -> bool:
        """``-x``/``--x`` is an option; ``-3`` or ``-2.5`` is a value."""
        if not arg.startswith("-") or arg in self.SEPARATORS:
            return False
        try:
            float(arg)
        except ValueError:
            return True
        return False

    def _option_name(self, arg: str) -> tuple[str, bool]:
        """Parameter name written in an option and whether it has ``=``.

        ``--max-chunk-size=5`` → ``("max_chunk_size", True)``.
        """
        body = arg.lstrip("-")
        name, equals, _ = body.partition("=")
        return name.replace("-", "_"), bool(equals)

    def _resolve(self, command: str, arg: str, name: str,
                 params: list[str]) -> str:
        """Match an option to a parameter, as Fire does.

        Accepts the full name or, when it is unique, its first letter.

        Raises:
            UsageError: No parameter matches.
        """
        if name in params:
            return name
        if len(name) == 1:
            matches = [param for param in params if param.startswith(name)]
            if len(matches) == 1:
                return matches[0]
        raise UsageError(
            f"Unknown option '{arg}' for '{command}'. "
            f"Valid options: {self._options(params)}"
        )

    def _options(self, params: list[str]) -> str:
        """``--a, --b`` list for error messages."""
        return ", ".join(f"--{param}" for param in params) or "none"
