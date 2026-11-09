"""Parsing and validation of the config file passed on the command line."""

from sys import argv
from mazegen import MazeGenError

VALID_KEYS = {
    "WIDTH", "HEIGHT", "ENTRY", "EXIT", "OUTPUT_FILE",
    "SEED", "PERFECT", "PLAYABLE", "COLOR", "TERMINAL"
}
OPTIONAL_KEYS = {"PERFECT", "PLAYABLE", "COLOR", "TERMINAL", "SEED"}
REQUIRED_KEYS = VALID_KEYS - OPTIONAL_KEYS


def parse_int(value: str, field: str) -> int:
    """Parse 'value' as an int, raising a MazeGenError on failure.

    Args:
        value (str): The raw string to parse.
        field (str): The human-readable name of the field 'value'
            came from, used in the error message.

    Returns:
        int: The parsed integer.

    Raises:
        MazeGenError: If 'value' isn't a valid integer literal.
    """
    try:
        return int(value)
    except ValueError:
        raise MazeGenError(f"{field} must be an integer, got '{value}'")


class Config():
    """Parsed and validated game configuration.

    Reads the config file path from 'sys.argv', parses its "KEY=value"
    lines, validates them, and exposes the results as read-only
    properties. Exits the process with status 1 and an explanatory
    message on any parsing or validation failure.

    Attributes:
        file (str): The path of the maze output file.
        seed (int): The random seed to generate the maze with.
        height (int): The height of the maze in cells.
        width (int): The width of the maze in cells.
        entry (tuple[int, int]): The (x, y) coordinates of the entry.
        exit (tuple[int, int]): The (x, y) coordinates of the exit.
        perfect (bool): Whether the maze must be perfect (no loops).
        playable (bool): Whether the maze is used as a playable board.
        color (bool): Whether to render the maze in color.
        terminal (bool): Whether to run the terminal session instead
            of the mlx graphical one.
    """

    def __init__(self) -> None:
        """Read, parse and validate the config file named in 'sys.argv'.

        Raises:
            SystemExit: If no config file is given, if it can't be
                read, or if its contents fail validation (unknown or
                duplicated keys, missing required keys, malformed
                values, or maze dimensions too large for the display).
        """
        self._seed: int = 42
        self._height: int = 0
        self._width: int = 0
        self._entry_coords: list[str] = []
        self._exit_coords: list[str] = []
        self._entry: tuple[int, int] = (0, 0)
        self._exit: tuple[int, int] = (0, 0)
        self._perfect: bool = False
        self._output_file: str = ""
        self._playable: bool = False
        self._color: bool = False
        self._terminal: bool = True
        if len(argv) == 2:
            try:
                with open(argv[1], 'r') as config:
                    parsed: list[str] = []
                    for num, line in enumerate(config):
                        if line.startswith(" "):
                            raise MazeGenError(
                                f"Line {num + 1}: Can't start with spaces."
                            )
                        if line == "\n":
                            raise MazeGenError(
                                f"Line {num + 1}: Found empty line."
                            )
                        parsed.append(line.strip())
                seen_keys: list[str] = []
                for num, line in enumerate(parsed):
                    if line == "" or line.startswith("#"):
                        continue
                    if "=" not in line or ' ' in line:
                        raise MazeGenError(
                            f"Line {num + 1}: Invalid line in config: {line}"
                            )
                    key = line.split("=")[0]
                    if key not in VALID_KEYS:
                        raise MazeGenError(
                            f"Line {num + 1}: Unknown key in config: {key}"
                            )
                    if key in seen_keys:
                        raise MazeGenError(
                            f"Duplicated key in config: {key}"
                            )
                    seen_keys.append(key)
                missing_keys = REQUIRED_KEYS - set(seen_keys)
                if missing_keys:
                    raise MazeGenError(
                        "Missing required key(s): "
                        f"{', '.join(sorted(missing_keys))}"
                        )
                for line in parsed:
                    upper_line = line.upper()
                    if upper_line.startswith("WIDTH="):
                        self._width = parse_int(
                            upper_line.removeprefix("WIDTH="), "WIDTH"
                            )
                    if upper_line.startswith("HEIGHT="):
                        self._height = parse_int(
                            upper_line.removeprefix("HEIGHT="), "HEIGHT"
                            )
                    if upper_line.startswith("ENTRY="):
                        entry_str = upper_line.removeprefix("ENTRY=")
                        self._entry_coords = entry_str.split(",")
                    if upper_line.startswith("EXIT="):
                        exit_str = upper_line.removeprefix("EXIT=")
                        self._exit_coords = exit_str.split(",")
                    if upper_line.startswith("OUTPUT_FILE="):
                        self._output_file = line[len("OUTPUT_FILE="):]
                    if upper_line.startswith("SEED="):
                        seed_str = upper_line.removeprefix("SEED=")
                        if seed_str != "":
                            self._seed = parse_int(
                                seed_str, "SEED"
                                )
                    if upper_line.startswith("PERFECT="):
                        perfect_str = upper_line.removeprefix("PERFECT=")
                        self._perfect = False
                        if perfect_str.capitalize() == "True":
                            self._perfect = True
                        elif perfect_str.capitalize() == "False":
                            self._perfect = False
                        elif perfect_str != "":
                            raise MazeGenError("PERFECT can be True or False")
                    if upper_line.startswith("PLAYABLE="):
                        playable_str = upper_line.removeprefix("PLAYABLE=")
                        if playable_str.capitalize() == "True":
                            self._playable = True
                        elif playable_str.capitalize() == "False":
                            self._playable = False
                        elif playable_str != "":
                            raise MazeGenError(
                                "PLAYABLE can be True or False"
                            )
                    if upper_line.startswith("COLOR="):
                        color_str = upper_line.removeprefix("COLOR=")
                        if color_str.capitalize() == "True":
                            self._color = True
                        elif color_str.capitalize() == "False":
                            self._color = False
                        elif color_str != "":
                            raise MazeGenError("COLOR can be True or False")
                    if upper_line.startswith("TERMINAL="):
                        terminal_str = upper_line.removeprefix("TERMINAL=")
                        if terminal_str.capitalize() == "True":
                            self._terminal = True
                        elif terminal_str.capitalize() == "False":
                            self._terminal = False
                        elif terminal_str != "":
                            raise MazeGenError("TERMINAL can be True or False")
                if len(self._entry_coords) != 2 or len(self._exit_coords) != 2:
                    raise MazeGenError(
                        "The coordinates only accept 2 params (n,n)"
                        )
                entry_x = parse_int(self._entry_coords[0], "ENTRY x")
                entry_y = parse_int(self._entry_coords[1], "ENTRY y")
                exit_x = parse_int(self._exit_coords[0], "EXIT x")
                exit_y = parse_int(self._exit_coords[1], "EXIT y")
                self._entry = (entry_x, entry_y)
                self._exit = (exit_x, exit_y)
                if self._output_file == "":
                    self._output_file = input(
                        "Enter a name for the output file: "
                        )
                if self._output_file == "":
                    raise MazeGenError("The file need to be saved!")
                if self.height > 28 or self.width > 60:
                    raise MazeGenError(
                        "This dimensions are not accepted. "
                        "The output is bigger than the size window."
                    )
                if ((self.height > 25 or self.width > 40)
                        and self.terminal):
                    raise MazeGenError(
                        "This dimensions are not accepted. "
                        "The output is bigger than the size window."
                    )
            except (MazeGenError, OSError) as e:
                print(f"Fail on the configuration: {e}")
                exit(1)
        else:
            print("The second argument is the name of the config file")
            print("Usage: python3 a-maze-ing.py <config-file>")
            exit(1)

    @property
    def file(self) -> str:
        """str: The path of the maze output file."""
        return self._output_file

    @property
    def seed(self) -> int:
        """int: The random seed to generate the maze with."""
        return self._seed

    @property
    def height(self) -> int:
        """int: The height of the maze in cells."""
        return self._height

    @property
    def width(self) -> int:
        """int: The width of the maze in cells."""
        return self._width

    @property
    def entry(self) -> tuple[int, int]:
        """tuple[int, int]: The (x, y) coordinates of the entry."""
        return self._entry

    @property
    def exit(self) -> tuple[int, int]:
        """tuple[int, int]: The (x, y) coordinates of the exit."""
        return self._exit

    @property
    def perfect(self) -> bool:
        """bool: Whether the maze must be perfect (no loops)."""
        return self._perfect

    @property
    def playable(self) -> bool:
        """bool: Whether the maze is used as a playable board."""
        return self._playable

    @property
    def color(self) -> bool:
        """bool: Whether to render the maze in color."""
        return self._color

    @property
    def terminal(self) -> bool:
        """bool: Whether to run the terminal session, not the mlx one."""
        return self._terminal
