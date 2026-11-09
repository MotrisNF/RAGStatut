"""Bootstraps the simulation: greets the user and loads the map file."""

from text_printer import Printer
from exceptions import FileError
from parser import Parser
from constants import TESTING


class StartProgram:
    """Coordinates the welcome prompt and map file loading."""

    def __init__(self) -> None:
        """Set up the printer and parser used by the welcome flow."""
        self.printer = Printer()
        self.parser = Parser()
        self.file: str = ""

    def start_simulation(self) -> None:
        """Greet the user, ask for a map filename, and check it opens.

        Raises:
            FileError: If no filename is entered, or the file cannot
                be opened.
        """
        self.printer.print_by_letter(
            "································",
            0.1,
            0.0,
            "\033[92m",
            TESTING
        )
        self.printer.erase_line(1, TESTING)
        self.printer.print_by_letter(
                                    "Welcome to the dron simulation...",
                                    0.02,
                                    1.0,
                                    "\033[92m",
                                    TESTING
                                    )
        self.printer.print_by_letter(
                                    "Introduce the name of the map: ",
                                    0.02,
                                    0.0,
                                    "\033[92m"
                                    )
        self.file = input().strip()
        if self.file == "":
            raise FileError("A name for the map file is mandatory.")

        try:
            with open(self.file, 'r'):
                ...

        except FileNotFoundError as e:
            raise FileError(f"The '{self.file}' file do not exist.") from e
        except OSError as e:
            raise FileError(f"The '{self.file}' file can't be opened") from e

        self.printer.erase_line(3, TESTING)

    def open_config(self) -> None:
        """Read and parse the previously validated map file.

        Raises:
            FileError: If the file can no longer be opened, isn't
                valid UTF-8 text, or its contents are malformed.
        """
        try:
            with open(self.file) as file:
                self.parser.read_file(file)
        except FileNotFoundError as e:
            raise FileError(f"The '{self.file}' file do not exist.") from e
        except UnicodeDecodeError as e:
            raise FileError(
                f"The '{self.file}' file is not a valid UTF-8 text "
                "file."
            ) from e
        except OSError as e:
            raise FileError(
                f"The '{self.file}' file can't be opened"
            ) from e
