"""Entry point: runs the simulation and handles Ctrl+C gracefully."""

from exceptions import ConfigError, PathError
from starter import StartProgram
from text_printer import Printer as Printer
from initiate_simulation import Initiator
from constants import TESTING
from show_simulation import Pyshow

import signal
import sys
from typing import Optional


class Application:
    """Loads a map, runs the simulation, and reports the outcome."""

    def __init__(self) -> None:
        """Set up the starter used to load and run a map."""
        self.starter = StartProgram()
        self.initiator: Optional[Initiator] = None

    def run(self) -> None:
        """Run the welcome flow, load the map, and print the outcome.

        Ends by offering the graphical animation (``_offer_graphical_
        view``).
        """
        self.starter.start_simulation()
        self.starter.printer.print_by_letter(
            f"Opening '{self.starter.file}'...",
            0.05,
            1.0,
            "\033[92m",
            TESTING
            )
        self.starter.open_config()
        self.initiator = Initiator(self.starter.parser.config)
        if not self.initiator.find_path_to_end():
            raise PathError("There is not a posible way to the end.")
        self.initiator.printer.print_by_letter(
            "Found a posible way on the map recived.",
            0.02,
            1,
            "\033[92m",
            TESTING
        )
        self.initiator.printer.erase_line(4, TESTING)
        self.initiator.printer.print_by_letter(
            "Starting the drone moves...",
            0.05,
            2,
            "\033[92m",
            TESTING
        )
        self.initiator.run_simulation()
        for step in self.initiator.turns:
            self.initiator.printer.print_by_letter(
                step,
                0.001,
                0.01
            )
        self.initiator.printer.print_by_letter(
            "Counting the movements...",
            0.05,
            1.5,
            testing=TESTING
        )
        total_moves: int = len(self.initiator.turns)
        self.initiator.printer.print_by_letter(
            f"The map was resolved in {total_moves} moves",
            0.02,
            2
        )
        self._offer_graphical_view()

    def _offer_graphical_view(self) -> None:
        """Ask whether to open the graphical animation, then run it.

        Re-prompts on anything other than 'y'/'n' instead of giving up.
        """
        assert self.initiator is not None
        printer = self.initiator.printer
        while True:
            printer.print_by_letter(
                "Show the graphical animation? (y/n): ",
                0.02,
                0.0,
                "\033[92m"
            )
            answer = input().strip().lower()
            if answer == "y":
                Pyshow(self.initiator).start()
                return
            if answer == "n":
                return


if __name__ == "__main__":
    app = Application()
    try:
        app.run()
    except ConfigError as e:
        Printer.print_by_letter(
            f"{type(e).__name__}: {e}",
            0.01,
            0.5,
            "\033[91m"
            )
        sys.exit(1)
    except KeyboardInterrupt:
        old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            print("\r\033[2K", end="")
            Printer.print_by_letter("Don't kill my f****** program...",
                                    0.4,
                                    0.0,
                                    "\033[91m",
                                    TESTING
                                    )
            sys.exit(0)
        finally:
            signal.signal(signal.SIGINT, old_handler)
    finally:
        printer = Printer()
        printer.print_by_letter(
            "End of the simulation...",
            0.02,
            0.0
        )
