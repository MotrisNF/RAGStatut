"""Benchmarks the solver against the subject's reference maps.

Run via ``make performance``. Loads each easy/medium/hard/challenger
reference map, solves it, and reports the turn count actually
achieved against the subject's own reference targets (Chapter VII.7)
-- without printing the turn-by-turn movement log, just the totals,
colored green when a target is met and red when it isn't.

No wildcards are used to discover the maps: each benchmark names its
file explicitly, so the reference list here is the single source of
truth for what "performance" checks.
"""

from dataclasses import dataclass

from constants import (
    ANSI_GREEN, ANSI_RED, PERFORMANCE_PRINT_PAUSE, PERFORMANCE_PRINT_SPEED
)
from exceptions import ConfigError, PathError
from initiate_simulation import Initiator
from parser import Parser
from text_printer import Printer

import initiate_simulation
import os
import sys
from time import sleep

# This report's own output should stay a clean, one-line-per-map
# table regardless of how TESTING is set for a normal run of main.py,
# so the noisy startup messages Initiator prints are always silenced
# here, independent of constants.TESTING.
initiate_simulation.TESTING = True


@dataclass(frozen=True)
class Benchmark:
    """One reference map to solve and check against a turn target.

    Attributes:
        path: Map file to load.
        label: Human-readable name, matching the subject's own list.
        target_turns: Turns the subject expects a good solver to
            reach. For ``optional`` maps this is a record to beat
            rather than a mandatory target.
        optional: True for the bonus Challenger map, which doesn't
            affect grading even if its target is missed.
    """

    path: str
    label: str
    target_turns: int
    optional: bool = False


class PerformanceReport:
    """Solves each reference map and prints its result against target."""

    _MAPS_DIR = "maps"
    _REQUIRED_SUBDIRS = ("easy", "medium", "hard", "challenger")

    _GROUPS: list[tuple[str, list[Benchmark]]] = [
        ("Easy Maps", [
            Benchmark(
                "maps/easy/01_linear_path.txt",
                "Linear path with 2 drones", 6
            ),
            Benchmark(
                "maps/easy/02_simple_fork.txt",
                "Simple fork with 4 drones", 8
            ),
            Benchmark(
                "maps/easy/03_basic_capacity.txt",
                "Basic capacity with 4 drones", 6
            ),
        ]),
        ("Medium Maps", [
            Benchmark(
                "maps/medium/01_dead_end_trap.txt",
                "Dead end trap with 5 drones", 12
            ),
            Benchmark(
                "maps/medium/02_circular_loop.txt",
                "Circular loop with 6 drones", 15
            ),
            Benchmark(
                "maps/medium/03_priority_puzzle.txt",
                "Priority puzzle with 5 drones", 12
            ),
        ]),
        ("Hard Maps", [
            Benchmark(
                "maps/hard/01_maze_nightmare.txt",
                "Maze nightmare with 8 drones", 30
            ),
            Benchmark(
                "maps/hard/02_capacity_hell.txt",
                "Capacity hell with 12 drones", 35
            ),
            Benchmark(
                "maps/hard/03_ultimate_challenge.txt",
                "Ultimate challenge with 15 drones", 45
            ),
        ]),
        ("Challenger Map (optional -- for exceptional implementations)", [
            Benchmark(
                "maps/challenger/01_the_impossible_dream.txt",
                "The Impossible Dream with 25 drones", 45,
                optional=True
            ),
        ]),
    ]

    def __init__(self) -> None:
        """Start assuming every mandatory benchmark will pass."""
        self._all_passed = True

    def _missing_directories(self) -> list[str]:
        """Names of required maps subdirectories that don't exist.

        Only meaningful once ``self._MAPS_DIR`` itself is confirmed to
        exist -- see ``run``.
        """
        return [
            name for name in self._REQUIRED_SUBDIRS
            if not os.path.isdir(os.path.join(self._MAPS_DIR, name))
        ]

    def _print(self, text: str, color: str) -> None:
        """Print one report line letter by letter, like main.py does."""
        Printer.print_by_letter(
            text, PERFORMANCE_PRINT_SPEED, PERFORMANCE_PRINT_PAUSE, color
        )

    def _solve(self, path: str) -> int:
        """Load and solve one map, returning its total turn count.

        Raises:
            ConfigError: If the map file can't be parsed.
            PathError: If no route exists, or the scheduler deadlocks.
        """
        with open(path) as file:
            config = Parser().read_file(file)
        initiator = Initiator(config)
        if not initiator.find_path_to_end():
            raise PathError("no route between start and end")
        initiator.run_simulation()
        return len(initiator.turns)

    def _report_one(self, benchmark: Benchmark) -> None:
        """Solve one benchmark and print its colored result line."""
        try:
            turns = self._solve(benchmark.path)
        except (ConfigError, PathError, OSError) as error:
            self._print(
                f"◦ {benchmark.label}: could not solve ({error})",
                ANSI_RED
            )
            if not benchmark.optional:
                self._all_passed = False
            return

        passed = turns <= benchmark.target_turns
        if not passed and not benchmark.optional:
            self._all_passed = False

        goal = (
            f"reference record: {benchmark.target_turns} turns"
            if benchmark.optional
            else f"target ≤{benchmark.target_turns} turns"
        )
        status = "PASS" if passed else "FAIL"
        color = ANSI_GREEN if passed else ANSI_RED
        self._print(
            f"◦ {benchmark.label}: {goal} "
            f"-- solved in {turns} turns [{status}]",
            color
        )

    def run(self) -> bool:
        """Print every group's results.

        Returns:
            True unless a required maps subdirectory is missing (a
            mandatory benchmark falling short of its target does not
            fail the report -- it's meant to be read, not to break
            the build).
        """
        if not os.path.isdir(self._MAPS_DIR):
            self._print(
                f"Missing '{self._MAPS_DIR}/' directory.", ANSI_RED
            )
            return False
        missing = self._missing_directories()
        if missing:
            self._print(
                f"Missing subdirectories under '{self._MAPS_DIR}/': "
                f"{', '.join(missing)}",
                ANSI_RED
            )
            return False

        for title, benchmarks in self._GROUPS:
            self._print(f"{title}:", "\033[0m")
            sleep(0.5)
            for benchmark in benchmarks:
                self._report_one(benchmark)
            sleep(0.5)
            print()

        sleep(2)
        summary = (
            "All mandatory targets met." if self._all_passed
            else "Some mandatory targets were missed."
        )

        self._print(summary, ANSI_GREEN if self._all_passed else ANSI_RED)
        return True


if __name__ == "__main__":
    sys.exit(0 if PerformanceReport().run() else 1)
