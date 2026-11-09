"""Module of the maze generator class

This module defines the MazeGen class, that can generate,
regenerate and solve a maze
"""

import random
from .exceptions import MazeGenError
from .maze_generator_dependences import get_maze
from .hexa_dependences import build_hexa_lines
from .maze_solver_dependences import MazeSolver


class MazeGen:
    """Generate, regenerate and solve a maze.

    This class is used for generate mazes for a height and a width given.
    The mazes can be perfect (one possible way) or non perfect(multiple ways).
    The maze will be different for each seed, but replicable with the same
    seed.

    Attributes:
        height (int): The height of the maze in squares.
        width (int): The width of the maze in squares.
        maze (list[list[int]]): Bidimensional array with the value of
            the cells in bits.
        solution (list[tuple[list[tuple[int,int]], str]]): Each index
            from the list contains a tuple. Each touple, contains a list
            of tuples(int, int), that contains the coordinates of the
            solution in order, and a str, that contains the solution
            writen in cardinal direction.
        grid (list[str]): The construction of the maze, square to square,
            on hexadecimal, giving the wall aviables on each square.
            Format for closed walls: N: 0001, E: 0010, S: 0100, W: 1000
        entry (tuple[int, int]): The (x, y) coordinates of the maze entry.
        exit (tuple[int, int]): The (x, y) coordinates of the maze exit.
    """
    def __init__(
            self,
            height: int,
            width: int,
            entry: tuple[int, int],
            exit: tuple[int, int],
            perfect: bool = False,
            seed: int | None = None
            ) -> None:
        """Inicialize the maze generator.

        Args:
            height (int): The height of the maze in squares.
            width (int): The width of the maze in squares.
            entry (tuple[int, int]): The (x, y) coordinates of the maze
                entry, used to solve the maze once generated.
            exit (tuple[int, int]): The (x, y) coordinates of the maze
                exit, used to solve the maze once generated.
            perfect (bool): That defines if the maze will be have one or more
                solution. False is by default.
            seed (int | None): This number will decide the random number
                generator. A positive number it's recomended. If None
                (default), a random seed is picked on each call, instead of
                once when the module is first imported.

        Raises:
            MazeGenError: If the height or the width are negative
                numbers, if 'entry'/'exit' fall outside the maze bounds
                (0 <= x < width, 0 <= y < height), or if 'entry' and
                'exit' are the same cell.
        """
        if height <= 0 and width <= 0:
            raise MazeGenError(
                "The height and the width need to be greaters than 0"
                )
        if height <= 0:
            raise MazeGenError("The height need to be greaters than 0")
        if width <= 0:
            raise MazeGenError("The width need to be greaters than 0")
        self._height = height
        self._width = width
        self._check_bounds(entry, "entry")
        self._check_bounds(exit, "exit")
        self._check_distinct(entry, exit)
        self._perfect = perfect
        self._seed = seed if seed is not None else random.randint(1, 1000)
        self._entry = entry
        self._exit = exit
        self._maze: list[list[int]] = []
        self._grid: list[str] = []
        self._solve: MazeSolver = MazeSolver(
            [],
            (self._width, self._height),
            entry,
            exit
        )
        self._solution: list[tuple[list[tuple[int, int]], str]] = []
        self._generated: bool = False

    def _check_bounds(self, coord: tuple[int, int], label: str) -> None:
        """Raise if 'coord' falls outside the maze's width/height.

        Used to validate 'entry'/'exit' independently of whatever
        checks the caller may or may not have already done, so MazeGen
        never has to trust external validation (e.g. a config file
        parser) to keep it from crashing on bad coordinates.

        Args:
            coord (tuple[int, int]): The (x, y) coordinate to check.
            label (str): Human-readable name used in the error message,
                e.g. "entry" or "exit".

        Raises:
            MazeGenError: If 'coord' is outside the maze bounds
                (0 <= x < width, 0 <= y < height).
        """
        x, y = coord
        if not (0 <= x < self._width and 0 <= y < self._height):
            raise MazeGenError(f"The {label} {coord} is outside the maze")

    def _check_distinct(
            self,
            entry: tuple[int, int],
            exit: tuple[int, int]
            ) -> None:
        """Raise if 'entry' and 'exit' are the same cell.

        Validated independently of external callers for the same reason
        as '_check_bounds': MazeGen shouldn't rely on a config parser or
        anything else to keep this invariant.

        Args:
            entry (tuple[int, int]): The (x, y) coordinates of the entry.
            exit (tuple[int, int]): The (x, y) coordinates of the exit.

        Raises:
            MazeGenError: If 'entry' equals 'exit'.
        """
        if entry == exit:
            raise MazeGenError(
                f"The entry and the exit can't be the same position: "
                f"{entry}"
            )

    def generate(self) -> None:
        """Generates a maze.

        Generates a mazewith the same seed given in the creation of the
        generator. That will actualize the _maze and the _grid values, and
        refresh the internal solver data used later by 'solve'. It doesn't
        compute the solution itself, so calling this repeatedly (e.g. while
        trying out seeds) stays cheap.
        """
        self._maze = get_maze(
            self._height,
            self._width,
            self._seed,
            self._perfect
        )
        self._grid = build_hexa_lines(
            self._height,
            self._width,
            self._maze
        )
        self._solve.set_data(
            self._grid,
            (self._width, self._height),
            self._entry,
            self._exit
        )
        self._generated = True

    def regenerate(
            self,
            entry: tuple[int, int] | None = None,
            exit: tuple[int, int] | None = None,
            seed: int = -1
            ) -> None:
        """Generates a maze with the new seed given.

        That will actualizethe _maze and the _grid values.
        If a seed it's not given, the last seed will be used.

        Any solution computed by a previous 'solve' call is cleared here,
        since it belonged to the maze layout that just got replaced —
        call 'solve' again afterward to get a solution for the new one.

        Raises:
            MazeGenError: If 'generate' hasn't been called yet, if a new
                'entry'/'exit' falls outside the maze bounds, or if the
                resulting entry and exit end up being the same cell.
        """
        if not self._generated:
            raise MazeGenError(
                "The maze must be generated before it can be regenerated"
            )
        if seed == -1:
            self._seed += random.randint(1, 50)
        else:
            self._seed = seed
        if entry is not None:
            self._check_bounds(entry, "entry")
            self._entry = entry
        if exit is not None:
            self._check_bounds(exit, "exit")
            self._exit = exit
        self._check_distinct(self._entry, self._exit)

        self._maze = get_maze(
                    self._height,
                    self._width,
                    self._seed,
                    self._perfect
                )
        self._grid = build_hexa_lines(
                    self._height,
                    self._width,
                    self._maze
                )
        self._solve.set_data(
                    self._grid,
                    (self._width, self._height),
                    self._entry,
                    self._exit
                )
        # The maze layout just changed, so any solution from before is for
        # a maze that no longer exists — drop it instead of leaving
        # 'solution' silently mismatched with the new 'maze'/'grid'.
        self._solution = []

    def solve(
            self,
            entry: tuple[int, int],
            exit: tuple[int, int]
            ) -> None:
        """Solves the maze for the given entry and exit points.

        Reuses the grid data already refreshed by 'generate' or
        'regenerate', so it only pays the cost of the path search itself.
        Updates the '_solution' value with the result.

        Args:
            entry (tuple[int, int]): The (x, y) coordinates of the entry.
            exit (tuple[int, int]): The (x, y) coordinates of the exit.

        Raises:
            MazeGenError: If 'generate' hasn't been called yet, if
                'entry' or 'exit' fall outside the maze, if they're the
                same cell, or if no path exists between them (e.g. one
                of them is locked inside the '42' pattern, or they're
                otherwise unreachable from each other).
        """
        if not self._generated:
            raise MazeGenError(
                "The maze must be generated before it can be solved"
            )
        self._check_bounds(entry, "entry")
        self._check_bounds(exit, "exit")
        self._check_distinct(entry, exit)
        self._entry = entry
        self._exit = exit
        self._solve.set_data(
            self._grid,
            (self._width, self._height),
            entry,
            exit
        )
        self._solution = self._solve.path_finder()
        if not self._solution:
            raise MazeGenError(
                f"No path exists between entry {entry} and exit {exit}"
            )

    @property
    def height(self) -> int:
        """Give the height of the maze.

        Returns:
            int: the height of the maze.
        """
        return self._height

    @property
    def width(self) -> int:
        """Give the width of the maze.

        Returns:
            int: The widht of the maze.
        """
        return self._width

    @property
    def grid(self) -> list[str]:
        """Give the grid of the maze, on hexadecimal format.
            N: 0001, E:0010, S:0100, W:1000

        Returns:
            list[str]: A list of str, one for row of the maze
        """
        return self._grid

    @property
    def maze(self) -> list[list[int]]:
        """Give the value of the maze, on integuer values.

        Returns:
            list[list[int]]: The value of each cell of the maze, needed to
                save the information of cells in bits, at the same way of
                grid.
        """
        return self._maze

    @property
    def entry(self) -> tuple[int, int]:
        """Give the entry point of the maze.

        Returns:
            tuple[int, int]: The (x, y) coordinates of the entry.
        """
        return self._entry

    @property
    def exit(self) -> tuple[int, int]:
        """Give the exit point of the maze.

        Returns:
            tuple[int, int]: The (x, y) coordinates of the exit.
        """
        return self._exit

    @property
    def solution(self) -> list[tuple[list[tuple[int, int]], str]]:
        """Give the positions and the path for the entry point given in solve
            method, to the exit point.

        Returns:
            list[tuple[list[tuple[int, int]], str]]: This list contains all
            the posibles solutions of the maze (always the shortest). Each
            index of the list, contains a tuple, with a list of the x,y
            coordinates for aall the path, and the path in a string writen
            with the cardinals points.
        """
        return self._solution
