"""Assembly of the initial maze, drawer and output file from a config."""

import io
from contextlib import redirect_stdout

from mazegen import MazeGen
from src.readfile import Config
from src.drawer_gestor import MazeDrawer
from src.maze_output import export_hexa


def build_maze(config: Config) -> tuple[MazeGen, MazeDrawer, str | None]:
    """Generate, solve and export a maze from the loaded configuration.

    Args:
        config (Config): The loaded configuration to build the maze
            from.

    Returns:
        tuple[MazeGen, MazeDrawer, str | None]: The generated and
            solved maze, its drawer, and any warning message printed
            during generation (e.g. about the "42" pattern not fitting
            the requested size), or None if generation printed
            nothing.

    Raises:
        MazeGenError: If the maze can't be generated or solved, or if
            the output file can't be written.
    """
    maze = MazeGen(
        config.height,
        config.width,
        config.entry,
        config.exit,
        config.perfect,
        config.seed
    )
    logo_output = io.StringIO()
    with redirect_stdout(logo_output):
        maze.generate()
    logo_warning = logo_output.getvalue().strip() or None
    drawer: MazeDrawer = MazeDrawer(
            maze,
            config.color,
            config.terminal,
            config.playable
        )
    maze.solve(maze.entry, maze.exit)
    export_hexa(
        maze.grid,
        maze.entry,
        maze.exit,
        maze.solution,
        config.file
    )
    return maze, drawer, logo_warning
