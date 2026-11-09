"""Entry point of the A-maze-ing game.

Reads the configuration file, builds the maze, and hands control to
either the mlx graphical session or the terminal session depending on
the configuration.
"""

from mazegen import MazeGen, MazeGenError
from src import Config, build_maze, run_terminal_session, MlxPrinter


def run_mlx_session(maze: MazeGen, color: bool) -> None:
    """Open and run the mlx graphical window for the given maze.

    Args:
        maze (MazeGen): The already generated maze to display.
        color (bool): Whether to render the maze in color.
    """
    printer = MlxPrinter(
        maze,
        color,
        (0, 0)
    )
    printer.run()


def main() -> None:
    """Load the configuration, build the maze, and start the session.

    Exits with status 1 and prints an explanatory message if the
    configuration file is invalid or the maze can't be built.
    """
    try:
        config = Config()
    except MazeGenError as e:
        print(f"Error on the configuration file: {e}")
        exit(1)
    try:
        maze, drawer, logo_warning = build_maze(config)
    except MazeGenError as e:
        print(f"Can't create the maze: {e}")
        exit(1)

    if config.terminal:
        run_terminal_session(config, maze, drawer, logo_warning)
    else:
        run_mlx_session(maze, config.color)


if __name__ == "__main__":
    main()
