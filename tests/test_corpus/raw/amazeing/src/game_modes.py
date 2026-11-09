"""Terminal game modes: normal, endless, and crazy.

Provides the helpers used by the terminal session to pick a game mode
and to drive endless mode's growing maze and crazy mode's countdown
timer.
"""

import io
from contextlib import redirect_stdout
from random import randint
import time

from mazegen import MazeGen
from src.readfile import Config
from src.drawer_gestor import MazeDrawer
from src.highscore import highscore


def prompt_game_mode() -> int:
    """Ask the user to pick a game mode from the terminal.

    Keeps prompting until the user enters "1", "2", "3" or "4".

    Returns:
        int: The chosen option: 1 (normal), 2 (endless), 3 (crazy) or
            4 (exit).
    """
    temp: str = ""
    while not (temp == "1" or temp == "2" or temp == "3" or temp == "4"):
        temp = input(
            "1: Normal mode\n"
            "2: Endless mode\n"
            "3: Crazy mode\n"
            "4: Exit\n"
        )
    return int(temp)


def setup_crazy_mode(
        config: Config
        ) -> tuple[MazeGen, MazeDrawer, float, int]:
    """Build the fixed 4x4 maze and initial state for crazy mode.

    Args:
        config (Config): The loaded configuration, used for its seed
            and display settings.

    Returns:
        tuple[MazeGen, MazeDrawer, float, int]: The generated maze,
            its drawer, the starting countdown timer (15.0 seconds)
            and the starting level (1).
    """
    maze = MazeGen(
                4,
                4,
                (0, 0),
                (3, 3),
                True,
                config.seed
            )
    maze.generate()
    maze.solve((0, 0), (1, 1))
    drawer = MazeDrawer(
                            maze,
                            config.color,
                            config.terminal,
                            config.playable
                        )
    timer: float = 15.0
    level: int = 1
    return maze, drawer, timer, level


def tick_timer(timer: float, level: int) -> float:
    """Print the countdown and level, and decrement the timer.

    If the timer reaches zero, prints the game-over message, records
    the reached level as a highscore, and exits the process.

    Args:
        timer (float): The seconds remaining before time runs out.
        level (int): The current crazy-mode level, used for display
            and as the recorded highscore on time out.

    Returns:
        float: The timer decremented by 0.1 seconds.
    """
    print(format(timer, ".1f"))
    print(f"You ar on level {level}", end="\n\n")
    timer -= 0.1
    if timer <= 0.0:
        print("\033[?1049l", end="", flush=True)
        print("TIME OUT!!")
        print(f"You reach level {level}!")
        highscore(level)
        time.sleep(2)
        exit(0)
    return timer


def pick_new_exit(
        maze: MazeGen,
        avoid: tuple[int, int]
        ) -> tuple[int, int]:
    """Pick a random cell to use as the next exit or entry.

    Keeps drawing coordinates until it finds one that isn't inside the
    "42" logo pattern and isn't 'avoid'.

    Args:
        maze (MazeGen): The maze to pick a cell from.
        avoid (tuple[int, int]): A (x, y) coordinate that must not be
            picked, typically the player's current position.

    Returns:
        tuple[int, int]: The (x, y) coordinates of the picked cell.
    """
    new_exit = (
        randint(0, maze.width - 1),
        randint(0, maze.height - 1)
    )
    while (maze.maze[new_exit[1]][new_exit[0]] == 31
            or new_exit == avoid):
        new_exit = (
            randint(0, maze.width - 1),
            randint(0, maze.height - 1)
        )
    return new_exit


def advance_endless_mode(
        maze: MazeGen,
        player_pos: tuple[int, int]
        ) -> None:
    """Regenerate the maze in place and solve it to a fresh exit.

    Used in endless mode once the player reaches the exit, so the
    same-sized maze gets a new layout and a new exit without changing
    the player's position.

    Args:
        maze (MazeGen): The maze to regenerate and solve, mutated
            in place.
        player_pos (tuple[int, int]): The (x, y) coordinates of the
            player, used as the new entry and to avoid picking it as
            the new exit.
    """
    with redirect_stdout(io.StringIO()):
        maze.regenerate()
    new_exit = pick_new_exit(maze, player_pos)
    maze.solve(player_pos, new_exit)


def announce_win() -> None:
    """Print the win message, pause briefly, and exit the process."""
    print("\033[?1049l", end="", flush=True)
    print("YOU WON!!")
    time.sleep(2)
    exit(0)


def grow_maze(
        maze: MazeGen,
        player_pos: tuple[int, int],
        config: Config
        ) -> tuple[MazeGen, MazeDrawer]:
    """Replace the maze with a new one, one cell taller and wider.

    Used in endless mode's growth variant: builds a fresh maze sized
    up by one in each dimension, entering where the player currently
    stands (moved to a valid cell if that one turns out to be inside
    the "42" pattern), and solves it to a new exit.

    Args:
        maze (MazeGen): The current maze, used only for its current
            size.
        player_pos (tuple[int, int]): The (x, y) coordinates of the
            player, used as the preferred new entry.
        config (Config): The loaded configuration, used for its seed
            and display settings.

    Returns:
        tuple[MazeGen, MazeDrawer]: The newly generated, solved maze
            and its drawer.
    """
    with redirect_stdout(io.StringIO()):
        placeholder_exit = pick_new_exit(maze, player_pos)
        maze = MazeGen(
            maze.height + 1,
            maze.width + 1,
            player_pos,
            placeholder_exit,
            True,
            config.seed
        )
    maze.generate()
    new_entry = player_pos
    if maze.maze[new_entry[1]][new_entry[0]] == 31:
        new_entry = pick_new_exit(maze, placeholder_exit)
    new_exit = pick_new_exit(maze, new_entry)
    maze.solve(new_entry, new_exit)
    drawer = MazeDrawer(
                            maze,
                            config.color,
                            config.terminal,
                            config.playable
                        )
    return maze, drawer
