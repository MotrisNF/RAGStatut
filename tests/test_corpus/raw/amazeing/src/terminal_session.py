"""Main interactive loop for the ASCII terminal game session."""

import io
from contextlib import redirect_stdout
import time

from mazegen import MazeGen
from src.readfile import Config
from src.drawer_gestor import MazeDrawer
from src.terminal_io import raw_mode, get_key
from src.player_movement import DIRECTIONS, decide_move
from src.color_controls import change_color, randomize_colors
from src.game_modes import (
    prompt_game_mode,
    setup_crazy_mode,
    tick_timer,
    advance_endless_mode,
    announce_win,
    grow_maze,
)
from src.highscore import print_leaderboard


def run_terminal_session(
        config: Config,
        maze: MazeGen,
        drawer: MazeDrawer,
        logo_warning: str | None
        ) -> None:
    """Run the terminal game loop until the user quits or wins.

    Reads one key per frame, moves the player, and redraws the maze,
    handling the viewer commands (toggle path, regenerate, cycle
    solutions, recolor) when not in playable mode, and the game-mode
    logic (win detection, endless regeneration, crazy timer, maze
    growth) when in playable mode. Restores the terminal screen and
    prints the leaderboard on exit if crazy mode was played.

    Args:
        config (Config): The loaded configuration.
        maze (MazeGen): The maze to play or view, already generated
            and solved.
        drawer (MazeDrawer): The drawer used to render 'maze' each
            frame.
        logo_warning (str | None): A warning message to print above
            the maze each frame (e.g. about the "42" pattern), or
            None to print nothing.
    """
    player_x: int = 0
    player_y: int = 0
    heading: str | None = None
    epilepsy: bool = False
    path = False
    actual_path: int = 0
    game_mode: int = 0

    if config.playable:
        game_mode = prompt_game_mode()
        if game_mode == 4:
            print("Exit sucesfully")
            exit(0)
        if config.playable and game_mode == 3:
            maze, drawer, timer, level = setup_crazy_mode(config)

    print("\033[?1049h\033[2J", end="", flush=True)
    try:
        with raw_mode():
            while 1:
                key = get_key()
                cell = maze.maze[player_y][player_x]
                move_dir = decide_move(cell, heading, key)
                if move_dir is not None:
                    dx, dy, _ = DIRECTIONS[move_dir]
                    player_x += dx
                    player_y += dy
                    heading = move_dir
                if key == 'p' and not config.playable:
                    path = not path
                if key == 'r' and not config.playable:
                    actual_path = 0
                    with redirect_stdout(io.StringIO()):
                        maze.regenerate()
                    maze.solve(maze.entry, maze.exit)
                if key == 'n' and not config.playable:
                    actual_path += 1
                    if actual_path >= len(maze.solution):
                        actual_path = 0
                if key == 'q':
                    break
                if key == 'e':
                    epilepsy = not epilepsy
                if key == 'm' and config.color:
                    change_color(drawer)
                    time.sleep(1)
                if epilepsy:
                    randomize_colors(drawer)
                print("\033[H\033[J", end="")
                if logo_warning:
                    print(logo_warning)
                drawer.draw(
                            (player_x, player_y),
                            maze.solution[actual_path][0],
                            path
                            )
                if config.playable:
                    if game_mode == 3:
                        timer = tick_timer(timer, level)
                    print("'w,a,s,d' - Move the player.")
                print()
                if config.color:
                    print("'e' - Try me!")
                    print("'m' - Select the colors of the maze.")
                if not config.playable:
                    print("'p' - Show the shorter way.")
                    print("'n' - Show the next shorter way.")
                if not config.playable:
                    print("'r' - Regenerate the maze.")
                print("'q' - Exit.")
                print()
                if (config.playable
                        and player_x == maze.exit[0]
                        and player_y == maze.exit[1]):
                    if game_mode == 2:
                        advance_endless_mode(maze, (player_x, player_y))
                        actual_path = 0
                        heading = None
                    elif game_mode == 1:
                        announce_win()
                    else:
                        actual_path = 0
                        maze, drawer = grow_maze(
                            maze, (player_x, player_y), config
                        )
                        player_x, player_y = maze.entry
                        timer += 2.5
                        level += 1
                        heading = None
    finally:
        print("\033[?1049l", end="")
        if config.playable and game_mode == 3:
            print("\n\n\n")
            print()
            print("Leaderboard:")

            print_leaderboard()
