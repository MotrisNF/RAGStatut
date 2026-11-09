"""Interactive terminal controls for the maze's ASCII color palette."""

from random import randint

from src.drawer_gestor import MazeDrawer
from src.terminal_io import cooked_input

BG_COLORS: list[str] = [f"\033[48;5;{i}m" for i in range(256)]


def change_color(drawer: MazeDrawer) -> None:
    """Prompt the user for a maze part and a color, then apply it.

    Reads the target part (wall, logo, path or passage) and a value
    between 0 and 255 from the terminal, and updates the matching
    color on 'drawer'. Prints an error message and leaves 'drawer'
    unchanged if the input isn't a valid value in that range.

    Args:
        drawer (MazeDrawer): The drawer whose color to update.
    """
    part = cooked_input(
            "1: wall, 2: logo, 3: path, 4: passage: "
        )
    valid_input: bool = True
    try:
        real_value = int(part)
        if real_value > 4 or real_value < 1:
            raise ValueError
    except ValueError:
        print("Invalid input.")
        valid_input = False
    if valid_input:
        value = cooked_input(
            "Inster a value between 0-255: "
            )
        try:
            color_value = int(value)
            if color_value > 255 or color_value < 0:
                raise ValueError()
            if part == "1":
                drawer.wall_color = BG_COLORS[color_value]
                print("Done!")
            elif part == "2":
                drawer.logo_color = BG_COLORS[color_value]
                print("Done!")
            elif part == "3":
                drawer.path_color = BG_COLORS[color_value]
                print("Done!")
            elif part == "4":
                drawer.passage_color = BG_COLORS[
                    color_value
                ]
                print("Done!")
            else:
                print("Not yet!")
        except ValueError:
            print("Instert a valid number.")


def randomize_colors(drawer: MazeDrawer) -> None:
    """Assign a random background color to each maze part.

    Args:
        drawer (MazeDrawer): The drawer whose colors to randomize.
    """
    drawer.wall_color = BG_COLORS[randint(0, 255)]
    drawer.logo_color = BG_COLORS[randint(0, 255)]
    drawer.path_color = BG_COLORS[randint(0, 255)]
    drawer.passage_color = BG_COLORS[randint(0, 255)]
