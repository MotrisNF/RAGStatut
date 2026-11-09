"""Rendering of a MazeGen maze to the terminal, in ASCII or color.

Draws the maze walls, entry, exit, player position and, optionally,
the solution path, either as plain box-drawing characters or as
colored background blocks.
"""

from mazegen import MazeGen, MazeGenError
from typing import Callable


RESET = "\033[0m"
BG_PASSAGE = "\033[100m"
BG_WALL = "\033[40m"
BG_RED = "\033[48;5;52m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_BLUE = "\033[44m"
BG_LOGO = "\033[45m"

NO = 1
E = 2
SO = 4
WE = 8
VISITED = 16
LOGO = 31

CORNER_CHARS = {
    (False, False, False, False): " ",
    (True, False, False, False): "║",
    (False, True, False, False): "║",
    (True, True, False, False): "║",
    (False, False, True, False): "═",
    (False, False, False, True): "═",
    (False, False, True, True): "═",
    (True, False, True, False): "╝",
    (True, False, False, True): "╚",
    (False, True, True, False): "╗",
    (False, True, False, True): "╔",
    (True, True, True, False): "╣",
    (True, True, False, True): "╠",
    (False, True, True, True): "╦",
    (True, False, True, True): "╩",
    (True, True, True, True): "╬",
}


class MazeDrawer:
    """Render a maze to the terminal, in ASCII or ANSI color.

    Attributes:
        maze (MazeGen): The maze to render.
        color (bool): Whether to render using colored background
            blocks instead of plain box-drawing characters.
        terminal (bool): Whether the drawer is allowed to draw (only
            terminal sessions support drawing).
        playable (bool): Whether to render the player marker.
    """

    def __init__(
            self,
            maze: MazeGen,
            color: bool = False,
            terminal: bool = True,
            playable: bool = False,
            ) -> None:
        """Initialize the drawer with default colors.

        Args:
            maze (MazeGen): The maze to render.
            color (bool): Whether to render in color. False by
                default.
            terminal (bool): Whether drawing is allowed. True by
                default.
            playable (bool): Whether to render the player marker.
                False by default.
        """
        self.maze = maze
        self.color = color
        self.terminal = terminal
        self.playable = playable
        self._logo_color = BG_LOGO
        self._wall_color = BG_WALL
        self._path_color = BG_RED
        self._passage_color = BG_PASSAGE
        self._path_drawable = True

    @property
    def logo_color(self) -> str:
        """str: The background color code used for the "42" logo cells."""
        return self._logo_color

    @logo_color.setter
    def logo_color(self, value: str) -> None:
        """Set the background color code used for the "42" logo cells."""
        self._logo_color = value

    @property
    def wall_color(self) -> str:
        """str: The background color code used for walls."""
        return self._wall_color

    @wall_color.setter
    def wall_color(self, value: str) -> None:
        """Set the background color code used for walls."""
        self._wall_color = value

    @property
    def path_color(self) -> str:
        """str: The background color code used for the solution path."""
        return self._path_color

    @path_color.setter
    def path_color(self, value: str) -> None:
        """Set the background color code used for the solution path."""
        self._path_color = value

    @property
    def passage_color(self) -> str:
        """str: The background color code used for open passages."""
        return self._passage_color

    @passage_color.setter
    def passage_color(self, value: str) -> None:
        """Set the background color code used for open passages."""
        self._passage_color = value

    @property
    def path_drawable(self) -> bool:
        """bool: Whether the solution path may currently be drawn."""
        return self._path_drawable

    @path_drawable.setter
    def path_drawable(self, value: bool) -> None:
        """Set whether the solution path may currently be drawn."""
        self._path_drawable = value

    def draw(
            self,
            player: tuple[int, int],
            path_coords: list[tuple[int, int]] | None = None,
            draw_path: bool = True,
            ) -> None:
        """Draw the maze to the terminal.

        Picks the ASCII or colored renderer depending on 'color'.

        Args:
            player (tuple[int, int]): The (x, y) coordinates of the
                player, drawn only when 'playable' is True.
            path_coords (list[tuple[int, int]] | None): The (x, y)
                coordinates of the solution path to highlight. None
                (default) draws no path.
            draw_path (bool): Whether to draw 'path_coords' at all,
                on top of 'path_drawable' also being True. True by
                default.

        Raises:
            MazeGenError: If 'terminal' is False.
        """
        if not self.terminal:
            raise MazeGenError(
            )
        path = (
            set(path_coords or [])
            if draw_path and self.path_drawable
            else set()
        )
        if self.color:
            self._draw_coloured(player, path)
        else:
            self._draw_ascii(player, path)

    @staticmethod
    def _wall_probes(
        matrix: list[list[int]],
        h: int,
        w: int
        ) -> tuple[
            Callable[[int, int], bool],
            Callable[[int, int], bool],
                ]:
        """Build closures that check for walls at a given grid line.

        Args:
            matrix (list[list[int]]): The maze cells, in bit-encoded
                wall format.
            h (int): The height of the maze in cells.
            w (int): The width of the maze in cells.

        Returns:
            tuple[Callable[[int, int], bool], Callable[[int, int], bool]]:
                A pair '(vertical_wall, horizontal_wall)' of functions.
                'vertical_wall(cx, y)' reports whether the vertical
                line at column 'cx', row 'y' is a wall; 'horizontal_wall
                (x, cy)' does the same for the horizontal line at
                column 'x', row 'cy'.
        """
        def vertical_wall(cx: int, y: int) -> bool:
            """Check whether the vertical line at column 'cx', row 'y' is
            a wall."""
            if cx == 0:
                return bool(matrix[y][0] & WE)
            if cx == w:
                return bool(matrix[y][w - 1] & E)
            return bool(matrix[y][cx - 1] & E)

        def horizontal_wall(x: int, cy: int) -> bool:
            """Check whether the horizontal line at column 'x', row 'cy'
            is a wall."""
            if cy == 0:
                return bool(matrix[0][x] & NO)
            if cy == h:
                return bool(matrix[h - 1][x] & SO)
            return bool(matrix[cy - 1][x] & SO)

        return vertical_wall, horizontal_wall

    def _draw_ascii(self,
                    player: tuple[int, int],
                    path: set[tuple[int, int]]
                    ) -> None:
        """Print the maze using plain box-drawing characters.

        Args:
            player (tuple[int, int]): The (x, y) coordinates of the
                player, drawn only when 'playable' is True.
            path (set[tuple[int, int]]): The (x, y) coordinates of the
                solution path to highlight.
        """
        matrix = self.maze.maze
        if not matrix:
            print("Empty matrix!")
            return

        entry = self.maze.entry
        exit_coord = self.maze.exit
        h = self.maze.height
        w = self.maze.width
        vertical_wall, horizontal_wall = self._wall_probes(matrix, h, w)

        def corner(cx: int, cy: int) -> str:
            """Pick the box-drawing character for the corner at grid
            point (cx, cy), based on which of its four adjacent walls
            are present."""
            up = cy > 0 and vertical_wall(cx, cy - 1)
            down = cy < h and vertical_wall(cx, cy)
            left = cx > 0 and horizontal_wall(cx - 1, cy)
            right = cx < w and horizontal_wall(cx, cy)
            return CORNER_CHARS[(up, down, left, right)]

        def border_line(cy: int) -> str:
            """Build one full horizontal border line of the maze, at
            row 'cy'."""
            line = ""
            for cx in range(w + 1):
                line += corner(cx, cy)
                if cx < w:
                    line += "═══" if horizontal_wall(cx, cy) else "   "
            return line

        print(border_line(0))
        for y in range(h):
            row_line = "║" if vertical_wall(0, y) else " "
            for x in range(w):
                if (x, y) == player and self.playable:
                    cell = "🧍 "
                elif (x, y) == entry:
                    cell = " 🏠"
                elif (x, y) == exit_coord:
                    cell = "🚪 "
                elif (x, y) in path:
                    cell = " ◉ "
                elif matrix[y][x] == 31:
                    cell = " ░ "
                else:
                    cell = "   "
                row_line += cell
                row_line += "║" if vertical_wall(x + 1, y) else " "
            print(row_line)
            print(border_line(y + 1))

    def _draw_coloured(self,
                       player: tuple[int, int],
                       path: set[tuple[int, int]]
                       ) -> None:
        """Print the maze using ANSI background-color blocks.

        Args:
            player (tuple[int, int]): The (x, y) coordinates of the
                player, drawn only when 'playable' is True.
            path (set[tuple[int, int]]): The (x, y) coordinates of the
                solution path to highlight.
        """
        matrix = self.maze.maze
        if not matrix:
            print("Empty matrix!")
            return

        entry = self.maze.entry
        exit_coord = self.maze.exit
        h = self.maze.height
        w = self.maze.width
        wall = self.wall_color
        passage = self.passage_color
        vertical_wall, horizontal_wall = self._wall_probes(matrix, h, w)

        def corner_color(cx: int, cy: int) -> str:
            """Pick the background color for the corner at grid point
            (cx, cy): 'wall' if any of its four adjacent walls is
            present, else 'passage'."""
            up = cy > 0 and vertical_wall(cx, cy - 1)
            down = cy < h and vertical_wall(cx, cy)
            left = cx > 0 and horizontal_wall(cx - 1, cy)
            right = cx < w and horizontal_wall(cx, cy)
            return wall if (up or down or left or right) else passage

        def h_segment_color(x: int, cy: int) -> str:
            """Pick the background color for the horizontal wall segment
            at column 'x', row 'cy': 'wall' if closed, 'path_color' if
            it connects two path cells, else 'passage'."""
            if horizontal_wall(x, cy):
                return wall
            if (x, cy - 1) in path and (x, cy) in path:
                return self.path_color
            return passage

        def v_segment_color(cx: int, y: int) -> str:
            """Pick the background color for the vertical wall segment
            at column 'cx', row 'y': 'wall' if closed, 'path_color' if
            it connects two path cells, else 'passage'."""
            if vertical_wall(cx, y):
                return wall
            if (cx - 1, y) in path and (cx, y) in path:
                return self.path_color
            return passage

        def border_line(cy: int) -> str:
            """Build one full horizontal border line of colored
            background blocks, at row 'cy'."""
            line = ""
            for cx in range(w + 1):
                line += f"{corner_color(cx, cy)} "
                if cx < w:
                    line += f"{h_segment_color(cx, cy)}   "
            return line

        lines = [border_line(0)]
        for y in range(h):
            row_line = f"{v_segment_color(0, y)} "
            for x in range(w):
                if (x, y) == player and self.playable:
                    row_line += f"{BG_BLUE}   "
                elif (x, y) == entry:
                    row_line += f"{BG_YELLOW}   "
                elif (x, y) == exit_coord:
                    row_line += f"{BG_GREEN}   "
                elif (x, y) in path:
                    row_line += f"{self.path_color}   "
                elif matrix[y][x] == LOGO:
                    row_line += f"{self.logo_color}   "
                else:
                    row_line += f"{passage}   "
                row_line += f"{v_segment_color(x + 1, y)} "
            lines.append(row_line)
            lines.append(border_line(y + 1))

        for line in lines:
            print(f"{line}{RESET}")
