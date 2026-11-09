"""Dependencies for solving the maze.

This module defines the MazeSolver class, used by MazeGen to compute the
shortest path(s) between the maze entry and exit.
"""

from collections import deque
from typing import Deque


class MazeSolver:
    """Solves hex-encoded mazes and computes valid navigation paths.

    Attributes:
        maze (list[str]): Hexadecimal string representation of the maze grid.
        size (tuple[int, int]): Dimensions of the maze grid as (width, height).
        enter_coord (tuple[int, int]): Starting coordinates as (x, y).
        exit_coord (tuple[int, int]): Destination coordinates as (x, y).
    """

    def __init__(
            self,
            maze: list[str],
            size: tuple[int, int],
            enter_coord: tuple[int, int],
            exit_coord: tuple[int, int]
    ) -> None:
        """Initializes the MazeSolver with maze and target points.

        Args:
            maze (list[str]): Grid layout encoded in hexadecimal per row.
            size (tuple[int, int]): Grid dimensions (width, height).
            enter_coord (tuple[int, int]): Coordinates of the entrance (x, y).
            exit_coord (tuple[int, int]): Coordinates of the exit (x, y).
        """
        self.maze = maze
        self.size = size
        self.enter_coord = enter_coord
        self.exit_coord = exit_coord

    def set_data(
            self,
            maze: list[str],
            size: tuple[int, int],
            enter_coord: tuple[int, int],
            exit_coord: tuple[int, int]
    ) -> None:
        """Updates internal solver state with new maze configuration data.

        Args:
            maze (list[str]): New hexadecimal maze layout.
            size (tuple[int, int]): New grid dimensions (width, height).
            enter_coord (tuple[int, int]): New entrance coordinates (x, y).
            exit_coord (tuple[int, int]): New exit coordinates (x, y).
        """
        self.maze = maze
        self.size = size
        self.enter_coord = enter_coord
        self.exit_coord = exit_coord

    def depth_first_search(self) -> list[list[float]]:
        """Calculates distance map from exit to entrance using an iterative
        algorithm.

        Returns:
            list[list[float]]: 2D matrix storing path distances from the exit
            to each cell, initialized with a large threshold for unreachable
            cells.
        """
        matrix: list[list[float]] = [[1e12] * self.size[0]
                                     for _ in range(self.size[1])]
        q: Deque[tuple[int, int]] = deque()
        q.append(self.exit_coord)
        matrix[self.exit_coord[1]][self.exit_coord[0]] = 0
        directions: list[tuple[int, int, int]] = [
            (0, -1, 1),
            (1, 0, 2),
            (0, 1, 4),
            (-1, 0, 8)
        ]
        while q:
            actual_coord: tuple[int, int] = q.popleft()
            actual_value: float = matrix[actual_coord[1]][actual_coord[0]]

            for direction in directions:
                next_x = actual_coord[0] + direction[0]
                next_y = actual_coord[1] + direction[1]

                if not (0 <= next_x < self.size[0]
                        and 0 <= next_y < self.size[1]):
                    continue

                next_value: float = matrix[next_y][next_x]
                if self.valid_movment(
                    actual_coord, direction, actual_value, next_value
                ):
                    next_coord: tuple[int, int] = (next_x, next_y)
                    q.append(next_coord)
                    matrix[next_y][next_x] = actual_value + 1
        return matrix

    def valid_movment(self,
                      actual_coord: tuple[int, int],
                      direction: tuple[int, int, int],
                      actual_value: float,
                      next_value: float
                      ) -> bool:
        """Checks whether moving from a cell towards a direction improves
        the neighbour's known distance.

        Used by 'depth_first_search' to decide whether to enqueue the
        neighbour cell: the wall on that side must be open, and the
        neighbour must not already have a distance equal or shorter than
        going through 'actual_coord' would give it.

        Args:
            actual_coord (tuple[int, int]): The (x, y) coordinates of the
                cell the flood fill is currently expanding from.
            direction (tuple[int, int, int]): The (dx, dy, wall bit) of the
                direction being probed.
            actual_value (float): The distance already assigned to
                'actual_coord'.
            next_value (float): The distance currently assigned to the
                neighbour cell, before this move.

        Returns:
            bool: True if the wall towards 'direction' is open and the
            neighbour's distance can be improved by going through
            'actual_coord'. Else False.
        """
        hex_char = self.maze[actual_coord[1]][actual_coord[0]]
        val = int(hex_char, 16)
        open_walls = (~val) & 15
        if open_walls & direction[2] != 0:
            if next_value > actual_value + 1:
                return True
        return False

    def path_finder(
            self
            ) -> list[tuple[list[tuple[int, int]], str]]:
        """Reconstructs all optimal paths from entrance to exit.

        Returns:
            list[tuple[list[tuple[int, int]], str]]: List of path solutions,
            where each solution consists of a coordinate list and a cardinal
            direction string (e.g., 'N', 'E', 'S', 'W').
        """
        matrix: list[list[float]] = self.depth_first_search()

        all_paths: list[tuple[list[tuple[int, int]], str]] = []
        start_x: int = self.enter_coord[0]
        start_y: int = self.enter_coord[1]

        if matrix[start_y][start_x] >= 1e12:
            return all_paths

        def backtrack(
                coord_x: int,
                coord_y: int,
                current_path_coords: list[tuple[int, int]],
                current_path_str: list[str]
        ) -> None:
            """Backtracks through the distance matrix to extract solution.

            Args:
                coord_x (int): Current X position.
                coord_y (int): Current Y position.
                current_path_coords (list[tuple[int, int]]): Coordinates list.
                current_path_str (list[str]): Cardinal directions list.
            """
            if (coord_x, coord_y) == self.exit_coord:
                all_paths.append((list(current_path_coords),
                                  "".join(current_path_str)))
                return

            current_val = matrix[coord_y][coord_x]

            hex_char = self.maze[coord_y][coord_x]
            val = int(hex_char, 16)
            open_walls = (~val) & 15
            directions = [
                (0, -1, 1, 'N'),
                (1, 0, 2, 'E'),
                (0, 1, 4, 'S'),
                (-1, 0, 8, 'W')
            ]

            for dir_x, dir_y, mask, move_char in directions:
                next_x = coord_x + dir_x
                next_y = coord_y + dir_y

                if ((0 <= next_x) and (next_x < self.size[0]) and
                   (0 <= next_y) and (next_y < self.size[1])):
                    if open_walls & mask:
                        if matrix[next_y][next_x] == current_val - 1:

                            current_path_coords.append((next_x, next_y))
                            current_path_str.append(move_char)

                            backtrack(next_x, next_y,
                                      current_path_coords, current_path_str)

                            current_path_coords.pop()
                            current_path_str.pop()
        backtrack(start_x, start_y, [self.enter_coord], [])
        return all_paths
