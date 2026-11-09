"""This module converts the maze on hexadecimal values.

This finctions convert the maze from decimal values to hexadecimal values.
"""


def build_hexa_lines(h: int,
                     w: int,
                     matrix: list[list[int]],
                     ) -> list[str]:
    """Take the maze and convert it in to hexadecimal values.

    This function is the base to this convertion. Use a loop and the 'format()'
    funtion for convert the maze in to a hexadecimal values. Only get the
    result from the first four bitn: N:0001, E: 0010, S: 0100, W: 1000.

    Args:
        h (int): The height of the maze.
        w (int): The width of the maz.
        matrix (lisd[list[str]]): A bidimensional array, with the value in bits
            from the walls of each cell: N: 0001, E: 0010, S: 0100, W: 1000.
            Bit 4 (VISITED) marks cells already carved or locked
            ('42' pattern).

    Returns:
        list[str]: A list that each index is a row of the maze, in hexadecimal
        value."""
    lines: list[str] = []
    for y in range(0, h):
        row = ""
        for x in range(0, w):
            temp: int = matrix[y][x]
            if temp >= 16:
                temp -= 16
            row += format(temp, "x")
        lines.append(row)
    lines.append("")
    return lines
