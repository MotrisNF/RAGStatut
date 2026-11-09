"""Export of a generated maze to the subject's hexadecimal file format."""

from mazegen import MazeGenError


def export_hexa(
        grid: list[str],
        entry: tuple[int, int],
        exit: tuple[int, int],
        path: list[tuple[list[tuple[int, int]], str]],
        output_file: str
        ) -> None:
    """Write the maze grid, entry, exit and solution to a file.

    Args:
        grid (list[str]): The maze rows in hexadecimal wall format.
        entry (tuple[int, int]): The (x, y) coordinates of the entry.
        exit (tuple[int, int]): The (x, y) coordinates of the exit.
        path (list[tuple[list[tuple[int, int]], str]]): The maze's
            solution(s), as returned by 'MazeGen.solution'; only the
            first solution's cardinal-direction string is written.
        output_file (str): The path of the file to write.

    Raises:
        MazeGenError: If 'path' is empty (no solution was computed),
            or if the file can't be written.
    """
    if not path:
        raise MazeGenError(
            "No solution found between the entry and the exit"
        )
    try:
        with open(output_file, 'w') as out:
            for line in grid:
                print(line, file=out)
            print(entry[0], ",", entry[1], file=out, sep="")
            print(exit[0], ",", exit[1], file=out, sep="")
            print(path[0][1], file=out)

    except OSError as e:
        raise MazeGenError(f"Can't write the output file: {e}")
