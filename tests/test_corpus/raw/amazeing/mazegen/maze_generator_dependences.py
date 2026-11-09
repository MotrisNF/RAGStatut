"""Dependencies for creating the maze.

This module has all the functions that the MazeGen class needs to create
the mazes.
"""

import random
from .exceptions import MazeGenError

# Wall bitmask per subject IV.5 (bit 0=N, 1=E, 2=S, 3=W); a closed
# wall sets its bit. VISITED (bit 4) is only used while carving the
# maze, it's not part of the 4-bit wall encoding written to the output file.
NO = 1
E = 2
SO = 4
WE = 8
VISITED = 16

# Each entry is (dx, dy, wall bit to open on this cell, wall bit to
# open on the neighbour on the opposite side).
DIRECTIONS: list[tuple[int, int, int, int]] = [
    (0, -1, NO, SO),
    (1, 0, E, WE),
    (0, 1, SO, NO),
    (-1, 0, WE, E),
]


def get_free_cells(h: int,
                   w: int,
                   matrix: list[list[int]]
                   ) -> list[tuple[int, int]]:
    """Get the cells not yet visited by the maze-carving algorithm.

    Used by the recursive backtracker to pick a starting cell and to
    check whether there is anything left to carve. Cells locked as part
    of the '42' pattern are excluded too, since they already have the
    VISITED bit set.

    Args:
        h (int): The height of the maze.
        w (int): The width of the maze.
        matrix (list[list[int]]): A bidimensional array, with the value in bits
            from the walls of each cell: N: 0001, E: 0010, S: 0100, W: 1000.
            Bit 4 (VISITED) marks cells already carved or locked
            ('42' pattern).

    Returns:
        list[tuple[int, int]]: A list with all the free cells on the maze,
            inside a tuple with each coordinate.
    """
    return [
        (x, y)
        for y in range(h)
        for x in range(w)
        if not matrix[y][x] & VISITED
    ]


def get_neighbors(h: int,
                  w: int,
                  x: int,
                  y: int,
                  matrix: list[list[int]]
                  ) -> list[tuple[int, int, int, int]]:
    """Get the not-yet-visited cells adjacent to a given cell.

    Used by the recursive backtracker to decide where to carve next.
    Neighbours already visited (either carved or locked as part of the
    '42' pattern) are excluded.

    Args:
        h (int): The height of the maze.
        w (int): The width of the maze.
        x (int): The x position of the cell to evaluate.
        y (int): The y position of the cell to evaluate.
        matrix (list[list[int]]): A bidimensional array, with the value in bits
            from the walls of each cell: N: 0001, E: 0010, S: 0100, W: 1000.
            Bit 4 (VISITED) marks cells already carved or locked
            ('42' pattern).

    Returns:
        list[tuple[int, int, int, int]]: One tuple per free neighbour, with
            its (x, y) coordinates and the wall bits to open on both sides
            (this cell's wall, the neighbour's opposite wall).
    """
    neighbors: list[tuple[int, int, int, int]] = []
    for dx, dy, wall, opp_wall in DIRECTIONS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h and not matrix[ny][nx] & VISITED:
            neighbors.append((nx, ny, wall, opp_wall))
    return neighbors


def maze_generator(h: int,
                   w: int,
                   rng: random.Random,
                   matrix: list[list[int]] | None = None
                   ) -> list[list[int]]:
    """This function creates a perfect maze.

    Uses an iterative loop to create a perfect maze. A stack holds the cells
    visited along the current path, and cells are popped from it when the
    path reaches a dead end. The loop ends when the stack is empty.

    Args:
        h (int): The height of the maze.
        w (int): The width of the maze.
        rng (random.Random): The caller's own random number generator,
            already seeded. Used instead of the 'random' module's global
            functions so generating a maze never reseeds or otherwise
            perturbs the interpreter's shared random state — anything
            else in the process using 'random' stays unaffected.
        matrix (list[list[int]] | None): A bidimensional array, with the
            value in bits from the walls of each cell: N: 0001, E: 0010,
            S: 0100, W: 1000. Bit 4 (VISITED) marks cells already carved or
            locked ('42' pattern). If None (default), a fresh matrix is
            created with matrix_generator.

    Returns:
        list[list[int]]: A bidimensional array, with the value in bits
            from the walls of each cell: N: 0001, E: 0010, S: 0100, W: 1000.
            Bit 4 (VISITED) marks cells already carved or locked
            ('42' pattern), but now, transformed with a perfect maze limited
            by walls.
    """
    if matrix is None:
        matrix = matrix_generator(h, w)

    free_cells = get_free_cells(h, w, matrix)
    if not free_cells:
        return matrix

    x, y = rng.choice(free_cells)
    matrix[y][x] |= VISITED
    stack: list[tuple[int, int]] = [(x, y)]

    while stack:
        x, y = stack[-1]
        neighbors = get_neighbors(h, w, x, y, matrix)
        if not neighbors:
            stack.pop()
            continue
        nx, ny, wall, opp_wall = rng.choice(neighbors)
        matrix[y][x] &= ~wall
        matrix[ny][nx] &= ~opp_wall
        matrix[ny][nx] |= VISITED
        stack.append((nx, ny))

    return matrix


def is_block_fully_open(matrix: list[list[int]], bx: int, by: int) -> bool:
    """This function checks whether a 3x3 block of cells is fully open.

    A 3x3 block, starting at cell (bx, by), is fully open when it has no
    internal walls, i.e. every cell inside it is connected to its neighbours
    within the block. Used to avoid opening walls that would create such a
    large open area in the maze.

    Args:
        matrix (list[list[int]]): A bidimensional array, with the value in bits
            from the walls of each cell: N: 0001, E: 0010, S: 0100, W: 1000.
            Bit 4 (VISITED) marks cells already carved or locked
            ('42' pattern), with a perfect maze limited by walls.
        bx (int): The x position of the top-left cell of the block.
        by (int): The y position of the top-left cell of the block.

    Returns:
        bool: True if the 3x3 block starting at (bx, by) is fully open.
            Else False.
    """
    for y in range(by, by + 3):
        for x in range(bx, bx + 2):
            if matrix[y][x] & E:
                return False
    for y in range(by, by + 2):
        for x in range(bx, bx + 3):
            if matrix[y][x] & SO:
                return False
    return True


def creates_open_area(h: int,
                      w: int,
                      matrix: list[list[int]],
                      x1: int, y1: int,
                      x2: int, y2: int,
                      wall: int,
                      opp_wall: int
                      ) -> bool:
    """Check whether opening a wall would create a fully open 3x3 area.

    This function temporarily opens the wall between the two given cells
    (both this cell's wall and the neighbour's opposite wall), checks if
    doing so would create a fully open 3x3 block anywhere near them, and
    then restores both walls to their previous state. It does not open
    anything permanently; it is only a check used by the caller to decide
    whether opening this wall is safe.

    Args:
        h (int): The height of the maze.
        w (int): The width of the maze.
        matrix (list[list[int]]): A bidimensional array, with the value in bits
            from the walls of each cell: N: 0001, E: 0010, S: 0100, W: 1000.
            Bit 4 (VISITED) marks cells already carved or locked
            ('42' pattern), with a perfect maze limited by walls.
        x1 (int): The x position of the first cell.
        y1 (int): The y position of the first cell.
        x2 (int): The x position of the second cell.
        y2 (int): The y position of the second cell.
        wall (int): The value in bits of this cell's wall to open:
            N: 0001, E: 0010, S: 0100, W: 1000.
        opp_wall (int): The value in bits of the neighbour's opposite wall
            to open: N: 0001, E: 0010, S: 0100, W: 1000.

    Returns:
        bool: True if opening this wall would create a fully open 3x3 block
            (i.e. it is not safe to open it). Else False.
    """
    matrix[y1][x1] &= ~wall
    matrix[y2][x2] &= ~opp_wall

    lo_x, hi_x = min(x1, x2), max(x1, x2)
    lo_y, hi_y = min(y1, y2), max(y1, y2)
    found = False
    for by in range(max(0, hi_y - 2), min(lo_y, h - 3) + 1):
        for bx in range(max(0, hi_x - 2), min(lo_x, w - 3) + 1):
            if is_block_fully_open(matrix, bx, by):
                found = True
                break
        if found:
            break

    matrix[y1][x1] |= wall
    matrix[y2][x2] |= opp_wall
    return found


def is_locked_cell(matrix: list[list[int]], x: int, y: int) -> bool:
    """Check if a cell is part of the '42' logo.

    This function checks that all four wall bits (N, E, S, W) are set, i.e.
    the lower nibble equals 0xF. In a perfect maze every cell keeps at
    least one wall open, so a cell with all its walls closed can only be
    one locked as part of the '42' pattern (place_42 sets it to 31, whose
    lower nibble is also 0xF).

    Args:
        matrix (list[list[int]]): A bidimensional array, with the value in bits
            from the walls of each cell: N: 0001, E: 0010, S: 0100, W: 1000.
            Bit 4 (VISITED) marks cells already carved or locked
            ('42' pattern), with a perfect maze limited by walls.
        x (int): The x position of the cell.
        y (int): The y position of the cell.

    Returns:
        bool: True if the cell is part of the '42' logo. Else False.
    """
    return matrix[y][x] & 0xF == 0xF


def valid_directions(h: int,
                     w: int,
                     x: int,
                     y: int
                     ) -> list[tuple[int, int, int, int]]:
    """Get all in-bounds neighbours of a cell, regardless of visited state.

    Used during the playability post-processing, where the maze is already
    fully carved and a wall may need to be opened between any two adjacent
    cells, not just towards unvisited ones.

    Args:
        h (int): The height of the maze.
        w (int): The width of the maze.
        x (int): The x position of the cell to evaluate.
        y (int): The y position of the cell to evaluate.

    Returns:
        list[tuple[int, int, int, int]]: One tuple per in-bounds neighbour,
            with its (x, y) coordinates and the wall bits to open on both
            sides (this cell's wall, the neighbour's opposite wall).
    """
    dirs: list[tuple[int, int, int, int]] = []
    for dx, dy, wall, opp_wall in DIRECTIONS:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            dirs.append((nx, ny, wall, opp_wall))
    return dirs


def open_degree(matrix: list[list[int]],
                h: int,
                w: int,
                x: int,
                y: int
                ) -> int:
    """Count how many walls of a cell are open.

    Used to check whether a cell has at least two openings, i.e. it is not
    a dead end.

    Args:
        matrix (list[list[int]]): A bidimensional array, with the value in bits
            from the walls of each cell: N: 0001, E: 0010, S: 0100, W: 1000.
            Bit 4 (VISITED) marks cells already carved or locked
            ('42' pattern), with a perfect maze limited by walls.
        h (int): The height of the maze.
        w (int): The width of the maze.
        x (int): The x position of the cell to evaluate.
        y (int): The y position of the cell to evaluate.

    Returns:
        int: The number of open walls (0 to 4) of the cell.
    """
    return sum(
        1 for _, _, wall, _ in valid_directions(h, w, x, y)
        if not matrix[y][x] & wall
    )


def ensure_open_node(h: int,
                     w: int,
                     matrix: list[list[int]],
                     x: int,
                     y: int,
                     rng: random.Random
                     ) -> None:
    """Make sure a cell has at least two openings.

    If the cell already has an open degree of 2 or more, do nothing.
    Otherwise, pick one of its closed walls at random (skipping walls that
    lead to a locked '42' cell) and open it, as long as doing so does not
    create a fully open 3x3 block of cells.

    Args:
        h (int): The height of the maze.
        w (int): The width of the maze.
        matrix (list[list[int]]): A bidimensional array, with the value in bits
            from the walls of each cell: N: 0001, E: 0010, S: 0100, W: 1000.
            Bit 4 (VISITED) marks cells already carved or locked
            ('42' pattern), with a perfect maze limited by walls.
        x (int): The x position of the cell to fix.
        y (int): The y position of the cell to fix.
        rng (random.Random): The caller's own random number generator, so
            this never touches the interpreter's global 'random' state.

    Returns:
        None
    """
    if open_degree(matrix, h, w, x, y) >= 2:
        return
    closed_dirs = [
        d for d in valid_directions(h, w, x, y)
        if matrix[y][x] & d[2] and not is_locked_cell(matrix, d[0], d[1])
    ]
    rng.shuffle(closed_dirs)
    for nx, ny, wall, opp_wall in closed_dirs:
        if not creates_open_area(h, w, matrix, x, y, nx, ny, wall, opp_wall):
            matrix[y][x] &= ~wall
            matrix[ny][nx] &= ~opp_wall
            return


def add_loop(h: int,
             w: int,
             matrix: list[list[int]],
             rng: random.Random
             ) -> bool:
    """Open one random wall to create a loop in the maze.

    Collects every closed wall between two non-locked adjacent cells, then
    tries them in random order until one can be opened without creating a
    fully open 3x3 block of cells.

    Args:
        h (int): The height of the maze.
        w (int): The width of the maze.
        matrix (list[list[int]]): A bidimensional array, with the value in bits
            from the walls of each cell: N: 0001, E: 0010, S: 0100, W: 1000.
            Bit 4 (VISITED) marks cells already carved or locked
            ('42' pattern), with a perfect maze limited by walls.
        rng (random.Random): The caller's own random number generator, so
            this never touches the interpreter's global 'random' state.

    Returns:
        bool: True if a wall was opened (a loop was added). Else False.
    """
    candidates: list[tuple[int, int, int, int, int, int]] = []
    for y in range(h):
        for x in range(w):
            if is_locked_cell(matrix, x, y):
                continue
            for dx, dy, wall, opp_wall in ((1, 0, E, WE), (0, 1, SO, NO)):
                nx, ny = x + dx, y + dy
                if (0 <= nx < w and 0 <= ny < h
                        and matrix[y][x] & wall
                        and not is_locked_cell(matrix, nx, ny)):
                    candidates.append((x, y, nx, ny, wall, opp_wall))

    rng.shuffle(candidates)
    for x, y, nx, ny, wall, opp_wall in candidates:
        if not creates_open_area(h, w, matrix, x, y, nx, ny, wall, opp_wall):
            matrix[y][x] &= ~wall
            matrix[ny][nx] &= ~opp_wall
            return True
    return False


def add_loops(h: int,
              w: int,
              matrix: list[list[int]],
              rng: random.Random,
              min_loops: int
              ) -> None:
    """Add loops to the maze until the minimum is reached.

    Calls add_loop repeatedly until either min_loops walls have been opened
    or no more walls can be opened without creating a fully open 3x3 block.

    Args:
        h (int): The height of the maze.
        w (int): The width of the maze.
        matrix (list[list[int]]): A bidimensional array, with the value in bits
            from the walls of each cell: N: 0001, E: 0010, S: 0100, W: 1000.
            Bit 4 (VISITED) marks cells already carved or locked
            ('42' pattern), with a perfect maze limited by walls.
        rng (random.Random): The caller's own random number generator, so
            this never touches the interpreter's global 'random' state.
        min_loops (int): The minimum number of loops to add.

    Returns:
        None
    """
    added = 0
    while added < min_loops and add_loop(h, w, matrix, rng):
        added += 1


def count_dead_ends(h: int,
                    w: int,
                    matrix: list[list[int]]
                    ) -> list[tuple[int, int]]:
    """Get all the dead-end cells of the maze.

    A cell is a dead end when it has exactly one open wall and it is not
    part of the locked '42' pattern.

    Args:
        h (int): The height of the maze.
        w (int): The width of the maze.
        matrix (list[list[int]]): A bidimensional array, with the value in bits
            from the walls of each cell: N: 0001, E: 0010, S: 0100, W: 1000.
            Bit 4 (VISITED) marks cells already carved or locked
            ('42' pattern), with a perfect maze limited by walls.

    Returns:
        list[tuple[int, int]]: A list with the (x, y) coordinates of every
            dead-end cell.
    """
    return [
        (x, y)
        for y in range(h)
        for x in range(w)
        if not is_locked_cell(matrix, x, y)
        and open_degree(matrix, h, w, x, y) == 1
    ]


def reduce_dead_ends(h: int,
                     w: int,
                     matrix: list[list[int]],
                     rng: random.Random,
                     max_dead_ends: int
                     ) -> None:
    """Reduce the number of dead ends in the maze down to a maximum.

    Repeatedly looks for a dead-end cell that can be opened up (through
    ensure_open_node) until the number of dead ends drops to max_dead_ends
    or no more dead ends can be fixed.

    Args:
        h (int): The height of the maze.
        w (int): The width of the maze.
        matrix (list[list[int]]): A bidimensional array, with the value in bits
            from the walls of each cell: N: 0001, E: 0010, S: 0100, W: 1000.
            Bit 4 (VISITED) marks cells already carved or locked
            ('42' pattern), with a perfect maze limited by walls.
        rng (random.Random): The caller's own random number generator, so
            this never touches the interpreter's global 'random' state.
        max_dead_ends (int): The maximum number of dead ends allowed to
            remain in the maze.

    Returns:
        None
    """
    for _ in range(h * w):
        dead_ends = count_dead_ends(h, w, matrix)
        if len(dead_ends) <= max_dead_ends:
            return
        rng.shuffle(dead_ends)
        fixed = False
        for x, y in dead_ends:
            before = matrix[y][x]
            ensure_open_node(h, w, matrix, x, y, rng)
            if matrix[y][x] != before:
                fixed = True
                break
        if not fixed:
            return


def make_playable(h: int,
                  w: int,
                  matrix: list[list[int]],
                  rng: random.Random,
                  min_loops: int = 2,
                  max_dead_ends: int = 0
                  ) -> list[list[int]]:
    """Turn a perfect maze into a playable one.

    Makes sure the four corners and the center cell each have at least two
    openings, adds loops so the maze is not reduced to a single perfect
    path, and then reduces the number of dead ends.

    Args:
        h (int): The height of the maze.
        w (int): The width of the maze.
        matrix (list[list[int]]): A bidimensional array, with the value in bits
            from the walls of each cell: N: 0001, E: 0010, S: 0100, W: 1000.
            Bit 4 (VISITED) marks cells already carved or locked
            ('42' pattern), with a perfect maze limited by walls.
        rng (random.Random): The caller's own random number generator, so
            this never touches the interpreter's global 'random' state.
        min_loops (int): The minimum number of loops to add. Defaults to 2.
        max_dead_ends (int): The maximum number of dead ends allowed to
            remain in the maze. Defaults to 0.

    Returns:
        list[list[int]]: The same matrix, modified in place, now playable.
    """
    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    center = (w // 2, h // 2)
    for x, y in corners + [center]:
        ensure_open_node(h, w, matrix, x, y, rng)

    add_loops(h, w, matrix, rng, min_loops)
    reduce_dead_ends(h, w, matrix, rng, max_dead_ends)

    return matrix


def place_42(matrix: list[list[int]], height: int, width: int) -> None:
    """Lock the '42' logo pattern in the center of the maze.

    Sets every cell that forms the '42' shape to 31 (all four walls closed
    plus the VISITED bit set), so the maze-carving algorithm treats them as
    already visited and never opens their walls.

    Args:
        matrix (list[list[int]]): A bidimensional array, with the value in bits
            from the walls of each cell: N: 0001, E: 0010, S: 0100, W: 1000.
            Bit 4 (VISITED) marks cells already carved or locked
            ('42' pattern).
        height (int): The height of the maze.
        width (int): The width of the maze.

    Returns:
        None

    Raises:
        MazeGenError: If the maze is smaller than 7 cells tall or 9 cells
            wide, since the '42' pattern would not fit.
    """
    if height < 7 or width < 9:
        raise MazeGenError(
            "The '42' pattern needs a maze at least 7 cells tall and "
            "9 cells wide"
        )
    coordinates: list[tuple[int, int]] = [
        (-3, -2), (1, -2), (2, -2), (3, -2),
        (-3, -1), (3, -1),
        (-3, 0), (-2, 0), (-1, 0), (1, 0), (2, 0), (3, 0),
        (-1, 1), (1, 1),
        (-1, 2), (1, 2), (2, 2), (3, 2)
        ]
    for cord in coordinates:
        y = cord[1] + height // 2
        x = cord[0] + width // 2
        matrix[y][x] = 31


def matrix_generator(h: int, w: int) -> list[list[int]]:
    """Create a fresh maze matrix with every wall closed.

    Every cell starts with all four walls closed (value 15) and nothing
    visited. If the maze is big enough, the '42' pattern is locked in the
    center; otherwise a warning is printed and the maze is left without it.

    Args:
        h (int): The height of the maze.
        w (int): The width of the maze.

    Returns:
        list[list[int]]: A bidimensional array, with the value in bits
            from the walls of each cell: N: 0001, E: 0010, S: 0100, W: 1000.
            Bit 4 (VISITED) marks cells locked as part of the '42' pattern.
    """
    matrix: list[list[int]] = [[15 for _ in range(w)] for _ in range(h)]
    try:
        place_42(matrix, h, w)
    except MazeGenError:
        print(
            f"The '42' is not available in {h} height and {w} width",
            flush=True
            )
    return matrix


def get_maze(h: int,
             w: int,
             seed: int,
             perfect: bool = True
             ) -> list[list[int]]:
    """Generate a maze, perfect or playable.

    Creates its own 'random.Random(seed)' instance and uses it for every
    random choice made while carving and, if requested, making the maze
    playable — the interpreter's global 'random' module is never touched,
    so generating a maze here never perturbs unrelated 'random' calls
    happening anywhere else in the process.

    Args:
        h (int): The height of the maze.
        w (int): The width of the maze.
        seed (int): The seed for the RNG. A positive value is recommended.
        perfect (bool): If True, return a perfect maze (a single unique
            path between any two cells). If False, turn it into a playable
            maze with loops and fewer dead ends. Defaults to True.

    Returns:
        list[list[int]]: A bidimensional array, with the value in bits
            from the walls of each cell: N: 0001, E: 0010, S: 0100, W: 1000.
            Bit 4 (VISITED) marks cells locked as part of the '42' pattern.
    """
    rng = random.Random(seed)
    maze = maze_generator(h, w, rng)
    if not perfect:
        make_playable(h, w, maze, rng)
    return maze
