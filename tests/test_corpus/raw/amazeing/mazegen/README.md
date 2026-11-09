# mazegen

`mazegen` is a small, dependency-free Python package that generates and
solves rectangular mazes. It can produce **perfect** mazes (exactly one
path between any two cells) or **playable** mazes (loops added and dead
ends trimmed, so there's more than one way through), and it can compute
every shortest path between any two cells of a generated maze.

This document describes the `mazegen` package on its own terms: what it
does, how it represents a maze internally, and how to use its public API.
It does not assume you are using the rest of the *A-Maze-ing* repository
(the terminal/graphical game front-ends under `src/`) — everything here
applies to `mazegen` as a standalone library.

## Table of contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Core concepts](#core-concepts)
  - [Coordinates](#coordinates)
  - [The wall bitmask](#the-wall-bitmask)
  - [`grid` vs `maze`](#grid-vs-maze)
  - [Perfect vs. playable mazes](#perfect-vs-playable-mazes)
  - [The "42" pattern](#the-42-pattern)
  - [Determinism and seeds](#determinism-and-seeds)
- [API reference](#api-reference)
  - [`MazeGen`](#mazegen-1)
  - [`MazeGenError`](#mazegenerror)
- [Solving: how paths are found](#solving-how-paths-are-found)
- [Rendering a maze yourself](#rendering-a-maze-yourself)
- [Error handling](#error-handling)
- [Internal / advanced API](#internal--advanced-api)
- [Package layout](#package-layout)
- [License](#license)

## Requirements

- Python 3.10 or newer.
- No third-party runtime dependencies — `mazegen` only uses the standard
  library (`random`, `collections`, `typing`).

## Installation

`mazegen` is packaged with [Hatchling](https://hatch.pypa.io/), configured
through the `pyproject.toml` at the root of this repository (the build
metadata has to live next to the package for the wheel build tool to find
it, even though the package's own code and behavior are fully contained
in this `mazegen/` directory).

From the repository root, you can:

- Install a prebuilt wheel, if one is present (e.g.
  `mazegen-1.0.0-py3-none-any.whl` at the repo root):

  ```sh
  pip install mazegen-1.0.0-py3-none-any.whl
  ```

- Build a fresh wheel from source and install it:

  ```sh
  python -m build          # produces dist/mazegen-*.whl
  pip install dist/mazegen-*.whl
  ```

- Or install directly from source in editable mode, for development:

  ```sh
  pip install -e .
  ```

Once installed, the package is imported simply as `mazegen`, regardless
of how it was installed.

## Quick start

```python
from mazegen import MazeGen, MazeGenError

# A 15x15 playable maze (loops + few dead ends), reproducible via seed.
maze = MazeGen(
    height=15,
    width=15,
    entry=(0, 0),
    exit=(14, 14),
    perfect=False,
    seed=42,
)

maze.generate()
maze.solve(maze.entry, maze.exit)

print(maze.grid)       # hexadecimal wall encoding, one string per row
print(maze.solution)   # every shortest path from entry to exit
```

`maze.solution` looks like:

```python
[
    ([(0, 0), (0, 1), (1, 1), ...], "SEE..."),
    # one entry per shortest path tied for the minimum length, if the
    # maze has more than one (only possible in a non-perfect maze)
]
```

## Core concepts

### Coordinates

Cells are addressed as `(x, y)` pairs, where `x` is the column (bounded
by `width`) and `y` is the row (bounded by `height`):
`0 <= x < width`, `0 <= y < height`. This is the convention used
everywhere in the public API: `entry`, `exit`, the coordinates inside
`solution`, etc.

### The wall bitmask

Each maze cell encodes which of its four walls are **closed** as a 4-bit
value:

| Direction | Bit    | Value |
|-----------|--------|-------|
| North (N) | bit 0  | `1`   |
| East (E)  | bit 1  | `2`   |
| South (S) | bit 2  | `4`   |
| West (W)  | bit 3  | `8`   |

A bit set to `1` means that wall is **closed**; a bit set to `0` means
that side is open (you can walk through it). The four bits combine into
a single value from `0` (all walls open) to `15`/`0xF` (all walls
closed), which is exactly what you see written out as a hex digit in
`grid` (see below).

### `grid` vs `maze`

`MazeGen` exposes the generated maze in two forms:

- **`grid`** (`list[str]`) — the maze encoded as one hexadecimal digit
  per cell, row by row: each character is the 4-bit wall value above,
  formatted as a single hex digit (`0`-`f`). This is the representation
  meant for external consumption (rendering, exporting to a file, etc.).
  For convenience when writing it out as text, `grid` has one extra,
  empty string appended after the last row (so `"\n".join(maze.grid)`
  ends with a trailing newline).

- **`maze`** (`list[list[int]]`) — the lower-level, still-generating
  representation: a 2D array of integers using the same 4 low bits for
  walls, but which may also carry an internal bit 4 (value `16`,
  "VISITED") left over from carving the maze. If you only care about
  walls, mask it with `& 0xF`. A cell whose low nibble equals `0xF`
  (all four walls closed) can only be a cell locked as part of the
  ["42" pattern](#the-42-pattern): a perfect maze's carving algorithm
  always leaves at least one wall open on every cell it visits.

Both are indexed as `[y][x]` (row, then column) to match how they are
built internally — this is the opposite order of the `(x, y)` tuples
used for `entry`, `exit` and path coordinates, so keep the distinction
in mind if you consume `grid`/`maze` directly.

### Perfect vs. playable mazes

Set with the `perfect` constructor argument:

- **`perfect=True`**: the maze is carved with a randomized, iterative
  recursive-backtracker algorithm, which produces a *spanning tree* over
  the grid — exactly one path between any two cells, no loops, and every
  non-entry/exit cell that's a dead end stays a dead end.

- **`perfect=False`** (default): the same perfect maze is carved first,
  then post-processed to make it more "playable":
  1. The four corners and the center cell are each guaranteed at least
     two openings.
  2. A minimum number of loops are added by opening extra walls (never
     opening a wall if doing so would create a fully open 3x3 block of
     cells, to avoid large, maze-less open areas).
  3. Dead ends are reduced down to a target maximum, by opening one more
     wall on dead-end cells where it's safe to do so.

  This means a playable maze can have **more than one shortest path**
  between two given cells — `solve()` returns all of them tied for the
  minimum length.

### The "42" pattern

If the maze is at least 7 cells tall and 9 cells wide, `mazegen` locks a
fixed set of cells in the middle of the grid, shaped like the digits
"42", before carving starts. Those cells are marked as already
visited/fully walled so the carving algorithm never touches them —
they show up in `grid` as `f` (all four walls closed) and act as solid
obstacles the rest of the maze is carved around.

If the maze is smaller than that, the pattern simply doesn't fit: a
notice is printed to stdout and generation proceeds without it — this is
not an error, the maze is still generated and perfectly usable.

### Determinism and seeds

Every random decision made while generating a maze (which cell to start
carving from, which wall to open next, which dead end to fix, etc.) goes
through a `random.Random(seed)` instance created just for that call. The
interpreter's shared, global `random` module state is never touched, so
generating mazes never has side effects on unrelated code elsewhere in
your process that also uses `random`.

- Passing an explicit `seed` makes the maze fully reproducible: the same
  `height`, `width`, `seed` and `perfect` always produce the same maze.
- Leaving `seed=None` (the default) picks a random seed once, when the
  `MazeGen` instance is created, and remembers it — so calling
  `generate()` again without calling `regenerate()` reproduces the same
  layout, rather than picking a new random seed every time.
- `regenerate()` moves on to a new seed (or accepts one explicitly).

## API reference

### `MazeGen`

```python
MazeGen(
    height: int,
    width: int,
    entry: tuple[int, int],
    exit: tuple[int, int],
    perfect: bool = False,
    seed: int | None = None,
)
```

Generate, regenerate and solve a maze of `height x width` cells.

**Raises** `MazeGenError` if `height` or `width` are not positive, if
`entry`/`exit` fall outside the maze bounds, or if `entry` and `exit`
are the same cell.

#### Properties (read-only)

| Property   | Type                                          | Description |
|------------|-----------------------------------------------|--------------|
| `height`   | `int`                                         | The maze's height, in cells. |
| `width`    | `int`                                         | The maze's width, in cells. |
| `grid`     | `list[str]`                                   | Hex-encoded rows of the maze (see [`grid` vs `maze`](#grid-vs-maze)). |
| `maze`     | `list[list[int]]`                             | Raw cell values as integers (see [`grid` vs `maze`](#grid-vs-maze)). |
| `entry`    | `tuple[int, int]`                             | The current entry coordinates. |
| `exit`     | `tuple[int, int]`                             | The current exit coordinates. |
| `solution` | `list[tuple[list[tuple[int, int]], str]]`     | Every shortest path found by the last `solve()` call. Each entry pairs the ordered list of `(x, y)` coordinates along a path with the same path written as a string of cardinal moves (e.g. `"SEEN"`). Empty until `solve()` is called. |

#### `generate() -> None`

Generates the maze using the seed given at construction time (or picked
automatically). Refreshes `maze` and `grid`, and resets the internal
solver state, but does **not** compute a solution — call `solve()`
afterwards if you need one. Calling `generate()` again re-derives the
same maze from the same seed.

#### `regenerate(entry=None, exit=None, seed=-1) -> None`

Generates a new maze, replacing the current one.

- If `seed` is omitted (default `-1`), the internal seed is advanced by
  a small random increment; pass an explicit `int` to pin it instead.
- `entry`/`exit` can optionally be moved to new coordinates; omit either
  to keep the current one.
- Any solution computed by a previous `solve()` call is discarded, since
  it belonged to the maze layout that just got replaced.

**Raises** `MazeGenError` if `generate()` hasn't been called yet, if a
new `entry`/`exit` falls outside the maze bounds, or if entry and exit
end up being the same cell.

#### `solve(entry, exit) -> None`

Computes every shortest path between `entry` and `exit` on the current
maze, and stores the result in `solution`. Also updates the `entry`/
`exit` properties to the values passed in.

**Raises** `MazeGenError` if `generate()` hasn't been called yet, if
`entry`/`exit` fall outside the maze, if they're the same cell, or if no
path exists between them (this can only happen if one of them falls
inside a locked "42" cell, or the maze is otherwise disconnected between
the two points).

### `MazeGenError`

The single exception type raised by everything in this package —
invalid constructor arguments, calling `solve()`/`regenerate()` before
`generate()`, out-of-bounds coordinates, an unsolvable entry/exit pair,
etc. Catch this one type if you just want to handle "something about
this maze request was invalid":

```python
try:
    maze = MazeGen(height=-1, width=10, entry=(0, 0), exit=(1, 1))
except MazeGenError as e:
    print(f"Could not build the maze: {e}")
```

## Solving: how paths are found

`solve()` is a thin wrapper around the internal `MazeSolver` class,
which works in two steps:

1. **Flood fill from the exit.** Starting at the exit cell, a
   breadth-first search expands outward through every open wall,
   recording in a matrix the shortest distance (in cells) from the exit
   to every reachable cell. Unreachable cells keep a very large sentinel
   distance.
2. **Backtrack from the entry.** Starting at the entry cell, the solver
   walks towards any neighbour whose recorded distance is exactly one
   less than the current cell's, branching into every such neighbour.
   Every branch that reaches the exit becomes one entry in `solution`;
   because all of them follow strictly decreasing distances, they are
   all guaranteed to be tied for the shortest possible length.

On a perfect maze there is only one such path. On a playable maze, loops
can create more than one path of the same minimal length, and all of
them are returned.

## Rendering a maze yourself

`mazegen` does not draw anything — that's left entirely to the caller.
Since `grid` is just hex digits encoding which walls are closed, turning
it into a picture only needs the wall bitmask from the
["The wall bitmask"](#the-wall-bitmask) section above. For example, a
minimal ASCII dump that just marks closed cell walls:

```python
from mazegen import MazeGen

maze = MazeGen(height=5, width=5, entry=(0, 0), exit=(4, 4), seed=1)
maze.generate()

N, E, S, W = 1, 2, 4, 8

for row in maze.grid[:-1]:      # grid's last element is an empty string
    top = ""
    mid = ""
    for ch in row:
        cell = int(ch, 16)
        top += "+---" if cell & N else "+   "
        mid += "|   " if cell & W else "    "
    print(top + "+")
    print(mid + "|")
print("+---" * maze.width + "+")
```

## Error handling

Every validation failure — bad dimensions, out-of-bounds coordinates,
calling methods out of order, an unsolvable entry/exit pair — raises
`MazeGenError` with a message describing what went wrong. There is no
other exception type exported by this package, so a single `except
MazeGenError` is enough to catch anything `mazegen` itself can raise.

## Internal / advanced API

The following modules back `MazeGen` and are not required for normal
use, but are available if you need finer control than the `MazeGen`
facade offers (e.g. building a maze matrix by hand, or plugging in a
custom carving step):

- `mazegen.maze_generator_dependences` — the carving algorithm
  (`maze_generator`), the playability post-processing (`make_playable`
  and its helpers: `ensure_open_node`, `add_loops`, `reduce_dead_ends`),
  the "42" pattern (`place_42`), and the top-level `get_maze(h, w, seed,
  perfect)` entry point `MazeGen` itself calls.
- `mazegen.hexa_dependences.build_hexa_lines(h, w, matrix)` — converts a
  raw `list[list[int]]` matrix into the hex-encoded `list[str]` grid
  format.
- `mazegen.maze_solver_dependences.MazeSolver` — the path-finding class
  described [above](#solving-how-paths-are-found), usable directly
  against any hex-encoded grid, not just one produced by `MazeGen`.

These are not re-exported from the top-level `mazegen` package (only
`MazeGen` and `MazeGenError` are, via `mazegen/__init__.py`); import them
from their specific submodule if needed, e.g.
`from mazegen.hexa_dependences import build_hexa_lines`.

## Package layout

```
mazegen/
├── __init__.py                     # public exports: MazeGen, MazeGenError
├── mazegen.py                      # the MazeGen facade class
├── exceptions.py                   # MazeGenError
├── maze_generator_dependences.py   # carving + playability + "42" pattern
├── hexa_dependences.py             # matrix -> hex grid conversion
└── maze_solver_dependences.py      # MazeSolver (shortest-path finder)
```

## License

`mazegen` is distributed under the terms in the repository's
[`LICENSE.md`](../LICENSE.md).
