*This project has been created as part of the 42 curriculum by albrodri, saperez-.*

# A-Maze-ing — This is the way

> Create your own maze generator and display its result!

## Description

**A-Maze-ing** is a Python program that procedurally generates a maze from a
plain-text configuration file, solves it, writes the result to a
hexadecimal output file, and displays it either live in the terminal or in
a graphical MiniLibX (MLX) window.

Two generation modes are supported:

- **`PERFECT=True`**: a perfect maze — exactly one path between the entry
  and the exit, no loops at all.
- **`PERFECT=False`** (default): a Pac-Man-style playable board — full
  connectivity, the four corners and the center are open corridors, at
  least two independent routes exist between any two points, and (by
  default) **no dead-ends at all**.

Every generated maze contains a hidden **"42"** pattern made of fully
closed cells in the middle of the grid (omitted with a console warning if
the maze is too small to fit it, i.e. under 7 cells tall or 9 cells wide),
and the whole maze is reproducible from a `SEED` value.

On top of the mandatory behaviour, the terminal side adds an
ANSI-coloured renderer with fully customisable colours, a WASD
player-movement mode, and three playable game modes (Normal, Endless,
Crazy/time-attack with a persisted highscore) — see
[Advanced / bonus features](#advanced--bonus-features) below.

## Instructions

### Requirements

- Python 3.10+ (this repository's `.venv` runs 3.10.12).
- A `make`-compatible shell (Linux/macOS, or WSL on Windows).
- Only needed for the graphical display: the MiniLibX Python bindings
  (`src/mlx-*.whl`, bundled in the repo) and their native dependencies
  (X11 on Linux).

### Setup and run

A `Makefile` is provided to automate everything:

```bash
make install      # creates a local .venv and installs dev/runtime dependencies
make install-mazegen      # create a local .venv and installs the mazegen package and the dependencies
make run           # runs a_maze_ing.py with CONFIG=config.txt (default)
make run CONFIG=my_config.txt   # or with a different config file
make debug         # same as `run`, but stopped at pdb on start
make lint          # flake8 + mypy over the whole project
make lint-strict    # flake8 + mypy --strict (optional, stricter)
make package       # rebuilds mazegen-*.whl from mazegen/'s current source
make clean         # removes caches and build artifacts (__pycache__, .mypy_cache, dist...)
make fclean         # Full clean: also removes the built mazegen wheel(s) from the repo root.
make uninstall      # removes the local .venv
```

`install` also picks up (if present at the repo root/`src/`) the optional
graphical dependency (`src/mlx-*.whl`, MiniLibX bindings) and the reusable
`mazegen-*.whl` package described below, so the whole repo is runnable from
a single command. See [`Makefile`](Makefile) for the exact rules.

Without `make`, the program can also be run directly, exactly as required
by the subject:

```bash
python3 a_maze_ing.py config.txt
```

`config.txt` is the only argument, and must be a plain-text file following
the format described below. A different filename can be passed instead.
Any error in the configuration, the maze parameters, or the maze itself
(missing/duplicated keys, non-integer values, out-of-bounds entry/exit,
an impossible maze size for the "42" pattern, etc.) is reported with a
clear message on the console and the program exits cleanly — it never
crashes with a raw traceback.

### Configuration file format

One `KEY=VALUE` pair per line; lines starting with `#` are ignored, and an
unknown, duplicated, or missing required key aborts the program with an
error message. Keys are matched case-insensitively, but the value of
`OUTPUT_FILE` is stored exactly as written (filenames are case-sensitive
on Linux/macOS). A default `config.txt` is provided at the repository
root.

| Key | Required | Description | Example |
|---|---|---|---|
| `WIDTH` | yes | Maze width, in cells | `WIDTH=16` |
| `HEIGHT` | yes | Maze height, in cells | `HEIGHT=15` |
| `ENTRY` | yes | Entry coordinates `x,y` | `ENTRY=0,0` |
| `EXIT` | yes | Exit coordinates `x,y` | `EXIT=4,4` |
| `OUTPUT_FILE` | yes | Output filename (if left empty, the program asks for one interactively) | `OUTPUT_FILE=maze.txt` |
| `SEED` | yes | RNG seed, for a reproducible maze | `SEED=42` |
| `PERFECT` | no (default `False`) | `True` for a perfect (single-path) maze, `False` for a playable board | `PERFECT=False` |
| `PLAYABLE` | no (default `False`) | Enables the WASD player-movement mode and the game-mode menu | `PLAYABLE=True` |
| `COLOR` | no (default `False`) | Enables the ANSI-coloured terminal renderer instead of plain ASCII | `COLOR=True` |
| `TERMINAL` | no (default `True`) | `True` displays the maze in the terminal, `False` opens a MiniLibX graphical window instead | `TERMINAL=True` |

`WIDTH`/`HEIGHT` are capped at 40×25 cells (in both `TERMINAL=True` and
`TERMINAL=False` mode): anything larger is rejected up front with a clear
error message, so the maze always fits the terminal/window it's drawn in.

Example (the `config.txt` shipped at the repo root):

```
WIDTH=16
HEIGHT=15
ENTRY=0,0
EXIT=4,4
OUTPUT_FILE=maze.txt
PERFECT=False
SEED=42
PLAYABLE=True
TERMINAL=True
COLOR=True
```

### Controls

#### Terminal renderer (`TERMINAL=True`)

Once running, the maze is displayed in the terminal and updates live,
several times a second:

| Key | Action | Availability |
|---|---|---|
| `w` `a` `s` `d` | Move the player | Only when `PLAYABLE=True` |
| `p` | Show/hide the shortest path from the entry | Only when `PLAYABLE=False` |
| `n` | Cycle to the next available shortest path | Only when `PLAYABLE=False` |
| `r` | Regenerate a new maze (next derived seed) | Only when `PLAYABLE=False` |
| `m` | Open a prompt to recolour walls / "42" pattern / path / passages (pick a part, then a value 0-255) | Only when `COLOR=True` |
| `e` | Toggle a "rainbow" mode that randomises all four colours every frame | Always |
| `q` | Quit | Always |

When `PLAYABLE=True`, the program first asks for a game mode:

1. **Normal**: reach the exit once to win.
2. **Endless**: reaching the exit regenerates a new maze of the same size
   with a fresh, randomly placed exit — the run never really ends.
3. **Crazy**: a time-attack mode. It starts on a small perfect 4×4 maze
   with a 15-second timer; every time you reach the exit the board grows
   by one row and one column — always as a perfect (single-path) maze,
   regardless of your config's `PERFECT` setting — the level counter goes
   up, and the timer gets a bit more time. Running out of time asks for a
   name and records the level reached in `highscore.txt` (sorted, best
   score first).
4. **Exit**.

Movement in `PLAYABLE` mode auto-continues in a straight corridor (the
player keeps moving forward when there is only one way to go that isn't
backtracking), Pac-Man style — `w`/`a`/`s`/`d` are only needed at forks.

#### Graphical renderer, MiniLibX (`TERMINAL=False`)

| Key | Action |
|---|---|
| `w` `a` `s` `d` | Move the player (smooth, animated movement) |
| `p` | Show/hide an animated reveal of the shortest path |
| `r` | Regenerate the maze |
| `m` | Change the theme of the maze
| `Esc` | Quit |

### Output file format

`OUTPUT_FILE` is written with one hexadecimal digit per cell — bit
0=North, 1=East, 2=South, 3=West; `1` means that wall is **closed**, `0`
means it's **open** — one row per line, followed by a blank line and then
the entry coordinates, the exit coordinates, and one valid shortest path
from entry to exit as a string of `N`/`E`/`S`/`W` letters. This matches
exactly the format `maze_analyzer.py` (provided with the subject) expects.
All lines end with `\n`.

Example (from the `config.txt` shipped in this repo, `SEED=42`):

```
9153939551139113
8692aac556aaaaaa
a96aac39556ac286
aa96c38293929683
ac4396c6ac6c696a
...
c44546c4446c6c56

0,0
4,4
EEESSSWSESEN
```

## Maze generation algorithm

The perfect maze is built with an **iterative recursive backtracker**:
starting from a random free cell, a stack tracks the current carving path;
at each step the algorithm carves into a random unvisited neighbour and
pushes it onto the stack, backtracking (popping the stack) whenever it
hits a dead end, until the stack is empty.

We chose it because:

- It always produces a **perfect maze** by construction (the carved cells
  and open walls form a spanning tree of the grid graph — one and only one
  path between any two cells), so `PERFECT=True` needs no extra
  post-processing or validation step, and it can never accidentally create
  a fully open 3×3 area (a spanning tree has no cycles, and a 3×3 open
  block requires one).
- It's iterative (an explicit stack), not recursive by function calls, so
  it can't blow the Python call stack on large mazes.
- Its bias towards long, winding corridors (versus, e.g., Prim's, which
  tends to produce many short branches) reads better visually and made it
  easier to reason about the "no 3×3 open area" and "42 pattern" rules
  while designing the generator.

The "42" pattern is locked in first, before any carving happens: every
cell that forms the digits is marked fully closed and "already visited",
so the backtracker simply never reaches or opens any of its walls.

For the default (`PERFECT=False`) Pac-Man mode, the perfect maze produced
above is used as a starting point and then adjusted, in order:

1. The four corners and the center cell are each forced to have at least
   two open walls (opening one at random, skipping the "42" cells, and
   only if it doesn't create a 3×3 open area).
2. Extra walls between two non-locked adjacent cells are removed (again,
   only if safe) to create at least two independent loops, guaranteeing an
   alternative route always exists.
3. Remaining dead-ends (cells with exactly one open wall) are opened up
   one more wall at a time, by default until **none are left at all** —
   this default (`max_dead_ends=0`) is what implements the "no dead-end
   at all" bonus described in the subject's Chapter VIII.

Every wall opened during this post-processing step is checked beforehand
against creating a fully open 3×3 block, so the "no corridor wider than 2
cells" rule holds for both maze modes, not just the perfect one.

The solver (`mazegen/maze_solver_dependences.py`) computes a distance map
with a recursive flood-fill from the exit back towards the entry —
relaxing a cell's distance whenever a shorter path through it is found —
then reconstructs every optimal path by walking from the entry through
neighbours whose distance decreases by exactly one at each step, so all
tied-shortest paths (not just one) are returned.

## Reusable module: `mazegen`

The class-based, pip-installable core lives in [`mazegen/`](mazegen/) and
is built as a standalone wheel (`mazegen-1.0.0-py3-none-any.whl`, at the
repository root) with no third-party dependencies. `a_maze_ing.py` and
every module under `src/` import and consume this package directly — it
is the actual maze engine the running program uses, not a parallel/unused
copy.

### Install

```bash
pip install ./mazegen-1.0.0-py3-none-any.whl
```

### Basic usage

```python
from mazegen import MazeGen, MazeGenError

maze = MazeGen(
    height=15,
    width=15,
    entry=(0, 0),
    exit=(14, 14),
    perfect=False,   # False = Pac-Man-style board (several routes, no dead-ends by default)
    seed=42,          # optional; a random seed is picked if omitted
)

maze.generate()                    # builds the grid
maze.solve(maze.entry, maze.exit)  # computes the shortest path(s)

maze.grid       # list[str]: one row per line, hexadecimal wall encoding (N=1, E=2, S=4, W=8 when closed)
maze.maze       # list[list[int]]: same data as integers, before hex-encoding
maze.solution   # list[tuple[list[tuple[int,int]], str]]: coords + N/E/S/W string, one entry per tied-shortest path
maze.regenerate()  # new maze, derived from the next seed (entry/exit can optionally be changed too)
```

Invalid parameters (e.g. `height`/`width` ≤ 0, `entry`/`exit` outside the
maze or equal to each other, or calling `solve()`/`regenerate()` before
`generate()`) raise `MazeGenError` instead of crashing or returning silent
garbage.

`maze.grid` is the module's own hexadecimal-per-row structure — the
subject explicitly allows this to differ from the on-disk output file
format. Writing *this project's* specific output-file format (hex rows +
entry/exit/path block) is intentionally kept out of `mazegen`: it's
app-specific I/O, not part of the reusable generator, so it lives in
[`src/maze_output.py`](src/maze_output.py) instead, consuming `maze.grid`
directly.

To rebuild the wheel from `mazegen/`'s current source, run `make package`
(it runs `python -m build`, then copies the resulting wheel to the repo
root and reinstalls it in `.venv` — see the `package` rule in the
[`Makefile`](Makefile) for the exact steps).

## Resources

### Classic references

- [Maze generation algorithm — Wikipedia](https://en.wikipedia.org/wiki/Maze_generation_algorithm)
- [Spanning tree — Wikipedia](https://en.wikipedia.org/wiki/Spanning_tree)
  (why a recursive-backtracker maze is always "perfect")
- Jamis Buck, *Mazes for Programmers* (book), and his maze-generation
  algorithm series on `weblog.jamisbuck.org` — widely cited reference for
  recursive backtracker, Prim's, and Kruskal's maze generation.
- MiniLibX documentation, for the graphical renderer.
- `docs/en.subject.pdf` — the project subject itself, primary source for
  every format/requirement referenced above.

### AI usage



## Team and project management

### Roles

- **saperez-**: the complete `MazeGen` generator side — the wall-bitmask
  matrix, the "42" pattern placement, the iterative recursive-backtracker
  carving, and the playable-mode post-processing (loops, corner/center
  opening, dead-end reduction) — plus everything on the terminal side:
  the config parser, the CLI entry point, the ASCII/colour terminal
  renderer, the WASD player-movement logic, the three game modes (Normal,
  Endless, Crazy), the highscore file, and the `Makefile`.
- **albrodri**: the maze solver (shortest-path flood fill and path
  reconstruction), and the entire MiniLibX-based graphical renderer —
  sprite loading, tile drawing, the animated player entity, and the
  animated shortest-path reveal.

### Planning and how it evolved

The project started 2026-07-24 with the repository/folder structure, then
moved through, roughly in this order: the wall-bitmask matrix generator
and the "42" pattern placement; the iterative recursive-backtracker
carving logic; the config-file parser and the first terminal printer;
the solver (recursive flood fill + path reconstruction); splitting the
single script into `inc/`/`src/` modules; player movement, the colour
system, and the three game modes; packaging a class-based,
pip-installable version of the generator (`mazegen/`) per Chapter VI in
parallel with the rest; the MiniLibX graphical renderer; and finally
reconnecting everything through a single `a_maze_ing.py` entry point,
with a last pass of `flake8`/`mypy` cleanup.

### What worked well / what to improve

Splitting the work along the generator/terminal vs. solver/MLX boundary
kept most changes in separate files and avoided the bulk of merge
conflicts. Two things cost more time than expected: the solver's
flood-fill is implemented recursively (with `sys.setrecursionlimit`
raised to cope), which worked but is fragile on very large mazes — an
iterative rewrite, like the one already used for the maze generator
itself, would be the safer long-term fix. The MiniLibX Python bindings
were also a significant source of friction (keycodes, image loading,
redraw flicker) before settling into the current renderer.

### Tools used

- Python 3.10+, `venv`, `pip`.
- `flake8` and `mypy` for linting/typing (see `make lint`).
- `hatchling` + `build` for packaging `mazegen` as a wheel.
- MiniLibX (`mlx`) Python bindings for the graphical display.

## Advanced / bonus features

- A default (non-perfect) maze with **no dead-end at all**: `make_playable`
  defaults to `max_dead_ends=0`, so the playable board is a fully
  "braided" maze by default — a chased player is never trapped anywhere.
  This is the bonus explicitly described in the subject's Chapter VIII.
- ANSI-coloured terminal renderer in addition to plain ASCII
  (`COLOR=True`/`False`), with all four visual elements (walls, "42"
  pattern, path, open passages) individually recolourable at runtime
  (`m` key), plus a "rainbow" mode that randomises them every frame
  (`e` key, an easter egg).
- Optional WASD player-movement mode (`PLAYABLE=True`) with three
  distinct game modes: Normal, Endless (infinite regeneration with a
  new random exit), and Crazy (time-attack with a growing maze and a
  persisted highscore).
- Cycling between multiple valid shortest paths (`n` key) in the terminal
  renderer, when not in `PLAYABLE` mode.
- A full MiniLibX graphical renderer (`src/maze_mlx_printer.py`), with
  sprite-based walls, smooth animated player movement, and an animated
  shortest-path reveal — selectable at runtime via the `TERMINAL` config
  key, as an alternative to the terminal renderer.
