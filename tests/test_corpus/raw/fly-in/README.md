*This project has been created as part of the 42 curriculum by saperez-.*

# Fly-in

## Description

Fly-in is a turn-based drone-fleet router and simulator. Given a map of
connected zones (a graph), a start hub, an end hub, and a fleet size, it
plans one route per drone and then runs a discrete, turn-by-turn
scheduler that moves the whole fleet from the start to the end while
respecting:

- **Zone occupancy limits** (`max_drones` per hub, with the start/end
  hubs left uncapped).
- **Connection capacity limits** (`max_link_capacity` per link).
- **Zone movement costs**: `normal` and `priority` hubs cost 1 turn to
  enter, `restricted` hubs cost 2 (the drone is "in flight" over the
  connection for the extra turn and cannot bail out early), and
  `blocked` hubs can never be entered.

The objective is to minimize the **total number of simulation turns**
needed to deliver every drone, without any drone ever violating a
capacity rule or entering a forbidden zone. The project is entirely
custom: no graph library (`networkx`, `graphlib`, ...) is used anywhere
in the pathfinding or simulation logic.

Once a map is solved, the result can be inspected two ways: a plain
turn-by-turn log in the terminal (the format the subject requires), and
an optional animated graphical view built with `pygame`.

## Instructions

### Project layout

```
src/             every module that makes up the deliverable — the parser,
                 the router, the scheduler, the graphical view, plus the
                 `main.py` and `performance.py` entry points
tests/           the pytest suite (see Testing below)
assets/          sprites and fonts, loaded by a path relative to the
                 working directory at runtime
maps/            the subject's reference maps under
                 maps/{easy,medium,hard,challenger}/ (read by
                 `make performance`), plus custom edge-case and
                 error-handling maps under maps/test/ and maps/bad_map/
                 (the subject recommends adding your own, Chapter VII.4)
pyproject.toml   project metadata, dependencies, and the flake8 / mypy /
                 pytest configuration
Makefile         the entry points listed under "Makefile targets"
```

The modules under `src/` import each other by bare name (e.g.
`from constants import ...`). `pyproject.toml` puts `src/` on the import
path — `tool.setuptools.package-dir` for `make install`, which does an
editable install, and `tool.pytest.ini_options.pythonpath` for the test
run — so no `src.` prefix is needed anywhere. `assets/` and `maps/` stay
at the repository root because the code resolves them relative to the
working directory, so every command is expected to run from the root.

The subject material itself (`en.subject.pdf`, `maps.tar.gz`, ...) is
kept under `docs/`, which is `.gitignore`d and not submitted.

### Requirements

- Python 3.10+
- A virtual environment is created and managed automatically by the
  `Makefile` (no manual `venv` setup needed).

### Makefile targets

| Target | Effect |
|---|---|
| `make install` | Creates the virtual environment and installs the project (editable) with its dependencies from `pyproject.toml`. |
| `make run` | Runs the simulator (`src/main.py`). |
| `make debug` | Runs the simulator under Python's built-in debugger (`pdb`). |
| `make performance` | Solves every reference map under `maps/{easy,medium,hard,challenger}/` and reports achieved turns against the subject's own targets (see [Performance](#performance) below). |
| `make lint` | Runs `flake8` and `mypy` (the subject's mandatory flag set, kept in `pyproject.toml`) over `src/` and `tests/`. |
| `make lint-strict` | Same, with `mypy --strict`. |
| `make clean` | Removes `__pycache__`, `.mypy_cache`, `.pytest_cache` and build artifacts. |
| `make destroy` / `make re-install` | Tear down / rebuild the virtual environment. |

### Running a simulation

```
$ make run
Welcome to the dron simulation...
Introduce the name of the map: maps/easy/01_linear_path.txt
...
D1-a D2-a
D1-b D2-b
The map was resolved in 2 moves
Show the graphical animation? (y/n): y
```

The program asks for a map file path, parses and validates it, plans
routes, runs the turn-based simulation, prints the resulting move log,
and finally offers to open the graphical animation.

### Map file format

A map is a plain text file:

```
# comments start with '#' and are ignored
nb_drones: 3

start_hub: base 0 0 [color=cyan]
end_hub: pad 4 0 [color=green]
hub: relay 2 0 [zone=restricted color=orange]
hub: bypass 2 2 [zone=priority max_drones=2]

connection: base-relay
connection: base-bypass [max_link_capacity=2]
connection: relay-pad
connection: bypass-pad
```

- The first line declares the fleet size: `nb_drones: <positive int>`.
- Exactly one `start_hub:` and one `end_hub:` are required; any other
  zone uses `hub:`. Each takes a unique name and integer coordinates.
- Optional metadata goes in `[...]`: `zone=` (`normal` (default),
  `priority`, `restricted`, or `blocked`), `color=` (any single word,
  used for the visual output), and `max_drones=` (ignored on the two
  gates, which are never capacity-limited).
- `connection: <a>-<b> [max_link_capacity=<n>]` declares a bidirectional
  link; zone names may not contain `-` or spaces, and a pair may only
  be connected once.
- Malformed input stops the program with a specific error naming the
  offending line, instead of a crash or a silent wrong answer.

## Algorithm & implementation strategy

### Route planning (`routing.Router`)

1. **Reachability check** (`find_path_to_end`): a plain DFS over the
   hub graph, ignoring costs and capacities but refusing `blocked`
   hubs, just to fail fast with a clear error if the map has no
   possible solution at all.
2. **Candidate generation** (`_generate_candidate_routes`): repeated
   Dijkstra searches (`_dijkstra`), where each successful route adds a
   penalty to the connections it used before searching again. This
   steers later searches toward unused parts of the graph, producing
   up to `MAX_CANDIDATE_PATHS` diverse start→end routes instead of the
   same shortest path over and over. A route is dropped once it costs
   too much more than the cheapest one found so far
   (`PATH_COST_CEILING_RATIO`), so the candidate pool stays sensible on
   graphs with many far-fetched detours. `priority` hubs get a
   negligible weight discount (`PRIORITY_BIAS`) so ties favor them, in
   line with the subject's "should be prioritized" rule.
3. **Assignment**: each drone is greedily assigned, one at a time, to
   whichever candidate route currently has the lowest *projected*
   cost — its own weight plus the number of drones already assigned to
   it, divided by that route's bottleneck. This spreads the fleet
   across all viable routes instead of funnelling every drone down a
   single shortest path and queuing them behind its capacity.
   Two variants differ only in what "bottleneck" means:
   - `plan_routes` uses raw capacity (`_bottleneck_capacity`): the
     tightest `max_drones`/`max_link_capacity` along the route.
   - `plan_routes_advanced` uses sustainable throughput
     (`_bottleneck_throughput`): capacity divided by crossing cost, so
     a capacity-1 `restricted` hub (2 turns to cross) is correctly
     weighed as half the throughput of an equally-capped `normal` one,
     not as equally tight.
4. **Complexity gate** (`is_complex_map`): compares fleet size against
   the single cheapest route's sustainable throughput. Below the
   threshold, `plan_routes`'s simpler heuristic is already close to
   optimal and cheap to compute. Past it, contention is severe enough
   that `plan_routes_advanced` plus a refinement pass are worth the
   extra cost.
5. **Local search refinement** (`refine_routes`, only for complex
   maps): the per-route heuristics above score each candidate in
   isolation, so they can't see two routes secretly sharing a
   bottleneck stage. This instead hill-climbs on the *real* outcome:
   it moves one drone at a time onto a different candidate route,
   actually re-runs the scheduler, and keeps the move only if it lowers
   the total turn count, until a full pass finds no more improvements
   or a trial budget is exhausted.

### Turn-based scheduling (`initiate_simulation.Initiator`)

Each turn: drones already mid-flight over a `restricted` connection
advance first (and free their connection slot on arrival). Then the
remaining drones are offered a move, ordered by how close they are to
the end — so a hub a leading drone just vacated can be used the same
turn by the drone right behind it, instead of wasting a turn. A drone
that has no legal move (capacity or connection full) simply waits and
is left out of that turn's log line, matching the subject's output
format. The simulation ends once every drone has reached the end hub;
a turn with no possible move anywhere raises an error instead of
looping forever, since that would mean the route assignment left a
drone permanently stuck.

## Visual representation

Fly-in follows the subject's "terminal and/or graphical" requirement
with both:

- **Terminal**: the move log is printed turn by turn in the
  `D<id>-<zone>` / `D<id>-<connection>` format the subject specifies,
  with colored, letter-by-letter animated status messages (via
  `text_printer.Printer`) so progress reads clearly instead of
  appearing all at once.
- **Graphical** (`show_simulation.Pyshow`, `pygame`): an animated,
  interactive playback of the whole simulation, meant to make the
  fleet's behavior legible at a glance rather than just numerically
  correct:
  - Zones are drawn as colored nodes (by `zone` type, or an explicit
    `color`), with capacity badges; connections are drawn as lines
    with a badge colored by whichever endpoint the wheel-favored crossing
    direction reaches (highlighting `priority`/`restricted` legs).
  - Drones animate smoothly between hubs frame by frame, rotate to face
    their direction of travel, and stack a shared count badge when
    several occupy the same spot.
  - **Playback controls**: play/pause (starts paused), and
    speed up/slow down buttons with a logarithmic speed bar (center =
    normal speed, left = slower, right = faster), plus a reset button
    that rewinds to turn zero.
  - **Camera controls**: the mouse wheel zooms in/out (from a
    somewhat-zoomed-out floor, through the default view, up to 4x),
    centered on the cursor; once zoomed in, click-and-drag pans the
    view. Zooming out past the default view doesn't just crop a flat
    background — a separately kept, larger starfield fills the extra
    margin so it still looks like open space, not a colored border.
  - A small animated "commander" mascot narrates the run in a
    dialog box, typing out random simulation-flavor lines
    letter by letter while its mandibles open and close, purely as
    a bit of personality for the sidebar.

## Performance

`make performance` solves every map under `maps/easy/`, `maps/medium/`,
`maps/hard/`, and `maps/challenger/` (each referenced explicitly, no
wildcard globbing) and checks the resulting turn count against the
subject's own reference targets (Chapter VII.7), printing each line in
green when the target is met and red when it isn't — without printing
the per-turn move log, just the totals. A missing `maps/` directory or
any of its four required subdirectories is reported instead of the
program crashing.

Latest local run:

| Map | Target | Result |
|---|---|---|
| Linear path (2 drones) | ≤6 turns | 4 — PASS |
| Simple fork (4 drones) | ≤8 turns | 5 — PASS |
| Basic capacity (4 drones) | ≤6 turns | 4 — PASS |
| Dead end trap (5 drones) | ≤12 turns | 8 — PASS |
| Circular loop (6 drones) | ≤15 turns | 15 — PASS |
| Priority puzzle (5 drones) | ≤12 turns | 7 — PASS |
| Maze nightmare (8 drones) | ≤30 turns | 13 — PASS |
| Capacity hell (12 drones) | ≤35 turns | 16 — PASS |
| Ultimate challenge (15 drones) | ≤45 turns | 27 — PASS |
| The Impossible Dream (25 drones, optional) | record: 45 turns | 47 — not beaten |

All mandatory targets are met; the optional Challenger map is solved,
just short of the reference record.

## Example

Input (`nb_drones: 2`, one route through a `restricted` relay):

```
nb_drones: 2

start_hub: base 0 0
hub: relay 2 0 [zone=restricted]
end_hub: pad 4 0

connection: base-relay
connection: relay-pad
```

Resulting move log:

```
D1-base-relay
D1-relay
D1-pad D2-base-relay
D2-relay
D2-pad
```

D1 immediately enters the two-turn `restricted` hop (reported as the
connection name while in flight, `D1-base-relay`), lands on turn 2
(`D1-relay`), and reaches the end on turn 3 (`D1-pad`) — the same turn
it vacates `relay`, freeing it for D2 to start its own crossing
(`D2-base-relay`). D2 then follows the same two-turn pattern, and the
simulation ends after 5 turns.

## Testing

The project is covered by a `pytest` suite (unit tests for the parser,
the router's pathfinding/planning, the turn scheduler, and the pure
animation-timing helpers in the graphical view) under `tests/`,
configured by the `[tool.pytest.ini_options]` table in `pyproject.toml`.
As noted in the subject (Chapter III.3), test programs are not submitted
or graded, but they're kept in the repo anyway. The suite reads its
fixtures from `maps/`, so run it from the repository root with:

```
python -m pytest
```

## Resources

- [Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)
  — the shortest-path search `Router._dijkstra` is built on.
- [Introduction to graph theory concepts used throughout routing.py](https://en.wikipedia.org/wiki/Graph_theory)
- [pygame documentation](https://www.pygame.org/docs/) — the graphical
  animation in `show_simulation.py`.
- [Pydantic documentation](https://docs.pydantic.dev/) — the validated
  map data model in `map_config.py`.

**AI usage**: the AI was used as a development assistant
throughout this project, always with the resulting code read, run
(including the `pytest` suite and manual/headless `pygame` checks), and
understood before being kept. Concretely, it helped with: iterating on
the `pygame` graphical view, editing map sprite assets, extending the
`Makefile`, and reorganising the repository into a `src/` layout with a
consolidated `pyproject.toml` (no source code changed in that move).
Also drafting this `README.md`. Every change was verified against the
existing test suite and `mypy --strict`/`flake8` before being accepted.
