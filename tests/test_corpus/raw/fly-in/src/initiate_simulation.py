"""Turn-based scheduling for the drone fleet.

Route planning lives in ``routing.py`` (see ``Router``); this module
only turns a chosen set of routes into an actual turn-by-turn
simulation.
"""

from constants import TESTING as TESTING
from drone import Drone, DronePosition
from map_config import MapConfig
from routing import Router
from text_printer import Printer
from typing import Optional
from exceptions import PathError


class Initiator:
    """Loads a validated map and runs the multi-drone simulation."""

    def __init__(self, config: Optional[MapConfig]) -> None:
        """Store the map config and prepare empty simulation results.

        Args:
            config: The parsed map, or ``None`` if loading failed.

        Attributes set here (besides ``config`` and ``printer``):
            drones: The fleet from the last ``run_simulation()`` call.
            turns: One formatted ``D<id>-<zone>`` line per turn, from
                the last run.
            turn_moves: Same data as ``turns``, as
                ``(drone_id, label)`` pairs per turn instead of a
                joined string.
            turn_positions: One ``DronePosition`` per drone per turn,
                in drone order, including delivered ones parked at
                the exit -- usable directly as animation frames.
        """
        self.config = config
        self.printer: Printer = Printer()
        self.drones: list[Drone] = []
        self.turns: list[str] = []
        self.turn_moves: list[list[tuple[int, str]]] = []
        self.turn_positions: list[list[DronePosition]] = []
        self.printer.print_by_letter(
            "Initiating the simulation...",
            0.1,
            0.5,
            "\033[92m",
            TESTING
        )

    def find_path_to_end(self) -> bool:
        """Check whether the end hub is reachable from the start hub.

        Returns:
            True if at least one valid route exists.

        Raises:
            PathError: If the config isn't loaded.
        """
        if self.config is None:
            raise PathError("The config is not loaded.")
        self.printer.print_by_letter(
            "Serching a valid way to the end...",
            0.02,
            2,
            "\033[92m",
            TESTING
        )
        return Router(self.config).find_path_to_end()

    def _create_drones(self, routes: list[list[str]]) -> list[Drone]:
        """Create the fleet, each following its own assigned route.

        Args:
            routes: One route per drone, in drone-id order.

        Returns:
            One ``Drone`` per drone requested by the map, numbered
            from 1.

        Raises:
            PathError: If the config isn't loaded.
        """
        if self.config is None:
            raise PathError("The config is not loaded.")
        drones = []
        for drone_id, path in enumerate(routes, start=1):
            drone = Drone(drone_id, self.config)
            drone.set_path(path)
            drones.append(drone)
        return drones

    def _simulate(
        self, routes: list[list[str]]
    ) -> tuple[
        list[Drone],
        list[str],
        list[list[tuple[int, str]]],
        list[list[DronePosition]],
    ]:
        """Run the turn-based scheduler for a fixed per-drone assignment.

        Every turn: in-flight drones crossing a ``restricted`` link
        advance first; then the remaining drones are offered a move,
        closest to the end hub first, so a hub freed by a drone ahead
        can be used the same turn by the drone behind it. A drone
        that can't move simply waits and is left out of that turn's
        line.

        Doesn't touch ``self`` -- only reads ``self.config`` and
        creates fresh ``Drone`` instances -- so ``Router.refine_
        routes`` can replay it for trial assignments before
        ``run_simulation`` calls it once for real.

        Args:
            routes: One route (inclusive of both gates) per drone, in
                drone-id order.

        Returns:
            The finished fleet, plus the per-turn logs
            ``run_simulation`` stores on ``self``.

        Raises:
            PathError: If the config isn't loaded, or a turn goes by
                with no drone able to move.
        """
        if self.config is None:
            raise PathError("The config is not loaded.")
        config = self.config
        drones = self._create_drones(routes)
        turns: list[str] = []
        turn_moves: list[list[tuple[int, str]]] = []
        turn_positions: list[list[DronePosition]] = []
        start_name = config.gates.entry.name
        end_name = config.gates.exit.name
        connection_capacity = Router.connection_capacities(config)
        connection_usage: dict[frozenset[str], int] = {}
        hub_occupancy: dict[str, int] = {}
        transit_keys: dict[int, frozenset[str]] = {}

        def hub_has_room(name: str) -> bool:
            """Whether one more drone can enter hub ``name`` right now."""
            if name in (start_name, end_name):
                return True
            limit = config.hubs[name].max_drones
            if limit is None:
                return False
            return hub_occupancy.get(name, 0) < limit

        def enter_hub(name: str) -> None:
            """Record one more drone occupying hub ``name``."""
            if name not in (start_name, end_name):
                hub_occupancy[name] = hub_occupancy.get(name, 0) + 1

        def leave_hub(name: str) -> None:
            """Record one drone freeing hub ``name``."""
            if name not in (start_name, end_name):
                hub_occupancy[name] = hub_occupancy.get(name, 0) - 1

        def connection_has_room(key: frozenset[str]) -> bool:
            """Whether one more drone can cross the connection ``key``."""
            return connection_usage.get(key, 0) < connection_capacity[key]

        while not all(drone.is_delivered for drone in drones):
            moves: list[tuple[int, str]] = []
            just_arrived: set[int] = set()

            for drone in drones:
                if not drone.is_in_transit:
                    continue
                arrived = drone.advance_transit()
                if arrived:
                    key = transit_keys.pop(drone.id)
                    connection_usage[key] -= 1
                    just_arrived.add(drone.id)
                moves.append((drone.id, drone.current_label))

            order = sorted(
                drones, key=lambda d: (d.remaining_hops, d.id)
            )
            one_turn_releases: list[frozenset[str]] = []
            for drone in order:
                if drone.is_delivered or drone.is_in_transit:
                    continue
                if drone.id in just_arrived:
                    continue
                next_hub = drone.next_hub()
                if next_hub is None:
                    continue
                current_hub = drone.current_hub
                key = frozenset((current_hub, next_hub))
                if not hub_has_room(next_hub):
                    continue
                if not connection_has_room(key):
                    continue

                cost = Router.zone_cost(config.hubs[next_hub].zone)
                assert cost is not None

                leave_hub(current_hub)
                enter_hub(next_hub)
                connection_usage[key] = connection_usage.get(key, 0) + 1

                if cost == 1:
                    drone.move_to(next_hub)
                    one_turn_releases.append(key)
                else:
                    drone.start_transit(
                        f"{current_hub}-{next_hub}", next_hub, cost - 1
                    )
                    transit_keys[drone.id] = key
                moves.append((drone.id, drone.current_label))

            for key in one_turn_releases:
                connection_usage[key] -= 1

            if not moves:
                raise PathError(
                    "Simulation deadlocked: no drone could move "
                    "this turn."
                )

            moves.sort()
            turn_moves.append(moves)
            turns.append(
                " ".join(f"D{drone_id}-{label}" for drone_id, label in moves)
            )
            turn_positions.append([drone.snapshot for drone in drones])

        return drones, turns, turn_moves, turn_positions

    def run_simulation(self) -> None:
        """Run the turn-based multi-drone simulation to completion.

        Plans routes with ``Router.plan_routes``, or with
        ``Router.plan_routes_advanced`` refined by
        ``Router.refine_routes`` on maps ``Router.is_complex_map``
        flags, then runs ``_simulate`` once for real.

        Stores the outcome in ``self.drones``, ``self.turns`` and
        ``self.turn_moves`` instead of returning it.

        Raises:
            PathError: If the config isn't loaded, no route exists,
                or (see ``_simulate``) a turn goes by with no drone
                able to move.
        """
        if self.config is None:
            raise PathError("The config is not loaded.")
        router = Router(self.config)

        if router.is_complex_map():
            routes = router.refine_routes(
                router.plan_routes_advanced(),
                lambda trial: len(self._simulate(trial)[1]),
            )
        else:
            routes = router.plan_routes()

        drones, turns, turn_moves, turn_positions = self._simulate(routes)
        self.drones = drones
        self.turns = turns
        self.turn_moves = turn_moves
        self.turn_positions = turn_positions
