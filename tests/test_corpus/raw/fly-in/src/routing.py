"""Graph algorithms that find and assign entry->exit routes.

Kept apart from ``initiate_simulation.py`` so the turn-based scheduler
(``Initiator``) doesn't have to live alongside the route-planning
algorithms (``Router``); the two only meet where ``Router.refine_routes``
needs to score a trial assignment by actually simulating it.
"""

import heapq
from typing import Callable, Optional

from constants import (
    ZoneType,
    NORMAL_ZONE_TURN_COST,
    RESTRICTED_ZONE_TURN_COST,
    MAX_CANDIDATE_PATHS,
    PATH_COST_CEILING_RATIO,
    PRIORITY_BIAS,
    COMPLEXITY_THRESHOLD,
    REFINE_TRIAL_BUDGET,
)
from map_config import MapConfig
from exceptions import PathError


class Router:
    """Finds and plans entry->exit routes over a validated map graph."""

    @staticmethod
    def zone_cost(zone: ZoneType) -> Optional[int]:
        """Return the turns needed to move into a hub of this zone type.

        Args:
            zone: The destination hub's zone type.

        Returns:
            The movement cost in turns, or ``None`` if the zone is
            ``blocked`` and can never be entered.
        """
        if zone == "blocked":
            return None
        if zone == "restricted":
            return RESTRICTED_ZONE_TURN_COST
        return NORMAL_ZONE_TURN_COST

    @staticmethod
    def connection_capacities(
        config: MapConfig
    ) -> dict[frozenset[str], int]:
        """Map each connection's endpoints to its crossing capacity.

        Args:
            config: The validated map.

        Returns:
            A lookup from ``{pos1, pos2}`` to ``max_link_capacity``.
        """
        return {
            frozenset((c.pos1, c.pos2)): c.capacity
            for c in config.connections
        }

    def __init__(self, config: MapConfig) -> None:
        """Attach every connection to the hubs it links.

        Args:
            config: The validated map to route drones through.
        """
        self.config = config
        self._fill_hub_connections()

    def _fill_hub_connections(self) -> None:
        """Rebuild each hub's connection list from ``self.config``."""
        for key, hub in self.config.hubs.items():
            hub.connection = []
            for connection in self.config.connections:
                if connection.pos1 == key:
                    hub.connection.append(connection)
                elif connection.pos2 == key:
                    hub.connection.append(connection)

    def find_path_to_end(self) -> bool:
        """Check whether the end hub is reachable from the start hub.

        A DFS over the hub graph that ignores movement costs and
        capacities and refuses to cross ``blocked`` hubs.

        Returns:
            True if at least one valid route exists.
        """
        start = self.config.gates.entry.name
        end = self.config.gates.exit.name
        visited: set[str] = set()
        to_visit = [start]
        while to_visit:
            current = to_visit.pop()
            if current == end:
                return True
            if current in visited:
                continue
            visited.add(current)
            hub = self.config.hubs.get(current)
            if hub is None:
                continue
            for connection in hub.connection:
                if connection.pos1 == current:
                    neighbor = connection.pos2
                else:
                    neighbor = connection.pos1
                if neighbor in visited:
                    continue
                neighbor_hub = self.config.hubs.get(neighbor)
                if neighbor_hub is not None:
                    if neighbor_hub.zone == "blocked":
                        continue
                to_visit.append(neighbor)
        return False

    @staticmethod
    def _route_weight(zone: ZoneType) -> Optional[float]:
        """Route-selection weight for entering a hub of this zone.

        Used only while searching for candidate routes (see
        ``_dijkstra``), never for the real per-turn cost the
        simulation charges (``zone_cost`` handles that). Matches
        ``zone_cost`` except ``priority`` gets a negligible discount
        so it's preferred on ties.

        Args:
            zone: The destination hub's zone type.

        Returns:
            The route weight, or ``None`` if the zone is ``blocked``
            and can never be entered.
        """
        if zone == "blocked":
            return None
        if zone == "restricted":
            return float(RESTRICTED_ZONE_TURN_COST)
        if zone == "priority":
            return float(NORMAL_ZONE_TURN_COST) - PRIORITY_BIAS
        return float(NORMAL_ZONE_TURN_COST)

    def _dijkstra(
        self, penalties: dict[frozenset[str], float]
    ) -> tuple[list[str], float]:
        """Find the cheapest entry->exit route under extra penalties.

        Plain Dijkstra over the hub graph, weighted by
        ``_route_weight`` plus, per connection, whatever extra cost
        ``penalties`` assigns it. Called with an empty ``penalties``
        this is just the single cheapest route; callers can steer
        successive calls away from already-used connections by
        growing their penalty.

        Args:
            penalties: Extra route weight per connection (keyed by
                its endpoint pair), added on top of ``_route_weight``.

        Returns:
            The hub-name path (inclusive of both gates) and its cost.

        Raises:
            PathError: If no route avoiding ``blocked`` hubs connects
                entry to exit.
        """
        config = self.config
        start = config.gates.entry.name
        end = config.gates.exit.name

        distances: dict[str, float] = {start: 0.0}
        previous: dict[str, str] = {}
        visited: set[str] = set()
        queue: list[tuple[float, str]] = [(0.0, start)]

        while queue:
            dist, current = heapq.heappop(queue)
            if current in visited:
                continue
            visited.add(current)
            if current == end:
                break
            hub = config.hubs.get(current)
            if hub is None:
                continue
            for connection in hub.connection:
                if connection.pos1 == current:
                    neighbor = connection.pos2
                else:
                    neighbor = connection.pos1
                if neighbor in visited:
                    continue
                neighbor_hub = config.hubs.get(neighbor)
                if neighbor_hub is None:
                    continue
                weight = self._route_weight(neighbor_hub.zone)
                if weight is None:
                    continue
                weight += penalties.get(frozenset((current, neighbor)), 0.0)
                new_dist = dist + weight
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(queue, (new_dist, neighbor))

        if end not in visited:
            raise PathError("No valid path between start and end.")

        path = [end]
        while path[-1] != start:
            path.append(previous[path[-1]])
        path.reverse()
        return path, distances[end]

    def _bottleneck_capacity(
        self, path: list[str], link_capacity: dict[frozenset[str], int]
    ) -> int:
        """How many drones a route can have in flight/parked at once.

        The tightest of every intermediate hub's ``max_drones`` and
        every connection's ``max_link_capacity`` along the route; the
        gates themselves are uncapped and skipped.

        Args:
            path: Hub names from entry to exit, inclusive.
            link_capacity: Lookup from ``connection_capacities``.

        Returns:
            The route's bottleneck capacity (at least 1).
        """
        config = self.config
        start_name = config.gates.entry.name
        end_name = config.gates.exit.name

        limits = [
            config.hubs[name].max_drones
            for name in path
            if name not in (start_name, end_name)
        ]
        limits += [
            link_capacity[frozenset((a, b))]
            for a, b in zip(path, path[1:])
        ]
        usable_limits = [limit for limit in limits if limit is not None]
        return min(usable_limits) if usable_limits else 1

    def _bottleneck_throughput(
        self, path: list[str], link_capacity: dict[frozenset[str], int]
    ) -> float:
        """Steady-state drones/turn this route can actually sustain.

        Unlike ``_bottleneck_capacity``, dividing each stage's
        capacity by its crossing cost (``zone_cost``) and keeping the
        smallest result correctly weighs a capacity-1 ``restricted``
        hub (2 turns to cross) as half the throughput of a capacity-1
        ``normal`` one, instead of counting them as equally tight.

        Args:
            path: Hub names from entry to exit, inclusive.
            link_capacity: Lookup from ``connection_capacities``.

        Returns:
            The route's sustainable throughput, in drones per turn.
        """
        config = self.config
        start_name = config.gates.entry.name
        end_name = config.gates.exit.name

        rates = []
        for name in path:
            if name in (start_name, end_name):
                continue
            hub = config.hubs[name]
            cost = self.zone_cost(hub.zone)
            if hub.max_drones is not None and cost is not None:
                rates.append(hub.max_drones / cost)
        for a, b in zip(path, path[1:]):
            cost = self.zone_cost(config.hubs[b].zone)
            if cost is not None:
                rates.append(link_capacity[frozenset((a, b))] / cost)
        return min(rates) if rates else 1.0

    def _generate_candidate_routes(self) -> list[tuple[list[str], float]]:
        """Find up to ``MAX_CANDIDATE_PATHS`` diverse entry->exit routes.

        Repeatedly calls ``_dijkstra``, penalizing each connection a
        route used by that connection's own weight (not the route's
        total cost, so a forced shared prefix isn't penalized far
        more than its one-time use warrants) before searching for the
        next one, steering later calls towards unused parts of the
        graph. Stops once a route repeats an earlier one, or gets too
        expensive relative to the cheapest one
        (``PATH_COST_CEILING_RATIO``).

        Shared by ``plan_routes`` and ``plan_routes_advanced``, which
        only differ in how they assign drones to these candidates.

        Returns:
            ``(path, cost)`` pairs, cheapest first found.
        """
        config = self.config

        candidates: list[tuple[list[str], float]] = []
        penalties: dict[frozenset[str], float] = {}
        base_cost: Optional[float] = None
        max_candidates = min(MAX_CANDIDATE_PATHS, config.nb_drones)

        while len(candidates) < max_candidates:
            path, cost = self._dijkstra(penalties)
            if base_cost is None:
                base_cost = cost
            elif cost > base_cost * PATH_COST_CEILING_RATIO:
                break
            if any(path == existing for existing, _ in candidates):
                break
            candidates.append((path, cost))
            for a, b in zip(path, path[1:]):
                key = frozenset((a, b))
                edge_weight = self._route_weight(config.hubs[b].zone) or 0.0
                penalties[key] = penalties.get(key, 0.0) + edge_weight

        return candidates

    def is_complex_map(self) -> bool:
        """Whether contention is severe enough to need the advanced planner.

        ``nb_drones`` divided by the single cheapest route's
        sustainable throughput (``_bottleneck_throughput``); past
        ``COMPLEXITY_THRESHOLD`` the simpler capacity count in
        ``plan_routes`` stops being an accurate enough proxy.

        Returns:
            True if ``plan_routes_advanced`` should be used instead
            of ``plan_routes``.
        """
        path, _ = self._dijkstra({})
        link_capacity = self.connection_capacities(self.config)
        throughput = self._bottleneck_throughput(path, link_capacity)
        return self.config.nb_drones / throughput > COMPLEXITY_THRESHOLD

    def plan_routes(self) -> list[list[str]]:
        """Compute one route per drone, spreading load across routes.

        The default planner. Assigns each drone, one at a time, to
        whichever candidate route (``_generate_candidate_routes``)
        currently has the lowest projected cost -- its route weight
        plus drones already assigned to it divided by its bottleneck
        capacity -- a greedy load balancer that spreads drones across
        routes instead of funnelling them all down one.

        Returns:
            One hub-name path (inclusive of both gates) per drone,
            ``nb_drones`` entries long, in drone-id order.
        """
        candidates = self._generate_candidate_routes()

        link_capacity = self.connection_capacities(self.config)
        bottlenecks = [
            self._bottleneck_capacity(path, link_capacity)
            for path, _ in candidates
        ]

        assigned_counts = [0] * len(candidates)
        routes: list[list[str]] = []
        for _ in range(self.config.nb_drones):
            best_index = min(
                range(len(candidates)),
                key=lambda i: (
                    candidates[i][1] + assigned_counts[i] / bottlenecks[i]
                ),
            )
            assigned_counts[best_index] += 1
            routes.append(candidates[best_index][0])
        return routes

    def plan_routes_advanced(self) -> list[list[str]]:
        """Compute one route per drone, using true per-route throughput.

        Same as ``plan_routes``, but divides by
        ``_bottleneck_throughput`` instead of raw capacity, so a
        ``restricted`` bottleneck is weighed as half the throughput
        of a capacity-equal ``normal`` one instead of equally tight.

        Returns:
            One hub-name path (inclusive of both gates) per drone,
            ``nb_drones`` entries long, in drone-id order.
        """
        candidates = self._generate_candidate_routes()

        link_capacity = self.connection_capacities(self.config)
        throughputs = [
            self._bottleneck_throughput(path, link_capacity)
            for path, _ in candidates
        ]

        assigned_counts = [0] * len(candidates)
        routes: list[list[str]] = []
        for _ in range(self.config.nb_drones):
            best_index = min(
                range(len(candidates)),
                key=lambda i: (
                    candidates[i][1] + assigned_counts[i] / throughputs[i]
                ),
            )
            assigned_counts[best_index] += 1
            routes.append(candidates[best_index][0])
        return routes

    def refine_routes(
        self,
        routes: list[list[str]],
        count_turns: Callable[[list[list[str]]], int],
    ) -> list[list[str]]:
        """Locally search for a better per-drone route assignment.

        ``plan_routes_advanced`` scores each candidate route in
        isolation, so it can't see two candidates sharing a
        bottleneck stage and competing for the same throughput. This
        instead measures the real outcome: it calls ``count_turns``
        for trial reassignments and keeps whatever lowers the total
        turn count.

        Hill-climbs from ``routes``: tries moving one drone at a time
        onto each other candidate route, keeping any swap that lowers
        the simulated turn count, until a full pass finds no further
        improvement or the trial budget runs out.

        Args:
            routes: The starting assignment, one route per drone.
            count_turns: Runs the real scheduler for a trial
                assignment and returns how many turns it took.

        Returns:
            The best assignment found -- never worse than ``routes``.
        """
        candidates = self._generate_candidate_routes()
        best_routes = list(routes)
        best_turns = count_turns(best_routes)

        trials_left = REFINE_TRIAL_BUDGET
        improved = True
        while improved and trials_left > 0:
            improved = False
            for drone_index in range(len(best_routes)):
                for candidate_path, _ in candidates:
                    if candidate_path == best_routes[drone_index]:
                        continue
                    if trials_left <= 0:
                        break
                    trial_routes = list(best_routes)
                    trial_routes[drone_index] = candidate_path
                    trials_left -= 1
                    trial_turns = count_turns(trial_routes)
                    if trial_turns < best_turns:
                        best_routes = trial_routes
                        best_turns = trial_turns
                        improved = True
        return best_routes
