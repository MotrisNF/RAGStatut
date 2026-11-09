"""State machine for a single drone travelling across the map."""

from typing import NamedTuple, Optional

from map_config import MapConfig
from constants import DroneStatus, ZoneType


class DronePosition(NamedTuple):
    """One drone's on-screen position at a given simulation turn.

    Attributes:
        drone_id: The drone this position belongs to.
        x: X coordinate to draw the drone at.
        y: Y coordinate to draw the drone at.
        hub: Name of the hub these coordinates belong to.
        zone: That hub's zone type (``"restricted"``, ``"blocked"``,
            ``"normal"``, ``"priority"``, ``"start"`` or ``"end"``) —
            check this instead of guessing from ``in_transit`` alone,
            since it names the actual hub involved.
        in_transit: True while the drone is mid-flight over a
            connection into ``hub``, in which case ``(x, y)`` is
            already that destination's position (the drone hasn't
            physically arrived yet). Callers that want to animate the
            crossing instead of snapping to the destination can use
            this flag to interpolate from the previous turn's
            position.
    """

    drone_id: int
    x: int
    y: int
    hub: str
    zone: ZoneType
    in_transit: bool


class Drone:
    """A single drone travelling from the start hub to the end hub.

    A drone follows a fixed path (assigned once by the scheduler via
    ``set_path``), one hop at a time. A hop into a
    ``normal``/``priority`` zone finishes in the same turn it starts
    (``move_to``); a hop into a ``restricted`` zone takes an extra
    turn during which the drone is "in flight" over the connection
    (``start_transit`` then ``advance_transit``).
    """

    def __init__(self, id: int, config: MapConfig) -> None:
        """Create a drone parked at the map's entry hub.

        Args:
            id: Unique identifier for this drone.
            config: The map configuration, used to read the entry
                and exit hub names.
        """
        self._id: int = id
        self._config: MapConfig = config
        self._status: DroneStatus = "waiting"
        self._total_moves: int = 0
        self._current_hub: str = config.gates.entry.name
        self._end_hub: str = config.gates.exit.name
        self._path: list[str] = []
        self._path_index: int = 0
        self._transit_connection: Optional[str] = None
        self._transit_destination: Optional[str] = None
        self._transit_remaining: int = 0

    @property
    def id(self) -> int:
        """Unique identifier, used to format output as ``D<id>``."""
        return self._id

    @property
    def current_hub(self) -> str:
        """Name of the hub this drone last departed from or reached."""
        return self._current_hub

    @property
    def display_hub(self) -> str:
        """Name of the hub these on-screen coordinates belong to.

        This is the hub it currently occupies, except while in
        transit over a connection, when it's the destination hub it's
        flying towards (see ``is_in_transit``).
        """
        return self._transit_destination or self._current_hub

    @property
    def position(self) -> tuple[int, int]:
        """(x, y) coordinates to draw this drone at right now.

        See ``display_hub`` for which hub these belong to.
        """
        hub = self._config.hubs[self.display_hub]
        return hub.x, hub.y

    @property
    def snapshot(self) -> DronePosition:
        """This drone's position and transit state, as a ``DronePosition``."""
        hub_name = self.display_hub
        hub = self._config.hubs[hub_name]
        return DronePosition(
            self._id, hub.x, hub.y, hub_name, hub.zone, self.is_in_transit
        )

    @property
    def total_moves(self) -> int:
        """Number of moves (single-turn or transit starts) taken."""
        return self._total_moves

    @property
    def path_progress(self) -> int:
        """Index of the current hub within this drone's assigned path."""
        return self._path_index

    @property
    def remaining_hops(self) -> int:
        """Hops still needed to reach the end of this drone's path.

        Unlike ``path_progress``, this is comparable across drones
        with differently-sized routes: the scheduler uses it to let
        whichever drone is actually closest to the end act first
        within a turn, so a hub it frees can be used the same turn
        by the drone behind it -- even when that drone took a longer
        prefix to get there.
        """
        return len(self._path) - 1 - self._path_index

    @property
    def is_delivered(self) -> bool:
        """Whether this drone has reached the end hub."""
        return self._status == "delivered"

    @property
    def is_in_transit(self) -> bool:
        """Whether this drone is mid-flight over a restricted link."""
        return self._status == "in_transit"

    @property
    def current_label(self) -> str:
        """Zone or connection name to report for this drone now."""
        if self._transit_connection is not None:
            return self._transit_connection
        return self._current_hub

    def set_path(self, path: list[str]) -> None:
        """Assign the hub sequence this drone will follow to the end.

        Args:
            path: Hub names from the drone's current hub to the end
                hub, inclusive of both endpoints.
        """
        self._path = path
        self._path_index = 0

    def next_hub(self) -> Optional[str]:
        """Return the next hub on this drone's path, if any remains."""
        next_index = self._path_index + 1
        if next_index >= len(self._path):
            return None
        return self._path[next_index]

    def move_to(self, hub_name: str) -> None:
        """Complete a single-turn move into an adjacent hub.

        Args:
            hub_name: Name of the hub the drone moves into.
        """
        self._total_moves += 1
        self._arrive(hub_name)

    def start_transit(
        self, connection_name: str, destination: str, extra_turns: int
    ) -> None:
        """Begin crossing a connection into a ``restricted`` hub.

        Args:
            connection_name: Name of the connection now being
                crossed, reported while the drone is in flight.
            destination: Hub the drone will reach once the transit
                finishes.
            extra_turns: Turns still needed, after this one, before
                arrival.
        """
        self._total_moves += 1
        self._status = "in_transit"
        self._transit_connection = connection_name
        self._transit_destination = destination
        self._transit_remaining = extra_turns

    def advance_transit(self) -> bool:
        """Progress one turn of an in-flight transit.

        Returns:
            True if the drone has just arrived at its destination.
        """
        self._transit_remaining -= 1
        if self._transit_remaining <= 0:
            assert self._transit_destination is not None
            self._arrive(self._transit_destination)
            return True
        return False

    def _arrive(self, hub_name: str) -> None:
        """Land the drone in ``hub_name`` and clear transit state.

        Args:
            hub_name: Name of the hub the drone now occupies.
        """
        self._current_hub = hub_name
        self._path_index += 1
        self._transit_connection = None
        self._transit_destination = None
        self._transit_remaining = 0
        self._status = (
            "delivered" if hub_name == self._end_hub else "waiting"
        )
