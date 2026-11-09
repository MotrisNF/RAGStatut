"""Pydantic models forming the validated representation of a map."""

from pydantic import BaseModel, field_validator
from typing import Optional

from constants import ZoneType


class Gate(BaseModel):
    """Position and color of a start or end zone.

    Attributes:
        x: X coordinate of the zone.
        y: Y coordinate of the zone.
        color: Optional visual color, ``None`` if not specified.
    """

    x: int
    y: int
    color: Optional[str] = None
    name: str


class Gates(BaseModel):
    """The two special gates of a map, kept apart for easy access.

    Attributes:
        entry: Gate matching the ``start_hub:`` definition.
        exit: Gate matching the ``end_hub:`` definition.
    """

    entry: Gate
    exit: Gate


class Connection(BaseModel):
    """A bidirectional link between two zones.

    Attributes:
        pos1: Name of one endpoint zone.
        pos2: Name of the other endpoint zone.
        capacity: Max drones allowed to cross simultaneously.
    """

    pos1: str
    pos2: str
    capacity: int = 1

    @field_validator("capacity")
    @classmethod
    def _capacity_positive(cls, value: int) -> int:
        """Reject a non-positive ``max_link_capacity`` value."""
        if value <= 0:
            raise ValueError(
                "max_link_capacity must be a positive integer"
            )
        return value


class Hub(BaseModel):
    """A single zone of the map.

    Attributes:
        x: X coordinate of the zone.
        y: Y coordinate of the zone.
        zone: Movement-cost type, or ``start``/``end`` for the gates.
        max_drones: Max simultaneous drones, ``None`` when unlimited
            (always the case for the ``start``/``end`` zones).
        color: Optional visual color, ``None`` if not specified.
    """

    x: int
    y: int
    zone: ZoneType = "normal"
    max_drones: Optional[int] = 1
    color: Optional[str] = None
    connection: list[Connection] = []

    @field_validator("max_drones")
    @classmethod
    def _max_drones_positive(cls, value: Optional[int]) -> Optional[int]:
        """Reject a non-positive ``max_drones`` value."""
        if value is not None and value <= 0:
            raise ValueError("max_drones must be a positive integer")
        return value


class MapConfig(BaseModel):
    """Fully validated representation of a parsed map file.

    Attributes:
        nb_drones: Number of drones to route from entry to exit.
        gates: The start and end zones.
        hubs: All zones (including the gates), keyed by name.
        connections: All links between zones.
        min_x: Smallest x coordinate used by any zone.
        max_x: Largest x coordinate used by any zone.
        min_y: Smallest y coordinate used by any zone.
        max_y: Largest y coordinate used by any zone.
        width: ``max_x - min_x``, the map's extent along x once
            shifted so ``min_x`` sits at 0.
        height: ``max_y - min_y``, the map's extent along y once
            shifted so ``min_y`` sits at 0.
    """

    nb_drones: int
    gates: Gates
    hubs: dict[str, Hub]
    connections: list[Connection]
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    width: int
    height: int

    @field_validator("nb_drones")
    @classmethod
    def _nb_drones_positive(cls, value: int) -> int:
        """Reject a non-positive ``nb_drones`` value."""
        if value <= 0:
            raise ValueError("nb_drones must be a positive integer")
        return value
