"""Shared fixtures: loading maps, or building tiny ones by hand."""

from typing import Callable, Optional

import pytest

from constants import ZoneType
from initiate_simulation import Initiator
from map_config import Connection, Gate, Gates, Hub, MapConfig
from parser import Parser

LoadMap = Callable[[str], MapConfig]
MakeInitiator = Callable[[str], Initiator]
MakeChainConfig = Callable[..., MapConfig]


@pytest.fixture
def load_map() -> LoadMap:
    """Return a helper that parses a map file into a ``MapConfig``."""
    def _load(path: str) -> MapConfig:
        with open(path) as file:
            return Parser().read_file(file)
    return _load


@pytest.fixture
def make_initiator(load_map: LoadMap) -> MakeInitiator:
    """Return a helper that loads a map straight into an ``Initiator``."""
    def _make(path: str) -> Initiator:
        return Initiator(load_map(path))
    return _make


@pytest.fixture
def make_chain_config() -> MakeChainConfig:
    """Return a helper that builds a straight-line map by hand.

    Useful for tests that need precise control over a hub's zone or
    capacity without maintaining a dedicated ``.txt`` fixture --
    builds ``start -> stage0 -> stage1 -> ... -> end`` in a single
    row, all connections capacity 1 unless overridden.
    """
    def _make(
        stages: list[tuple[ZoneType, Optional[int]]],
        nb_drones: int = 1,
        connection_capacities: Optional[list[int]] = None,
    ) -> MapConfig:
        names = ["start"] + [f"h{i}" for i in range(len(stages))] + ["end"]
        capacities = connection_capacities or [1] * (len(names) - 1)

        hubs: dict[str, Hub] = {
            "start": Hub(x=0, y=0, zone="start", max_drones=None),
            "end": Hub(
                x=len(names) - 1, y=0, zone="end", max_drones=None
            ),
        }
        for i, (zone, max_drones) in enumerate(stages):
            hubs[f"h{i}"] = Hub(
                x=i + 1, y=0, zone=zone, max_drones=max_drones
            )

        connections = [
            Connection(pos1=a, pos2=b, capacity=capacity)
            for a, b, capacity in zip(names, names[1:], capacities)
        ]

        return MapConfig(
            nb_drones=nb_drones,
            gates=Gates(
                entry=Gate(x=0, y=0, name="start"),
                exit=Gate(x=len(names) - 1, y=0, name="end"),
            ),
            hubs=hubs,
            connections=connections,
            min_x=0, max_x=len(names) - 1, min_y=0, max_y=0,
            width=len(names) - 1, height=0,
        )
    return _make
