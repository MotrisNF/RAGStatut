"""MapConfig's own pydantic validators, independent of the parser.

The parser wraps every ``ValidationError`` these raise into a
``FileError`` (see tests/test_parser.py's bad_map cases) -- these
tests check the models' own guarantees hold even when built directly.
"""

import pytest
from pydantic import ValidationError

from conftest import LoadMap
from map_config import Connection, Hub, MapConfig


def test_hub_rejects_zero_max_drones() -> None:
    with pytest.raises(ValidationError):
        Hub(x=0, y=0, max_drones=0)


def test_hub_rejects_negative_max_drones() -> None:
    with pytest.raises(ValidationError):
        Hub(x=0, y=0, max_drones=-1)


def test_hub_allows_unlimited_max_drones() -> None:
    hub = Hub(x=0, y=0, zone="start", max_drones=None)
    assert hub.max_drones is None


def test_hub_defaults_to_one_drone_and_normal_zone() -> None:
    hub = Hub(x=0, y=0)
    assert hub.max_drones == 1
    assert hub.zone == "normal"


def test_connection_rejects_zero_capacity() -> None:
    with pytest.raises(ValidationError):
        Connection(pos1="a", pos2="b", capacity=0)


def test_connection_rejects_negative_capacity() -> None:
    with pytest.raises(ValidationError):
        Connection(pos1="a", pos2="b", capacity=-3)


def test_connection_defaults_to_capacity_one() -> None:
    connection = Connection(pos1="a", pos2="b")
    assert connection.capacity == 1


def test_map_config_rejects_zero_drones(load_map: LoadMap) -> None:
    config = load_map("maps/test/1.txt")
    data = config.model_dump()
    data["nb_drones"] = 0
    with pytest.raises(ValidationError):
        MapConfig(**data)
