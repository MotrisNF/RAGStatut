"""Parser: valid maps load cleanly, malformed ones raise FileError."""

import glob

import pytest

from conftest import LoadMap
from exceptions import FileError
from parser import Parser

_BAD_MAPS = sorted(
    glob.glob("maps/bad_map/*.txt"),
    key=lambda path: int(path.split("/")[-1].split(".")[0]),
)


@pytest.mark.parametrize("path", _BAD_MAPS)
def test_bad_map_raises_file_error(path: str) -> None:
    with open(path) as file:
        with pytest.raises(FileError):
            Parser().read_file(file)


def test_valid_map_parses(load_map: LoadMap) -> None:
    config = load_map("maps/test/1.txt")
    assert config.nb_drones == 2
    assert config.gates.entry.name == "a"
    assert config.gates.exit.name == "b"
    assert set(config.hubs) == {"a", "b", "medium", "other"}
    assert len(config.connections) == 3


def test_hub_metadata_is_read(load_map: LoadMap) -> None:
    config = load_map("maps/test/3.txt")
    assert config.hubs["gate"].zone == "restricted"
    assert config.hubs["open"].zone == "normal"


def test_max_drones_is_ignored_on_the_gates(load_map: LoadMap) -> None:
    # VII.4: max_drones on start_hub/end_hub is ignored, not an error,
    # even though it's meaningless there (the gates are uncapped).
    config = load_map("maps/test/gate_max_drones_ignored.txt")
    assert config.hubs[config.gates.entry.name].max_drones is None
    assert config.hubs[config.gates.exit.name].max_drones is None
