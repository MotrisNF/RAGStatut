"""Initiator: the full run -- capacity, output format, completion."""

import re
from collections import Counter

import pytest

from conftest import MakeChainConfig, MakeInitiator
from initiate_simulation import Initiator

_TURN_LINE = re.compile(r"^D\d+-\S+( D\d+-\S+)*$")

_LIGHT_MAPS = [
    f"maps/test/{number}.txt" for number in range(1, 9)
]


@pytest.mark.parametrize("path", _LIGHT_MAPS)
def test_run_simulation_produces_consistent_logs(
    path: str, make_initiator: MakeInitiator
) -> None:
    initiator = make_initiator(path)
    initiator.find_path_to_end()
    initiator.run_simulation()

    assert len(initiator.turns) == len(initiator.turn_moves)
    assert len(initiator.turns) == len(initiator.turn_positions)
    assert len(initiator.turns) > 0
    for line in initiator.turns:
        assert _TURN_LINE.match(line), line


@pytest.mark.parametrize("path", _LIGHT_MAPS)
def test_every_drone_is_delivered_by_the_last_turn(
    path: str, make_initiator: MakeInitiator
) -> None:
    initiator = make_initiator(path)
    initiator.find_path_to_end()
    initiator.run_simulation()

    assert initiator.config is not None
    exit_name = initiator.config.gates.exit.name
    last_turn = initiator.turn_positions[-1]
    assert all(position.hub == exit_name for position in last_turn)


@pytest.mark.parametrize("path", _LIGHT_MAPS)
def test_hub_capacity_is_never_exceeded(
    path: str, make_initiator: MakeInitiator
) -> None:
    initiator = make_initiator(path)
    initiator.find_path_to_end()
    initiator.run_simulation()
    config = initiator.config
    assert config is not None

    gate_names = {config.gates.entry.name, config.gates.exit.name}
    for turn in initiator.turn_positions:
        occupancy: dict[str, int] = {}
        for position in turn:
            if position.in_transit or position.hub in gate_names:
                continue
            occupancy[position.hub] = occupancy.get(position.hub, 0) + 1
        for hub_name, count in occupancy.items():
            limit = config.hubs[hub_name].max_drones
            assert limit is not None
            assert count <= limit


def test_restricted_hop_takes_two_turns(
    make_initiator: MakeInitiator,
) -> None:
    # maps/test/3.txt: start(normal) -> gate(restricted) -> open -> goal.
    initiator = make_initiator("maps/test/3.txt")
    initiator.find_path_to_end()
    initiator.run_simulation()

    def label_for(drone_id: int, turn_index: int) -> str:
        return dict(initiator.turn_moves[turn_index])[drone_id]

    # Turn 0: still crossing (label names the connection). Turn 1:
    # arrived (label names the hub).
    assert label_for(1, 0) == "start-gate"
    assert label_for(1, 1) == "gate"


def test_complex_map_completes_without_deadlocking(
    make_initiator: MakeInitiator,
) -> None:
    initiator = make_initiator("maps/test/complex/9.txt")
    initiator.find_path_to_end()
    initiator.run_simulation()
    assert initiator.config is not None
    exit_name = initiator.config.gates.exit.name

    assert len(initiator.turns) > 0
    assert initiator.turn_positions[-1][0].hub == exit_name


def test_connection_capacity_limits_simultaneous_transit(
    make_chain_config: MakeChainConfig,
) -> None:
    # A restricted stage's crossing (2 turns) shows up in turn_moves
    # as the connection's own label ("start-h0") while in flight, so
    # its capacity is directly observable there, unlike hub capacity.
    config = make_chain_config(
        [("restricted", 3)], nb_drones=3, connection_capacities=[1, 1]
    )
    initiator = Initiator(config)
    initiator.run_simulation()

    for turn in initiator.turn_moves:
        in_flight = sum(1 for _, label in turn if label == "start-h0")
        assert in_flight <= 1


def test_create_drones_assigns_routes_in_id_order(
    make_initiator: MakeInitiator,
) -> None:
    initiator = make_initiator("maps/test/6.txt")
    initiator.find_path_to_end()
    initiator.run_simulation()

    by_second_hop = Counter(drone._path[1] for drone in initiator.drones)
    assert set(by_second_hop) == {"top", "bottom"}
    assert sum(by_second_hop.values()) == len(initiator.drones)
    for index, drone in enumerate(initiator.drones, start=1):
        assert drone.id == index
