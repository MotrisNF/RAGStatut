"""Drone: the per-drone state machine (moves, transit, delivery)."""

from conftest import LoadMap
from drone import Drone


def test_new_drone_starts_at_entry_waiting(load_map: LoadMap) -> None:
    config = load_map("maps/test/2.txt")
    drone = Drone(1, config)
    assert drone.current_hub == config.gates.entry.name
    assert drone.is_delivered is False
    assert drone.is_in_transit is False
    assert drone.total_moves == 0


def test_move_to_advances_a_single_turn_hop(load_map: LoadMap) -> None:
    config = load_map("maps/test/2.txt")
    drone = Drone(1, config)
    drone.set_path(["entry", "mid", "exit"])
    drone.move_to("mid")
    assert drone.current_hub == "mid"
    assert drone.path_progress == 1
    assert drone.total_moves == 1
    assert drone.is_delivered is False


def test_move_to_the_end_hub_delivers(load_map: LoadMap) -> None:
    config = load_map("maps/test/2.txt")
    drone = Drone(1, config)
    drone.set_path(["entry", "mid", "exit"])
    drone.move_to("mid")
    drone.move_to("exit")
    assert drone.is_delivered is True
    assert drone.current_hub == config.gates.exit.name


def test_transit_takes_the_declared_extra_turns_to_land(
    load_map: LoadMap,
) -> None:
    config = load_map("maps/test/3.txt")
    drone = Drone(1, config)
    drone.set_path(["start", "gate", "open", "goal"])

    # A restricted hop costs 2 turns total; start_transit is called
    # on the first of them, so extra_turns=1 is the one still owed.
    drone.start_transit("start-gate", "gate", extra_turns=1)
    assert drone.is_in_transit is True
    assert drone.display_hub == "gate"

    arrived = drone.advance_transit()
    assert arrived is True
    assert drone.is_in_transit is False
    assert drone.current_hub == "gate"


def test_transit_with_more_extra_turns_stays_in_flight_longer(
    load_map: LoadMap,
) -> None:
    config = load_map("maps/test/3.txt")
    drone = Drone(1, config)
    drone.set_path(["start", "gate", "open", "goal"])

    drone.start_transit("start-gate", "gate", extra_turns=2)
    arrived_early = drone.advance_transit()
    assert arrived_early is False
    assert drone.is_in_transit is True

    arrived = drone.advance_transit()
    assert arrived is True
    assert drone.is_in_transit is False


def test_snapshot_reports_the_destination_while_in_transit(
    load_map: LoadMap,
) -> None:
    config = load_map("maps/test/3.txt")
    drone = Drone(1, config)
    drone.set_path(["start", "gate", "open", "goal"])
    drone.start_transit("start-gate", "gate", extra_turns=1)

    snapshot = drone.snapshot
    assert snapshot.hub == "gate"
    assert snapshot.in_transit is True
    assert snapshot.zone == "restricted"


def test_next_hub_is_none_once_the_path_is_exhausted(
    load_map: LoadMap,
) -> None:
    config = load_map("maps/test/2.txt")
    drone = Drone(1, config)
    drone.set_path(["entry", "mid", "exit"])
    drone.move_to("mid")
    drone.move_to("exit")
    assert drone.next_hub() is None


def test_current_label_is_the_hub_at_rest_and_the_link_in_flight(
    load_map: LoadMap,
) -> None:
    config = load_map("maps/test/3.txt")
    drone = Drone(1, config)
    drone.set_path(["start", "gate", "open", "goal"])
    assert drone.current_label == "start"

    drone.start_transit("start-gate", "gate", extra_turns=1)
    assert drone.current_label == "start-gate"

    drone.advance_transit()
    assert drone.current_label == "gate"


def test_position_matches_the_display_hubs_coordinates(
    load_map: LoadMap,
) -> None:
    config = load_map("maps/test/3.txt")
    drone = Drone(1, config)
    drone.set_path(["start", "gate", "open", "goal"])

    gate = config.hubs["gate"]
    drone.start_transit("start-gate", "gate", extra_turns=1)
    assert drone.position == (gate.x, gate.y)


def test_drones_track_independent_state_by_id(load_map: LoadMap) -> None:
    config = load_map("maps/test/2.txt")
    first = Drone(1, config)
    second = Drone(2, config)
    first.set_path(["entry", "mid", "exit"])
    second.set_path(["entry", "mid", "exit"])

    first.move_to("mid")

    assert first.id == 1
    assert second.id == 2
    assert first.current_hub == "mid"
    assert second.current_hub == "entry"
    assert first.total_moves == 1
    assert second.total_moves == 0
