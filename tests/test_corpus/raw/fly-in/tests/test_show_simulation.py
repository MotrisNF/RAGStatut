"""Pyshow: the pure animation-timing helpers (no pygame window needed)."""

from conftest import MakeInitiator
from constants import TURN_DURATION_FRAMES, TURN_PAUSE_FRAMES
from show_simulation import Pyshow

_MOVE_FRAMES = TURN_DURATION_FRAMES - TURN_PAUSE_FRAMES


def _pyshow(make_initiator: MakeInitiator) -> Pyshow:
    initiator = make_initiator("maps/test/1.txt")
    initiator.find_path_to_end()
    initiator.run_simulation()
    return Pyshow(initiator)


def test_interpolate_stays_put_before_its_first_waypoint(
    make_initiator: MakeInitiator,
) -> None:
    show = _pyshow(make_initiator)
    show._frame_count = 0
    position, direction = show._interpolate([(-1, 0, 0), (0, 100, 0)])
    assert position == (0, 0)
    assert direction == (100, 0)


def test_interpolate_reaches_the_target_once_the_turn_completes(
    make_initiator: MakeInitiator,
) -> None:
    show = _pyshow(make_initiator)
    show._frame_count = TURN_DURATION_FRAMES
    position, _ = show._interpolate([(-1, 0, 0), (0, 100, 0)])
    assert position == (100, 0)


def test_interpolate_is_partway_through_mid_turn(
    make_initiator: MakeInitiator,
) -> None:
    show = _pyshow(make_initiator)
    show._frame_count = _MOVE_FRAMES // 2
    position, _ = show._interpolate([(-1, 0, 0), (0, 100, 0)])
    assert 0 < position[0] < 100


def test_interpolate_keeps_facing_the_last_real_direction_once_stopped(
    make_initiator: MakeInitiator,
) -> None:
    show = _pyshow(make_initiator)
    waypoints = [(-1, 0, 0), (0, 100, 0), (1, 100, 0)]
    show._frame_count = 2 * TURN_DURATION_FRAMES
    position, direction = show._interpolate(waypoints)
    assert position == (100, 0)
    assert direction == (100, 0)


def test_interpolate_spans_two_turns_at_half_speed_for_a_restricted_hop(
    make_initiator: MakeInitiator,
) -> None:
    show = _pyshow(make_initiator)
    # Waypoints skip turn 0, exactly like a drone that starts a
    # restricted (2-turn) hop on turn 0 and doesn't rest again until
    # turn 1 -- the same shape _simulate produces for that case.
    waypoints = [(-1, 0, 0), (1, 100, 0)]
    move_frames = 2 * _MOVE_FRAMES

    show._frame_count = move_frames // 2
    halfway, _ = show._interpolate(waypoints)
    assert 0 < halfway[0] < 100

    show._frame_count = move_frames
    arrived, _ = show._interpolate(waypoints)
    assert arrived == (100, 0)


def test_interpolate_direction_points_along_the_vertical_axis(
    make_initiator: MakeInitiator,
) -> None:
    show = _pyshow(make_initiator)
    show._frame_count = 0
    _, direction = show._interpolate([(-1, 0, 0), (0, 0, -50)])
    assert direction == (0, -50)


def test_to_screen_places_the_min_corner_at_the_offset(
    make_initiator: MakeInitiator,
) -> None:
    show = _pyshow(make_initiator)
    show._offset_x = 10
    show._offset_y = 20
    show._spacing_x = 5.0
    show._spacing_y = 5.0
    assert show._to_screen(show._c.min_x, show._c.min_y) == (10, 20)
