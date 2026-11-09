"""Router: pathfinding and route planning, independent of scheduling."""

from conftest import LoadMap, MakeChainConfig
from map_config import Connection, Gate, Gates, Hub, MapConfig
from routing import Router


def test_find_path_to_end_true_when_reachable(load_map: LoadMap) -> None:
    router = Router(load_map("maps/test/1.txt"))
    assert router.find_path_to_end() is True


def test_find_path_to_end_false_when_end_hub_is_unreachable(
    load_map: LoadMap,
) -> None:
    router = Router(load_map("maps/test/unreachable.txt"))
    assert router.find_path_to_end() is False


def test_find_path_to_end_false_when_the_only_route_is_blocked(
    load_map: LoadMap,
) -> None:
    # maps/test/blocked_only_path.txt: start and goal are
    # topologically connected, but the sole hub between them is
    # zone=blocked -- a different failure mode from unreachable.txt,
    # where nothing connects to the end hub at all.
    router = Router(load_map("maps/test/blocked_only_path.txt"))
    assert router.find_path_to_end() is False


def test_find_path_to_end_ignores_blocked_detours(load_map: LoadMap) -> None:
    # maps/test/6.txt's middle hub is zone=blocked and isn't on the
    # only route from start to goal.
    router = Router(load_map("maps/test/6.txt"))
    assert router.find_path_to_end() is True


def test_plan_routes_returns_one_route_per_drone(load_map: LoadMap) -> None:
    config = load_map("maps/test/7.txt")
    router = Router(config)
    routes = router.plan_routes()
    assert len(routes) == config.nb_drones
    for route in routes:
        assert route[0] == config.gates.entry.name
        assert route[-1] == config.gates.exit.name


def test_plan_routes_never_uses_a_blocked_hub(load_map: LoadMap) -> None:
    # maps/test/complex/5.txt: a grid where several blocked cells are
    # still connected to their neighbours -- routing must skip them
    # because of their zone, not because they're unreachable.
    config = load_map("maps/test/complex/5.txt")
    router = Router(config)
    for route in router.plan_routes():
        for name in route:
            assert config.hubs[name].zone != "blocked"


def test_plan_routes_spreads_drones_when_branches_exist(
    load_map: LoadMap,
) -> None:
    # maps/test/6.txt: two branches of different capacity from start
    # to goal, five drones -- both should end up used, not just one.
    config = load_map("maps/test/6.txt")
    router = Router(config)
    routes = router.plan_routes()
    used_second_hops = {route[1] for route in routes}
    assert len(used_second_hops) > 1


def test_is_complex_map_false_for_a_light_map(load_map: LoadMap) -> None:
    router = Router(load_map("maps/test/1.txt"))
    assert router.is_complex_map() is False


def test_is_complex_map_true_for_a_severe_bottleneck(
    load_map: LoadMap,
) -> None:
    router = Router(load_map("maps/test/complex/9.txt"))
    assert router.is_complex_map() is True


def test_refine_routes_never_makes_the_starting_assignment_worse(
    load_map: LoadMap,
) -> None:
    config = load_map("maps/test/complex/9.txt")
    router = Router(config)
    starting_routes = router.plan_routes_advanced()

    def count_turns(routes: list[list[str]]) -> int:
        # A cheap stand-in for the real scheduler: just checks the
        # search only ever proposes valid, same-shaped assignments.
        assert len(routes) == config.nb_drones
        return sum(len(route) for route in routes)

    refined = router.refine_routes(starting_routes, count_turns)
    assert count_turns(refined) <= count_turns(starting_routes)


def test_zone_cost_matches_the_subject_rules() -> None:
    assert Router.zone_cost("normal") == 1
    assert Router.zone_cost("priority") == 1
    assert Router.zone_cost("restricted") == 2
    assert Router.zone_cost("blocked") is None


def test_bottleneck_throughput_halves_for_a_restricted_stage(
    make_chain_config: MakeChainConfig,
) -> None:
    normal_config = make_chain_config([("normal", 1)])
    restricted_config = make_chain_config([("restricted", 1)])

    normal_throughput = Router(normal_config)._bottleneck_throughput(
        ["start", "h0", "end"],
        Router.connection_capacities(normal_config)
    )
    restricted_throughput = Router(
        restricted_config
    )._bottleneck_throughput(
        ["start", "h0", "end"],
        Router.connection_capacities(restricted_config)
    )
    assert restricted_throughput == normal_throughput / 2


def test_generate_candidate_routes_finds_both_branches(
    load_map: LoadMap,
) -> None:
    config = load_map("maps/test/6.txt")
    router = Router(config)
    candidates = router._generate_candidate_routes()
    branch_hops = {path[1] for path, _ in candidates}
    assert branch_hops == {"top", "bottom"}


def test_dijkstra_prefers_priority_hubs_on_ties() -> None:
    # Two equal-length branches from start to end; only the second
    # one is zone=priority. Real per-turn cost is a tie (both cost 1
    # to enter), but the route-selection weight isn't.
    config = MapConfig(
        nb_drones=1,
        gates=Gates(
            entry=Gate(x=0, y=0, name="start"),
            exit=Gate(x=2, y=0, name="end"),
        ),
        hubs={
            "start": Hub(x=0, y=0, zone="start", max_drones=None),
            "normal_branch": Hub(x=1, y=-1, zone="normal", max_drones=1),
            "priority_branch": Hub(
                x=1, y=1, zone="priority", max_drones=1
            ),
            "end": Hub(x=2, y=0, zone="end", max_drones=None),
        },
        connections=[
            Connection(pos1="start", pos2="normal_branch", capacity=1),
            Connection(pos1="normal_branch", pos2="end", capacity=1),
            Connection(pos1="start", pos2="priority_branch", capacity=1),
            Connection(pos1="priority_branch", pos2="end", capacity=1),
        ],
        min_x=0, max_x=2, min_y=-1, max_y=1, width=2, height=2,
    )
    path, _ = Router(config)._dijkstra({})
    assert "priority_branch" in path
    assert "normal_branch" not in path


def test_advanced_planner_favours_the_higher_throughput_branch() -> None:
    # Same shape as the priority test above, but one branch is
    # restricted (half the throughput of the other despite equal
    # capacity) instead of merely priority.
    config = MapConfig(
        nb_drones=20,
        gates=Gates(
            entry=Gate(x=0, y=0, name="start"),
            exit=Gate(x=2, y=0, name="end"),
        ),
        hubs={
            "start": Hub(x=0, y=0, zone="start", max_drones=None),
            "normal_branch": Hub(x=1, y=-1, zone="normal", max_drones=1),
            "restricted_branch": Hub(
                x=1, y=1, zone="restricted", max_drones=1
            ),
            "end": Hub(x=2, y=0, zone="end", max_drones=None),
        },
        connections=[
            Connection(pos1="start", pos2="normal_branch", capacity=1),
            Connection(pos1="normal_branch", pos2="end", capacity=1),
            Connection(
                pos1="start", pos2="restricted_branch", capacity=1
            ),
            Connection(pos1="restricted_branch", pos2="end", capacity=1),
        ],
        min_x=0, max_x=2, min_y=-1, max_y=1, width=2, height=2,
    )
    router = Router(config)

    def on_restricted_branch(routes: list[list[str]]) -> int:
        return sum(1 for route in routes if "restricted_branch" in route)

    simple_count = on_restricted_branch(router.plan_routes())
    advanced_count = on_restricted_branch(router.plan_routes_advanced())
    assert advanced_count < simple_count
