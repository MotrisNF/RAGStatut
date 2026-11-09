"""Application: the CLI orchestration, especially the y/n retry loop."""

import pytest

import main
from conftest import MakeInitiator
from initiate_simulation import Initiator


def _ready_initiator(make_initiator: MakeInitiator) -> Initiator:
    initiator = make_initiator("maps/test/1.txt")
    initiator.find_path_to_end()
    initiator.run_simulation()
    return initiator


@pytest.fixture(autouse=True)
def _silence_printer(monkeypatch: pytest.MonkeyPatch) -> None:
    # _offer_graphical_view's prompt doesn't pass testing=True, so it
    # would otherwise touch sys.stdin.fileno(), which pytest's
    # captured stdin can't provide. These tests care about the retry
    # logic, not the animation, so silence it instead.
    monkeypatch.setattr(
        main.Printer, "print_by_letter", staticmethod(lambda *a, **k: None)
    )


def test_reprompts_until_a_valid_answer_then_declines(
    make_initiator: MakeInitiator, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = main.Application()
    app.initiator = _ready_initiator(make_initiator)

    answers = iter(["maybe", "X", "n"])
    monkeypatch.setattr("builtins.input", lambda: next(answers))
    monkeypatch.setattr(
        main, "Pyshow", lambda initiator: pytest.fail("should not launch")
    )

    app._offer_graphical_view()


def test_launches_pyshow_on_y(
    make_initiator: MakeInitiator, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = main.Application()
    app.initiator = _ready_initiator(make_initiator)
    monkeypatch.setattr("builtins.input", lambda: "y")

    class _FakeShow:
        started = False

        def __init__(self, initiator: Initiator) -> None:
            self.initiator = initiator

        def start(self) -> None:
            _FakeShow.started = True

    monkeypatch.setattr(main, "Pyshow", _FakeShow)

    app._offer_graphical_view()
    assert _FakeShow.started is True


def test_accepts_uppercase_and_surrounding_whitespace(
    make_initiator: MakeInitiator, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = main.Application()
    app.initiator = _ready_initiator(make_initiator)

    monkeypatch.setattr("builtins.input", lambda: "  N  ")
    monkeypatch.setattr(
        main, "Pyshow", lambda initiator: pytest.fail("should not launch")
    )

    app._offer_graphical_view()
