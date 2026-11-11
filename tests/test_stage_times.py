"""The learned duration of each local stage."""
from pathlib import Path

from src.stage_times import StageTimes


def test_average_of_the_last(tmp_path: Path) -> None:
    """The mean of what was seen, and it survives a restart."""
    path = tmp_path / "t.json"
    times = StageTimes(path)
    times.add("convenio", 30)
    times.add("convenio", 40)
    times.add("respuesta", 20)
    assert StageTimes(path).averages() == {"convenio": 35.0,
                                           "respuesta": 20.0}


def test_only_the_last_are_kept(tmp_path: Path) -> None:
    """Old durations stop counting."""
    times = StageTimes(tmp_path / "t.json")
    for _ in range(StageTimes.KEEP):
        times.add("convenio", 100)
    times.add("convenio", 10)
    assert times.averages()["convenio"] < 100
    assert len(times._times["convenio"]) == StageTimes.KEEP


def test_broken_file_means_nothing(tmp_path: Path) -> None:
    """A broken file is ignored, never an error."""
    path = tmp_path / "t.json"
    path.write_text("roto")
    assert StageTimes(path).averages() == {}


def test_instant_stages_are_ignored(tmp_path: Path) -> None:
    """A stage of a few milliseconds never becomes the estimate."""
    path = tmp_path / "t.json"
    times = StageTimes(path)
    times.add("convenio", 0.0)
    path.write_text('{"convenio": [0.0, 0.2]}')
    assert StageTimes(path).averages() == {}
