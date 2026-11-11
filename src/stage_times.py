"""How long each stage of a local answer takes, learned from use.

The page shows a progress bar for each stage of a local answer. The model
cannot say how far it is, so the bar estimates it from the time the same
stage took before on this computer.
"""
import json
from pathlib import Path


class StageTimes:
    """The last durations of each stage, kept in a small JSON file.

    Attributes:
        FILE_NAME: Name of the file, in the folder of the generation logs.
        KEEP: How many durations of each stage are averaged.
    """

    FILE_NAME = "stage_times.json"
    KEEP = 20
    # Shorter is not a real model call (a test, or an empty stage).
    MIN_SECONDS = 1.0

    def __init__(self, path: Path) -> None:
        """Read the saved durations; a missing or broken file means none.

        Args:
            path: The JSON file.
        """
        self._path = path
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._times: dict[str, list[float]] = {
                str(stage): [float(t) for t in times]
                for stage, times in data.items()}
        except (OSError, ValueError, AttributeError, TypeError):
            self._times = {}

    def add(self, stage: str, seconds: float) -> None:
        """Remember how long a stage took, and save it.

        A file that cannot be written is ignored: the bar only uses the
        old estimate. Durations under ``MIN_SECONDS`` are not kept.
        """
        if seconds < self.MIN_SECONDS:
            return
        times = self._times.setdefault(stage, [])
        times.append(round(seconds, 1))
        del times[:-self.KEEP]
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._times), encoding="utf-8")
        except OSError:
            pass

    def averages(self) -> dict[str, float]:
        """The mean duration of each stage seen so far, in seconds."""
        kept = {stage: [t for t in times if t >= self.MIN_SECONDS]
                for stage, times in self._times.items()}
        return {stage: round(sum(times) / len(times), 1)
                for stage, times in kept.items() if times}
