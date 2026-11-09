"""Request scheduler."""
from enum import Enum


class PreemptionMode(Enum):
    """How a running request gives its memory back when space runs out.

    SWAP copies its key-value blocks to CPU memory and back later.
    RECOMPUTE drops the blocks and recomputes them when the request resumes;
    it is the default because recomputing is usually faster than copying.
    """

    SWAP = "swap"
    RECOMPUTE = "recompute"


class Scheduler:
    """Decides which requests run in the next step."""

    def __init__(self, max_num_seqs: int = 256) -> None:
        """Keep at most ``max_num_seqs`` requests running at once."""
        self.max_num_seqs = max_num_seqs
        self.waiting: list[str] = []
        self.running: list[str] = []

    def schedule(self) -> list[str]:
        """Move waiting requests to running while there is room.

        Requests are served in arrival order (first come, first served).
        """
        while self.waiting and len(self.running) < self.max_num_seqs:
            self.running.append(self.waiting.pop(0))
        return list(self.running)
