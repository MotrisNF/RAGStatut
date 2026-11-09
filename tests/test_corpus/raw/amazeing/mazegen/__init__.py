"""Package exposing the public maze generation and solving API."""

from .mazegen import MazeGen
from .exceptions import MazeGenError

__all__ = [
    "MazeGen",
    "MazeGenError",
]
