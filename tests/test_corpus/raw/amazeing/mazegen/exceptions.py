"""Module of the custom exception used across the maze generator package.

This module defines MazeGenError, the exception raised by MazeGen and its
dependencies instead of letting generic exceptions (ValueError, IndexError,
OSError...) propagate without context.
"""


class MazeGenError(Exception):
    """Custom exception for errors raised by the maze generator package.

    Raised instead of generic exceptions so that any code using MazeGen
    can catch a single, well-defined exception type with a clear message,
    e.g. invalid constructor parameters, or calling 'solve' before
    'generate'.
    """
    def __init__(self, msg: str = "Uknown maze error"):
        """Initialize the exception with a message.

        Args:
            msg (str): Description of the error. Defaults to
                "Uknown maze error".
        """
        super().__init__(msg)
