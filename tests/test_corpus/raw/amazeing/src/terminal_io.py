"""Low-level terminal I/O: raw-mode key reads and cooked-mode prompts."""

from contextlib import contextmanager
import select
import sys
import termios
from typing import Generator

REFRESH_SECONDS: float = 0.1


@contextmanager
def raw_mode() -> Generator[None, None, None]:
    """Put stdin into raw, unbuffered, non-echoing mode for the block.

    Disables canonical mode and echo so single key presses can be read
    without waiting for Enter, and restores the previous terminal
    settings on exit even if the block raises.

    Yields:
        None
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    new_settings = termios.tcgetattr(fd)
    new_settings[3] &= ~(termios.ICANON | termios.ECHO)
    new_settings[6][termios.VMIN] = 1
    new_settings[6][termios.VTIME] = 0
    try:
        termios.tcsetattr(fd, termios.TCSANOW, new_settings)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def get_key(timeout: float = REFRESH_SECONDS) -> str:
    """Read a single key from stdin without blocking past 'timeout'.

    Requires 'raw_mode' to be active to read one key at a time.

    Args:
        timeout (float): Seconds to wait for a key press.
            'REFRESH_SECONDS' by default.

    Returns:
        str: The character read, or "" if 'timeout' elapsed with no
            key pressed.
    """
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if ready:
        return sys.stdin.read(1)
    return ""


def cooked_input(prompt: str = "") -> str:
    """Read a line from stdin, temporarily leaving raw mode.

    Restores canonical mode and echo so the user can see what they
    type and edit it with backspace, then restores raw mode
    afterward, so it can be called from within a 'raw_mode' block.

    Args:
        prompt (str): The prompt to display before reading. Empty by
            default.

    Returns:
        str: The line entered by the user.
    """
    fd = sys.stdin.fileno()
    raw_settings = termios.tcgetattr(fd)
    cooked_settings = termios.tcgetattr(fd)
    cooked_settings[3] |= (termios.ICANON | termios.ECHO)
    termios.tcsetattr(fd, termios.TCSADRAIN, cooked_settings)
    try:
        return input(prompt)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, raw_settings)
