"""Translation of raw key presses into a valid player move direction."""

DIRECTIONS: dict[str, tuple[int, int, int]] = {
    'w': (0, -1, 1),
    'd': (1, 0, 2),
    's': (0, 1, 4),
    'a': (-1, 0, 8),
}

OPPOSITE_KEY: dict[str, str] = {'w': 's', 's': 'w', 'a': 'd', 'd': 'a'}


def decide_move(cell: int, heading: str | None, key: str) -> str | None:
    """Decide which direction the player should move towards, if any.

    Moves the player in the key's direction if that direction is open.
    Otherwise, auto-advances through a corridor: if exactly one open
    direction (besides the one behind the player) exists, moves that
    way instead, so the player doesn't get stuck pressing into a wall
    at a bend.

    Args:
        cell (int): The current cell's value, bit-encoded per wall
            (see 'DIRECTIONS').
        heading (str | None): The direction key the player was last
            moving in, or None if it hasn't moved yet.
        key (str): The key the player just pressed.

    Returns:
        str | None: The direction key to move towards ('w', 'a', 's'
            or 'd'), or None if the player shouldn't move.
    """
    open_dirs = [
        d for d, (_, _, bit) in DIRECTIONS.items()
        if cell & bit == 0
    ]
    opposite = OPPOSITE_KEY[heading] if heading else None
    forward_dirs = [d for d in open_dirs if d != opposite]
    if key in DIRECTIONS and key in open_dirs:
        return key
    if len(forward_dirs) == 1:
        return forward_dirs[0]
    return None
