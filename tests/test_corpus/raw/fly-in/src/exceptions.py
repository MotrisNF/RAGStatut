"""Custom exception hierarchy for configuration and file errors."""


class ConfigError(Exception):
    """Base exception for any invalid simulation configuration."""

    def __init__(self, msg: str = "Default ConfigError."):
        """Store the error message.

        Args:
            msg: Human-readable description of what went wrong.
        """
        super().__init__(msg)


class FileError(ConfigError):
    """Raised when the map file is missing, unreadable, or malformed."""

    def __init__(self, msg: str = "Default FileError.") -> None:
        """Store the error message.

        Args:
            msg: Human-readable description of what went wrong.
        """
        super().__init__(msg)


class PathError(ConfigError):
    """Raised when no valid drone route exists between the gates."""

    def __init__(self, msg: str = "Default PathError.") -> None:
        """Store the error message.

        Args:
            msg: Human-readable description of what went wrong.
        """
        super().__init__(msg)
