"""Parser for the drone-simulation map file format.

Reads the ``nb_drones:``, ``start_hub:``, ``end_hub:``, ``hub:`` and
``connection:`` directives described in the subject, validates them
with pydantic models, and builds a :class:`MapConfig`.
"""

from typing import Literal, Optional, TextIO

from pydantic import ValidationError

from exceptions import FileError
from map_config import MapConfig, Hub, Connection, Gate, Gates
from constants import ZoneType, _HUB_METADATA_KEYS, _CONNECTION_METADATA_KEYS


class Parser:
    """Reads a map file and turns it into a validated MapConfig."""

    def __init__(self) -> None:
        """Create a parser with no map loaded yet."""
        self.config: Optional[MapConfig] = None

    def read_file(self, file: TextIO) -> MapConfig:
        """Parse an open map file into a MapConfig.

        Args:
            file: An open, readable text handle to the map file.

        Returns:
            The validated MapConfig, also stored in ``self.config``.

        Raises:
            FileError: If any line violates the expected map syntax
                (see VII.4 of the subject), naming the offending line
                and the cause.
        """
        nb_drones: Optional[int] = None
        hubs: dict[str, Hub] = {}
        connections: list[Connection] = []
        start_name: Optional[str] = None
        end_name: Optional[str] = None
        seen_connections: set[frozenset[str]] = set()
        seen_coordinates: dict[tuple[int, int], str] = {}

        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue

            if nb_drones is None:
                nb_drones = self._parse_nb_drones(line, line_number)
                continue

            if line.startswith("start_hub:"):
                if start_name is not None:
                    raise FileError(
                        f"Line {line_number}: duplicate 'start_hub' "
                        "definition."
                    )
                start_name = self._parse_hub(
                    line[len("start_hub:"):],
                    "start",
                    hubs,
                    seen_coordinates,
                    line_number,
                )
            elif line.startswith("end_hub:"):
                if end_name is not None:
                    raise FileError(
                        f"Line {line_number}: duplicate 'end_hub' "
                        "definition."
                    )
                end_name = self._parse_hub(
                    line[len("end_hub:"):],
                    "end",
                    hubs,
                    seen_coordinates,
                    line_number,
                )
            elif line.startswith("hub:"):
                self._parse_hub(
                    line[len("hub:"):],
                    None,
                    hubs,
                    seen_coordinates,
                    line_number,
                )
            elif line.startswith("connection:"):
                self._parse_connection(
                    line[len("connection:"):],
                    hubs,
                    connections,
                    seen_connections,
                    line_number,
                )
            else:
                raise FileError(
                    f"Line {line_number}: unrecognized directive."
                )

        if nb_drones is None:
            raise FileError("The file dont have any key")
        if start_name is None:
            raise FileError(
                "The file is missing a 'start_hub' definition."
            )
        if end_name is None:
            raise FileError(
                "The file is missing an 'end_hub' definition."
            )

        start_hub = hubs[start_name]
        end_hub = hubs[end_name]

        xs = [x for x, _y in seen_coordinates]
        ys = [y for _x, y in seen_coordinates]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        try:
            self.config = MapConfig(
                nb_drones=nb_drones,
                gates=Gates(
                    entry=Gate(
                        x=start_hub.x,
                        y=start_hub.y,
                        color=start_hub.color,
                        name=start_name,
                    ),
                    exit=Gate(
                        x=end_hub.x,
                        y=end_hub.y,
                        color=end_hub.color,
                        name=end_name,
                    ),
                ),
                hubs=hubs,
                connections=connections,
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
                width=max_x - min_x,
                height=max_y - min_y,
            )
        except ValidationError as e:
            raise FileError(str(e)) from e

        return self.config

    @staticmethod
    def _parse_nb_drones(line: str, line_number: int) -> int:
        """Parse the mandatory first ``nb_drones: <n>`` line.

        Args:
            line: The full line (comment already stripped).
            line_number: 1-based line number, for error messages.

        Returns:
            The parsed, strictly positive number of drones.

        Raises:
            FileError: If the line isn't a valid ``nb_drones:``
                directive, or its value isn't a positive integer.
        """
        if not line.startswith("nb_drones:"):
            raise FileError(
                f"Line {line_number}: expected "
                "'nb_drones: <positive_integer>' as the first line."
            )
        value_raw = line[len("nb_drones:"):].strip()
        return Parser._parse_positive_int(value_raw, "nb_drones", line_number)

    def _parse_hub(
        self,
        remainder: str,
        forced_zone: Optional[Literal["start", "end"]],
        hubs: dict[str, Hub],
        seen_coordinates: dict[tuple[int, int], str],
        line_number: int,
    ) -> str:
        """Parse a ``hub:``/``start_hub:``/``end_hub:`` line body.

        Args:
            remainder: The line with its directive prefix stripped.
            forced_zone: ``"start"``/``"end"`` for a gate line, or
                ``None`` for a regular ``hub:`` line.
            hubs: Zones parsed so far; the new hub is added to it.
            seen_coordinates: ``(x, y)`` positions already used by an
                earlier zone, mapped to that zone's name. Not
                required by the subject, but two zones sharing a
                position doesn't make physical sense, so it's
                rejected too.
            line_number: 1-based line number, for error messages.

        Returns:
            The name of the newly parsed zone.

        Raises:
            FileError: If the line is malformed, its metadata is
                invalid, or its coordinates are already used by
                another zone.
        """
        body, meta = self._split_metadata_block(remainder, line_number)

        tokens = body.split()
        if len(tokens) != 3:
            raise FileError(
                f"Line {line_number}: malformed hub definition, "
                "expected '<name> <x> <y> [metadata]'."
            )
        name, x_raw, y_raw = tokens
        if "-" in name:
            raise FileError(
                f"Line {line_number}: zone name '{name}' cannot "
                "contain a hyphen."
            )
        if name in hubs:
            raise FileError(
                f"Line {line_number}: duplicate zone name '{name}'."
            )
        x = self._parse_int(x_raw, "x", line_number)
        y = self._parse_int(y_raw, "y", line_number)
        if (x, y) in seen_coordinates:
            raise FileError(
                f"Line {line_number}: zone '{name}' has the same "
                f"coordinates ({x}, {y}) as zone "
                f"'{seen_coordinates[(x, y)]}'."
            )

        metadata = self._parse_metadata(meta, _HUB_METADATA_KEYS, line_number)
        zone = self._parse_zone_type(
            metadata.get("zone", "normal"), line_number
        )

        max_drones: Optional[int] = None
        if forced_zone is None:
            max_drones = 1
            if "max_drones" in metadata:
                max_drones = self._parse_positive_int(
                    metadata["max_drones"], "max_drones", line_number
                )

        try:
            hub = Hub(
                x=x,
                y=y,
                zone=forced_zone if forced_zone is not None else zone,
                max_drones=max_drones,
                color=metadata.get("color"),
            )
        except ValidationError as e:
            raise FileError(f"Line {line_number}: {e}") from e

        hubs[name] = hub
        seen_coordinates[(x, y)] = name
        return name

    def _parse_connection(
        self,
        remainder: str,
        hubs: dict[str, Hub],
        connections: list[Connection],
        seen_connections: set[frozenset[str]],
        line_number: int,
    ) -> None:
        """Parse a ``connection:`` line body and record the link.

        Args:
            remainder: The line with the ``connection:`` prefix
                stripped.
            hubs: Zones parsed so far; both endpoints must already
                be present.
            connections: Connections parsed so far; the new one is
                appended to it.
            seen_connections: Endpoint pairs already seen, used to
                reject ``a-b``/``b-a`` duplicates.
            line_number: 1-based line number, for error messages.

        Raises:
            FileError: If the line is malformed, references an
                undefined zone, links a zone to itself, duplicates
                an existing connection, or has invalid metadata.
        """
        body, meta = self._split_metadata_block(remainder, line_number)

        pos1, separator, pos2 = body.partition("-")
        pos1, pos2 = pos1.strip(), pos2.strip()
        if not separator or not pos1 or not pos2 or "-" in pos2:
            raise FileError(
                f"Line {line_number}: malformed connection definition, "
                "expected '<zone1>-<zone2> [metadata]'."
            )

        for name in (pos1, pos2):
            if name not in hubs:
                raise FileError(
                    f"Line {line_number}: connection references "
                    f"undefined zone '{name}'."
                )
        if pos1 == pos2:
            raise FileError(
                f"Line {line_number}: a connection cannot link a zone "
                "to itself."
            )

        link = frozenset((pos1, pos2))
        if link in seen_connections:
            raise FileError(
                f"Line {line_number}: duplicate connection between "
                f"'{pos1}' and '{pos2}'."
            )
        seen_connections.add(link)

        metadata = self._parse_metadata(
            meta, _CONNECTION_METADATA_KEYS, line_number
        )
        capacity = 1
        if "max_link_capacity" in metadata:
            capacity = self._parse_positive_int(
                metadata["max_link_capacity"],
                "max_link_capacity",
                line_number,
            )

        try:
            connections.append(
                Connection(pos1=pos1, pos2=pos2, capacity=capacity)
            )
        except ValidationError as e:
            raise FileError(f"Line {line_number}: {e}") from e

    @staticmethod
    def _split_metadata_block(
        remainder: str, line_number: int
    ) -> tuple[str, Optional[str]]:
        """Split ``'<body> [key=value ...]'`` into body and metadata.

        Args:
            remainder: The directive line with its prefix stripped.
            line_number: 1-based line number, for error messages.

        Returns:
            A ``(body, meta)`` pair. ``meta`` is ``None`` when the
            line has no ``[...]`` block, otherwise the raw text found
            between the brackets.

        Raises:
            FileError: If a ``[`` is present without a matching
                closing ``]``, or the metadata contains stray
                brackets.
        """
        remainder = remainder.strip()
        if "[" not in remainder:
            return remainder, None

        body, _, bracket_part = remainder.partition("[")
        if not bracket_part.endswith("]"):
            raise FileError(
                f"Line {line_number}: metadata block is missing its "
                "closing ']'."
            )

        meta = bracket_part[:-1]
        if "[" in meta or "]" in meta:
            raise FileError(
                f"Line {line_number}: malformed metadata block."
            )
        return body.strip(), meta

    @staticmethod
    def _parse_metadata(
        meta: Optional[str],
        allowed_keys: set[str],
        line_number: int,
    ) -> dict[str, str]:
        """Parse a ``key=value ...`` metadata block into a dict.

        Args:
            meta: The raw text between ``[`` and ``]``, or ``None``
                if the line had no metadata block.
            allowed_keys: Metadata keys accepted for this directive.
            line_number: 1-based line number, for error messages.

        Returns:
            The parsed ``key: value`` pairs, empty if ``meta`` was
            ``None`` or empty.

        Raises:
            FileError: If a token has no ``=``, uses an unknown key,
                or a key is repeated.
        """
        result: dict[str, str] = {}
        if not meta:
            return result
        for token in meta.split():
            if "=" not in token:
                raise FileError(
                    f"Line {line_number}: malformed metadata token "
                    f"'{token}'."
                )
            key, _, value = token.partition("=")
            if key not in allowed_keys:
                raise FileError(
                    f"Line {line_number}: unknown metadata key '{key}'."
                )
            if key in result:
                raise FileError(
                    f"Line {line_number}: duplicate metadata key "
                    f"'{key}'."
                )
            result[key] = value
        return result

    @staticmethod
    def _parse_zone_type(value: str, line_number: int) -> ZoneType:
        """Validate a ``zone=`` metadata value against the four types.

        Written as an explicit chain (instead of a lookup + cast) so
        that each branch narrows straight to a `ZoneType` literal.

        Args:
            value: The raw ``zone=`` value from the metadata.
            line_number: 1-based line number, for error messages.

        Returns:
            The matching ``ZoneType`` literal.

        Raises:
            FileError: If ``value`` isn't one of the four zone types.
        """
        if value == "normal":
            return "normal"
        if value == "blocked":
            return "blocked"
        if value == "restricted":
            return "restricted"
        if value == "priority":
            return "priority"
        raise FileError(
            f"Line {line_number}: invalid zone type '{value}'."
        )

    @staticmethod
    def _parse_int(raw: str, field: str, line_number: int) -> int:
        """Parse a token as an integer, raising a clear FileError.

        Args:
            raw: The raw string value to parse.
            field: Name of the field being parsed, for error
                messages.
            line_number: 1-based line number, for error messages.

        Returns:
            The parsed integer.

        Raises:
            FileError: If ``raw`` isn't a valid integer literal.
        """
        try:
            return int(raw)
        except ValueError:
            raise FileError(
                f"Line {line_number}: '{field}' must be an integer, "
                f"got '{raw}'."
            ) from None

    @staticmethod
    def _parse_positive_int(raw: str, field: str, line_number: int) -> int:
        """Parse a token as a strictly positive integer.

        Args:
            raw: The raw string value to parse.
            field: Name of the field being parsed, for error
                messages.
            line_number: 1-based line number, for error messages.

        Returns:
            The parsed positive integer.

        Raises:
            FileError: If ``raw`` isn't an integer, or isn't > 0.
        """
        value = Parser._parse_int(raw, field, line_number)
        if value <= 0:
            raise FileError(
                f"Line {line_number}: '{field}' must be a positive "
                f"integer, got '{raw}'."
            )
        return value
