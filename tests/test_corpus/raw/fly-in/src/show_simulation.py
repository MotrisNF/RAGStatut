"""Pygame-based graphical playback of a finished drone simulation.

Draws the static map once, then animates the recorded per-turn drone
positions frame by frame, with a zoom/pan camera and sidebar playback
controls (play/pause, speed, reset) alongside a small narrating
mascot.
"""

from initiate_simulation import Initiator
from exceptions import PathError
from constants import (
    COLORS, ZONE_COLORS, DESKTOP_OFFSET, MARGIN_X, MARGIN_Y, MAX_GRID,
    SIDEBAR_RATIO, TURN_DURATION_FRAMES, TURN_PAUSE_FRAMES,
    START_DELAY_FRAMES, FRAME_RATE, NODE_SPRITE_SCALE, DRONE_SPRITE_SCALE,
    MOON_SIZE_DIVISOR, CONNECTION_LINE_WIDTH, CONNECTION_LINE_INNER_WIDTH,
    BADGE_BORDER_WIDTH, CAPACITY_BADGE_RADIUS_FLOOR, CAPACITY_BADGE_FONT_FLOOR,
    CAPACITY_BADGE_GRID_DIVISOR, CAPACITY_BADGE_SIZE_CAP_DIVISOR,
    CAPACITY_BADGE_RADIUS_SIZE_CAP_MIN, CAPACITY_BADGE_FONT_SIZE_CAP_MIN,
    DRONE_BADGE_RADIUS_FLOOR, DRONE_BADGE_FONT_FLOOR, DRONE_BADGE_GRID_DIVISOR,
    DRONE_BADGE_SIZE_CAP_DIVISOR, DRONE_BADGE_RADIUS_SIZE_CAP_MIN,
    DRONE_BADGE_FONT_SIZE_CAP_MIN, TURN_LABEL_FONT_FLOOR,
    TURN_LABEL_SIDEBAR_DIVISOR, TURN_LABEL_MARGIN_TOP,
    TURN_LABEL_MARGIN_SIDE, SIMULATION_DEFAULT_SPEED, SIMULATION_MIN_SPEED,
    SIMULATION_MAX_SPEED, SIMULATION_SPEED_STEP, COMMANDER_SPRITE_SCALE,
    COMMANDER_MARGIN_TOP, CONTROL_BUTTON_SCALE, CONTROL_BUTTON_SPACING,
    CONTROL_PANEL_MARGIN_BOTTOM, CONTROL_BUTTON_GROUP_GAP,
    SPEED_BAR_MARGIN_TOP, SPEED_BAR_HEIGHT, SPEED_BAR_BORDER_WIDTH,
    SPEED_BAR_TICK_WIDTH, SPEED_BAR_INDICATOR_WIDTH, MAP_ZOOM_DEFAULT,
    MAP_ZOOM_MIN, MAP_ZOOM_MAX, MAP_ZOOM_STEP_FACTOR, COMMANDER_FRAME_PADDING,
    COMMANDER_FRAME_BORDER_WIDTH, DIALOG_BOX_MARGIN_SIDE,
    DIALOG_BOX_MARGIN_TOP, DIALOG_BOX_MARGIN_BOTTOM, DIALOG_BOX_PADDING,
    DIALOG_BOX_BORDER_WIDTH, DIALOG_FONT_SIDEBAR_DIVISOR, DIALOG_FONT_FLOOR,
    DIALOG_LINE_SPACING, DIALOG_TYPE_FRAMES_PER_CHAR, DIALOG_HOLD_FRAMES,
    MANDIBLE_TOGGLE_FRAMES, SIMULATION_DIALOG_PHRASES
)

from collections import deque

import math
import os
import random

# Must be set before ``import pygame`` to suppress the "Hello from the
# pygame community" support-prompt banner it otherwise prints on import.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402


class Pyshow():
    """Renders the map and animates the simulated drone fleet."""

    def __init__(self, initiator: Initiator) -> None:
        """Store the finished simulation to render.

        Args:
            initiator: A simulation that has already been run.

        Raises:
            PathError: If the config isn't loaded.
        """
        if initiator.config is None:
            raise PathError("The config is not loaded.")
        self._initiator = initiator
        self._c = initiator.config
        self._dimentions: tuple[int, int] = (0, 0)
        self._running: bool = True
        self._clock = pygame.time.Clock()
        self._ticks: float = FRAME_RATE
        self._grid: int = 0
        self._spacing_x: float = 0.0
        self._spacing_y: float = 0.0
        self._offset_x: int = 0
        self._offset_y: int = 0
        self._centers: dict[str, tuple[int, int]] = {}
        self._frame_count: float = 0.0
        self._paused: bool = True
        self._speed: float = SIMULATION_DEFAULT_SPEED
        self._zoom: float = MAP_ZOOM_DEFAULT
        self._camera_x: float = 0.0
        self._camera_y: float = 0.0

    def _to_screen(self, x: int, y: int) -> tuple[int, int]:
        """Convert a hub's map coordinates to a screen pixel position."""
        col = x - self._c.min_x
        row = y - self._c.min_y
        return (
            self._offset_x + int(col * self._spacing_x),
            self._offset_y + int(row * self._spacing_y),
        )

    def _to_view(
        self, x: int, y: int, pad_x: int = 0, pad_y: int = 0
    ) -> tuple[int, int]:
        """Map an unzoomed map-area screen position to the current view.

        Applies the current zoom level and camera pan, so drones and
        other overlays drawn fresh each frame line up with the
        (pre-rendered, then zoomed) map background.

        Args:
            pad_x: Extra horizontal offset -- the letterbox margin
                added when the map is zoomed out past its default
                size and centered rather than panned (see
                ``_clamp_camera``).
            pad_y: Same, vertically.
        """
        return (
            int((x - self._camera_x) * self._zoom) + pad_x,
            int((y - self._camera_y) * self._zoom) + pad_y,
        )

    def _clamp_camera(self, map_width: int, map_height: int) -> None:
        """Keep the camera within the range the current zoom allows.

        Once the view is at least as large as the map (zoomed out to
        or past the default), there's nothing to pan to, so the
        camera is pinned to (0, 0) -- centering the shrunk map is
        handled separately, at render time.
        """
        view_width = map_width / self._zoom
        view_height = map_height / self._zoom
        max_camera_x = max(0.0, map_width - view_width)
        max_camera_y = max(0.0, map_height - view_height)
        self._camera_x = min(max(0.0, self._camera_x), max_camera_x)
        self._camera_y = min(max(0.0, self._camera_y), max_camera_y)

    def _pan_camera(
        self, dx: int, dy: int, map_width: int, map_height: int
    ) -> None:
        """Pan the camera by a screen-pixel drag delta.

        Args:
            dx: Horizontal mouse movement, in screen pixels.
            dy: Same, vertically.
            map_width: Width of the map area (excludes the sidebar).
            map_height: Height of the map area.
        """
        self._camera_x -= dx / self._zoom
        self._camera_y -= dy / self._zoom
        self._clamp_camera(map_width, map_height)

    def _apply_zoom(
        self, notches: int, mouse_pos: tuple[int, int],
        map_width: int, map_height: int
    ) -> None:
        """Zoom the map view in or out, cursor position held fixed.

        Args:
            notches: Mouse wheel ``event.y`` -- positive zooms in,
                negative zooms out.
            mouse_pos: Cursor position, in map-area screen pixels.
            map_width: Width of the map area (excludes the sidebar).
            map_height: Height of the map area.
        """
        old_zoom = self._zoom
        new_zoom = old_zoom * (MAP_ZOOM_STEP_FACTOR ** notches)
        new_zoom = max(MAP_ZOOM_MIN, min(new_zoom, MAP_ZOOM_MAX))
        if new_zoom == old_zoom:
            return

        mouse_x, mouse_y = mouse_pos
        map_x = self._camera_x + mouse_x / old_zoom
        map_y = self._camera_y + mouse_y / old_zoom

        self._zoom = new_zoom
        self._camera_x = map_x - mouse_x / new_zoom
        self._camera_y = map_y - mouse_y / new_zoom
        self._clamp_camera(map_width, map_height)

    @staticmethod
    def _wrap_text(
        text: str, font: pygame.font.Font, max_width: int
    ) -> list[str]:
        """Word-wrap ``text`` into lines no wider than ``max_width``."""
        lines: list[str] = []
        current = ""
        for word in text.split(" "):
            candidate = f"{current} {word}" if current else word
            if not current or font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def _interpolate(
        self, waypoints: list[tuple[int, int, int]]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Find a drone's current screen position and facing direction.

        Locates which pair of waypoints brackets the current frame
        and linearly interpolates between them, so a move animates
        smoothly across the turns it takes to complete.

        Args:
            waypoints: A drone's ``(turn, x, y)`` rest positions, in
                order.

        Returns:
            The interpolated screen position, and the direction of
            travel for that segment.
        """
        for i in range(len(waypoints) - 1):
            turn0, x0, y0 = waypoints[i]
            turn1, x1, y1 = waypoints[i + 1]
            turns_spanned = turn1 - turn0
            frame0 = (turn0 + 1) * TURN_DURATION_FRAMES
            frame1 = (turn1 + 1) * TURN_DURATION_FRAMES
            move_frames = turns_spanned * (
                TURN_DURATION_FRAMES - TURN_PAUSE_FRAMES
            )
            if self._frame_count <= frame1 or i == len(waypoints) - 2:
                t = (self._frame_count - frame0) / move_frames
                t = max(0.0, min(1.0, t))
                position = (
                    int(x0 + (x1 - x0) * t),
                    int(y0 + (y1 - y0) * t),
                )
                direction = (x1 - x0, y1 - y0)
                if direction == (0, 0):
                    direction = self._last_direction(waypoints, i)
                return position, direction
        last = (waypoints[-1][1], waypoints[-1][2])
        return last, (0, -1)

    def _last_direction(
        self, waypoints: list[tuple[int, int, int]], i: int
    ) -> tuple[int, int]:
        """Find the last real movement direction up to waypoint ``i``.

        Used while a drone is stationary (arrived or waiting) so its
        sprite keeps facing the way it was last moving, instead of
        resetting to a default orientation.
        """
        for k in range(i, -1, -1):
            x0, y0 = waypoints[k][1], waypoints[k][2]
            x1, y1 = waypoints[k + 1][1], waypoints[k + 1][2]
            if (x1 - x0, y1 - y0) != (0, 0):
                return (x1 - x0, y1 - y0)
        return (0, -1)

    def _distances_from_entry(self) -> dict[str, int]:
        """Hop-distance from the entry hub to every other hub.

        Used to tell, for a given connection, which endpoint is
        "downstream" -- the one a drone would be advancing towards
        when moving away from the entry -- so its zone (not the
        upstream one) decides the connection badge's color.
        """
        adjacency: dict[str, list[str]] = {name: [] for name in self._c.hubs}
        for connection in self._c.connections:
            adjacency[connection.pos1].append(connection.pos2)
            adjacency[connection.pos2].append(connection.pos1)

        start_name = self._c.gates.entry.name
        distances = {start_name: 0}
        queue = deque([start_name])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def start(self) -> None:
        """Open the window, draw the static map, and animate the fleet."""
        pygame.init()

        desktop = pygame.display.Info()
        self._dimentions = (
            desktop.current_w - DESKTOP_OFFSET,
            desktop.current_h - DESKTOP_OFFSET
            )

        screen = pygame.display.set_mode(self._dimentions)
        back_ground = pygame.Surface(self._dimentions)
        back_ground.fill(COLORS["DARK_BLUE"])

        sidebar_width = int(self._dimentions[0] * SIDEBAR_RATIO)
        map_width = self._dimentions[0] - sidebar_width
        map_height = self._dimentions[1]

        star_tile = pygame.image.load("assets/star_tile.png").convert()
        tile_w, tile_h = star_tile.get_size()
        for tile_x in range(0, map_width, tile_w):
            for tile_y in range(0, map_height, tile_h):
                back_ground.blit(star_tile, (tile_x, tile_y))

        # A separate, larger starfield -- big enough to cover the
        # margins revealed once zoomed out past MAP_ZOOM_MIN -- so
        # those margins show more stars instead of a flat fill.
        star_field_width = int(map_width / MAP_ZOOM_MIN)
        star_field_height = int(map_height / MAP_ZOOM_MIN)
        star_field = pygame.Surface((star_field_width, star_field_height))
        star_field.fill(COLORS["DARK_BLUE"])
        for tile_x in range(0, star_field_width, tile_w):
            for tile_y in range(0, star_field_height, tile_h):
                star_field.blit(star_tile, (tile_x, tile_y))

        moon_sprite = pygame.image.load("assets/moon.png").convert_alpha()
        moon_size = min(map_width, map_height) // MOON_SIZE_DIVISOR
        moon_scaled = pygame.transform.scale(
            moon_sprite, (moon_size, moon_size)
        )
        moon_rect = moon_scaled.get_rect(
            center=(map_width // 2, map_height // 2)
        )
        back_ground.blit(moon_scaled, moon_rect)

        sidebar_rect = pygame.Rect(map_width, 0, sidebar_width, map_height)
        pygame.draw.rect(back_ground, COLORS["BLACK"], sidebar_rect)
        pygame.draw.rect(back_ground, COLORS["BROWN"], sidebar_rect, width=24)

        usable_width = map_width - MARGIN_X * 2
        usable_height = map_height - MARGIN_Y * 2

        cols = self._c.width + 1
        rows = self._c.height + 1

        grid_candidates: list[float] = [MAX_GRID]
        if cols > 1:
            self._spacing_x = usable_width / (cols - 1)
            self._offset_x = MARGIN_X
            grid_candidates.append(self._spacing_x)
        else:
            self._spacing_x = 0.0
            self._offset_x = MARGIN_X + usable_width // 2

        if rows > 1:
            self._spacing_y = usable_height / (rows - 1)
            self._offset_y = MARGIN_Y
            grid_candidates.append(self._spacing_y)
        else:
            self._spacing_y = 0.0
            self._offset_y = MARGIN_Y + usable_height // 2

        self._grid = int(min(grid_candidates))

        node_sprite = pygame.image.load("assets/node.png").convert_alpha()
        size = int(self._grid * NODE_SPRITE_SCALE)
        scaled_node_sprite = pygame.transform.scale(node_sprite, (size, size))

        for name, hub in self._c.hubs.items():
            self._centers[name] = self._to_screen(hub.x, hub.y)

        capacity_radius = min(
            max(
                CAPACITY_BADGE_RADIUS_FLOOR,
                self._grid // CAPACITY_BADGE_GRID_DIVISOR
            ),
            max(
                CAPACITY_BADGE_RADIUS_SIZE_CAP_MIN,
                size // CAPACITY_BADGE_SIZE_CAP_DIVISOR
            )
        )
        capacity_font_px = min(
            max(
                CAPACITY_BADGE_FONT_FLOOR,
                self._grid // CAPACITY_BADGE_GRID_DIVISOR
            ),
            max(
                CAPACITY_BADGE_FONT_SIZE_CAP_MIN,
                size // CAPACITY_BADGE_SIZE_CAP_DIVISOR
            )
        )
        capacity_font = pygame.font.Font(
            "assets/PressStart2P-Regular.ttf",
            capacity_font_px)
        entry_distances = self._distances_from_entry()
        restricted_badges: list[
            tuple[tuple[int, int], int, tuple[int, int, int]]
        ] = []
        other_badges: list[
            tuple[tuple[int, int], int, tuple[int, int, int]]
        ] = []
        for connection in self._c.connections:
            pos1 = self._centers[connection.pos1]
            pos2 = self._centers[connection.pos2]
            pygame.draw.line(
                back_ground, COLORS["BLACK"], pos1, pos2,
                CONNECTION_LINE_WIDTH
            )
            pygame.draw.line(
                back_ground, COLORS["WHITE"], pos1, pos2,
                CONNECTION_LINE_INNER_WIDTH
            )

            midpoint = (
                (pos1[0] + pos2[0]) // 2,
                (pos1[1] + pos2[1]) // 2,
            )

            downstream_name = connection.pos2
            if (
                entry_distances.get(connection.pos1, 0)
                > entry_distances.get(connection.pos2, 0)
            ):
                downstream_name = connection.pos1
            downstream_zone = self._c.hubs[downstream_name].zone

            if downstream_zone == "restricted":
                badge_fill = ZONE_COLORS["restricted"]
                restricted_badges.append(
                    (midpoint, connection.capacity, badge_fill)
                )
            else:
                badge_fill = (
                    ZONE_COLORS["priority"] if downstream_zone == "priority"
                    else COLORS["WHITE"]
                )
                other_badges.append(
                    (midpoint, connection.capacity, badge_fill)
                )

        for midpoint, capacity, badge_fill in restricted_badges + other_badges:
            pygame.draw.circle(
                back_ground, badge_fill, midpoint, capacity_radius
            )
            pygame.draw.circle(
                back_ground, COLORS["BLACK"], midpoint, capacity_radius,
                BADGE_BORDER_WIDTH
            )

            capacity_label = capacity_font.render(
                str(capacity), False, COLORS["BLACK"]
            )
            capacity_rect = capacity_label.get_rect(center=midpoint)
            back_ground.blit(capacity_label, capacity_rect)

        for name, hub in self._c.hubs.items():
            center = self._centers[name]
            color = pygame.Color(ZONE_COLORS[hub.zone])
            if hub.color:
                try:
                    color = pygame.Color(hub.color)
                except ValueError:
                    pass

            tinted_node_sprite = scaled_node_sprite.copy()
            tinted_node_sprite.fill(
                color, special_flags=pygame.BLEND_RGBA_MULT
            )
            node_rect = tinted_node_sprite.get_rect(center=center)
            back_ground.blit(tinted_node_sprite, node_rect)

            if hub.zone not in ("start", "end"):
                badge_center = (
                    node_rect.right - capacity_radius,
                    node_rect.bottom - capacity_radius,
                )
                pygame.draw.circle(
                    back_ground, COLORS["WHITE"], badge_center,
                    capacity_radius
                )
                pygame.draw.circle(
                    back_ground, COLORS["BLACK"], badge_center,
                    capacity_radius, BADGE_BORDER_WIDTH
                )
                capacity_label = capacity_font.render(
                    str(hub.max_drones), False, COLORS["BLACK"]
                )
                capacity_rect = capacity_label.get_rect(center=badge_center)
                back_ground.blit(capacity_label, capacity_rect)

        drone_sprite = pygame.image.load("assets/drone.png").convert_alpha()
        drone_size = int(self._grid * DRONE_SPRITE_SCALE)
        scaled_drone_sprite = pygame.transform.scale(
            drone_sprite, (drone_size, drone_size)
        )
        drone_badge_radius = min(
            max(
                DRONE_BADGE_RADIUS_FLOOR,
                self._grid // DRONE_BADGE_GRID_DIVISOR
            ),
            max(
                DRONE_BADGE_RADIUS_SIZE_CAP_MIN,
                drone_size // DRONE_BADGE_SIZE_CAP_DIVISOR
            )
        )
        drone_badge_font_px = min(
            max(
                DRONE_BADGE_FONT_FLOOR,
                self._grid // DRONE_BADGE_GRID_DIVISOR
            ),
            max(
                DRONE_BADGE_FONT_SIZE_CAP_MIN,
                drone_size // DRONE_BADGE_SIZE_CAP_DIVISOR
            )
        )
        drone_badge_font = pygame.font.Font(
            "assets/PressStart2P-Regular.ttf",
            drone_badge_font_px)

        entry = self._c.gates.entry
        turn_positions = self._initiator.turn_positions
        total_turns = len(turn_positions)

        entry_x, entry_y = self._to_screen(entry.x, entry.y)
        drone_waypoints: dict[int, list[tuple[int, int, int]]] = {
            drone_id: [(-1, entry_x, entry_y)]
            for drone_id in range(1, self._c.nb_drones + 1)
        }
        for turn_index, positions in enumerate(turn_positions):
            for position in positions:
                if position.in_transit:
                    continue
                sx, sy = self._to_screen(position.x, position.y)
                drone_waypoints[position.drone_id].append(
                    (turn_index, sx, sy)
                )

        turn_font = pygame.font.Font(
            "assets/PressStart2P-Regular.ttf",
            max(
                TURN_LABEL_FONT_FLOOR,
                sidebar_width // TURN_LABEL_SIDEBAR_DIVISOR
            )
        )

        sidebar_center_x = map_width + sidebar_width // 2

        commander_size = int(sidebar_width * COMMANDER_SPRITE_SCALE)
        commander_open_sprite = pygame.transform.scale(
            pygame.image.load("assets/comander.png").convert_alpha(),
            (commander_size, commander_size)
        )
        commander_closed_sprite = pygame.transform.scale(
            pygame.image.load("assets/comander_closed.png").convert_alpha(),
            (commander_size, commander_size)
        )
        commander_rect = commander_open_sprite.get_rect(
            midtop=(
                sidebar_center_x,
                TURN_LABEL_MARGIN_TOP + turn_font.get_height()
                + COMMANDER_MARGIN_TOP
            )
        )
        # The portrait alternates open/closed jaws each frame while the
        # commander "talks", so it's drawn fresh in the main loop
        # rather than baked in here -- only its frame is static.
        commander_frame_rect = commander_rect.inflate(
            COMMANDER_FRAME_PADDING * 2, COMMANDER_FRAME_PADDING * 2
        )
        pygame.draw.rect(
            back_ground, COLORS["BROWN"], commander_frame_rect,
            COMMANDER_FRAME_BORDER_WIDTH
        )

        button_width = int(sidebar_width * CONTROL_BUTTON_SCALE)
        button_height = button_width // 2

        def load_button(name: str) -> pygame.Surface:
            sprite = pygame.image.load(f"assets/{name}.png").convert_alpha()
            return pygame.transform.scale(
                sprite, (button_width, button_height)
            )

        reset_sprite = load_button("reset")
        slow_sprite = load_button("slow")
        play_sprite = load_button("play")
        pause_sprite = load_button("pause")
        fast_sprite = load_button("fast")

        speed_group_width = (
            3 * button_width + 2 * CONTROL_BUTTON_SPACING
        )
        row_width = button_width + CONTROL_BUTTON_GROUP_GAP + speed_group_width
        row_left = sidebar_center_x - row_width // 2

        # Anchored to the bottom of the sidebar: the speed bar sits
        # right above the bottom margin, and the button row sits
        # right above the speed bar.
        speed_bar_bottom = map_height - CONTROL_PANEL_MARGIN_BOTTOM
        speed_bar_top = speed_bar_bottom - SPEED_BAR_HEIGHT
        row_top = speed_bar_top - SPEED_BAR_MARGIN_TOP - button_height

        reset_rect = reset_sprite.get_rect(topleft=(row_left, row_top))
        slow_rect = slow_sprite.get_rect(
            topleft=(reset_rect.right + CONTROL_BUTTON_GROUP_GAP, row_top)
        )
        playpause_rect = play_sprite.get_rect(
            topleft=(slow_rect.right + CONTROL_BUTTON_SPACING, row_top)
        )
        fast_rect = fast_sprite.get_rect(
            topleft=(playpause_rect.right + CONTROL_BUTTON_SPACING, row_top)
        )
        button_rects = {
            "reset": reset_rect,
            "slow": slow_rect,
            "playpause": playpause_rect,
            "fast": fast_rect,
        }

        back_ground.blit(reset_sprite, reset_rect)
        back_ground.blit(slow_sprite, slow_rect)
        back_ground.blit(fast_sprite, fast_rect)
        # The play/pause icon reflects live state, so it's drawn fresh
        # each frame in the main loop instead of being baked in here.

        speed_bar_rect = pygame.Rect(
            slow_rect.left,
            speed_bar_top,
            fast_rect.right - slow_rect.left,
            SPEED_BAR_HEIGHT
        )
        pygame.draw.rect(back_ground, COLORS["CYAN"], speed_bar_rect)
        pygame.draw.rect(
            back_ground, COLORS["BROWN"], speed_bar_rect,
            SPEED_BAR_BORDER_WIDTH
        )
        pygame.draw.line(
            back_ground, COLORS["BROWN"],
            (speed_bar_rect.centerx, speed_bar_rect.top),
            (speed_bar_rect.centerx, speed_bar_rect.bottom),
            SPEED_BAR_TICK_WIDTH
        )
        speed_log_min = math.log2(SIMULATION_MIN_SPEED)
        speed_log_range = math.log2(SIMULATION_MAX_SPEED) - speed_log_min

        dialog_box_rect = pygame.Rect(
            map_width + DIALOG_BOX_MARGIN_SIDE,
            commander_frame_rect.bottom + DIALOG_BOX_MARGIN_TOP,
            sidebar_width - 2 * DIALOG_BOX_MARGIN_SIDE,
            row_top - DIALOG_BOX_MARGIN_BOTTOM
            - (commander_frame_rect.bottom + DIALOG_BOX_MARGIN_TOP)
        )
        pygame.draw.rect(back_ground, COLORS["BLACK"], dialog_box_rect)
        pygame.draw.rect(
            back_ground, COLORS["ORANGE"], dialog_box_rect,
            DIALOG_BOX_BORDER_WIDTH
        )
        dialog_font = pygame.font.Font(
            "assets/PressStart2P-Regular.ttf",
            max(
                DIALOG_FONT_FLOOR,
                sidebar_width // DIALOG_FONT_SIDEBAR_DIVISOR
            )
        )
        dialog_text_width = dialog_box_rect.width - 2 * DIALOG_BOX_PADDING

        dialog_phrase = random.choice(SIMULATION_DIALOG_PHRASES)
        dialog_full_text = "\n".join(
            self._wrap_text(dialog_phrase, dialog_font, dialog_text_width)
        )
        dialog_chars_shown = 0
        dialog_char_timer = 0
        dialog_hold_timer = 0
        dialog_typing = True
        mandible_open = False
        mandible_timer = 0

        dragging = False

        loop_frame = 0

        while self._running:
            self._clock.tick(self._ticks)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                elif (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                ):
                    if button_rects["playpause"].collidepoint(event.pos):
                        self._paused = not self._paused
                    elif button_rects["fast"].collidepoint(event.pos):
                        self._speed = min(
                            self._speed * SIMULATION_SPEED_STEP,
                            SIMULATION_MAX_SPEED
                        )
                    elif button_rects["slow"].collidepoint(event.pos):
                        self._speed = max(
                            self._speed / SIMULATION_SPEED_STEP,
                            SIMULATION_MIN_SPEED
                        )
                    elif button_rects["reset"].collidepoint(event.pos):
                        self._frame_count = 0.0
                        self._speed = SIMULATION_DEFAULT_SPEED
                        self._paused = True
                        loop_frame = 0
                    elif (
                        event.pos[0] < map_width
                        and event.pos[1] < map_height
                        and self._zoom > MAP_ZOOM_DEFAULT
                    ):
                        # Only past the default zoom is there anything
                        # to pan to -- at or below it the whole map
                        # already fits, so dragging there is a no-op.
                        dragging = True
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    dragging = False
                elif event.type == pygame.MOUSEMOTION:
                    if dragging and event.buttons[0]:
                        self._pan_camera(
                            event.rel[0], event.rel[1],
                            map_width, map_height
                        )
                    elif dragging:
                        # The button was released outside the window,
                        # so no MOUSEBUTTONUP arrived -- recover here.
                        dragging = False
                elif event.type == pygame.MOUSEWHEEL:
                    mouse_pos = pygame.mouse.get_pos()
                    if mouse_pos[0] < map_width and mouse_pos[1] < map_height:
                        self._apply_zoom(
                            event.y, mouse_pos, map_width, map_height
                        )

            loop_frame += 1
            if loop_frame > START_DELAY_FRAMES and not self._paused:
                self._frame_count += self._speed
            elapsed_turns = int(self._frame_count) // TURN_DURATION_FRAMES

            # The commander's dialog/mandibles are ambient flavor, kept
            # running regardless of the drone playback's own pause state.
            if dialog_typing:
                dialog_char_timer += 1
                if dialog_char_timer >= DIALOG_TYPE_FRAMES_PER_CHAR:
                    dialog_char_timer = 0
                    dialog_chars_shown = min(
                        dialog_chars_shown + 1, len(dialog_full_text)
                    )
                    if dialog_chars_shown >= len(dialog_full_text):
                        dialog_typing = False
                        dialog_hold_timer = 0
                mandible_timer += 1
                if mandible_timer >= MANDIBLE_TOGGLE_FRAMES:
                    mandible_timer = 0
                    mandible_open = not mandible_open
            else:
                mandible_open = False
                dialog_hold_timer += 1
                if dialog_hold_timer >= DIALOG_HOLD_FRAMES:
                    dialog_phrase = random.choice(SIMULATION_DIALOG_PHRASES)
                    dialog_full_text = "\n".join(
                        self._wrap_text(
                            dialog_phrase, dialog_font, dialog_text_width
                        )
                    )
                    dialog_chars_shown = 0
                    dialog_char_timer = 0
                    dialog_typing = True

            drone_counts: dict[tuple[int, int], int] = {}
            drone_directions: dict[tuple[int, int], tuple[int, int]] = {}
            for waypoints in drone_waypoints.values():
                drone_pos, direction = self._interpolate(waypoints)
                drone_counts[drone_pos] = drone_counts.get(drone_pos, 0) + 1
                if drone_pos not in drone_directions:
                    drone_directions[drone_pos] = direction

            pad_x = pad_y = 0
            if self._zoom == MAP_ZOOM_DEFAULT:
                screen.blit(back_ground, (0, 0))
            elif self._zoom > MAP_ZOOM_DEFAULT:
                view_width = map_width / self._zoom
                view_height = map_height / self._zoom
                source_rect = pygame.Rect(
                    int(self._camera_x), int(self._camera_y),
                    max(1, round(view_width)), max(1, round(view_height))
                ).clip(pygame.Rect(0, 0, map_width, map_height))
                zoomed_map = pygame.transform.scale(
                    back_ground.subsurface(source_rect),
                    (map_width, map_height)
                )
                screen.blit(zoomed_map, (0, 0))
                screen.blit(
                    back_ground, (map_width, 0),
                    area=pygame.Rect(
                        map_width, 0, sidebar_width, map_height
                    )
                )
            else:
                # Zoomed out past the default: there's no extra map
                # content to reveal, so the whole map is shrunk and
                # centered instead of panned. The margins this opens
                # up are filled with more of the (separately kept,
                # larger) starfield rather than a flat background.
                shrunk_width = max(1, round(map_width * self._zoom))
                shrunk_height = max(1, round(map_height * self._zoom))
                shrunk_map = pygame.transform.scale(
                    back_ground.subsurface(
                        pygame.Rect(0, 0, map_width, map_height)
                    ),
                    (shrunk_width, shrunk_height)
                )
                pad_x = (map_width - shrunk_width) // 2
                pad_y = (map_height - shrunk_height) // 2

                field_width = max(1, round(star_field_width * self._zoom))
                field_height = max(1, round(star_field_height * self._zoom))
                scaled_field = pygame.transform.scale(
                    star_field, (field_width, field_height)
                )
                field_x = (map_width - field_width) // 2
                field_y = (map_height - field_height) // 2
                screen.set_clip(pygame.Rect(0, 0, map_width, map_height))
                screen.blit(scaled_field, (field_x, field_y))
                screen.set_clip(None)

                screen.blit(shrunk_map, (pad_x, pad_y))
                screen.blit(
                    back_ground, (map_width, 0),
                    area=pygame.Rect(
                        map_width, 0, sidebar_width, map_height
                    )
                )

            screen.blit(
                pause_sprite if not self._paused else play_sprite,
                playpause_rect
            )

            speed_fraction = (
                (math.log2(self._speed) - speed_log_min) / speed_log_range
            )
            speed_fraction = max(0.0, min(1.0, speed_fraction))
            indicator_x = speed_bar_rect.left + int(
                speed_fraction * speed_bar_rect.width
            )
            indicator_rect = pygame.Rect(
                indicator_x - SPEED_BAR_INDICATOR_WIDTH // 2,
                speed_bar_rect.top,
                SPEED_BAR_INDICATOR_WIDTH,
                speed_bar_rect.height
            )
            pygame.draw.rect(screen, COLORS["GOLD"], indicator_rect)
            pygame.draw.rect(
                screen, COLORS["BLACK"], indicator_rect, BADGE_BORDER_WIDTH
            )

            screen.blit(
                commander_open_sprite if mandible_open
                else commander_closed_sprite,
                commander_rect
            )

            dialog_visible_text = dialog_full_text[:dialog_chars_shown]
            text_x = dialog_box_rect.left + DIALOG_BOX_PADDING
            text_y = dialog_box_rect.top + DIALOG_BOX_PADDING
            for line in dialog_visible_text.split("\n"):
                line_label = dialog_font.render(line, False, COLORS["WHITE"])
                screen.blit(line_label, (text_x, text_y))
                text_y += line_label.get_height() + DIALOG_LINE_SPACING

            screen.set_clip(pygame.Rect(0, 0, map_width, map_height))
            for drone_pos, count in drone_counts.items():
                view_pos = self._to_view(*drone_pos, pad_x, pad_y)
                dx, dy = drone_directions[drone_pos]
                if dx == 0 and dy == 0:
                    angle = 0.0
                else:
                    angle = -math.degrees(math.atan2(dx, -dy))
                drone_sprite = scaled_drone_sprite
                zoomed_radius = drone_badge_radius
                if self._zoom != MAP_ZOOM_DEFAULT:
                    zoomed_size = max(1, int(drone_size * self._zoom))
                    drone_sprite = pygame.transform.scale(
                        scaled_drone_sprite, (zoomed_size, zoomed_size)
                    )
                    zoomed_radius = max(
                        1, int(drone_badge_radius * self._zoom)
                    )
                rotated_sprite = pygame.transform.rotate(
                    drone_sprite, angle
                )
                drone_rect = rotated_sprite.get_rect(center=view_pos)
                screen.blit(rotated_sprite, drone_rect)

                pygame.draw.circle(
                    screen, COLORS["WHITE"], view_pos, zoomed_radius
                )
                pygame.draw.circle(
                    screen, COLORS["BLACK"], view_pos,
                    zoomed_radius, BADGE_BORDER_WIDTH
                )
                drone_label = drone_badge_font.render(
                    str(count), False, COLORS["BLACK"]
                )
                drone_label_rect = drone_label.get_rect(center=view_pos)
                screen.blit(drone_label, drone_label_rect)
            screen.set_clip(None)

            turn_label = turn_font.render(
                f"Turno {min(elapsed_turns, total_turns)}",
                False, COLORS["WHITE"]
            )
            screen.blit(
                turn_label, (
                    map_width + TURN_LABEL_MARGIN_SIDE,
                    TURN_LABEL_MARGIN_TOP)
            )

            pygame.display.flip()
        pygame.quit()
