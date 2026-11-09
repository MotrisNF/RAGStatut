"""Shared literal types and metadata keys used across the project."""

from typing import Literal

ZoneType = Literal[
    "normal", "blocked", "restricted", "priority", "start", "end"
]
"""Movement-cost category of a hub; ``start``/``end`` mark the gates."""

DroneStatus = Literal[
    "waiting", "in_transit", "delivered"
]
"""Lifecycle state of a drone within a simulation turn."""

_HUB_METADATA_KEYS = {"zone", "color", "max_drones"}
"""Metadata keys accepted inside a ``hub:`` line's ``[...]`` block."""

_CONNECTION_METADATA_KEYS = {"max_link_capacity"}
"""Metadata keys accepted inside a ``connection:`` line's ``[...]`` block."""

TESTING = False
"""Passed to ``Printer.print_by_letter`` to skip the letter-by-letter
animation and terminal echo handling, both incompatible with pytest's
captured stdin."""

COLORS = {
    "GREEN": (99, 127, 103),
    "BLACK": (36, 34, 46),
    "WHITE": (230, 235, 214),
    "RED": (130, 14, 16),
    "ORANGE": (255, 140, 0),
    "GOLD": (255, 215, 0),
    "CYAN": (106, 138, 153),
    "MAGENTA": (148, 30, 168),
    "GRAY": (128, 128, 128),
    "DARK_BLUE": (39, 46, 69),
    "BROWN": (148, 90, 80),
    "LIGTH_BLUE": (132, 173, 167),
    "YELLOW": (217, 215, 161),
    "LIGHT_GREEN": (129, 150, 126)
}
"""RGB palette shared by sprites, badges and zone coloring."""

ZONE_COLORS: dict[ZoneType, tuple[int, int, int]] = {
    "normal": COLORS["YELLOW"],
    "blocked": COLORS["RED"],
    "restricted": COLORS["ORANGE"],
    "priority": COLORS["LIGHT_GREEN"],
    "start": COLORS["CYAN"],
    "end": COLORS["BROWN"],
}
"""Fill color drawn for each zone type."""

DESKTOP_OFFSET = 200
"""Pixels subtracted from the desktop resolution when sizing the window."""

MARGIN_X = 60
"""Horizontal padding, in pixels, around the map grid."""

MARGIN_Y = 200
"""Vertical padding, in pixels, around the map grid."""

MAX_GRID = 200
"""Largest cell size, in pixels, a map grid is ever drawn at."""

SIDEBAR_RATIO = 1 / 3
"""Fraction of the window width reserved for the sidebar panel."""

TURN_DURATION_FRAMES = 45
"""Frames a single simulated turn takes to animate, pause included."""

TURN_PAUSE_FRAMES = 8
"""Frames a drone holds still at a hub before the next turn starts."""

START_DELAY_FRAMES = 60
"""Frames the animation waits before the first turn begins."""

FRAME_RATE = 30.0
"""Target frames per second for the pygame window."""

NODE_SPRITE_SCALE = 0.55
"""Node sprite size, as a fraction of the grid cell size."""

DRONE_SPRITE_SCALE = 0.55
"""Drone sprite size, as a fraction of the grid cell size."""

MOON_SIZE_DIVISOR = 4
"""The moon's diameter is the shorter map dimension divided by this."""

CONNECTION_LINE_WIDTH = 10
"""Width of the black outline drawn under each connection line."""

CONNECTION_LINE_INNER_WIDTH = 4
"""Width of the white line drawn on top, giving the outlined look."""

BADGE_BORDER_WIDTH = 2
"""Shared outline thickness for every white-circle badge."""

CAPACITY_BADGE_RADIUS_FLOOR = 6
"""Smallest radius a connection/node capacity badge is ever drawn at."""

CAPACITY_BADGE_FONT_FLOOR = 9
"""Smallest font size a connection/node capacity badge ever uses."""

CAPACITY_BADGE_GRID_DIVISOR = 8
"""Divides the grid cell size to get the badge's proportional radius."""

CAPACITY_BADGE_SIZE_CAP_DIVISOR = 3
"""Divides the node sprite size to cap how large the badge can grow."""

CAPACITY_BADGE_RADIUS_SIZE_CAP_MIN = 2
"""Margin kept clear when capping the badge radius to the node size."""

CAPACITY_BADGE_FONT_SIZE_CAP_MIN = 6
"""Margin kept clear when capping the badge font to the node size."""

DRONE_BADGE_RADIUS_FLOOR = 6
"""Smallest radius the drone count badge is ever drawn at."""

DRONE_BADGE_FONT_FLOOR = 9
"""Smallest font size the drone count badge ever uses."""

DRONE_BADGE_GRID_DIVISOR = 10
"""Divides the grid cell size to get the badge's proportional radius."""

DRONE_BADGE_SIZE_CAP_DIVISOR = 3
"""Divides the drone sprite size to cap how large the badge can grow."""

DRONE_BADGE_RADIUS_SIZE_CAP_MIN = 2
"""Margin kept clear when capping the badge radius to the drone size."""

DRONE_BADGE_FONT_SIZE_CAP_MIN = 6
"""Margin kept clear when capping the badge font to the drone size."""

TURN_LABEL_FONT_FLOOR = 24
"""Smallest font size the turn counter label ever uses."""

TURN_LABEL_SIDEBAR_DIVISOR = 12
"""Divides the sidebar width to get the label's proportional font size."""

TURN_LABEL_MARGIN_TOP = 50
TURN_LABEL_MARGIN_SIDE = 250
"""Pixels between the turn counter label and the sidebar's edges."""

NORMAL_ZONE_TURN_COST = 1
"""Turns spent crossing a normal or priority hub."""

RESTRICTED_ZONE_TURN_COST = 2
"""Turns spent crossing a restricted hub."""

MAX_CANDIDATE_PATHS = 8
"""Most alternate routes Router keeps per drone during route planning."""

PATH_COST_CEILING_RATIO = 3.5
"""A candidate route is dropped once it costs this many times the best."""

PRIORITY_BIAS = 1e-3
"""Tiny weight discount making Dijkstra prefer priority hubs on ties."""

COMPLEXITY_THRESHOLD = 20.0
"""Bottleneck severity above which a map is routed by the advanced planner."""

REFINE_TRIAL_BUDGET = 400
"""Local-search trials refine_routes runs before giving up on improving."""

SIMULATION_DEFAULT_SPEED = 1.0
"""Turn-playback speed multiplier the simulation starts at."""

SIMULATION_MIN_SPEED = 0.25
"""Slowest playback speed the "slow" button can reach."""

SIMULATION_MAX_SPEED = 4.0
"""Fastest playback speed the "fast" button can reach."""

SIMULATION_SPEED_STEP = 2.0
"""Factor the "fast"/"slow" buttons multiply or divide the speed by."""

COMMANDER_SPRITE_SCALE = 0.6
"""Commander portrait size, as a fraction of the sidebar width."""

COMMANDER_MARGIN_TOP = 40
"""Pixel gap between the turn counter label and the commander portrait."""

CONTROL_BUTTON_SCALE = 0.18
"""Each control button's width, as a fraction of the sidebar width."""

CONTROL_BUTTON_SPACING = 20
"""Pixel gap between neighboring control buttons."""

CONTROL_PANEL_MARGIN_BOTTOM = 50
"""Pixel gap between the bottom of the sidebar and the speed bar above it."""

CONTROL_BUTTON_GROUP_GAP = 50
"""Pixel gap separating the reset button from the speed control group."""

SPEED_BAR_MARGIN_TOP = 30
"""Pixel gap between the control buttons and the speed bar below them."""

SPEED_BAR_HEIGHT = 28
"""Height, in pixels, of the speed bar track."""

SPEED_BAR_BORDER_WIDTH = 4
"""Outline thickness of the speed bar track."""

SPEED_BAR_TICK_WIDTH = 4
"""Width of the center tick marking the default (standard) speed."""

SPEED_BAR_INDICATOR_WIDTH = 10
"""Width of the marker showing the current speed on the bar."""

MAP_ZOOM_DEFAULT = 1.0
"""Zoom level the map view starts at -- the initial, unzoomed state."""

MAP_ZOOM_MIN = 0.75
"""Smallest zoom level the mouse wheel can reach on the map view, letting
it pull back a bit further than the initial state."""

MAP_ZOOM_MAX = 4.0
"""Largest zoom level the mouse wheel can reach on the map view."""

MAP_ZOOM_STEP_FACTOR = 1.1
"""Multiplier applied to the zoom level per mouse wheel notch."""

COMMANDER_FRAME_PADDING = 10
"""Padding, in pixels, between the commander portrait and its frame."""

COMMANDER_FRAME_BORDER_WIDTH = 4
"""Outline thickness of the frame drawn around the commander portrait."""

DIALOG_BOX_MARGIN_SIDE = 20
"""Horizontal margin, in pixels, between the dialog box and the sidebar's
edges."""

DIALOG_BOX_MARGIN_TOP = 30
"""Pixel gap between the commander frame and the dialog box below it."""

DIALOG_BOX_MARGIN_BOTTOM = 30
"""Pixel gap between the dialog box and the control buttons below it."""

DIALOG_BOX_PADDING = 16
"""Padding, in pixels, between the dialog box border and its text."""

DIALOG_BOX_BORDER_WIDTH = 4
"""Outline thickness of the dialog box border."""

DIALOG_FONT_SIDEBAR_DIVISOR = 22
"""Divides the sidebar width to get the dialog text's font size."""

DIALOG_FONT_FLOOR = 10
"""Smallest font size the dialog text ever uses."""

DIALOG_LINE_SPACING = 8
"""Pixel gap between successive lines of the dialog text."""

DIALOG_TYPE_FRAMES_PER_CHAR = 3
"""Frames held between revealing each new letter of the dialog text."""

DIALOG_HOLD_FRAMES = 90
"""Frames a fully-typed dialog phrase stays on screen before changing."""

MANDIBLE_TOGGLE_FRAMES = 4
"""Frames between each mandible open/closed flip while the commander
"talks"."""

SIMULATION_DIALOG_PHRASES = [
    "Welcome to the simulation",
    "Drones inbound to the hub",
    "Watch for restricted zones",
    "Priority routes save time",
    "Plotting the fastest path",
    "Blocked zones are off limits",
    "Stand by for departure",
    "All systems operational",
    "Hub capacity is holding",
    "Reset returns to turn zero",
]
"""Random flavor phrases the commander "says" in the sidebar dialog box."""

PERFORMANCE_PRINT_SPEED = 0.005
"""Seconds between each letter of a ``make performance`` report line."""

PERFORMANCE_PRINT_PAUSE = 0.05
"""Seconds paused after a ``make performance`` report line finishes."""

ANSI_GREEN = "\033[92m"
"""ANSI escape code for bright green terminal text (a target met)."""

ANSI_RED = "\033[91m"
"""ANSI escape code for bright red terminal text (a target missed)."""
