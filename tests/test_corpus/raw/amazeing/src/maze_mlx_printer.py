"""Rendering and game loop for the MiniLibX (graphical) game session."""

import time
from typing import Any
from os import _exit, path
from mlx import Mlx
from mazegen import MazeGen, MazeGenError


class MovingEntity:
    """Tracks and animates the smooth, tile-by-tile movement of an actor.

    Movement is driven by a requested direction (set through 'set_input')
    that gets applied one tile at a time: the entity keeps sliding towards
    the current tile's neighbour until it reaches the next tile's center,
    at which point it can pick up a newly requested direction or stop if
    a wall blocks the way.

    Attributes:
        maze (list[str]): Hexadecimal grid string representation of the
            maze the entity moves through.
        size (tuple[int, int]): Dimensions of the maze grid as
            (width, height).
        tile_size (int): The size, in pixels, of one maze cell.
        speed (float): Fraction of a tile crossed on each 'update' call.
        current_cell (list[int]): The (x, y) grid cell the entity is
            currently on (or leaving from).
        position (list[float]): The (x, y) offset, in tile units, from
            'current_cell' towards the next cell.
        current_direction (tuple[int, int] | None): The direction the
            entity is actively moving towards, or None if stopped.
        requested_direction (tuple[int, int] | None): The last direction
            requested through 'set_input', applied as soon as it becomes
            possible.
    """

    def __init__(
        self,
        maze: list[str],
        size: tuple[int, int],
        start_coord: tuple[int, int],
        tile_size: int = 64,
        speed: float = 0.1
    ) -> None:
        """Initialize the entity at 'start_coord', stopped.

        Args:
            maze (list[str]): Hexadecimal grid string representation of
                the maze the entity moves through.
            size (tuple[int, int]): Dimensions of the maze grid as
                (width, height).
            start_coord (tuple[int, int]): The (x, y) grid cell the
                entity starts on.
            tile_size (int): The size, in pixels, of one maze cell.
                Defaults to 64.
            speed (float): Fraction of a tile crossed on each 'update'
                call. Defaults to 0.1.
        """
        self.maze = maze
        self.size = size
        self.tile_size = tile_size
        self.speed = speed

        self.current_cell = [int(start_coord[0]), int(start_coord[1])]
        self.position = [0.0, 0.0]

        self.current_direction: tuple[int, int] | None = None
        self.requested_direction: tuple[int, int] | None = None

    def set_input(self, direction: tuple[int, int] | None) -> None:
        """Record the direction the player wants to move towards next.

        Doesn't move the entity immediately; 'update' picks it up once
        the entity is free to change direction (e.g. at a tile center).

        Args:
            direction (tuple[int, int] | None): The (dx, dy) direction
                requested, or None to request no movement.
        """
        self.requested_direction = direction

    def screen_position(self) -> tuple[float, float]:
        """Give the entity's current pixel position on screen.

        Returns:
            tuple[float, float]: The (x, y) pixel coordinates, combining
                'current_cell' with the in-progress offset in 'position'.
        """
        return (
            (self.current_cell[0] + self.position[0]) * self.tile_size,
            (self.current_cell[1] + self.position[1]) * self.tile_size,
        )

    def at_tile_center(self) -> bool:
        """Check whether the entity is currently centered on a tile.

        Returns:
            bool: True if 'position' is (0, 0), within a small tolerance,
                i.e. the entity isn't mid-transition between two cells.
                Else False.
        """
        return abs(self.position[0]) < 1e-6 and abs(self.position[1]) < 1e-6

    def is_valid_move(self, new_x: int, new_y: int) -> bool:
        """Check whether the entity can move from 'current_cell' to
        (new_x, new_y).

        Args:
            new_x (int): The x position of the destination cell.
            new_y (int): The y position of the destination cell.

        Returns:
            bool: True if (new_x, new_y) is inside the maze bounds and no
                wall blocks the move from 'current_cell' to it. Else
                False.
        """
        if not (0 <= new_x < self.size[0] and 0 <= new_y < self.size[1]):
            return False

        curr_x, curr_y = self.current_cell
        curr_tile = int(self.maze[curr_y][curr_x], 16)
        dest_tile = int(self.maze[new_y][new_x], 16)

        dx, dy = new_x - curr_x, new_y - curr_y

        if dy == -1 and ((curr_tile & 1) or (dest_tile & 4)):
            return False
        if dy == 1 and ((curr_tile & 4) or (dest_tile & 1)):
            return False
        if dx == 1 and ((curr_tile & 2) or (dest_tile & 8)):
            return False
        if dx == -1 and ((curr_tile & 8) or (dest_tile & 2)):
            return False
        return True

    def can_move_towards(self, direction: tuple[int, int] | None) -> bool:
        """Check whether the entity can move one cell towards 'direction'.

        Args:
            direction (tuple[int, int] | None): The (dx, dy) direction to
                check, or None.

        Returns:
            bool: True if 'direction' is not None and the neighbouring
                cell in that direction is reachable from 'current_cell'.
                Else False.
        """
        if not direction:
            return False
        return self.is_valid_move(self.current_cell[0] + direction[0],
                                  self.current_cell[1] + direction[1])

    def update(self) -> bool:
        """Advance the entity's animation by one step.

        Picks up 'requested_direction' when possible, slides 'position'
        towards the next tile at 'speed', and once a full tile has been
        crossed, moves 'current_cell' over and resets 'position'. Stops
        the entity ('current_direction' set to None) whenever a wall
        blocks further movement in the current direction.

        Returns:
            bool: True if the entity moved or is moving. Else False.
        """
        if self.current_direction is None:
            if self.can_move_towards(self.requested_direction):
                self.current_direction = self.requested_direction
                self.position = [0.0, 0.0]
            else:
                return False

        if self.at_tile_center():
            if (self.requested_direction and
               self.requested_direction != self.current_direction):
                if self.can_move_towards(self.requested_direction):
                    self.current_direction = self.requested_direction

            if not self.can_move_towards(self.current_direction):
                self.current_direction = None
                return False

        if self.current_direction is not None:
            self.position[0] += self.current_direction[0] * self.speed
            self.position[1] += self.current_direction[1] * self.speed

        for i in range(2):
            if self.position[i] >= 1.0:
                self.current_cell[i] += 1
                self.position[i] = 0.0
                if not self.can_move_towards(self.current_direction):
                    self.current_direction = None
            elif self.position[i] <= -1.0:
                self.current_cell[i] -= 1
                self.position[i] = 0.0
                if not self.can_move_towards(self.current_direction):
                    self.current_direction = None

        return True


class InputConfig:
    """Stores numerical constants for keyboard event mapping.

    This class serves as a static configuration register, defining hardware
    keycodes for Linux (X11) systems by default, with additional support for
    on-demand reconfiguration for macOS (Cocoa) platforms.

    Attributes:
        KEY_W (int): Keycode for character 'W' (Up action movement).
        KEY_A (int): Keycode for character 'A' (Left action movement).
        KEY_S (int): Keycode for character 'S' (Down action movement).
        KEY_D (int): Keycode for character 'D' (Right action movement).
        KEY_P (int): Keycode for character 'P' (Print solution).
        KEY_R (int): Keycode for character 'R' (Regenerate maze layout).
        KEY_ESC (int): Keycode for the Escape key (terminate application).
    """
    KEY_W: int = 119
    KEY_A: int = 97
    KEY_S: int = 115
    KEY_D: int = 100
    KEY_P: int = 112
    KEY_R: int = 114
    KEY_M: int = 109
    KEY_ESC: int = 65307


class AssetManager:
    """Manages exclusively the loading and storage of game sprites.

    This class automates the retrieval of wall and entity textures from disk.

    Attributes:
        m (Mlx): The MiniLibX framework wrapper instance.
        mlx_ptr (Any): The connection identifier pointer initialization.
        base_path (str): The root directory where asset subfolders are located.
        walls_sprites (dict[str, Any]): Hexadecimal tiles mapped to images.
        in_sprite (Any): The entry point tile graphical resource pointer.
        out_sprite (Any): The exit point tile graphical resource pointer.
        player_sprite (Any): The entity avatar graphical resource pointer.
    """
    def __init__(
            self,
            mlx: Mlx,
            mlx_ptr: Any,
            color: bool
            ) -> None:
        """Initializes the AssetManager with core graphical references.

        Args:
            mlx (Mlx): The wrapper utility class instance for MiniLibX.
            mlx_ptr (Any): Connection handle generated by mlx_init.
            base_path (str): Root folder directory path for assets.
            Defaults to "assets".
        """
        self.color = color
        self.m = mlx
        self.mlx_ptr = mlx_ptr
        self.base_path = self._base_path_for(self.color)

        self.walls_sprites: dict[str, Any] = {}
        self.in_sprite: Any = None
        self.out_sprite: Any = None
        self.player_sprite: Any = None

    @staticmethod
    def _base_path_for(color: bool) -> str:
        """Give the asset root folder for a given sprite theme.

        Args:
            color (bool): True for the colored ('x1') sprite set, False
                for the plain ('x2') one.

        Returns:
            str: The root directory path for the requested sprite theme.
        """
        return "assets/x1" if color else "assets/x2"

    def toggle_color(self) -> None:
        """Switch between the colored and plain sprite themes.

        Flips 'color', points 'base_path' at the other theme's folder,
        drops the cached wall sprites and reloads every texture from the
        new theme.
        """
        self.color = not self.color
        self.base_path = self._base_path_for(self.color)
        self.walls_sprites = {}
        self.load_all()

    def load_all(self) -> None:
        """Loads all required PNG textures from disk into memory storage.

        Parses all hexadecimal variations of structural maze walls, as well as
        the interface targets for entrance, exit, and the controllable player.
        """
        wall_types = {
            '0': "walls/floor.png",
            '1': "walls/wall_n.png",
            '2': "walls/wall_e.png",
            '4': "walls/wall_s.png",
            '8': "walls/wall_w.png",
            '3': "walls/wall_ne.png",
            '5': "walls/wall_ns.png",
            '9': "walls/wall_nw.png",
            '6': "walls/wall_es.png",
            'A': "walls/wall_ew.png",
            'C': "walls/wall_sw.png",
            '7': "walls/wall_nes.png",
            'B': "walls/wall_new.png",
            'D': "walls/wall_nsw.png",
            'E': "walls/wall_esw.png",
            'F': "walls/wall_nesw.png",
        }

        for key, relative_path in wall_types.items():
            full_path = f"{self.base_path}/{relative_path}"
            if not path.exists(full_path):
                raise MazeGenError(f"Sprite missing: {full_path}")
            self.walls_sprites[key] = self.m.mlx_png_file_to_image(
                self.mlx_ptr, full_path)[0]

        if not path.exists(f"{self.base_path}/in_out/in.png"):
            raise MazeGenError(
                f"Sprite missing: {self.base_path}/in_out/in.png"
                )
        if not path.exists(f"{self.base_path}/in_out/out.png"):
            raise MazeGenError(
                f"Sprite missing: {self.base_path}/in_out/out.png"
                )
        if not path.exists(f"{self.base_path}/player/player.png"):
            raise MazeGenError(
                f"Sprite missing: {self.base_path}/player/player.png"
                )
        self.in_sprite = self.m.mlx_png_file_to_image(
            self.mlx_ptr, f"{self.base_path}/in_out/in.png")[0]
        self.out_sprite = self.m.mlx_png_file_to_image(
            self.mlx_ptr, f"{self.base_path}/in_out/out.png")[0]
        self.player_sprite = self.m.mlx_png_file_to_image(
            self.mlx_ptr, f"{self.base_path}/player/player.png")[0]


class Renderer:
    """Thin wrapper around MiniLibX drawing calls, on the tile grid.

    Attributes:
        m (Any): The MiniLibX framework wrapper instance.
        mlx_ptr (Any): The connection identifier pointer.
        win_ptr (Any): The target window pointer.
        tile_size (int): The size, in pixels, of one maze cell.
    """

    def __init__(
            self,
            mlx: Any,
            mlx_ptr: Any,
            win_ptr: Any,
            tile_size: int
            ) -> None:
        """Initialize the renderer with its MiniLibX connection.

        Args:
            mlx (Any): The MiniLibX framework wrapper instance.
            mlx_ptr (Any): The connection identifier pointer.
            win_ptr (Any): The target window pointer.
            tile_size (int): The size, in pixels, of one maze cell.
        """
        self.m = mlx
        self.mlx_ptr = mlx_ptr
        self.win_ptr = win_ptr
        self.tile_size = tile_size

    def draw_tile(self, sprite: Any, grid_x: int, grid_y: int) -> None:
        """Draws a sprite snapped exactly to the integer coordinate grid."""
        self.m.mlx_put_image_to_window(
            self.mlx_ptr,
            self.win_ptr,
            sprite,
            grid_x * self.tile_size,
            grid_y * self.tile_size,
        )

    def draw_entity(self, sprite: Any, screen_pos: tuple[float, float]
                    ) -> None:
        """Draws a sprite at a free-floating pixel position.

        Used for moving actors, whose position doesn't sit exactly on the
        tile grid while they animate between cells.

        Args:
            sprite (Any): The image resource to draw.
            screen_pos (tuple[float, float]): The (x, y) pixel position,
                rounded to the nearest integer before drawing.
        """
        self.m.mlx_put_image_to_window(
            self.mlx_ptr,
            self.win_ptr,
            sprite,
            int(round(screen_pos[0])),
            int(round(screen_pos[1])),
        )

    def draw_path_marker(self, grid_x: int, grid_y: int, text: str, color: int
                         ) -> None:
        """Draws a solution-path marker centered on a maze cell.

        Args:
            grid_x (int): The x position of the cell, in grid units.
            grid_y (int): The y position of the cell, in grid units.
            text (str): The marker text to draw (e.g. "*").
            color (int): The 0xRRGGBB text color.
        """
        center_x = int((grid_x * self.tile_size) + (self.tile_size / 2) - 5)
        center_y = int((grid_y * self.tile_size) + (self.tile_size / 2) - 10)
        self.m.mlx_string_put(
            self.mlx_ptr,
            self.win_ptr,
            center_x,
            center_y,
            color,
            text,
        )
        self.m.mlx_do_sync(self.mlx_ptr)

    def draw_text(self, x: int, y: int, text: str, color: int = 0xFFFFFF
                  ) -> None:
        """Draws a line of text at a raw pixel position.

        Args:
            x (int): The x pixel position.
            y (int): The y pixel position.
            text (str): The text to draw.
            color (int): The 0xRRGGBB text color. Defaults to white
                (0xFFFFFF).
        """
        self.m.mlx_string_put(
            self.mlx_ptr,
            self.win_ptr,
            x,
            y,
            color,
            text,
        )
        self.m.mlx_do_sync(self.mlx_ptr)


class Menu:
    """Draws the fixed help text shown at the bottom of the game window.

    Attributes:
        renderer (Renderer): The renderer used to draw the menu text.
        width (int): The width, in pixels, of the game window.
        height (int): The height, in pixels, of the game window.
        padding (int): The margin, in pixels, around the menu text.
        line_height (int): The vertical spacing, in pixels, between
            menu lines.
    """

    def __init__(
            self,
            renderer: Renderer,
            width: int,
            height: int,
            padding: int = 10,
            line_height: int = 18,
    ) -> None:
        """Initialize the menu with its renderer and layout.

        Args:
            renderer (Renderer): The renderer used to draw the menu text.
            width (int): The width, in pixels, of the game window.
            height (int): The height, in pixels, of the game window.
            padding (int): The margin, in pixels, around the menu text.
                Defaults to 10.
            line_height (int): The vertical spacing, in pixels, between
                menu lines. Defaults to 18.
        """
        self.renderer = renderer
        self.width = width
        self.height = height
        self.padding = padding
        self.line_height = line_height

    def draw(self) -> None:
        """Draws the help text, anchored to the bottom of the window."""
        lines = [
            "",
            "'w,a,s,d' - Move the player.",
            "'m' - Select the stile of the maze.",
            "'p' - Show the shorter way.",
            "'r' - Regenerate the maze.",
            "'Esc' - Exit.",
        ]

        y = self.height - self.padding - self.line_height * len(lines)
        for line in lines:
            self.renderer.draw_text(self.padding, y, line, 0xFFFFFF)
            y += self.line_height


class MlxPrinter:
    """Orchestrates the core game loop, event handles, and sub-system
    communication.

    This class coordinates the initialization of the graphics context, anchors
    the operational inputs to the physical layout, updates actors under a timed
    FPS constraint, and invokes ordered draw calls.

    Attributes:
        m (Mlx): The MiniLibX framework utility library instance.
        mlx_ptr (Any): Connection handle.
        tile_size (int): The operational cell size in pixels.
        generator (MazeGen): Core maze generation data reference.
        maze (list[str]): Hexadecimal grid string representation layout.
        size (tuple[int, int]): Total dimensions (width, height) of the map.
        enter_coord (tuple[int, int]): Horizontal and vertical entry points.
        exit_coord (tuple[int, int]): Horizontal and vertical exit points.
        show_path (bool): save if path solution must be shown.
        path_coords (list[tuple[int, int]]): Ordered steps for solution.
        animated_steps (int): Counter tracking the animated path progression.
        last_frame_time (float): Acurate timestamp of last update.
        win_ptr (Any): Current active window resource context pointer.
        assets (AssetManager): Memory resource storage object for textures.
        renderer (Renderer): Direct drawing wrapper utility pipeline.
        player_entity (MovingEntity): The logic controller for actor physics.
    """
    def __init__(
            self,
            generator: MazeGen,
            color: bool,
            player_coord: tuple[int, int],
            tile_size: int = 64
    ) -> None:
        """Initializes the game window, loads assets and draws the first
        frame.

        Args:
            generator (MazeGen): The already-generated and solved maze to
                play.
            color (bool): Whether to load the colored ('x1') sprite
                theme instead of the plain ('x2') one.
            player_coord (tuple[int, int]): The (x, y) starting cell
                recorded as the player's last known position.
            tile_size (int): The operational cell size in pixels.
                Defaults to 64.

        Raises:
            SystemExit: If a required sprite file is missing on disk.
        """
        self.m = Mlx()
        self.mlx_ptr = self.m.mlx_init()
        self.tile_size = tile_size
        self.menu_height = 128
        self.window_width = self.tile_size * generator.width
        if self.window_width < 400:
            self.window_width = 400
        self.window_height = (self.tile_size * generator.height +
                              self.menu_height)

        self.generator = generator
        self.maze = generator.grid
        self.size = (generator.width, generator.height)
        self.enter_coord = generator.entry
        self.exit_coord = generator.exit

        self.show_path: bool = False
        self.path_coords: list[tuple[int, int]] = []
        self.animated_steps: int = 0
        self.frame_counter: int = 0
        self.path_redraw_enabled: bool = False
        self.path_visible_cells: set[tuple[int, int]] = set()
        self.path_clear_queue: list[tuple[int, int]] = []

        self.last_frame_time: float = time.perf_counter()

        self.win_ptr = self.create_window()
        self.assets = AssetManager(self.m, self.mlx_ptr, color)

        self.renderer = Renderer(self.m, self.mlx_ptr,
                                 self.win_ptr, self.tile_size)
        self.menu = Menu(self.renderer, self.window_width, self.window_height)

        self.player_entity = MovingEntity(
            maze=self.maze,
            size=self.size,
            start_coord=self.enter_coord,
            tile_size=self.tile_size,
        )
        self.last_player_cell = tuple(player_coord)
        try:
            self.assets.load_all()
        except MazeGenError as e:
            print(f"Asset error: {e}")
            exit(1)
        self.path_coords = self.parse_min_path()
        self.setup_hooks()

        self._needs_full_redraw = True
        self._render_needed = True
        self.menu.draw()
        self.draw_frame()
        self.menu.draw()

    def create_background(self) -> None:
        """Flag the next frame to redraw the whole scene from scratch.

        Used after the maze layout changes (e.g. on 'regenerate'), where
        the previous frame's drawn tiles are no longer valid.
        """
        self._needs_full_redraw = True
        self._render_needed = True

    def create_window(self) -> Any:
        """Open the MiniLibX game window, sized for the current maze.

        Returns:
            Any: The window pointer returned by MiniLibX.
        """
        return self.m.mlx_new_window(
            self.mlx_ptr,
            self.window_width,
            self.window_height,
            "A-Maze-ing"
        )

    def parse_min_path(self) -> list[tuple[int, int]]:
        """Turn the maze's cardinal-direction solution string into cell
        coordinates.

        Walks the first solution returned by 'MazeGen.solve', starting
        from 'enter_coord', turning each 'N'/'S'/'E'/'W' move into the
        (x, y) coordinates of the cell it leads to.

        Returns:
            list[tuple[int, int]]: The (x, y) coordinates of every cell
                on the solution path, from entry to exit.
        """
        solution_string = self.generator.solution[0][1]
        coords = [self.enter_coord]
        curr_x, curr_y = self.enter_coord
        move_map = {'N': (0, -1), 'S': (0, 1), 'E': (1, 0), 'W': (-1, 0)}

        for move in solution_string:
            if move in move_map:
                dx, dy = move_map[move]
                curr_x += dx
                curr_y += dy
                coords.append((curr_x, curr_y))
        return coords

    def setup_hooks(self) -> None:
        """Register the MiniLibX key and loop event callbacks.

        Wires 'key_handler' to keyboard events and 'loop_handler' to the
        per-frame idle loop, both bound to this instance.
        """
        self.m.mlx_key_hook(self.win_ptr, self.key_handler, self)
        self.m.mlx_loop_hook(self.mlx_ptr, self.loop_handler, self)

    def _draw_background_tile(self, cell: tuple[int, ...]) -> None:
        """Redraws a single maze cell's wall sprite (and entry/exit
        marker, if applicable).

        Used to restore a cell to its base appearance, e.g. after the
        player has moved off of it.

        Args:
            cell (tuple[int, ...]): The (x, y) coordinates of the cell
                to redraw.
        """
        x, y = cell

        if cell == self.enter_coord:
            self.renderer.draw_tile(
                self.assets.walls_sprites[self.maze[y][x].upper()],
                x,
                y,
            )
            self.renderer.draw_tile(self.assets.in_sprite, x, y)
        elif cell == self.exit_coord:
            self.renderer.draw_tile(
                self.assets.walls_sprites[self.maze[y][x].upper()],
                x,
                y,
            )
            self.renderer.draw_tile(self.assets.out_sprite, x, y)
        else:
            self.renderer.draw_tile(
                self.assets.walls_sprites[self.maze[y][x].upper()],
                x,
                y,
            )

    def _draw_path_marker(self, cell: tuple[int, int]) -> None:
        """Draws a single solution-path marker and records it as visible.

        Args:
            cell (tuple[int, int]): The (x, y) coordinates of the cell to
                mark.
        """
        self.renderer.draw_path_marker(cell[0], cell[1], "*", 0x00FFFF)
        self.path_visible_cells.add(cell)

    def _restore_cell(self, cell: tuple[int, ...]) -> None:
        """Redraws a cell's background tile, re-adding its path marker
        if one was showing there.

        Args:
            cell (tuple[int, ...]): The (x, y) coordinates of the cell to
                restore.
        """
        self._draw_background_tile(cell)
        if (self.show_path and self.path_redraw_enabled and
           cell in self.path_visible_cells):
            self._draw_path_marker(cell)

    def _draw_path_markers(self) -> None:
        """Draws a marker on every currently visible solution-path cell."""
        for cell in self.path_visible_cells:
            self.renderer.draw_path_marker(cell[0], cell[1], "*", 0x00FFFF)

    def _draw_full_scene(self) -> None:
        """Redraws every maze tile, the entry/exit markers, the visible
        solution path and the player, from scratch.

        Used for the initial frame and whenever 'create_background' has
        flagged a full redraw (e.g. after regenerating the maze or
        toggling the sprite theme).
        """
        for y_row, row in enumerate(self.maze):
            for x_col, tile in enumerate(row):
                self.renderer.draw_tile(
                    self.assets.walls_sprites[tile.upper()],
                    x_col,
                    y_row,
                )

        if self.show_path:
            self.path_visible_cells = set(
                self.path_coords[:self.animated_steps])
            self._draw_path_markers()

        self.renderer.draw_tile(
            self.assets.in_sprite, self.enter_coord[0], self.enter_coord[1])
        self.renderer.draw_tile(
            self.assets.out_sprite, self.exit_coord[0], self.exit_coord[1])

        player_x, player_y = self.player_entity.screen_position()
        self.renderer.draw_entity(
            self.assets.player_sprite, (player_x, player_y))
        self.last_player_cell = tuple(self.player_entity.current_cell)

    def draw_frame(self) -> None:
        """Renders the current frame, doing only the work actually needed.

        Does nothing if '_render_needed' is False. Otherwise, either
        redraws the whole scene (when '_needs_full_redraw' is set) or
        does an incremental update: restores the tiles the player is
        leaving/passing through/heading towards and redraws the player
        sprite at its new position. Flushes the drawing to the window
        either way.
        """
        if not getattr(self, "_render_needed", True):
            return

        if getattr(self, "_needs_full_redraw", False):
            self._draw_full_scene()
            self._needs_full_redraw = False
        else:
            previous_cell = self.last_player_cell
            current_cell = tuple(self.player_entity.current_cell)
            movement_target = None
            if self.player_entity.current_direction is not None:
                dx, dy = self.player_entity.current_direction
                movement_target = (current_cell[0] + dx, current_cell[1] + dy)

            cells_to_restore = {previous_cell, current_cell}
            if movement_target:
                cells_to_restore.add(movement_target)
                cells_to_restore.add(self.enter_coord)
                cells_to_restore.add(self.exit_coord)

            for cell in cells_to_restore:
                self._restore_cell(cell)

            if (not self.path_redraw_enabled and
               current_cell in self.path_visible_cells):
                self.path_visible_cells.remove(current_cell)
                self._draw_background_tile(current_cell)

            player_x, player_y = self.player_entity.screen_position()
            self.renderer.draw_entity(self.assets.player_sprite,
                                      (player_x, player_y))
            self.last_player_cell = current_cell

        self.m.mlx_do_sync(self.mlx_ptr)
        self._render_needed = False

    def regenerate(self, generator: MazeGen) -> None:
        """Replace the current maze with a freshly regenerated one.

        Regenerates 'generator' in place, resets every piece of state
        derived from the previous maze (player entity, solution path,
        path-animation progress) and schedules a full redraw.

        Args:
            generator (MazeGen): The maze generator to regenerate. Must
                be the same generator this printer was built with (it's
                also stored as 'self.generator').
        """
        generator.regenerate()
        self.maze = generator.grid
        self.size = (generator.width, generator.height)
        self.enter_coord = generator.entry
        self.exit_coord = generator.exit
        self.player_entity = MovingEntity(
            maze=self.maze,
            size=self.size,
            start_coord=self.enter_coord,
            tile_size=self.tile_size,
        )
        self.generator.solve(self.enter_coord, self.exit_coord)
        self.path_coords = self.parse_min_path()
        self.animated_steps = 0
        self.frame_counter = 0
        self.path_visible_cells.clear()
        self.path_clear_queue.clear()
        self.create_background()
        self.draw_frame()

    def key_handler(self, keycode: int, param: Any = None) -> int:
        """Handles a single keyboard event from MiniLibX.

        Maps 'w'/'a'/'s'/'d' to a movement request, 'm' to toggling the
        sprite theme, 'p' to toggling the solution-path animation, 'r'
        to regenerating the maze, and Escape to quitting.

        Args:
            keycode (int): The raw keycode reported by MiniLibX. Compared
                against the constants in 'InputConfig'.
            param (Any): Unused; required by the MiniLibX hook signature.
                Defaults to None.

        Returns:
            int: Always 0, as required by the MiniLibX hook signature.
        """
        if keycode == InputConfig.KEY_W:
            self.player_entity.set_input((0, -1))
        elif keycode == InputConfig.KEY_S:
            self.player_entity.set_input((0, 1))
        elif keycode == InputConfig.KEY_A:
            self.player_entity.set_input((-1, 0))
        elif keycode == InputConfig.KEY_D:
            self.player_entity.set_input((1, 0))
        elif keycode == InputConfig.KEY_M:
            try:
                self.assets.toggle_color()
            except MazeGenError as e:
                print(f"Asset error: {e}")
                exit(1)
            self._needs_full_redraw = True
            self._render_needed = True
            self.draw_frame()

        elif keycode == InputConfig.KEY_P:
            self.show_path = not self.show_path
            if self.show_path:
                self.animated_steps = 0
                self.frame_counter = 0
                self.path_visible_cells.clear()
                self.path_clear_queue.clear()
            else:
                self.path_clear_queue = list(
                    self.path_coords[:self.animated_steps])
                self.path_visible_cells.clear()
            self._needs_full_redraw = True
            self._render_needed = True
            self.draw_frame()

        elif keycode == InputConfig.KEY_R:
            self.regenerate(self.generator)
            for _ in range(0, 2):
                self.show_path = not self.show_path
                if self.show_path:
                    self.animated_steps = 0
                    self.frame_counter = 0
                    self.path_visible_cells.clear()
                    self.path_clear_queue.clear()
                else:
                    self.path_clear_queue = list(
                        self.path_coords[:self.animated_steps])
                    self.path_visible_cells.clear()
                self._needs_full_redraw = True
                self._render_needed = True
                self.draw_frame()
        elif keycode == InputConfig.KEY_ESC:
            _exit(0)
        return 0

    def loop_handler(self, param: Any = None) -> int:
        """Advances one tick of the game's idle/animation loop.

        Called by MiniLibX on every idle cycle. Drains one pending
        solution-path cell to erase per call, otherwise updates the
        player entity's movement and, if the solution path is being
        shown, advances its reveal animation by one cell; then renders a
        frame if anything changed.

        Args:
            param (Any): Unused; required by the MiniLibX hook signature.
                Defaults to None.

        Returns:
            int: Always 0, as required by the MiniLibX hook signature.
        """
        current_time = time.perf_counter()

        self.last_frame_time = current_time
        if self.path_clear_queue:
            cell = self.path_clear_queue.pop(0)
            self._draw_background_tile(cell)
            self.path_visible_cells.discard(cell)
            self._render_needed = True
            self.draw_frame()
            return 0

        player_moved = self.player_entity.update()
        path_advanced = False
        if self.show_path and self.animated_steps < len(self.path_coords):
            self.frame_counter += 1
            if self.frame_counter >= 1:
                self.animated_steps += 1
                self.frame_counter = 0
                new_cell = self.path_coords[self.animated_steps - 1]
                self._draw_path_marker(new_cell)
                path_advanced = True

        self._render_needed = (self._render_needed or player_moved or
                               path_advanced)
        if self._render_needed:
            self.draw_frame()

        return 0

    def run(self) -> None:
        """Starts the MiniLibX event loop, blocking for the game's
        lifetime.

        If the loop ever returns, toggles 'show_path' twice (a no-op on
        the state but forces a couple of extra redraws) so the last
        frame on screen is left fully up to date.
        """
        self.m.mlx_loop(self.mlx_ptr)
        self.draw_frame()
        for _ in range(0, 2):
            self.show_path = not self.show_path
            if self.show_path:
                self.animated_steps = 0
                self.frame_counter = 0
                self.path_visible_cells.clear()
                self.path_clear_queue.clear()
            else:
                self.path_clear_queue = list(
                    self.path_coords[:self.animated_steps])
                self.path_visible_cells.clear()
            self._needs_full_redraw = True
            self._render_needed = True
            self.draw_frame()
