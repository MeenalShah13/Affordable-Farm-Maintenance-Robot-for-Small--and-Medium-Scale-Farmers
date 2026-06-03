"""
Boustrophedon controller with obstacle-aware grid and Dijkstra return path.

Key improvements:
  - Obstacle INFLATION: before path planning, obstacles are expanded by 1 cell
    in all directions so the robot doesn't clip tree edges.
  - Obstacle detection during RETURN: if the robot hits an obstacle while
    following the return path, it backs up, marks the cell, and replans.
  - Proper PNG grid map using PIL with color coding and return path overlay.
"""

from controller import Robot
import math
from typing import List, Tuple, Optional
from enum import IntEnum
import heapq
import copy

# ─────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


# ─────────────────────────────────────────────────────────────────────
# Field geometry
# ─────────────────────────────────────────────────────────────────────
FIELD_CENTER_X = 0.59
FIELD_CENTER_Y = 0.0
FIELD_W = 2.0
FIELD_H = 3.0

FIELD_MIN_X = FIELD_CENTER_X - FIELD_W / 2.0
FIELD_MAX_X = FIELD_CENTER_X + FIELD_W / 2.0
FIELD_MIN_Y = FIELD_CENTER_Y - FIELD_H / 2.0
FIELD_MAX_Y = FIELD_CENTER_Y + FIELD_H / 2.0

FIELD_MARGIN = 0.15

DOCK_X = -0.7
DOCK_Y = -1.36

# ─────────────────────────────────────────────────────────────────────
# Grid
# ─────────────────────────────────────────────────────────────────────
CELL_SIZE = 0.2
GRID_COLS = int(FIELD_W / CELL_SIZE)
GRID_ROWS = int(FIELD_H / CELL_SIZE)

# ─────────────────────────────────────────────────────────────────────
# Anchors
# ─────────────────────────────────────────────────────────────────────
A1_X, A1_Y = -0.50,  1.54
A2_X, A2_Y =  1.64, -1.57
A3_X, A3_Y =  1.59,  1.53
Z_ROBOT = 0.07
Z_A1 = 0.22
Z_A2 = 0.22
Z_A3 = 0.22

# ─────────────────────────────────────────────────────────────────────
# Drive
# ─────────────────────────────────────────────────────────────────────
WHEEL_RADIUS = 0.07
TRACK_WIDTH  = 0.18
MAX_WHEEL_W  = 8.0

V_BASE  = 0.10
V_CREEP = 0.03

CROSS_KP    = 1.8
MAX_CROSS_W = 1.8

HEADING_MIN_DIST   = 0.03
HEADING_FRESH_TIME = 0.8
KP_FRESH = 1.8
KP_STALE = 0.6

STRIP_WIDTH = CELL_SIZE
ARRIVE_TOL  = 0.05

# ─────────────────────────────────────────────────────────────────────
# Obstacle detection
# ─────────────────────────────────────────────────────────────────────
OBSTACLE_DEFLECT_THRESH = 400
OBSTACLE_AVOID_THRESH   = 200
OBSTACLE_CLEAR_THRESH   = 500

DEFLECT_W_BIAS = 1.2

AVOID_BACKUP_TIME = 0.5
AVOID_BACKUP_V    = -0.06
AVOID_CURVE_TIME  = 1.0
AVOID_CURVE_V     = 0.05
AVOID_CURVE_W     = 1.5

OBSTACLE_MARK_DIST = 0.15

# Inflation radius for path planning (in cells).
# 1 = obstacles expanded by 1 cell in each direction (3x3 block).
INFLATE_RADIUS = 1

DOCK_TOL      = 0.15
BATTERY_CAP_S = 180.0
CHARGE_TIME_S = 6.0


# ─────────────────────────────────────────────────────────────────────
def trilaterate(ax1, ay1, d1, ax2, ay2, d2, ax3, ay3, d3):
    A = 2 * (ax2 - ax1)
    B = 2 * (ay2 - ay1)
    C = d1**2 - d2**2 + ax2**2 - ax1**2 + ay2**2 - ay1**2
    D = 2 * (ax3 - ax1)
    E = 2 * (ay3 - ay1)
    F = d1**2 - d3**2 + ax3**2 - ax1**2 + ay3**2 - ay1**2
    denom = A * E - B * D
    if abs(denom) < 1e-9:
        return None
    return ((C * E - F * B) / denom,
            (A * F - D * C) / denom)


# ─────────────────────────────────────────────────────────────────────
# Grid
# ─────────────────────────────────────────────────────────────────────
class CellState(IntEnum):
    UNKNOWN  = 0
    VISITED  = 1
    OBSTACLE = -1


class Grid:
    def __init__(self):
        self.cells = [[CellState.UNKNOWN] * GRID_COLS for _ in range(GRID_ROWS)]
        self.path: List[Tuple[int, int]] = []

    @staticmethod
    def world_to_cell(wx, wy):
        col = int((wx - FIELD_MIN_X) / CELL_SIZE)
        row = int((wy - FIELD_MIN_Y) / CELL_SIZE)
        return col, row

    @staticmethod
    def cell_center(col, row):
        return (FIELD_MIN_X + (col + 0.5) * CELL_SIZE,
                FIELD_MIN_Y + (row + 0.5) * CELL_SIZE)

    def in_bounds(self, col, row):
        return 0 <= col < GRID_COLS and 0 <= row < GRID_ROWS

    def mark_visited(self, wx, wy):
        col, row = self.world_to_cell(wx, wy)
        if self.in_bounds(col, row):
            if self.cells[row][col] != CellState.OBSTACLE:
                self.cells[row][col] = CellState.VISITED
            cell = (col, row)
            if not self.path or self.path[-1] != cell:
                self.path.append(cell)

    def mark_obstacle(self, wx, wy):
        col, row = self.world_to_cell(wx, wy)
        if self.in_bounds(col, row):
            self.cells[row][col] = CellState.OBSTACLE

    def get(self, col, row):
        if self.in_bounds(col, row):
            return self.cells[row][col]
        return CellState.OBSTACLE

    def is_passable(self, col, row):
        return self.in_bounds(col, row) and self.cells[row][col] != CellState.OBSTACLE

    def coverage_fraction(self):
        visited = sum(1 for r in range(GRID_ROWS) for c in range(GRID_COLS)
                      if self.cells[r][c] == CellState.VISITED)
        return visited / (GRID_ROWS * GRID_COLS)

    def current_cell(self, wx, wy):
        col, row = self.world_to_cell(wx, wy)
        return clamp(col, 0, GRID_COLS - 1), clamp(row, 0, GRID_ROWS - 1)

    def find_next_free_col(self, row, start_col, direction):
        col = start_col
        while 0 <= col < GRID_COLS:
            if self.cells[row][col] != CellState.OBSTACLE:
                return col
            col += direction
        return None

    def row_has_free_cells(self, row):
        if not (0 <= row < GRID_ROWS):
            return False
        return any(self.cells[row][c] != CellState.OBSTACLE for c in range(GRID_COLS))

    def neighbors4(self, col, row):
        result = []
        for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nc, nr = col + dc, row + dr
            if self.is_passable(nc, nr):
                result.append((nc, nr))
        return result

    def nearest_passable_to_world(self, wx, wy):
        best = None
        best_dist = float('inf')
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if self.cells[r][c] != CellState.OBSTACLE:
                    cx, cy = self.cell_center(c, r)
                    d = math.hypot(cx - wx, cy - wy)
                    if d < best_dist:
                        best_dist = d
                        best = (c, r)
        return best

    def inflated_copy(self, radius: int = 1) -> 'Grid':
        """
        Return a copy of the grid where every obstacle cell has been
        expanded by `radius` cells in all directions.  The robot's
        physical size is ~0.2m, so inflating by 1 cell (0.2m) ensures
        Dijkstra paths keep 1 cell clearance from obstacles.
        """
        inflated = Grid()
        # Copy non-obstacle cells first
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                inflated.cells[r][c] = self.cells[r][c]

        # Expand obstacles
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if self.cells[r][c] == CellState.OBSTACLE:
                    for dr in range(-radius, radius + 1):
                        for dc in range(-radius, radius + 1):
                            nr, nc = r + dr, c + dc
                            if inflated.in_bounds(nc, nr):
                                inflated.cells[nr][nc] = CellState.OBSTACLE
        return inflated


# ─────────────────────────────────────────────────────────────────────
# Dijkstra
# ─────────────────────────────────────────────────────────────────────
def dijkstra(grid: Grid, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    if start == goal:
        return [start]
    if not grid.is_passable(start[0], start[1]):
        return None
    if not grid.is_passable(goal[0], goal[1]):
        return None

    counter = 0
    pq = [(0, counter, start)]
    dist = {start: 0}
    parent = {}

    while pq:
        d, _, u = heapq.heappop(pq)
        if u == goal:
            break
        if d > dist.get(u, float('inf')):
            continue
        for v in grid.neighbors4(u[0], u[1]):
            nd = d + 1
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                parent[v] = u
                counter += 1
                heapq.heappush(pq, (nd, counter, v))

    if goal not in parent and goal != start:
        return None

    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    path.reverse()
    return path


# ─────────────────────────────────────────────────────────────────────
# PNG grid map
# ─────────────────────────────────────────────────────────────────────
def build_grid_png(grid: Grid, return_path_cells=None, filename="grid_map.png"):
    """
    Save a color PNG of the grid.
      - Gray   = unknown
      - Green  = visited
      - Black  = obstacle (original)
      - Dark red = inflated obstacle margin
      - Blue   = return path
      - Cyan   = start/end of return path
    """
    try:
        from PIL import Image
    except ImportError:
        print("  !! PIL not available, skipping grid PNG")
        return

    # Colors (R, G, B)
    COLOR_UNKNOWN  = (160, 160, 160) # light gray
    COLOR_VISITED  = (100, 200, 100) # light green
    COLOR_OBSTACLE = (30, 30, 30) # black
    # COLOR_PATH     = (60, 120, 255) # blue
    # COLOR_ENDPOINT = (0, 255, 255) # cyan

    scale = 20  # pixels per cell
    img_w = GRID_COLS * scale
    img_h = GRID_ROWS * scale
    img = Image.new("RGB", (img_w, img_h), COLOR_UNKNOWN)

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            state = grid.cells[r][c]
            if state == CellState.OBSTACLE:
                color = COLOR_OBSTACLE
            elif state == CellState.VISITED:
                color = COLOR_VISITED
            else:
                color = COLOR_UNKNOWN

            # Row 0 = bottom of field = bottom of image
            # PIL y=0 is top, so flip: img_row = GRID_ROWS - 1 - r
            img_r = GRID_ROWS - 1 - r
            for py in range(img_r * scale, (img_r + 1) * scale):
                for px in range(c * scale, (c + 1) * scale):
                    img.putpixel((px, py), color)

    # # Draw return path
    # if return_path_cells:
    #     for i, (c, r) in enumerate(return_path_cells):
    #         if not (0 <= c < GRID_COLS and 0 <= r < GRID_ROWS):
    #             continue
    #         is_endpoint = (i == 0 or i == len(return_path_cells) - 1)
    #         color = COLOR_ENDPOINT if is_endpoint else COLOR_PATH
    #         img_r = GRID_ROWS - 1 - r
    #         # Draw a smaller square in the center of the cell
    #         margin = scale // 4
    #         for py in range(img_r * scale + margin, (img_r + 1) * scale - margin):
    #             for px in range(c * scale + margin, (c + 1) * scale - margin):
    #                 img.putpixel((px, py), color)

    # Draw grid lines
    for r in range(GRID_ROWS + 1):
        img_y = (GRID_ROWS - r) * scale
        if 0 <= img_y < img_h:
            for px in range(img_w):
                img.putpixel((px, min(img_y, img_h - 1)), (80, 80, 80))
    for c in range(GRID_COLS + 1):
        img_x = c * scale
        if 0 <= img_x < img_w:
            for py in range(img_h):
                img.putpixel((min(img_x, img_w - 1), py), (80, 80, 80))

    img.save(filename)
    print(f"  >> Grid map saved to {filename} ({img_w}x{img_h})")


# ─────────────────────────────────────────────────────────────────────
# Mowing state
# ─────────────────────────────────────────────────────────────────────
class MowPhase(IntEnum):
    X_PASS  = 0
    Y_SHIFT = 1


class MowState:
    def __init__(self):
        self.row = 0
        self.phase = MowPhase.X_PASS
        self.total_rows = GRID_ROWS
        self.going_east = True
        self.x_waypoint: Optional[float] = None

    @property
    def target_y(self):
        return FIELD_MIN_Y + (self.row + 0.5) * CELL_SIZE

    @property
    def x_end(self):
        m = FIELD_MARGIN
        return (FIELD_MAX_X - m) if self.going_east else (FIELD_MIN_X + m)

    @property
    def x_direction(self):
        return +1 if self.going_east else -1

    @property
    def next_row_y(self):
        return FIELD_MIN_Y + (self.row + 1 + 0.5) * CELL_SIZE

    @property
    def shift_hold_x(self):
        m = FIELD_MARGIN
        return (FIELD_MAX_X - m) if self.going_east else (FIELD_MIN_X + m)

    def advance(self):
        self.x_waypoint = None
        if self.phase == MowPhase.X_PASS:
            if self.row + 1 >= self.total_rows:
                return False
            self.phase = MowPhase.Y_SHIFT
        else:
            self.row += 1
            self.going_east = not self.going_east
            self.phase = MowPhase.X_PASS
        return True

    def skip_to_row(self, new_row):
        self.x_waypoint = None
        self.row = new_row
        self.going_east = (new_row % 2 == 0)
        self.phase = MowPhase.X_PASS

    @property
    def is_complete(self):
        return self.row >= self.total_rows


# ─────────────────────────────────────────────────────────────────────
# Avoidance state
# ─────────────────────────────────────────────────────────────────────
class AvoidPhase(IntEnum):
    NONE   = 0
    BACKUP = 1
    CURVE  = 2


class AvoidState:
    def __init__(self):
        self.phase = AvoidPhase.NONE
        self.phase_end = 0.0
        self.curve_sign = 1.0


# ─────────────────────────────────────────────────────────────────────
# Robot
# ─────────────────────────────────────────────────────────────────────
class SimpleRobot:
    def __init__(self):
        self.robot = Robot()
        self.dt_ms = int(self.robot.getBasicTimeStep())

        self.motors = []
        for name in ("wheel1", "wheel2", "wheel3", "wheel4"):
            m = self.robot.getDevice(name)
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
            self.motors.append(m)
        self.mL1, self.mR1, self.mL2, self.mR2 = self.motors

        for name in ("wheel1_ps", "wheel2_ps", "wheel3_ps", "wheel4_ps"):
            self.robot.getDevice(name).enable(self.dt_ms)

        self.dsL = self.robot.getDevice("ds_left")
        self.dsR = self.robot.getDevice("ds_right")
        self.dsL.enable(self.dt_ms)
        self.dsR.enable(self.dt_ms)

        self.robot.getDevice("camera").enable(self.dt_ms)

        self.receiver = self.robot.getDevice("receiver")
        self.receiver.enable(self.dt_ms)
        self.receiver.setChannel(100)
        self.dists = {
            "anchor_radio1": None,
            "anchor_radio2": None,
            "anchor_radio3": None,
        }

        self.x = DOCK_X
        self.y = DOCK_Y
        self.th = 0.0
        self._head_anchor_x = self.x
        self._head_anchor_y = self.y
        self._head_last_t = 0.0
        self.fix_count = 0

        self.batt = BATTERY_CAP_S
        self.charge_until = 0.0

        self.return_path: Optional[List[Tuple[float, float]]] = None
        self.return_path_cells: Optional[List[Tuple[int, int]]] = None
        self.return_idx = 0

        self.grid = Grid()
        self.mow = MowState()
        self.avoid = AvoidState()

        self.state = "INIT"

    # ── helpers ──────────────────────────────────────────────────────
    def time(self):
        return self.robot.getTime()

    def stop(self):
        for m in self.motors:
            m.setVelocity(0.0)

    def set_vw(self, v, w):
        vL = v - 0.5 * w * TRACK_WIDTH
        vR = v + 0.5 * w * TRACK_WIDTH
        wL = clamp(vL / WHEEL_RADIUS, -MAX_WHEEL_W, MAX_WHEEL_W)
        wR = clamp(vR / WHEEL_RADIUS, -MAX_WHEEL_W, MAX_WHEEL_W)
        self.mL1.setVelocity(wL)
        self.mL2.setVelocity(wL)
        self.mR1.setVelocity(wR)
        self.mR2.setVelocity(wR)

    def _heading_fresh(self):
        return (self.time() - self._head_last_t) < HEADING_FRESH_TIME

    def _kp(self):
        return KP_FRESH if self._heading_fresh() else KP_STALE

    def _sL(self):
        return self.dsL.getValue()

    def _sR(self):
        return self.dsR.getValue()

    def _needs_hard_avoid(self):
        return self._sL() < OBSTACLE_AVOID_THRESH or self._sR() < OBSTACLE_AVOID_THRESH

    def _is_clear(self):
        return self._sL() > OBSTACLE_CLEAR_THRESH and self._sR() > OBSTACLE_CLEAR_THRESH

    def _deflect_bias(self):
        sL, sR = self._sL(), self._sR()
        if sL >= OBSTACLE_DEFLECT_THRESH and sR >= OBSTACLE_DEFLECT_THRESH:
            return 0.0
        return -DEFLECT_W_BIAS if sL < sR else +DEFLECT_W_BIAS

    # ── UWB ──────────────────────────────────────────────────────────
    def _flat_dist(self, d3d, za, zr):
        return math.sqrt(max(0.0, d3d**2 - (za - zr)**2))

    def update_position(self):
        while self.receiver.getQueueLength() > 0:
            data = self.receiver.getString()
            s = self.receiver.getSignalStrength()
            if s > 0 and data in self.dists:
                self.dists[data] = 1.0 / math.sqrt(s)
            self.receiver.nextPacket()

        if not all(v is not None for v in self.dists.values()):
            return False

        d1 = self._flat_dist(self.dists["anchor_radio1"], Z_A1, Z_ROBOT)
        d2 = self._flat_dist(self.dists["anchor_radio2"], Z_A2, Z_ROBOT)
        d3 = self._flat_dist(self.dists["anchor_radio3"], Z_A3, Z_ROBOT)
        r = trilaterate(A1_X, A1_Y, d1, A2_X, A2_Y, d2, A3_X, A3_Y, d3)
        if r is None:
            return False

        nx, ny = r
        alpha = 0.35
        if self.fix_count < 3:
            self.x, self.y = nx, ny
        else:
            self.x = self.x * (1 - alpha) + nx * alpha
            self.y = self.y * (1 - alpha) + ny * alpha

        dx = self.x - self._head_anchor_x
        dy = self.y - self._head_anchor_y
        if math.hypot(dx, dy) > HEADING_MIN_DIST:
            self.th = math.atan2(dy, dx)
            self._head_anchor_x = self.x
            self._head_anchor_y = self.y
            self._head_last_t = self.time()

        self.fix_count += 1
        return True

    # ── steering ─────────────────────────────────────────────────────
    def steer_toward(self, desired_heading, extra_w=0.0):
        err = wrap_pi(desired_heading - self.th)
        w = clamp(self._kp() * err + extra_w, -MAX_CROSS_W, MAX_CROSS_W)
        alignment = math.cos(err)
        v = V_CREEP + (V_BASE - V_CREEP) * clamp(alignment, 0.0, 1.0)
        self.set_vw(v, w)

    def steer_to_point(self, tx, ty, tol=0.15):
        dx, dy = tx - self.x, ty - self.y
        if math.hypot(dx, dy) < tol:
            self.stop()
            return True
        self.steer_toward(math.atan2(dy, dx))
        return False

    # ── obstacle marking ─────────────────────────────────────────────
    def mark_obstacle_ahead(self):
        sL, sR = self._sL(), self._sR()
        if sL < OBSTACLE_DEFLECT_THRESH:
            a = self.th + 0.3
            self.grid.mark_obstacle(self.x + OBSTACLE_MARK_DIST * math.cos(a),
                                    self.y + OBSTACLE_MARK_DIST * math.sin(a))
        if sR < OBSTACLE_DEFLECT_THRESH:
            a = self.th - 0.3
            self.grid.mark_obstacle(self.x + OBSTACLE_MARK_DIST * math.cos(a),
                                    self.y + OBSTACLE_MARK_DIST * math.sin(a))
        self.grid.mark_obstacle(self.x + OBSTACLE_MARK_DIST * math.cos(self.th),
                                self.y + OBSTACLE_MARK_DIST * math.sin(self.th))

    # ── hard avoidance (used in MOWING and RETURN) ───────────────────
    def begin_hard_avoid(self):
        self.mark_obstacle_ahead()
        sL, sR = self._sL(), self._sR()
        self.avoid.curve_sign = -1.0 if sL < sR else +1.0
        self.avoid.phase = AvoidPhase.BACKUP
        self.avoid.phase_end = self.time() + AVOID_BACKUP_TIME

    def run_hard_avoid(self) -> bool:
        t = self.time()
        if self.avoid.phase == AvoidPhase.BACKUP:
            self.set_vw(AVOID_BACKUP_V, 0.0)
            if t >= self.avoid.phase_end:
                self.avoid.phase = AvoidPhase.CURVE
                self.avoid.phase_end = t + AVOID_CURVE_TIME
            return False
        if self.avoid.phase == AvoidPhase.CURVE:
            self.set_vw(AVOID_CURVE_V, AVOID_CURVE_W * self.avoid.curve_sign)
            if t >= self.avoid.phase_end:
                if self._is_clear():
                    self.avoid.phase = AvoidPhase.NONE
                    return True
                else:
                    self.avoid.phase_end = t + AVOID_CURVE_TIME * 0.5
            return False
        self.avoid.phase = AvoidPhase.NONE
        return True

    # ── mowing replanning ────────────────────────────────────────────
    def _replan_after_mow_avoid(self):
        cur_col, _ = self.grid.current_cell(self.x, self.y)
        direction = self.mow.x_direction
        if self.mow.phase == MowPhase.X_PASS:
            search_col = cur_col + direction
            free_col = self.grid.find_next_free_col(self.mow.row, search_col, direction)
            if free_col is not None:
                wx, _ = Grid.cell_center(free_col, self.mow.row)
                self.mow.x_waypoint = wx
            else:
                self._advance_to_next_free_row()

    def _advance_to_next_free_row(self):
        for r in range(self.mow.row + 1, GRID_ROWS):
            if self.grid.row_has_free_cells(r):
                self.mow.skip_to_row(r)
                return
        self.mow.row = self.mow.total_rows

    def _check_obstacle_ahead_on_grid(self):
        cur_col, _ = self.grid.current_cell(self.x, self.y)
        mow_row = self.mow.row
        direction = self.mow.x_direction
        next_col = cur_col + direction
        if not self.grid.in_bounds(next_col, mow_row):
            return None
        if self.grid.get(next_col, mow_row) == CellState.OBSTACLE:
            free_col = self.grid.find_next_free_col(mow_row, next_col + direction, direction)
            if free_col is not None:
                wx, _ = Grid.cell_center(free_col, mow_row)
                return wx
        return None

    # ── mowing segments ──────────────────────────────────────────────
    def run_x_pass(self):
        target_y = self.mow.target_y
        y_err = target_y - self.y
        x_target = self.mow.x_waypoint if self.mow.x_waypoint is not None else self.mow.x_end
        grid_skip = self._check_obstacle_ahead_on_grid()
        if grid_skip is not None and self.mow.x_waypoint is None:
            self.mow.x_waypoint = grid_skip
            x_target = grid_skip

        desired_vy = clamp(CROSS_KP * y_err, -0.5, 0.5)
        desired_vx = 1.0 * self.mow.x_direction
        desired_heading = math.atan2(desired_vy, desired_vx)
        bias = self._deflect_bias()
        if bias != 0.0:
            self.mark_obstacle_ahead()
        self.steer_toward(desired_heading, extra_w=bias)

        if self.mow.going_east:
            arrived = self.x >= x_target - ARRIVE_TOL
        else:
            arrived = self.x <= x_target + ARRIVE_TOL

        if arrived and self.mow.x_waypoint is not None:
            self.mow.x_waypoint = None
            next_skip = self._check_obstacle_ahead_on_grid()
            if next_skip is not None:
                self.mow.x_waypoint = next_skip
            return False
        return arrived

    def run_y_shift(self):
        hold_x = self.mow.shift_hold_x
        target_y = self.mow.next_row_y
        x_err = hold_x - self.x
        desired_vx = clamp(CROSS_KP * x_err, -0.5, 0.5)
        desired_vy = 1.0
        desired_heading = math.atan2(desired_vy, desired_vx)
        bias = self._deflect_bias()
        if bias != 0.0:
            self.mark_obstacle_ahead()
        self.steer_toward(desired_heading, extra_w=bias)
        return self.y >= target_y - ARRIVE_TOL

    # ── return path planning ─────────────────────────────────────────
    def plan_return_path(self):
        """
        Plan return using Dijkstra on an INFLATED copy of the grid.
        Inflation expands obstacles by INFLATE_RADIUS cells so the
        path keeps clearance from trees the robot would clip.
        """
        inflated = self.grid.inflated_copy(INFLATE_RADIUS)
        start_cell = inflated.current_cell(self.x, self.y)

        # If start is inside inflated obstacle, find nearest passable
        if not inflated.is_passable(start_cell[0], start_cell[1]):
            start_cell = inflated.nearest_passable_to_world(self.x, self.y)
            if start_cell is None:
                print("  !! No passable start — direct drive")
                self.return_path = [(DOCK_X, DOCK_Y)]
                self.return_path_cells = []
                self.return_idx = 0
                return

        exit_cell = inflated.nearest_passable_to_world(DOCK_X, DOCK_Y)
        if exit_cell is None:
            print("  !! No passable exit — direct drive")
            self.return_path = [(DOCK_X, DOCK_Y)]
            self.return_path_cells = []
            self.return_idx = 0
            return

        print(f"  >> Planning return: {start_cell} → {exit_cell} → dock (inflated grid)")

        if start_cell == exit_cell:
            grid_path = [start_cell]
        else:
            grid_path = dijkstra(inflated, start_cell, exit_cell)

        if grid_path is None:
            # Fallback: try without inflation
            print("  !! Inflated path failed — trying raw grid")
            exit_cell = self.grid.nearest_passable_to_world(DOCK_X, DOCK_Y)
            start_cell = self.grid.current_cell(self.x, self.y)
            if exit_cell:
                grid_path = dijkstra(self.grid, start_cell, exit_cell)
            if grid_path is None:
                print("  !! No path found — direct drive")
                self.return_path = [(DOCK_X, DOCK_Y)]
                self.return_path_cells = []
                self.return_idx = 0
                return

        self.return_path_cells = list(grid_path)
        world_path = [Grid.cell_center(c, r) for c, r in grid_path]
        world_path.append((DOCK_X, DOCK_Y))

        self.return_path = world_path
        self.return_idx = 0
        print(f"  >> Return path: {len(world_path)} waypoints")

        # Save map PNG with return path
        build_grid_png(self.grid, self.return_path_cells, "grid_map.png")

    def run_return(self) -> bool:
        """Follow return path. Returns True when dock reached."""
        if self.return_path is None or self.return_idx >= len(self.return_path):
            return self.steer_to_point(DOCK_X, DOCK_Y, tol=DOCK_TOL)

        wx, wy = self.return_path[self.return_idx]
        is_last = (self.return_idx == len(self.return_path) - 1)
        tol = DOCK_TOL if is_last else CELL_SIZE * 0.6

        arrived = self.steer_to_point(wx, wy, tol=tol)
        if arrived:
            self.return_idx += 1
            if self.return_idx >= len(self.return_path):
                return True
        return False

    def replan_return(self):
        """Re-plan return path after hitting obstacle during RETURN."""
        print("  !! Return obstacle hit — replanning")
        self.plan_return_path()

    # ── main loop ────────────────────────────────────────────────────
    def run(self):
        last_t = self.time()
        start_x = FIELD_MIN_X + FIELD_MARGIN
        start_y = FIELD_MIN_Y + FIELD_MARGIN

        while self.robot.step(self.dt_ms) != -1:
            t = self.time()
            dt = max(1e-3, t - last_t)
            last_t = t

            self.update_position()

            if (FIELD_MIN_X <= self.x <= FIELD_MAX_X and
                    FIELD_MIN_Y <= self.y <= FIELD_MAX_Y):
                self.grid.mark_visited(self.x, self.y)

            col, row = self.grid.current_cell(self.x, self.y)
            fresh = "F" if self._heading_fresh() else "S"
            phase_str = ""
            if self.state == "MOWING":
                if self.avoid.phase != AvoidPhase.NONE:
                    phase_str = f"AVOID:{self.avoid.phase.name}"
                else:
                    phase_str = self.mow.phase.name
                    if self.mow.x_waypoint is not None:
                        phase_str += f" wp={self.mow.x_waypoint:.2f}"
            elif self.state == "RETURN":
                if self.avoid.phase != AvoidPhase.NONE:
                    phase_str = f"AVOID:{self.avoid.phase.name}"
                elif self.return_path:
                    phase_str = f"wp {self.return_idx}/{len(self.return_path)}"
            print(f"[{t:6.1f}] {self.state:12s}  ({self.x:.2f},{self.y:.2f})  "
                  f"th={math.degrees(self.th):+6.1f}[{fresh}]  "
                  f"cell=({col},{row})  {phase_str}  "
                  f"sL={self._sL():.0f} sR={self._sR():.0f}  "
                  f"coverage={self.grid.coverage_fraction():.0%}")

            # ── INIT ──
            if self.state == "INIT":
                self.stop()
                if self.fix_count >= 5:
                    self.state = "GO_TO_START"
                continue

            # ── GO_TO_START ──
            if self.state == "GO_TO_START":
                self.batt -= dt
                if self.steer_to_point(start_x, start_y):
                    self.state = "MOWING"
                continue

            # ── MOWING ──
            if self.state == "MOWING":
                self.batt -= dt

                if self.mow.is_complete:
                    print(f"*** Mowing complete ({self.grid.coverage_fraction():.0%}) ***")
                    self.plan_return_path()
                    self.state = "RETURN"
                    continue

                # Hard avoidance in progress
                if self.avoid.phase != AvoidPhase.NONE:
                    done = self.run_hard_avoid()
                    if done:
                        self._replan_after_mow_avoid()
                    else:
                        continue

                if self._needs_hard_avoid():
                    self.begin_hard_avoid()
                    continue

                if self.mow.phase == MowPhase.X_PASS:
                    cur_col, _ = self.grid.current_cell(self.x, self.y)
                    next_col = cur_col + self.mow.x_direction
                    free = self.grid.find_next_free_col(self.mow.row, next_col, self.mow.x_direction)
                    at_edge = (self.mow.going_east and self.x >= self.mow.x_end - ARRIVE_TOL) or \
                              (not self.mow.going_east and self.x <= self.mow.x_end + ARRIVE_TOL)
                    if free is None and not at_edge:
                        still_going = self.mow.advance()
                        if not still_going:
                            self._advance_to_next_free_row()
                        continue
                    done = self.run_x_pass()
                else:
                    done = self.run_y_shift()

                if done:
                    print(f"  >> finished {self.mow.phase.name} at row {self.mow.row}")
                    still_going = self.mow.advance()
                    if not still_going:
                        print(f"*** Mowing complete ({self.grid.coverage_fraction():.0%}) ***")
                        self.plan_return_path()
                        self.state = "RETURN"
                continue

            # ── RETURN (with obstacle detection!) ──
            if self.state == "RETURN":
                self.batt -= dt

                # Hard avoidance during return
                if self.avoid.phase != AvoidPhase.NONE:
                    done = self.run_hard_avoid()
                    if done:
                        # Obstacle cleared — replan the return path
                        self.replan_return()
                    else:
                        continue

                # Check for new obstacles during return
                if self._needs_hard_avoid():
                    self.mark_obstacle_ahead()
                    self.begin_hard_avoid()
                    continue

                if self.run_return():
                    print("*** At dock ***")
                    self.state = "CHARGE"
                    self.charge_until = self.time() + CHARGE_TIME_S
                    self.stop()
                continue

            # ── CHARGE ──
            if self.state == "CHARGE":
                self.stop()
                if self.time() >= self.charge_until:
                    self.batt = BATTERY_CAP_S
                    self.state = "DONE"
                continue

            # ── DONE ──
            if self.state == "DONE":
                self.stop()


if __name__ == "__main__":
    SimpleRobot().run()