"""
navigation/grid.py + navigation/planner.py — combined for clarity.

Grid          : Occupancy grid with serialisation to/from JSON (persisted
                across charging cycles).
dijkstra()    : 4-connected shortest-path planner.
MowState      : Boustrophedon (lawnmower) coverage planner.
AvoidState    : Two-phase backup-and-curve obstacle avoidance FSM.
ReturnPlanner : Plans and follows Dijkstra path back to dock.
StuckRecovery : Detects and escapes stuck conditions (encoder + IMU).
"""

from __future__ import annotations

import heapq
import json
import math
import logging
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)


def _clamp(x, lo, hi): return max(lo, min(hi, x))
def _wrap_pi(a):
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


# ─────────────────────────────────────────────────────────────────────
# Occupancy Grid
# ─────────────────────────────────────────────────────────────────────

class CellState(IntEnum):
    UNKNOWN  =  0
    VISITED  =  1
    OBSTACLE = -1


class Grid:
    """
    2-D occupancy grid.  Rows = south→north, cols = west→east.
    Serialised to JSON so the grid survives the robot charging cycle.

    Obstacle decay
    ──────────────
    Each obstacle cell has a timestamp recording when it was last marked.
    decay_obstacles(now) reverts cells whose timestamp is older than
    OBSTACLE_DECAY_S back to UNKNOWN, so the robot will re-visit and
    re-check whether the obstacle is still present on its next pass.

    Re-marking a cell (the IR sensor sees the obstacle again) resets
    its clock, so persistent obstacles stay permanent as long as they
    keep being detected.  Temporary obstacles (a person walking through,
    a moving plant, a sensor glitch) fade out automatically.
    """

    def __init__(self):
        from config import GRID_ROWS, GRID_COLS
        self.rows = GRID_ROWS
        self.cols = GRID_COLS
        self.cells: List[List[CellState]] = [
            [CellState.UNKNOWN] * GRID_COLS for _ in range(GRID_ROWS)
        ]
        # Maps (col, row) → monotonic time when obstacle was last marked.
        # Only obstacle cells have an entry; cleared when the cell decays.
        self._obstacle_times: dict = {}

    # ── Coordinate conversion ─────────────────────────────────────────

    @staticmethod
    def world_to_cell(wx: float, wy: float) -> Tuple[int, int]:
        from config import FIELD_MIN_X, FIELD_MIN_Y, CELL_SIZE
        col = int((wx - FIELD_MIN_X) / CELL_SIZE)
        row = int((wy - FIELD_MIN_Y) / CELL_SIZE)
        return col, row

    @staticmethod
    def cell_centre(col: int, row: int) -> Tuple[float, float]:
        from config import FIELD_MIN_X, FIELD_MIN_Y, CELL_SIZE
        return (FIELD_MIN_X + (col + 0.5) * CELL_SIZE,
                FIELD_MIN_Y + (row + 0.5) * CELL_SIZE)

    def in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.cols and 0 <= row < self.rows

    def current_cell(self, wx: float, wy: float) -> Tuple[int, int]:
        col, row = self.world_to_cell(wx, wy)
        return _clamp(col, 0, self.cols - 1), _clamp(row, 0, self.rows - 1)

    # ── State mutation ────────────────────────────────────────────────

    def mark_visited(self, wx: float, wy: float) -> None:
        col, row = self.world_to_cell(wx, wy)
        if self.in_bounds(col, row) and self.cells[row][col] != CellState.OBSTACLE:
            self.cells[row][col] = CellState.VISITED

    def mark_obstacle(self, wx: float, wy: float) -> None:
        """
        Mark a world-coordinate position as an obstacle and record the
        current time.  Re-marking resets the decay clock so persistent
        obstacles remain as long as the sensor keeps detecting them.
        """
        import time as _time
        col, row = self.world_to_cell(wx, wy)
        if self.in_bounds(col, row):
            self.cells[row][col] = CellState.OBSTACLE
            self._obstacle_times[(col, row)] = _time.monotonic()

    # ── Obstacle decay ────────────────────────────────────────────────

    def decay_obstacles(self, now: float) -> int:
        """
        Revert obstacle cells that have not been re-confirmed within
        OBSTACLE_DECAY_S seconds back to UNKNOWN.

        Parameters
        ----------
        now : float
            Current monotonic time (time.monotonic()).

        Returns
        -------
        int — number of cells that decayed this call (useful for logging).

        How it works
        ────────────
        When the robot re-mows a decayed cell, the IR sensors will fire
        again if the obstacle is still there (re-marking it with a fresh
        timestamp), or they won't fire if the obstacle is gone (the cell
        stays UNKNOWN and gets marked VISITED on the next pass).
        """
        from config import OBSTACLE_DECAY_S

        decayed = 0
        expired = [
            cell for cell, t in self._obstacle_times.items()
            if (now - t) >= OBSTACLE_DECAY_S
        ]
        for (col, row) in expired:
            if self.in_bounds(col, row):
                self.cells[row][col] = CellState.UNKNOWN
                log.debug("Obstacle decayed at cell (%d,%d) after %.0fs",
                          col, row, now - self._obstacle_times[(col, row)])
            del self._obstacle_times[(col, row)]
            decayed += 1

        if decayed:
            log.info("Obstacle decay: %d cell(s) reverted to UNKNOWN "
                     "(%d obstacle(s) remaining)",
                     decayed, len(self._obstacle_times))
        return decayed

    def get(self, col: int, row: int) -> CellState:
        if self.in_bounds(col, row):
            return self.cells[row][col]
        return CellState.OBSTACLE

    def is_passable(self, col: int, row: int) -> bool:
        return self.in_bounds(col, row) and self.cells[row][col] != CellState.OBSTACLE

    # ── Queries ───────────────────────────────────────────────────────

    def coverage_fraction(self) -> float:
        visited = sum(self.cells[r][c] == CellState.VISITED
                      for r in range(self.rows) for c in range(self.cols))
        return visited / (self.rows * self.cols)

    def find_next_free_col(self, row: int, start_col: int, direction: int) -> Optional[int]:
        if not (0 <= row < self.rows):
            return None
        col = start_col
        while 0 <= col < self.cols:
            if self.cells[row][col] != CellState.OBSTACLE:
                return col
            col += direction
        return None

    def row_has_free_cells(self, row: int) -> bool:
        return (0 <= row < self.rows and
                any(self.cells[row][c] != CellState.OBSTACLE for c in range(self.cols)))

    def neighbours4(self, col: int, row: int) -> List[Tuple[int, int]]:
        result = []
        for dc, dr in [(1,0),(-1,0),(0,1),(0,-1)]:
            nc, nr = col+dc, row+dr
            if self.is_passable(nc, nr):
                result.append((nc, nr))
        return result

    def nearest_passable(self, wx: float, wy: float) -> Optional[Tuple[int, int]]:
        best, best_d = None, float("inf")
        for r in range(self.rows):
            for c in range(self.cols):
                if self.cells[r][c] != CellState.OBSTACLE:
                    cx, cy = self.cell_centre(c, r)
                    d = math.hypot(cx-wx, cy-wy)
                    if d < best_d:
                        best_d = d; best = (c, r)
        return best

    def inflated(self, radius: int = 1) -> "Grid":
        from config import GRID_ROWS, GRID_COLS
        g = Grid()
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                g.cells[r][c] = self.cells[r][c]
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if self.cells[r][c] == CellState.OBSTACLE:
                    for dr in range(-radius, radius+1):
                        for dc in range(-radius, radius+1):
                            nr, nc = r+dr, c+dc
                            if g.in_bounds(nc, nr):
                                g.cells[nr][nc] = CellState.OBSTACLE
        return g

    # ── Serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Serialise grid to a JSON-safe dict.

        obstacle_times is stored as ages (seconds since last mark) rather
        than raw monotonic timestamps, so the values remain meaningful
        across process restarts and can be safely stored in Firestore.
        """
        import time as _time
        from config import CELL_SIZE
        now = _time.monotonic()
        return {
            "rows":      self.rows,
            "cols":      self.cols,
            "cell_size": CELL_SIZE,
            "cells":     [[int(self.cells[r][c]) for c in range(self.cols)]
                           for r in range(self.rows)],
            # Age = seconds elapsed since the obstacle was last confirmed.
            # Stored as string keys "col_row" to avoid Firestore/JSON issues.
            "obstacle_ages": {
                f"{col}_{row}": round(now - t, 2)
                for (col, row), t in self._obstacle_times.items()
            },
        }

    def save(self, path: str) -> None:
        import time as _time
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Resolve ages at save time so the dict comprehension has access to now
        now = _time.monotonic()
        d = {
            "rows":      self.rows,
            "cols":      self.cols,
            "cell_size": __import__("config").CELL_SIZE,
            "cells":     [[int(self.cells[r][c]) for c in range(self.cols)]
                           for r in range(self.rows)],
            "obstacle_ages": {
                f"{col}_{row}": round(now - t, 2)
                for (col, row), t in self._obstacle_times.items()
            },
        }
        with open(path, "w") as f:
            json.dump(d, f)
        log.info("Grid saved to %s (%d obstacles, %d with timestamps)",
                 path, sum(self.cells[r][c] == CellState.OBSTACLE
                           for r in range(self.rows) for c in range(self.cols)),
                 len(self._obstacle_times))

    def load(self, path: str) -> bool:
        """Load grid and obstacle timestamps from JSON. Returns True on success."""
        import time as _time
        try:
            with open(path) as f:
                d = json.load(f)

            # Restore cell states
            for r in range(self.rows):
                for c in range(self.cols):
                    self.cells[r][c] = CellState(d["cells"][r][c])

            # Restore obstacle timestamps from stored ages.
            # age = seconds since the obstacle was last marked.
            # Reconstruct as now - age so decay counting continues correctly.
            now = _time.monotonic()
            self._obstacle_times = {}
            for key, age in d.get("obstacle_ages", {}).items():
                try:
                    col_s, row_s = key.split("_")
                    col, row = int(col_s), int(row_s)
                    if self.in_bounds(col, row):
                        self._obstacle_times[(col, row)] = now - float(age)
                except (ValueError, TypeError):
                    pass

            log.info("Grid loaded from %s (%.0f%% coverage, %d obstacles, "
                     "%d with timestamps)",
                     path, self.coverage_fraction() * 100,
                     sum(self.cells[r][c] == CellState.OBSTACLE
                         for r in range(self.rows) for c in range(self.cols)),
                     len(self._obstacle_times))
            return True
        except Exception as e:
            log.warning("Could not load grid from %s: %s", path, e)
            return False


# ─────────────────────────────────────────────────────────────────────
# Dijkstra
# ─────────────────────────────────────────────────────────────────────

def dijkstra(grid: Grid, start: Tuple[int, int],
             goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    if start == goal:
        return [start]
    if not grid.is_passable(*start) or not grid.is_passable(*goal):
        return None
    counter = 0
    pq = [(0, counter, start)]
    dist: dict = {start: 0}
    parent: dict = {}
    while pq:
        d, _, u = heapq.heappop(pq)
        if u == goal: break
        if d > dist.get(u, float("inf")): continue
        for v in grid.neighbours4(*u):
            nd = d + 1
            if nd < dist.get(v, float("inf")):
                dist[v] = nd; parent[v] = u
                counter += 1
                heapq.heappush(pq, (nd, counter, v))
    if goal not in parent:
        return None
    path = []
    cur = goal
    while cur != start:
        path.append(cur); cur = parent[cur]
    path.append(start)
    path.reverse()
    return path


# ─────────────────────────────────────────────────────────────────────
# Mowing Planner  (boustrophedon)
# ─────────────────────────────────────────────────────────────────────

class MowPhase(IntEnum):
    X_PASS  = 0
    Y_SHIFT = 1


class MowState:
    """Snake-pattern mowing planner."""

    def __init__(self):
        from config import GRID_ROWS
        self.row:         int       = 0
        self.phase:       MowPhase  = MowPhase.X_PASS
        self.going_east:  bool      = True
        self.x_waypoint:  Optional[float] = None
        self._total_rows: int       = GRID_ROWS

    @property
    def target_y(self) -> float:
        from config import FIELD_MIN_Y, CELL_SIZE
        return FIELD_MIN_Y + (self.row + 0.5) * CELL_SIZE

    @property
    def x_end(self) -> float:
        from config import FIELD_MAX_X, FIELD_MIN_X, FIELD_MARGIN
        return (FIELD_MAX_X - FIELD_MARGIN) if self.going_east else (FIELD_MIN_X + FIELD_MARGIN)

    @property
    def x_direction(self) -> int:
        return +1 if self.going_east else -1

    @property
    def next_row_y(self) -> float:
        from config import FIELD_MIN_Y, CELL_SIZE
        return FIELD_MIN_Y + (self.row + 1 + 0.5) * CELL_SIZE

    @property
    def shift_hold_x(self) -> float:
        from config import FIELD_MAX_X, FIELD_MIN_X, FIELD_MARGIN
        return (FIELD_MAX_X - FIELD_MARGIN) if self.going_east else (FIELD_MIN_X + FIELD_MARGIN)

    @property
    def x_pass_num(self) -> int:
        """1-based X-pass index (== row + 1).  Total passes == _total_rows."""
        return self.row + 1

    @property
    def y_shift_num(self) -> int:
        """1-based Y-shift index (shift N moves from row N-1 → row N).
        Total Y-shifts == _total_rows - 1."""
        return self.row + 1   # same value; row has already incremented when printed

    def advance(self) -> bool:
        self.x_waypoint = None
        if self.phase == MowPhase.X_PASS:
            if self.row + 1 >= self._total_rows:
                return False
            self.phase = MowPhase.Y_SHIFT
        else:
            self.row       += 1
            self.going_east = not self.going_east
            self.phase      = MowPhase.X_PASS
        return True

    def skip_to_row(self, new_row: int) -> None:
        self.x_waypoint = None
        self.row        = new_row
        self.going_east = (new_row % 2 == 0)
        self.phase      = MowPhase.X_PASS

    @property
    def is_complete(self) -> bool:
        return self.row >= self._total_rows


# ─────────────────────────────────────────────────────────────────────
# Obstacle Avoidance FSM
# ─────────────────────────────────────────────────────────────────────

class AvoidPhase(IntEnum):
    NONE   = 0
    BACKUP = 1
    CURVE  = 2


class AvoidState:
    """Backup then curve around obstacle."""

    def __init__(self):
        self.phase:           AvoidPhase = AvoidPhase.NONE
        self.phase_end:       float      = 0.0
        self.curve_sign:      float      = 1.0
        self._side:           str        = "both"   # "left" | "right" | "both"
        self._hard_deadline:  float      = 0.0
        self._extensions:     int        = 0

    @property
    def active(self) -> bool:
        return self.phase != AvoidPhase.NONE

    def begin(self, t: float, side: str) -> None:
        """
        Start an avoidance manoeuvre.

        Parameters
        ----------
        t    : current monotonic time
        side : which side detected the obstacle — "left", "right", or "both".
               Comes from classify_ir() in robot.py.
               "left"  → curve right (away from left obstacle)
               "right" → curve left
               "both"  → curve left (arbitrary; could also reverse fully)
        """
        from config import AVOID_BACKUP_TIME_S, AVOID_CURVE_TIME_S
        self._side          = side
        # curve_sign: -1 = curve right, +1 = curve left (used by arc mode)
        self.curve_sign     = -1.0 if side == "left" else +1.0

        # "both" blocked: reverse for twice as long to clear the gap,
        # then always curve left.
        backup_t = AVOID_BACKUP_TIME_S * 2.0 if side == "both" else AVOID_BACKUP_TIME_S

        self.phase          = AvoidPhase.BACKUP
        self.phase_end      = t + backup_t
        self._hard_deadline = t + backup_t + AVOID_CURVE_TIME_S * 4
        self._extensions    = 0
        log.info("Avoidance: side=%s curve=%s backup=%.1fs (deadline %.1fs)",
                 side,
                 "right" if self.curve_sign < 0 else "left",
                 backup_t,
                 backup_t + AVOID_CURVE_TIME_S * 4)

    def update(self, t: float, drive, is_clear_fn) -> bool:
        """
        Returns True when avoidance is complete.

        Exits when ANY of the following is true:
          (a) is_clear_fn() returns True  (path confirmed clear)
          (b) Hard deadline reached       (max 4× AVOID_CURVE_TIME_S total)

        is_clear_fn() is checked on EVERY tick during the CURVE phase so
        the robot exits as soon as the obstacle is removed, not just at the
        end of a time window.
        """
        from config import AVOID_BACKUP_DUTY, AVOID_CURVE_DUTY, AVOID_CURVE_TIME_S

        # Hard deadline — always exit, prevents infinite avoidance loop
        if t >= self._hard_deadline:
            drive.stop()
            self.phase = AvoidPhase.NONE
            log.warning("Avoidance hard deadline reached — forcing exit "
                        "(check IR threshold or obstacle clearance distance)")
            return True

        if self.phase == AvoidPhase.BACKUP:
            drive.set_vw(-AVOID_BACKUP_DUTY, 0.0)
            if t >= self.phase_end:
                self.phase     = AvoidPhase.CURVE
                self.phase_end = t + AVOID_CURVE_TIME_S
            return False

        if self.phase == AvoidPhase.CURVE:
            from config import PIVOT_IN_PLACE
            if PIVOT_IN_PLACE:
                # Pivot in place: one side forward, other side reversed.
                # Near-zero radius — reliable even on heavy robots where
                # low slow-side duty stalls the motor (0.35× was too low).
                if self._side == "left":
                    # obstacle left → turn right: left fwd, right rev
                    drive.set_left_right(AVOID_CURVE_DUTY, -AVOID_CURVE_DUTY)
                else:
                    # obstacle right/both → turn left: left rev, right fwd
                    drive.set_left_right(-AVOID_CURVE_DUTY, AVOID_CURVE_DUTY)
            else:
                # Gentle arc: slow side stays forward at 60% duty.
                # 0.35× was observed to stall; 0.60× spins reliably.
                slow = AVOID_CURVE_DUTY * 0.60
                if self._side == "left":
                    drive.set_left_right(AVOID_CURVE_DUTY, slow)
                else:
                    drive.set_left_right(slow, AVOID_CURVE_DUTY)

            # Check for clear path every tick — exit immediately when clear
            if is_clear_fn():
                drive.stop()
                self.phase = AvoidPhase.NONE
                log.info("Avoidance complete — path clear")
                return True

            # If current curve window expired but not clear, extend once more
            if t >= self.phase_end:
                self._extensions += 1
                self.phase_end    = t + AVOID_CURVE_TIME_S * 0.5
                log.info("Avoidance: still not clear, extending "
                         "(extension %d)", self._extensions)
            return False

        self.phase = AvoidPhase.NONE
        return True


# ─────────────────────────────────────────────────────────────────────
# Stuck Recovery
# ─────────────────────────────────────────────────────────────────────

class StuckRecovery:
    """
    Detects when the robot is stuck (motors running, no encoder movement
    OR IMU shows no acceleration despite commanded movement) and executes
    a randomised escape manoeuvre.
    """

    def __init__(self):
        self._active        = False
        self._phase_end     = 0.0
        self._escape_sign   = 1.0

    @property
    def active(self) -> bool:
        return self._active

    def begin(self, t: float) -> None:
        import random
        self._active      = True
        self._phase_end   = t + 2.0
        self._escape_sign = random.choice([-1.0, 1.0])
        log.warning("Stuck detected — executing escape manoeuvre")

    def update(self, t: float, drive) -> bool:
        """Returns True when escape is done."""
        if not self._active:
            return True
        drive.set_vw(-0.45, 0.5 * self._escape_sign)
        if t >= self._phase_end:
            drive.stop()
            self._active = False
            log.info("Stuck recovery complete")
            return True
        return False


# ─────────────────────────────────────────────────────────────────────
# Return Path Planner
# ─────────────────────────────────────────────────────────────────────

class ReturnPlanner:
    """Dijkstra-based return path to dock."""

    def __init__(self):
        self._waypoints: List[Tuple[float, float]] = []
        self._cells:     List[Tuple[int, int]]     = []
        self._idx:       int                        = 0

    def plan(self, grid: Grid, rx: float, ry: float) -> None:
        from config import DOCK_X, DOCK_Y, INFLATE_RADIUS, CELL_SIZE
        inflated   = grid.inflated(INFLATE_RADIUS)
        start_cell = inflated.current_cell(rx, ry)
        if not inflated.is_passable(*start_cell):
            start_cell = inflated.nearest_passable(rx, ry) or start_cell

        exit_cell = inflated.nearest_passable(DOCK_X, DOCK_Y)
        if exit_cell is None:
            self._fallback(DOCK_X, DOCK_Y); return

        path = dijkstra(inflated, start_cell, exit_cell)
        if path is None:
            path = dijkstra(grid, grid.current_cell(rx, ry),
                             grid.nearest_passable(DOCK_X, DOCK_Y)
                             or grid.current_cell(DOCK_X, DOCK_Y))
        if path is None:
            self._fallback(DOCK_X, DOCK_Y); return

        self._cells     = list(path)
        world           = [Grid.cell_centre(c, r) for c, r in path]
        world.append((DOCK_X, DOCK_Y))
        self._waypoints = world
        self._idx       = 0
        log.info("Return path: %d waypoints", len(world))

    def _fallback(self, dock_x: float, dock_y: float) -> None:
        log.warning("ReturnPlanner: direct line to dock")
        self._waypoints = [(dock_x, dock_y)]
        self._cells     = []
        self._idx       = 0

    def follow(self, drive, rx: float, ry: float) -> bool:
        """Returns True when docked."""
        from config import DOCK_X, DOCK_Y, DOCK_TOL, CELL_SIZE
        if not self._waypoints or self._idx >= len(self._waypoints):
            return drive.steer_to_point(DOCK_X, DOCK_Y, tol=DOCK_TOL)

        tx, ty  = self._waypoints[self._idx]
        is_last = self._idx == len(self._waypoints) - 1
        tol     = DOCK_TOL if is_last else CELL_SIZE * 0.6

        if drive.steer_to_point(tx, ty, tol=tol):
            self._idx += 1
            if self._idx >= len(self._waypoints):
                return True
        return False

    def replan(self, grid: Grid, rx: float, ry: float) -> None:
        log.info("ReturnPlanner replanning due to obstacle")
        self.plan(grid, rx, ry)

    @property
    def cells(self) -> List[Tuple[int, int]]:
        return list(self._cells)