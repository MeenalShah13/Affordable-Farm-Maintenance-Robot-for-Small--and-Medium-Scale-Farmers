from controller import Robot
import math, heapq
from typing import List, Tuple, Dict, Optional, Set

Cell = Tuple[int, int]

def clamp(x, lo, hi): return max(lo, min(hi, x))
def wrap_pi(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

# --------------------------
# FIELD (from your world)
# --------------------------
FIELD_CENTER_X = 0.59
FIELD_CENTER_Y = 0
FIELD_W_M = 2.0
FIELD_H_M = 3.0

CELL_M = 0.1
GRID_W = int(math.ceil(FIELD_W_M / CELL_M))
GRID_H = int(math.ceil(FIELD_H_M / CELL_M))

# Field bottom-left corner in world XY
# ROBOT_MARGIN_M = 0.06
# FIELD_MIN_X = FIELD_CENTER_X - FIELD_W_M / 2.0 - ROBOT_MARGIN_M  # = -0.35
# FIELD_MIN_Y = FIELD_CENTER_Y - FIELD_H_M / 2.0 - ROBOT_MARGIN_M  # = -1.56
FIELD_MIN_X = -0.42
FIELD_MIN_Y = -1.14

# Dock (from your world) - note: dock is OUTSIDE the field
DOCK_WORLD_X = -0.57
DOCK_WORLD_Y = -0.07

# Field entry point - closest point on field edge to dock
# FIELD_ENTRY_X = FIELD_MIN_X + 0.15  # Just inside field left edge
# FIELD_ENTRY_Y = FIELD_MIN_Y + FIELD_H_M - 0.15  # Near top of field

FIELD_ENTRY_X = -0.22
FIELD_ENTRY_Y = -0.2

# Robot + drive params
WHEEL_RADIUS_M = 0.07
TRACK_WIDTH_M = 0.19
MAX_WHEEL_RAD_S = 8.0

V_FWD = 0.12
# Turning behavior
TURN_KP = 3.5
MAX_W_CMD = 4.5
TURN_IN_PLACE_ERR = 0.6   # rad (~35 deg)

# Boundary recovery
FIELD_MARGIN_M = 0.12     # keep this much inside the boundary
BOUNDARY_BACKUP_S = 0.8   # time to back up when boundary triggers
BOUNDARY_TURN_S = 1.0     # time to turn inward after backup
BOUNDARY_BACKUP_V = -0.10
BOUNDARY_TURN_W = 2.5

# Distance sensor threshold
# Webots "generic" DistanceSensor returns LOWER values when objects are CLOSER
# Default lookup table: 0 (close) to 1000 (far)
OBSTACLE_THRESH = 300.0  # Trigger when value is ABOVE this (obstacle close)

# Stuck detection
STUCK_TIME_THRESH = 3.0   # if no progress for this long, we're stuck
STUCK_DIST_THRESH = 0.02  # minimum movement to count as progress

# Obstacle avoidance when stuck
AVOID_BACKUP_S = 0.5
AVOID_TURN_S = 0.7
AVOID_TURN_W = 3.0

# Dock camera verification
RED_RATIO_THRESH = 0.02
VERIFY_SPIN_W = 0.8
VERIFY_TIMEOUT_S = 8.0

# Battery simulation
BATTERY_CAPACITY_S = 180.0
RETURN_AT_FRAC = 0.30
RETURN_MARGIN = 1.5
CHARGE_TIME_S = 6.0

# Planner
PLANNER_MODE = "stc"


class GridMap:
    UNKNOWN = 0
    FREE = 1
    OCC = 2

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.occ = [[GridMap.UNKNOWN for _ in range(h)] for _ in range(w)]
        self.vis = [[False for _ in range(h)] for _ in range(w)]

    def in_bounds(self, c: Cell) -> bool:
        x, y = c
        return 0 <= x < self.w and 0 <= y < self.h

    def get(self, c: Cell) -> int:
        x, y = c
        return self.occ[x][y]

    def set_occ(self, c: Cell, v: int):
        if self.in_bounds(c):
            x, y = c
            self.occ[x][y] = v

    def mark_visited(self, c: Cell):
        if self.in_bounds(c):
            x, y = c
            self.vis[x][y] = True
            if self.occ[x][y] == GridMap.UNKNOWN:
                self.occ[x][y] = GridMap.FREE

    def neighbors4(self, c: Cell) -> List[Cell]:
        x, y = c
        nbr = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
        out = []
        for n in nbr:
            if self.in_bounds(n) and self.get(n) != GridMap.OCC:
                out.append(n)
        return out

    def coverage_done(self) -> bool:
        for x in range(self.w):
            for y in range(self.h):
                if (self.occ[x][y] == GridMap.FREE or self.occ[x][y] == GridMap.UNKNOWN) and not self.vis[x][y]:
                    return False
        return True


def dijkstra(grid: GridMap, start: Cell, goal: Cell) -> Optional[List[Cell]]:
    if not grid.in_bounds(start) or not grid.in_bounds(goal):
        return None
    if grid.get(start) == GridMap.OCC or grid.get(goal) == GridMap.OCC:
        return None

    pq = [(0, start)]
    dist: Dict[Cell, int] = {start: 0}
    parent: Dict[Cell, Cell] = {}

    while pq:
        d, u = heapq.heappop(pq)
        if u == goal:
            break
        if d != dist.get(u, 10**9):
            continue
        for v in grid.neighbors4(u):
            nd = d + 1
            if nd < dist.get(v, 10**9):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    if goal not in dist:
        return None

    path = [goal]
    cur = goal
    while cur != start:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def build_spanning_tree(grid: GridMap, root: Cell) -> Dict[Cell, Cell]:
    parent: Dict[Cell, Cell] = {}
    stack = [root]
    seen: Set[Cell] = {root}

    while stack:
        u = stack.pop()
        nbrs = grid.neighbors4(u)
        # prefer unvisited
        nbrs.sort(key=lambda c: 0 if not grid.vis[c[0]][c[1]] else 1)
        for v in nbrs:
            if v in seen:
                continue
            if grid.get(v) == GridMap.OCC:
                continue
            seen.add(v)
            parent[v] = u
            stack.append(v)

    return parent


def stc_coverage_path(grid: GridMap, start: Cell) -> List[Cell]:
    parent = build_spanning_tree(grid, start)
    children: Dict[Cell, List[Cell]] = {}
    for c, p in parent.items():
        children.setdefault(p, []).append(c)
    for k in children:
        children[k].sort()

    tour: List[Cell] = [start]
    stack = [(start, 0)]
    while stack:
        node, idx = stack[-1]
        kids = children.get(node, [])
        if idx < len(kids):
            nxt = kids[idx]
            stack[-1] = (node, idx + 1)
            tour.append(nxt)
            stack.append((nxt, 0))
        else:
            stack.pop()
            if stack:
                tour.append(stack[-1][0])

    out = []
    for c in tour:
        if not out or out[-1] != c:
            out.append(c)
    return out


def red_ratio(camera) -> float:
    img = camera.getImage()
    if img is None:
        return 0.0
    w = camera.getWidth()
    h = camera.getHeight()
    stride = 4
    red = 0
    total = 0
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            r = camera.imageGetRed(img, w, x, y)
            g = camera.imageGetGreen(img, w, x, y)
            b = camera.imageGetBlue(img, w, x, y)
            total += 1
            if r > 160 and g < 80 and b < 80:
                red += 1
    return red / max(1, total)


class InitialRobot:
    def __init__(self):
        self.robot = Robot()
        self.dt_ms = int(self.robot.getBasicTimeStep())
        self.dt = self.dt_ms / 1000.0
        self.has_left_dock = False
        self.entered_field = False  # NEW: Track if we've entered the field

        # Motors
        self.mL1 = self.robot.getDevice("wheel1")
        self.mR1 = self.robot.getDevice("wheel2")
        self.mL2 = self.robot.getDevice("wheel3")
        self.mR2 = self.robot.getDevice("wheel4")
        for m in (self.mL1, self.mR1, self.mL2, self.mR2):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)

        # Encoders
        self.psL1 = self.robot.getDevice("wheel1_ps")
        self.psR1 = self.robot.getDevice("wheel2_ps")
        self.psL2 = self.robot.getDevice("wheel3_ps")
        self.psR2 = self.robot.getDevice("wheel4_ps")
        for ps in (self.psL1, self.psR1, self.psL2, self.psR2):
            ps.enable(self.dt_ms)

        # Distance sensors
        self.dsL = self.robot.getDevice("ds_left")
        self.dsR = self.robot.getDevice("ds_right")
        self.dsL.enable(self.dt_ms)
        self.dsR.enable(self.dt_ms)

        # Camera
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(self.dt_ms)

        # State - start with GOTO_FIELD to navigate from dock to field
        self.state = "GOTO_FIELD"
        self.x, self.y, self.th = DOCK_WORLD_X, DOCK_WORLD_Y, -math.pi / 2
        self.prev_L = None
        self.prev_R = None
        self.batt = BATTERY_CAPACITY_S
        self.charging_until = 0.0

        # Grid
        self.grid = GridMap(GRID_W, GRID_H)
        self.dock_cell = self.world_to_cell_in_field(DOCK_WORLD_X, DOCK_WORLD_Y)
        self.dock_cell_clamped = (
            clamp(self.dock_cell[0], 0, GRID_W-1),
            clamp(self.dock_cell[1], 0, GRID_H-1)
        )

        # Plan
        self.plan: List[Cell] = []
        self.plan_i = 0

        # Boundary recovery
        self.boundary_mode = "NONE"  # NONE | BACKUP | TURN | GOTO
        self.boundary_until = 0.0
        self.boundary_target = None

        # Stuck detection
        self.stuck_check_start = 0.0
        self.stuck_last_pos = (self.x, self.y)
        self.stuck_mode = "NONE"  # NONE | BACKUP | TURN
        self.stuck_until = 0.0

    def time(self) -> float:
        return self.robot.getTime()

    def stop(self):
        for m in (self.mL1, self.mR1, self.mL2, self.mR2):
            m.setVelocity(0.0)

    def set_vw(self, v: float, w: float):
        """v = forward vel (m/s), w = angular vel (rad/s)"""
        v_L = v - 0.5 * w * TRACK_WIDTH_M
        v_R = v + 0.5 * w * TRACK_WIDTH_M
        omega_L = v_L / WHEEL_RADIUS_M
        omega_R = v_R / WHEEL_RADIUS_M
        omega_L = clamp(omega_L, -MAX_WHEEL_RAD_S, MAX_WHEEL_RAD_S)
        omega_R = clamp(omega_R, -MAX_WHEEL_RAD_S, MAX_WHEEL_RAD_S)
        self.mL1.setVelocity(omega_L)
        self.mL2.setVelocity(omega_L)
        self.mR1.setVelocity(omega_R)
        self.mR2.setVelocity(omega_R)

    def update_odometry(self):
        eL = (self.psL1.getValue() + self.psL2.getValue()) / 2.0
        eR = (self.psR1.getValue() + self.psR2.getValue()) / 2.0
        
        if self.prev_L is None:
            self.prev_L, self.prev_R = eL, eR
            return
        
        dL = (eL - self.prev_L) * WHEEL_RADIUS_M
        dR = (eR - self.prev_R) * WHEEL_RADIUS_M
        self.prev_L, self.prev_R = eL, eR
        
        ds = (dL + dR) / 2.0
        dth = (dR - dL) / TRACK_WIDTH_M
        print(f"Odometry update: eL={eL:.4f} eR={eR:.4f} dL={dL:.4f} dR={dR:.4f} ds={ds:.4f} dth={dth:.4f}")
        self.x += ds * math.cos(self.th + dth / 2.0)
        self.y += ds * math.sin(self.th + dth / 2.0)
        self.th = wrap_pi(self.th + dth)

    def obstacle_close(self) -> bool:
        """
        Webots 'generic' DistanceSensor returns HIGHER values when objects are CLOSER.
        Default lookup table maps distance to value: far=0, close=1000.
        Obstacle detected if EITHER sensor reads ABOVE threshold.
        """
        vL = self.dsL.getValue()
        vR = self.dsR.getValue()
        return (vL < OBSTACLE_THRESH) or (vR < OBSTACLE_THRESH)

    def world_to_cell_in_field(self, wx: float, wy: float) -> Cell:
        cx = int((wx - FIELD_MIN_X) / CELL_M)
        cy = int((wy - FIELD_MIN_Y) / CELL_M)
        return (cx, cy)

    def cell_center_world(self, c: Cell) -> Tuple[float, float]:
        cx, cy = c
        wx = FIELD_MIN_X + (cx + 0.5) * CELL_M
        wy = FIELD_MIN_Y + (cy + 0.5) * CELL_M
        return (wx, wy)

    def update_map(self):
        """Mark current cell as visited and update obstacle info"""
        cur = self.world_to_cell_in_field(self.x, self.y)
        if self.grid.in_bounds(cur):
            self.grid.mark_visited(cur)

        # If obstacle detected, mark cells ahead as occupied
        if self.obstacle_close():
            # Mark cell ~0.15m ahead as occupied
            ahead_x = self.x + 0.15 * math.cos(self.th)
            ahead_y = self.y + 0.15 * math.sin(self.th)
            ahead_cell = self.world_to_cell_in_field(ahead_x, ahead_y)
            if self.grid.in_bounds(ahead_cell):
                self.grid.set_occ(ahead_cell, GridMap.OCC)

    def inside_field_world(self) -> bool:
        return (FIELD_MIN_X <= self.x <= FIELD_MIN_X + FIELD_W_M and
                FIELD_MIN_Y <= self.y <= FIELD_MIN_Y + FIELD_H_M)

    def heading_points_outside(self) -> bool:
        """Check if heading points out of field"""
        margin = 0.05
        if self.x < FIELD_MIN_X + margin and math.cos(self.th) < -0.5:
            return True
        if self.x > FIELD_MIN_X + FIELD_W_M - margin and math.cos(self.th) > 0.5:
            return True
        if self.y < FIELD_MIN_Y + margin and math.sin(self.th) < -0.5:
            return True
        if self.y > FIELD_MIN_Y + FIELD_H_M - margin and math.sin(self.th) > 0.5:
            return True
        return False

    def nearest_safe_point_inside(self) -> Tuple[float, float]:
        """Find nearest point well inside field boundaries"""
        cx = clamp(self.x, FIELD_MIN_X + FIELD_MARGIN_M,
                   FIELD_MIN_X + FIELD_W_M - FIELD_MARGIN_M)
        cy = clamp(self.y, FIELD_MIN_Y + FIELD_MARGIN_M,
                   FIELD_MIN_Y + FIELD_H_M - FIELD_MARGIN_M)
        return (cx, cy)

    def drive_to_world_point(self, tx: float, ty: float, ignore_obstacles: bool = False) -> bool:
        """Drive to a world coordinate. Returns True if arrived."""
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        
        if dist < 0.05:
            self.stop()
            return True

        desired = math.atan2(dy, dx)
        err = wrap_pi(desired - self.th)
        w = clamp(TURN_KP * err, -MAX_W_CMD, MAX_W_CMD)
        
        heading_scale = clamp(1.0 - 1.5 * abs(err), 0.1, 1.0)
        v = V_FWD * heading_scale

        if abs(err) > TURN_IN_PLACE_ERR:
            v = 0.0

        # Check obstacles unless explicitly ignored
        if not ignore_obstacles and self.boundary_mode == "NONE" and self.stuck_mode == "NONE":
            if self.obstacle_close():
                self.stop()
                return False

        self.set_vw(v, w)
        return False

    def check_stuck(self) -> bool:
        """Detect if robot is stuck in same position"""
        if self.stuck_mode != "NONE":
            return False  # Already handling stuck
        
        current_time = self.time()
        dx = self.x - self.stuck_last_pos[0]
        dy = self.y - self.stuck_last_pos[1]
        dist_moved = math.hypot(dx, dy)
        
        if dist_moved > STUCK_DIST_THRESH:
            # Made progress, reset timer
            self.stuck_check_start = current_time
            self.stuck_last_pos = (self.x, self.y)
            return False
        
        # No progress - check if stuck
        if current_time - self.stuck_check_start > STUCK_TIME_THRESH:
            return True
        
        return False

    def need_return(self) -> bool:
        if not self.has_left_dock:
            return False
        
        frac = self.batt / BATTERY_CAPACITY_S
        if frac < RETURN_AT_FRAC:
            print("Battery low, need to return")
            return True

        cur = self.world_to_cell_in_field(self.x, self.y)
        cur_clamped = (clamp(cur[0], 0, GRID_W-1), clamp(cur[1], 0, GRID_H-1))
        path = dijkstra(self.grid, cur_clamped, self.dock_cell_clamped)
        if not path:
            print("No path to dock, need to return")
            return True
        dist_m = len(path) * CELL_M
        est_time = (dist_m / max(0.05, V_FWD)) * RETURN_MARGIN
        print(est_time > self.batt)
        return est_time > self.batt

    def make_coverage_plan(self):
        cur = self.world_to_cell_in_field(self.x, self.y)
        if not self.grid.in_bounds(cur):
            cur = (clamp(cur[0], 0, GRID_W-1), clamp(cur[1], 0, GRID_H-1))

        self.plan = stc_coverage_path(self.grid, cur)
        self.plan_i = 0

    def make_return_plan(self):
        cur = self.world_to_cell_in_field(self.x, self.y)
        cur = (clamp(cur[0], 0, GRID_W-1), clamp(cur[1], 0, GRID_H-1))
        self.plan = dijkstra(self.grid, cur, self.dock_cell_clamped) or [cur]
        self.plan_i = 0

    def drive_plan_step(self):
        if self.plan_i >= len(self.plan):
            self.stop()
            return

        target = self.plan[self.plan_i]
        tx, ty = self.cell_center_world(target)

        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        if dist < 0.05:
            self.plan_i += 1
            self.grid.mark_visited(target)
            self.stop()
            return

        desired = math.atan2(dy, dx)
        err = wrap_pi(desired - self.th)

        w = clamp(TURN_KP * err, -MAX_W_CMD, MAX_W_CMD)
        heading_scale = clamp(1.0 - 1.5 * abs(err), 0.1, 1.0)
        v = V_FWD * heading_scale

        if abs(err) > TURN_IN_PLACE_ERR:
            v = 0.0

        if self.obstacle_close():
            self.stop()
            return

        self.set_vw(v, w)

    def at_dock_estimate(self) -> bool:
        if not self.has_left_dock:
            return False
        return math.hypot(self.x - DOCK_WORLD_X, self.y - DOCK_WORLD_Y) < 0.12

    def verify_dock_camera(self) -> bool:
        start = self.time()
        while self.robot.step(self.dt_ms) != -1:
            self.update_odometry()
            ratio = red_ratio(self.camera)
            if ratio >= RED_RATIO_THRESH:
                self.stop()
                return True
            if self.time() - start > VERIFY_TIMEOUT_S:
                self.stop()
                return False
            self.set_vw(0.0, VERIFY_SPIN_W)
        return False

    def run(self):
        last_t = self.time()

        while self.robot.step(self.dt_ms) != -1:
            t = self.time()
            dt = max(1e-3, t - last_t)
            last_t = t

            self.update_odometry()
            
            # Track when we've left the dock area
            if math.hypot(self.x - DOCK_WORLD_X, self.y - DOCK_WORLD_Y) > 0.25:
                self.has_left_dock = True
            
            # Track when we've entered the field
            if self.inside_field_world():
                self.entered_field = True
            
            # Only update map when inside field
            if self.inside_field_world():
                self.update_map()
            
            print(f"state={self.state} batt={self.batt:.1f}s "
                  f"pos=({self.x:.2f},{self.y:.2f}) th={self.th:.2f} "
                  f"obsL={self.dsL.getValue():.0f} obsR={self.dsR.getValue():.0f} "
                  f"obstClose={self.obstacle_close()} boundary={self.boundary_mode} "
                  f"stuck={self.stuck_mode} inField={self.inside_field_world()}",
                  f"plan_i={self.plan_i}/{len(self.plan)}")
            
            # -----------------
            # STUCK RECOVERY (highest priority, but only when in field)
            # -----------------
            if self.stuck_mode != "NONE" and self.entered_field:
                print("Line 617: In stuck recovery mode:", self.stuck_mode)
                if self.stuck_mode == "BACKUP":
                    self.set_vw(BOUNDARY_BACKUP_V, 0.0)
                    if self.time() >= self.stuck_until:
                        self.stuck_mode = "TURN"
                        self.stuck_until = self.time() + AVOID_TURN_S
                
                elif self.stuck_mode == "TURN":
                    print("Line 625: Executing TURN phase of stuck recovery")
                    # Turn away from obstacles
                    vL = self.dsL.getValue()
                    vR = self.dsR.getValue()
                    # Turn toward the side with LESS obstacle (LOWER sensor value = farther)
                    turn_dir = 1.0 if vL < vR else -1.0
                    self.set_vw(0.0, AVOID_TURN_W * turn_dir)
                    if self.time() >= self.stuck_until:
                        self.stuck_mode = "NONE"
                        self.stuck_check_start = self.time()
                        self.stuck_last_pos = (self.x, self.y)
                        # Replan after stuck recovery
                        if self.state == "COVER":
                            self.make_coverage_plan()
                        elif self.state == "RETURN":
                            self.make_return_plan()
                continue

            # -----------------
            # BOUNDARY RECOVERY (only when in COVER or already entered field)
            # -----------------
            if self.boundary_mode != "NONE" and self.entered_field:
                print("Line 651: In boundary recovery mode:", self.boundary_mode)
                if self.boundary_mode == "BACKUP":
                    self.set_vw(BOUNDARY_BACKUP_V, 0.0)
                    if self.time() >= self.boundary_until:
                        self.boundary_mode = "TURN"
                        self.boundary_until = self.time() + BOUNDARY_TURN_S

                elif self.boundary_mode == "TURN":
                    # Intelligent turn: turn toward field center
                    print("Line 657: Executing TURN phase of boundary recovery")
                    cx = (FIELD_MIN_X + FIELD_W_M/2) - self.x
                    cy = (FIELD_MIN_Y + FIELD_H_M/2) - self.y
                    desired = math.atan2(cy, cx)
                    err = wrap_pi(desired - self.th)
                    turn_dir = 1.0 if err > 0 else -1.0
                    self.set_vw(0.0, BOUNDARY_TURN_W * turn_dir)
                    
                    if self.time() >= self.boundary_until:
                        self.boundary_mode = "GOTO"
                        self.boundary_target = self.nearest_safe_point_inside()

                elif self.boundary_mode == "GOTO":
                    print("Line 671: Executing GOTO phase of boundary recovery")
                    tx, ty = self.boundary_target
                    done = self.drive_to_world_point(tx, ty)
                    if done:
                        self.boundary_mode = "NONE"
                        self.boundary_target = None
                        if self.state == "COVER":
                            self.make_coverage_plan()
                        elif self.state == "RETURN":
                            self.make_return_plan()
                continue

            # Trigger boundary recovery ONLY if we've entered the field and now left it
            # or if we're near the edge and heading out
            if self.entered_field and self.state == "COVER":
                print("Line 688: Checking for boundary recovery trigger")
                if not self.inside_field_world():
                    self.boundary_mode = "BACKUP"
                    self.boundary_until = self.time() + BOUNDARY_BACKUP_S
                    self.stop()
                    continue

                near_edge = (
                    self.x < FIELD_MIN_X - FIELD_MARGIN_M or
                    self.x > FIELD_MIN_X + FIELD_W_M - FIELD_MARGIN_M or
                    self.y < FIELD_MIN_Y - FIELD_MARGIN_M or
                    self.y > FIELD_MIN_Y + FIELD_H_M - FIELD_MARGIN_M
                )
                if near_edge and self.heading_points_outside():
                    self.boundary_mode = "BACKUP"
                    self.boundary_until = self.time() + BOUNDARY_BACKUP_S
                    self.stop()
                    continue

            # Battery drain when not charging
            if self.state != "CHARGE":
                self.batt = max(0.0, self.batt - dt)

            # -----------------
            # MAIN STATE MACHINE
            # -----------------
            
            if self.state == "GOTO_FIELD":
                # Navigate from dock to field entry point
                arrived = self.drive_to_world_point(FIELD_ENTRY_X, FIELD_ENTRY_Y, ignore_obstacles=False)
                
                if self.inside_field_world():
                    # We've entered the field, switch to coverage
                    print("*** ENTERED FIELD - STARTING COVERAGE ***")
                    self.state = "COVER"
                    self.entered_field = True
                    self.stuck_check_start = self.time()
                    self.stuck_last_pos = (self.x, self.y)
                    self.make_coverage_plan()
                    print("path", self.plan)
                elif arrived:
                    print("*** ARRIVED AT FIELD ENTRY POINT ***")
                    # Arrived at entry point (should be inside field)
                    self.state = "COVER"
                    self.entered_field = True
                    self.stuck_check_start = self.time()
                    self.stuck_last_pos = (self.x, self.y)
                    self.make_coverage_plan()
                    print("path", self.plan)
            elif self.state == "COVER":
                # Check if stuck
                if self.check_stuck():
                    print("*** STUCK DETECTED - INITIATING RECOVERY ***")
                    self.stuck_mode = "BACKUP"
                    self.stuck_until = self.time() + AVOID_BACKUP_S
                    self.stop()
                    continue

                # Check if need to return
                if self.has_left_dock and (self.grid.coverage_done() or self.need_return()):
                    print(self.has_left_dock, self.grid.coverage_done(), self.need_return())
                    print("Line 762: Coverage complete or need to return - switching to RETURN state")
                    self.state = "RETURN"
                    self.make_return_plan()
                    print("return path", self.plan)
                    continue

                # Drive coverage plan
                if self.obstacle_close():
                    print("Line 768: Obstacle detected during COVER - updating map and replanning")
                    self.stop()
                    self.update_map()
                    self.make_coverage_plan()
                    print("new path", self.plan)
                else:
                    self.drive_plan_step()

            elif self.state == "RETURN":
                print("Line 777: In RETURN state")
                # Check if stuck
                if self.check_stuck():
                    print("*** STUCK WHILE RETURNING - INITIATING RECOVERY ***")
                    self.stuck_mode = "BACKUP"
                    self.stuck_until = self.time() + AVOID_BACKUP_S
                    self.stop()
                    continue

                if self.at_dock_estimate():
                    print("*** ARRIVED AT DOCK ESTIMATE - VERIFYING ***")
                    self.stop()
                    self.state = "VERIFY"
                else:
                    print("Line 790: Driving return plan")
                    # If outside field, drive directly to dock
                    if not self.inside_field_world():
                        self.drive_to_world_point(DOCK_WORLD_X, DOCK_WORLD_Y)
                    else:
                        # Follow grid plan to field edge, then go to dock
                        if self.plan_i >= len(self.plan) or len(self.plan) <= 1:
                            self.drive_to_world_point(DOCK_WORLD_X, DOCK_WORLD_Y)
                        else:
                            if self.obstacle_close():
                                print("Line 797: Obstacle detected during RETURN - updating map and replanning")
                                self.update_map()
                                self.make_return_plan()
                                self.stop()
                            else:
                                self.drive_plan_step()

            elif self.state == "VERIFY":
                print("Line 804: In VERIFY state")
                ok = self.verify_dock_camera()
                self.state = "CHARGE"
                self.charging_until = self.time() + CHARGE_TIME_S
                self.stop()

            elif self.state == "CHARGE":
                print("Line 813: In CHARGE state")
                self.stop()
                if self.time() >= self.charging_until:
                    # Reset for next coverage run
                    self.has_left_dock = False
                    self.entered_field = False
                    self.batt = BATTERY_CAPACITY_S
                    self.x, self.y, self.th = DOCK_WORLD_X, DOCK_WORLD_Y, 0.0
                    self.prev_L = None
                    self.prev_R = None
                    self.stuck_check_start = self.time()
                    self.stuck_last_pos = (self.x, self.y)
                    self.state = "GOTO_FIELD"  # Go back to field

if __name__ == "__main__":
    InitialRobot().run()