"""
robot.py — FieldBot main state machine.

States
──────
  INIT        Warm up IMU, load persisted grid, init hardware
  GO_TO_START Navigate from dock to first mowing position
  MOWING      Boustrophedon coverage — sample every SAMPLE_INTERVAL_S
  SAMPLING    Stationary sample collection (blocking, ~45–90 s)
  AVOIDING    Obstacle backup+curve FSM
  STUCK       Unstuck escape manoeuvre
  RETURN      Dijkstra path back to dock
  DOCKING     Back into dock (reverse manoeuvre + charging detection)
  CHARGING    Sit on dock; upload data; purge images; save grid
  DONE        Idle — grid preserved for next run

Obstacle detection
──────────────────
  1. IR sensors    (ADS1015, ch0=left, ch1=right)   → hard avoid
  2. Accelerometer (IMU bump)                        → hard avoid
  3. Encoder-zero with non-zero duty                 → stuck recovery
"""

from __future__ import annotations

import math
import logging
import os
import time
from enum import IntEnum, auto
from typing import Optional

log = logging.getLogger(__name__)


def _clamp(x, lo, hi): return max(lo, min(hi, x))
def _wrap_pi(a):
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


class RobotState(IntEnum):
    INIT        = auto()
    GO_TO_START = auto()
    MOWING      = auto()
    SAMPLING    = auto()
    AVOIDING    = auto()
    STUCK       = auto()
    RETURN      = auto()
    DOCKING     = auto()
    CHARGING    = auto()
    DONE        = auto()
    PAUSED      = auto()   # remote stop commanded — waiting for dashboard "start"
    RESUMING    = auto()   # navigating back to interrupted mowing position after recharge


class FieldBot:
    """Instantiate once; call run() to start the blocking control loop."""

    def __init__(self):
        log.info("FieldBot initialising…")
        self._init_hardware()
        self._init_data()        # Firebase must be ready before _init_navigation()
        self._init_navigation()  # grid cloud check uses self.firebase
        self._init_sampling()

        self.state        = RobotState.INIT
        self._imu_ticks   = 0
        self._imu_ready   = False
        self._commanding_move = False
        self._last_status_push: float = 0.0
        self._last_decay_check: float = 0.0
        self._remote_stop:  bool  = False   # True when dashboard sends "stop"
        self._last_cmd_poll: float = 0.0    # monotonic time of last command poll
        self._last_grid_push: float = 0.0   # monotonic time of last in-field grid push
        # Battery-return resume tracking
        self._return_reason: str   = "complete"   # "battery" | "complete"
        self._resume_x:      float = 0.0          # mowing position when battery ran low
        self._resume_y:      float = 0.0
        self._resume_mow               = None     # copy of MowState at interruption
        self._resume_path: list        = []       # Dijkstra waypoints dock → interrupt point
        self._resume_idx:  int         = 0
        self._tick_count:  int         = 0

        log.info("FieldBot ready")

    # ── Hardware init ─────────────────────────────────────────────────

    def _init_hardware(self):
        """
        Create the I2C bus and both PCA9685 objects ONCE, then inject them
        into every subsystem that needs them.

        This is the only place in the entire codebase that calls board.I2C()
        or PCA9685().  All other classes receive these objects as parameters.
        Creating multiple board.I2C() handles to the same physical bus causes
        BlockingIOError (errno 11) when handles collide mid-transaction.
        """
        import board
        from adafruit_pca9685 import PCA9685
        from config import PCA41_ADDR, PCA42_ADDR, PCA_FREQ

        # ── Single shared I2C bus ──────────────────────────────────────
        i2c = board.I2C()

        # ── Single shared PCA9685 objects ──────────────────────────────
        self._pca41 = PCA9685(i2c, address=PCA41_ADDR)
        self._pca42 = PCA9685(i2c, address=PCA42_ADDR)
        self._pca41.frequency = PCA_FREQ
        self._pca42.frequency = PCA_FREQ
        log.info("PCA41 %.1f Hz | PCA42 %.1f Hz",
                 self._pca41.frequency, self._pca42.frequency)

        # ── Drive system (receives pre-built PCA objects) ──────────────
        from hardware.pca_motor import DriveSystem, AuxMotor
        from config import (
            M3_PWM, M3_IN1, M3_IN2, ENC_M3,
            M4_PWM, M4_IN1, M4_IN2, ENC_M4,
            M3_DUTY_SCALE, PROBE_DITHER_S, PROBE_TAP_N,
        )
        self.drive      = DriveSystem(self._pca41, self._pca42)
        self.soil_probe = AuxMotor(self._pca42, M3_PWM, M3_IN1, M3_IN2,
                                   enc_pins=ENC_M3, label="M3-SoilProbe",
                                   duty_scale=M3_DUTY_SCALE,
                                   dither_s=PROBE_DITHER_S,
                                   tap_n=PROBE_TAP_N)
        self.cam_arm    = AuxMotor(self._pca42, M4_PWM, M4_IN1, M4_IN2,
                                   enc_pins=ENC_M4, label="M4-CamArm")

        # ── Sensors (receive the shared I2C handle) ────────────────────
        from hardware.sensors import SensorHub
        self.sensors = SensorHub(i2c)

        # ── Accessories (receive the shared PCA handles) ───────────────
        from hardware.peripherals import BatteryMonitor, StatusLEDs, Solenoids, Camera
        self.battery   = BatteryMonitor()   # uses smbus2 separately — no conflict
        self.leds      = StatusLEDs(self._pca41)
        self.solenoids = Solenoids(self._pca41)
        self.camera    = Camera()

    def _init_navigation(self):
        from navigation.planner import (
            Grid, MowState, MowPhase, AvoidState, ReturnPlanner, StuckRecovery
        )
        from config import DOCK_X, DOCK_Y, GRID_SAVE_PATH

        self.grid     = Grid()
        self.mow      = MowState()
        self.avoid    = AvoidState()
        self.stuck    = StuckRecovery()
        self.returner = ReturnPlanner()

        # ── Grid loading priority ──────────────────────────────────────
        # 1. Try local grid.json first (instant, no network required).
        # 2. If not present, try to download from Firebase (one-time check
        #    at boot — acceptable latency while the robot is still at dock).
        # 3. If neither is available, start with a blank grid.

        local_loaded = self.grid.load(GRID_SAVE_PATH)
        if local_loaded:
            log.info("Grid loaded from local file — resuming previous pattern")
        else:
            log.info("No local grid found — checking Firebase cloud…")
            self._load_grid_from_cloud()

        # Set initial pose to dock
        self.drive.set_pose(DOCK_X, DOCK_Y, 0.0)

    def _init_sampling(self):
        from sampling.ai_pipeline import AIPipeline
        from sampling.sampler import Sampler
        self.ai_pipeline = AIPipeline()
        self.sampler     = Sampler(
            sensors=self.sensors,
            camera=self.camera,
            cam_arm=self.cam_arm,
            soil_probe=self.soil_probe,
            ai_pipeline=self.ai_pipeline,
            leds=self.leds,     # LED only lit during soil probe
        )

    def _init_data(self):
        from data.firebase_uploader import FirebaseUploader, StatusPusher
        from data.logger import SessionLogger
        from config import FIREBASE_STATUS_INTERVAL_S

        self.firebase = FirebaseUploader()
        self.session  = SessionLogger(firebase_uploader=self.firebase)

        # Start the background status pusher thread (daemon, won't block exit)
        self.status_pusher = StatusPusher(
            self.firebase, interval_s=FIREBASE_STATUS_INTERVAL_S
        )
        self.status_pusher.start()

    # ── Obstacle / sensor shorthands ─────────────────────────────────

    def _classify_ir(self, left: int, right: int, bump: bool = False) -> str:
        """
        Six-way IR classification (from drive_and_sample.py classify_ir).
        Returns: "near_left"|"near_right"|"near_both"|
                 "warn_left"|"warn_right"|"warn_both"|"clear".
        bump maps to "near_both".
        """
        from config import IR_NEAR_THRESH, IR_WARN_THRESH
        if bump:
            return "near_both"
        l_near = left  > IR_NEAR_THRESH
        r_near = right > IR_NEAR_THRESH
        if l_near and r_near: return "near_both"
        if l_near:            return "near_left"
        if r_near:            return "near_right"
        l_warn = left  > IR_WARN_THRESH
        r_warn = right > IR_WARN_THRESH
        if l_warn and r_warn: return "warn_both"
        if l_warn:            return "warn_left"
        if r_warn:            return "warn_right"
        return "clear"

    def _ir_near(self) -> bool:
        from config import IR_NEAR_THRESH
        return self.sensors.ir.is_obstacle_near(IR_NEAR_THRESH)

    def _ir_warn(self) -> bool:
        from config import IR_WARN_THRESH
        return self.sensors.ir.is_obstacle_warn(IR_WARN_THRESH)

    def _ir_clear(self) -> bool:
        """
        Hysteresis clear check — from drive_and_sample.py ir_is_clear().
        Returns True only when BOTH sensors drop to IR_WARN_THRESH - IR_CLEAR_MARGIN.
        The 500-count margin prevents rapid flicker at the detection boundary.
        """
        from config import IR_WARN_THRESH, IR_CLEAR_MARGIN
        l, r = self._ir_values()
        threshold = IR_WARN_THRESH - IR_CLEAR_MARGIN
        return l < threshold and r < threshold

    def _bump(self) -> bool:
        from config import BUMP_ACCEL_THRESH
        return self.sensors.check_bump(BUMP_ACCEL_THRESH)

    def _ir_values(self):
        r = self.sensors.ir.read()
        return r.left_raw, r.right_raw

    # ── Obstacle marking ─────────────────────────────────────────────

    def _mark_obstacle_ahead(self) -> None:
        from config import OBSTACLE_MARK_DIST
        x, y, th = self.drive.x, self.drive.y, self.drive.th
        for offset in (0.0, 0.3, -0.3):
            a = th + offset
            self.grid.mark_obstacle(
                x + OBSTACLE_MARK_DIST * math.cos(a),
                y + OBSTACLE_MARK_DIST * math.sin(a),
            )

    # ── Mowing helpers ────────────────────────────────────────────────

    def _run_x_pass(self) -> bool:
        """Drive eastward/westward along the current row. Returns True when done."""
        from config import (FIELD_MIN_X, FIELD_MAX_X, FIELD_MARGIN,
                            CROSS_KP, ARRIVE_TOL, CELL_SIZE)
        drive = self.drive
        y_err    = self.mow.target_y - drive.y
        x_target = (self.mow.x_waypoint if self.mow.x_waypoint is not None
                    else self.mow.x_end)

        direction_str = "→ EAST" if self.mow.going_east else "← WEST"
        log.debug("[X-PASS %d/%d] %s  x=%.2f→%.2f  y=%.2f(tgt %.2f err%+.3f)  cov=%.0f%%",
                  self.mow.x_pass_num, self.mow._total_rows,
                  direction_str,
                  drive.x, x_target,
                  drive.y, self.mow.target_y, y_err,
                  self.grid.coverage_fraction() * 100)
        desired_vx = 1.0 * self.mow.x_direction
        desired_vy = _clamp(CROSS_KP * y_err, -0.4, 0.4)
        desired_th = math.atan2(desired_vy, desired_vx)

        # Soft deflection — side-specific steering away from warn-level IR.
        # Uses classify_ir for exact side; hard avoidance handled above in _tick.
        ir_l, ir_r = self._ir_values()
        ir_cls_w   = self._classify_ir(ir_l, ir_r)
        extra_w    = 0.0
        if ir_cls_w.startswith("warn") or ir_cls_w.startswith("near"):
            self._mark_obstacle_ahead()
            if ir_cls_w in ("warn_left",  "near_left"):   extra_w = -0.4
            elif ir_cls_w in ("warn_right", "near_right"): extra_w = +0.4
            # warn_both / near_both: no extra_w; full avoidance handles it

        drive.steer_toward_heading(desired_th, extra_w=extra_w)

        arrived = (drive.x >= x_target - ARRIVE_TOL if self.mow.going_east
                   else drive.x <= x_target + ARRIVE_TOL)
        if arrived and self.mow.x_waypoint is not None:
            self.mow.x_waypoint = None
        return arrived and self.mow.x_waypoint is None

    def _run_y_shift(self) -> bool:
        from config import CROSS_KP, ARRIVE_TOL
        drive    = self.drive
        x_err    = self.mow.shift_hold_x - drive.x
        desired_vx = _clamp(CROSS_KP * x_err, -0.4, 0.4)
        desired_vy = 1.0
        drive.steer_toward_heading(math.atan2(desired_vy, desired_vx))

        log.info("[Y-SHIFT %d/%d] row %d→%d  y=%.2f→%.2f (need +%.3f)  x=%.2f(hold %.2f err%+.3f)",
                  self.mow.row, self.mow._total_rows - 1,
                  self.mow.row, self.mow.row + 1,
                  drive.y, self.mow.next_row_y,
                  self.mow.next_row_y - drive.y,
                  drive.x, self.mow.shift_hold_x, x_err)

        return drive.y >= self.mow.next_row_y - ARRIVE_TOL

    def _replan_after_avoid(self) -> None:
        from navigation.planner import Grid
        if self.mow.row >= self.grid.rows:
            # Mowing already finished; avoidance completed after the last pass.
            self.mow.row = self.mow._total_rows
            return
        cur_col, _ = self.grid.current_cell(self.drive.x, self.drive.y)
        direction  = self.mow.x_direction
        free = self.grid.find_next_free_col(
            self.mow.row, cur_col + direction, direction)
        if free is not None:
            wx, _ = Grid.cell_centre(free, self.mow.row)
            self.mow.x_waypoint = wx
        else:
            for r in range(self.mow.row + 1, self.grid.rows):
                if self.grid.row_has_free_cells(r):
                    self.mow.skip_to_row(r)
                    return
            self.mow.row = self.mow._total_rows

    # ── Main control loop ─────────────────────────────────────────────

    def run(self) -> None:
        from config import (FIELD_MIN_X, FIELD_MIN_Y, FIELD_MARGIN,
                            LOOP_DT, IMU_WARMUP_TICKS, GRID_SAVE_PATH,
                            DOCK_X, DOCK_Y, DOCK_TOL, BUMP_ACCEL_THRESH)

        start_x = FIELD_MIN_X + FIELD_MARGIN
        start_y = FIELD_MIN_Y + FIELD_MARGIN
        last_t  = time.monotonic()

        self.leds.blue()

        try:
            while True:
                now = time.monotonic()
                dt  = max(1e-6, now - last_t)
                last_t = now

                # ── IMU warm-up ──────────────────────────────────────
                if not self._imu_ready:
                    self._imu_ticks += 1
                    if self._imu_ticks >= IMU_WARMUP_TICKS:
                        self._imu_ready = True
                        log.info("IMU ready — pose (%.2f, %.2f) th=%.1f°",
                                 self.drive.x, self.drive.y,
                                 math.degrees(self.drive.th))
                    time.sleep(LOOP_DT)
                    continue

                # ── Odometry update ──────────────────────────────────
                imu = self.sensors.imu.read()
                # Clamp dt to at most 3× the nominal period before passing to
                # odometry.  Blocking operations (sampler.collect, session.begin,
                # Firebase writes) can stall the loop for tens of seconds.  The
                # accel term in update_odometry is proportional to dt², so even
                # small IMU noise causes a position jump of hundreds of metres
                # when dt is that large.  During a stall the robot is stationary,
                # so discarding the extra time is correct.
                odom_dt = min(dt, LOOP_DT * 3)
                # Pass body-frame forward acceleration so update_odometry can
                # build a secondary IMU dead-reckoning estimate alongside the
                # encoder estimate; the two are averaged into self.drive.x/y.
                self.drive.update_odometry(imu.gyro_z, odom_dt, -imu.accel_y)

                x, y, th = self.drive.x, self.drive.y, self.drive.th
                from config import FIELD_MIN_X, FIELD_MAX_X, FIELD_MIN_Y, FIELD_MAX_Y
                if FIELD_MIN_X <= x <= FIELD_MAX_X and FIELD_MIN_Y <= y <= FIELD_MAX_Y:
                    self.grid.mark_visited(x, y)

                # Compute bump from the already-read IMU to avoid a second I2C read.
                accel_mag = math.hypot(imu.accel_x, imu.accel_y)
                bump = accel_mag > BUMP_ACCEL_THRESH

                # ── State machine ────────────────────────────────────
                try:
                    self._tick(now, dt, start_x, start_y, bump, accel_mag)
                except OSError as io_err:
                    # Transient I2C error (errno 121 EREMOTEIO or errno 11
                    # EAGAIN) — stop motors as a safe fallback and continue
                    # the loop.  A single bad I2C transaction should not
                    # crash the robot; the next tick will retry normally.
                    log.warning(
                        "Transient I2C error in _tick (errno %d: %s) — "
                        "motors stopped, resuming next tick",
                        io_err.errno or 0, io_err,
                    )
                    try:
                        self.drive.stop()
                    except Exception:
                        pass

                # ── Feed latest status to the pusher thread ──────────
                # update() is instant (dict copy under a lock) — no I/O.
                # The StatusPusher thread reads this and pushes every 30 s.
                if hasattr(self, "status_pusher"):
                    self.status_pusher.update(self._build_status())

                # ── Obstacle decay pass ───────────────────────────────
                from config import OBSTACLE_DECAY_S, OBSTACLE_DECAY_CHECK_S
                if (OBSTACLE_DECAY_S > 0
                        and now - self._last_decay_check >= OBSTACLE_DECAY_CHECK_S):
                    self.grid.decay_obstacles(now)
                    self._last_decay_check = now

                # ── Remote command poll ───────────────────────────────
                # Polls Firestore every COMMAND_POLL_S for stop/start
                # commands sent from the laptop dashboard.
                from config import COMMAND_POLL_S
                if (COMMAND_POLL_S > 0
                        and now - self._last_cmd_poll >= COMMAND_POLL_S):
                    self._check_remote_command()
                    self._last_cmd_poll = now

                # ── Periodic grid push to Firestore ──────────────────
                # The grid is normally uploaded only at dock, so the
                # dashboard map stays blank during the entire field run.
                # Push grid/current every GRID_PUSH_INTERVAL_S so the
                # laptop map updates in near-real-time while mowing.
                from config import GRID_PUSH_INTERVAL_S
                if (GRID_PUSH_INTERVAL_S > 0
                        and now - self._last_grid_push >= GRID_PUSH_INTERVAL_S
                        and hasattr(self, 'firebase') and self.firebase):
                    try:
                        self.firebase._save_grid_to_cloud(
                            self.grid, self.session.session_id)
                        self._last_grid_push = now
                    except Exception as _ge:
                        log.debug("Periodic grid push failed: %s", _ge)

                # Telemetry every 5 ticks (~4 Hz at 20 Hz loop)
                self._tick_count += 1
                if self._tick_count % 5 == 0:
                    ir_l_t, ir_r_t = self._ir_values()
                    log.info(
                        "[%s] pos=(%.2f,%.2f) th=%+.0f°  "
                        "IR L=%-5d R=%-5d  "
                        "accel=(%.2f,%.2f,%.2f)m/s²  bump_mag=%.2f/%.0f  "
                        "gyro=(%.1f,%.1f,%.1f)°/s  "
                        "pass=%d/%d  batt=%.0f%%  cov=%.0f%%",
                        self.state.name, x, y, math.degrees(th),
                        ir_l_t, ir_r_t,
                        imu.accel_x, imu.accel_y, imu.accel_z,
                        accel_mag, BUMP_ACCEL_THRESH,
                        math.degrees(imu.gyro_x), math.degrees(imu.gyro_y), math.degrees(imu.gyro_z),
                        self.mow.x_pass_num, self.mow._total_rows,
                        self.battery.percentage,
                        self.grid.coverage_fraction() * 100,
                    )

                time.sleep(max(0, LOOP_DT - (time.monotonic() - now)))

        except KeyboardInterrupt:
            log.info("Keyboard interrupt — initiating emergency shutdown")
            self._shutdown(reason="KeyboardInterrupt")
        except Exception as exc:
            log.exception("Unhandled exception — initiating emergency shutdown: %s", exc)
            self._shutdown(reason=f"Exception: {exc}")
        else:
            # Normal exit (DONE state reached)
            self._shutdown(reason="normal")

    def _tick(self, t: float, dt: float,
               start_x: float, start_y: float, bump: bool,
               accel_mag: float = 0.0) -> None:

        from config import BUMP_ACCEL_THRESH
        drive = self.drive
        # ── Read IR sensors once per tick ─────────────────────────────
        ir_l, ir_r = self._ir_values()
        ir_class = self._classify_ir(ir_l, ir_r, bump)

        # ── Hard obstacle gate — checked BEFORE any state logic ───────
        # The robot must react to a close obstacle regardless of what the
        # grid says, what state it is in, or whether avoidance is already
        # running.  The only exceptions are states where the robot is not
        # moving (SAMPLING, CHARGING, DONE) or already avoiding.
        _moving_states = {
            RobotState.GO_TO_START,
            RobotState.MOWING,
            RobotState.RETURN,
            RobotState.DOCKING,
            RobotState.RESUMING,
        }
        if ir_class.startswith("near") and self.state in _moving_states:
            if self.avoid.phase.name == "NONE":   # don't re-trigger mid-avoid
                self._mark_obstacle_ahead()
                # Pass specific side to AvoidState so it curves correctly
                side = ("left"  if ir_class == "near_left"  else
                        "right" if ir_class == "near_right" else "both")
                self.avoid.begin(t, side)
                drive.stop()
                self.leds.red()
                # Include mowing pass info when obstacle hits during a mow pass
                if self.state == RobotState.MOWING:
                    _phase = self.mow.phase.name
                    _pnum  = (self.mow.x_pass_num if _phase == "X_PASS"
                              else self.mow.row)
                    _ptot  = (self.mow._total_rows if _phase == "X_PASS"
                              else self.mow._total_rows - 1)
                    log.info(
                        "Obstacle! IR L=%d R=%d  bump=%s(mag=%.2f/thresh=%.0f)"
                        "  class=%s side=%s → AVOIDING  [%s %d/%d pos=(%.2f,%.2f)]",
                        ir_l, ir_r, bump, accel_mag, BUMP_ACCEL_THRESH,
                        ir_class, side,
                        _phase, _pnum, _ptot, drive.x, drive.y)
                else:
                    log.info(
                        "Obstacle! IR L=%d R=%d  bump=%s(mag=%.2f/thresh=%.0f)"
                        "  class=%s side=%s → AVOIDING",
                        ir_l, ir_r, bump, accel_mag, BUMP_ACCEL_THRESH,
                        ir_class, side)
                self.state = RobotState.AVOIDING
                return

        # ── INIT ─────────────────────────────────────────────────────
        if self.state == RobotState.INIT:
            drive.stop()
            self.state = RobotState.GO_TO_START
            self.leds.off()
            return

        # ── GO_TO_START ───────────────────────────────────────────────
        if self.state == RobotState.GO_TO_START:
            self._commanding_move = True
            if drive.steer_to_point(start_x, start_y):
                log.info("At start position")
                self._commanding_move = False
                self.session.begin()   # creates Firestore session doc
                self.state = RobotState.MOWING
                self.leds.off()
            return

        # ── MOWING ───────────────────────────────────────────────────
        if self.state == RobotState.MOWING:
            self._commanding_move = True

            # Battery check
            if self.battery.is_low():
                log.warning("Battery low — returning to dock (will resume after charge)")
                self._start_return(reason="battery")
                return

            # Coverage complete
            if self.mow.is_complete:
                log.info("Mowing complete (%.0f%% coverage)",
                         self.grid.coverage_fraction() * 100)
                self._start_return(reason="complete")
                return

            # Sample due
            if self.sampler.is_due(t):
                _phase = self.mow.phase.name
                _pnum  = self.mow.x_pass_num if self.mow.phase.name == "X_PASS" else self.mow.row
                _ptot  = self.mow._total_rows if self.mow.phase.name == "X_PASS" else self.mow._total_rows - 1
                log.info("[SAMPLING] pausing %s %d/%d at (%.2f,%.2f) for sample",
                         _phase, _pnum, _ptot, drive.x, drive.y)
                drive.stop()
                self._commanding_move = False
                self.state = RobotState.SAMPLING
                return

            # Stuck check
            if drive.is_stuck(self._commanding_move):
                log.warning("[STUCK] detected during %s %d/%d at (%.2f,%.2f)",
                            self.mow.phase.name,
                            self.mow.x_pass_num if self.mow.phase.name == "X_PASS" else self.mow.row,
                            self.mow._total_rows, drive.x, drive.y)
                self.stuck.begin(t)
                self.state = RobotState.STUCK
                self.leds.yellow()
                return

            # Normal mowing
            from navigation.planner import MowPhase
            if self.mow.phase == MowPhase.X_PASS:
                # Emit a START log only on the first tick of each new pass.
                if not getattr(self, "_x_pass_announced", None) == self.mow.row:
                    direction_str = "→ EAST" if self.mow.going_east else "← WEST"
                    log.info("[X-PASS %d/%d START] %s  y_target=%.2f  cov=%.0f%%",
                             self.mow.x_pass_num, self.mow._total_rows,
                             direction_str, self.mow.target_y,
                             self.grid.coverage_fraction() * 100)
                    self._x_pass_announced = self.mow.row
                done = self._run_x_pass()
            else:
                if not getattr(self, "_y_shift_announced", None) == self.mow.row:
                    log.info("[Y-SHIFT %d/%d START] row %d → %d  y=%.2f→%.2f",
                             self.mow.row, self.mow._total_rows - 1,
                             self.mow.row, self.mow.row + 1,
                             self.mow.target_y, self.mow.next_row_y)
                    self._y_shift_announced = self.mow.row
                done = self._run_y_shift()

            if done:
                if self.mow.phase == MowPhase.X_PASS:
                    log.info("[X-PASS %d/%d DONE] cov=%.0f%%  pos=(%.2f,%.2f)",
                             self.mow.x_pass_num, self.mow._total_rows,
                             self.grid.coverage_fraction() * 100,
                             self.drive.x, self.drive.y)
                else:
                    log.info("[Y-SHIFT %d/%d DONE] now at y=%.2f",
                             self.mow.row, self.mow._total_rows - 1,
                             self.drive.y)
                if not self.mow.advance():
                    log.info("[MOWING] all %d passes done — full coverage achieved",
                             self.mow._total_rows)
                    self._start_return()
            return

        # ── SAMPLING ─────────────────────────────────────────────────
        if self.state == RobotState.SAMPLING:
            # Blocking — runs to completion within this tick
            record = self.sampler.collect(drive.x, drive.y, drive.th, t)
            self.session.add_sample(record)
            self.state = RobotState.MOWING
            return

        # ── AVOIDING ─────────────────────────────────────────────────
        if self.state == RobotState.AVOIDING:
            done = self.avoid.update(t, drive, self._ir_clear)
            if done:
                self._replan_after_avoid()
                self.state = RobotState.MOWING
                self.leds.off()
            return

        # ── STUCK ────────────────────────────────────────────────────
        if self.state == RobotState.STUCK:
            done = self.stuck.update(t, drive)
            if done:
                self.state = RobotState.MOWING
            return

        # ── RETURN ───────────────────────────────────────────────────
        if self.state == RobotState.RETURN:
            self._commanding_move = True

            if self.returner.follow(drive, drive.x, drive.y):
                log.info("Near dock — starting docking manoeuvre")
                self.state = RobotState.DOCKING
            return

        # ── DOCKING ──────────────────────────────────────────────────
        if self.state == RobotState.DOCKING:
            from config import DOCK_X, DOCK_Y, DOCK_TOL

            docked = drive.reverse_to_point(DOCK_X, DOCK_Y, tol=DOCK_TOL)
            if docked:
                drive.stop()

                # If the dashboard sent a remote stop, park here and wait —
                # do not wait for AC charging to begin.
                if self._remote_stop:
                    log.info("Remote stop: robot parked at dock → PAUSED")
                    self._remote_stop = False
                    drive.set_pose(DOCK_X, DOCK_Y, drive.th)
                    self._push_status_sync({
                        "state":       "PAUSED",
                        "pos_x":        round(drive.x, 3),
                        "pos_y":        round(drive.y, 3),
                        "battery_pct":  round(self.battery.percentage, 1),
                        "coverage_pct": round(self.grid.coverage_fraction() * 100, 1),
                        "sample_count": self.session.sample_count,
                    })
                    self.state = RobotState.PAUSED
                    self.leds.off()
                    return

                # Normal path: enable charging and wait for AC detection.
                self.battery.enable_charging()
                log.info("At dock — charging enabled, waiting for AC contact")

            # Check the PLD pin (GPIO 6): HIGH = AC adapter present = charging.
            # This is instant — no voltage comparison, no 4-second delay.
            # Stop motors immediately — charging contact can occur before the
            # position tolerance is reached (docked may still be False), so
            # drive.stop() inside the docked block may not have run yet.
            if self.battery.is_charging():
                log.info("*** AC power detected — charging confirmed ***")
                drive.stop()
                self.battery.enable_charging()
                drive.set_pose(DOCK_X, DOCK_Y, drive.th)
                self._on_dock()
                self.state = RobotState.CHARGING
                self.leds.blue()
            return

        # ── CHARGING ─────────────────────────────────────────────────
        if self.state == RobotState.CHARGING:
            drive.stop()
            # Check both: SOC threshold AND still plugged in
            if self.battery.is_full() and self.battery.is_charging():
                # Disable charging to protect battery (avoid overcharge)
                self.battery.disable_charging()
                log.info("Charge complete (%.0f%%) — charging disabled",
                         self.battery.percentage)
                self.state = RobotState.DONE
                self.leds.white()
            elif not self.battery.is_charging():
                # AC adapter was unplugged while still charging — warn but wait
                log.warning("AC adapter unplugged during charging (%.0f%%)!",
                            self.battery.percentage)
                self.leds.yellow()
            return

        # ── DONE ─────────────────────────────────────────────────────
        if self.state == RobotState.DONE:
            drive.stop()
            from config import AUTO_RESTART
            if AUTO_RESTART:
                if self._return_reason == "battery" and self._resume_mow is not None:
                    # Robot returned early due to low battery.
                    # Navigate back to where it was interrupted, then resume.
                    log.info("Battery-return: navigating back to interrupted "
                             "position (%.2f, %.2f)", self._resume_x, self._resume_y)
                    self._plan_resume_path()
                    self.state = RobotState.RESUMING
                else:
                    # Natural completion — restart the full field run.
                    log.info("Charge complete — auto-restarting field run")
                    self._begin_new_run()
            return

        # ── RESUMING ──────────────────────────────────────────────────
        if self.state == RobotState.RESUMING:
            # Navigate from dock back to the interrupted mowing position.
            # Obstacle avoidance is active (RESUMING is in _moving_states).
            self._commanding_move = True

            if not self._resume_path or self._resume_idx >= len(self._resume_path):
                # No path or arrived — restore mow state and resume
                self._restore_mow_and_resume()
                return

            tx, ty = self._resume_path[self._resume_idx]
            from config import ARRIVE_TOL, CELL_SIZE
            is_last = (self._resume_idx == len(self._resume_path) - 1)
            tol = ARRIVE_TOL * 2 if is_last else CELL_SIZE * 0.6

            if drive.steer_to_point(tx, ty, tol=tol):
                self._resume_idx += 1
                if self._resume_idx >= len(self._resume_path):
                    self._restore_mow_and_resume()
            return

        # ── PAUSED ───────────────────────────────────────────────────
        if self.state == RobotState.PAUSED:
            # Motors off; command polling in the main loop will call
            # _check_remote_command() which transitions us to GO_TO_START
            # when a "start" command arrives from the dashboard.
            drive.stop()
            return

    # ── Remote command handling ───────────────────────────────────────

    def _check_remote_command(self) -> None:
        """
        Poll Firestore for a dashboard command and act on it.

        "stop"  — valid when the robot is actively working (MOWING,
                  GO_TO_START, SAMPLING, AVOIDING, STUCK).
                  Sets _remote_stop = True and triggers return to dock.
                  The robot parks at DOCK_X, DOCK_Y and enters PAUSED.

        "start" — valid only in PAUSED state.
                  Resets visited cells (keeps obstacles), creates a new
                  session, and resumes the coverage run.
        """
        _moving = {
            RobotState.GO_TO_START, RobotState.MOWING,
            RobotState.SAMPLING, RobotState.AVOIDING, RobotState.STUCK,
        }
        if not hasattr(self, "firebase") or self.firebase is None:
            return

        cmd = self.firebase.poll_command()
        if cmd is None:
            return

        if cmd == "stop" and self.state in _moving:
            log.info("Dashboard command: STOP — returning to dock")
            self._remote_stop = True
            self.firebase.acknowledge_command()
            self._start_return()   # existing method: plan path, → RETURN

        elif cmd == "start" and self.state == RobotState.PAUSED:
            log.info("Dashboard command: START — resuming field run")
            self.firebase.acknowledge_command()
            self._begin_new_run()

        elif cmd is not None:
            log.debug("Ignoring command %r in state %s", cmd, self.state.name)

    # ── New run / restart ─────────────────────────────────────────────

    def _begin_new_run(self) -> None:
        """
        Reset for another complete field run without power-cycling.

        Called from:
          • DONE state (auto-restart after full charge)
          • PAUSED state (dashboard sends "start")

        What resets
        ───────────
          Visited cells → UNKNOWN   (robot re-mows the whole field)
          Obstacle cells → kept     (robot remembers where obstacles are,
                                     subject to normal decay)
          Mow state     → row 0     (restart boustrophedon from the top)
          Session       → new ID    (fresh Firestore session for new data)

        What persists
          grid.json + obstacle timestamps (decay continues correctly)
        """
        from navigation.planner import MowState, CellState
        from data.logger import SessionLogger

        # Reset visited cells; keep obstacles (with their decay timestamps)
        for r in range(self.grid.rows):
            for c in range(self.grid.cols):
                if self.grid.cells[r][c] == CellState.VISITED:
                    self.grid.cells[r][c] = CellState.UNKNOWN

        self.mow = MowState()

        # New session logger with a fresh session ID
        self.session = SessionLogger(firebase_uploader=self.firebase)

        # Disable charging so the HAT doesn't try to charge while moving
        try:
            self.battery.disable_charging()
        except Exception:
            pass

        self.state = RobotState.GO_TO_START
        self.leds.off()
        log.info("New run started — session %s, obstacles retained",
                 self.session.session_id)

    # ── On-dock handler ───────────────────────────────────────────────

    def _on_dock(self) -> None:
        """
        Bulk-upload everything to Firebase, purge local data, save grid.
        Called once when charging is confirmed at the dock.
        """
        from config import GRID_SAVE_PATH

        # Push a definitive "CHARGING" status before the upload starts
        self._push_status_sync({
            "state":       "CHARGING",
            "pos_x":        round(self.drive.x, 3),
            "pos_y":        round(self.drive.y, 3),
            "heading_deg":  round(math.degrees(self.drive.th), 1),
            "battery_pct":  round(self.battery.percentage, 1),
            "coverage_pct": round(self.grid.coverage_fraction() * 100, 1),
            "sample_count": self.session.sample_count,
        })

        # ── Bulk Firebase upload ───────────────────────────────────────
        # session.finalise() calls firebase.batch_upload_session() which
        # uploads all samples, images, and the grid in one pass.
        log.info("Docked — starting bulk Firebase upload…")
        upload_ok = self.session.finalise(self.grid)

        # ── Purge local data ───────────────────────────────────────────
        # Always purge images (they are large); keep session JSON if upload
        # failed so it can be retried on the next charging cycle.
        if upload_ok:
            log.info("Upload successful — purging local sample data & images")
            self.session.purge()
        else:
            log.warning("Upload incomplete — keeping local JSON for next retry")
            # Still delete local images to free space; Firestore data
            # will be incomplete but local JSON is the backup.
            from config import IMAGE_SAVE_DIR
            from pathlib import Path
            img_dir = Path(IMAGE_SAVE_DIR)
            if img_dir.exists():
                deleted = 0
                for f in img_dir.glob("*.jpg"):
                    try: f.unlink(); deleted += 1
                    except OSError: pass
                log.info("Freed %d images from local storage", deleted)

        # ── Always save grid locally ───────────────────────────────────
        # grid.json is the robot's memory — preserved regardless of
        # whether the upload succeeded.
        self.grid.save(GRID_SAVE_PATH)
        log.info("Grid saved locally to %s", GRID_SAVE_PATH)

    def _start_return(self, reason: str = "complete") -> None:
        """
        Begin the return journey to the dock.

        Parameters
        ----------
        reason : "battery" | "complete"
            "battery"  — robot ran low mid-mow; save position + MowState so
                         we can navigate back and resume after charging.
            "complete" — natural end of coverage; full restart after charging.
        """
        import copy
        self._return_reason = reason

        if reason == "battery":
            # Snapshot exactly where mowing was interrupted.
            # copy() is safe — MowState only contains primitives + one Optional[float].
            self._resume_x   = self.drive.x
            self._resume_y   = self.drive.y
            self._resume_mow = copy.copy(self.mow)
            log.info("Battery return: saving resume point (%.2f, %.2f) row=%d phase=%s",
                     self._resume_x, self._resume_y,
                     self.mow.row, self.mow.phase.name)

        self.session._flush_local(self.grid)
        self.returner.plan(self.grid, self.drive.x, self.drive.y)
        self.state = RobotState.RETURN
        self.leds.off()
        log.info("Heading back to dock (reason=%s)", reason)

    def _plan_resume_path(self) -> None:
        """
        Plan a Dijkstra path from the dock to the saved mowing position.
        Called from DONE state when returning after a battery-low event.
        Populates self._resume_path and resets self._resume_idx.
        """
        from navigation.planner import dijkstra, Grid
        from config import DOCK_X, DOCK_Y, INFLATE_RADIUS

        # Disable charging: robot is about to move
        try:
            self.battery.disable_charging()
        except Exception:
            pass

        inflated   = self.grid.inflated(INFLATE_RADIUS)
        start_cell = inflated.current_cell(DOCK_X, DOCK_Y)
        goal_cell  = inflated.current_cell(self._resume_x, self._resume_y)

        if not inflated.is_passable(*start_cell):
            start_cell = inflated.nearest_passable(DOCK_X, DOCK_Y) or start_cell
        if not inflated.is_passable(*goal_cell):
            goal_cell = inflated.nearest_passable(self._resume_x, self._resume_y) or goal_cell

        path = dijkstra(inflated, start_cell, goal_cell)
        if path is None:
            # Fall back to uninflated grid
            path = dijkstra(self.grid,
                            self.grid.current_cell(DOCK_X, DOCK_Y),
                            self.grid.current_cell(self._resume_x, self._resume_y))

        if path:
            self._resume_path = [Grid.cell_centre(c, r) for c, r in path]
            # Final waypoint is the exact interrupted world position
            self._resume_path.append((self._resume_x, self._resume_y))
        else:
            log.warning("Resume path planning failed — going direct")
            self._resume_path = [(self._resume_x, self._resume_y)]

        self._resume_idx = 0
        log.info("Resume path: %d waypoints to (%.2f, %.2f)",
                 len(self._resume_path), self._resume_x, self._resume_y)

        # New session for post-resume data (visited cells are NOT reset)
        from data.logger import SessionLogger
        self.session = SessionLogger(firebase_uploader=self.firebase)

    def _restore_mow_and_resume(self) -> None:
        """
        Called when RESUMING completes (robot has arrived at the interrupted
        mowing position).  Restores MowState and transitions to MOWING.
        """
        log.info("Arrived at resume position (%.2f, %.2f) — restoring mow state",
                 self.drive.x, self.drive.y)

        if self._resume_mow is not None:
            self.mow = self._resume_mow
            self._resume_mow = None

        # Start collecting data again
        self.session.begin()

        # Reset bookkeeping for next battery event
        self._return_reason = "complete"
        self._resume_path   = []
        self._resume_idx    = 0

        self.state = RobotState.MOWING
        log.info("Mowing resumed from row %d, phase %s, going_east=%s",
                 self.mow.row, self.mow.phase.name, self.mow.going_east)

    # ── Grid cloud download ──────────────────────────────────────────

    def _load_grid_from_cloud(self) -> None:
        """
        Download the latest grid from Firestore and apply it.
        Called from _init_navigation() only when no local grid.json exists.
        _init_data() runs before _init_navigation() so self.firebase is
        always initialised when this method is reached.
        """
        if self.firebase is None:
            log.info("Firebase not configured — skipping cloud grid check")
            return

        cloud_grid = self.firebase.download_grid()
        if cloud_grid is None:
            log.info("No cloud grid available — starting with blank grid")
            return

        from config import GRID_ROWS, GRID_COLS, GRID_SAVE_PATH
        from navigation.planner import CellState
        try:
            rows = cloud_grid.get("rows", GRID_ROWS)
            cols = cloud_grid.get("cols", GRID_COLS)
            cells = cloud_grid.get("cells", [])
            for r in range(min(rows, self.grid.rows)):
                for c in range(min(cols, self.grid.cols)):
                    self.grid.cells[r][c] = CellState(cells[r][c])
            # Save locally so next boot uses the local file
            self.grid.save(GRID_SAVE_PATH)
            log.info(
                "Cloud grid applied (%.0f%% coverage) and saved locally",
                self.grid.coverage_fraction() * 100,
            )
        except Exception as exc:
            log.warning("Failed to apply cloud grid: %s", exc)

    # ── Status push helpers ───────────────────────────────────────────

    def _build_status(self) -> dict:
        """Assemble the current robot status payload."""
        return {
            "state":        self.state.name,
            "pos_x":         round(self.drive.x, 3),
            "pos_y":         round(self.drive.y, 3),
            "heading_deg":   round(math.degrees(self.drive.th), 1),
            "battery_pct":   round(self.battery.percentage, 1),
            "coverage_pct":  round(self.grid.coverage_fraction() * 100, 1),
            "sample_count":  self.session.sample_count,
        }

    def _push_status_sync(self, status: dict) -> None:
        """
        Synchronous status push — used at dock where blocking is acceptable.
        Ensures the dashboard shows the correct state before the upload begins.
        """
        if hasattr(self, 'firebase') and self.firebase:
            self.firebase.push_status(status)

    # ── Shutdown ──────────────────────────────────────────────────────

    def _shutdown(self, reason: str = "unknown") -> None:
        """
        Safe shutdown sequence, called on every exit path.

        Order
        ─────
        1. Stop all motors immediately (safety first).
        2. LED → red to signal unexpected stop, or off for normal.
        3. Flush local JSON so samples are on disk.
        4. Attempt emergency Firebase upload (synchronous, with timeout).
        5. Save grid locally.
        6. Release GPIO resources.
        """
        log.info("=== Shutdown triggered: %s ===", reason)

        # ── Step 1: stop everything physical ──────────────────────────
        try:
            self.drive.stop()
        except Exception:
            pass
        try:
            self.cam_arm.stop()
            self.soil_probe.stop()
        except Exception:
            pass
        try:
            self.solenoids.latch_both()
        except Exception:
            pass

        # ── Step 2: LED feedback ───────────────────────────────────────
        try:
            if reason == "normal":
                self.leds.off()
            else:
                self.leds.red()   # unexpected stop — make it visible
        except Exception:
            pass

        # ── Step 3 + 4: emergency upload ──────────────────────────────
        self._emergency_upload(reason)

        # ── Step 5: Stop the status pusher thread ────────────────────────
        if hasattr(self, "status_pusher"):
            try:
                self.status_pusher.stop()
                self.status_pusher.join(timeout=5)
            except Exception:
                pass

        # ── Step 6: GPIO cleanup ───────────────────────────────────────
        try:
            self.drive.cleanup()
        except Exception:
            pass
        try:
            if hasattr(self, "battery"):
                self.battery.cleanup()
        except Exception:
            pass

        try:
            self.leds.off()
        except Exception:
            pass

        log.info("FieldBot shutdown complete")

    def _emergency_upload(self, reason: str) -> None:
        """
        Best-effort Firebase upload on any exit — normal, interrupt, or crash.

        Sequence
        ────────
        1. Flush session data to local JSON (instant, always succeeds).
        2. Push a STOPPED status document to Firestore so the dashboard
           shows the robot went offline and why.
        3. Call session.finalise(grid) which runs batch_upload_session().
           This is the same code path as the normal dock upload.
        4. Purge local images if upload succeeded.
        5. Always save grid.json locally regardless of upload outcome.

        A threading.Timer enforces a hard 90-second deadline — if Firebase
        is unreachable the program will not hang indefinitely.
        """
        import threading
        from config import GRID_SAVE_PATH

        # Nothing to do if data layer was never initialised
        if not hasattr(self, "session") or not hasattr(self, "grid"):
            log.warning("Emergency upload skipped — session/grid not initialised")
            return

        log.info("Emergency upload starting (90 s deadline)…")

        # ── Flush local JSON first — always succeeds ───────────────────
        try:
            self.session._flush_local(self.grid)
            log.info("Local session JSON flushed")
        except Exception as exc:
            log.warning("Local flush failed: %s", exc)

        # ── Push STOPPED status to Firestore (best-effort) ────────────
        self._push_status_sync({
            "state":       "STOPPED",
            "stop_reason": reason,
            "pos_x":        round(getattr(getattr(self, "drive", None), "x", 0), 3),
            "pos_y":        round(getattr(getattr(self, "drive", None), "y", 0), 3),
            "battery_pct":  round(getattr(self.battery, "percentage", 0)
                                  if hasattr(self, "battery") else 0, 1),
            "coverage_pct": round(self.grid.coverage_fraction() * 100, 1),
            "sample_count": self.session.sample_count,
        })

        # ── Attempt Firebase bulk upload with a hard timeout ──────────
        upload_ok = False
        upload_done = threading.Event()

        def _do_upload():
            nonlocal upload_ok
            try:
                upload_ok = self.session.finalise(self.grid)
            except Exception as exc:
                log.warning("Emergency upload exception: %s", exc)
            finally:
                upload_done.set()

        upload_thread = threading.Thread(
            target=_do_upload,
            name="emergency-upload",
            daemon=True,
        )
        upload_thread.start()
        finished = upload_done.wait(timeout=90.0)

        if not finished:
            log.warning("Emergency upload timed out after 90 s — "
                        "data preserved in local JSON for next run")
        elif upload_ok:
            log.info("Emergency upload succeeded — purging local images")
            try:
                self.session.purge()
            except Exception as exc:
                log.warning("Purge after emergency upload failed: %s", exc)
        else:
            log.warning("Emergency upload failed — local JSON preserved at %s",
                        self.session.log_path)

        # ── Always save grid locally ───────────────────────────────────
        try:
            self.grid.save(GRID_SAVE_PATH)
            log.info("Grid saved to %s", GRID_SAVE_PATH)
        except Exception as exc:
            log.warning("Grid save failed: %s", exc)