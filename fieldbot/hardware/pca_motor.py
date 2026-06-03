"""
hardware/pca_motor.py — Motor control via PCA9685.

Matches the exact wiring from integ2.py:
  • PCA 0x41 — M1 (back-right), M2 (front-right), LEDs, solenoids
  • PCA 0x42 — M3 (soil-probe), M4 (camera-arm), M5 (back-left), M6 (front-left)

Each motor uses three PCA channels:
  PWM channel  → speed (duty 0–0xFFFF)
  IN1 channel  → direction bit A
  IN2 channel  → direction bit B

Direction convention:
  Forward: IN1=FULL, IN2=OFF
  Reverse: IN1=OFF,  IN2=FULL
  Brake:   IN1=FULL, IN2=FULL   (active low brake, H-bridge dependent)
  Coast:   PWM=OFF (default stop)
"""

from __future__ import annotations
import logging
import math
from typing import Dict

log = logging.getLogger(__name__)


def _duty(fraction: float) -> int:
    """Convert 0.0–1.0 fraction to PCA9685 16-bit duty value (0–0xFFFF)."""
    return int(max(0.0, min(1.0, fraction)) * 0xFFFF)


def _write_channel(pca, channel: int, value: int,
                   retries: int = 2, delay_s: float = 0.02) -> bool:
    """
    Write a duty-cycle value to a single PCA9685 channel with retry.

    OSError errno 121 (EREMOTEIO) means the device did not ACK the I2C
    transaction — typically a momentary power glitch, loose wire, or bus
    noise.  Retrying after a short delay usually succeeds.

    Returns True on success, False if all retries exhausted.
    The caller decides how to handle failure (usually: stop the motor and
    continue rather than crashing the robot).
    """
    import time as _time
    for attempt in range(retries + 1):
        try:
            pca.channels[channel].duty_cycle = value
            return True
        except OSError as exc:
            if attempt < retries:
                _time.sleep(delay_s)
            else:
                log.warning(
                    "PCA channel %d write failed after %d retries: %s "
                    "(errno %d — check I2C wiring, power supply, pull-ups)",
                    channel, retries, exc, exc.errno or 0,
                )
    return False


class PCAMotor:
    """
    Controls a single N20 motor wired to a PCA9685 H-bridge channel.

    All channel writes go through _write_channel() which retries on
    transient EREMOTEIO errors rather than propagating the exception.
    If all retries fail the motor is left in its last state and the
    robot continues — a stalled motor is better than a crashed program.
    """

    def __init__(self, pca, pwm_ch: int, in1_ch: int, in2_ch: int,
                 label: str = ""):
        self._pca   = pca
        self._pwm   = pwm_ch
        self._in1   = in1_ch
        self._in2   = in2_ch
        self._label = label

    def _w(self, ch: int, val: int) -> None:
        """Shorthand: write channel with retry."""
        _write_channel(self._pca, ch, val)

    def forward(self, duty_frac: float = 0.55) -> None:
        """Drive motor forward at duty_frac (0.0–1.0)."""
        self._w(self._in1, 0xFFFF)
        self._w(self._in2, 0x0000)
        self._w(self._pwm, _duty(duty_frac))

    def reverse(self, duty_frac: float = 0.55) -> None:
        """Drive motor in reverse at duty_frac (0.0–1.0)."""
        self._w(self._in1, 0x0000)
        self._w(self._in2, 0xFFFF)
        self._w(self._pwm, _duty(duty_frac))

    def stop(self) -> None:
        """Coast stop — remove all drive."""
        self._w(self._pwm, 0x0000)
        self._w(self._in1, 0x0000)
        self._w(self._in2, 0x0000)

    def brake(self) -> None:
        """Active brake — short motor terminals to hold shaft position."""
        self._w(self._pwm, 0xFFFF)
        self._w(self._in1, 0xFFFF)
        self._w(self._in2, 0xFFFF)

    def set_signed(self, signed_duty: float) -> None:
        """positive = forward, negative = reverse, 0 = stop."""
        if signed_duty > 0:
            self.forward(signed_duty)
        elif signed_duty < 0:
            self.reverse(-signed_duty)
        else:
            self.stop()


class DriveSystem:
    """
    Differential drive system using all four wheel motors.

    Left  side: M5 (back-left,  PCA42) + M6 (front-left,  PCA42)
    Right side: M1 (back-right, PCA41) + M2 (front-right, PCA41)

    Odometry
    --------
    Heading  : integrated from LSM6DSOX gyro-Z (passed in each tick).
    Distance : computed from left/right encoder tick deltas.
    Both are maintained in self.x, self.y, self.th.
    """

    def __init__(self, pca41, pca42):
        """
        Parameters
        ----------
        pca41 : PCA9685 object at address 0x41 (shared, created once in robot.py)
        pca42 : PCA9685 object at address 0x42 (shared, created once in robot.py)

        I2C and PCA objects must be created once and injected here.
        Never call board.I2C() or PCA9685() inside this constructor —
        doing so creates a second handle to the same bus, causing
        BlockingIOError (errno 11) when two handles try to talk simultaneously.
        """
        from gpiozero import RotaryEncoder
        from config import (
            M1_PWM, M1_IN1, M1_IN2,
            M2_PWM, M2_IN1, M2_IN2,
            M5_PWM, M5_IN1, M5_IN2,
            M6_PWM, M6_IN1, M6_IN2,
            ENC_M1, ENC_M2, ENC_M5, ENC_M6,
            METRES_PER_TICK,
            TRACK_WIDTH,
            DRIVE_DUTY_BASE, DRIVE_DUTY_CREEP,
            HEADING_KP, ARRIVE_TOL,
        )
        self._pca41 = pca41
        self._pca42 = pca42

        # Drive motors
        # Left side = M5 (Back-Left) + M6 (Front-Left) — both on PCA 0x42
        # Right side = M1 (Back-Right) + M2 (Front-Right) — both on PCA 0x41
        # Source: drive_and_sample.py "left = M5+M6, right = M1+M2"
        self._left_back  = PCAMotor(self._pca42, M5_PWM, M5_IN1, M5_IN2, "M5-BL")
        self._left_front = PCAMotor(self._pca42, M6_PWM, M6_IN1, M6_IN2, "M6-FL")
        self._right_back  = PCAMotor(self._pca41, M1_PWM, M1_IN1, M1_IN2, "M1-BR")
        self._right_front = PCAMotor(self._pca41, M2_PWM, M2_IN1, M2_IN2, "M2-FR")

        # Encoders — gpiozero RotaryEncoder (efficient interrupt-based)
        # Each encoder is assigned to the motor on the SAME physical side.
        # RotaryEncoder is inherently bidirectional (positive forward, negative
        # backward) so no manual direction correction is needed or applied.
        self._enc_L_back  = RotaryEncoder(*ENC_M5, max_steps=0)  # M5 = Back-Left
        self._enc_L_front = RotaryEncoder(*ENC_M6, max_steps=0)  # M6 = Front-Left
        self._enc_R_back  = RotaryEncoder(*ENC_M1, max_steps=0)  # M1 = Back-Right
        self._enc_R_front = RotaryEncoder(*ENC_M2, max_steps=0)  # M2 = Front-Right

        # Per-encoder prev-tick snapshots (one per motor, not averaged per side).
        # Averaging before delta was computed previously caused cancellation when
        # mirrored motors on the same side count in opposite quadrature directions.
        self._prev_L_back:  int = 0
        self._prev_L_front: int = 0
        self._prev_R_back:  int = 0
        self._prev_R_front: int = 0
        self._left_fwd:  bool = True
        self._right_fwd: bool = True

        # Pose (world frame) — fused average of encoder + IMU estimates
        self.x:  float = 0.0
        self.y:  float = 0.0
        self.th: float = 0.0

        # Separate position accumulators for per-tick kinematic fusion
        self._enc_x: float = 0.0   # encoder dead-reckoning
        self._enc_y: float = 0.0
        self._imu_x: float = 0.0   # IMU kinematic dead-reckoning (per-tick, no cumulative velocity)
        self._imu_y: float = 0.0

        # Stuck detection — snapshot taken at start of each check window,
        # independent of _prev_*_steps used for odometry.
        self._stuck_iters:    int  = 0
        self._stuck_snap_Lb:  int  = 0
        self._stuck_snap_Lf:  int  = 0
        self._stuck_snap_Rb:  int  = 0
        self._stuck_snap_Rf:  int  = 0
        self._stuck_snap_set: bool = False

        self._metres_per_tick    = METRES_PER_TICK
        self._track           = TRACK_WIDTH
        self._kp_heading      = HEADING_KP
        self._duty_base       = DRIVE_DUTY_BASE
        self._duty_creep      = DRIVE_DUTY_CREEP
        self._arrive_tol      = ARRIVE_TOL

        log.info("DriveSystem ready — pose (%.2f, %.2f) th=0°", self.x, self.y)

    # ── Odometry ─────────────────────────────────────────────────────

    def update_odometry(self, gyro_z: float, dt: float,
                        accel_fwd: float = 0.0) -> None:
        """
        Update (x, y, th) each control loop tick.

        gyro_z    : yaw rate from IMU in rad s⁻¹ (bias-corrected)
        dt        : seconds since last call
        accel_fwd : body-frame forward acceleration from IMU (m s⁻²).
                    Used for a secondary dead-reckoning estimate that is
                    averaged with the encoder estimate to reduce error.
        """
        import math

        # 1. Heading from gyro (accurate, immune to wheel slip)
        self.th = _wrap_pi(self.th + gyro_z * dt)

        # 2. Encoder-based distance — compute each encoder's delta independently,
        # take abs() of each, then average the magnitudes per side.
        # Averaging the raw step counts BEFORE differencing caused cancellation
        # when the two motors on the same side are physically mirrored and their
        # encoders therefore count in opposite quadrature directions.
        dL_back  = abs(self._enc_L_back.steps  - self._prev_L_back)  * self._metres_per_tick
        dL_front = abs(self._enc_L_front.steps - self._prev_L_front) * self._metres_per_tick
        dR_back  = abs(self._enc_R_back.steps  - self._prev_R_back)  * self._metres_per_tick
        dR_front = abs(self._enc_R_front.steps - self._prev_R_front) * self._metres_per_tick

        self._prev_L_back  = self._enc_L_back.steps
        self._prev_L_front = self._enc_L_front.steps
        self._prev_R_back  = self._enc_R_back.steps
        self._prev_R_front = self._enc_R_front.steps

        dL = (dL_back + dL_front) / 2.0
        dR = (dR_back + dR_front) / 2.0

        if not self._left_fwd:  dL = -dL
        if not self._right_fwd: dR = -dR

        d_enc = (dL + dR) / 2.0

        # 3. Per-tick kinematic fusion of encoder and IMU.
        #
        # Why NOT cumulative velocity integration:
        #   Integrating accel → velocity → position accumulates sensor noise
        #   without bound.  After a few seconds the IMU position drifts metres
        #   away from reality; averaging it with the correct encoder position
        #   pulls the robot off-course and causes the steering controller to
        #   over-correct (the "going in circles" symptom).
        #
        # What we do instead — per-tick kinematics:
        #   Use the encoder-derived velocity as the known initial velocity for
        #   this tick, then apply the acceleration correction on top.
        #   This resets the "initial velocity" every tick from the encoder,
        #   so IMU noise never accumulates.
        #
        #   d_imu_tick = v_enc * dt + ½ * accel_fwd * dt²
        #              = d_enc       + ½ * accel_fwd * dt²
        #
        #   The second term is the only new information from the IMU; it is
        #   small by design (at 20 Hz and 2 m/s² peak accel it is ~0.1 mm)
        #   but keeps the estimate consistent during acceleration/deceleration.
        v_enc = d_enc / dt if dt > 1e-9 else 0.0
        d_imu_tick = v_enc * dt + 0.5 * accel_fwd * dt * dt

        d_fused = (d_enc + d_imu_tick) / 2.0

        self._enc_x += d_enc    * math.cos(self.th)
        self._enc_y += d_enc    * math.sin(self.th)
        self._imu_x += d_imu_tick * math.cos(self.th)
        self._imu_y += d_imu_tick * math.sin(self.th)

        self.x = (self._enc_x + self._imu_x) / 2.0
        self.y = (self._enc_y + self._imu_y) / 2.0

        log.debug(
            "odom dt=%.3fs dL=%+.5f dR=%+.5f d_enc=%+.5f d_imu=%+.5f d_fused=%+.5f"
            " → enc=(%.3f,%.3f) imu=(%.3f,%.3f) fused=(%.3f,%.3f) th=%+.1f°",
            dt, dL, dR, d_enc, d_imu_tick, d_fused,
            self._enc_x, self._enc_y,
            self._imu_x, self._imu_y,
            self.x, self.y, math.degrees(self.th),
        )

    def is_stuck(self, commanded_moving: bool) -> bool:
        """
        Return True if motors are commanded but cumulative encoder movement
        over STUCK_CHECK_ITERS ticks is below STUCK_TICK_THRESH.

        Uses its own per-encoder snapshots, taken independently of the
        _prev_*_steps used by update_odometry(), so the check window spans
        a full STUCK_CHECK_ITERS ticks rather than a single tick.
        """
        from config import STUCK_TICK_THRESH, STUCK_CHECK_ITERS
        if not commanded_moving:
            self._stuck_iters    = 0
            self._stuck_snap_set = False
            return False

        if not self._stuck_snap_set:
            self._stuck_snap_Lb  = self._enc_L_back.steps
            self._stuck_snap_Lf  = self._enc_L_front.steps
            self._stuck_snap_Rb  = self._enc_R_back.steps
            self._stuck_snap_Rf  = self._enc_R_front.steps
            self._stuck_snap_set = True
            self._stuck_iters    = 0
            return False

        self._stuck_iters += 1
        if self._stuck_iters < STUCK_CHECK_ITERS:
            return False

        # Window elapsed — measure cumulative per-encoder movement
        total_delta = (abs(self._enc_L_back.steps  - self._stuck_snap_Lb) +
                       abs(self._enc_L_front.steps - self._stuck_snap_Lf) +
                       abs(self._enc_R_back.steps  - self._stuck_snap_Rb) +
                       abs(self._enc_R_front.steps - self._stuck_snap_Rf))

        # Reset snapshot for next window
        self._stuck_snap_Lb  = self._enc_L_back.steps
        self._stuck_snap_Lf  = self._enc_L_front.steps
        self._stuck_snap_Rb  = self._enc_R_back.steps
        self._stuck_snap_Rf  = self._enc_R_front.steps
        self._stuck_iters    = 0

        return total_delta < STUCK_TICK_THRESH

    def set_pose(self, x: float, y: float, th: float) -> None:
        self.x, self.y, self.th = x, y, th
        self._enc_x, self._enc_y = x, y
        self._imu_x, self._imu_y = x, y

    # ── Motor commands ───────────────────────────────────────────────

    def set_left_right(self, left_duty: float, right_duty: float) -> None:
        """
        Set signed duty (-1.0 to 1.0) for each side.
        Positive = forward.
        """
        self._left_fwd  = left_duty >= 0
        self._right_fwd = right_duty >= 0

        self._left_back.set_signed(left_duty)
        self._left_front.set_signed(left_duty)
        for m in (self._right_back, self._right_front):
            m.set_signed(right_duty)

    def set_vw(self, v_duty: float, w_duty: float) -> None:
        """
        v_duty (0–1) = forward translation duty
        w_duty (+/-) = rotational correction (+ = turn left/CCW)
        """
        left  = _clamp(v_duty - w_duty, -1.0, 1.0)
        right = _clamp(v_duty + w_duty, -1.0, 1.0)
        self.set_left_right(left, right)

    def stop(self) -> None:
        for m in (self._left_back, self._left_front,
                  self._right_back, self._right_front):
            m.stop()

    # ── Steering ─────────────────────────────────────────────────────

    def steer_toward_heading(self, desired_th: float,
                              extra_w: float = 0.0) -> None:
        """Proportional heading controller. Robot moves while correcting."""
        import math
        err = _wrap_pi(desired_th - self.th)
        w   = _clamp(self._kp_heading * err + extra_w, -0.5, 0.5)
        alignment = math.cos(err)
        v = self._duty_creep + (self._duty_base - self._duty_creep) * max(0.0, alignment)
        self.set_vw(v, w)

    def steer_to_point(self, tx: float, ty: float,
                        tol: float = None) -> bool:
        """Navigate to world point. Returns True when arrived."""
        import math
        tol = tol or self._arrive_tol
        dx, dy = tx - self.x, ty - self.y
        if math.hypot(dx, dy) < tol:
            self.stop()
            return True
        self.steer_toward_heading(math.atan2(dy, dx))
        return False

    def reverse_to_point(self, tx: float, ty: float,
                          tol: float = None) -> bool:
        """Back up toward a point (for docking). Returns True when arrived."""
        import math
        tol = tol or self._arrive_tol * 1.5
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        if dist < tol:
            self.stop()
            return True
        # Desired heading is AWAY from target (reversing toward it)
        reverse_th = _wrap_pi(math.atan2(dy, dx) + math.pi)
        err = _wrap_pi(reverse_th - self.th)
        w   = _clamp(self._kp_heading * err, -0.5, 0.5)
        self.set_vw(-self._duty_creep, w)
        return False

    def cleanup(self) -> None:
        self.stop()
        log.info("DriveSystem cleaned up")


class AuxMotor:
    """
    Wrapper for non-drive motors (M3 soil-probe, M4 camera-arm).
    Provides timed run methods.
    """

    def __init__(self, pca, pwm_ch: int, in1_ch: int, in2_ch: int,
                 enc_pins: tuple = None, label: str = "",
                 duty_scale: float = 1.0,
                 dither_s: float = 0.0,
                 tap_n: int = 0):
        self._motor      = PCAMotor(pca, pwm_ch, in1_ch, in2_ch, label)
        self._label      = label
        self._duty_scale = duty_scale   # gear-ratio compensation (e.g. 15.0 for 1500:1 vs 100:1)
        self._dither_s   = dither_s     # seconds of reverse kick before each forward run (stiction fix)
        self._tap_n      = tap_n        # ON/OFF pulses before each forward run (stiction fix)
        self._enc        = None
        if enc_pins:
            from gpiozero import RotaryEncoder
            self._enc = RotaryEncoder(*enc_pins, max_steps=0)

    def _scale(self, duty: float) -> float:
        """Apply gear-ratio compensation, clamped to [0, 1]."""
        return min(1.0, duty * self._duty_scale)

    def _startup_sequence(self, scaled_duty: float) -> None:
        """
        Break gearbox stiction before a forward run.

        Phase 1 — dither: brief full-power reverse pulse shifts the gear teeth
          off their resting mesh angle so static friction is lower when forward
          torque is applied.
        Phase 2 — tap: rapid ON/OFF pulses deliver current spikes; each is a
          micro-impulse against the remaining stiction.
        """
        import time
        if self._dither_s > 0:
            log.info("[%s] stiction dither: %.0f ms reverse",
                     self._label, self._dither_s * 1000)
            self._motor.reverse(1.0)
            time.sleep(self._dither_s)
            self._motor.stop()
            time.sleep(0.03)
        if self._tap_n > 0:
            log.info("[%s] stiction tap: %d pulses", self._label, self._tap_n)
            for _ in range(self._tap_n):
                self._motor.forward(scaled_duty)
                time.sleep(0.025)   # 25 ms on
                self._motor.stop()
                time.sleep(0.015)   # 15 ms off
            time.sleep(0.03)

    def run_forward(self, duration_s: float, duty: float = 0.6) -> None:
        """Run forward for duration_s seconds (blocking)."""
        import time
        scaled = self._scale(duty)
        self._startup_sequence(scaled)
        log.info("[%s] running forward %.1fs @ %.0f%% (scaled from %.0f%%)",
                 self._label, duration_s, scaled * 100, duty * 100)
        self._motor.forward(scaled)
        time.sleep(duration_s)
        self._motor.stop()

    def run_reverse(self, duration_s: float, duty: float = 0.6) -> None:
        """Run in reverse for duration_s seconds (blocking)."""
        import time
        scaled = self._scale(duty)
        log.info("[%s] running reverse %.1fs @ %.0f%% (scaled from %.0f%%)",
                 self._label, duration_s, scaled * 100, duty * 100)
        self._motor.reverse(scaled)
        time.sleep(duration_s)
        self._motor.stop()

    def run_forward_until(self, condition_fn, max_time_s: float,
                           duty: float = 0.6) -> float:
        """
        Run forward until condition_fn() returns True, then hold position.
        Returns elapsed seconds (needed to reverse by same amount).
        condition_fn : callable → bool
        """
        import time
        scaled = self._scale(duty)
        self._startup_sequence(scaled)
        log.info("[%s] running forward until condition (max %.1fs) @ %.0f%%",
                 self._label, max_time_s, scaled * 100)
        self._motor.forward(scaled)
        t0 = time.monotonic()
        while time.monotonic() - t0 < max_time_s:
            if condition_fn():
                break
            time.sleep(0.05)
        elapsed = time.monotonic() - t0
        self._motor.brake()   # hold position — prevents probe drifting under gravity/soil pressure
        log.info("[%s] condition met after %.2fs (holding)", self._label, elapsed)
        return elapsed

    def hold(self) -> None:
        """Actively brake to hold the current shaft position."""
        self._motor.brake()

    def stop(self) -> None:
        self._motor.stop()


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _wrap_pi(a: float) -> float:
    import math
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))