"""
hardware/peripherals.py — Battery monitor, LEDs, solenoids, and camera.

BatteryMonitor  — Suptronics x120x UPS HAT (MAX17040 fuel gauge)
StatusLEDs      — RGB LED on PCA 0x41
Solenoids       — latch solenoids on PCA 0x41
Camera          — Raspberry Pi Camera via rpicam-still

Battery hardware reference: github.com/suptronics/x120x
─────────────────────────────────────────────────────────
  MAX17040 at I²C 0x36 (smbus2)
    Register 0x02 → cell voltage  (12-bit, × 1.25 mV per bit)
    Register 0x04 → state-of-charge (MSB = integer %, LSB = fraction/256)
    Register 0x06 → QuickStart command (write 0x4000 to force SOC recalc)
    Register 0xFE → PowerOnReset      (write 0x0054 to reset IC)

  GPIO 6  (BCM) — PLD pin (Power Loss Detection)
    HIGH = AC adapter present  →  robot is at dock and CHARGING
    LOW  = AC adapter removed  →  robot is running on battery

  GPIO 16 (BCM) — Charging ON/OFF control
    LOW  (output, drive-low)  →  charging ENABLED
    HIGH (output, drive-high) →  charging DISABLED
    Default on boot: HIGH (disabled) — must explicitly enable charging.

Charging detection strategy
────────────────────────────
  Reading the PLD GPIO pin is instant and reliable — no voltage comparison,
  no 4-second window, no false positives.  is_charging() returns True the
  moment the AC adapter is plugged in, and False the moment it is removed.
"""

from __future__ import annotations

import logging
import os
import time

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Battery monitor  (x120x UPS HAT)
# ─────────────────────────────────────────────────────────────────────

class BatteryMonitor:
    """
    Reads voltage and state-of-charge from the MAX17040 fuel gauge on the
    Suptronics x120x UPS HAT, and detects charging via the PLD GPIO pin.

    Methods
    -------
    read()           → (voltage_V, soc_pct)  — single I²C transaction
    voltage          → float V
    percentage       → float %
    is_charging()    → bool  — True when AC adapter is plugged in (GPIO 6 HIGH)
    is_low()         → bool  — True when SOC < BATTERY_LOW_PCT
    is_critical()    → bool  — True when voltage < BATTERY_CRIT_VOLTAGE
    enable_charging()  — set GPIO 16 LOW  (start charging)
    disable_charging() — set GPIO 16 HIGH (stop charging)
    """

    def __init__(self):
        from config import (
            BATTERY_I2C_ADDR, BATTERY_PLD_PIN, BATTERY_CHG_PIN,
        )
        import smbus2
        from gpiozero import Button, OutputDevice

        self._addr   = BATTERY_I2C_ADDR
        self._bus    = smbus2.SMBus(1)

        # PLD pin: pulled HIGH on the HAT when AC adapter is present.
        # Using gpiozero Button with pull_up=None because the HAT supplies
        # its own pull-up; we just read the logic level.
        # active_high=True means .is_active (used internally) = pin HIGH.
        self._pld = Button(
            BATTERY_PLD_PIN,
            pull_up=None,          # HAT provides pull-up — do not add Pi pull-up
            active_state=True,     # True = HIGH = AC present
        )

        # CHG pin: OutputDevice, active_low=True so .on() enables charging
        # (drives pin LOW) and .off() disables charging (drives pin HIGH).
        self._chg = OutputDevice(
            BATTERY_CHG_PIN,
            active_high=False,     # active = LOW = charging ENABLED
            initial_value=False,   # start with charging disabled (pin HIGH)
        )

        # Initialise the fuel gauge (mirrors bat.py PowerOnReset + QuickStart)
        self._power_on_reset()
        self._quick_start()

        log.info(
            "BatteryMonitor ready — %.2fV  %.1f%%  charging=%s",
            self.voltage, self.percentage,
            "YES" if self.is_charging() else "NO",
        )

    # ── Fuel gauge init ───────────────────────────────────────────────

    def _power_on_reset(self) -> None:
        """Send POR command to MAX17040 (resets all registers to defaults)."""
        from config import BATTERY_CMD_REG
        try:
            self._bus.write_word_data(self._addr, BATTERY_CMD_REG, 0x5400)
            time.sleep(0.01)
        except Exception as exc:
            log.warning("MAX17040 PowerOnReset failed: %s", exc)

    def _quick_start(self) -> None:
        """
        Force the MAX17040 to restart its SOC algorithm from scratch.
        Use after power-on or if SOC reading looks wrong.
        """
        from config import BATTERY_MODE_REG
        try:
            self._bus.write_word_data(self._addr, BATTERY_MODE_REG, 0x0040)
            time.sleep(0.125)   # give the IC time to complete measurement
        except Exception as exc:
            log.warning("MAX17040 QuickStart failed: %s", exc)

    # ── Register reads  (bat.py register layout) ──────────────────────

    def read(self) -> tuple[float, float]:
        """
        Read voltage and state-of-charge in a single I²C transaction pair.

        Returns
        -------
        (voltage_V, soc_pct) — both floats, (0.0, 0.0) on read error.

        Register math (from bat.py / MAX17040 datasheet):
          VCELL (0x02): 12-bit value in top 12 bits of 16-bit word.
                        voltage = ((byte0 << 4) | (byte1 >> 4)) * 1.25 mV
          SOC   (0x04): MSB = integer percent, LSB = fractional (n/256).
        """
        from config import BATTERY_VCELL_REG, BATTERY_SOC_REG
        try:
            raw_v   = self._bus.read_i2c_block_data(self._addr, BATTERY_VCELL_REG, 2)
            voltage = ((raw_v[0] << 4) | (raw_v[1] >> 4)) * 0.00125

            raw_s   = self._bus.read_i2c_block_data(self._addr, BATTERY_SOC_REG, 2)
            soc_pct = raw_s[0] + (raw_s[1] / 256.0)

            return voltage, soc_pct
        except Exception as exc:
            log.warning("MAX17040 read error: %s", exc)
            return 0.0, 0.0

    @property
    def voltage(self) -> float:
        v, _ = self.read()
        return v

    @property
    def percentage(self) -> float:
        _, pct = self.read()
        return pct

    # ── Charging detection via PLD GPIO pin ───────────────────────────

    def is_charging(self) -> bool:
        """
        True when the AC adapter is plugged in (GPIO 6 is HIGH).

        This is instant — no voltage comparison, no delay.
        The HAT pulls GPIO 6 HIGH via hardware pull-up when AC is present,
        and lets it float LOW when AC is removed.
        """
        try:
            return self._pld.is_active   # True = pin HIGH = AC present
        except Exception as exc:
            log.warning("PLD pin read error: %s", exc)
            return False

    def ac_present(self) -> bool:
        """Alias for is_charging() — clearer name in some contexts."""
        return self.is_charging()

    # ── Threshold checks ──────────────────────────────────────────────

    def is_low(self) -> bool:
        """True when SOC is below BATTERY_LOW_PCT (robot should return to dock)."""
        from config import BATTERY_LOW_PCT
        return self.percentage < BATTERY_LOW_PCT

    def is_warning(self) -> bool:
        """True when SOC is below BATTERY_WARN_PCT (show yellow LED)."""
        from config import BATTERY_WARN_PCT
        return self.percentage < BATTERY_WARN_PCT

    def is_critical(self) -> bool:
        """True when voltage is critically low (emergency shutdown)."""
        from config import BATTERY_CRIT_VOLTAGE
        return self.voltage < BATTERY_CRIT_VOLTAGE

    def is_full(self) -> bool:
        """True when SOC has reached CHARGE_COMPLETE_PCT."""
        from config import CHARGE_COMPLETE_PCT
        return self.percentage >= CHARGE_COMPLETE_PCT

    # ── Charging control ──────────────────────────────────────────────

    def enable_charging(self) -> None:
        """
        Enable battery charging by driving GPIO 16 LOW.
        Equivalent to: pinctrl set 16 op dl
        """
        try:
            self._chg.on()   # active_high=False → .on() drives pin LOW
            log.info("Battery charging ENABLED (GPIO16 LOW)")
        except Exception as exc:
            log.warning("enable_charging failed: %s", exc)

    def disable_charging(self) -> None:
        """
        Disable battery charging by driving GPIO 16 HIGH.
        Equivalent to: pinctrl set 16 op dh
        """
        try:
            self._chg.off()  # active_high=False → .off() drives pin HIGH
            log.info("Battery charging DISABLED (GPIO16 HIGH)")
        except Exception as exc:
            log.warning("disable_charging failed: %s", exc)

    def cleanup(self) -> None:
        """Release GPIO resources."""
        try:
            self._pld.close()
            self._chg.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────
# LEDs + Solenoids  (PCA 0x41)
# ─────────────────────────────────────────────────────────────────────

class StatusLEDs:
    """
    RGB LED on PCA 0x41.
    Channel numbers read from config: LED_RED_CH, LED_GREEN_CH, LED_BLUE_CH.
    """

    def __init__(self, pca41):
        from config import LED_RED_CH, LED_GREEN_CH, LED_BLUE_CH
        self._pca  = pca41
        self._r_ch = LED_RED_CH
        self._g_ch = LED_GREEN_CH
        self._b_ch = LED_BLUE_CH
        self.off()

    def _set(self, r: bool, g: bool, b: bool) -> None:
        self._pca.channels[self._r_ch].duty_cycle = 0xFFFF if r else 0x0000
        self._pca.channels[self._g_ch].duty_cycle = 0xFFFF if g else 0x0000
        self._pca.channels[self._b_ch].duty_cycle = 0xFFFF if b else 0x0000

    def off(self):       self._set(False, False, False)
    def red(self):       self._set(True,  False, False)
    def green(self):     self._set(False, True,  False)
    def blue(self):      self._set(False, False, True)
    def yellow(self):    self._set(True,  True,  False)
    def cyan(self):      self._set(False, True,  True)
    def white(self):     self._set(True,  True,  True)

    def blink(self, color_fn, times: int = 3, interval_s: float = 0.3):
        for _ in range(times):
            color_fn()
            time.sleep(interval_s)
            self.off()
            time.sleep(interval_s)


class Solenoids:
    """
    Solenoid latch control on PCA 0x41.
    HIGH = energised/unlocked, LOW = latched (default safe state).
    """

    def __init__(self, pca41):
        from config import SOL_LEFT_CH, SOL_RIGHT_CH
        self._pca  = pca41
        self._l_ch = SOL_LEFT_CH
        self._r_ch = SOL_RIGHT_CH
        self.latch_both()   # safe default on init

    def unlock_left(self):  self._pca.channels[self._l_ch].duty_cycle = 0xFFFF
    def lock_left(self):    self._pca.channels[self._l_ch].duty_cycle = 0x0000
    def unlock_right(self): self._pca.channels[self._r_ch].duty_cycle = 0xFFFF
    def lock_right(self):   self._pca.channels[self._r_ch].duty_cycle = 0x0000
    def latch_both(self):   self.lock_left();  self.lock_right()
    def unlock_both(self):  self.unlock_left(); self.unlock_right()


# ─────────────────────────────────────────────────────────────────────
# Camera  (rpicam-still)
# ─────────────────────────────────────────────────────────────────────

class Camera:
    """
    Captures JPEG images using rpicam-still (Raspberry Pi camera stack).

    Key fixes over the naive os.system() approach
    ──────────────────────────────────────────────
    1. Longer --timeout (2000 ms default) so the sensor has time to
       auto-expose before capturing.  500 ms was the root cause of the
       "Failed to queue buffer / Device timeout" errors.

    2. Settle delay between consecutive captures.  The camera hardware
       holds a lock on /dev/video* for a short window after a process
       exits.  Calling rpicam-still again before that releases causes
       I/O errors on the second capture.

    3. Automatic retry with escalating wait.  If rpicam-still exits
       non-zero (including the "attempting a restart" recovery path),
       we wait and try again up to CAMERA_MAX_RETRIES times.

    4. subprocess instead of os.system() so we get the actual exit code
       and stderr for logging, and can enforce a hard Python-side timeout
       as a safety net against hung camera processes.

    5. kill_lingering() kills any orphaned rpicam-still processes before
       the first capture in a sampling session.
    """

    def __init__(self, image_dir: str = None):
        from config import (IMAGE_SAVE_DIR, IMAGE_WIDTH, IMAGE_HEIGHT,
                            IMAGE_QUALITY, CAMERA_TIMEOUT_MS,
                            CAMERA_SETTLE_S, CAMERA_MAX_RETRIES,
                            CAMERA_RETRY_WAIT_S, CAMERA_AUTOFOCUS,
                            CAMERA_FOCUS_WAIT_S, CAMERA_FOCUS_LENS_POS)
        from pathlib import Path
        self._dir         = image_dir or IMAGE_SAVE_DIR
        self._w           = IMAGE_WIDTH
        self._h           = IMAGE_HEIGHT
        self._qual        = IMAGE_QUALITY
        self._timeout_ms  = CAMERA_TIMEOUT_MS
        self._settle_s    = CAMERA_SETTLE_S
        self._max_retry   = CAMERA_MAX_RETRIES
        self._retry_wait  = CAMERA_RETRY_WAIT_S
        self._autofocus   = CAMERA_AUTOFOCUS
        self._focus_wait  = CAMERA_FOCUS_WAIT_S
        self._lens_pos    = CAMERA_FOCUS_LENS_POS
        self._last_cap_t: float = 0.0

        Path(self._dir).mkdir(parents=True, exist_ok=True)
        log.info("Camera ready — dir=%s timeout=%dms AF=%s settle=%.1fs",
                 self._dir, self._timeout_ms,
                 "on" if self._autofocus else "off", self._settle_s)

    # ── Public API ────────────────────────────────────────────────────

    def kill_lingering(self) -> None:
        """
        Kill any orphaned rpicam-still processes left over from a previous
        crash or incomplete capture.  Call once before a sampling session.
        """
        import subprocess
        try:
            result = subprocess.run(
                ["pkill", "-f", "rpicam-still"],
                capture_output=True,
            )
            if result.returncode == 0:
                log.info("Killed lingering rpicam-still process(es)")
                time.sleep(0.5)   # let the kernel release /dev/video*
        except Exception as exc:
            log.debug("kill_lingering: %s", exc)

    def capture(self, label: str) -> str:
        """
        Capture a single JPEG with retry logic.

        Enforces a settle delay since the last capture so consecutive
        calls (front image, then back image) don't race on the device.

        Returns the filepath.  If all retries fail the filepath is still
        returned (caller checks os.path.exists before using it).
        """
        import subprocess

        path = os.path.join(self._dir, f"{label}.jpg")

        # ── Settle delay ───────────────────────────────────────────────
        elapsed = time.monotonic() - self._last_cap_t
        remaining = self._settle_s - elapsed
        if remaining > 0:
            log.debug("Camera settle: waiting %.1fs", remaining)
            time.sleep(remaining)

        # ── Build command ──────────────────────────────────────────────
        # --timeout controls how long rpicam-still runs before capturing.
        # With autofocus enabled we extend it so AF has time to converge,
        # then add an explicit Python-side sleep for focus settle.
        # Without AF the standard timeout covers auto-exposure settle only.
        af_extra_ms = int(self._focus_wait * 1000) if self._autofocus else 0
        effective_timeout_ms = self._timeout_ms + af_extra_ms

        cmd = [
            "rpicam-still",
            "-o",        path,
            "--width",   str(self._w),
            "--height",  str(self._h),
            "--quality", str(self._qual),
            "--timeout", str(effective_timeout_ms),
            "-n",                               # no preview window
            "--nopreview",                      # belt-and-braces on headless Pi
        ]

        if self._autofocus:
            # --autofocus-mode auto     : run one AF scan at start of timeout,
            #                             wait for lock, then expose + capture.
            #                             'continuous' keeps scanning but does NOT
            #                             guarantee focus is locked at capture.
            # --autofocus-on-capture    : trigger another AF cycle right before
            #                             the shutter fires — ensures focus at
            #                             capture even if the first scan drifted.
            cmd += [
                "--autofocus-mode",    "auto",
                "--autofocus-on-capture",
            ]
            # Optional: seed the lens position so AF starts near the expected
            # subject distance instead of scanning from infinity.
            if self._lens_pos > 0.0:
                cmd += ["--lens-position", f"{self._lens_pos:.1f}"]
            log.debug("AF enabled — effective timeout=%d ms (base %d + focus %d ms)",
                      effective_timeout_ms, self._timeout_ms, af_extra_ms)

        # ── Retry loop ─────────────────────────────────────────────────
        # Hard Python-side timeout = effective_timeout_ms + 10 s grace
        hard_timeout = effective_timeout_ms / 1000.0 + 10.0

        for attempt in range(1, self._max_retry + 1):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=hard_timeout,
                )
                self._last_cap_t = time.monotonic()

                if result.returncode == 0 and os.path.exists(path):
                    log.info("Captured [attempt %d]: %s", attempt, path)
                    return path

                # Non-zero exit — log stderr for diagnosis
                stderr_snippet = (result.stderr or "")[-300:]
                log.warning(
                    "rpicam-still attempt %d/%d failed (exit %d): %s",
                    attempt, self._max_retry,
                    result.returncode, stderr_snippet,
                )

            except subprocess.TimeoutExpired:
                log.warning(
                    "rpicam-still attempt %d/%d timed out after %.0fs",
                    attempt, self._max_retry, hard_timeout,
                )
                # Kill the hung process so the next attempt starts clean
                self.kill_lingering()

            except Exception as exc:
                log.warning("rpicam-still attempt %d/%d error: %s",
                            attempt, self._max_retry, exc)

            # Wait before retrying so the camera stack can recover
            if attempt < self._max_retry:
                log.info("Retrying capture in %.1fs…", self._retry_wait)
                time.sleep(self._retry_wait)

        log.error("All %d capture attempts failed for label=%s",
                  self._max_retry, label)
        return path   # caller will see file does not exist