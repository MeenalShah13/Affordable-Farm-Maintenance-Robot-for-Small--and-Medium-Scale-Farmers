"""
hardware/sensors.py — All I²C sensor wrappers, matching integ2.py exactly.

Sensor addresses (from integ2.py test_sensors()):
  Seesaw  0x37 — STEMMA capacitive soil moisture
  HTU21D  0x40 — temperature + humidity
  TSL2561 0x39 — luminosity
  LSM6DSOX 0x6A — IMU (accel + gyro)
  ADS1015  0x48 — ADC → IR distance sensors (ch0=left, ch1=right)

Note: battery sensor at 0x36 is in battery.py (separate I²C device).
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Reading dataclasses
# ─────────────────────────────────────────────────────────────────────

@dataclass
class IMUReading:
    accel_x: float = 0.0   # m s⁻²
    accel_y: float = 0.0
    accel_z: float = 9.81
    gyro_x:  float = 0.0   # rad s⁻¹
    gyro_y:  float = 0.0
    gyro_z:  float = 0.0   # yaw rate — used for heading integration


@dataclass
class EnvReading:
    temperature_c: float = 0.0
    humidity_pct:  float = 0.0


@dataclass
class SoilReading:
    moisture_raw: int   = 0       # capacitance counts (~200–2000)
    temperature_c: float = 0.0   # on-chip temp


@dataclass
class IRReading:
    left_raw:  int = 0    # ADS1015 raw value  (higher = closer)
    right_raw: int = 0


@dataclass
class SensorSnapshot:
    timestamp:    float      = field(default_factory=time.time)
    lux:          float      = 0.0
    imu:          IMUReading  = field(default_factory=IMUReading)
    environment:  EnvReading  = field(default_factory=EnvReading)
    soil:         SoilReading = field(default_factory=SoilReading)
    ir:           IRReading   = field(default_factory=IRReading)


# ─────────────────────────────────────────────────────────────────────
# Individual sensor classes
# ─────────────────────────────────────────────────────────────────────

class LuminositySensor:
    """TSL2561 ambient light sensor at I2C 0x39."""

    def __init__(self, i2c):
        from adafruit_tsl2561 import TSL2561
        self._dev = TSL2561(i2c, address=0x39)
        self._dev.enabled = True
        self._dev.gain = 0   # 1× gain for outdoor use
        log.info("TSL2561 ready")

    def read_lux(self) -> float:
        try:
            val = self._dev.lux
            return float(val) if val is not None else 0.0
        except Exception as e:
            log.warning("TSL2561 error: %s", e)
            return 0.0


class IMUSensor:
    """LSM6DSOX at 0x6A. Auto-calibrates gyro bias on init."""

    def __init__(self, i2c, bias_samples: int = 100):
        from adafruit_lsm6ds.lsm6dsox import LSM6DSOX
        self._dev = LSM6DSOX(i2c)
        self._bias = self._calibrate(bias_samples)
        log.info("LSM6DSOX ready, gyro bias=(%+.4f, %+.4f, %+.4f)",
                 *self._bias)

    def _calibrate(self, n: int) -> tuple:
        bx = by = bz = 0.0
        for _ in range(n):
            try:
                gx, gy, gz = self._dev.gyro
                bx += gx; by += gy; bz += gz
            except Exception:
                pass
            time.sleep(0.005)
        return bx / n, by / n, bz / n

    def read(self) -> IMUReading:
        try:
            ax, ay, az = self._dev.acceleration
            gx, gy, gz = self._dev.gyro
            bx, by, bz = self._bias
            return IMUReading(
                accel_x=ax, accel_y=ay, accel_z=az,
                gyro_x=gx - bx, gyro_y=gy - by, gyro_z=gz - bz,
            )
        except Exception as e:
            log.warning("LSM6DSOX error: %s", e)
            return IMUReading()

    def check_bump(self, threshold: float = 6.0) -> bool:
        r = self.read()
        return math.hypot(r.accel_x, r.accel_y) > threshold


class EnvSensor:
    """
    Adafruit HTU21D temperature + humidity sensor at 0x40.
    Note: integ2.py uses HTU21D (not Si7021).
    """

    def __init__(self, i2c):
        from adafruit_htu21d import HTU21D
        self._dev = HTU21D(i2c)
        log.info("HTU21D ready")

    def read(self) -> EnvReading:
        try:
            return EnvReading(
                temperature_c=self._dev.temperature,
                humidity_pct=self._dev.relative_humidity,
            )
        except Exception as e:
            log.warning("HTU21D error: %s", e)
            return EnvReading()


class SoilSensor:
    """
    Adafruit STEMMA Seesaw soil sensor at 0x37.
    Note: integ2.py uses 0x37, NOT the default 0x36 (which is the battery gauge).
    """

    def __init__(self, i2c):
        from adafruit_seesaw.seesaw import Seesaw
        self._dev = Seesaw(i2c, addr=0x37)
        log.info("STEMMA soil sensor (0x37) ready")

    def read(self) -> SoilReading:
        try:
            return SoilReading(
                moisture_raw=self._dev.moisture_read(),
                temperature_c=self._dev.get_temp(),
            )
        except Exception as e:
            log.warning("Seesaw error: %s", e)
            return SoilReading()


class IRSensors:
    """
    Dual IR distance sensors (Sharp GP2Y0A21YK0F) via ADS1015 at 0x48.
    ch0 = left IR,  ch1 = right IR

    ADS1015 .value scale: 0–32767 maps to 0–4.096 V (default ±4.096 V gain).
    Voltage → distance (cm): 29.988 × V^(−1.173)  [valid 10–80 cm]
    At threshold scale: 1 V ≈ 8000 counts  →  IR_NEAR_THRESH=10000 ≈ 1.25 V ≈ 21 cm

    Reliability notes
    -----------------
    The ADS1015 runs in single-shot mode.  Each .value access:
      1. Writes a new config register (channel + start-conversion bit)
      2. Polls the OS bit until conversion is done  (~1.2 ms at 860 SPS)
      3. Reads the result register
    Reading two channels back-to-back without a gap causes EREMOTEIO (errno 121)
    because the chip hasn't finished step 2 before the next transaction arrives.

    Fixes applied:
      • data_rate=860 SPS  — faster than default 1600, more reliable
      • _INTER_CH_S gap    — let the chip settle before switching channel
      • Per-channel retry  — only retry the failed channel, not the pair
      • Same-tick cache    — all read() callers within one control tick share
                             one physical I2C read, reducing bus load ~3×
      • Stale-value fallback — on total failure return last good reading
                               instead of zeros (zeros = "path clear" = unsafe)
    """

    _INTER_CH_S   = 0.003   # gap between channel reads  (> 1 conversion at 860 SPS)
    _RETRY_WAIT_S = 0.020   # delay before retry         (> 1 conversion + I2C overhead)
    _CACHE_TTL_S  = 0.045   # reuse cached reading within this window (~one 20 Hz tick)

    def __init__(self, i2c):
        from adafruit_ads1x15.ads1015 import ADS1015
        from adafruit_ads1x15.analog_in import AnalogIn
        self._ads  = ADS1015(i2c, address=0x48)
        self._ads.data_rate = 1600
        self._left  = AnalogIn(self._ads, 0)
        self._right = AnalogIn(self._ads, 1)
        self._cache:      IRReading = IRReading()
        self._cache_time: float     = 0.0
        log.info("ADS1015 + IR sensors ready (data_rate=%d SPS)", self._ads.data_rate)

    def _read_one(self, chan, label: str):
        """Read one channel with per-channel retry. Returns int or None on failure."""
        for attempt in range(3):
            try:
                return chan.value
            except OSError as exc:
                if attempt < 2:
                    time.sleep(self._RETRY_WAIT_S)
                else:
                    log.warning("ADS1015 %s channel failed after 3 attempts: %s",
                                label, exc)
        return None

    def read(self) -> IRReading:
        now = time.monotonic()
        if now - self._cache_time < self._CACHE_TTL_S:
            return self._cache   # same tick — avoid redundant I2C reads

        left_val = self._read_one(self._left, "left")
        time.sleep(self._INTER_CH_S)
        right_val = self._read_one(self._right, "right")

        if left_val is not None and right_val is not None:
            self._cache      = IRReading(left_raw=left_val, right_raw=right_val)
            self._cache_time = now
        # else: keep returning the last good reading — safer than returning zeros

        return self._cache

    def is_obstacle_near(self, thresh: int = 1500) -> bool:
        """True when either sensor reads ABOVE thresh (higher = closer)."""
        r = self.read()
        return r.left_raw > thresh or r.right_raw > thresh

    def is_obstacle_warn(self, thresh: int = 1000) -> bool:
        """True when either sensor reads ABOVE thresh."""
        r = self.read()
        return r.left_raw > thresh or r.right_raw > thresh

    def is_clear(self, thresh: int = 800) -> bool:
        """True when BOTH sensors read BELOW thresh (low value = nothing nearby)."""
        r = self.read()
        return r.left_raw < thresh and r.right_raw < thresh


# ─────────────────────────────────────────────────────────────────────
# Facade
# ─────────────────────────────────────────────────────────────────────

class SensorHub:
    """
    Owns all sensors. Call snapshot() for a complete reading.

    Parameters
    ----------
    i2c : board.I2C() object — created once in robot.py and shared.
          Never create a second board.I2C() handle; doing so causes
          BlockingIOError (errno 11) when two handles share the bus.
    """

    def __init__(self, i2c):
        self.lux   = LuminositySensor(i2c)
        self.imu   = IMUSensor(i2c)
        self.env   = EnvSensor(i2c)
        self.soil  = SoilSensor(i2c)
        self.ir    = IRSensors(i2c)

        log.info("SensorHub ready — all 5 sensors online")

    def snapshot(self) -> SensorSnapshot:
        return SensorSnapshot(
            timestamp=time.time(),
            lux=self.lux.read_lux(),
            imu=self.imu.read(),
            environment=self.env.read(),
            soil=self.soil.read(),
            ir=self.ir.read(),
        )

    def check_bump(self, threshold: float = 6.0) -> bool:
        return self.imu.check_bump(threshold)