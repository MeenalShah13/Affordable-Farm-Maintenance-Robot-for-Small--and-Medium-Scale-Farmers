"""
sampling/sampler.py — Orchestrates a complete field sample.

Sampling sequence (Requirement 4)
──────────────────────────────────
1.  Stop the drive motors
2.  Read temperature + humidity (HTU21D)
3.  Motor4 FORWARD 10 s   → capture image (plant from front)
4.  Motor4 REVERSE 20 s   → capture image (plant from behind/other angle)
5.  Motor4 FORWARD 10 s   → return arm to centre position
6.  Motor3 FORWARD at PROBE_DUTY_DOWN until lux < PROBE_LUX_THRESHOLD
    (gravity assists descent — use light duty only)
    Record time taken (probe_t)
7.  Read soil moisture (Seesaw)
8.  Motor3 REVERSE — two-phase ascent:
      Phase 1: PROBE_SOFT_START_DUTY for PROBE_SOFT_START_S  (unstick, no current spike)
      Phase 2: PROBE_DUTY_UP for remaining time               (overcome gravity + friction)
    Source: drive_and_sample.py raise_probe() soft-start pattern
9.  Run AI pipeline on both captured images
10. Store SampleRecord with (x, y) position and all readings

All motor operations in this module are BLOCKING (time.sleep).
The robot is stationary for the entire sampling sequence.
"""

from __future__ import annotations

import math
import os
import time
import uuid
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SampleRecord:
    """Complete data for one sampling stop. JSON-serialisable."""

    sample_id:      str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp:      float = field(default_factory=time.time)

    # Position (metres from origin)
    pos_x:          float = 0.0
    pos_y:          float = 0.0
    heading_deg:    float = 0.0

    # Environment
    temperature_c:  float = 0.0
    humidity_pct:   float = 0.0

    # Soil probe
    soil_moisture:  int   = 0
    soil_temp_c:    float = 0.0
    probe_time_s:   float = 0.0   # how long probe was lowering

    # Luminosity at time of soil probe
    lux_at_probe:   float = 0.0

    # Images
    image_front:    str   = ""    # filepath
    image_back:     str   = ""    # filepath

    # AI results
    ai_front:       Optional[dict] = None
    ai_back:        Optional[dict] = None
    disease_detected: bool         = False
    disease_summary:  str          = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────
# Sampler
# ─────────────────────────────────────────────────────────────────────

class Sampler:
    """
    Runs a complete sampling stop at the robot's current position.

    Dependencies (injected)
    -----------------------
    sensors     : SensorHub   — for env, soil, lux readings
    camera      : Camera      — rpicam-still capture
    cam_arm     : AuxMotor    — Motor 4, camera arm
    soil_probe  : AuxMotor    — Motor 3, soil probe
    ai_pipeline : AIPipeline  — leaf detection + disease classification
    """

    def __init__(self, sensors, camera, cam_arm, soil_probe, ai_pipeline,
                 leds=None):
        self._sensors    = sensors
        self._camera     = camera
        self._arm        = cam_arm
        self._probe      = soil_probe
        self._ai         = ai_pipeline
        self._leds       = leds    # optional StatusLEDs — only lit during probe
        self._last_t:    float = 0.0

    def is_due(self, t: float) -> bool:
        from config import SAMPLE_INTERVAL_S
        return (t - self._last_t) >= SAMPLE_INTERVAL_S

    def collect(self, robot_x: float, robot_y: float,
                robot_th_rad: float, t: float) -> SampleRecord:
        """
        Execute the full sampling sequence.
        BLOCKING — robot must be stopped before calling.
        """
        from config import (
            CAM_ARM_FWD_TIME_S, CAM_ARM_REV_TIME_S, CAM_ARM_RET_TIME_S,
            CAM_ARM_DUTY,
            PROBE_LUX_THRESHOLD, PROBE_MAX_TIME_S,
            PROBE_DUTY_DOWN,
            PROBE_DUTY_UP,
            PROBE_SOFT_START_S,
            PROBE_SOFT_START_DUTY,
        )

        record = SampleRecord(
            timestamp=time.time(),   # wall-clock Unix timestamp for dashboard display
            pos_x=robot_x,
            pos_y=robot_y,
            heading_deg=math.degrees(robot_th_rad),
        )

        log.info("=== SAMPLE %s at (%.2f, %.2f) ===",
                 record.sample_id, robot_x, robot_y)

        # Kill any leftover rpicam-still processes before capturing.
        # Orphaned processes hold the camera device and cause "Failed to
        # queue buffer" errors on the first capture of each sample.
        self._camera.kill_lingering()

        # ── Step 2: temp + humidity ────────────────────────────────────
        env = self._sensors.env.read()
        record.temperature_c = env.temperature_c
        record.humidity_pct  = env.humidity_pct
        log.info("  Env: %.1f°C  %.1f%%RH",
                 env.temperature_c, env.humidity_pct)

        # ── Step 3: camera arm forward 10 s → capture front image ─────
        log.info("  Camera arm: forward %.0fs", CAM_ARM_FWD_TIME_S)
        self._arm.run_forward(CAM_ARM_FWD_TIME_S, CAM_ARM_DUTY)
        front_label    = f"{record.sample_id}_front"
        record.image_front = self._camera.capture(front_label)
        log.info("  Captured front: %s", record.image_front)

        # ── Step 4: camera arm reverse 20 s → capture back image ──────
        log.info("  Camera arm: reverse %.0fs", CAM_ARM_REV_TIME_S)
        self._arm.run_reverse(CAM_ARM_REV_TIME_S, CAM_ARM_DUTY)
        back_label     = f"{record.sample_id}_back"
        record.image_back = self._camera.capture(back_label)
        log.info("  Captured back: %s", record.image_back)

        # ── Step 5: camera arm forward 10 s → return to centre ────────
        log.info("  Camera arm: return to centre %.0fs", CAM_ARM_RET_TIME_S)
        self._arm.run_forward(CAM_ARM_RET_TIME_S, CAM_ARM_DUTY)

        # ── Step 6: lower soil probe ────────────────────────────────────
        # Use PROBE_DUTY_DOWN — gravity assists the descent so only light
        # torque is needed.  This gives a controlled soft landing.
        # (Heavier ascent duty is applied in Step 8.)
        log.info("  Soil probe: lowering @ %.2f duty until lux < %.0f",
                 PROBE_DUTY_DOWN, PROBE_LUX_THRESHOLD)
        self._leds.white()
        lux_condition = lambda: self._sensors.lux.read_lux() < PROBE_LUX_THRESHOLD
        probe_t = self._probe.run_forward_until(
            condition_fn=lux_condition,
            max_time_s=PROBE_MAX_TIME_S,
            duty=PROBE_DUTY_DOWN,
        )
        record.probe_time_s = probe_t
        record.lux_at_probe = self._sensors.lux.read_lux()

        # ── Step 7: read soil moisture ──────────────────────────────────
        time.sleep(0.5)   # brief settle before reading
        soil = self._sensors.soil.read()
        record.soil_moisture = soil.moisture_raw
        record.soil_temp_c   = soil.temperature_c
        log.info("  Soil: moisture=%d  temp=%.1f°C",
                 soil.moisture_raw, soil.temperature_c)

        # ── Step 8: retract soil probe — two-phase ascent ───────────────
        # From drive_and_sample.py raise_probe():
        #   Phase 1 (soft-start): brief lower-duty kick to unstick the
        #     motor from rest under load without a current spike.
        #   Phase 2 (full ascent): high duty to overcome gravity and any
        #     soil friction on the shaft for the remaining time.
        soft_t    = min(PROBE_SOFT_START_S, probe_t)
        remaining = 3 * (probe_t - soft_t)
        log.info("  Soil probe: retracting "
                 "(soft-start %.2f duty × %.2fs → full %.2f duty × %.2fs)",
                 PROBE_SOFT_START_DUTY, soft_t, PROBE_DUTY_UP, remaining)

        self._probe.run_reverse(soft_t, duty=PROBE_SOFT_START_DUTY)
        if remaining > 0:
            self._probe.run_reverse(remaining, duty=PROBE_DUTY_UP)

        self._leds.off()

        # ── Step 9: AI pipeline on both images ─────────────────────────
        log.info("  Running AI pipeline…")
        if os.path.exists(record.image_front):
            result_f = self._ai.run(record.image_front)
            record.ai_front = result_f.to_dict()
        if os.path.exists(record.image_back):
            result_b = self._ai.run(record.image_back)
            record.ai_back  = result_b.to_dict()

        # Aggregate disease detection
        diseased = []
        for res_dict in [record.ai_front, record.ai_back]:
            if res_dict and res_dict.get("is_diseased"):
                diseased.append(res_dict)
        record.disease_detected = len(diseased) > 0
        if diseased:
            top = max(diseased, key=lambda r: r["disease_conf"])
            record.disease_summary = (f"{top['disease_label']} "
                                      f"({top['disease_conf']:.0%})")
            log.warning("  *** DISEASE DETECTED: %s ***", record.disease_summary)
        else:
            record.disease_summary = "healthy"
            log.info("  No disease detected")

        self._last_t = t
        return record
