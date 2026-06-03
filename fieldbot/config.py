"""
config.py — Central configuration for FieldBot.

Source of truth for all hardware constants, thresholds, and tuning
parameters.  Change wiring or behaviour → edit only this file.

Motor groupings verified against drive_and_sample.py:
  left  side = M5 (back-left)  + M6 (front-left)   both on PCA 0x42
  right side = M1 (back-right) + M2 (front-right)  both on PCA 0x41
"""

import math

# ─────────────────────────────────────────────────────────────────────
# Encoder odometry
# ─────────────────────────────────────────────────────────────────────
WHEEL_RADIUS:    float = 0.045    # metres
TRACK_WIDTH:     float = 0.22     # metres (left–right separation)
ENCODER_PPR:     int   = 3000     # pulses per revolution (N20 encoder)
GEAR_RATIO:      float = 100.0    # reference gear ratio (M1, M2, M4, M6)
TICKS_PER_REV:   float = ENCODER_PPR * GEAR_RATIO   # 300000
METRES_PER_TICK: float = (2 * math.pi * WHEEL_RADIUS) / TICKS_PER_REV

# ── Replacement motor (1500:1 gear ratio) ─────────────────────────────
# M3 (soil probe) was replaced with a 1500:1 unit.
# All drive motors (M1, M2, M5, M6) are 100:1 (GEAR_RATIO above).
GEAR_RATIO_M3: float = 1500.0

# M3 duty scale — theoretical (1500/100 = 15×).
# M3 is a mechanism motor (probe only), so theoretical scaling is acceptable.
M3_DUTY_SCALE: float = GEAR_RATIO_M3 / GEAR_RATIO   # 15.0

# ─────────────────────────────────────────────────────────────────────
# Field geometry  (metres)
# ─────────────────────────────────────────────────────────────────────
FIELD_W: float = 1.0
FIELD_H: float = 1.0
FIELD_CENTER_X: float = 0.0
FIELD_CENTER_Y: float = 0.0

FIELD_MIN_X: float = FIELD_CENTER_X - FIELD_W / 2.0
FIELD_MAX_X: float = FIELD_CENTER_X + FIELD_W / 2.0
FIELD_MIN_Y: float = FIELD_CENTER_Y - FIELD_H / 2.0
FIELD_MAX_Y: float = FIELD_CENTER_Y + FIELD_H / 2.0
FIELD_MARGIN: float = 0.25

DOCK_X: float = FIELD_MIN_X - 0.05
DOCK_Y: float = FIELD_MIN_Y + 0.05
DOCK_TOL: float = 0.18    # metres — "close enough to dock"

# ─────────────────────────────────────────────────────────────────────
# Coverage grid
# ─────────────────────────────────────────────────────────────────────
CELL_SIZE: float = 0.30
GRID_COLS: int   = int(FIELD_W / CELL_SIZE)
GRID_ROWS: int   = int(FIELD_H / CELL_SIZE)

# ─────────────────────────────────────────────────────────────────────
# PCA9685 PWM controllers  (from integ2.py)
# ─────────────────────────────────────────────────────────────────────
PCA_FREQ:   int = 50    # 50 Hz — lower than default 60 Hz; gives higher peak current
                        # per pulse for inductive motor loads (more starting torque)
PCA41_ADDR: int = 0x41    # right-side drive motors + LEDs + solenoids
PCA42_ADDR: int = 0x42    # left-side drive motors + M3 (probe) + M4 (arm)

PCA_OFF:  int = 0x0000
PCA_FULL: int = 0xFFFF

# Motor duty levels  (fraction of 0xFFFF)
DRIVE_DUTY_BASE:   float = 0.65   # 65% — normal cruise
DRIVE_DUTY_CREEP:  float = 0.60   # 60% — slow manoeuvre / approach
DRIVE_DUTY_AVOID:  float = 0.55

# ─────────────────────────────────────────────────────────────────────
# Motor wiring  (from integ2.py run_motor calls)
#
# Each motor has three PCA channels: (pwm_ch, in1_ch, in2_ch)
# in1=HIGH / in2=LOW → FORWARD
# in1=LOW  / in2=HIGH → REVERSE
# ─────────────────────────────────────────────────────────────────────

# ── RIGHT side — PCA 0x41 ────────────────────────────────────────────
M1_PCA = "pca41";  M1_PWM = 3;  M1_IN1 = 4;  M1_IN2 = 5   # Back-Right
M2_PCA = "pca41";  M2_PWM = 6;  M2_IN1 = 7;  M2_IN2 = 8   # Front-Right

# ── LEFT side + mechanisms — PCA 0x42 ────────────────────────────────
M3_PCA = "pca42";  M3_PWM = 0;  M3_IN1 = 1;  M3_IN2 = 2   # Soil Probe
M4_PCA = "pca42";  M4_PWM = 3;  M4_IN1 = 4;  M4_IN2 = 5   # Camera Arm
M5_PCA = "pca42";  M5_PWM = 6;  M5_IN1 = 7;  M5_IN2 = 8   # Back-Left
M6_PCA = "pca42";  M6_PWM = 9;  M6_IN1 = 10; M6_IN2 = 11  # Front-Left

# Drive groupings — from drive_and_sample.py (authoritative):
#   left  = M5 (Back-Left)  + M6 (Front-Left)   ← both on PCA 0x42
#   right = M1 (Back-Right) + M2 (Front-Right)  ← both on PCA 0x41
# WARNING: uploaded config had LEFT=M1+M5, RIGHT=M2+M6 — that mixes
# sides and makes the robot spin. The groupings below are correct.
LEFT_MOTORS  = [("pca42", M5_PWM, M5_IN1, M5_IN2),   # Back-Left
                ("pca42", M6_PWM, M6_IN1, M6_IN2)]    # Front-Left
RIGHT_MOTORS = [("pca41", M1_PWM, M1_IN1, M1_IN2),   # Back-Right
                ("pca41", M2_PWM, M2_IN1, M2_IN2)]    # Front-Right

# ─────────────────────────────────────────────────────────────────────
# Encoder GPIO pins  (BCM, from integ2.py)
# ─────────────────────────────────────────────────────────────────────
ENC_M1 = (17, 27)   # back-right
ENC_M2 = (22, 10)   # front-right
ENC_M3 = ( 9, 11)   # soil probe
ENC_M4 = (23, 24)   # camera arm
ENC_M5 = (13, 19)   # back-left
ENC_M6 = (25,  8)   # front-left

# ─────────────────────────────────────────────────────────────────────
# LEDs and Solenoids  (all on PCA 0x41)
# ─────────────────────────────────────────────────────────────────────
LED_RED_CH:   int = 0
LED_GREEN_CH: int = 1
LED_BLUE_CH:  int = 2

SOL_LEFT_CH:  int = 12  # solenoid left
SOL_RIGHT_CH: int = 9   # solenoid right

# ─────────────────────────────────────────────────────────────────────
# I²C sensor addresses  (from integ2.py)
# ─────────────────────────────────────────────────────────────────────
I2C_BUS:         int = 1
ADDR_SEESAW:     int = 0x37   # STEMMA capacitive soil (note: not default 0x36!)
ADDR_HTU21D:     int = 0x40   # temperature + humidity
ADDR_TSL2561:    int = 0x39   # luminosity
ADDR_LSM6DSOX:   int = 0x6A   # IMU
ADDR_ADS1015:    int = 0x48   # ADC (for IR sensors)
ADDR_BATTERY:    int = 0x36   # MAX17043 fuel gauge  ← different from soil!

# ADS1015 channels for IR sensors
ADC_IR_LEFT:  int = 0
ADC_IR_RIGHT: int = 1

# ─────────────────────────────────────────────────────────────────────
# IMU / Localisation
# ─────────────────────────────────────────────────────────────────────
GYRO_BIAS_SAMPLES: int   = 100
BUMP_ACCEL_THRESH: float = 20.0     # m s⁻² — spike → avoidance

# Stuck detection: if motor duty > 0 but ticks < this in N iterations
STUCK_TICK_THRESH: int   = 5       # encoder ticks per check interval
STUCK_CHECK_ITERS: int   = 20      # check over this many loop ticks

# ─────────────────────────────────────────────────────────────────────
# IR obstacle thresholds  (ADS1015 raw value, 0–2048)
# Larger value = closer object (GP2Y type response)
# Calibrate with objects at known distances
# ─────────────────────────────────────────────────────────────────────
IR_NEAR_THRESH:  int = 22000   # HIGHER = closer. Stop + avoid when reading > this
IR_WARN_THRESH:  int = 20000   # Slow + deflect when reading > this
# Clear when BOTH sensors read below IR_WARN_THRESH - margin.
# Calibrate by printing ir.read() at known distances.
# Typical ADS1015 range for this sensor type: ~2000 (very close) → ~100 (open)

IR_CLEAR_MARGIN: int = 500   # hysteresis — only "clear" once both sensors drop this far below IR_WARN_THRESH

# ── IR debug logging ──────────────────────────────────────────────────
# When True, the control loop logs raw IR sensor values every tick.
# Set True to calibrate thresholds; set False during normal operation
# to avoid flooding the log.
IR_DEBUG_LOG: bool = True
OBSTACLE_MARK_DIST: float = 0.15   # metres ahead to mark obstacle cell
INFLATE_RADIUS:  int = 1

# ── Obstacle decay ────────────────────────────────────────────────────
# Obstacle cells revert to UNKNOWN after OBSTACLE_DECAY_S seconds if the
# IR sensor does not re-confirm them.  The robot will re-visit and check
# whether the obstacle is still present on its next coverage pass.
#
# Tune to suit your field:
#   Short decay (60–120 s)  : good for dynamic environments (people, animals)
#   Long decay  (300–600 s) : good for static fields (pots, permanent fixtures)
#   0 = decay disabled (obstacles are permanent for the session)
OBSTACLE_DECAY_S:       float = 3600.0   # seconds before an obstacle cell expires
OBSTACLE_DECAY_CHECK_S: float = 30.0    # how often the decay pass runs (seconds)

# ─────────────────────────────────────────────────────────────────────
# Navigation / steering
# ─────────────────────────────────────────────────────────────────────
HEADING_KP:   float = 2.0
CROSS_KP:     float = 1.8
ARRIVE_TOL:   float = 0.08    # metres

AVOID_BACKUP_TIME_S: float = 1.0
AVOID_CURVE_TIME_S:  float = 2.0
AVOID_BACKUP_DUTY:   float = 0.45
AVOID_CURVE_DUTY:    float = 0.55

# Avoidance turn style (from drive_and_sample.py PIVOT_IN_PLACE):
#   False — gentle arc: slow side runs forward at AVOID_CURVE_DUTY × 0.60.
#           Smooth but needs enough slow-side duty to overcome stiction.
#   True  — pivot in place: slow side reverses at same duty.
#           Near-zero radius. More reliable — 0.35× was observed to stall
#           the slow side and cause the robot to go nearly straight.
PIVOT_IN_PLACE: bool = True

# ─────────────────────────────────────────────────────────────────────
# Sampling  (Requirement 4)
# ─────────────────────────────────────────────────────────────────────
SAMPLE_INTERVAL_S: float = 30.0     # seconds between samples during mow

# Camera arm  (Motor 4)
CAM_ARM_FWD_TIME_S:  float = 0.5   # extend arm forward
CAM_ARM_REV_TIME_S:  float = 1.5   # reverse past centre to other side
CAM_ARM_RET_TIME_S:  float = 0.25  # return to centre
CAM_ARM_DUTY:        float = 0.50

# Soil probe  (Motor 3)
PROBE_LUX_THRESHOLD: float = 20.0
PROBE_MAX_TIME_S:    float = 15.0
# Probe duties are asymmetric — gravity assists descent but fights ascent.
# Source: drive_and_sample.py probe constants.
PROBE_DUTY_DOWN:       float = 0.35   # descent: light duty, gravity helps
PROBE_DUTY_UP:         float = 0.90   # ascent: high torque (gravity + soil friction)
PROBE_SOFT_START_S:    float = 0.20   # brief lower-duty kick before full ascent
PROBE_SOFT_START_DUTY: float = 0.50   # duty during the soft-start window

# Stiction-breaking sequence applied before each probe descent.
# The 1500:1 gearbox has higher static friction than dynamic; the motor cannot
# start under the probe box weight without first breaking the static mesh.
# Sequence: reverse dither → tap pulses → forward run.
# Set either to 0 to disable that phase.
PROBE_DITHER_S: float = 0.12   # seconds of full-power reverse kick before descent
PROBE_TAP_N:    int   = 8      # ON/OFF pulses (25 ms on, 15 ms off) before descent

# ─────────────────────────────────────────────────────────────────────
# Camera  (rpicam-still)
# ─────────────────────────────────────────────────────────────────────
IMAGE_WIDTH:   int = 1920
IMAGE_HEIGHT:  int = 1080
IMAGE_QUALITY: int = 85
IMAGE_SAVE_DIR: str = "robo_data/images"

# Thumbnail settings for Firestore storage (base64-encoded inside each sample doc).
# Full-resolution images are too large for Firestore (1 MB document limit).
# 640×360 @ quality 70 ≈ 20–50 KB per image → 27–67 KB as base64 — well within limits.
IMAGE_THUMB_SIZE:    tuple = (640, 360)
IMAGE_THUMB_QUALITY: int   = 70

# Camera capture settings
# TIMEOUT: milliseconds rpicam-still runs before capturing.
#   500 ms is too short — the camera sensor needs time to auto-expose,
#   especially indoors or in low-contrast scenes.  2000 ms is reliable.
CAMERA_TIMEOUT_MS:   int   = 3000   # ms — must be > AF convergence time (typically 1-2 s)

# Settle delay between consecutive captures (front then back image).
# Without this the camera hardware doesn't fully release between calls,
# causing "Failed to queue buffer" / "Device timeout" errors.
CAMERA_SETTLE_S:     float = 1.5

# Retry settings — if rpicam-still exits non-zero, wait and try again.
CAMERA_MAX_RETRIES:  int   = 3
CAMERA_RETRY_WAIT_S: float = 2.0

# ── Autofocus (Raspberry Pi Camera Module 3 only) ─────────────────────
# The Camera Module 3 has a motorised lens with PDAF (phase-detection AF).
# True  → passes --autofocus-mode auto --autofocus-on-capture to rpicam-still.
#   'auto'         : run a single AF scan at the start of the timeout window
#                    and wait for focus to lock before the shutter fires.
#   'on-capture'   : also triggers AF immediately before the shutter — belt-and-
#                    braces for the stationary-leaf use case.
# False → fixed focus (default lens position) — faster, no AF hardware used.
CAMERA_AUTOFOCUS:       bool  = True
CAMERA_FOCUS_WAIT_S:    float = 2.0   # extra Python-side settle after AF (seconds)
# Lens position hint for manual focus assist.
# 0.0  = let autofocus decide freely (recommended).
# Use dioptres (1/distance_in_metres): leaves at 0.3 m → ~3.3, at 0.5 m → ~2.0.
# Only used when CAMERA_AUTOFOCUS is True — acts as a starting-point hint.
CAMERA_FOCUS_LENS_POS:  float = 0.0   # 0.0 = no hint, full scan

# ─────────────────────────────────────────────────────────────────────
# AI pipeline
# ─────────────────────────────────────────────────────────────────────
MODEL_PATH: str = "data/best_litecshuffle.keras"
AI_CONFIDENCE_THRESH: float = 0.60
IMG_PREPROCESS_SIZE: tuple = (160, 160)   # LiteCShuffle input size

# Plant disease class labels — update to match your training dataset
DISEASE_CLASSES: list = [
    "Bacterial_spot_Pepper_Bell",
    "healthy_Pepper_Bell",
    "Early_blight_Potato",
    "Late_blight_Potato",
    "healthy_Potato",
    "Bacterial_spot_Tomato",
    "Early_blight_Tomato",
    "Late_blight_Tomato",
    "Leaf_Mold_Tomato",
    "Septoria_leaf_spot_Tomato",
    "Spider_mites_Two_spotted_spider_mite_Tomato",
    "Target_Spot_Tomato",
    "YellowLeaf__Curl_Virus_Tomato",
    "mosaic_virus_Tomato",
    "healthy_Tomato",
]

# ─────────────────────────────────────────────────────────────────────
# Battery  (Suptronics x120x UPS HAT — MAX17040 fuel gauge)
#
# Hardware interface (from github.com/suptronics/x120x):
#
#   I²C 0x36         — MAX17040 fuel gauge (voltage + state of charge)
#   GPIO 6  (BCM)    — PLD pin (Power Loss Detection)
#                       HIGH = AC adapter present → robot is charging
#                       LOW  = AC adapter removed → running on battery
#   GPIO 16 (BCM)    — Charging ON/OFF control
#                       LOW  (drive-low)  = charging ENABLED
#                       HIGH (drive-high) = charging DISABLED
# ─────────────────────────────────────────────────────────────────────

# MAX17040 I²C registers (read via smbus2, same as bat.py)
BATTERY_I2C_ADDR:   int   = 0x36   # confirmed — soil sensor is at 0x37
BATTERY_VCELL_REG:  int   = 0x02   # voltage register
BATTERY_SOC_REG:    int   = 0x04   # state-of-charge register
BATTERY_MODE_REG:   int   = 0x06   # QuickStart command register
BATTERY_CMD_REG:    int   = 0xFE   # PowerOnReset command register

# GPIO pins (BCM numbering, from x120x documentation)
BATTERY_PLD_PIN:    int   = 6      # Power Loss Detection — HIGH = AC present
BATTERY_CHG_PIN:    int   = 16     # Charge control — LOW = charging enabled

# Thresholds
BATTERY_LOW_PCT:    float = 30.0   # % SOC — robot returns to dock
BATTERY_WARN_PCT:   float = 40.0   # % SOC — LED warning
CHARGE_COMPLETE_PCT: float = 95.0  # % SOC — considered fully charged
BATTERY_CRIT_VOLTAGE: float = 3.20 # V  — emergency shutdown (from bat.py default)

# ─────────────────────────────────────────────────────────────────────
# Local data paths
# ─────────────────────────────────────────────────────────────────────
DATA_DIR:       str = "robo_data"
GRID_SAVE_PATH: str = "robo_data/grid.json"
SESSION_PATH:   str = "robo_data/session.json"

# ─────────────────────────────────────────────────────────────────────
# Firebase  ← fill these in from your Firebase project console
# ─────────────────────────────────────────────────────────────────────

# Path to the Service Account JSON key file downloaded from:
#   Firebase Console → Project Settings → Service Accounts → Generate New Key
FIREBASE_SERVICE_ACCOUNT_PATH: str = "data/firebase_service_account.json"

# A unique ID for this robot (allows multiple robots in one Firebase project)
ROBOT_ID: str = "fieldbot-01"

FIREBASE_STATUS_INTERVAL_S:    float = 30.0

COMMAND_POLL_S: float = 10.0
GRID_PUSH_INTERVAL_S: float = 60.0   # push grid/current every N seconds while
                                     # mowing so dashboard map updates mid-run
                                     # (grid normally only written at dock)

# AUTO_RESTART : after a full charge cycle (DONE state), automatically start
#                the next field run instead of sitting idle.
#                Visited cells are reset to UNKNOWN; obstacles are kept.
#                Set False if you want the robot to stay docked until you
#                manually restart main.py.
AUTO_RESTART: bool = False

# ─────────────────────────────────────────────────────────────────────
# Main control loop
# ─────────────────────────────────────────────────────────────────────
LOOP_HZ:  int   = 20
LOOP_DT:  float = 1.0 / LOOP_HZ
IMU_WARMUP_TICKS: int = 5
