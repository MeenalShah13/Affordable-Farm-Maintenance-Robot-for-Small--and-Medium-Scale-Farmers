# FieldBot — Raspberry Pi Robot Control System

The heart of the FieldBot autonomous robot. This module runs on a Raspberry Pi 4/5 and implements the complete control loop: navigation, obstacle detection, sampling orchestration, and cloud synchronization.

## Quick Start

### Installation
```bash
pip3 install -r requirements.txt --break-system-packages
```

### Run Robot
```bash
python3 main.py
```

Monitor the log file:
```bash
tail -f robo_data/fieldbot.log
```

## Module Overview

### Core Files

| File | Purpose |
|------|---------|
| **main.py** | Entry point — sets up logging, signal handlers, starts robot loop |
| **robot.py** | State machine (INIT → MOWING → SAMPLING → DOCKING → CHARGING) |
| **config.py** | All hardware constants, tuning parameters, thresholds |

### hardware/

Motor and sensor drivers (Raspberry Pi HAT interface):

| Module | Provides |
|--------|----------|
| **pca_motor.py** | `DriveSystem` (left/right wheel pairs), `AuxMotor` (probe M3, arm M4) |
| **sensors.py** | `SensorHub` — IMU, soil moisture, temperature, humidity, light, IR |
| **peripherals.py** | `BatteryMonitor`, `StatusLEDs`, `Solenoids`, `Camera` |

**Note**: All hardware uses a single shared I2C bus and dual PCA9685 PWM controllers to avoid conflicts.

### navigation/

Path planning and coverage:

| Module | Provides |
|--------|----------|
| **planner.py** | `Grid`, `MowState`, `AvoidState`, `ReturnPlanner`, `StuckRecovery` |

**Features**:
- Boustrophedon (back-and-forth) grid coverage
- Dijkstra path planning around obstacles
- Obstacle decay (cells expire after OBSTACLE_DECAY_S seconds)
- Stuck detection & escape manoeuvres

### sampling/

Image sampling & AI inference pipeline:

| Module | Provides |
|--------|----------|
| **ai_pipeline.py** | `AIPipeline`, `LeafDetector`, `AIResult` |
| **preprocessing.py** | `PreprocessingPipeline` — denoise, contrast, background removal |
| **sampler.py** | `Sampler` — orchestrates arm extension, camera capture, AI inference |
| **model_training.py** | Local convenience script to load + run trained models |

### data/

Cloud sync and logging:

| Module | Provides |
|--------|----------|
| **logger.py** | Logging setup (stdout + file) |
| **firebase_uploader.py** | `FirebaseUploader` — sync to Firestore + Storage |

### dashboard/

Web UI for monitoring & remote control (laptop-side):

- **app.py**: Flask server (serves static HTML)
- **templates/index.html**: Single-page app with map, grid, samples, Firebase realtime binding

---

## State Machine

### States (robot.py:RobotState)

```python
INIT        # Warm up IMU, load persisted grid, init hardware
GO_TO_START # Navigate dock → first mowing position
MOWING      # Boustrophedon coverage loop
SAMPLING    # Stationary: extend arm, capture image, AI inference
AVOIDING    # Obstacle detected: backup + curve manoeuvre
STUCK       # Encoders stalled: reverse + pivot escape
RETURN      # Dijkstra path back to dock
DOCKING     # Reverse into dock, detect charging contact
CHARGING    # On dock: upload data to Firebase, purge images, save grid
DONE        # Idle — grid preserved for next run
PAUSED      # Remote stop via dashboard
RESUMING    # Returning to interrupted mowing position after battery recharge
```

### Key Loop Invariants

- **20 Hz loop** (50ms ticks)
- Each tick: read sensors → update pose/grid → decide action → command motors
- Blocking operations (sampling, Firebase upload) only in CHARGING state
- Obstacle decay runs every OBSTACLE_DECAY_CHECK_S = 30 seconds

---

## Configuration

All constants are in **config.py**. Key sections:

### Encoder Odometry
```python
WHEEL_RADIUS = 0.045       # metres
TRACK_WIDTH = 0.22         # left–right separation
ENCODER_PPR = 3000         # N20 motor encoder pulses
GEAR_RATIO = 100.0         # 100:1 reduction (drive motors)
METRES_PER_TICK = 0.00001874  # calculated from above
```

### Field Geometry
```python
FIELD_W, FIELD_H = 1.0, 1.0       # 1m × 1m field
CELL_SIZE = 0.30                  # 30cm cells → 3×3 grid
DOCK_X, DOCK_Y = -0.55, -0.55     # Dock position
DOCK_TOL = 0.18                   # "Close enough to dock"
```

### Motor Duty (fraction of 0xFFFF)
```python
DRIVE_DUTY_BASE = 0.65    # Normal cruise
DRIVE_DUTY_CREEP = 0.60   # Slow manoeuvre
DRIVE_DUTY_AVOID = 0.55   # Obstacle avoidance
```

### Sampling
```python
SAMPLE_INTERVAL_S = 30.0           # Every 30 seconds
CAM_ARM_FWD_TIME_S = 0.5           # Extend
CAM_ARM_REV_TIME_S = 1.5           # Retract past centre
CAM_ARM_RET_TIME_S = 0.25          # Return to home
PROBE_DUTY_UP = 0.90               # Ascent (gravity fights)
PROBE_DUTY_DOWN = 0.35             # Descent (gravity helps)
```

### IR Obstacle Thresholds
```python
IR_NEAR_THRESH = 22000    # Hard stop + avoid
IR_WARN_THRESH = 20000    # Slow + deflect
IR_CLEAR_MARGIN = 500     # Hysteresis
```

### Battery
```python
BATTERY_LOW_PCT = 30.0       # Return to dock
BATTERY_WARN_PCT = 40.0      # LED warning
CHARGE_COMPLETE_PCT = 95.0   # Considered fully charged
```

### AI Pipeline
```python
MODEL_PATH = "data/best_litecshuffle.keras"
AI_CONFIDENCE_THRESH = 0.60
IMG_PREPROCESS_SIZE = (160, 160)
DISEASE_CLASSES = [...]  # 15 crop disease classes
```

---

## Hardware Architecture

### Motor Wiring (verified in config.py)

**PCA41 (0x41)** — Right side + accessories:
- M1: Back-Right (PWM=3, IN1=4, IN2=5)
- M2: Front-Right (PWM=6, IN1=7, IN2=8)
- LEDs: R=0, G=1, B=2 (all on PCA41)
- Solenoids: LEFT=12, RIGHT=9

**PCA42 (0x42)** — Left side + mechanisms:
- M3: Soil Probe (PWM=0, IN1=1, IN2=2, duty_scale=15× for 1500:1 gear)
- M4: Camera Arm (PWM=3, IN1=4, IN2=5)
- M5: Back-Left (PWM=6, IN1=7, IN2=8)
- M6: Front-Left (PWM=9, IN1=10, IN2=11)

**Drive Groupings** (authoritative):
- LEFT = M5 + M6 (both on PCA42)
- RIGHT = M1 + M2 (both on PCA41)

### I2C Sensors
```python
I2C_BUS = 1
ADDR_SEESAW = 0x37      # Soil moisture (STEMMA, note: not 0x36!)
ADDR_HTU21D = 0x40      # Temp + humidity
ADDR_TSL2561 = 0x39     # Luminosity
ADDR_LSM6DSOX = 0x6A    # IMU (accelerometer + gyro)
ADDR_ADS1015 = 0x48     # ADC for IR sensors
ADDR_BATTERY = 0x36     # MAX17043 fuel gauge (different from soil!)
```

### Encoder Pins (GPIO, BCM numbering)
```python
ENC_M1 = (17, 27)   # Back-right
ENC_M2 = (22, 10)   # Front-right
ENC_M3 = (9, 11)    # Soil probe
ENC_M4 = (23, 24)   # Camera arm
ENC_M5 = (13, 19)   # Back-left
ENC_M6 = (25, 8)    # Front-left
```

---

## Sampling Pipeline

**Triggered every SAMPLE_INTERVAL_S (default 30s) during MOWING state:**

1. **Arm extension** (SAMPLING state, ~3s total):
   - Extend forward (0.5s)
   - Retreat past centre to opposite side (1.5s)
   - Return to home position (0.25s)
   
2. **Camera capture** (during arm repositioning):
   - Front image (autofocus + quality 85)
   - 1.5s settle time
   - Rear image
   - Save to `robo_data/images/`

3. **AI inference** (preprocessing + model):
   - PreprocessingPipeline: denoise → contrast → background removal → resize to 160×160
   - LeafDetector: heuristic (green dominance + edge density) → `(is_leaf, confidence)`
   - DiseaseClassifier (LiteCShuffle): → `(label, confidence, is_diseased)`
   - Store result with thumbnail in Firestore

---

## Obstacle Avoidance

### Detection Methods
1. **IR sensors** (ADS1015, left/right): continuous monitoring in MOWING state
2. **IMU bump** (accelerometer spike >20 m/s²): hard collision detection
3. **Stuck detection** (encoders stalled but duty > 0): escape manoeuvre

### Avoidance FSM (avoid.py:AvoidState)
```
Obstacle detected
  ├─> BACKUP (1.0s at duty 0.45, reverse)
  └─> CURVE (2.0s at duty 0.55, turn away)
         └─> Resume MOWING once clear
```

**Obstacle cell** marked at OBSTACLE_MARK_DIST = 0.15m ahead.  
**Obstacle decay**: Cells expire after OBSTACLE_DECAY_S = 3600s (1 hour) unless re-confirmed.

---

## Stuck Recovery

**Triggered when**:
- Motor duty > 0 for STUCK_CHECK_ITERS (20 iterations / 1 second)
- But encoder ticks < STUCK_TICK_THRESH (5 ticks per check)

**Escape sequence**:
1. Stop and back up
2. Pivot 90° in place
3. Resume forward
4. If still stuck after 3 attempts, log warning and try next cell

---

## Battery Management

**Monitoring** (BatteryMonitor, max17043 fuel gauge at 0x36):
- Voltage (VCELL_REG = 0x02)
- State of Charge % (SOC_REG = 0x04)
- Read every loop tick

**Thresholds**:
- 30% SOC → return to dock immediately (BATTERY_LOW_PCT)
- 40% SOC → LED warning (BATTERY_WARN_PCT)
- 95% SOC → charge complete (CHARGE_COMPLETE_PCT)
- 3.2V → emergency shutdown (BATTERY_CRIT_VOLTAGE)

**Charging control**:
- GPIO 16 (BCM) = charge control (LOW = enabled)
- GPIO 6 (BCM) = power-loss detection (HIGH = AC present)

---

## Firebase Integration

**On-dock (CHARGING state)**:

1. **Authenticate** using service account JSON (fieldbot/data/firebase_service_account.json)
2. **Upload images**:
   - Full-resolution to Firebase Storage (`gs://project.appspot.com/fieldbot-01/images/...`)
   - Thumbnail (640×360 @ quality 70, base64) embedded in Firestore doc
3. **Push sample records** to Firestore:
   - Collection: `robots/{ROBOT_ID}/samples`
   - Fields: timestamp, image_path, is_leaf, leaf_conf, disease_label, disease_conf, is_diseased
4. **Sync grid state** to Firestore:
   - Collection: `robots/{ROBOT_ID}/grids`
   - Latest visited/obstacle/boundary cells
5. **Purge old images** from Pi storage (keep only last N)

**Periodic cloud updates** (during MOWING):
- Grid position every GRID_PUSH_INTERVAL_S = 60s
- Robot status every FIREBASE_STATUS_INTERVAL_S = 30s
- Command polling every COMMAND_POLL_S = 10s

---

## Local Data Storage

All data persisted in `robo_data/` directory:

```
robo_data/
├── fieldbot.log              # Rotating log (stdout + file)
├── session.json              # Current session metadata
├── grid.json                 # Visited/obstacle cells (persisted across runs)
└── images/                   # Captured leaf samples (deleted on dock upload)
    ├── 2024-12-03_14-30-45_front.jpg
    └── 2024-12-03_14-30-45_rear.jpg
```

---

## Running on Boot (Systemd)

Create `/etc/systemd/system/fieldbot.service`:

```ini
[Unit]
Description=FieldBot Autonomous Robot
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/fieldbot
ExecStart=/usr/bin/python3 /home/pi/fieldbot/main.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable fieldbot.service
sudo systemctl start fieldbot.service
sudo journalctl -u fieldbot.service -f  # Monitor
```

---

## Debugging

### Enable Debug Logging
```bash
export FIELDBOT_LOG_LEVEL=DEBUG
python3 main.py
```

### Check Sensor Readings
```python
from hardware.sensors import SensorHub
import board
imu = SensorHub(board.I2C()).imu
print(imu.acceleration, imu.gyro)
```

### Test Motors
```python
from hardware.pca_motor import DriveSystem
import board
from adafruit_pca9685 import PCA9685
pca41 = PCA9685(board.I2C(), address=0x41)
pca42 = PCA9685(board.I2C(), address=0x42)
drive = DriveSystem(pca41, pca42)
drive.left(0.5)   # 50% duty, forward
drive.right(0.3)  # 30% duty, forward
```

### Monitor Odometry
Check `robo_data/fieldbot.log` for pose updates:
```
[INFO] Pose: x=0.123 y=-0.456 θ=1.234
```

---

## Troubleshooting

### Robot Spins Instead of Going Straight
**Cause**: Motor groupings mixed (LEFT/RIGHT sides paired wrong).  
**Fix**: Verify `config.py` LEFT/RIGHT motor tuples match your physical wiring. Run integration test.

### Camera Timeouts
**Cause**: rpicam-still slow to autofocus or return to ready state.  
**Symptoms**: "Failed to queue buffer" / "Device timeout" in logs.  
**Fix**: Increase `CAMERA_TIMEOUT_MS` (default 3000) or `CAMERA_SETTLE_S` (default 1.5).

### Soil Probe Won't Move
**Cause**: High-torque motor (1500:1) stalls; gravity + soil friction exceed starting torque.  
**Symptoms**: M3 stops mid-descent or mid-ascent.  
**Fix**: Enable `PROBE_DITHER_S` (brief reverse kick) and `PROBE_TAP_N` (ON/OFF pulses) to break stiction. Reduce `PROBE_MAX_TIME_S` if stuck detection fires repeatedly.

### Firebase Upload Fails
**Cause**: Service account JSON missing, expired, or wrong project.  
**Symptoms**: Exception in logs: "Failed to initialize Firebase App".  
**Fix**: Regenerate key in Firebase console, save to `data/firebase_service_account.json`, restart robot.

### Low Battery Not Triggering Return
**Cause**: MAX17043 fuel gauge not reading or threshold misconfigured.  
**Symptoms**: Robot keeps mowing below 30% SOC.  
**Fix**: Check battery voltage: `cat /sys/kernel/debug/regulator/battery` (if available). Recalibrate max17043 or lower `BATTERY_LOW_PCT`.

---

## Performance Tuning

### Increase Coverage Speed
```python
DRIVE_DUTY_BASE = 0.75  # 75% instead of 65%
LOOP_HZ = 25             # 25 Hz instead of 20 Hz
```

### Improve Localization Accuracy
```python
GYRO_BIAS_SAMPLES = 200  # More IMU warmup samples
```

### Faster Obstacle Detection
```python
IR_DEBUG_LOG = True      # Log every sensor reading (for tuning thresholds)
OBSTACLE_DECAY_S = 1200  # Expire obstacles after 20 min instead of 1 hour
```

### Reduce Sampling Time
```python
CAM_ARM_FWD_TIME_S = 0.3
CAM_ARM_REV_TIME_S = 1.0
CAM_ARM_RET_TIME_S = 0.15
CAMERA_TIMEOUT_MS = 2000
```

---

## Testing & Integration

See **root README.md** for:
- Training AI models with **ai-model/model_training.py**
- Running web dashboard with **fieldbot/dashboard/app.py**

---

## License

GNU Affero General Public License v3 License. See root LICENSE file.
