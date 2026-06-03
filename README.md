# FieldBot — Autonomous Crop Health Monitoring Robot

![Status](https://img.shields.io/badge/status-Active-brightgreen)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A low-cost autonomous ground robot for small-to-medium-scale farms that performs autonomous field coverage (mowing) and collects crop health samples for AI-driven disease detection. The robot operates independently for hours, uploads data to cloud (Firebase), and automatically returns to dock for charging.

## Quick Start

### On Raspberry Pi
```bash
cd fieldbot
pip3 install -r requirements.txt
python3 main.py
```

### On Laptop (Dashboard)
```bash
cd fieldbot/dashboard
pip install -r requirements.txt
python app.py
# Open http://localhost:8080 in browser
```

### Train AI Models
```bash
cd ai-model
python model_training.py              # Multi-model benchmarking
python experiments.py                 # Feature selection & comparisons
```

## Project Structure

```
.
├── fieldbot/                          # Pi-side robot control (Raspberry Pi 4/5)
│   ├── main.py                        # Entry point — starts control loop
│   ├── robot.py                       # State machine (MOWING → SAMPLING → DOCKING, etc.)
│   ├── config.py                      # All hardware constants & tuning params
│   ├── requirements.txt                # Pi dependencies (TensorFlow, PCA9685, sensors)
│   ├── hardware/                      # Motor, sensor, battery, camera drivers
│   ├── navigation/                    # Grid coverage, path planning (Dijkstra)
│   ├── sampling/                      # AI pipeline, preprocessing, sampler orchestration
│   ├── data/                          # Logging, Firebase uploader
│   └── dashboard/                     # Web UI (Flask server + HTML/Firebase)
│
├── ai-model/                          # Model training & evaluation
│   ├── model_training.py              # 9 models: VGG, CNN, ELM, Capsule, etc.
│   ├── preprocessing.py               # Image augmentation & feature extraction
│   ├── data_loader.py                 # HDF5 lazy-loading for large datasets
│   ├── feature_selection.py           # L1/L2 feature filtering
│   ├── experiments.py                 # Benchmarks & comparisons
│   └── outputs/
│       ├── models/                    # Saved .keras and .pkl checkpoints
│       └── cache/                     # HDF5 feature cache
│
├── simulation/                        # Webots simulation (prototyping only)
│   ├── initial_design/                # Early robot design
│   └── intermediate/                  # Intermediate robot simulation
│
└── README.md                          # This file
```

## Features

### Autonomous Navigation
- **Boustrophedon coverage**: Back-and-forth lawn mowing pattern over configurable grid
- **Odometry-based localization**: Encoder ticks → pose estimates with ~5cm accuracy
- **Obstacle detection**: IR rangefinders + IMU bump detection + stuck recovery
- **Dynamic path planning**: Dijkstra routing around obstacles, obstacle decay over time
- **Dock detection**: Charging dock IR beacons for return-to-dock homing

### Crop Health Sampling
- **Automated leaf collection**: Motor-driven camera arm positions leaves for imaging every 30s during mowing
- **Soil probe**: Motorized probe collects subsurface samples (~15m depth max)
- **Multi-modal sensing**: Soil moisture, temperature, humidity, light, battery state

### AI Disease Detection
- **9 trained models** for crop disease classification (Pepper, Tomato, Potato)
- **LiteCShuffle** (deployed): Lightweight attention-based CNN, <50ms inference
- **Preprocessing pipeline**: Denoise (cellular automaton) → contrast (CLAHE) → background removal
- **Heuristic leaf detection**: Green dominance + edge density scoring
- **15 disease classes** + healthy variants for target crops

### Cloud Integration
- **Firebase realtime sync**: Images, sensor readings, grid state, session metadata
- **On-dock auto-upload**: Full-resolution + thumbnail images (base64 in Firestore)
- **Web dashboard**: Realtime map, grid heatmap, sample history, battery status, remote control

## Hardware

### Main Components
| Component | Model | Purpose |
|-----------|-------|---------|
| CPU | Raspberry Pi 4/5 | Main controller |
| Motors (drive) | N20 100:1 + encoders | Wheel propulsion (M1–M6) |
| Motor (probe) | N20 100:1 + encoders | Soil sampling |
| Motor (arm) | N20 100:1 + encoders | Camera positioning |
| Camera | Pi Camera Module 3 | Leaf imaging (autofocus) |
| IMU | LSM6DSOX | Bump detection, heading |
| Soil moisture | STEMMA SEESAW | Subsurface moisture |
| Temp/humidity | HTU21D | Ambient conditions |
| Light | TSL2561 | Luminosity for probe trigger |
| IR rangefinders | 2× GP2Y (via ADS1015) | Obstacle detection |
| Battery | Suptronics x120x UPS | Monitoring + charging HAT |
| PWM controllers | 2× PCA9685 | Motor driver (I2C bus) |

### Pinout & I2C Addresses
See **fieldbot/config.py** for:
- Motor PCA channels (LEFT: M5+M6 on PCA42, RIGHT: M1+M2 on PCA41)
- Encoder GPIO pins (BCM numbering)
- I2C sensor addresses (0x37 soil, 0x40 temp, 0x39 light, 0x6A IMU, 0x48 ADC, 0x36 battery)
- Solenoid/LED channels

## Configuration

All constants are in **fieldbot/config.py**. Common tuning parameters:

```python
# Field geometry (1m × 1m default)
FIELD_W, FIELD_H = 1.0, 1.0
CELL_SIZE = 0.30  # 30cm grid cells → 3×3 coverage

# Motor duty cycles (% of 0xFFFF)
DRIVE_DUTY_BASE = 0.65    # Cruise
DRIVE_DUTY_CREEP = 0.60   # Slow manoeuvre
DRIVE_DUTY_AVOID = 0.55   # Obstacle avoidance

# Sampling
SAMPLE_INTERVAL_S = 30.0   # Sample every 30s during mowing
CAMERA_TIMEOUT_MS = 3000   # Autofocus settle time
PROBE_DUTY_UP = 0.90       # High torque for ascent (gravity fights)
PROBE_DUTY_DOWN = 0.35     # Light duty for descent (gravity helps)

# Battery thresholds
BATTERY_LOW_PCT = 30.0     # Return to dock
BATTERY_WARN_PCT = 40.0    # LED warning
CHARGE_COMPLETE_PCT = 95.0

# IR obstacle thresholds (ADS1015 raw, 0–2048)
IR_NEAR_THRESH = 22000     # Hard stop + avoid
IR_WARN_THRESH = 20000     # Slow + deflect

# AI pipeline
MODEL_PATH = "data/best_litecshuffle.keras"
AI_CONFIDENCE_THRESH = 0.60
IMG_PREPROCESS_SIZE = (160, 160)
DISEASE_CLASSES = [
    "Bacterial_spot_Pepper_Bell",
    "healthy_Pepper_Bell",
    "Early_blight_Potato",
    # ... (15 total)
]
```

## Control Flow

### Robot State Machine (robot.py)
```
INIT
  └─> Warm up IMU, load persisted grid, init hardware
      └─> GO_TO_START (if not at start)
          └─> Navigate dock → first mow position
              └─> MOWING
                  ├─> SAMPLING (every 30s)
                  │   └─> Extend arm, capture image, AI inference, store sample
                  ├─> AVOIDING (if IR detects obstacle)
                  │   └─> Backup + curve manoeuvre
                  └─> STUCK (if encoders stall)
                      └─> Escape sequence (reverse, pivot)
              └─> RETURN (once grid covered or battery low)
                  └─> Dijkstra path back to dock
              └─> DOCKING
                  └─> Reverse into dock, detect charging contact
              └─> CHARGING
                  └─> Upload data to Firebase, purge images, save grid
              └─> DONE (idle until manual restart or AUTO_RESTART=True)
```

**Loop frequency**: 20 Hz (50ms ticks)

## AI Models

Train & evaluate 9 models in **ai-model/model_training.py**:

| Model | Type | Input | Key Features |
|-------|------|-------|--------------|
| SimpleVGG16 | CNN | Images | Transfer learning baseline |
| VGG19_SVM | Hybrid | Images | VGG19 features → RBF-SVM |
| HybridCNN | Custom | Images | Nature 2025 paper architecture |
| KCNet | SNN proxy | Features | Sparse random Kenyon cells (2000×) |
| FlyCaps | Capsule | Images/Features | Firefly-optimised ELM |
| **LiteCShuffle** | **Attention CNN** | **Images** | **Deployed** — lightweight, <10s |
| YangViT | SNN proxy | Images | Winner-take-all lateral inhibition |
| HybridGrasshopperABC | MLP | Features | Grasshopper + ABC (paywall) |
| MantisSearch | ELM | Features | Mantis Search algorithm (paywall) |

**Training from scratch**:
```python
from ai_model.model_training import ModelTrainer

trainer = ModelTrainer(
    model_type="litecshuffle",
    n_classes=15,
    image_input_shape=(160, 160, 3)
)
trainer.run(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
trainer.save("outputs/models")
```

**Evaluation**:
```python
results = trainer.evaluate(X_test, y_test)
predictions = trainer.predict(X_test)
```

## Firebase Setup

1. Create a Firebase project at [console.firebase.google.com](https://console.firebase.google.com)
2. Download service account JSON key:
   - **Project Settings** → **Service Accounts** → **Generate New Key**
3. Save as **fieldbot/data/firebase_service_account.json**
4. Enable Firestore, Storage, and Realtime Database in console
5. Update **fieldbot/config.py**:
   ```python
   FIREBASE_SERVICE_ACCOUNT_PATH = "data/firebase_service_account.json"
   ROBOT_ID = "fieldbot-01"  # Unique per robot
   ```

## Running the Robot

### First Time
```bash
cd fieldbot
sudo python3 main.py
```

**Monitor logs** in **robo_data/fieldbot.log** or stdout.

### Environment Variables
```bash
export FIELDBOT_LOG_LEVEL=DEBUG  # INFO | WARNING | DEBUG
python3 main.py
```

### Systemd Service (Autostart on Boot)
```ini
# /etc/systemd/system/fieldbot.service
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

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable fieldbot.service
sudo systemctl start fieldbot.service
sudo systemctl status fieldbot.service
```

## Web Dashboard

**Laptop side** — view realtime robot status & samples:

```bash
cd fieldbot/dashboard
pip install flask
python app.py
```

Open **http://localhost:8080** in browser.

Features:
- Map with robot position, grid heatmap, obstacles
- Sample history with thumbnails
- Battery, sensor readings
- Remote start/stop commands

## Troubleshooting

### Robot spins instead of going straight
**Issue**: Drive motor groupings mixed (M1+M5 vs M2+M6).  
**Fix**: Verify **config.py** LEFT/RIGHT motor groupings match your wiring.

### Camera timeouts during sampling
**Issue**: rpicam-still fails with "Failed to queue buffer" or "Device timeout".  
**Fix**: Increase **CAMERA_SETTLE_S** (default 1.5s) or **CAMERA_TIMEOUT_MS** (default 3000ms). Reduce image resolution if needed.

### Soil probe won't descend
**Issue**: High-torque motor stalls under gravity + soil friction.  
**Fix**: Enable **PROBE_DITHER_S** (reverse kick before descent) and **PROBE_TAP_N** (ON/OFF pulses) in config.py.

### Firebase upload fails
**Issue**: Service account JSON missing or expired credentials.  
**Fix**: Regenerate key in Firebase console, update **firebase_service_account.json**, restart robot.

### Low accuracy on new crops
**Issue**: LiteCShuffle trained on Pepper/Tomato/Potato; new crops not in dataset.  
**Fix**: Collect 500+ leaf images per disease class, retrain model using **ai-model/model_training.py** or use transfer learning fine-tuning.

## Key Papers & References

1. **LiteCShuffle** (Cogent F&A 2025): Lightweight CNN with channel attention & shuffle
2. **FlyCaps** (IJRITCC 2023): Capsule network + Firefly metaheuristic + ELM
3. **HybridCNN** (Nature 2025): WOA-APSO optimization paper
4. **KCNet** (arxiv:2108.07554): Insect-inspired sparse random networks
5. **YangViT** (IEEE TSMC 2024): SNN-inspired visual perception proxy

## License

GNU Affero General Public License v3 License — See LICENSE file for details.

## Authors

**Meenal Shah** — CSS 595 Capstone Project  
**Institution**: University of Washington

---

**Questions or issues?** Open an issue on GitHub or contact the maintainers.
