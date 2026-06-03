# FieldBot Web Dashboard

Realtime web interface for monitoring and controlling the FieldBot robot. View live robot position, grid coverage, sensor readings, sample history, and remote control.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Dashboard Server
```bash
python app.py
```

Open **http://localhost:8080** in your browser.

---

## Architecture

### Backend (Flask)
- **app.py**: Simple Flask server that serves the single-page app (SPA)
- No backend API — all data fetching done in browser via Firebase

### Frontend (HTML + JavaScript)
- **templates/index.html**: Complete SPA with Bootstrap + Leaflet + Firebase SDK
- Realtime data binding to Firestore
- No build step — vanilla JavaScript

---

## Features

### Map View
- **Robot position**: Live location on field (from odometry)
- **Grid overlay**: Visited (green), unknown (gray), obstacle (red), boundary (blue) cells
- **Obstacle markers**: Current obstacle locations with expiry countdown
- **Dock location**: Fixed landmark on map

**Controls**:
- Drag to pan, scroll to zoom
- Click cells to inspect (visited count, last update)

### Sensor Readings
- **Battery**: SOC % + voltage
- **IMU**: Acceleration, angular velocity (gyro)
- **Soil moisture**: % saturation
- **Temperature/humidity**: Ambient conditions
- **Luminosity**: Light level (lux)
- **Motor encoders**: Tick counts per motor
- **Time**: Timestamp of last sensor read

### Sample History
- **Timeline**: All samples collected during session with thumbnails
- **Details**: Per sample:
  - Image (front + rear)
  - Leaf detection score (heuristic)
  - Disease classification (label + confidence)
  - Is diseased? (boolean, based on confidence threshold)
  - Timestamp

### Status Panel
- **Robot state**: Current state (MOWING, SAMPLING, DOCKING, etc.)
- **Pose**: x, y, θ (heading)
- **Battery**: % SOC + voltage + charge status
- **Session**: Start time, duration, cells visited, obstacles found

### Remote Control
- **Start**: Begin autonomous mowing from dock
- **Stop**: Pause mowing (returns to dock on next cycle)
- **Resume**: Continue paused session
- **Emergency stop**: Immediate motor shutdown (requires manual reset)

---

## Setup: Firebase Configuration

### Step 1: Create Firebase Project
1. Go to [console.firebase.google.com](https://console.firebase.google.com)
2. Create new project (enable Firestore, Storage, Realtime Database)
3. Go to **Project Settings** → **Accounts & Access** → **Service Accounts**
4. Click **Generate New Key** (downloads JSON)
5. Save as **fieldbot/data/firebase_service_account.json** (Pi-side upload)

### Step 2: Get Web Config
1. In Firebase console, go to **Project Settings** → **General**
2. Under **Your Apps**, find your web app or create one
3. Copy the Firebase config:
```json
{
  "apiKey": "...",
  "authDomain": "your-project.firebaseapp.com",
  "projectId": "your-project",
  "storageBucket": "your-project.appspot.com",
  "messagingSenderId": "...",
  "appId": "..."
}
```

### Step 3: Update Dashboard Config
Edit **templates/index.html**, find the `firebaseConfig` object and update:

```javascript
// In templates/index.html, around line 20
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_PROJECT.firebaseapp.com",
  projectId: "YOUR_PROJECT",
  storageBucket: "YOUR_PROJECT.appspot.com",
  messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
  appId: "YOUR_APP_ID"
};
```

### Step 4: Security Rules (Firestore)
Set permissive rules for testing (restrict in production):

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /robots/{document=**} {
      allow read, write: if request.auth != null;
    }
  }
}
```

---

## Data Flow

### Pi → Firebase (FieldBot uploads)
**On dock (CHARGING state)**:
- Images: Full-res to Storage, thumbnail (base64) to Firestore doc
- Samples: Each with leaf_conf, disease_label, disease_conf, is_diseased
- Grid state: Visited, obstacle, boundary cell list
- Session metadata: Start time, duration, final stats

**Periodic (during mowing)**:
- Robot status: Pose (x, y, θ), battery %, state
- Grid snapshot: Every 60s for realtime map updates
- Sensor readings: Every 20–30 ticks

**Collections**:
```
/robots/{ROBOT_ID}/
  ├── samples/          ← Per-sample documents (images, AI results)
  ├── grids/            ← Grid snapshots (visited cells, obstacles)
  ├── status/           ← Current robot pose & battery
  └── sessions/         ← Session metadata (start, end, stats)
```

### Firebase → Browser (Dashboard reads)
**Subscriptions** (realtime listeners):
- `robots/{ROBOT_ID}/status` → update map & sensor panel
- `robots/{ROBOT_ID}/grids/{latest}` → refresh grid overlay
- `robots/{ROBOT_ID}/samples` → populate sample history with thumbnails

**Firestore queries**:
- Fetch last N samples: `where("timestamp", ">", cutoff_date)`
- Grid cells: `where("visited", "==", true)` (count coverage)

---

## Templates Structure

### HTML Layout (templates/index.html)
```html
<!DOCTYPE html>
<html>
<head>
  <title>FieldBot Dashboard</title>
  <link rel="stylesheet" href="...bootstrap.min.css">
  <link rel="stylesheet" href="...leaflet.min.css">
  <style>/* custom styles */</style>
</head>
<body>
  <header>FieldBot Dashboard</header>
  
  <div class="container-fluid">
    <div class="row">
      <!-- Left: Map (Leaflet) -->
      <div class="col-md-8">
        <div id="map"></div>
      </div>
      
      <!-- Right: Status Panel -->
      <div class="col-md-4">
        <div id="status-panel">
          <h3>Robot Status</h3>
          <p>State: <span id="state"></span></p>
          <p>Pose: <span id="pose"></span></p>
          <p>Battery: <span id="battery"></span></p>
        </div>
        
        <div id="sensors-panel">
          <h3>Sensors</h3>
          <ul id="sensor-list"></ul>
        </div>
        
        <div id="controls">
          <button onclick="startRobot()">Start</button>
          <button onclick="stopRobot()">Stop</button>
          <button onclick="resumeRobot()">Resume</button>
        </div>
      </div>
    </div>
    
    <div class="row">
      <!-- Sample History -->
      <div class="col-12">
        <div id="samples-panel">
          <h3>Sample History</h3>
          <div id="sample-gallery"></div>
        </div>
      </div>
    </div>
  </div>
  
  <script src="...firebase-app.js"></script>
  <script src="...firebase-firestore.js"></script>
  <script src="...firebase-storage.js"></script>
  <script src="...leaflet.min.js"></script>
  <script>
    // Initialize Firebase, Firestore listeners, Leaflet map
    // Realtime UI updates
  </script>
</body>
</html>
```

---

## Customization

### Change Field Dimensions
Edit **fieldbot/config.py** before deploying:
```python
FIELD_W = 1.5  # 1.5m wide instead of 1m
FIELD_H = 2.0  # 2.0m tall instead of 1m
CELL_SIZE = 0.40  # 40cm cells instead of 30cm
```

Dashboard auto-fetches from Firestore, so no changes needed.

### Change Robot ID
Edit **fieldbot/config.py**:
```python
ROBOT_ID = "fieldbot-02"  # For a second robot
```

Dashboard must fetch from same `ROBOT_ID`. Edit **templates/index.html**:
```javascript
const ROBOT_ID = "fieldbot-02";  // Match Pi config
```

### Customize Map Markers
Edit **templates/index.html** Leaflet initialization:
```javascript
// Change robot icon
const robotIcon = L.icon({
  iconUrl: 'path/to/robot.png',
  iconSize: [32, 32]
});

// Change dock icon
const dockIcon = L.icon({
  iconUrl: 'path/to/dock.png',
  iconSize: [40, 40]
});
```

### Customize Color Scheme
Edit **templates/index.html** CSS:
```css
:root {
  --color-visited: #4CAF50;    /* Green */
  --color-obstacle: #F44336;   /* Red */
  --color-unknown: #BDBDBD;    /* Gray */
  --color-boundary: #2196F3;   /* Blue */
}
```

---

## Development

### Local Testing (without Raspberry Pi)
1. Set up a test Firestore with mock data:
```python
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("path/to/serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Add test robot data
db.collection("robots").document("fieldbot-01").set({
    "status": {
        "x": 0.5,
        "y": 0.5,
        "theta": 0.0,
        "battery_pct": 85.0,
        "state": "MOWING"
    }
})
```

2. Run dashboard:
```bash
python app.py
```

3. Open http://localhost:8080 and verify realtime updates

### Add New Sensor Display
1. Add a field to robot status in **fieldbot/robot.py**:
```python
status_doc["new_sensor_value"] = self.sensors.new_sensor.read()
```

2. Update dashboard subscription in **templates/index.html**:
```javascript
db.collection("robots").doc(ROBOT_ID).collection("status")
  .orderBy("timestamp", "desc").limit(1).onSnapshot((snap) => {
    const status = snap.docs[0].data();
    document.getElementById("new-sensor").textContent = status.new_sensor_value;
  });
```

### Deploy to Cloud (Google Cloud / Heroku)
Replace `localhost:8080` with deployed URL in **index.html** if needed. Firebase automatically works cross-domain.

Example with Google Cloud Run:
```bash
gcloud run deploy fieldbot-dashboard \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## Troubleshooting

### Dashboard shows "No data" or blank map
**Cause**: Firebase config incorrect or robot hasn't uploaded data yet.  
**Fix**:
1. Verify **firebase_service_account.json** is valid (Pi side)
2. Check Firestore has documents under `/robots/{ROBOT_ID}/`
3. Verify **firebaseConfig** in **templates/index.html** matches your Firebase project

### Map doesn't update in realtime
**Cause**: Firestore listener not attached or connection dropped.  
**Fix**:
1. Check browser console for errors: Press F12 → Console tab
2. Verify Firebase SDK loaded: `firebase.app()` should not throw
3. Check Firestore Security Rules allow reads
4. Restart dashboard: `python app.py`

### Images show as broken thumbnails
**Cause**: Firebase Storage path incorrect or access denied.  
**Fix**:
1. Verify image upload in **fieldbot/data/firebase_uploader.py** points to correct bucket
2. Check Storage Security Rules:
```
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    match /robots/{document=**} {
      allow read, write: if request.auth != null;
    }
  }
}
```

### Remote control buttons don't work
**Cause**: Dashboard can't write to Firestore (permission denied) or robot not listening for commands.  
**Fix**:
1. Verify Firestore writes allowed in Security Rules
2. Check Pi is polling command collection: `robot.py` line ~200
3. Try writing test command to Firestore:
```python
db.collection("robots").document("fieldbot-01").collection("commands").add({
    "command": "start",
    "timestamp": datetime.now()
})
```

---

## Performance Tips

### Large Sample History (1000+ images)
Dashboard fetches only last N samples by default. Adjust in **templates/index.html**:
```javascript
const SAMPLE_LIMIT = 50;  // Reduce from 100 to 50
```

### Slow Map Rendering
Simplify grid cell rendering. Instead of updating each cell individually:
```javascript
// Draw all cells in one batch
const cellGroup = L.featureGroup(allCells);
cellGroup.addTo(map);
```

### High Firebase Costs
- Limit status update frequency: Increase `FIREBASE_STATUS_INTERVAL_S` in config.py (default 30s)
- Archive old samples to Cloud Storage (automatic retention policy)
- Enable Firestore TTL for temporary documents (30 days default)

---

## License

GNU Affero General Public License v3 License. See root LICENSE file.
