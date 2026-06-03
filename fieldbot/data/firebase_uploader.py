"""
data/firebase_uploader.py — Firebase integration for FieldBot (Pi side).

Changes from previous version
──────────────────────────────
  • Firebase Storage REMOVED — images are stored as base64-encoded JPEG
    thumbnails (640×360, quality 70) directly inside each Firestore sample
    document.  No Storage bucket needed; no cost beyond the free-tier
    Firestore quota (1 GB total, 20 K writes/day).

  • Status push is now a long-lived daemon thread (StatusPusher) rather
    than a new thread spawned every N seconds.  The control loop calls
    pusher.update(dict) every tick (fast, no I/O); the thread pushes to
    Firestore every 30 s on its own schedule.

Three responsibilities
──────────────────────
1. download_grid()          — boot-time cloud grid fetch (blocking, once)
2. batch_upload_session()   — dock-time bulk upload (all samples in Firestore)
3. StatusPusher             — daemon thread, push every 30 s

Firestore layout
─────────────────
  robots/{ROBOT_ID}/
    grid/current            ← saved grid, overwritten every dock
    status/current          ← live telemetry, overwritten every 30 s
    sessions/{id}/          ← one document per run
      samples/{id}/         ← one document per sample
                               fields: image_front_b64, image_back_b64

Design rules
─────────────
  • Every public method catches ALL exceptions.
  • SDK initialised lazily — safe to import with no service account.
  • Wi-Fi check gates every network call; fails fast (1.5 s timeout).
"""

from __future__ import annotations

import base64
import logging
import os
import socket
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)

# ── SDK singleton ──────────────────────────────────────────────────────
_app         = None
_db          = None
_sdk_ready   = False
_sdk_checked = False


def _init_sdk() -> bool:
    """Initialise firebase_admin once. Returns True if SDK is ready."""
    global _app, _db, _sdk_ready, _sdk_checked
    if _sdk_checked:
        return _sdk_ready
    _sdk_checked = True

    from config import FIREBASE_SERVICE_ACCOUNT_PATH
    if not os.path.exists(FIREBASE_SERVICE_ACCOUNT_PATH):
        log.warning(
            "Firebase service account not found at %s — offline-only mode.\n"
            "Download from: Firebase Console → Project Settings → "
            "Service Accounts → Generate New Key",
            FIREBASE_SERVICE_ACCOUNT_PATH,
        )
        return False

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_PATH)
        # No storageBucket — Storage is not used
        _app = firebase_admin.initialize_app(cred)
        _db  = firestore.client()
        _sdk_ready = True
        log.info("Firebase SDK ready (Firestore only — no Storage)")
        return True

    except Exception as exc:
        log.error("Firebase SDK init failed: %s", exc)
        return False


# ── Connectivity check ─────────────────────────────────────────────────

def _is_online() -> bool:
    """
    Check whether the Pi has a routable network address without making any
    outbound connection.

    Strategy: `hostname -I` returns all non-loopback IP addresses assigned to
    the host (space-separated).  A non-empty result means at least one network
    interface is up and addressed, so we proceed with the Firebase call.

    Why not TCP-probe an external host?
      * Outbound TCP to arbitrary IPs/ports is blocked on many shared,
        university, and VPN networks even when HTTPS to known services works.
      * A local command completes in < 50 ms, requires no routing, and cannot
        be blocked by any firewall.

    If the Firebase call subsequently fails (no internet path despite having an
    IP), the SDK exception is caught in the callers below, local data is
    preserved, and the upload retries on the next dock.
    """
    import subprocess

    try:
        out = subprocess.check_output(
            ["hostname", "-I"],
            text=True, timeout=2,
            stderr=subprocess.DEVNULL,
        ).strip()

        if out:
            log.debug("Network interface OK — hostname -I: %s", out)
            return True

        log.warning(
            "hostname -I returned no addresses — no network interface is up. "
            "Run `hostname -I` on the Pi to diagnose."
        )
        return False

    except Exception as exc:
        # hostname -I unavailable; fall back to the socket resolver (no
        # external connection made — just a local name → address lookup).
        log.debug("hostname -I failed (%s) — falling back to gethostbyname", exc)
        try:
            addr = socket.gethostbyname(socket.gethostname())
            if addr and not addr.startswith("127."):
                log.debug("Network interface OK via gethostbyname — %s", addr)
                return True
        except OSError:
            pass
        log.warning("Could not determine network state — assuming offline.")
        return False


# ── Firestore path helpers ─────────────────────────────────────────────

def _robot_doc():
    from config import ROBOT_ID
    return _db.collection("robots").document(ROBOT_ID)

def _session_ref(sid: str):
    return _robot_doc().collection("sessions").document(sid)

def _sample_ref(sid: str, sample_id: str):
    return _session_ref(sid).collection("samples").document(sample_id)


# ── Image encoding (replaces Firebase Storage) ─────────────────────────

def _encode_image_b64(local_path: str) -> Optional[str]:
    """
    Resize a JPEG to IMAGE_THUMB_SIZE, encode at IMAGE_THUMB_QUALITY,
    and return a base64 ASCII string.

    Storing the string directly in Firestore replaces Firebase Storage.
    Typical size for 640×360 @ quality 70: 20–50 KB → 27–67 KB as base64.
    Firestore document limit is 1 MB, so two images per sample is fine.

    Returns None on any error so callers can store an empty string instead.
    """
    if not local_path or not os.path.exists(local_path):
        return None
    try:
        import cv2
        from config import IMAGE_THUMB_SIZE, IMAGE_THUMB_QUALITY

        img = cv2.imread(local_path)
        if img is None:
            log.warning("Cannot read image for encoding: %s", local_path)
            return None

        w, h = IMAGE_THUMB_SIZE
        thumb = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(
            ".jpg", thumb,
            [cv2.IMWRITE_JPEG_QUALITY, IMAGE_THUMB_QUALITY],
        )
        if not ok:
            log.warning("JPEG encode failed: %s", local_path)
            return None

        result = base64.b64encode(buf.tobytes()).decode("ascii")
        log.debug("Encoded %s → %d chars", os.path.basename(local_path), len(result))
        return result

    except Exception as exc:
        log.warning("Image encode error (%s): %s", local_path, exc)
        return None


# ══════════════════════════════════════════════════════════════════════
# StatusPusher — long-lived daemon thread
# ══════════════════════════════════════════════════════════════════════

class StatusPusher(threading.Thread):
    """
    Daemon thread that pushes robot telemetry to Firestore every 30 seconds.

    The control loop calls update() every tick with the latest status dict.
    update() is instant (just updates a shared dict under a lock; no I/O).
    The thread wakes every INTERVAL seconds and writes whatever is latest.

    Using threading.Event.wait(timeout) instead of time.sleep() means
    stop() causes the thread to exit within milliseconds rather than
    waiting up to 30 s.

    Usage in robot.py
    ──────────────────
        # Start once during init
        self.status_pusher = StatusPusher(self.firebase, interval_s=30)
        self.status_pusher.start()

        # Inside control loop (every tick — no I/O, instant)
        self.status_pusher.update(self._build_status())

        # During shutdown (before GPIO cleanup)
        self.status_pusher.stop()
        self.status_pusher.join(timeout=5)
    """

    def __init__(self, firebase_uploader: "FirebaseUploader",
                 interval_s: float = 30.0):
        super().__init__(name="status-pusher", daemon=True)
        self._uploader = firebase_uploader
        self._interval = interval_s
        self._stop     = threading.Event()
        self._lock     = threading.Lock()
        self._latest:  dict = {}

    def update(self, status_dict: dict) -> None:
        """
        Store the latest status for the next push.
        Called from the control loop — MUST return instantly.
        """
        with self._lock:
            self._latest = dict(status_dict)

    def run(self) -> None:
        log.info("StatusPusher started — interval=%.0fs", self._interval)
        # Loop until stop() is called; wait() returns True when stopped
        while not self._stop.wait(timeout=self._interval):
            self._push_now()
        # Final push on shutdown so dashboard shows the stopped state
        self._push_now()
        log.info("StatusPusher stopped")

    def _push_now(self) -> None:
        with self._lock:
            status = dict(self._latest)
        if status:
            self._uploader.push_status(status)

    def stop(self) -> None:
        """Signal the thread to exit after its current sleep."""
        self._stop.set()


# ══════════════════════════════════════════════════════════════════════
# FirebaseUploader
# ══════════════════════════════════════════════════════════════════════

class FirebaseUploader:
    """
    Firebase operations for FieldBot:
      1. download_grid()         — boot-time grid fetch
      2. batch_upload_session()  — dock-time bulk upload
      3. push_status()           — called by StatusPusher thread
    """

    def __init__(self):
        self._ready = _init_sdk()

    # ── 1. Boot-time grid download ────────────────────────────────────

    def download_grid(self) -> Optional[dict]:
        """
        Fetch the latest grid from Firestore at startup.

        Reconstructs the 2D cells list from the flattened 'cells_flat'
        field stored in Firestore (nested arrays are not permitted there).
        Returns a Grid.to_dict()-compatible dict (with 'cells' as a 2D
        list) or None if unavailable.
        """
        if not self._ready:
            return None
        if not _is_online():
            log.info("Boot grid check: offline — using local grid")
            return None
        try:
            doc = _robot_doc().collection("grid").document("current").get()
            if doc.exists:
                data = doc.to_dict()
                log.info(
                    "Boot: cloud grid found (%.0f%% coverage, updated %s)",
                    data.get("coverage_pct", 0),
                    _fmt_ts(data.get("updated_at", 0)),
                )

                # Reconstruct 2D list from flat storage so callers can use
                # data["cells"][r][c] exactly as with Grid.to_dict()
                flat  = data.get("cells_flat", [])
                rows  = data.get("rows", 0)
                cols  = data.get("cols", 0)
                if flat and rows and cols and len(flat) == rows * cols:
                    data["cells"] = [
                        flat[r * cols: (r + 1) * cols]
                        for r in range(rows)
                    ]
                elif "cells" not in data:
                    log.warning("Boot grid: no cells data found in cloud document")
                    return None

                return data
            log.info("Boot: no cloud grid yet — starting fresh")
            return None
        except Exception as exc:
            log.warning("Boot grid check failed: %s", exc)
            return None

    # ── 2. Dock-only bulk upload ──────────────────────────────────────

    def batch_upload_session(self, session_logger, grid) -> bool:
        """
        Upload the entire session to Firestore when the robot reaches the dock.

        Images are base64-encoded at thumbnail resolution and stored as
        string fields inside the sample document — no Firebase Storage used.

        Returns True if every sample was written successfully.
        """
        if not self._ready:
            log.warning("Batch upload skipped — Firebase SDK not ready")
            return False
        if not _is_online():
            log.warning("Batch upload skipped — network unreachable (see probe details above)")
            return False

        session_id = session_logger.session_id
        samples    = list(session_logger._samples)
        total      = len(samples)
        log.info("=== Dock upload: %d samples, session=%s ===", total, session_id)

        # Create session document
        try:
            coverage = grid.coverage_fraction() * 100 if grid else 0.0
            _session_ref(session_id).set({
                "session_id":   session_id,
                "robot_id":     _robot_id(),
                "started_at":   session_logger.started_at,
                "ended_at":     time.time(),
                "status":       "uploading",
                "sample_count": total,
                "coverage_pct": round(coverage, 1),
            })
        except Exception as exc:
            log.error("Could not create session doc: %s", exc)
            return False

        # Upload each sample
        uploaded = 0
        for i, sample in enumerate(samples, 1):
            sid = sample.get("sample_id", f"s{i}")
            log.info("  Sample %d/%d: %s", i, total, sid)
            try:
                doc = dict(sample)
                doc.pop("_firebase_uploaded", None)

                # Encode images as base64 thumbnails — stored in Firestore
                doc["image_front_b64"] = _encode_image_b64(
                    sample.get("image_front", "")) or ""
                doc["image_back_b64"]  = _encode_image_b64(
                    sample.get("image_back",  "")) or ""

                # Remove local file paths (meaningless on the laptop)
                doc.pop("image_front", None)
                doc.pop("image_back",  None)

                _sample_ref(session_id, sid).set(doc)
                uploaded += 1

            except Exception as exc:
                log.warning("  Sample %s failed: %s", sid, exc)

        log.info("  Samples uploaded: %d/%d", uploaded, total)

        # Save grid
        self._save_grid_to_cloud(grid, session_id)

        # Mark session complete
        try:
            _session_ref(session_id).update({
                "status":           "complete",
                "uploaded_at":      time.time(),
                "samples_uploaded": uploaded,
            })
            log.info("Session %s marked complete", session_id)
        except Exception as exc:
            log.warning("Could not mark session complete: %s", exc)

        return uploaded == total

    def _save_grid_to_cloud(self, grid, session_id: str) -> None:
        """
        Write grid to robots/{ROBOT_ID}/grid/current.

        Cells are flattened to 1D (Firestore rejects nested arrays).
        Obstacle ages are stored as a flat string-keyed dict so the
        laptop dashboard and next boot can reconstruct decay correctly.
        """
        if grid is None:
            return
        try:
            d = grid.to_dict()   # includes obstacle_ages and 2D cells

            # Flatten 2D cells → 1D list (Firestore-safe)
            flat_cells = [
                int(grid.cells[r][c])
                for r in range(grid.rows)
                for c in range(grid.cols)
            ]

            payload = {
                "rows":           d["rows"],
                "cols":           d["cols"],
                "cell_size":      d["cell_size"],
                "cells_flat":     flat_cells,
                # obstacle_ages is already a flat string-keyed dict — Firestore OK
                "obstacle_ages":  d.get("obstacle_ages", {}),
                "updated_at":     time.time(),
                "session_id":     session_id,
                "coverage_pct":   round(grid.coverage_fraction() * 100, 1),
            }
            _robot_doc().collection("grid").document("current").set(payload)
            log.info("Grid saved to Firestore (%.0f%% coverage, %d cells, "
                     "%d obstacle timestamps)",
                     payload["coverage_pct"], len(flat_cells),
                     len(payload["obstacle_ages"]))
        except Exception as exc:
            log.warning("Grid save failed: %s", exc)

    # ── Remote command channel ────────────────────────────────────────

    def poll_command(self) -> Optional[str]:
        """
        Read the current unacknowledged command from Firestore.

        The laptop dashboard writes to:
            robots/{ROBOT_ID}/commands/control
              { "command": "stop"|"start", "issued_at": float,
                "acknowledged": false }

        Returns the command string if there is an unacknowledged one,
        or None if no command is pending, SDK is not ready, or offline.
        """
        if not self._ready:
            return None
        if not _is_online():
            return None
        try:
            doc = _robot_doc().collection("commands").document("control").get()
            if doc.exists:
                d = doc.to_dict()
                if not d.get("acknowledged", True):
                    cmd = d.get("command")
                    log.info("Received command: %r", cmd)
                    return cmd
        except Exception as exc:
            log.debug("poll_command error (non-fatal): %s", exc)
        return None

    def acknowledge_command(self) -> None:
        """
        Mark the current command as acknowledged so it is not re-processed.
        Call this after acting on the command.
        """
        if not self._ready:
            return
        try:
            _robot_doc().collection("commands").document("control").update({
                "acknowledged":    True,
                "acknowledged_at": time.time(),
            })
            log.debug("Command acknowledged")
        except Exception as exc:
            log.debug("acknowledge_command error (non-fatal): %s", exc)

    # ── 3. Status push (called by StatusPusher thread) ────────────────

    def push_status(self, status_dict: dict) -> None:
        """
        Write telemetry to robots/{ROBOT_ID}/status/current.
        Called from StatusPusher daemon thread — never the control loop.
        Silently skips when offline or on any error.
        """
        if not self._ready:
            return
        if not _is_online():
            return
        try:
            payload = dict(status_dict)
            payload["updated_at"] = time.time()
            _robot_doc().collection("status").document("current").set(payload)
            log.debug("Status pushed: %s batt=%.0f%%",
                      payload.get("state"), payload.get("battery_pct", 0))
        except Exception as exc:
            log.debug("Status push failed (non-fatal): %s", exc)


# ── Module helpers ─────────────────────────────────────────────────────

def _robot_id() -> str:
    from config import ROBOT_ID
    return ROBOT_ID

def _fmt_ts(ts: float) -> str:
    try:
        import datetime
        return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)