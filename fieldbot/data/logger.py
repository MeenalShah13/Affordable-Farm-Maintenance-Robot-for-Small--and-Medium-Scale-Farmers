"""
data/logger.py — Session logger for FieldBot.

Responsibilities
─────────────────
  During the field run  → write everything to local JSON only.
                          NO network calls, NO Firebase, NO Wi-Fi dependency.

  At the dock           → hand everything to FirebaseUploader.batch_upload_session()
                          which does the complete bulk upload, then call purge()
                          to clean up local files.

  Grid is NEVER deleted here.  grid.json is managed exclusively by robot.py.

Local JSON format
──────────────────
  {
    "session_id": "...",
    "started_at": 1234567890.0,
    "updated_at": 1234567890.0,
    "sample_count": 5,
    "samples": [ {SampleRecord fields...}, ... ]
  }

Crash safety: every add_sample() write is atomic (write tmp → rename).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import List

log = logging.getLogger(__name__)


class SessionLogger:
    """
    Manages one robot run session.

    add_sample()  → local JSON only (fast, no network)
    finalise()    → triggers bulk Firebase upload, then local cleanup
    purge()       → deletes local images and session JSON
    """

    def __init__(self, firebase_uploader=None):
        from config import SESSION_PATH, DATA_DIR
        Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

        self._path      = SESSION_PATH
        self.session_id = str(uuid.uuid4())[:12]
        self.started_at = time.time()
        self._samples:  List[dict] = []
        self._firebase  = firebase_uploader   # may be None in offline mode

        log.info("SessionLogger ready — session_id=%s", self.session_id)

    # ── Session start ─────────────────────────────────────────────────

    def begin(self) -> None:
        """
        Mark the session as started and write an initial local JSON file.

        Called once when the robot leaves the dock and begins mowing.
        No Firebase call is made here — all uploads happen at dock.
        The initial flush creates the session file on disk immediately so
        that data is preserved even if the robot crashes before collecting
        any samples.
        """
        self._flush_local()
        log.info("Session %s begun — logging locally until dock",
                 self.session_id)

    # ── Field run ─────────────────────────────────────────────────────

    def add_sample(self, record) -> None:
        """
        Store a completed sample.  Local JSON only — zero network activity.

        The atomic write (tmp → rename) means a sudden power loss never
        leaves a corrupt JSON file.
        """
        d = record.to_dict() if hasattr(record, "to_dict") else dict(record)
        self._samples.append(d)
        self._flush_local()
        log.debug("Sample %s logged locally (%d total)",
                  d.get("sample_id", "?"), len(self._samples))

    # ── Dock ──────────────────────────────────────────────────────────

    def finalise(self, grid) -> bool:
        """
        Called when the robot reaches the dock and charging is confirmed.

        Triggers the full Firebase bulk upload (all samples + images +
        grid).  Returns True if the upload succeeded.

        Even if upload fails, the local JSON is preserved so it can be
        retried on the next charging cycle.
        """
        log.info("Finalising session %s (%d samples)…",
                 self.session_id, len(self._samples))

        # Write a final local snapshot with grid data
        self._flush_local(grid)

        if self._firebase is None:
            log.warning("No Firebase uploader configured — "
                        "data will remain local only")
            return False

        # Delegate the entire upload to FirebaseUploader
        ok = self._firebase.batch_upload_session(self, grid)
        if ok:
            log.info("Firebase upload complete for session %s",
                     self.session_id)
        else:
            log.warning("Firebase upload incomplete — "
                        "local data preserved for next attempt")
        return ok

    # ── Cleanup ───────────────────────────────────────────────────────

    def purge(self) -> None:
        """
        Delete local session JSON and captured images.
        Called after a successful Firebase upload.
        NEVER touches grid.json.
        """
        from config import IMAGE_SAVE_DIR

        deleted_images = 0

        # Delete image files referenced in sample records
        for s in self._samples:
            for key in ("image_front", "image_back"):
                p = s.get(key, "")
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                        deleted_images += 1
                    except OSError as exc:
                        log.warning("Could not delete image %s: %s", p, exc)

        # Sweep for any orphaned images not referenced in records
        img_dir = Path(IMAGE_SAVE_DIR)
        if img_dir.exists():
            for f in img_dir.glob("*.jpg"):
                try:
                    f.unlink()
                    deleted_images += 1
                except OSError:
                    pass

        # Clear in-memory samples
        self._samples.clear()

        # Remove local session JSON and any temp file
        for p in [self._path, self._path + ".tmp"]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

        log.info("Purge complete — %d images deleted, session JSON removed",
                 deleted_images)

    # ── Internal ──────────────────────────────────────────────────────

    def _flush_local(self, grid=None) -> None:
        """Atomic write of session state to disk."""
        payload = {
            "session_id":   self.session_id,
            "started_at":   self.started_at,
            "updated_at":   time.time(),
            "sample_count": len(self._samples),
            "samples":      self._samples,
            # Grid is only included in the final finalise() write
            "grid": grid.to_dict() if grid is not None else None,
        }
        tmp = self._path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp, self._path)
        except OSError as exc:
            log.error("Local session flush failed: %s", exc)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def log_path(self) -> str:
        return self._path

    @property
    def sample_count(self) -> int:
        return len(self._samples)