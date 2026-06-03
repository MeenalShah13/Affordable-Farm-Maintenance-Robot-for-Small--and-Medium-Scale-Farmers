"""
main.py — FieldBot entry point (Pi only).

Starts the robot control loop and nothing else.
All data is uploaded to Firebase when the robot docks, or as a
best-effort emergency upload on any unexpected shutdown.

Handled exit paths
──────────────────
  Ctrl+C           → KeyboardInterrupt caught inside robot.run()
  kill <pid>       → SIGTERM caught here, re-raises KeyboardInterrupt
  systemctl stop   → SIGTERM (same as above)
  Unhandled crash  → Exception caught inside robot.run()
  Normal finish    → DONE state reached, run() returns cleanly

All paths end in robot._shutdown(reason=...) → _emergency_upload().

Environment variables
─────────────────────
  FIELDBOT_LOG_LEVEL  — DEBUG | INFO | WARNING  (default INFO)
"""

import logging
import os
import signal
import sys
from pathlib import Path
from config import DATA_DIR

# Module-level reference so the SIGTERM handler can reach the robot
_bot = None


def _setup_logging() -> None:
    level = getattr(
        logging,
        os.environ.get("FIELDBOT_LOG_LEVEL", "INFO").upper(),
        logging.INFO,
    )
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(DATA_DIR, "fieldbot.log"), mode="a"
            ),
        ],
    )


def _sigterm_handler(signum, frame):
    """
    Convert SIGTERM (sent by systemctl stop / kill) into a KeyboardInterrupt
    so robot.run()'s except block triggers the normal emergency-upload path
    rather than the process dying instantly with no cleanup.
    """
    logging.getLogger("main").warning(
        "SIGTERM received — converting to KeyboardInterrupt for clean shutdown"
    )
    raise KeyboardInterrupt("SIGTERM")


def main() -> None:
    global _bot
    _setup_logging()
    log = logging.getLogger("main")
    log.info("=" * 60)
    log.info("FieldBot starting")
    log.info("=" * 60)

    # Register SIGTERM handler before instantiating the robot so any
    # kill signal during startup is also handled cleanly.
    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        from robot import FieldBot
        _bot = FieldBot()
        _bot.run()   # blocks; handles KeyboardInterrupt and exceptions internally
    except SystemExit:
        # sys.exit() called somewhere — let it propagate normally
        raise
    except Exception as exc:
        # Catch anything that escaped robot.run() (e.g. crash in __init__)
        log.exception("Fatal error before robot loop started: %s", exc)
        if _bot is not None:
            log.info("Attempting emergency shutdown from main()…")
            try:
                _bot._shutdown(reason=f"InitError: {exc}")
            except Exception as shutdown_exc:
                log.warning("Emergency shutdown also failed: %s", shutdown_exc)
        sys.exit(1)


if __name__ == "__main__":
    main()