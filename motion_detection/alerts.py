"""
alerts.py
---------
Small helper module used by detect.py. Handles throttled console
logging (so the terminal doesn't get spammed every single frame)
and saves a snapshot photo of each motion event to the reports folder.
"""

import os
import cv2
from datetime import datetime

# Build the reports path relative to THIS file's location, not the terminal's
# current folder. This avoids errors caused by running the script from a
# different working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(SCRIPT_DIR, "..", "reports", "motion_events")
LOG_COOLDOWN_SECONDS = 3


class MotionLogger:
    def __init__(self, save_snapshots=True):
        self.last_log_time = None
        self.save_snapshots = save_snapshots

        if self.save_snapshots:
            try:
                os.makedirs(REPORTS_DIR, exist_ok=True)
            except OSError as e:
                print(f"[WARNING] Could not create reports folder ({e}). Snapshots will be disabled.")
                self.save_snapshots = False

    def log(self, frame=None):
        now = datetime.now()

        if self.last_log_time is not None:
            seconds_since_last = (now - self.last_log_time).total_seconds()
            if seconds_since_last < LOG_COOLDOWN_SECONDS:
                return  # too soon since last log, skip

        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[MOTION] Motion detected at {timestamp_str}")

        if self.save_snapshots and frame is not None:
            filename = now.strftime("motion_%Y%m%d_%H%M%S.jpg")
            filepath = os.path.join(REPORTS_DIR, filename)
            cv2.imwrite(filepath, frame)
            print(f"  [SAVED] Snapshot saved to {filepath}")

        self.last_log_time = now