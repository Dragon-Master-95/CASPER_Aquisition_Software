"""Schedule helper: fires a callback at a given UTC instant, in its own
background thread.
"""

import time
import threading
from datetime import datetime

from .timeutil import UTC


class ScheduleTrigger:
    def __init__(self, target_dt_utc, callback):
        self.target = target_dt_utc
        self.callback = callback
        self._cancelled = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def cancel(self):
        self._cancelled = True

    def _run(self):
        while not self._cancelled:
            remaining = (self.target - datetime.now(UTC)).total_seconds()
            if remaining <= 0:
                if not self._cancelled:
                    self.callback()
                return
            time.sleep(min(remaining, 1.0))
