"""Background acquisition. The FPGA itself performs the full integration,
so nothing here does software averaging — see AcquisitionEngine.
"""

import os
import threading
import queue
import time


class AcquisitionEngine:
    """Reads continuously. The FPGA itself performs the full integration —
    via the 'acc_len' vector accumulator, configured once by
    SpectrometerCore.configure_accumulation() — so every read_raw_block()
    call already returns one fully-integrated scan. There is no software
    averaging step here.

    The loop paces itself to the hardware's own scan period
    (core.integration_ms) so consecutive frames stay evenly spaced in
    wall-clock time — that even spacing is what the metadata-based
    timestamp reconstruction (start_time + frame_index * integration_ms)
    assumes. If a read takes longer than the scan period (the loop can't
    keep up with the hardware), it simply reads back-to-back and resyncs
    rather than trying to catch up.
    """

    def __init__(self, core, live_dir):
        self.core = core
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None
        self._record_queue = None
        self.stats = {'n_reads': 0, 'last_read_ms': 0.0, 'n_frames': 0, 'overruns': 0}

        os.makedirs(live_dir, exist_ok=True)
        self.live_data_path = os.path.join(live_dir, 'live_frame.bin')
        self.live_meta_path = os.path.join(live_dir, 'live_frame.ctr')
        self._frame_counter = 0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def attach_recorder(self, q):
        with self._lock:
            self._record_queue = q

    def detach_recorder(self):
        with self._lock:
            self._record_queue = None

    def _write_live_buffer(self, block):
        # Data first, counter second: the plot process only trusts a frame
        # once it sees the counter change, so a torn write is at worst one
        # skipped/stale preview frame — never a crash.
        with open(self.live_data_path, 'wb') as f:
            f.write(block.tobytes())
        self._frame_counter += 1
        with open(self.live_meta_path, 'wb') as f:
            f.write(self._frame_counter.to_bytes(8, 'little'))

    def _loop(self):
        next_due = time.monotonic()
        while self._running:
            t0 = time.monotonic()
            raw_block = self.core.read_raw_block()   # already hardware-integrated
            self.stats['last_read_ms'] = (time.monotonic() - t0) * 1000.0
            self.stats['n_reads'] += 1
            self.stats['n_frames'] += 1

            self._write_live_buffer(raw_block)

            with self._lock:
                rq = self._record_queue
            if rq is not None:
                try:
                    rq.put_nowait(raw_block)
                except queue.Full:
                    pass  # drop rather than ever block acquisition

            # Pace to the hardware's own scan period so frames stay evenly
            # spaced. If the read itself already took longer than that, we
            # can't keep up — resync instead of trying to catch up.
            next_due += self.core.integration_ms / 1000.0
            sleep_s = next_due - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                self.stats['overruns'] += 1
                next_due = time.monotonic()
