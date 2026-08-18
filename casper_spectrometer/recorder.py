"""Recording: one flat raw file per session, header-free, plus a metadata
JSON that carries everything needed to reconstruct/segregate/FITS-ify it
later (see :mod:`casper_spectrometer.postprocessing`).
"""

import os
import time
import json
import threading
import queue
from datetime import datetime

import numpy as np

from .timeutil import UTC, IST


class Recorder:
    def __init__(self, directory, core, file_format='auto'):
        self.directory   = directory or '.'
        self.core        = core
        self.file_format = file_format   # 'bin' | 'csv' | 'auto'
        self.active_format = None
        self.queue = queue.Queue(maxsize=512)
        self._thread  = None
        self._running = False
        self._fh = None
        self.raw_path  = None
        self.meta_path = None
        self.n_written = 0
        self.start_utc = self.start_ist = None
        self.stop_utc  = self.stop_ist  = None
        self.schedule_info = None   # filled in by GUI if scheduled

    @staticmethod
    def _benchmark_format(sample_len, directory):
        data = np.random.rand(sample_len).astype(np.float32)
        tmp_bin = os.path.join(directory, '._fmt_bench.bin')
        tmp_csv = os.path.join(directory, '._fmt_bench.csv')
        try:
            t0 = time.perf_counter()
            with open(tmp_bin, 'wb') as f:
                for _ in range(20):
                    data.tofile(f)
            t_bin = time.perf_counter() - t0

            t0 = time.perf_counter()
            with open(tmp_csv, 'w') as f:
                for _ in range(20):
                    f.write(','.join(map(str, data)) + '\n')
            t_csv = time.perf_counter() - t0
        finally:
            for p in (tmp_bin, tmp_csv):
                try:
                    os.remove(p)
                except OSError:
                    pass
        return 'bin' if t_bin <= t_csv else 'csv'

    def start(self, schedule_info=None):
        core = self.core
        fmt = self.file_format
        if fmt == 'auto':
            fmt = self._benchmark_format(core.n_blocks * core.chunk, self.directory)
        self.active_format = fmt
        self.schedule_info = schedule_info

        self.start_utc = datetime.now(UTC)
        self.start_ist = self.start_utc.astimezone(IST)
        stamp = self.start_utc.strftime('%Y%m%d-%H%M%S')

        ext = 'bin' if fmt == 'bin' else 'csv'
        self.raw_path  = os.path.join(self.directory, 'raw_{}.{}'.format(stamp, ext))
        self.meta_path = os.path.join(self.directory, 'meta_{}.json'.format(stamp))
        self._fh = open(self.raw_path, 'wb' if fmt == 'bin' else 'w')

        self._write_metadata(final=False)

        self.n_written = 0
        self._running = True
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        return self.active_format, self.raw_path

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._fh:
            self._fh.close()
        self.stop_utc = datetime.now(UTC)
        self.stop_ist = self.stop_utc.astimezone(IST)
        self._write_metadata(final=True)

    def _write_metadata(self, final):
        core = self.core
        meta = dict(
            hostname=core.hostname, bitfile=str(core.bitfile),
            nchannels=core.nchannels, nfft=core.nfft, adc_nos=core.adc_nos,
            n_blocks=core.n_blocks, chunk=core.chunk, cx_design=core.cx,
            nominal_sample_rate_mhz=core.fs_in, acc_len=core.acc_len,
            integration_ms=core.integration_ms,  # hardware-integrated scan period
            start_freq_mhz=core.start_freq, stop_freq_mhz=core.stop_freq,
            data_products=list(getattr(core, 'data_products', [])),
            record_dtype='float32', record_shape=[core.n_blocks, core.chunk],
            file_format=self.active_format,
            start_utc=self.start_utc.isoformat() if self.start_utc else None,
            start_ist=self.start_ist.isoformat() if self.start_ist else None,
            schedule_info=self.schedule_info,
        )
        if final:
            meta.update(
                stop_utc=self.stop_utc.isoformat() if self.stop_utc else None,
                stop_ist=self.stop_ist.isoformat() if self.stop_ist else None,
                n_frames=self.n_written,
            )
        with open(self.meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    def _writer_loop(self):
        while self._running or not self.queue.empty():
            try:
                mean_block = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.active_format == 'bin':
                mean_block.tofile(self._fh)
            else:
                self._fh.write(','.join(map(str, mean_block.reshape(-1))) + '\n')
            self.n_written += 1
