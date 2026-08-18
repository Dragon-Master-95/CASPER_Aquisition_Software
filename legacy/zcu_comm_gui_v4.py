"""
CASPER Spectrometer control GUI — v4.

Design summary (per revision after review):

  * Hardware reads stay sequential per-channel (FPGA/BRAM constraint), but
    NOTHING else happens in that hot loop any more: no interleaving, no
    per-frame timestamp/acc_n/n_avg bookkeeping, no disk I/O. Those are all
    deferred to after recording stops.
  * Integration happens entirely on the FPGA, via the 'acc_len' vector-
    accumulator register — there is no software averaging step. The user
    enters a desired accumulation time in ms; SpectrometerCore solves for
    the (integer) acc_len value that produces it, using the fixed
    hardware calibration acc_len=1 -> ~2 ms/scan (is not independent of
    nfft one has to check the design and aquisition first), and writes it to
    the FPGA, with the mandatory reset+settle sequence, before acquisition starts. 
    AcquisitionEngine then just reads continuously — every read is already a 
    fully-integrated scan — paced to that scan period, and does exactly two 
    things per frame: (1) overwrite a small "live buffer" file pair with the 
    raw block, (2) if recording, hand the raw block to the recorder's queue.
  * Recording writes ONE flat file per session — the raw, un-interleaved,
    header-free block stream. No epoch/acc_n/n_avg is stored per record;
    the wall-clock time of any frame is reconstructable from
    start_time + frame_index * integration_ms, which is stored once in a
    metadata JSON file next to the recording.
  * Only once recording stops does post-processing run: segregate the raw
    dump into per-ADC arrays, write them in the chosen format (bin/csv,
    whichever benchmarks faster, or a forced choice), zip {raw dump +
    segregated files}, and — compulsorily — convert the result to a FITS
    file whose header/description is built entirely from the metadata JSON
    (nothing free-typed by the user). Once the zip and FITS file both exist,
    every intermediate file/folder that led to them (raw dump, metadata
    JSON, segregated per-channel directory) is deleted — the zip and the
    FITS file are the only things left on disk for that recording.
  * Live plotting never touches the acquisition thread's memory. A separate
    OS process (multiprocessing.Process) polls the live-buffer file pair on
    disk and redraws independently — no Queue/Pipe array transfer, no GIL
    contention with the read loop.
  * Recording can be started/stopped manually, or scheduled with an
    explicit date + hh:mm:ss + timezone (IST/UTC) picker for both start and
    stop.
  * For multi-BRAM-block setups, one name per channel doubles as both the
    plot subplot title and the FITS/metadata data-product name.

Requires: numpy, matplotlib, casperfpga, tkinter, and (for FITS export,
compulsory at the end of every recording) astropy.
"""

import os, sys, time, math, json, zipfile, shutil, threading, queue, multiprocessing
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import casperfpga
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime, timezone, timedelta

try:
    from astropy.io import fits
    _HAVE_ASTROPY = True
except ImportError:
    _HAVE_ASTROPY = False

UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))
TZ_MAP = {'UTC': UTC, 'IST': IST}


# ─────────────────────────────────────────────
#  Hardware-facing core
# ─────────────────────────────────────────────
class SpectrometerCore:
    """Owns the FPGA connection and the pure numpy math. Deliberately has
    NO per-frame bookkeeping — read_raw_block() is as cheap as the
    mandated sequential reads allow, and interleave() is a pure function
    reused identically by the live-plot process and the post-recording
    segregation step, so both reconstruct data exactly the same way.
    """

    # Hardware calibration: one accumulation count (acc_len=1) is one scan,
    # ~0.046 ms, independent of nfft. acc_len must be an integer register
    # value, so the ms<->acc_len conversion is a simple linear one.
    MS_PER_ACC_LEN = 2   # ms per single acc_len count

    def __init__(self, hostname='169.254.192.228', bitfile=None,
                 nchannels=4, nfft=2048, cx=True, adc_nos=1, fs=2000.0,
                 start_freq=0.0, stop_freq=1000.0, integration_ms=None):
        self.fpga      = casperfpga.CasperFpga(hostname)
        time.sleep(0.2)
        self.hostname  = hostname
        self.bitfile   = bitfile
        self.nchannels = nchannels
        self.nfft      = nfft
        self.cx        = cx
        self.adc_nos   = adc_nos
        self.fs_in     = fs               # nominal sample rate, metadata only
        self.start_freq = start_freq      # MHz — defines the plotted/FITS axis
        self.stop_freq  = stop_freq       # MHz
        self.chunk      = nfft // nchannels
        self.n_blocks   = nchannels * adc_nos
        # The FPGA design integrates in hardware via the 'acc_len' vector-
        # accumulator register — there is no separate software integration
        # step. The user gives a desired accumulation time in ms; we solve
        # for the acc_len register value that produces it.
        if integration_ms is None:
            self.acc_len = self.default_acc_len(nfft)
        else:
            self.acc_len = self.acc_len_for_ms(integration_ms)
        self.integration_ms = self.scan_ms_for_acc_len(self.acc_len)
        # Per user decision: start/stop freq OVERRIDE the axis outright.
        self.faxis = np.linspace(self.start_freq, self.stop_freq, self.nfft)

    @classmethod
    def acc_len_for_ms(cls, target_ms):
        """Desired accumulation time (ms) -> acc_len register value
        (acc_len=1 ~ 0.046 ms/scan; acc_len must be an integer)."""
        return max(1, round(target_ms / cls.MS_PER_ACC_LEN))

    @classmethod
    def scan_ms_for_acc_len(cls, acc_len):
        """acc_len register value -> actual scan duration (ms). Rounding in
        acc_len_for_ms means this can differ slightly from what was asked
        for — this is the value that should actually be recorded/trusted.
        """
        return acc_len * cls.MS_PER_ACC_LEN

    @classmethod
    def default_acc_len(cls, nfft):
        """The FPGA design's own built-in default register value, used
        only when no accumulation time is specified at all."""
        return 2 * (2 ** 28) // nfft

    def program_fpga(self):
        self.fpga.upload_to_ram_and_program(self.bitfile)
        print('FPGA programmed with bitfile: {}'.format(self.bitfile))

    def configure_accumulation(self, acc_len=None, reset_wait=5.0, on_status=None):
        """Program the hardware accumulation length and let the vector
        accumulator settle. This MUST run once, after program_fpga() and
        before any read_raw_block() call: acc_len sets how many FFT frames
        the FPGA itself averages into one readable (already fully
        integrated) scan, and the mandatory reset_wait (~5s) is how long
        the accumulator takes to produce its first valid scan after being
        reset. Skipping or mistiming this was the real source of the
        "extra" acquisition latency/garbage-first-read seen previously —
        it's a one-off cost (at startup, or whenever integration time is
        changed), not per-frame.
        """
        if acc_len is None:
            acc_len = self.acc_len

        def _status(msg):
            if on_status:
                on_status(msg)
            print(msg)

        _status('Configuring accumulation period (acc_len={})…'.format(acc_len))
        self.fpga.write_int('acc_len', acc_len)
        time.sleep(0.1)
        _status('Accumulation period configured.')

        _status('Resetting counters…')
        self.fpga.write_int('cnt_rst', 1)
        self.fpga.write_int('cnt_rst', 0)
        _status('Waiting {:.1f}s for accumulator to settle…'.format(reset_wait))
        time.sleep(reset_wait)
        _status('Counters reset, accumulator ready.')

        self.acc_len = acc_len
        self.integration_ms = self.scan_ms_for_acc_len(acc_len)

    def read_raw_block(self):
        """The only place that does the mandated sequential per-channel
        reads. Returns the raw (n_blocks, chunk) float32 matrix, completely
        un-interleaved — no other work happens here.
        """
        raw = np.empty((self.n_blocks, self.chunk), dtype='>u8')
        for i in range(self.n_blocks):
            raw[i] = np.frombuffer(
                self.fpga.read('q{:d}'.format(i + 1), self.chunk * 8, 0), dtype='>u8')
        return raw.astype(np.float32)

    def interleave(self, raw_block):
        """(n_blocks, chunk) -> {adc_idx: 1D array of length nfft}."""
        spectrum = {}
        for k in range(self.adc_nos):
            block = raw_block[self.nchannels * k: self.nchannels * (k + 1)]
            spectrum[k] = block.T.reshape(-1)
        return spectrum


# ─────────────────────────────────────────────
#  Background acquisition (hardware already integrates — no software
#  averaging happens here)
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
#  Recording: one flat raw file, no per-frame headers
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
#  Post-processing: segregate, zip, compulsory FITS
# ─────────────────────────────────────────────
def load_raw_recording(raw_path, meta):
    """Read the flat raw dump back into (n_frames, n_blocks, chunk) float32."""
    n_blocks, chunk = meta['record_shape']
    frame_len = n_blocks * chunk
    if meta['file_format'] == 'bin':
        flat = np.fromfile(raw_path, dtype=np.float32)
        n_frames = flat.size // frame_len
        return flat[:n_frames * frame_len].reshape(n_frames, n_blocks, chunk)
    rows = []
    with open(raw_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split(',')])
    arr = np.array(rows, dtype=np.float32)
    return arr.reshape(arr.shape[0], n_blocks, chunk)


def segregate_and_zip(raw_path, meta_path, out_dir):
    """Split the raw dump into per-ADC files (same format as the raw dump),
    then zip {raw dump, segregated files} together. Returns
    (segregated_paths, zip_path).
    """
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    frames = load_raw_recording(raw_path, meta)          # (n_frames, n_blocks, chunk)
    nchannels = meta['nchannels']
    adc_nos   = meta['adc_nos']
    fmt       = meta['file_format']
    stamp     = os.path.splitext(os.path.basename(raw_path))[0].replace('raw_', '')

    seg_dir = os.path.join(out_dir, 'segregated_{}'.format(stamp))
    os.makedirs(seg_dir, exist_ok=True)

    segregated = {}
    ext = 'bin' if fmt == 'bin' else 'csv'
    for k in range(adc_nos):
        block = frames[:, nchannels * k:nchannels * (k + 1), :]     # (n_frames, nchannels, chunk)
        spectra = block.transpose(0, 2, 1).reshape(block.shape[0], -1)  # (n_frames, nfft) interleaved
        path = os.path.join(seg_dir, 'adc{}_{}.{}'.format(k, stamp, ext))
        if fmt == 'bin':
            spectra.astype(np.float32).tofile(path)
        else:
            with open(path, 'w') as f:
                for row in spectra:
                    f.write(','.join(map(str, row)) + '\n')
        segregated[k] = path

    zip_path = os.path.join(out_dir, 'recording_{}.zip'.format(stamp))
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(raw_path, arcname=os.path.basename(raw_path))
        zf.write(meta_path, arcname=os.path.basename(meta_path))
        for path in segregated.values():
            zf.write(path, arcname=os.path.join('segregated', os.path.basename(path)))

    return segregated, zip_path


def convert_to_fits(meta_path, segregated_paths, output_path=None):
    """Build a multi-extension FITS file. All header/description content
    comes from the metadata JSON written during recording — nothing here
    is free-typed by the user.
    """
    if not _HAVE_ASTROPY:
        raise RuntimeError('astropy is required for FITS export (pip install astropy)')

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    nfft = meta['nfft']
    integ_ms = meta.get('integration_ms')  # optional, filled by caller if known

    hdus = [fits.PrimaryHDU()]
    ph = hdus[0].header
    ph['TELESCOP'] = 'CASPER-ZCU'
    ph['INSTRUME'] = meta['hostname']
    ph['BITFILE']  = meta['bitfile']
    ph['DATE-OBS'] = meta.get('start_utc') or ''
    ph['DOBS-IST'] = meta.get('start_ist') or ''
    ph['DATE-END'] = meta.get('stop_utc') or ''
    ph['DEND-IST'] = meta.get('stop_ist') or ''
    ph['NFRAMES']  = meta.get('n_frames', 0)
    ph['NFFT']     = nfft
    ph['NCHAN']    = meta['nchannels']
    ph['NADC']     = meta['adc_nos']
    ph['CXDESIGN'] = meta['cx_design']
    ph['FSMHZ']    = meta['nominal_sample_rate_mhz']
    ph['ACCLEN']   = (meta.get('acc_len'), 'Hardware vector-accumulator length')
    ph['INTTIME']  = (meta.get('integration_ms'), 'Hardware-integrated scan period [ms]')
    ph['FSTART']   = (meta['start_freq_mhz'], 'Acquisition start freq [MHz]')
    ph['FSTOP']    = (meta['stop_freq_mhz'], 'Acquisition stop freq [MHz]')
    ph['FORMAT']   = meta['file_format']
    if meta.get('schedule_info'):
        ph['SCHED']    = json.dumps(meta['schedule_info'])[:68]
    ph['COMMENT']  = ('CASPER spectrometer recording: {}ch x {} ADC(s), nfft={}, '
                       'freq {:.3f}-{:.3f} MHz'.format(
                           meta['nchannels'], meta['adc_nos'], nfft,
                           meta['start_freq_mhz'], meta['stop_freq_mhz']))

    faxis = np.linspace(meta['start_freq_mhz'], meta['stop_freq_mhz'], nfft)
    freq_col = fits.Column(name='freq_MHz', format='D', array=faxis)
    hdus.append(fits.BinTableHDU.from_columns([freq_col], name='FREQAXIS'))

    products = meta.get('data_products') or []
    for k, path in segregated_paths.items():
        name = products[k] if k < len(products) and products[k] else 'ADC{}'.format(k)
        safe_name = ''.join(c if c.isalnum() else '_' for c in name).upper()[:20]
        if meta['file_format'] == 'bin':
            data = np.fromfile(path, dtype=np.float32).reshape(-1, nfft)
        else:
            rows = []
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append([float(x) for x in line.split(',')])
            data = np.array(rows, dtype=np.float32)

        img = fits.ImageHDU(data=data, name='{}_SPEC'.format(safe_name))
        img.header['ADCIDX']  = k
        img.header['DPROD']   = name
        img.header['NROWS']   = data.shape[0]
        hdus.append(img)

    if output_path is None:
        stamp = meta.get('start_utc', datetime.now(UTC).isoformat())
        stamp = stamp.replace(':', '').replace('-', '').split('.')[0].replace('T', '-')
        output_path = os.path.join(os.path.dirname(list(segregated_paths.values())[0]),
                                    '..', 'spectrometer_{}.fits'.format(stamp))
        output_path = os.path.normpath(output_path)

    fits.HDUList(hdus).writeto(output_path, overwrite=True)
    return output_path


def _cleanup_intermediates(raw_path, meta_path, segregated_paths):
    """Delete everything a recording created except the final zip and FITS
    file — the raw dump, the metadata JSON, and the segregated per-channel
    directory are all already preserved inside the zip, so they're
    redundant on disk once the zip and FITS exist.
    """
    for p in (raw_path, meta_path):
        try:
            if p and os.path.isfile(p):
                os.remove(p)
        except OSError:
            pass

    seg_dir = None
    for p in segregated_paths.values():
        seg_dir = os.path.dirname(p)
        break
    if seg_dir and os.path.isdir(seg_dir):
        shutil.rmtree(seg_dir, ignore_errors=True)


def run_postprocessing(raw_path, meta_path, out_dir):
    """Compulsory pipeline run once recording stops: segregate -> zip -> FITS
    -> delete every intermediate, leaving only the zip and the FITS file.
    integration_ms/acc_len are already in the metadata JSON (written by
    Recorder at start, from the hardware-configured core) — nothing to
    stamp in here.
    """
    segregated, zip_path = segregate_and_zip(raw_path, meta_path, out_dir)
    fits_path = convert_to_fits(meta_path, segregated)
    _cleanup_intermediates(raw_path, meta_path, segregated)
    return zip_path, fits_path


# ─────────────────────────────────────────────
#  Multiprocessing live-plot worker (module level: must be picklable)
# ─────────────────────────────────────────────
def _plot_worker(cfg):
    """Runs in its own OS process. Polls the live-buffer file pair on disk
    and redraws — no array is ever passed through a Queue/Pipe from the
    acquisition thread, so this process cannot add latency to acquisition
    and acquisition (a different process entirely) cannot stall this plot.
    """
    import numpy as _np
    import math as _math
    import matplotlib as _mpl
    _mpl.use('TkAgg')
    import matplotlib.pyplot as _plt
    import matplotlib.animation as _anim

    nchannels = cfg['nchannels']; adc_nos = cfg['adc_nos']; nfft = cfg['nfft']
    chunk = cfg['chunk']; n_blocks = cfg['n_blocks']
    faxis = _np.array(cfg['faxis'])
    titles = cfg['titles']
    scale = cfg['scale']
    data_path, meta_path = cfg['live_data_path'], cfg['live_meta_path']
    frame_bytes = n_blocks * chunk * 4

    def safe_log(x):
        x = _np.asarray(x, dtype=_np.float64)
        if scale == 'log10':
            return 10.0 * _np.log10(_np.where(x <= 0, 1, x))
        return _np.nan_to_num(x)

    def interleave(block):
        spec = {}
        for k in range(adc_nos):
            b = block[nchannels * k:nchannels * (k + 1)]
            spec[k] = b.T.reshape(-1)
        return spec

    def read_latest():
        try:
            with open(meta_path, 'rb') as mf:
                raw = mf.read(8)
                if len(raw) < 8:
                    return None, None
                counter = int.from_bytes(raw, 'little')
            with open(data_path, 'rb') as df:
                buf = df.read(frame_bytes)
                if len(buf) < frame_bytes:
                    return None, None
            block = _np.frombuffer(buf, dtype=_np.float32).reshape(n_blocks, chunk)
            return counter, block
        except (FileNotFoundError, ValueError):
            return None, None

    cols = int(_math.ceil(_math.sqrt(adc_nos)))
    rows = int(_math.ceil(adc_nos / cols))
    fig, axes_grid = _plt.subplots(rows, cols, constrained_layout=True,
                                   figsize=(5 * cols, 3.5 * rows))
    axes_flat = [axes_grid] if adc_nos == 1 else _np.array(axes_grid).flatten().tolist()
    for idx in range(adc_nos, rows * cols):
        axes_flat[idx].set_visible(False)

    _, block0 = read_latest()
    spec0 = interleave(block0) if block0 is not None else {k: _np.zeros(nfft) for k in range(adc_nos)}

    lines, axes = [], []
    for k in range(adc_nos):
        ax = axes_flat[k]
        ax.grid(True)
        ax.set_xlabel('Frequency (MHz)', fontsize=8)
        ax.set_ylabel('Power (dB arb.)' if scale == 'log10' else 'Norm Power', fontsize=8)
        ax.tick_params(labelsize=7)
        data = safe_log(spec0[k])
        l, = ax.plot(faxis, data, '-', linewidth=0.8)
        title = titles[k] if k < len(titles) else 'ADC {}'.format(k)
        ax.set_title('{}\n(waiting for data…)'.format(title), fontsize=8, pad=3)
        lines.append(l); axes.append(ax)

    last_counter = [-1]

    def update(_frame):
        counter, block = read_latest()
        if counter is None or counter == last_counter[0]:
            return
        last_counter[0] = counter
        spec = interleave(block)
        for k in range(adc_nos):
            data = safe_log(spec[k])
            lines[k].set_ydata(data)
            ymin, ymax = _np.min(data), _np.max(data)
            margin = (ymax - ymin) * 0.1 if (ymax - ymin) != 0 else 1.0
            axes[k].set_ylim(ymin - margin, ymax + margin)
            title = titles[k] if k < len(titles) else 'ADC {}'.format(k)
            axes[k].set_title('{}\nlive frame: {:d}'.format(title, counter), fontsize=8, pad=3)

    _keep = _anim.FuncAnimation(fig, update, interval=250, cache_frame_data=False)
    _plt.show()


# ─────────────────────────────────────────────
#  Schedule helper: fires a callback at a given UTC instant
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────
class SpectrometerGUI:
    BG       = '#0f1117'
    PANEL    = '#1a1d27'
    ACCENT   = '#00d4ff'
    ACCENT2  = '#7b61ff'
    TEXT     = '#e8eaf0'
    SUBTEXT  = '#6b7280'
    ENTRY_BG = '#252836'
    BORDER   = '#2e3245'
    SUCCESS  = '#22c55e'
    WARNING  = '#f59e0b'
    DANGER   = '#ef4444'
    FONT_H   = ('Courier New', 11, 'bold')
    FONT_B   = ('Courier New', 10)
    FONT_S   = ('Courier New', 9)

    MAX_ADC = 8

    def __init__(self, root):
        self.root = root
        self.root.title('CASPER Spectrometer Control v4')
        self.root.configure(bg=self.BG)
        self.root.resizable(False, False)
        self.root.protocol('WM_DELETE_WINDOW', self._on_exit)

        self.core       = None
        self.engine     = None
        self.recorder   = None
        self.fpga_ready = False
        self.recording  = False
        self.bitfile_var = tk.StringVar(value='')
        self._plot_proc  = None
        self._start_trigger = None
        self._stop_trigger  = None

        self.title_vars = [tk.StringVar(value='ADC {}'.format(i)) for i in range(self.MAX_ADC)]
        self.title_rows = []

        self._build_ui()

    # ── UI construction ────────────────────────
    def _build_ui(self):
        root = self.root
        tk.Frame(root, bg=self.ACCENT, height=3).pack(fill='x')

        header = tk.Frame(root, bg=self.BG, pady=14)
        header.pack(fill='x', padx=24)
        tk.Label(header, text='◈  CASPER', font=('Courier New', 15, 'bold'),
                 fg=self.ACCENT, bg=self.BG).pack(side='left')
        tk.Label(header, text='control interface · v4', font=self.FONT_S,
                 fg=self.SUBTEXT, bg=self.BG).pack(side='left', padx=10)

        body = tk.Frame(root, bg=self.BG)
        body.pack(fill='both', expand=True, padx=24, pady=(0, 20))
        left  = tk.Frame(body, bg=self.BG)
        mid   = tk.Frame(body, bg=self.BG)
        right = tk.Frame(body, bg=self.BG)
        left.pack(side='left', fill='both', expand=True, padx=(0, 8))
        mid.pack(side='left', fill='both', expand=True, padx=8)
        right.pack(side='left', fill='both', expand=True, padx=(8, 0))

        # ── PARAMETERS ──
        self._section(left, 'PARAMETERS')
        param_frame = self._card(left)
        params = [
            ('Hostname / IP',       'hostname', '169.254.192.228'),
            ('N Channels',          'nchannels', '4'),
            ('N FFT Points',        'nfft', '2048'),
            ('Sample Rate (MHz)*',  'fs', '2000.0'),
            ('Start Freq (MHz)',    'start_freq', '0'),
            ('Stop Freq (MHz)',     'stop_freq', '1000'),
        ]
        self.vars = {}
        for label, key, default in params:
            self._labeled_entry(param_frame, label, key, default)
        tk.Label(param_frame, text='*metadata only — axis comes from Start/Stop Freq',
                 font=('Courier New', 7), fg=self.SUBTEXT, bg=self.PANEL,
                 wraplength=220, justify='left').pack(fill='x', padx=4, pady=(2, 0))

        adc_row = tk.Frame(param_frame, bg=self.PANEL)
        adc_row.pack(fill='x', pady=2, padx=4)
        tk.Label(adc_row, text='BRAM Block Channels\n(4ch in one block)', font=self.FONT_S,
                 fg=self.SUBTEXT, bg=self.PANEL, width=22, anchor='w').pack(side='left')
        self.adc_var = tk.StringVar(value='1')
        self.adc_var.trace_add('write', self._on_adc_change)
        tk.Entry(adc_row, textvariable=self.adc_var, font=self.FONT_B,
                 bg=self.ENTRY_BG, fg=self.TEXT, insertbackground=self.ACCENT,
                 relief='flat', bd=4, width=16).pack(side='left', padx=(4, 0))

        toggle_row = tk.Frame(param_frame, bg=self.PANEL)
        toggle_row.pack(fill='x', pady=(8, 0))
        self.cx_var    = tk.BooleanVar(value=False)
        self.scale_var = tk.StringVar(value='log10')
        self._toggle(toggle_row, 'Log Scale', self.scale_var, side='left')
        self._toggle(toggle_row, 'CX Design', self.cx_var, side='left')

        # ── ACQUISITION ──
        # Integration is entirely hardware-driven: the FPGA's own 'acc_len'
        # vector accumulator does the averaging, so every read is already a
        # fully-integrated scan. The user only ever enters a desired
        # accumulation time in ms; the backend solves for acc_len.
        self._section(left, 'ACQUISITION (hardware-integrated)')
        acq_frame = self._card(left)

        int_row = tk.Frame(acq_frame, bg=self.PANEL)
        int_row.pack(fill='x', pady=2, padx=4)
        tk.Label(int_row, text='Accumulation time (ms)', font=self.FONT_S,
                 fg=self.SUBTEXT, bg=self.PANEL, width=22, anchor='w').pack(side='left')
        self.integ_var = tk.StringVar(value='1000')
        tk.Entry(int_row, textvariable=self.integ_var, font=self.FONT_B,
                 bg=self.ENTRY_BG, fg=self.TEXT, insertbackground=self.ACCENT,
                 relief='flat', bd=4, width=10).pack(side='left', padx=(4, 0))
        tk.Button(int_row, text='Apply', font=self.FONT_S, bg=self.ENTRY_BG,
                  fg=self.ACCENT, relief='flat', bd=0, padx=8, cursor='hand2',
                  command=self._apply_integration).pack(side='left', padx=6)

        self.acc_preview_lbl = tk.Label(acq_frame, text='', font=('Courier New', 7),
                                        fg=self.SUBTEXT, bg=self.PANEL, anchor='w',
                                        justify='left')
        self.acc_preview_lbl.pack(fill='x', padx=4, pady=(2, 0))

        rstwait_row = tk.Frame(acq_frame, bg=self.PANEL)
        rstwait_row.pack(fill='x', pady=(6, 2), padx=4)
        tk.Label(rstwait_row, text='Accumulator settle wait (s)', font=self.FONT_S,
                 fg=self.SUBTEXT, bg=self.PANEL, width=22, anchor='w').pack(side='left')
        self.reset_wait_var = tk.StringVar(value='5.0')
        tk.Entry(rstwait_row, textvariable=self.reset_wait_var, font=self.FONT_B,
                 bg=self.ENTRY_BG, fg=self.TEXT, insertbackground=self.ACCENT,
                 relief='flat', bd=4, width=10).pack(side='left', padx=(4, 0))
        tk.Label(acq_frame, text='"Apply" (re)writes acc_len, resets counters, then\n'
                                  'waits the settle time above — a one-off cost at\n'
                                  'program time or whenever this changes, not per-frame.',
                 font=('Courier New', 7), fg=self.SUBTEXT, bg=self.PANEL,
                 justify='left').pack(fill='x', padx=4, pady=(2, 6))

        self.integ_var.trace_add('write', self._update_acc_preview)
        self._update_acc_preview()

        plot_toggle_row = tk.Frame(acq_frame, bg=self.PANEL)
        plot_toggle_row.pack(fill='x', pady=(8, 0))
        self.enable_plot_var = tk.BooleanVar(value=True)
        self._toggle(plot_toggle_row, 'Enable Plot (separate process)', self.enable_plot_var, side='left')

        # ── SUBPLOT TITLES / DATA PRODUCTS ──
        self._section(left, 'CHANNEL NAMES (plot title = data product)')
        self.title_card = self._card(left)
        for i in range(self.MAX_ADC):
            row = tk.Frame(self.title_card, bg=self.PANEL)
            tk.Label(row, text='Ch {:d} name'.format(i), font=self.FONT_S,
                     fg=self.SUBTEXT, bg=self.PANEL, width=22, anchor='w').pack(side='left')
            tk.Entry(row, textvariable=self.title_vars[i], font=self.FONT_B,
                      bg=self.ENTRY_BG, fg=self.TEXT, insertbackground=self.ACCENT,
                      relief='flat', bd=4, width=16).pack(side='left', padx=(4, 0))
            self.title_rows.append(row)
        self._refresh_title_rows(1)

        # ── BITFILE (mid) ──
        self._section(mid, 'BITFILE')
        bit_frame = self._card(mid)
        bf_row = tk.Frame(bit_frame, bg=self.PANEL)
        bf_row.pack(fill='x')
        tk.Label(bf_row, textvariable=self.bitfile_var, font=self.FONT_S,
                 fg=self.SUBTEXT, bg=self.PANEL, anchor='w', wraplength=220,
                 justify='left').pack(side='left', fill='x', expand=True, pady=4)
        tk.Button(bf_row, text='Browse…', font=self.FONT_S, bg=self.ENTRY_BG,
                  fg=self.ACCENT, activebackground=self.BORDER,
                  activeforeground=self.ACCENT, relief='flat', bd=0, padx=10,
                  pady=4, cursor='hand2', command=self._pick_bitfile).pack(side='right')

        # ── RECORDING (mid) ──
        self._section(mid, 'RECORDING')
        rec_frame = self._card(mid)

        dir_row = tk.Frame(rec_frame, bg=self.PANEL)
        dir_row.pack(fill='x', pady=(0, 4))
        self.dir_var = tk.StringVar(value='')
        tk.Label(dir_row, text='Save Dir', font=self.FONT_S, fg=self.SUBTEXT,
                 bg=self.PANEL, width=22, anchor='w').pack(side='left')
        tk.Label(dir_row, textvariable=self.dir_var, font=self.FONT_S, fg=self.TEXT,
                 bg=self.PANEL, anchor='w', width=14).pack(side='left', padx=(4, 0))
        tk.Button(dir_row, text='…', font=self.FONT_S, bg=self.ENTRY_BG,
                  fg=self.ACCENT, relief='flat', bd=0, padx=6, pady=2,
                  cursor='hand2', command=self._pick_directory).pack(side='left', padx=4)

        fmt_row = tk.Frame(rec_frame, bg=self.PANEL)
        fmt_row.pack(fill='x', pady=(4, 4))
        tk.Label(fmt_row, text='File format', font=self.FONT_S, fg=self.SUBTEXT,
                 bg=self.PANEL, width=22, anchor='w').pack(side='left')
        self.format_var = tk.StringVar(value='auto')
        for val in ('auto', 'bin', 'csv'):
            tk.Radiobutton(fmt_row, text=val, variable=self.format_var, value=val,
                           font=self.FONT_S, fg=self.TEXT, bg=self.PANEL,
                           selectcolor=self.ENTRY_BG, activebackground=self.PANEL,
                           activeforeground=self.ACCENT, cursor='hand2').pack(side='left')

        rec_btn_row = tk.Frame(rec_frame, bg=self.PANEL)
        rec_btn_row.pack(fill='x', pady=(6, 0))
        self.start_rec_btn = tk.Button(rec_btn_row, text='● Start Rec', font=self.FONT_S,
                  bg=self.SUCCESS, fg=self.BG, relief='flat', bd=0, padx=8, pady=6,
                  cursor='hand2', command=lambda: self._on_start_recording())
        self.start_rec_btn.pack(side='left', fill='x', expand=True, padx=(0, 4))
        self.stop_rec_btn = tk.Button(rec_btn_row, text='■ Stop Rec', font=self.FONT_S,
                  bg=self.BORDER, fg=self.SUBTEXT, relief='flat', bd=0, padx=8, pady=6,
                  cursor='hand2', command=lambda: self._on_stop_recording(), state='disabled')
        self.stop_rec_btn.pack(side='left', fill='x', expand=True, padx=(4, 0))

        if not _HAVE_ASTROPY:
            tk.Label(rec_frame, text='(astropy not installed — FITS step of the\n'
                                      'compulsory post-processing pipeline will fail)',
                     font=self.FONT_S, fg=self.WARNING, bg=self.PANEL,
                     justify='left').pack(fill='x', pady=(6, 0))

        # ── SCHEDULE (mid) ──
        self._section(mid, 'SCHEDULE (optional)')
        sched_frame = self._card(mid)
        self.sched_enable_var = tk.BooleanVar(value=False)
        self._toggle(sched_frame, 'Use scheduled start/stop', self.sched_enable_var, side='left')

        self.sched_start = self._datetime_picker(sched_frame, 'Start')
        self.sched_stop  = self._datetime_picker(sched_frame, 'Stop')

        sched_btn_row = tk.Frame(sched_frame, bg=self.PANEL)
        sched_btn_row.pack(fill='x', pady=(6, 0))
        tk.Button(sched_btn_row, text='Arm Schedule', font=self.FONT_S, bg=self.ACCENT2,
                  fg=self.BG, relief='flat', bd=0, padx=8, pady=6, cursor='hand2',
                  command=self._on_arm_schedule).pack(side='left', fill='x', expand=True, padx=(0, 4))
        tk.Button(sched_btn_row, text='Cancel', font=self.FONT_S, bg=self.ENTRY_BG,
                  fg=self.DANGER, relief='flat', bd=0, padx=8, pady=6, cursor='hand2',
                  command=self._on_cancel_schedule).pack(side='left', fill='x', expand=True, padx=(4, 0))
        self.sched_status_lbl = tk.Label(sched_frame, text='not armed', font=self.FONT_S,
                                         fg=self.SUBTEXT, bg=self.PANEL)
        self.sched_status_lbl.pack(fill='x', pady=(4, 0))

        # ── STATUS (right) ──
        self._section(right, 'STATUS')
        status_frame = self._card(right)
        fpga_row = tk.Frame(status_frame, bg=self.PANEL)
        fpga_row.pack(fill='x', pady=4)
        tk.Label(fpga_row, text='FPGA', font=self.FONT_B, fg=self.TEXT,
                 bg=self.PANEL).pack(side='left')
        self.led = tk.Canvas(fpga_row, width=16, height=16, bg=self.PANEL,
                             highlightthickness=0)
        self.led.pack(side='left', padx=8)
        self._draw_led(False)
        self.fpga_status_lbl = tk.Label(fpga_row, text='not programmed', font=self.FONT_S,
                                        fg=self.SUBTEXT, bg=self.PANEL)
        self.fpga_status_lbl.pack(side='left')

        self.log = tk.Text(status_frame, height=20, width=38, bg=self.ENTRY_BG,
                           fg=self.SUBTEXT, font=('Courier New', 8), relief='flat',
                           bd=0, state='disabled', wrap='word')
        self.log.pack(fill='both', expand=True, pady=(8, 0))

        # ── ACTIONS (right) ──
        self._section(right, 'ACTIONS')
        btn_frame = self._card(right)
        self.prog_btn = self._action_btn(btn_frame, '⬡  PROGRAM FPGA', self.ACCENT2,
                                         self._on_program)
        self.prog_btn.pack(fill='x', pady=(0, 6))
        self.plot_btn = self._action_btn(btn_frame, '⬡  PLOT SPECTRUM', self.ACCENT,
                                         self._on_plot)
        self.plot_btn.pack(fill='x', pady=(0, 6))
        self.plot_btn.config(state='disabled', fg=self.SUBTEXT, bg=self.BORDER)
        tk.Button(btn_frame, text='✕  EXIT', font=self.FONT_H, bg=self.DANGER,
                  fg='white', activebackground='#b91c1c', activeforeground='white',
                  relief='flat', bd=0, padx=12, pady=10, cursor='hand2',
                  command=self._on_exit).pack(fill='x')

        tk.Frame(root, bg=self.BORDER, height=1).pack(fill='x')
        tk.Label(root, text='CASPER FPGA · Python Spectrometer Interface · v4',
                 font=('Courier New', 8), fg=self.SUBTEXT, bg=self.BG,
                 pady=6).pack()

    # ── date+time+tz picker widget ─────────────
    def _datetime_picker(self, parent, label):
        now_ist = datetime.now(IST)
        row = tk.Frame(parent, bg=self.PANEL)
        row.pack(fill='x', pady=(6, 0))
        tk.Label(row, text=label, font=self.FONT_S, fg=self.SUBTEXT, bg=self.PANEL,
                 width=6, anchor='w').pack(side='left')

        y = tk.StringVar(value=str(now_ist.year))
        mo = tk.StringVar(value='{:02d}'.format(now_ist.month))
        d = tk.StringVar(value='{:02d}'.format(now_ist.day))
        h = tk.StringVar(value='{:02d}'.format(now_ist.hour))
        mi = tk.StringVar(value='{:02d}'.format(now_ist.minute))
        s = tk.StringVar(value='{:02d}'.format(now_ist.second))
        tzv = tk.StringVar(value='IST')

        specs = [(y, [str(v) for v in range(now_ist.year, now_ist.year + 3)], 5),
                 (mo, ['{:02d}'.format(v) for v in range(1, 13)], 3),
                 (d,  ['{:02d}'.format(v) for v in range(1, 32)], 3),
                 (h,  ['{:02d}'.format(v) for v in range(0, 24)], 3),
                 (mi, ['{:02d}'.format(v) for v in range(0, 60)], 3),
                 (s,  ['{:02d}'.format(v) for v in range(0, 60)], 3)]
        for var, values, width in specs:
            cb = ttk.Combobox(row, textvariable=var, values=values, width=width,
                              state='readonly', font=self.FONT_S)
            cb.pack(side='left', padx=1)
        tz_cb = ttk.Combobox(row, textvariable=tzv, values=['IST', 'UTC'], width=4,
                             state='readonly', font=self.FONT_S)
        tz_cb.pack(side='left', padx=(4, 0))

        return dict(year=y, month=mo, day=d, hour=h, minute=mi, second=s, tz=tzv)

    def _picker_to_utc(self, picker):
        tzinfo = TZ_MAP[picker['tz'].get()]
        dt = datetime(int(picker['year'].get()), int(picker['month'].get()),
                      int(picker['day'].get()), int(picker['hour'].get()),
                      int(picker['minute'].get()), int(picker['second'].get()),
                      tzinfo=tzinfo)
        return dt.astimezone(UTC)

    # ── subplot title row show/hide ────────────
    def _refresh_title_rows(self, n):
        for i, row in enumerate(self.title_rows):
            if i < n:
                row.pack(fill='x', pady=2, padx=4)
            else:
                row.pack_forget()

    def _on_adc_change(self, *_):
        try:
            n = int(self.adc_var.get().strip())
            n = max(1, min(n, self.MAX_ADC))
            self._refresh_title_rows(n)
        except ValueError:
            pass

    def _update_acc_preview(self, *_):
        try:
            ms = float(self.integ_var.get().strip())
            acc_len = SpectrometerCore.acc_len_for_ms(ms)
            actual_ms = SpectrometerCore.scan_ms_for_acc_len(acc_len)
            self.acc_preview_lbl.config(
                text='-> acc_len = {} (actual scan ≈ {:.4f} ms)'.format(acc_len, actual_ms))
        except (ValueError, ZeroDivisionError):
            self.acc_preview_lbl.config(text='-> enter a valid accumulation time and N FFT')

    # ── widget helpers ─────────────────────────
    def _section(self, parent, title):
        row = tk.Frame(parent, bg=self.BG)
        row.pack(fill='x', pady=(12, 2))
        tk.Label(row, text=title, font=('Courier New', 8, 'bold'), fg=self.ACCENT2,
                 bg=self.BG).pack(side='left')
        tk.Frame(row, bg=self.BORDER, height=1).pack(side='left', fill='x',
                                                       expand=True, padx=(8, 0), pady=6)

    def _card(self, parent):
        f = tk.Frame(parent, bg=self.PANEL, bd=0, highlightthickness=1,
                     highlightbackground=self.BORDER)
        f.pack(fill='x', pady=2, ipady=8, ipadx=10)
        return f

    def _labeled_entry(self, parent, label, key, default):
        row = tk.Frame(parent, bg=self.PANEL)
        row.pack(fill='x', pady=2, padx=4)
        tk.Label(row, text=label, font=self.FONT_S, fg=self.SUBTEXT, bg=self.PANEL,
                 width=22, anchor='w').pack(side='left')
        var = tk.StringVar(value=default)
        tk.Entry(row, textvariable=var, font=self.FONT_B, bg=self.ENTRY_BG,
                 fg=self.TEXT, insertbackground=self.ACCENT, relief='flat', bd=4,
                 width=16).pack(side='left', padx=(4, 0))
        self.vars[key] = var

    def _toggle(self, parent, label, var, side):
        tk.Checkbutton(parent, text=label, variable=var, font=self.FONT_S,
                       fg=self.TEXT, bg=self.PANEL, selectcolor=self.ENTRY_BG,
                       activebackground=self.PANEL, activeforeground=self.ACCENT,
                       cursor='hand2').pack(side=side, padx=6)

    def _action_btn(self, parent, text, color, cmd):
        return tk.Button(parent, text=text, font=self.FONT_H, bg=color, fg=self.BG,
                         activebackground=self.ACCENT, activeforeground=self.BG,
                         relief='flat', bd=0, padx=12, pady=10, cursor='hand2', command=cmd)

    def _draw_led(self, armed):
        self.led.delete('all')
        color = self.SUCCESS if armed else self.DANGER
        glow  = '#86efac' if armed else '#fca5a5'
        self.led.create_oval(2, 2, 14, 14, fill=color, outline='')
        self.led.create_oval(4, 4, 8, 8, fill=glow, outline='')

    def _log(self, msg, color=None):
        self.log.config(state='normal')
        tag = 'tag{}'.format(self.log.index('end'))
        self.log.insert('end', msg + '\n', tag)
        if color:
            self.log.tag_config(tag, foreground=color)
        self.log.see('end')
        self.log.config(state='disabled')

    def _pick_bitfile(self):
        path = filedialog.askopenfilename(title='Select bitfile',
            filetypes=[('FPGA bitfiles', '*.fpg *.dtbo'), ('All', '*.*')])
        if path:
            self.bitfile_var.set(path)

    def _pick_directory(self):
        path = filedialog.askdirectory(title='Select save directory')
        if path:
            self.dir_var.set(path)

    # ── collect params ─────────────────────────
    def _collect_params(self):
        def intv(key):
            return int(self.vars[key].get().strip())
        def floatv(key):
            return float(self.vars[key].get().strip())

        adc_nos = int(self.adc_var.get().strip())
        titles = [self.title_vars[i].get().strip() or 'ADC {}'.format(i)
                  for i in range(adc_nos)]
        return dict(
            hostname       = self.vars['hostname'].get().strip(),
            nchannels      = intv('nchannels'),
            nfft           = intv('nfft'),
            fs             = floatv('fs'),
            start_freq     = floatv('start_freq'),
            stop_freq      = floatv('stop_freq'),
            adc_nos        = adc_nos,
            cx             = self.cx_var.get(),
            bitfile        = self.bitfile_var.get() or None,
            integration_ms = float(self.integ_var.get().strip()),
        ), titles

    # ── PROGRAM ─────────────────────────────────
    def _on_program(self):
        if not self.bitfile_var.get().strip():
            messagebox.showwarning('No bitfile', 'Please select a bitfile first.')
            return
        self.prog_btn.config(state='disabled', text='Programming…')
        self._log('Connecting to FPGA…', self.WARNING)

        def _status(msg):
            self.root.after(0, lambda m=msg: self._log(m, self.WARNING))

        def _work():
            try:
                params, titles = self._collect_params()
                reset_wait = float(self.reset_wait_var.get().strip() or 5.0)
                self.core = SpectrometerCore(**params)
                self.core.data_products = titles
                self.core.program_fpga()
                # Mandatory before any read: program acc_len (already computed
                # from the requested accumulation ms in SpectrometerCore.__init__),
                # reset counters, and wait for the hardware accumulator to settle.
                self.core.configure_accumulation(reset_wait=reset_wait, on_status=_status)
                live_dir = os.path.join(self.dir_var.get().strip() or '.', '.live')
                self.engine = AcquisitionEngine(self.core, live_dir)
                self.engine.start()
                self.root.after(0, self._on_program_success)
            except Exception as exc:
                self.root.after(0, lambda e=str(exc): self._on_program_fail(e))

        threading.Thread(target=_work, daemon=True).start()

    def _on_program_success(self):
        self.fpga_ready = True
        self._draw_led(True)
        self.fpga_status_lbl.config(text='programmed ✓', fg=self.SUCCESS)
        self.prog_btn.config(state='normal', text='⬡  PROGRAM FPGA')
        self.plot_btn.config(state='normal', fg=self.BG, bg=self.ACCENT)
        self.start_rec_btn.config(state='normal', bg=self.SUCCESS, fg=self.BG)
        self._log('FPGA programmed. Background acquisition running '
                   '(hardware-integrated: acc_len={}, actual scan ≈ {:.4f} ms).'.format(
                       self.core.acc_len, self.core.integration_ms), self.SUCCESS)
        self._log('Freq axis: {} - {} MHz.'.format(
            self.core.start_freq, self.core.stop_freq))

    def _on_program_fail(self, err):
        self._draw_led(False)
        self.fpga_status_lbl.config(text='error', fg=self.DANGER)
        self.prog_btn.config(state='normal', text='⬡  PROGRAM FPGA')
        self._log('ERROR: ' + err, self.DANGER)
        messagebox.showerror('Programming failed', err)

    # ── integration time ───────────────────────
    def _apply_integration(self):
        """Integration is hardware-driven, so changing it means rewriting
        acc_len on the FPGA and letting the accumulator settle again — not
        a software-only change. Briefly pauses acquisition while that
        happens.
        """
        if self.engine is None or self.core is None:
            messagebox.showwarning('Not ready', 'Program the FPGA first.')
            return
        if self.recording:
            messagebox.showwarning('Recording active', 'Stop recording before changing '
                                    'the accumulation time.')
            return
        try:
            ms = float(self.integ_var.get().strip())
        except ValueError:
            messagebox.showerror('Invalid value', 'Accumulation time must be a number (ms).')
            return
        reset_wait = float(self.reset_wait_var.get().strip() or 5.0)
        acc_len = SpectrometerCore.acc_len_for_ms(ms)

        self._log('Reconfiguring hardware accumulation for {:.4f} ms '
                   '(acc_len={})…'.format(ms, acc_len), self.WARNING)

        def _status(msg):
            self.root.after(0, lambda m=msg: self._log(m, self.WARNING))

        def _work():
            self.engine.stop()
            self.core.configure_accumulation(acc_len=acc_len, reset_wait=reset_wait,
                                             on_status=_status)
            self.engine.start()
            self.root.after(0, lambda: self._log(
                'Accumulation updated: acc_len={}, actual scan ≈ {:.4f} ms.'.format(
                    self.core.acc_len, self.core.integration_ms), self.SUCCESS))

        threading.Thread(target=_work, daemon=True).start()

    # ── schedule ────────────────────────────────
    def _on_arm_schedule(self):
        if not self.sched_enable_var.get():
            messagebox.showinfo('Schedule disabled', 'Check "Use scheduled start/stop" first.')
            return
        try:
            start_utc = self._picker_to_utc(self.sched_start)
            stop_utc  = self._picker_to_utc(self.sched_stop)
        except ValueError as exc:
            messagebox.showerror('Invalid date/time', str(exc))
            return
        if stop_utc <= start_utc:
            messagebox.showerror('Invalid schedule', 'Stop time must be after start time.')
            return

        self._on_cancel_schedule()  # clear any previous armed triggers
        info = dict(start_utc=start_utc.isoformat(), stop_utc=stop_utc.isoformat())

        def _fire_start():
            self.root.after(0, lambda: self._on_start_recording(schedule_info=info))

        def _fire_stop():
            self.root.after(0, self._on_stop_recording)

        self._start_trigger = ScheduleTrigger(start_utc, _fire_start)
        self._stop_trigger  = ScheduleTrigger(stop_utc, _fire_stop)
        self._start_trigger.start()
        self._stop_trigger.start()

        self.sched_status_lbl.config(
            text='armed: {} -> {} UTC'.format(
                start_utc.strftime('%Y-%m-%d %H:%M:%S'),
                stop_utc.strftime('%Y-%m-%d %H:%M:%S')), fg=self.SUCCESS)
        self._log('Schedule armed (UTC): {} -> {}'.format(
            start_utc.isoformat(), stop_utc.isoformat()), self.ACCENT)

    def _on_cancel_schedule(self):
        if self._start_trigger:
            self._start_trigger.cancel()
        if self._stop_trigger:
            self._stop_trigger.cancel()
        self._start_trigger = self._stop_trigger = None
        self.sched_status_lbl.config(text='not armed', fg=self.SUBTEXT)

    # ── recording ───────────────────────────────
    def _on_start_recording(self, schedule_info=None):
        if self.engine is None:
            messagebox.showwarning('Not ready', 'Program the FPGA first.')
            return
        if self.recording:
            return
        directory = self.dir_var.get().strip() or '.'
        self.recorder = Recorder(directory, self.core, file_format=self.format_var.get())
        fmt, raw_path = self.recorder.start(schedule_info=schedule_info)
        self.engine.attach_recorder(self.recorder.queue)
        self.recording = True
        self.start_rec_btn.config(state='disabled', bg=self.BORDER, fg=self.SUBTEXT)
        self.stop_rec_btn.config(state='normal', bg=self.DANGER, fg='white')
        self._log('Recording started ({}) -> single raw dump, no per-frame headers.'.format(fmt),
                   self.SUCCESS)
        self._log('  UTC start: {}'.format(
            self.recorder.start_utc.strftime('%Y-%m-%d %H:%M:%S.%f')))
        self._log('  IST start: {}'.format(
            self.recorder.start_ist.strftime('%Y-%m-%d %H:%M:%S.%f')))
        self._log('  -> {}'.format(raw_path))
        self._log('  -> {}'.format(self.recorder.meta_path))

    def _on_stop_recording(self):
        if not self.recording:
            return
        self.engine.detach_recorder()
        self.recorder.stop()
        self.recording = False
        self.start_rec_btn.config(state='normal', bg=self.SUCCESS, fg=self.BG)
        self.stop_rec_btn.config(state='disabled', bg=self.BORDER, fg=self.SUBTEXT)
        self._log('Recording stopped. {} frames written.'.format(self.recorder.n_written),
                   self.WARNING)
        self._log('  UTC stop: {}'.format(
            self.recorder.stop_utc.strftime('%Y-%m-%d %H:%M:%S.%f')))
        self._log('  IST stop: {}'.format(
            self.recorder.stop_ist.strftime('%Y-%m-%d %H:%M:%S.%f')))
        self._start_postprocessing()

    # ── compulsory post-processing (segregate -> zip -> FITS) ─────────
    def _start_postprocessing(self):
        recorder = self.recorder
        directory = recorder.directory
        self._log('Post-processing: segregating channels, zipping, converting to FITS, '
                   'then removing intermediates…', self.ACCENT)

        def _work():
            try:
                zip_path, fits_path = run_postprocessing(
                    recorder.raw_path, recorder.meta_path, directory)
                self.root.after(0, lambda: self._on_postprocess_success(zip_path, fits_path))
            except Exception as exc:
                self.root.after(0, lambda e=str(exc): self._on_postprocess_fail(e))

        threading.Thread(target=_work, daemon=True).start()

    def _on_postprocess_success(self, zip_path, fits_path):
        self._log('Post-processing complete — raw dump, metadata JSON and the '
                   'segregated folder were deleted; only the zip and FITS remain.',
                   self.SUCCESS)
        self._log('  zip: {}'.format(zip_path))
        self._log('  FITS: {}'.format(fits_path))

    def _on_postprocess_fail(self, err):
        self._log('Post-processing error: ' + err, self.DANGER)
        messagebox.showerror('Post-processing failed', err)

    # ── PLOT (separate process) ────────────────
    def _on_plot(self):
        if not self.fpga_ready or self.engine is None:
            messagebox.showwarning('Not ready', 'Program the FPGA first.')
            return
        if not self.enable_plot_var.get():
            messagebox.showinfo('Plot disabled', '"Enable Plot" is unchecked — '
                                 'acquisition keeps running headless.')
            return
        if self._plot_proc is not None and self._plot_proc.is_alive():
            self._log('Plot window already open.', self.WARNING)
            return

        cfg = dict(
            nchannels=self.core.nchannels, adc_nos=self.core.adc_nos, nfft=self.core.nfft,
            chunk=self.core.chunk, n_blocks=self.core.n_blocks,
            faxis=self.core.faxis.tolist(), titles=self.core.data_products,
            scale=self.scale_var.get(),
            live_data_path=self.engine.live_data_path,
            live_meta_path=self.engine.live_meta_path,
        )
        self._plot_proc = multiprocessing.Process(target=_plot_worker, args=(cfg,), daemon=True)
        self._plot_proc.start()
        self._log('Spectrum window opened in a separate process (reads live-buffer '
                   'file on disk; no in-memory transfer from the acquisition thread).',
                   self.SUCCESS)
        self._poll_plot_process()

    def _poll_plot_process(self):
        if self._plot_proc is not None and self._plot_proc.is_alive():
            self.root.after(500, self._poll_plot_process)
        else:
            self._plot_proc = None
            self._log('Spectrum window closed.', self.SUBTEXT)

    # ── EXIT ─────────────────────────────────────
    def _on_exit(self):
        if messagebox.askokcancel('Exit', 'Close the spectrometer interface?'):
            self._on_cancel_schedule()
            if self.recording:
                self._on_stop_recording()
            if self.engine:
                self.engine.stop()
            if self._plot_proc is not None and self._plot_proc.is_alive():
                self._plot_proc.terminate()
            plt.close('all')
            self.root.destroy()


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    root = tk.Tk()
    app = SpectrometerGUI(root)
    root.mainloop()
