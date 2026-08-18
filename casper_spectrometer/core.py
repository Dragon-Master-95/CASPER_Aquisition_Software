"""Hardware-facing core: owns the FPGA connection and the pure numpy math.

SpectrometerCore is deliberately free of any per-frame bookkeeping —
``read_raw_block()`` is as cheap as the mandated sequential reads allow,
and ``interleave()`` is a pure function reused identically by the live-plot
process and the post-recording segregation step, so both reconstruct data
exactly the same way.
"""

import time

import numpy as np
import casperfpga


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
