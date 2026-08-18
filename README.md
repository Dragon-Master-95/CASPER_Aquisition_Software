# CASPER Spectrometer Control GUI

A Tkinter control application for a CASPER/CasperFPGA-based spectrometer
(ZCU216-class boards). It programs the FPGA, configures hardware-side
integration, streams and live-plots spectra, records raw acquisitions to
disk on a manual or scheduled start/stop, and — once a recording stops —
compulsorily post-processes it into a self-describing FITS file.

This repo is the modular version of the original single-file
`zcu_comm_gui_v4.py` script: the same behavior, split into a package so
each concern can be read, tested, and reused independently of the GUI.

## Features

- **FPGA programming & hardware-side integration.** Accumulation is done
  entirely on the FPGA via the `acc_len` vector-accumulator register — you
  enter a desired accumulation time in milliseconds and the software
  solves for the integer `acc_len` that produces it, then performs the
  mandatory register write + counter reset + settle-time sequence.
- **Continuous background acquisition** in its own thread, paced to the
  hardware's own scan period so frames stay evenly spaced in wall-clock
  time.
- **Live spectrum plotting** in a separate OS process (not thread) that
  polls a small live-buffer file pair on disk — it never touches the
  acquisition thread's memory, so plotting cannot add latency to
  acquisition and vice versa.
- **Recording** to a single flat, header-free raw file per session, with
  one metadata JSON capturing everything needed to reconstruct or
  interpret it later (hostname, bitfile, nfft, acc_len, frequency axis,
  channel/data-product names, start/stop times in UTC and IST, ...).
- **Manual or scheduled** recording start/stop, with an explicit
  date + time + timezone (IST/UTC) picker for both edges.
- **Compulsory post-processing** once a recording stops: the raw dump is
  segregated into per-ADC arrays, zipped together with its metadata, and
  converted to a multi-extension FITS file whose header is built entirely
  from the metadata JSON. Once the zip and FITS file exist, every
  intermediate (raw dump, metadata JSON, segregated directory) is deleted
  — only the zip and the FITS file remain on disk.

## Repository layout

```
.
├── main.py                       # entry point — run this to launch the GUI
├── requirements.txt
├── README.md
└── casper_spectrometer/          # the application package
    ├── __init__.py
    ├── timeutil.py                # UTC/IST timezone constants
    ├── core.py                    # SpectrometerCore: FPGA connection + pure numpy math
    ├── acquisition.py             # AcquisitionEngine: background read loop
    ├── recorder.py                # Recorder: raw file + metadata JSON writer
    ├── postprocessing.py          # segregate -> zip -> FITS pipeline
    ├── plotting.py                # live-plot worker (runs in its own process)
    ├── scheduler.py                # ScheduleTrigger: fires a callback at a UTC instant
    └── gui.py                     # SpectrometerGUI: the Tkinter application
```

Each module can be imported and used on its own — e.g. `core.py` and
`postprocessing.py` have no Tkinter dependency, so they can be scripted or
unit-tested headlessly.

## Installation

1. **Python.** Tested on Python 3.8+.

2. **System package for Tkinter** (only needed if `python3 -c "import
   tkinter"` fails). On Debian/Ubuntu:

   ```bash
   sudo apt install python3-tk
   ```

3. **Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **casperfpga.** Not distributed on PyPI — install it from the CASPER
   GitHub repository:

   ```bash
   pip install git+https://github.com/casper-astro/casperfpga.git
   ```

   or clone it locally and `pip install -e .` from the checkout if you
   need a specific branch/version.

## Usage

```bash
python3 main.py
```

This opens the control window with three columns:

- **Left — Parameters / Acquisition / Channel names.** FPGA hostname/IP,
  channel and FFT geometry, sample rate (metadata only — the plotted/FITS
  frequency axis is always `linspace(start_freq, stop_freq, nfft)`),
  desired accumulation time in ms, and a name per ADC/data product (this
  name doubles as the plot subplot title and the FITS extension name).
- **Middle — Bitfile / Recording / Schedule.** Pick a `.fpg`/`.dtbo`
  bitfile, pick a save directory and raw file format (`auto` benchmarks
  bin vs. csv write speed on your disk and picks the faster one), and
  optionally arm a scheduled start/stop with an explicit date + time +
  timezone.
- **Right — Status / Actions.** FPGA state LED, a scrolling log, and the
  `PROGRAM FPGA` / `PLOT SPECTRUM` / `EXIT` buttons.

Typical flow:

1. Select a bitfile and press **PROGRAM FPGA**. This uploads the bitfile,
   writes `acc_len`, resets counters, waits the settle time (default 5s),
   and starts background acquisition.
2. Press **PLOT SPECTRUM** to open a live view (a separate process; safe
   to leave closed if you only want to record).
3. Press **● Start Rec** to begin recording; **■ Stop Rec** to end it, or
   arm a schedule instead of doing this manually.
4. When recording stops, post-processing runs automatically. Watch the
   log for the final `.zip` and `.fits` paths.

### Note on accumulation time vs. `acc_len`

`acc_len` is an integer hardware register; `SpectrometerCore.MS_PER_ACC_LEN`
is the calibration constant (ms per `acc_len` count) for the current FPGA
design. The GUI shows a live "-> acc_len = N (actual scan ≈ X ms)" preview
as you type, because rounding to the nearest integer register value means
the actual scan period can differ slightly from what you asked for — the
*actual* value (not your requested one) is what gets recorded in the
metadata JSON and, later, in the FITS header.

## Output files

For a recording started at `20260101-120000`:

- During recording: `raw_20260101-120000.bin` (or `.csv`) and
  `meta_20260101-120000.json` in the chosen save directory.
- After post-processing (automatic): `recording_20260101-120000.zip`
  (containing the raw dump, metadata JSON, and per-ADC segregated files)
  and `spectrometer_<UTC-start>.fits`. The raw dump, metadata JSON, and
  segregated directory are deleted once both of these exist.

## Requirements

See [`requirements.txt`](requirements.txt). In short: `numpy`,
`matplotlib`, `astropy` (FITS export is compulsory, not optional), Tkinter
(stdlib/OS package), and `casperfpga` (installed from GitHub, not PyPI).

## License

Add a license file appropriate for your institution/project before
publishing this repository publicly.
