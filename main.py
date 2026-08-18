#!/usr/bin/env python3
"""
CASPER Spectrometer control GUI — entry point.

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

See README.md for setup/usage and requirements.txt for dependencies.
"""

import multiprocessing
import tkinter as tk

from casper_spectrometer.gui import SpectrometerGUI


def main():
    multiprocessing.set_start_method('spawn', force=True)
    root = tk.Tk()
    SpectrometerGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
