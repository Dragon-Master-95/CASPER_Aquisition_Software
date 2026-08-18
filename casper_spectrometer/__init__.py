"""
casper_spectrometer
====================

Modular control software for a CASPER/CasperFPGA-based spectrometer
(ZCU216-class boards). This package was split out of the original
monolithic ``zcu_comm_gui_v4.py`` script into separate modules so each
concern (hardware I/O, acquisition, recording, post-processing, live
plotting, scheduling, GUI) can be read, tested, and reused on its own.

Modules
-------
timeutil        Timezone constants (UTC/IST) shared across the package.
core            SpectrometerCore — owns the FPGA connection and the pure
                numpy math (acc_len<->ms conversion, block interleaving).
acquisition     AcquisitionEngine — background thread that reads
                hardware-integrated scans and feeds the live buffer /
                recorder queue.
recorder        Recorder — writes one flat raw file per recording session
                plus its metadata JSON.
postprocessing  Segregation, zip packaging, and compulsory FITS export
                run once a recording stops.
plotting        Live spectrum plot, run in its own OS process.
scheduler       ScheduleTrigger — fires a callback at a target UTC time.
gui             SpectrometerGUI — the Tkinter control interface tying all
                of the above together.

See the top-level ``main.py`` for the application entry point, and the
project README for setup and usage instructions.
"""

__version__ = '4.0.0'
