"""Tkinter control interface tying together core/acquisition/recorder/
postprocessing/plotting/scheduler into one application.
"""

import os
import threading
import multiprocessing

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

from .core import SpectrometerCore
from .acquisition import AcquisitionEngine
from .recorder import Recorder
from .postprocessing import run_postprocessing, HAVE_ASTROPY
from .plotting import plot_worker
from .scheduler import ScheduleTrigger
from .timeutil import UTC, IST, TZ_MAP


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

        if not HAVE_ASTROPY:
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
        self._plot_proc = multiprocessing.Process(target=plot_worker, args=(cfg,), daemon=True)
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
