"""Multi-processing live-plot worker.

``plot_worker`` runs in its own OS process (spawned by the GUI). It polls
the live-buffer file pair written by
:class:`casper_spectrometer.acquisition.AcquisitionEngine` on disk and
redraws — no array is ever passed through a Queue/Pipe from the
acquisition thread, so this process cannot add latency to acquisition and
acquisition (a different process entirely) cannot stall this plot.

Kept at module level (rather than as a method) because it must be
picklable for ``multiprocessing.Process``.
"""


def plot_worker(cfg):
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
