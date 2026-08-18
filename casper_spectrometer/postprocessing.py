"""Post-processing: segregate, zip, and (compulsorily) convert to FITS.

This all runs once a recording stops. Every header/description field in
the FITS output is built entirely from the metadata JSON written by
:class:`casper_spectrometer.recorder.Recorder` — nothing here is
free-typed by the user.
"""

import os
import json
import zipfile
import shutil
from datetime import datetime

import numpy as np

try:
    from astropy.io import fits
    HAVE_ASTROPY = True
except ImportError:
    HAVE_ASTROPY = False

from .timeutil import UTC


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
    if not HAVE_ASTROPY:
        raise RuntimeError('astropy is required for FITS export (pip install astropy)')

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    nfft = meta['nfft']

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


def cleanup_intermediates(raw_path, meta_path, segregated_paths):
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
    cleanup_intermediates(raw_path, meta_path, segregated)
    return zip_path, fits_path
