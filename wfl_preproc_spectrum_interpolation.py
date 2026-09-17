import numpy as np
import mne
from mne.io.pick import _picks_to_idx


def spectrum_interpolation(raw, Fl, bandwidth, neighbourwidth, exclude=None):
    """
    Spectrum interpolation for removing narrow-band interference.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object.
    Fl : float or array-like
        Frequency (or frequencies) to interpolate.
    bandwidth : float or array-like
        Half bandwidth (Hz) around each target frequency.
    neighbourwidth : float or array-like
        Width (Hz) of neighbouring frequency bands used for interpolation.
    exclude : list | tuple | None
        Channels to exclude.

    Returns
    -------
    raw_interpolation : mne.io.Raw
        Spectrum interpolated Raw object.

    Notes
    -----
    For detailed usage instructions and examples, please refer to the
    GitHub repository:
    https://github.com/FulongWangBuaa/spectrum-interpolation-for-python
    """

    Fs = raw.info['sfreq']
    picks = None

    if exclude is not None:
        picks = _picks_to_idx(raw.info, picks, exclude=exclude)
    else:
        picks = _picks_to_idx(raw.info, picks, exclude=())

    picks_good, picks_bad = list(), list()

    for ii, pi in enumerate(picks):
        if raw.ch_names[pi] in raw.info["bads"]:
            picks_bad.append(ii)
        else:
            picks_good.append(ii)

    picks_good = np.array(picks_good, int)
    picks_bad = np.array(picks_bad, int)

    dat = raw.get_data()[picks_good, :]

    nchans, nsamples = dat.shape

    if np.any(np.isnan(dat)):
        print("Warning: data contains NaN values.")

    Fl = np.atleast_1d(Fl)
    bandwidth = np.atleast_1d(bandwidth)
    neighbourwidth = np.atleast_1d(neighbourwidth)

    # ------------------------------------------------------------------
    # Calculate the largest data length containing an integer number
    # of cycles for each interpolation frequency
    # ------------------------------------------------------------------
    n = np.round(
        np.floor(nsamples * (Fl / Fs + 100 * np.finfo(float).eps))
        * Fs / Fl
    ).astype(int)

    # ------------------------------------------------------------------
    # Multiple frequencies -> recursive processing
    # ------------------------------------------------------------------
    if np.size(n) > 1 or not np.all(n == n[0]):
        filt = raw.copy()

        for i in range(np.size(Fl)):
            filt = spectrum_interpolation(
                raw=filt,
                Fl=Fl[i],
                bandwidth=bandwidth[i],
                neighbourwidth=neighbourwidth[i],
                exclude=exclude
            )

        return filt

    # ------------------------------------------------------------------
    # Automatically crop to integer number of cycles
    # ------------------------------------------------------------------
    n = int(n[0])

    if n != nsamples:

        print(
            f"Spectrum interpolation: crop data from "
            f"{nsamples} -> {n} samples "
            f"({nsamples/Fs:.3f}s -> {n/Fs:.3f}s)"
        )

        dat = dat[:, :n]
        nsamples = n

    if np.size(Fl) < np.size(bandwidth):
        bandwidth = bandwidth[:np.size(Fl)]

    if np.size(Fl) < np.size(neighbourwidth):
        neighbourwidth = neighbourwidth[:np.size(Fl)]

    if np.size(Fl) != np.size(bandwidth) or np.size(Fl) != np.size(neighbourwidth):
        raise ValueError(
            "The number of frequencies to interpolate should equal "
            "the number of bandwidths and neighbourwidths."
        )

    nfft = nsamples

    # frequencies to interpolate
    f2int = np.array([Fl - bandwidth, Fl + bandwidth]).T

    # neighbouring frequencies
    f4int = np.array([
        f2int[:, 0] - neighbourwidth,
        f2int[:, 0],
        f2int[:, 1],
        f2int[:, 1] + neighbourwidth
    ]).T

    data_fft = np.fft.fft(dat, n=nfft, axis=1)

    frq = np.linspace(
        start=0,
        stop=1,
        num=nfft + 1
    ) * Fs

    # ------------------------------------------------------------------
    # Spectrum interpolation
    # ------------------------------------------------------------------
    for i in range(np.size(Fl)):

        smpl2int = np.arange(
            np.argmin(np.abs(frq - f2int[i, 0])),
            np.argmin(np.abs(frq - f2int[i, 1])) + 1
        )

        low_neighbouring = np.arange(
            np.argmin(np.abs(frq - f4int[i, 0])),
            np.argmin(np.abs(frq - f4int[i, 1]))
        )

        high_neighbouring = np.arange(
            np.argmin(np.abs(frq - f4int[i, 2])) + 1,
            np.argmin(np.abs(frq - f4int[i, 3])) + 1
        )

        smpl4int = np.concatenate(
            (low_neighbouring, high_neighbouring)
        )

        mean_amp = np.mean(
            np.abs(data_fft[:, smpl4int]),
            axis=1,
            keepdims=True
        )

        mean_amp = np.ones(
            data_fft[:, smpl2int].shape
        ) * mean_amp

        data_fft[:, smpl2int] = (
            np.exp(1j * np.angle(data_fft[:, smpl2int]))
            * mean_amp
        )

    # ------------------------------------------------------------------
    # Reconstruct conjugate symmetric spectrum
    # ------------------------------------------------------------------
    half_len = data_fft.shape[1] // 2

    data_fft = np.column_stack((
        data_fft[:, 0],
        data_fft[:, 1:half_len],
        data_fft[:, half_len],
        np.conj(np.flip(data_fft[:, 1:half_len], axis=1))
    ))

    filt = np.fft.ifft(data_fft, axis=1).real

    # ------------------------------------------------------------------
    # Create output Raw
    # ------------------------------------------------------------------
    raw_interpolation = raw.copy()

    # Crop Raw to the new duration
    raw_interpolation.crop(
        tmin=0,
        tmax=(nsamples - 1) / Fs
    )

    raw_interpolation._data[picks_good, :] = filt

    return raw_interpolation