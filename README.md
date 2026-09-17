# 1. Overview
Suppression of abnormal spectral peaks in the power spectral density of MEG/EEG signals for improved spectral analysis and noise characterization.

---

# 2. Usage Examples

## S3P
```python
from wfl_preproc_s3p import s3p
# import data
raw = mne.io.read_raw_fif(raw_path,preload=True)
raw_room = mne.io.read_raw_fif(raw_room_path,preload=True)

# filter
raw_filt = raw.copy().filter(2, 45, fir_design='firwin').notch_filter(50)
raw_room_filt = raw_room.copy().filter(2, 45, fir_design='firwin').notch_filter(50)

# s3p
# Since the spike interference is centered at 30 Hz, S3P is applied exclusively to the 28–33 Hz frequency band
# n_noise denotes the number of noise components removed
raw_s3p = s3p(raw_filt,raw_room_filt,fmin=28,fmax=33,n_noise=2)
```

## Spectrum interpolation
```python
from wfl_preproc_spectrum_interpolation import spectrum_interpolation
raw = mne.io.read_raw_fif(raw_path,preload=True)
Fl = [50]
dftbandwidth = [2]
dftneighbourwidth = [2]
raw_interpolation = spectrum_interpolation(raw,Fl, dftbandwidth, dftneighbourwidth)
```

# Acknowledgements
- [1] Leske S., Dalal S. S. Reducing Power Line Noise in EEG and MEG Data via Spectrum Interpolation[J]. NeuroImage, 2019,189: 763-776. [DOI: https://doi.org/10.1016/j.neuroimage.2019.01.026](https://doi.org/10.1016/j.neuroimage.2019.01.026)
- [2] Ramírez R R, Kopell B H, Butson C R, et al. Spectral signal space projection algorithm for frequency domain MEG and EEG denoising, whitening, and source imaging[J]. NeuroImage, 2011, 56(1): 78-92. [DOI: https://doi.org/10.1016/j.neuroimage.2011.02.002](https://doi.org/10.1016/j.neuroimage.2011.02.002)
