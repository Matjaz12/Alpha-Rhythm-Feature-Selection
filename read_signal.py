import mne
import numpy as np

FREQ_BAND = (3.0, 30.0) # [Hz]

def read_sig(sig_path, band_pass_filter=True):
    raw_edf = mne.io.read_raw_edf(sig_path, preload=True)
    if band_pass_filter:
        raw_edf.filter(FREQ_BAND[0], FREQ_BAND[1], fir_design='firwin', skip_by_annotation='edge', verbose=False)

    data = raw_edf.get_data() # shape (n_signals x n_samples)
    ch_names = raw_edf.ch_names
    times = raw_edf.times
    fs = 1 / abs(times[1] - times[0])
    annot = raw_edf.annotations # OrderedDict with keys: onset, duration, description

    return data, ch_names, annot, times, fs

if __name__ == "__main__":
    SIG_PATH = "./data/S001R04.edf"
    data, ch_names, annot, times, fs = read_sig(sig_path=SIG_PATH)
    print(f"data.shape: {data.shape}")
    print(f"sampling freq [Hz]: {data.shape}")
    print(f"annotations: {np.unique(annot.description)}")
