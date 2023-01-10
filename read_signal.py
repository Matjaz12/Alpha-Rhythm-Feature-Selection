import mne
import numpy as np


def read_sig(sig_path, freq_band=(None, None)):
    raw_edf = mne.io.read_raw_edf(sig_path, preload=True)
    
    if freq_band[0] and freq_band[1]:
        print(f"Filtering signal using freq_band: {freq_band}")
        raw_edf.filter(freq_band[0], freq_band[1], fir_design='firwin', skip_by_annotation='edge', verbose=False)

    data = raw_edf.get_data() # shape (n_signals x n_samples)
    ch_names = raw_edf.ch_names
    times = raw_edf.times
    fs = 1 / abs(times[1] - times[0])
    annot = raw_edf.annotations # OrderedDict with keys: onset, duration, description

    return data, ch_names, annot, times, fs


if __name__ == "__main__":
    SIG_PATH = "./data/S001R04.edf"
    data, ch_names, annot, times, fs = read_sig(sig_path=SIG_PATH, freq_band=(3.0, 30.0))
    print(f"data.shape: {data.shape}")
    print(f"sampling freq [Hz]: {data.shape}")
    print(f"annotations: {np.unique(annot.description)}")
