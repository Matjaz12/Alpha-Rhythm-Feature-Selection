from feature_selection import compute_R2_map, plot_R2_map
from read_signal import read_sig
from utils import get_intervals


def main(sig1_path, sig2_path, freq_band):
    data1, ch_names1, annot1, times1, fs1 = read_sig(sig1_path, freq_band)
    intervals1 = get_intervals(data1, annot1, fs1, n_intervals=15)
    data2, ch_names2, annot2, times2, fs2 = read_sig(sig2_path, freq_band)
    intervals2 = get_intervals(data2, annot2, fs2, n_intervals=15)

    R2 = compute_R2_map(intervals1, intervals2, fs=fs1)
    print(f"R2.min(): {R2.min()}, R2.max(): {R2.max()}")
    print(f"R2.shape: {R2.shape}")
    plot_R2_map(R2, max_freq=30, fs=fs1)


if __name__ == "__main__":
    SIG1_PATH = "./data/S001R01.edf" # eyes open
    SIG2_PATH = "./data/S001R02.edf" # eyes closes
    FREQ_BAND = (3.0, 30.0)          # freq. band to keep [Hz]

    main(SIG1_PATH, SIG2_PATH, FREQ_BAND)
