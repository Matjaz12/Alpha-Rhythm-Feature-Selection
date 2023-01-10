import matplotlib.pyplot as plt
import numpy as np
import scipy

from read_signal import read_sig


def get_intervals(data, annot, fs, n_intervals):
    intervals = []

    # In this case we only have one interval (i.e interval T0)
    dur = annot[0]["duration"] * fs     # unit: [smp]
    start = int(annot[0]["onset"] * fs)
    end = int(start + dur)

    if annot[0]["description"] == "T0":
        intervals.append(data[:, start:end])

    # Split into n_intervals
    # intervals = np.array_split(intervals[0], n_intervals, axis=1)
    # min_n = intervals[0].shape[1] - 1
    # intervals_new = np.array([inter[:, 0:min_n] for inter in intervals])

    intervals = np.array(intervals[0])
    # delta is at most (n_intervals - 1) which is not large compared to signal length
    delta = intervals.shape[1] % n_intervals
    print(delta)
    intervals = intervals[:, 0:intervals.shape[1] - delta]
    intervals = np.split(intervals, n_intervals, axis=1)
    intervals = np.array(intervals)
    
    return intervals


if __name__ == "__main__":
    SIG_PATH = "./data/S001R01.edf"
    data, ch_names, annot, times, fs = read_sig(sig_path=SIG_PATH)
    intervals1 = get_intervals(data, annot, fs, n_intervals=15)
    print(intervals1.shape)
