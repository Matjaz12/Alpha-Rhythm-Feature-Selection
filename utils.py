import matplotlib.pyplot as plt
import numpy as np
import scipy

from read_signal import read_sig


def get_intervals(data, annot, fs):
    intervals_t0, intervals_t1, intervals_t2 = [], [], []

    for idx in range(len(annot)):
        dur = annot[idx]["duration"] * fs # unit: [smp]
        start = int(annot[idx]["onset"] * fs)
        end = int(start + dur)

        if annot[idx]["description"] == "T0":
            intervals_t0.append(data[:, start:end])

        elif annot[idx]["description"] == "T1":
            intervals_t1.append(data[:, start:end])

        elif annot[idx]["description"] == "T2":
            intervals_t2.append(data[:, start:end])

    intervals_t0 = np.array(intervals_t0) 
    intervals_t1 = np.array(intervals_t1)
    intervals_t2 = np.array(intervals_t2)
    
    return intervals_t0, intervals_t1, intervals_t2

if __name__ == "__main__":
    SIG_PATH = "./data/S001R03.edf"
    data, ch_names, annot, times, fs = read_sig(sig_path=SIG_PATH)
    intervals_t0, intervals_t1, intervals_t2 = get_intervals(data, annot, fs)
    print(intervals_t0.shape)
    print(intervals_t1.shape)
    print(intervals_t2.shape)