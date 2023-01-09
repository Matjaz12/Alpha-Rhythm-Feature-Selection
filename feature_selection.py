import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy

from read_signal import read_sig
from utils import get_intervals


def compute_R2(mean_psd1, mean_psd2, psd1, psd2):
    # Compute number of examples for each activity
    N1 = psd1.shape[0]
    N2 = psd2.shape[0]
    N = min(N1, N2) # assume this for now !
    
    n_samples1, n_samples2 = len(mean_psd1), len(mean_psd2) 
    n_samples = min(n_samples1, n_samples2)
    r2 = np.zeros(n_samples)
    
    for idx in range(n_samples):
        s1 = np.sum(psd1[:, idx])
        s2 = np.sum(psd2[:, idx])
        g1 = np.sum(psd1[:, idx] ** 2)
        g2 = np.sum(psd2[:, idx] ** 2)
        # eq. assumes that Var(annot) = 1.
        r2[idx] = ((s1 - s2) ** 2) / (2 * N * (g1 + g2) - (s1 + s2) ** 2 + 1e-8)

    return r2


def compute_R2_map(intervals_t1, intervals_t2, fs):
    n_channels, n_samples = intervals_t1.shape[1:]
    print(n_channels, n_samples)

    R2 = []

    # Iterate over all channels    
    for channel_idx in range(n_channels):        
        # Compute power spectrum for each signal
        psd1, freq1 = compute_power_spectrum(intervals_t1[:, channel_idx, :], fs)
        psd2, freq2 = compute_power_spectrum(intervals_t2[:, channel_idx, :], fs)

        # Compute the mean power spectrums
        mean_psd1 = np.mean(psd1, axis=0) 
        mean_psd2 = np.mean(psd2, axis=0) 
        # plt.plot(mean_psd1, label="mean power spectrum 1")
        # plt.plot(mean_psd2, label="mean power spectrum 2")

        # Compute r2 for current channel_idx
        r2 = compute_R2(mean_psd1, mean_psd2, psd1, psd2)
        # plt.plot(r2, label="r2")
        # plt.show()
        # break

        R2.append(r2)

    R2 = np.array(R2)
    return R2


def compute_power_spectrum(sig, fs):
    #  output already in range 0 to fs / 2
    psd, freq = mne.time_frequency.psd_array_multitaper(sig, sfreq=fs, verbose=False)
    return psd, freq



def plot_R2_map(R2, max_freq, fs):
    if max_freq > (fs / 2):
       max_freq = fs / 2

    plt.figure(figsize=(10, 10))
    plt.title("R2 as a function of channel number and frequency")
    plt.ylabel("Channel number")
    plt.xlabel("Frequency [Hz]")
    max_samples = int(max_freq / (fs / 2) * R2.shape[1])
    R2_cropped = R2[:, 0:max_samples]
    print(R2_cropped.shape)
    plt.imshow(R2_cropped)
    k = fs / (2 * R2.shape[1])
    STEP = 10
    freqs = np.array([round(k * x, 2) for x in range(0, R2_cropped.shape[1], STEP)])
    plt.xticks(list(range(0, R2_cropped.shape[1], STEP)), freqs)
    plt.show()



if __name__ == "__main__":
    SIG_PATH = "./data/S001R03.edf"
    data, ch_names, annot, times, fs = read_sig(sig_path=SIG_PATH)
    intervals_t0, intervals_t1, intervals_t2 = get_intervals(data, annot, fs)
    print(f"intervals_t1.shape: {intervals_t1.shape}")
    print(f"intervals_t2.shape: {intervals_t2.shape}")
    R2 = compute_R2_map(intervals_t0, intervals_t1, fs=fs)
    print(f"R2.min(): {R2.min()}, R2.max(): {R2.max()}")
    plot_R2_map(R2, max_freq=20, fs=fs)