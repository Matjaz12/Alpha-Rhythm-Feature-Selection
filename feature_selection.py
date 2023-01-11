import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy

def compute_R2_signal(mean_psd1, mean_psd2, psd1, psd2, eps=1e-10):
    if mean_psd1.shape != mean_psd2.shape:
        raise Exception("shape of mean_psd1 and mean_psd2 is not the same")

    # Compute number of examples for each activity
    N = psd1.shape[0]
    n_samples = len(mean_psd1)
    r2 = np.zeros(n_samples)
    
    for idx in range(n_samples):
        s1 = np.sum(psd1[:, idx])
        s2 = np.sum(psd2[:, idx])
        g1 = np.sum(psd1[:, idx] ** 2)
        g2 = np.sum(psd2[:, idx] ** 2)
        r2[idx] = ((s1 - s2) ** 2) / (2 * N * (g1 + g2) - (s1 + s2) ** 2 + eps)

    return r2


def compute_R2_map(intervals1, intevals2, fs):
    n_channels, n_samples = intervals1.shape[1:]
    R2 = []

    # Iterate over all channels    
    for channel_idx in range(n_channels):        
        # Compute power spectrum for each signal
        psd1, _ = compute_power_spectrum(intervals1[:, channel_idx, :], fs)
        psd2, _ = compute_power_spectrum(intevals2[:, channel_idx, :], fs)

        # Compute the mean power spectrums
        mean_psd1 = np.mean(psd1, axis=0)
        mean_psd2 = np.mean(psd2, axis=0)

        # Compute r2 for current channel_idx
        r2 = compute_R2_signal(mean_psd1, mean_psd2, psd1, psd2)
        if channel_idx == 60: 
            plot_psd_and_r2(mean_psd1, mean_psd2, r2, fs, channel_idx)

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

    plt.figure(figsize=(10, 8))
    plt.title("R2 as a function of channel number and frequency")
    plt.ylabel("Channel number")
    plt.xlabel("Frequency [Hz]")
    max_samples = int(max_freq / (fs / 2) * R2.shape[1])
    R2_cropped = R2[:, 0:max_samples]
    # print(R2_cropped.shape)
    plt.imshow(R2_cropped, interpolation="nearest")
    k = fs / (2 * R2.shape[1])
    STEP = 20
    freqs = np.array([round(k * x, 2) for x in range(0, R2_cropped.shape[1], STEP)])
    plt.xticks(list(range(0, R2_cropped.shape[1], STEP)), freqs)
    plt.yticks(list(range(0, R2_cropped.shape[0], 5)))
    plt.tight_layout()
    plt.savefig("./results/r2_map.pdf")
    plt.show()


def plot_psd_and_r2(mean_psd1, mean_psd2, r2, fs, channel_idx):
    STEP = 20
    k = fs / (2 * len(mean_psd1))
    freqs = np.array([round(k * x, 1) for x in range(0, len(mean_psd1), STEP)])
    plt.figure(1, figsize=(10, 8))
    plt.subplot(211)
    plt.title(f"Mean power spectrums and corresponing R2 signal for channel {channel_idx}")
    plt.plot(mean_psd1, label="Mean power spectrum 1")
    plt.plot(mean_psd2, label="Mean power spectrum 2")
    plt.xticks(list(range(0, len(mean_psd1), STEP)), freqs)
    plt.xlabel("Frequency [Hz]")
    plt.legend()
    plt.subplot(212)
    plt.plot(r2, label="R2")
    plt.xticks(list(range(0, len(mean_psd1), STEP)), freqs)
    plt.xlabel("Frequency [Hz]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./results/r2_signal_{channel_idx}.pdf")
