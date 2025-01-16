import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Specify the file path
from scipy.signal import stft
from scipy.interpolate import interp1d
from scipy.signal.windows import hann
import json
import h5py

# --- User-defined parameters ---
file_path    = r"/Users/ru/FFT/Fe-56_MT_102_0K_cross_sections.csv"
end_index     = 1000
start_index   = 250
sampling_size = end_index - start_index    # base sampling freq
window_size   = 1
step_size     = 0.1

# --- 1) Read the CSV and extract columns ---
try:
    data = pd.read_csv(file_path)
    # print("Data preview:")
    print(data.head())

    if 'ERG' in data.columns and 'XS' in data.columns:
        E  = data['ERG'].to_numpy()
        xs = data['XS'].to_numpy()
        # print("\nEnergy (E):", E)
        # print("\nCross Section (xs):", xs)
    else:
        print("\nRequired columns 'ERG' and 'XS' not found.")
except FileNotFoundError:
    print(f"File not found at: {file_path}. Please check the file path and try again.")
except Exception as e:
    print(f"An error occurred: {e}")

# # --- 2) Plot the initial non-uniform data snippet ---
# plt.figure()
# plt.plot(E[start_index:end_index], xs[start_index:end_index])
# plt.xscale("log")
# plt.yscale("log")
# plt.title("Non-Uniform Data (Snippet)")
# plt.xlabel("Energy (eV) - Log Scale")
# plt.ylabel("Cross Section (barns) - Log Scale")
# plt.show()

# --- 3) Generate Non-Uniformly Sampled Data ---
t_nonuniform = E[start_index:end_index]
print(t_nonuniform)
signal_nonuniform = xs[start_index:end_index]
T = t_nonuniform.max()

# --- 4) Define a higher-rate Uniform Grid & Interpolate ---
fs_nonuniform = sampling_size              # original approx. sampling
fs_uniform    = sampling_size * 3          # desired uniform sampling
t_uniform     = np.linspace(E[start_index], T, int(fs_uniform * T), endpoint=False)

interp_func    = interp1d(t_nonuniform, signal_nonuniform, kind='linear', fill_value="extrapolate")
signal_uniform = interp_func(t_uniform)

# --- 5) Prepare Sliding Window FFT (Spectrogram) ---
fs           = fs_uniform                  # effectively the same as above
window_samps = int(window_size * fs)
step_samps   = int(step_size   * fs)

# Pad the signal
pad          = window_samps // 2
padded_sig   = np.pad(signal_uniform, (pad, pad), mode='constant', constant_values=0)
padded_t     = np.linspace(-pad/fs, T + pad/fs, len(padded_sig), endpoint=False)

# Frequencies for the rFFT
frequencies  = np.fft.rfftfreq(window_samps, d=1/fs)
hann_window  = hann(window_samps)

# -- Vectorized Spectrogram Computation --
starts       = range(0, len(padded_sig) - window_samps + 1, step_samps)
windowed     = [padded_sig[s : s + window_samps] * hann_window for s in starts]
spectrogram  = np.array([np.fft.rfft(w) for w in windowed])  # shape => (time_bins, frequencies)
spectrogram  = spectrogram.T                                  # shape => (frequencies, time_bins)

# Centered time-bins for each window
time_bins    = np.array([padded_t[s + window_samps // 2] for s in starts])
tmin, tmax   = time_bins[0], time_bins[-1]

# Remap the time_bins so they match the original uniform time scale
time_bins    = np.array([t_uniform[-1]*(tb - tmin)/(tmax - tmin) + t_uniform[0] for tb in time_bins])

# -- Save Spectrogram to HDF5 --

with h5py.File("spectrogram.h5", "w") as h5f:
    # Frequencies and time_bins are 1D arrays
    h5f.create_dataset("time_bins", data=time_bins)
    h5f.create_dataset("frequencies", data=frequencies)
    
    # Spectrogram is a 2D complex array, so we can store real/imag parts separately
    h5f.create_dataset("spectrogram_real", data=spectrogram.real)
    h5f.create_dataset("spectrogram_imag", data=spectrogram.imag)

print("Spectrogram saved to spectrogram.h5")

# # -- Save Spectrogram to JSON --

# spectrogram_dict = {
#     "time_bins": time_bins.tolist(),
#     "frequencies": frequencies.tolist(),
#     # Convert the complex array into separate lists for real and imaginary parts.
#     "spectrogram_real": spectrogram.real.tolist(),
#     "spectrogram_imag": spectrogram.imag.tolist(),
# }

# with open("spectrogram.json", "w") as json_file:
#     json.dump(spectrogram_dict, json_file)

# print("Spectrogram saved to spectrogram.json")

# # --- 6) Plot the Spectrogram ---
plt.figure(figsize=(12, 6))
plt.pcolormesh(
    time_bins,
    frequencies,
    10 * np.log10(np.abs(spectrogram) + 1e-12),
    shading='auto',
    cmap='viridis'
)
plt.colorbar(label='Power (dB)')
plt.title('Time-Frequency Heatmap (Spectrogram)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.xscale("log")
plt.tight_layout()
plt.show()

# --- 7) Reconstruct the Signal (Inverse Windowed-FFT) ---
reconstructed_signal = np.zeros_like(padded_sig, dtype=np.float64)
overlap_factor       = np.zeros_like(padded_sig, dtype=np.float64)

for idx, s in enumerate(starts):
    fft_result           = spectrogram[:, idx]
    rec_window           = np.fft.irfft(fft_result, n=window_samps) * hann_window
    reconstructed_signal[s : s + window_samps] += rec_window
    overlap_factor[s : s + window_samps]       += hann_window

# Normalize by overlap factor and Hann window RMS
window_rms           = np.sqrt(0.5)  # RMS of Hann window
reconstructed_signal /= np.maximum(overlap_factor, 1)
reconstructed_signal /= window_rms

# Trim the padding
reconstructed_signal = reconstructed_signal[pad : -pad]

# Final amplitude correction
amp_corr             = np.max(signal_uniform) / np.max(reconstructed_signal)
reconstructed_signal *= amp_corr

# # --- 8) Plot Original vs Reconstructed Signal ---
plt.figure(figsize=(12, 6))
plt.plot(t_uniform, signal_uniform, label='Original Signal', alpha=0.8)
plt.plot(t_uniform, reconstructed_signal, label='Reconstructed Signal', linestyle='--', alpha=0.8)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original vs Reconstructed Signal')
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.show()

# Example calculation for relative errors
relative_error = np.abs(signal_uniform - reconstructed_signal) / np.abs(signal_uniform)

# --- Plot Relative Errors ---
plt.figure(figsize=(12, 6))
plt.plot(t_uniform, relative_error, label='Relative Error', color='red', alpha=0.8)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Relative Error')
plt.title('Relative Error: Original vs Reconstructed Signal')
plt.xscale("log")
plt.yscale("log")  # Optional, if errors span multiple orders of magnitude
plt.tight_layout()
plt.show()
