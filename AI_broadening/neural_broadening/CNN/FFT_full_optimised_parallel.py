import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import h5py

from scipy.interpolate import interp1d
from scipy.signal.windows import hann
from concurrent.futures import ThreadPoolExecutor

# ------------------------------------------------------------------
# 0) User parameters
# ------------------------------------------------------------------

file_path     = r"/Users/ru/FFT/Fe-56_MT_102_0K_cross_sections.csv"
end_index     = 1000
start_index   = 250
sampling_size = end_index - start_index    # base sampling freq
window_size   = 1.0     # seconds for each FFT window
step_size     = 0.1     # seconds step for sliding window
num_threads   = 4     # None => use default number of threads

# ------------------------------------------------------------------
# 1) Load CSV and extract columns
# ------------------------------------------------------------------
try:
    data = pd.read_csv(file_path)
    if 'ERG' in data.columns and 'XS' in data.columns:
        E  = data['ERG'].to_numpy()
        xs = data['XS'].to_numpy()
    else:
        raise ValueError("Required columns 'ERG' and 'XS' not found in CSV.")
except Exception as e:
    print(f"Error reading CSV: {e}")
    raise SystemExit

print(f"Loaded CSV with {len(E)} rows. Using rows {start_index} to {end_index} for demonstration.")

# ------------------------------------------------------------------
# 2) Quick plot of the non-uniform snippet
# ------------------------------------------------------------------
plt.figure()
plt.plot(E[start_index:end_index], xs[start_index:end_index])
plt.title("Non-Uniform Data (Snippet)")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Energy (log)")
plt.ylabel("Cross Section (log)")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 3) Non-uniform to uniform interpolation
# ------------------------------------------------------------------
t_nonuniform      = E[start_index:end_index]
signal_nonuniform = xs[start_index:end_index]
T                 = t_nonuniform.max()

fs_nonuniform = sampling_size               # approximate sampling freq
fs_uniform    = sampling_size * 3           # up-sample for uniform grid
t_uniform     = np.linspace(E[start_index], T, int(fs_uniform * T), endpoint=False)

interp_func    = interp1d(t_nonuniform, signal_nonuniform, kind='linear', fill_value="extrapolate")
signal_uniform = interp_func(t_uniform)

# ------------------------------------------------------------------
# 4) Prepare for spectrogram (windowed FFT)
# ------------------------------------------------------------------
fs           = fs_uniform
window_samps = int(window_size * fs)           # samples in each window
step_samps   = int(step_size   * fs)           # step between windows

# Pad the signal for sliding windows
pad          = window_samps // 2
padded_sig   = np.pad(signal_uniform, (pad, pad), mode='constant', constant_values=0)
padded_t     = np.linspace(-pad/fs, T + pad/fs, len(padded_sig), endpoint=False)

# Frequencies for rFFT
frequencies  = np.fft.rfftfreq(window_samps, d=1/fs)
hann_window  = hann(window_samps)

# Indices where each window starts
starts = list(range(0, len(padded_sig) - window_samps + 1, step_samps))

def compute_single_rfft(start_index):
    """Applies window + Hann, returns rFFT for the given start_index."""
    segment = padded_sig[start_index : start_index + window_samps] * hann_window
    return np.fft.rfft(segment)

# ------------------------------------------------------------------
# 5) PARALLEL Spectrogram computation using threads
# ------------------------------------------------------------------
# Depending on your system, you can set num_threads explicitly, e.g., num_threads=4
# If num_threads=None, it uses max available threads
print("Computing spectrogram in parallel (ThreadPoolExecutor)...")
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    # 'results' => list of rFFT results, each result is 1D array (complex)
    results = list(executor.map(compute_single_rfft, starts))

# Convert list of 1D FFTs => 2D array => (time_bins, frequencies)
spectrogram = np.array(results, dtype=np.complex128)
# Transpose => (frequencies, time_bins) if you prefer that shape
spectrogram = spectrogram.T

# Time bins at window centers
time_bins = np.array([padded_t[s + window_samps // 2] for s in starts])
tmin, tmax = time_bins[0], time_bins[-1]

# Remap time_bins to match the uniform scale range
time_bins = np.array([
    t_uniform[-1] * (tb - tmin) / (tmax - tmin) + t_uniform[0]
    for tb in time_bins
])

print(f"Spectrogram shape = {spectrogram.shape} => (frequencies, time_bins)")

# ------------------------------------------------------------------
# 6) Plot the spectrogram
# ------------------------------------------------------------------
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
plt.xlabel('Time (s) [log-scale]')
plt.ylabel('Frequency (Hz)')
plt.xscale("log")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 7) (Optional) Save spectrogram to HDF5 and JSON
# ------------------------------------------------------------------
with h5py.File("spectrogram.h5", "w") as h5f:
    h5f.create_dataset("time_bins", data=time_bins)
    h5f.create_dataset("frequencies", data=frequencies)
    h5f.create_dataset("spectrogram_real", data=spectrogram.real)
    h5f.create_dataset("spectrogram_imag", data=spectrogram.imag)

# spectrogram_dict = {
#     "time_bins": time_bins.tolist(),
#     "frequencies": frequencies.tolist(),
#     "spectrogram_real": spectrogram.real.tolist(),
#     "spectrogram_imag": spectrogram.imag.tolist(),
# }

# with open("spectrogram.json", "w") as json_file:
#     json.dump(spectrogram_dict, json_file)

print("Saved spectrogram to spectrogram.h5 and spectrogram.json")

# ------------------------------------------------------------------
# 8) PARALLEL Reconstruction
# ------------------------------------------------------------------
# Inverse rFFT + Overlap-Add
# We'll chunk the list of windows among threads. Each thread creates
# its own partial sum. Then we combine them.

def partial_reconstruction(start_indices, spec_slice, length):
    """
    Reconstructs a partial signal from a slice of the spectrogram
    (which is shape => (#windows_in_this_chunk, frequencies)).
    Returns (reconstructed_local, overlap_local).
    """
    recon_local   = np.zeros(length, dtype=np.float64)
    overlap_local = np.zeros(length, dtype=np.float64)

    for i, s in enumerate(start_indices):
        fft_result = spec_slice[i]
        rec_window = np.fft.irfft(fft_result, n=window_samps) * hann_window
        recon_local[s : s + window_samps]   += rec_window
        overlap_local[s : s + window_samps] += hann_window
    return recon_local, overlap_local

# Reshape spectrogram => (#time_bins, frequencies)
# so that spectrogram[i] => 1D array = one window's FFT
spectrogram_t = spectrogram.T  # shape => (#time_bins, frequencies)

length_padded = len(padded_sig)
num_chunks     = 4  # pick the number of chunks to distribute
chunk_size     = len(starts) // num_chunks if num_chunks > 0 else len(starts)

chunks     = [starts[i : i + chunk_size] for i in range(0, len(starts), chunk_size)]
spec_slices = [spectrogram_t[i : i + chunk_size] for i in range(0, len(starts), chunk_size)]

print("Reconstructing in parallel (ThreadPoolExecutor)...")

results_partial = []
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = []
    for chunk_starts, spec_slc in zip(chunks, spec_slices):
        futures.append(
            executor.submit(
                partial_reconstruction,
                chunk_starts,
                spec_slc,
                length_padded
            )
        )
    results_partial = [f.result() for f in futures]

# Now sum up the partial reconstructions
reconstructed_signal = np.zeros(length_padded, dtype=np.float64)
overlap_factor       = np.zeros(length_padded, dtype=np.float64)

for recon_piece, overlap_piece in results_partial:
    reconstructed_signal += recon_piece
    overlap_factor       += overlap_piece

# Normalization by overlap and Hann window RMS
window_rms = np.sqrt(0.5)  # RMS of Hann
reconstructed_signal /= np.maximum(overlap_factor, 1e-12)
reconstructed_signal /= window_rms

# Remove the padding
reconstructed_signal = reconstructed_signal[pad : -pad]

# Final amplitude correction
max_orig = np.max(signal_uniform) if np.max(signal_uniform) != 0 else 1e-12
max_recon = np.max(reconstructed_signal) if np.max(reconstructed_signal) != 0 else 1e-12
amp_corr = max_orig / max_recon
reconstructed_signal *= amp_corr

print("Parallel reconstruction completed.")

# ------------------------------------------------------------------
# 9) Plot Original vs Reconstructed
# ------------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(t_uniform, signal_uniform, label='Original Signal', alpha=0.8)
plt.plot(t_uniform, reconstructed_signal, label='Reconstructed Signal', linestyle='--', alpha=0.8)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original vs Reconstructed Signal (Parallel)')
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()
plt.show()

print("Done!")
