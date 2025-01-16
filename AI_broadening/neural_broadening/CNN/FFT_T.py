import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.interpolate import interp1d
from scipy.signal.windows import hann
import json
import h5py
import os

# ---------------------------
# 1) User-defined parameters
# ---------------------------
# file_path    = r"/Users/ru/FFT/Fe-56_MT_102_0K_cross_sections.csv"
file_path    = r"/Users/ru/FFT/Fe56_MT_102_0K_to_4000K_Delta20K.csv"

# Energy range selection
E_min = 500 #* 10**(-6)
E_max = 6000 #* 10**(-6)

# Upsampling factor
upsample_factor = 3  # e.g. was 10 before, now we do 3 for demonstration

# Sliding window parameters
window_size = 1
step_size   = 0.1

# ---------------------------
# 2) Read the CSV
# ---------------------------
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found at: {file_path}. Check path and try again.")
    raise SystemExit(1)
except Exception as e:
    print(f"An error occurred while reading the CSV: {e}")
    raise SystemExit(1)

# Ensure required columns exist
required_cols = {'ERG', 'XS', 'T'}
if not required_cols.issubset(data.columns):
    raise ValueError(f"CSV must contain columns {required_cols}.")

# ---------------------------
# 3) Get unique T values
# ---------------------------
# unique_T_values = data["T"].unique()
# unique_T_values = [80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
unique_T_values = [t for t in range(300, 4000, 20)]
print(f"Unique T values: {unique_T_values}")

# Create an output directory for plots/HDF5 if desired
# (Optional) e.g.:
# os.makedirs("output_spectrograms", exist_ok=True)

data["ERG"] = data["ERG"] * 1e6

# ---------------------------
# 4) Loop over each T value
# ---------------------------
for T_val in unique_T_values:
    print(f"\nProcessing T = {T_val} ...")

    # (a) Filter the DataFrame by the current T
    subset_T = data[data["T"] == T_val].copy()

    # (b) Also filter by energy range [E_min, E_max]
    mask = (subset_T["ERG"] >= E_min) & (subset_T["ERG"] <= E_max)
    subset_range = subset_T[mask].copy()

    # If we don't have enough points in that subset, skip
    # For debugging:
    print(f"Currently processing T={T_val}...")

    subset_T = data[data["T"] == T_val].copy()
    print(f" subset_T has {len(subset_T)} rows for T={T_val}.")
    # if len(subset_T) > 0:
    #     print(" subset_T energies:", subset_T["ERG"].values)
    #     print(" T dtype:", subset_T["T"].dtype)

    mask = (subset_T["ERG"] >= E_min) & (subset_T["ERG"] <= E_max)
    # print(f" mask sum: {mask.sum()} (should match the # of rows in range).")

    subset_range = subset_T[mask].copy()
    # print(f" subset_range has {len(subset_range)} rows within [{E_min}, {E_max}].")
    # if len(subset_range) > 0:
    #     print(" subset_range energies:", subset_range["ERG"].values)
    #     print(" min(E) =", subset_range["ERG"].min(), "max(E) =", subset_range["ERG"].max())

    if len(subset_range) < 2:
        print(f" [Warning] Insufficient data for T={T_val} in range [{E_min}, {E_max}]. Skipping.")
        continue


    # Extract the columns as NumPy arrays
    E = subset_range["ERG"].to_numpy()
    xs = subset_range["XS"].to_numpy()

    # Sort by ascending energy if not already
    sort_idx = np.argsort(E)
    E  = E[sort_idx]
    xs = xs[sort_idx]

    E_selected_min = E.min()
    E_selected_max = E.max()

    # ---------------------------
    # 5) Generate Non-Uniformly Sampled Data
    #    ("time" = E in your context)
    # ---------------------------
    t_nonuniform      = E
    signal_nonuniform = xs
    T_max             = t_nonuniform.max()

    # Base "sampling" is simply the # of points in this energy range
    sampling_size = len(t_nonuniform)
    fs_nonuniform = sampling_size

    # Upsample factor
    fs_uniform = sampling_size * upsample_factor

    # ---------------------------
    # 6) Define a higher-rate uniform grid & interpolate
    # ---------------------------
    # The # of points is int(fs_uniform * E_selected_max).
    # If E_selected_max is large, you may want a different approach
    # (like a fixed number of points instead).
    if E_selected_max <= E_selected_min:
        print(f"  [Warning] E_selected_max <= E_selected_min for T={T_val}. Skipping.")
        continue

    t_uniform = np.linspace(
        E_selected_min,
        E_selected_max,
        int(fs_uniform * E_selected_max),
        endpoint=False
    )

    # Linear interpolation
    interp_func = interp1d(
        t_nonuniform,
        signal_nonuniform,
        kind="linear",
        fill_value="extrapolate"
    )
    signal_uniform = interp_func(t_uniform)

    # ---------------------------
    # 7) Prepare Sliding Window FFT (Spectrogram)
    # ---------------------------
    fs = fs_uniform
    window_samps = int(window_size * fs)
    step_samps   = int(step_size   * fs)

    # Pad the signal for overlap-add
    pad        = window_samps // 2
    padded_sig = np.pad(
        signal_uniform,
        (pad, pad),
        mode='constant',
        constant_values=0
    )

    # "padded_t" covers from -pad/fs to T_max + pad/fs
    padded_t = np.linspace(
        -pad/fs,
        T_max + pad/fs,
        len(padded_sig),
        endpoint=False
    )

    # Frequencies for the rFFT
    frequencies = np.fft.rfftfreq(window_samps, d=1/fs)
    hann_window = hann(window_samps)

    # Vectorized spectrogram computation
    starts      = range(0, len(padded_sig) - window_samps + 1, step_samps)
    windowed    = [padded_sig[s : s + window_samps] * hann_window for s in starts]
    spectrogram = np.array([np.fft.rfft(w) for w in windowed])  # => (time_bins, freq-bins)
    spectrogram = spectrogram.T                                  # => (freq-bins, time_bins)

    # Centered time-bins for each window
    time_bins = np.array([padded_t[s + window_samps // 2] for s in starts])
    tmin, tmax = time_bins[0], time_bins[-1]

    # Remap time_bins to match the uniform time axis
    time_bins = np.array([
        t_uniform[-1] * (tb - tmin)/(tmax - tmin) + t_uniform[0]
        for tb in time_bins
    ])

    # ---------------------------
    # 8) Save Spectrogram to HDF5
    # ---------------------------
    # e.g., each T in a separate file
    h5_filename = f"spectrogram_T_{T_val}.h5"
    with h5py.File(h5_filename, "w") as h5f:
        h5f.create_dataset("time_bins",      data=time_bins)
        h5f.create_dataset("frequencies",    data=frequencies)
        h5f.create_dataset("spectrogram_real", data=spectrogram.real)
        h5f.create_dataset("spectrogram_imag", data=spectrogram.imag)

    print(f"  [Info] Spectrogram saved: {h5_filename}")

    # ---------------------------
    # 9) Plot the Spectrogram
    # ---------------------------
    # plt.figure(figsize=(12, 6))
    # plt.pcolormesh(
    #     time_bins,
    #     frequencies,
    #     10 * np.log10(np.abs(spectrogram) + 1e-12),
    #     shading='auto',
    #     cmap='viridis'
    # )
    # plt.colorbar(label='Power (dB)')
    # plt.title(f"Spectrogram (T={T_val})\nE in [{E_min}, {E_max}]")
    # plt.xlabel('Energy (eV) [Log Scale]')
    # plt.ylabel('Frequency (Hz)')
    # plt.xscale("log")
    # plt.tight_layout()

    # spectrogram_plot = f"spectrogram_T_{T_val}.png"
    # plt.savefig(spectrogram_plot, dpi=150)
    # plt.close()
    # print(f"  [Info] Spectrogram plot saved: {spectrogram_plot}")

    # ---------------------------
    # 10) Reconstruct the Signal (Inverse Windowed-FFT)
    # ---------------------------
    reconstructed_signal = np.zeros_like(padded_sig, dtype=np.float64)
    overlap_factor       = np.zeros_like(padded_sig, dtype=np.float64)

    for idx, s in enumerate(starts):
        fft_result = spectrogram[:, idx]
        rec_window = np.fft.irfft(fft_result, n=window_samps) * hann_window
        reconstructed_signal[s : s + window_samps] += rec_window
        overlap_factor[s : s + window_samps]       += hann_window

    # Normalize by overlap factor & Hann window RMS
    window_rms           = np.sqrt(0.5)
    reconstructed_signal /= np.maximum(overlap_factor, 1e-12)
    reconstructed_signal /= window_rms

    # Trim padding
    reconstructed_signal = reconstructed_signal[pad : -pad]

    # Final amplitude correction
    max_orig = np.max(signal_uniform) if len(signal_uniform) > 0 else 1e-12
    max_reco = np.max(reconstructed_signal) if len(reconstructed_signal) > 0 else 1e-12
    amp_corr = max_orig / max_reco if max_reco != 0 else 1.0
    reconstructed_signal *= amp_corr

    # ---------------------------
    # 11) Plot Original vs Reconstructed
    # ---------------------------
    # plt.figure(figsize=(12, 6))
    # plt.plot(t_uniform, signal_uniform, label='Original Signal', alpha=0.8)
    # plt.plot(t_uniform, reconstructed_signal, label='Reconstructed Signal', linestyle='--', alpha=0.8)
    # plt.legend()
    # plt.xlabel('Energy (eV)')
    # plt.ylabel('Cross Section')
    # plt.title(f"Original vs Reconstructed (T={T_val})\nE in [{E_min}, {E_max}]")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.tight_layout()

    # orig_recon_plot = f"orig_vs_recon_T_{T_val}.png"
    # plt.savefig(orig_recon_plot, dpi=150)
    # plt.close()
    # print(f"  [Info] Original vs Reconstructed plot saved: {orig_recon_plot}")

    # ---------------------------
    # 12) Calculate & Plot Relative Errors
    # ---------------------------
    signal_uniform_safe = np.where(np.abs(signal_uniform) < 1e-12, 1e-12, signal_uniform)
    relative_error = np.abs(signal_uniform - reconstructed_signal) / np.abs(signal_uniform_safe)

    # plt.figure(figsize=(12, 6))
    # plt.plot(t_uniform, relative_error, label='Relative Error', color='red', alpha=0.8)
    # plt.legend()
    # plt.xlabel('Energy (eV)')
    # plt.ylabel('Relative Error')
    # plt.title(f"Relative Error (T={T_val})\nE in [{E_min}, {E_max}]")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.tight_layout()

    # error_plot = f"relative_error_T_{T_val}.png"
    # plt.savefig(error_plot, dpi=150)
    # plt.close()
    # print(f"  [Info] Relative error plot saved: {error_plot}")

print("\nAll T values processed!")
