import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
import os
import tensorflow as tf

# Check if GPU is available and set memory growth
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(e)

# ---------------------------
# 1) User-defined parameters
# ---------------------------
file_path = r"/Users/ru/FFT/Fe56_MT_102_0K_to_4000K_Delta20K.csv"

# Energy range selection
E_min = 500  # eV
E_max = 6000  # eV

# Upsampling factor
upsample_factor = 3  # e.g., was 10 before, now we do 3 for demonstration

# Sliding window parameters
window_size = 1  # in energy units
step_size = 0.1   # in energy units

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
unique_T_values = [t for t in range(300, 4000, 20)]
print(f"Unique T values: {unique_T_values}")

# Optional: Create an output directory for plots/HDF5
# os.makedirs("output_spectrograms", exist_ok=True)

# Convert energy units if necessary
data["ERG"] = data["ERG"] * 1e6  # Adjust based on your data's units

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
    print(f"Currently processing T={T_val}...")
    print(f" subset_T has {len(subset_T)} rows for T={T_val}.")

    mask = (subset_T["ERG"] >= E_min) & (subset_T["ERG"] <= E_max)
    subset_range = subset_T[mask].copy()

    if len(subset_range) < 2:
        print(f" [Warning] Insufficient data for T={T_val} in range [{E_min}, {E_max}]. Skipping.")
        continue

    # Extract the columns as NumPy arrays
    E = subset_range["ERG"].to_numpy()
    xs = subset_range["XS"].to_numpy()

    # Sort by ascending energy if not already
    sort_idx = np.argsort(E)
    E = E[sort_idx]
    xs = xs[sort_idx]

    E_selected_min = E.min()
    E_selected_max = E.max()

    # ---------------------------
    # 5) Generate Non-Uniformly Sampled Data
    #    ("time" = E in your context)
    # ---------------------------
    t_nonuniform = E
    signal_nonuniform = xs
    T_max = tf.constant(t_nonuniform.max(), dtype=tf.float32)

    # Convert to TensorFlow tensors
    t_nonuniform_tf = tf.constant(t_nonuniform, dtype=tf.float32)
    signal_nonuniform_tf = tf.constant(signal_nonuniform, dtype=tf.float32)

    # Base "sampling" is simply the # of points in this energy range
    sampling_size = len(t_nonuniform)
    fs_nonuniform = sampling_size

    # Upsample factor
    fs_uniform = sampling_size * upsample_factor

    # ---------------------------
    # 6) Define a higher-rate uniform grid & interpolate
    # ---------------------------
    if E_selected_max <= E_selected_min:
        print(f"  [Warning] E_selected_max <= E_selected_min for T={T_val}. Skipping.")
        continue

    # Define the uniform grid
    num_uniform_points = int(fs_uniform * E_selected_max)
    t_uniform_np = np.linspace(
        E_selected_min,
        E_selected_max,
        num_uniform_points,
        endpoint=False
    )
    t_uniform_tf = tf.constant(t_uniform_np, dtype=tf.float32)

    # Implement linear interpolation manually using TensorFlow
    # Find indices where t_uniform falls between t_nonuniform
    indices = tf.searchsorted(t_nonuniform_tf, t_uniform_tf, side='left')
    indices = tf.clip_by_value(indices, 1, len(t_nonuniform_tf)-1)

    # Get the two surrounding points for interpolation
    E_left = tf.gather(t_nonuniform_tf, indices - 1)
    E_right = tf.gather(t_nonuniform_tf, indices)
    xs_left = tf.gather(signal_nonuniform_tf, indices - 1)
    xs_right = tf.gather(signal_nonuniform_tf, indices)

    # Compute the weights
    weight_right = (t_uniform_tf - E_left) / (E_right - E_left)
    weight_left = 1.0 - weight_right

    # Perform linear interpolation
    signal_uniform_tf = xs_left * weight_left + xs_right * weight_right

    # Move tensors to GPU
    with tf.device('/GPU:0'):
        signal_uniform = signal_uniform_tf  # This tensor is on GPU

    # ---------------------------
    # 7) Prepare Sliding Window FFT (Spectrogram)
    # ---------------------------
    fs = fs_uniform
    window_samps = int(window_size * fs)
    
    # Ensure window_samps is even to avoid shape mismatches
    if window_samps % 2 != 0:
        window_samps -= 1
        print(f"Adjusted window_samps to be even: {window_samps}")

    step_samps = int(step_size * fs)

    # Pad the signal for overlap-add
    pad = window_samps // 2
    padded_sig = tf.pad(signal_uniform, [[pad, pad]], mode='CONSTANT', constant_values=0.0)

    # "padded_t" covers from -pad/fs to T_max + pad/fs
    padded_t_np = np.linspace(
        -pad/fs,
        E_selected_max + pad/fs,
        num_uniform_points + 2 * pad,
        endpoint=False
    )
    padded_t = tf.constant(padded_t_np, dtype=tf.float32)

    # Frequencies for the rFFT
    frequencies_np = np.fft.rfftfreq(window_samps, d=1/fs)
    frequencies_tf = tf.constant(frequencies_np, dtype=tf.float32)

    # Create Hann window using TensorFlow
    hann_window_np = np.hanning(window_samps).astype(np.float32)
    hann_window_tf = tf.constant(hann_window_np, dtype=tf.float32)

    # Vectorized spectrogram computation
    starts = tf.range(0, len(padded_sig) - window_samps + 1, step_samps)
    num_windows = len(starts)

    # Extract windows using TensorFlow's extract_patches
    windows = tf.image.extract_patches(
        images=tf.reshape(padded_sig, [1, -1, 1, 1]),
        sizes=[1, window_samps, 1, 1],
        strides=[1, step_samps, 1, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    windows = tf.reshape(windows, [num_windows, window_samps])
    windows = windows * hann_window_tf  # Apply Hann window

    # Compute FFT using TensorFlow
    spectrogram = tf.signal.rfft(windows)  # Shape: (num_windows, freq_bins)
    # Do not transpose the spectrogram
    # spectrogram = tf.transpose(spectrogram)  # Removed

    # Debugging: Print shapes
    print(f"Spectrogram shape before iFFT: {spectrogram.shape}")
    print(f"Hann window shape: {hann_window_tf.shape}")

    # Centered time-bins for each window
    window_centers = starts + (window_samps // 2)
    time_bins = tf.gather(padded_t, window_centers)

    # Remap time_bins to match the uniform time axis
    tmin = time_bins[0]
    tmax = time_bins[-1]
    time_bins = (time_bins - tmin) / (tmax - tmin) * E_selected_max

    # ---------------------------
    # 8) Save Spectrogram to HDF5
    # ---------------------------
    # Convert TensorFlow tensors back to NumPy for saving
    spectrogram_np = spectrogram.numpy()
    frequencies_np = frequencies_tf.numpy()
    time_bins_np = time_bins.numpy()

    # e.g., each T in a separate file
    h5_filename = f"spectrogram_T_{T_val}.h5"
    with h5py.File(h5_filename, "w") as h5f:
        h5f.create_dataset("time_bins", data=time_bins_np)
        h5f.create_dataset("frequencies", data=frequencies_np)
        h5f.create_dataset("spectrogram_real", data=spectrogram_np.real)
        h5f.create_dataset("spectrogram_imag", data=spectrogram_np.imag)

    print(f"  [Info] Spectrogram saved: {h5_filename}")

    # ---------------------------
    # 9) Plot the Spectrogram using imshow
    # ---------------------------
    plt.figure(figsize=(12, 6))

    # Calculate the power in dB
    C = 10 * np.log10(np.abs(spectrogram_np.T) + 1e-12)  # Shape: (freq_bins, num_windows)

    # Plot using imshow
    plt.imshow(
        C,
        aspect='auto',
        origin='lower',
        extent=[
            time_bins_np[0],
            time_bins_np[-1],
            frequencies_np[0],
            frequencies_np[-1]
        ],
        cmap='viridis'
    )

    plt.colorbar(label='Power (dB)')
    plt.title(f"Spectrogram (T={T_val})\nE in [{E_min}, {E_max}]")
    plt.xlabel('Energy (eV) [Log Scale]')
    plt.ylabel('Frequency (Hz)')
    plt.xscale("log")
    plt.tight_layout()

    spectrogram_plot = f"spectrogram_T_{T_val}.png"
    plt.savefig(spectrogram_plot, dpi=150)
    plt.close()
    print(f"  [Info] Spectrogram plot saved: {spectrogram_plot}")

    # ---------------------------
    # 10) Reconstruct the Signal (Inverse Windowed-FFT)
    # ---------------------------
    try:
        # Perform inverse FFT using TensorFlow
        reconstructed_windows = tf.signal.irfft(spectrogram) * hann_window_tf  # Shape: (num_windows, window_samps)
    except Exception as e:
        print(f"  [Error] Inverse FFT failed for T={T_val}: {e}")
        continue

    # Initialize reconstructed signal and overlap factor using NumPy
    reconstructed_signal_np = np.zeros(len(padded_sig), dtype=np.float32)
    overlap_factor_np = np.zeros(len(padded_sig), dtype=np.float32)

    # Convert TensorFlow tensors to NumPy for reconstruction
    spectrogram_np = spectrogram.numpy()  # Shape: (num_windows, freq_bins)
    hann_window_np = hann_window_np  # Already defined

    for idx in range(num_windows):
        s = starts[idx].numpy()
        window_fft = spectrogram_np[idx]
        rec_window = np.fft.irfft(window_fft) * hann_window_np
        reconstructed_signal_np[s : s + window_samps] += rec_window
        overlap_factor_np[s : s + window_samps] += hann_window_np

    # Normalize by overlap factor & Hann window RMS
    window_rms = np.sqrt(0.5)
    reconstructed_signal_np /= np.maximum(overlap_factor_np, 1e-12)
    reconstructed_signal_np /= window_rms

    # Trim padding
    reconstructed_signal_np = reconstructed_signal_np[pad : -pad]

    # Final amplitude correction
    signal_uniform_np = signal_uniform.numpy()
    max_orig = np.max(signal_uniform_np) if len(signal_uniform_np) > 0 else 1e-12
    max_reco = np.max(reconstructed_signal_np) if len(reconstructed_signal_np) > 0 else 1e-12
    amp_corr = max_orig / max_reco if max_reco != 0 else 1.0
    reconstructed_signal_np *= amp_corr

    # ---------------------------
    # 11) Plot Original vs Reconstructed
    # ---------------------------
    plt.figure(figsize=(12, 6))
    plt.plot(t_uniform_np, signal_uniform_np, label='Original Signal', alpha=0.8)
    plt.plot(t_uniform_np, reconstructed_signal_np, label='Reconstructed Signal', linestyle='--', alpha=0.8)
    plt.legend()
    plt.xlabel('Energy (eV)')
    plt.ylabel('Cross Section')
    plt.title(f"Original vs Reconstructed (T={T_val})\nE in [{E_min}, {E_max}]")
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()

    orig_recon_plot = f"orig_vs_recon_T_{T_val}.png"
    plt.savefig(orig_recon_plot, dpi=150)
    plt.close()
    print(f"  [Info] Original vs Reconstructed plot saved: {orig_recon_plot}")

    # ---------------------------
    # 12) Calculate & Plot Relative Errors
    # ---------------------------
    signal_uniform_safe = np.where(np.abs(signal_uniform_np) < 1e-12, 1e-12, signal_uniform_np)
    relative_error = np.abs(signal_uniform_np - reconstructed_signal_np) / np.abs(signal_uniform_safe)

    plt.figure(figsize=(12, 6))
    plt.plot(t_uniform_np, relative_error, label='Relative Error', color='red', alpha=0.8)
    plt.legend()
    plt.xlabel('Energy (eV)')
    plt.ylabel('Relative Error')
    plt.title(f"Relative Error (T={T_val})\nE in [{E_min}, {E_max}]")
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()

    error_plot = f"relative_error_T_{T_val}.png"
    plt.savefig(error_plot, dpi=150)
    plt.close()
    print(f"  [Info] Relative error plot saved: {error_plot}")

print("\nAll T values processed!")
