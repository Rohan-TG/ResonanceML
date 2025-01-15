import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Specify the file path
from scipy.signal import stft
from scipy.interpolate import interp1d
from scipy.signal.windows import hann


file_path = r"/Users/ru/FFT/Fe-56_MT_102_0K_cross_sections.csv"

end_index     = 1000
start_index   = 200
sampling_size = 1000
window_size   = 1
step_size     = 0.1

try:
    # Load the CSV file into a pandas DataFrame
    data = pd.read_csv(file_path)
    
    # Display the first few rows of the DataFrame
    print("Data preview:")
    print(data.head())

    # Convert specific columns to numpy arrays (update column names as needed)
    if 'ERG' in data.columns and 'XS' in data.columns:
        E = np.array(data['ERG'])
        xs = np.array(data['XS'])
        print("\nEnergy (E):", E)
        print("\nCross Section (xs):", xs)
    else:
        print("\nThe required columns 'Energy (eV)' and 'Cross Section (barns)' were not found.")
except FileNotFoundError:
    print(f"File not found at: {file_path}. Please check the file path and try again.")
except Exception as e:
    print(f"An error occurred: {e}")


plt.figure()
plt.plot( E[start_index:end_index], xs[start_index:end_index])
plt.xscale("log")
plt.yscale("log")
plt.show()


# 1) Generate Non-Uniformly Sampled Data
fs_nonuniform = sampling_size  # Approximate sampling frequency
               # Total duration in seconds
t_nonuniform = E[start_index:end_index]
T = max(t_nonuniform)

# Generate the non-uniform signal
#signal_nonuniform = amp1 * np.sin(2 * np.pi * freq1 * t_nonuniform) + amp2 * np.sin(2 * np.pi * freq2 * t_nonuniform)
signal_nonuniform = xs[start_index:end_index]

# 2) Define Uniform Time Grid
fs_uniform = sampling_size * 5  # Desired uniform sampling frequency
t_uniform = np.linspace(E[start_index], T, int(fs_uniform * T), endpoint=False)

# 3) Interpolate Signal
# Create interpolation function
interpolation_function = interp1d(t_nonuniform, signal_nonuniform, kind='linear', fill_value="extrapolate") #cubic

# Interpolate to uniform time grid
signal_uniform = interpolation_function(t_uniform)

#plt.figure(figsize=(12, 6))
#
#
## Plot non-uniform signal
#plt.scatter(t_nonuniform, signal_nonuniform, label='Non-Uniform Signal', color='blue', s=10, alpha=0.7)
#
## Plot uniform signal
#plt.plot(t_uniform, signal_uniform, label='Uniformly Reconstructed Signal', color='orange', linestyle='--')
#
#plt.title('Non-Uniform to Uniform Signal Reconstruction')
#plt.xlabel('Time (s)')
#plt.ylabel('Amplitude')
#plt.legend()
#plt.tight_layout()
#plt.show()

# plt.figure(figsize=(12, 6))
# plt.plot(t_uniform, signal_uniform, label='Reconstructed Uniform Signal', alpha=0.8, linestyle='--')
# plt.scatter(t_nonuniform, signal_nonuniform, label='Original Non-Uniform Signal', color='blue', s=10, alpha=0.7)
# plt.legend()
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Comparison of Non-Uniform and Reconstructed Uniform Signals')
# plt.tight_layout()
# plt.yscale("log")
# plt.xscale("log")
# plt.show()

# 1) Generate Uniformly Sampled Data
fs = sampling_size * 5 # 5000
T = 1.0    # Total duration
t = t_uniform  # Time array

signal = signal_uniform
window_samples = int(window_size * fs)
step_samples = int(step_size * fs)
padding = window_samples // 2
padded_signal = np.pad(signal, (padding, padding), mode='constant', constant_values=0)
padded_t = np.linspace(-padding / fs, T + padding / fs, len(padded_signal), endpoint=False)

# 2) Sliding-Window FFT and Spectrogram
frequencies = np.fft.rfftfreq(window_samples, d=1/fs)
hann_window = hann(window_samples)

spectrogram = []
time_bins = []

for start in range(0, len(padded_signal) - window_samples + 1, step_samples):
    window_signal = padded_signal[start:start + window_samples] * hann_window
    fft_result = np.fft.rfft(window_signal)  # FFT on the window
    spectrogram.append(fft_result)
    time_bins.append(padded_t[start + window_samples // 2])

spectrogram = np.array(spectrogram).T  # Shape (frequencies, time bins)

time_bins = np.array(time_bins)
tmin = time_bins[0]
tmax = time_bins[-1]
time_bins = np.array([t_uniform[-1]*(i-tmin)/(tmax-tmin) + t_uniform[0] for i in time_bins])
# Plot the spectrogram
plt.figure(figsize=(12, 6))
plt.pcolormesh(
    time_bins, frequencies, 10 * np.log10(np.abs(spectrogram) + 1e-12),
    shading='auto', cmap='viridis'
)
plt.colorbar(label='Power (dB)')
plt.title('Time-Frequency Heatmap (Spectrogram)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()
plt.xscale("log")
plt.show()

# 3) Reconstruct the Signal with Padding
reconstructed_signal = np.zeros_like(padded_signal)
overlap_factor = np.zeros_like(padded_signal)

for idx, start in enumerate(range(0, len(padded_signal) - window_samples + 1, step_samples)):
    fft_result = spectrogram[:, idx]
    reconstructed_window = np.fft.irfft(fft_result, n=window_samples) * hann_window
    reconstructed_signal[start:start + window_samples] += reconstructed_window
    overlap_factor[start:start + window_samples] += hann_window

# Normalize by overlap factor and Hann window RMS
window_gain = np.sqrt(0.5)  # Theoretical RMS of the Hann window
reconstructed_signal /= np.maximum(overlap_factor, 1)  # Normalize by overlap
reconstructed_signal /= window_gain  # Normalize by Hann window gain

# Trim padding
reconstructed_signal = reconstructed_signal[padding:-padding]
overlap_factor = overlap_factor[padding:-padding]

# Apply final amplitude correction
amplitude_correction = np.max(signal) / np.max(reconstructed_signal)
reconstructed_signal *= amplitude_correction

# 4) Plot Original vs Reconstructed Signal
plt.figure(figsize=(12, 6))
plt.plot(t, signal, label='Original Signal', alpha=0.8)
plt.plot(t, reconstructed_signal, label='Reconstructed Signal', linestyle='--', alpha=0.8)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original vs Reconstructed Signal')
plt.tight_layout()
plt.yscale("log")
plt.xscale("log")
plt.show()

