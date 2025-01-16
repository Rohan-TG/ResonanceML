import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import os
import sys

def read_spectrogram_h5(h5_filepath):
    """
    Reads spectrogram data from an HDF5 file.

    Parameters:
        h5_filepath (str): Path to the HDF5 file.

    Returns:
        tuple: (time_bins, frequencies, spectrogram_complex)
    """
    try:
        with h5py.File(h5_filepath, 'r') as h5f:
            # Define required datasets
            required_datasets = {'time_bins', 'frequencies', 'spectrogram_real', 'spectrogram_imag'}
            
            # Check for missing datasets
            missing = required_datasets - set(h5f.keys())
            if missing:
                raise KeyError(f"Missing datasets in HDF5 file: {missing}")
            
            # Read datasets
            time_bins = h5f['time_bins'][:]
            frequencies = h5f['frequencies'][:]
            spectrogram_real = h5f['spectrogram_real'][:]
            spectrogram_imag = h5f['spectrogram_imag'][:]
        
        # Reconstruct complex spectrogram
        spectrogram_complex = spectrogram_real + 1j * spectrogram_imag
        
        return time_bins, frequencies, spectrogram_complex

    except Exception as e:
        print(f"Error reading HDF5 file '{h5_filepath}': {e}")
        sys.exit(1)

def plot_spectrogram(time_bins, frequencies, spectrogram, title='Spectrogram', output_path=None):
    """
    Plots the spectrogram.

    Parameters:
        time_bins (np.ndarray): 1D array of time (or energy) bins.
        frequencies (np.ndarray): 1D array of frequency bins.
        spectrogram (np.ndarray): 2D complex array representing the spectrogram.
        title (str): Title of the plot.
        output_path (str): Path to save the plot image. If None, displays the plot.
    """
    # Configure Matplotlib to handle large plots
    mpl.rcParams['agg.path.chunksize'] = 2000  # Increase chunk size to prevent OverflowError
    mpl.rcParams['path.simplify_threshold'] = 0.2  # Increase simplify threshold

    plt.figure(figsize=(12, 6))
    
    # Convert spectrogram to power (dB)
    power_db = 10 * np.log10(np.abs(spectrogram) + 1e-12)  # Add small constant to avoid log(0)
    
    # Use imshow for efficient rendering of large spectrograms
    extent = [time_bins.min(), time_bins.max(), frequencies.min(), frequencies.max()]
    aspect = 'auto'
    
    plt.imshow(
        power_db,
        extent=extent,
        origin='lower',
        aspect=aspect,
        cmap='viridis'
    )
    
    plt.colorbar(label='Power (dB)')
    plt.title(title)
    plt.xlabel('Energy (scaled) [Log Scale]')
    plt.ylabel('Frequency (Hz)')
    
    # Apply logarithmic scales if appropriate
    plt.xscale('log')  # Comment out if linear scale is desired
    plt.yscale('log')  # Comment out if linear scale is desired
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Spectrogram plot saved to '{output_path}'")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='HDF5 Spectrogram Reader and Plotter')
    parser.add_argument('--file', '-f', type=str, required=True, help='Path to the HDF5 spectrogram file.')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output path to save the spectrogram plot. If not provided, the plot will be saved alongside the HDF5 file with a .png extension.')
    
    args = parser.parse_args()
    
    h5_filepath = args.file
    output_path = args.output
    
    if not os.path.isfile(h5_filepath):
        print(f"Error: File '{h5_filepath}' does not exist.")
        sys.exit(1)
    
    # Read spectrogram data from HDF5
    time_bins, frequencies, spectrogram = read_spectrogram_h5(h5_filepath)
    
    # Define plot title based on filename or other metadata
    base_filename = os.path.basename(h5_filepath)
    title = f"Spectrogram: {base_filename}"
    
    # Define default output path if not specified
    if output_path is None:
        # Save plot in the same directory as h5 file, with .png extension
        output_dir = os.path.dirname(h5_filepath)
        plot_filename = os.path.splitext(base_filename)[0] + '.png'
        output_path = os.path.join(output_dir, plot_filename)
    
    # Plot and save the spectrogram
    plot_spectrogram(time_bins, frequencies, spectrogram, title=title, output_path=output_path)

if __name__ == '__main__':
    main()
