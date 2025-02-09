import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from datetime import datetime
import noisereduce as nr
import soundfile as sf
from scipy.io import wavfile
from scipy.io.wavfile import write
import scipy.signal as signal
import scipy.io.wavfile as wav
from IPython.display import Audio
import pickle
from scipy.signal import find_peaks
from scipy.signal import butter
from scipy import signal  # Correct way to import signal processing functions



# reduce noise with relate to the first sec
def reduce_noise(y,sr):
    noise_section = y[:sr]  # First 1 second

    # Reduce noise
    y_reduced_noise = nr.reduce_noise(y=y, sr=sr, y_noise=noise_section)
    
    return y_reduced_noise

# high pass
def high_pass_filter(y, sr=48000, cutoff=2048, order=5):
    """
    Apply a high-pass Butterworth filter to remove frequencies below a given cutoff.

    Parameters:
    - y: Input audio signal (numpy array)
    - sr: Sampling rate (Hz)
    - cutoff: Cutoff frequency (Hz) - Default is 2048 Hz
    - order: Filter order - Default is 5 (higher order gives a sharper cutoff)

    Returns:
    - Filtered signal (numpy array)
    """
    nyquist = 0.5 * sr  # Nyquist frequency (half of sampling rate)
    normal_cutoff = cutoff / nyquist  # Normalize cutoff frequency (0 to 1 range)
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    filtered_y = signal.filtfilt(b, a, y)  # Apply the filter forward & backward
    return filtered_y

# Matche filter
def matched_filter(y, template, threshold_factor=3.0):

    y = np.asarray(y)
    template = np.asarray(template)

    N, M = len(y), len(template)

    L = N + M - 1  

    signal_fft = np.fft.fft(y, L)  # Implicit zero-padding
    template_fft = np.fft.fft(template, L)
    result = np.fft.ifft(signal_fft * np.conjugate(template_fft))

    energy = np.sum(np.abs(template)**2)
    if energy:
        result /= energy

    mag = np.abs(result)
    noise_std = np.std(mag)  # noise estimate
    threshold = threshold_factor * noise_std

    return result, mag, threshold


start_time = time.time()

print("loading files")
template = np.loadtxt("bit_sound.txt", delimiter=",")  # Adjust delimiter as needed

file_paths = [


    r"C:\Users\dror.e\Documents\wav_files\241105-T034.WAV",
    r"C:\Users\dror.e\Documents\wav_files\241105-T035.WAV",
    r"C:\Users\dror.e\Documents\wav_files\241105-T0607.WAV",
    r"C:\Users\dror.e\Documents\wav_files\241105-T3435.WAV"]

# Iterate through each file path
for path_wav in file_paths:
    start_time = time.time()  # Start timing
    print(f"Processing: {path_wav}")

    print("✅done - loading files")

    print("loading sr and signal")
    sample_rate, data = wavfile.read(path_wav)
    
    # Ensure the data has at least two channels before indexing
    if len(data.shape) > 1:
        y = data[:, 0]  # Take first channel
    else:
        y = data  # If mono, use directly

    print("✅done - loading sr and signal")
    print("channel 1 is ready")

    print("saving file name")
    file_name = os.path.basename(path_wav)
    name_without_extension, _ = os.path.splitext(file_name)
    print("✅done - saving file name")

    print("reducing noise")
    template_cleaned = reduce_noise(template, sample_rate)
    signal_cleaned = reduce_noise(y, sample_rate)
    print("✅done - reducing noise")

    print("high pass")
    template_high_pass = high_pass_filter(template_cleaned)
    signal_high_pass = high_pass_filter(signal_cleaned)
    print("✅done - high pass")

    np.savez(f"{name_without_extension}_clean_signal.npz", arr1=signal_high_pass)
    print("✅saved reduced signal")

    result, mag, threshold = matched_filter(signal_high_pass, template_high_pass)
    np.savez(f"{name_without_extension}_matched_filter_results.npz", arr1=result, arr2=mag, var=threshold)
    print("✅saved matched filter")

    end_time = time.time()  # End timing
    print(f"Time taken for {file_name}: {end_time - start_time:.4f} seconds\n")


