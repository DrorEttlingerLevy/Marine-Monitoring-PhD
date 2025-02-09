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
import multiprocessing as mp

## Function 1 - Loading WAV with wavfile and csv['StartTime'] for time + creat channels list

def loading_file(audio_path, csv_path):
    # Path to the wav file
    print('Loading audio file')
    sr, y = wavfile.read(audio_path)  # Read the audio file
    print('Finish loading audio file')

    # Path to the csv file
    print('Loading csv file')
    csv_df = pd.read_csv(csv_path)
    csv_df['StartTime'] = pd.to_datetime(csv_df['StartTime'])
    print('Finish loading csv file')

    channel_list = [y[:,i] for i in range(y.shape[1])]
    
    return y, sr, csv_df, channel_list

## Function 2 - feching the real time from the csv file

def wav_real_time(path,csv_df):
    x_axis_time = []
    file_name = path[-15:]

    matching_row = csv_df[csv_df['FileName']==file_name]
    start_time = matching_row['StartTime'].dt.strftime("%Y-%m-%d %H:%M:%S").iloc[0]
    end_matching_row = csv_df.iloc[matching_row.index+1]
    end_time = end_matching_row['StartTime'].dt.strftime("%Y-%m-%d %H:%M:%S").iloc[0]
    start_time = pd.to_datetime(start_time, format="%Y-%m-%d %H:%M:%S")
    end_time = pd.to_datetime(end_time, format="%Y-%m-%d %H:%M:%S")

    x_axis_time.extend([start_time,end_time])
    return x_axis_time[0],x_axis_time[1]

## Function 3 - ploting raw data with real time

def plot_row_audio(y, sr, start_time, end_time, start_min=0, end_min=None):
    # Convert start_time to datetime
    start_time = pd.to_datetime(start_time)

    # Calculate total duration
    duration = librosa.get_duration(y=y, sr=sr)

    # Calculate sample indices for the specified range
    start_sample = int(start_min * 60 * sr)  # Convert start_min to samples
    end_sample = int(end_min * 60 * sr) if end_min is not None else len(y)  # Convert end_min to samples or default to full length

    # Extract the segment of the signal
    y_segment = y[start_sample:end_sample]

    # Calculate the time axis for the segment
    segment_duration = librosa.get_duration(y=y_segment, sr=sr)
    time_axis_segment = np.linspace(start_min, start_min + segment_duration / 60, len(y_segment))

    # Generate datetime labels for the x-axis
    segment_start_time = start_time + pd.to_timedelta(start_min, unit="m")
    segment_end_time = segment_start_time + pd.to_timedelta(segment_duration / 60, unit="m")
    time_labels = pd.date_range(start=segment_start_time, end=segment_end_time, periods=6)  # 6 evenly spaced labels

    # Plot the segment
    print('Start Plotting')
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis_segment, y_segment, alpha=0.5, label='Signal')
    plt.xticks(np.linspace(start_min, start_min + segment_duration / 60, len(time_labels)),
               time_labels.strftime('%H:%M'), rotation=45)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(f'Signal from Minute {start_min} to {end_min if end_min else round(duration / 60)} on date {start_time.date()}')
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()

## Function 4 - ploting mel spectogram with real time and by channels

def plot_spectrogram_or_mel(y, sr, start_time=None, end_time=None, start_min=0, end_min=None, cmap='magma', mel=False):
    """
    Plots spectrograms or mel spectrograms for each channel in `y` if it is a dictionary of arrays, 
    or a single spectrogram if `y` is a 1D array.

    Parameters:
        y (dict or np.array): Dictionary of audio time series (multi-channel) or a 1D numpy array (single channel).
        sr (int): Sampling rate of the audio.
        start_time (pd.Timestamp, optional): Real-world start time of the audio segment. Defaults to None.
        end_time (pd.Timestamp, optional): Real-world end time of the audio segment. Defaults to None.
        start_min (int, optional): Start time in minutes relative to the beginning of the audio. Default is 0.
        end_min (int, optional): End time in minutes relative to the beginning of the audio. Defaults to full length.
        cmap (str, optional): Colormap to use for the spectrogram. Default is 'magma'.
        mel (bool, optional): Whether to plot a mel spectrogram. Default is False.
    """

    # Convert single-channel input into a dictionary for uniform processing
    if isinstance(y, np.ndarray):
        y = {"Channel_1": y}  # Wrap in a dictionary
    elif not isinstance(y, dict) or len(y) == 0:
        print("No valid audio data provided.")
        return

    num_channels = len(y)  # Number of channels

    # Create subplots (at least one plot must exist even if empty)
    fig, axes = plt.subplots(max(1, num_channels), 1, figsize=(15, 5 * max(1, num_channels)))
    if num_channels == 1:
        axes = [axes]  # Ensure axes is iterable

    for idx, (channel_name, channel_data) in enumerate(y.items()):
        if len(channel_data) == 0:
            print(f"Warning: Channel {channel_name} has no data to plot.")
            continue

        # Convert start and end time to sample indices
        start_sample = int(start_min * 60 * sr) if start_min is not None else 0
        end_sample = int(end_min * 60 * sr) if end_min is not None else len(channel_data)

        # Ensure start and end are within bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(channel_data), end_sample)

        # Extract audio segment
        y_segment = channel_data[start_sample:end_sample]

        # Compute the spectrogram or mel spectrogram
        if mel:
            S = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_mels=256, fmax=sr)
            S_db = librosa.power_to_db(S, ref=np.max)
        else:
            S = librosa.stft(y_segment)
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

        # Compute duration and time axis
        segment_duration = librosa.get_duration(y=y_segment, sr=sr)
        num_minutes = int(segment_duration // 60) + 1
        minute_ticks = np.linspace(0, segment_duration, num=num_minutes)
        time_labels = pd.date_range(start="00:00", periods=num_minutes, freq='min').strftime('%H:%M')

        # Handle time labeling if real-world timestamps are provided
        if start_time is not None:
            segment_start_time = pd.to_datetime(start_time) + pd.to_timedelta(start_min, unit="m")
            time_labels = pd.date_range(start=segment_start_time, periods=num_minutes, freq='min').strftime('%H:%M')

        # Plot the spectrogram or mel spectrogram
        ax = axes[idx]
        if mel:
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', cmap=cmap, fmax=sr, ax=ax)
            ax.set_ylabel('Frequency (Mel)')
        else:
            frequencies = librosa.fft_frequencies(sr=sr)
            ax.pcolormesh(np.linspace(0, segment_duration, S_db.shape[1]), frequencies, S_db, cmap=cmap)
            ax.set_ylabel('Frequency (Hz)')
        
        #title = f'Channel {channel_name}' if start_time is None else f'Channel {channel_name} on {start_time.date()}'
        #ax.set_title(title)
        ax.set_xticks(minute_ticks)
        ax.set_xticklabels(time_labels, rotation=45)
        ax.set_xlabel('Time')

    plt.tight_layout()
    plt.show()


## Function 5 - reduce noise with relate to the first sec
def reduce_noise(y,sr):
    noise_section = y[:sr]  # First 1 second

    # Reduce noise
    y_reduced_noise = nr.reduce_noise(y=y, sr=sr, y_noise=noise_section)
    
    return y_reduced_noise

## Function 6 - Matched filter
def matched_filter(signal, template, threshold_factor=3.0):
    signal = np.asarray(signal)
    template = np.asarray(template)

    N, M = len(signal), len(template)

    L = N + M - 1  

    signal_fft = np.fft.fft(signal, L)  # Implicit zero-padding
    template_fft = np.fft.fft(template, L)
    result = np.fft.ifft(signal_fft * np.conjugate(template_fft))

    energy = np.sum(np.abs(template)**2)
    if energy:
        result /= energy

    mag = np.abs(result)
    noise_std = np.std(mag)  # noise estimate
    threshold = threshold_factor * noise_std
    peaks, _ = find_peaks(mag, height=threshold)

    return result, mag, np.angle(result), threshold, peaks

## Function 7 - high pass
def high_pass_filter(y, sr, cutoff, order):
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

## Function 8 - energy graph
def process_array(arr, threshold, sampling_rate):
    """
    Processes an array by applying a threshold, squaring values above it, and summing them within a sliding window.
    
    Parameters:
        arr (numpy array): Input array of numerical values.
        threshold (float): Threshold value.
        sampling_rate (int): Number of samples per second (default: 1 Hz).
        
    Returns:
        numpy array: Array of summed squared values for each window.
    """
    window_size = 10 * 60 * sampling_rate  # 10-minute window size
    step_size = 1 * 60 * sampling_rate  # 1-minute step size
    num_windows = (len(arr) - window_size) // step_size + 1  # Number of sliding windows
    
    result = np.zeros(num_windows)
    
    for i in range(num_windows):
        start = i * step_size
        window = arr[start:start + window_size]
        squared_values = np.where(window > threshold, window**2, 0)  # Square values above threshold
        result[i] = np.sum(squared_values)  # Sum all squared values
    
    return result

## Function 9 - combin WAV MISMATCH
def check_wav_mismatch(wav1, wav2):
    """
    Compare the key parameters of two WAV files and return a detailed mismatch report.
    """
    mismatches = []
    
    if wav1.getnchannels() != wav2.getnchannels():
        mismatches.append(f"Channels Mismatch: {wav1.getnchannels()} vs {wav2.getnchannels()}")
    
    if wav1.getsampwidth() != wav2.getsampwidth():
        mismatches.append(f"Sample Width Mismatch: {wav1.getsampwidth()} bytes vs {wav2.getsampwidth()} bytes")
    
    if wav1.getframerate() != wav2.getframerate():
        mismatches.append(f"Sample Rate Mismatch: {wav1.getframerate()} Hz vs {wav2.getframerate()} Hz")
    
    if wav1.getcomptype() != wav2.getcomptype():
        mismatches.append(f"Compression Type Mismatch: {wav1.getcomptype()} vs {wav2.getcomptype()}")
    
    return mismatches

## Function 10 - combin WAV
def combine_wav_files(wav1_path, wav2_path, output_path):
    with wave.open(wav1_path, 'rb') as wav1, wave.open(wav2_path, 'rb') as wav2:
        
        # Check if parameters match and provide a detailed report
        mismatches = check_wav_mismatch(wav1, wav2)
        if mismatches:
            raise ValueError("WAV files do not match:\n" + "\n".join(mismatches))

        # Open output WAV file
        with wave.open(output_path, 'wb') as output_wav:
            output_wav.setparams(wav1.getparams())  # Use first file's parameters

            # Chunk size for reading (to avoid memory issues)
            chunk_size = 1024 * 1024  # 1MB chunks

            # Copy data from first WAV file
            while True:
                data = wav1.readframes(chunk_size)
                if not data:
                    break
                output_wav.writeframes(data)

            # Copy data from second WAV file
            while True:
                data = wav2.readframes(chunk_size)
                if not data:
                    break
                output_wav.writeframes(data)

    print(f"Combined file saved at: {output_path}")


## Fucntion (???) not used maybe delete
def channels_split(audio_path, csv_path):
    y,sr,csv_df ,channel_list= loading_file(audio_path,csv_path)
    start_time, end_time = wav_real_time(audio_path,csv_df)
    y_reduced_noise = {}
    for i, channel in enumerate(channel_list):
        y_reduced_chanel = reduce_noise(channel, sr)
        y_reduced_noise[f"y_reduced_noise_{i+1}"] = np.asarray(y_reduced_chanel, dtype=np.float32)


## Usege Example
def usege_example():

    audio_path = r"C:\Users\dror.e\Documents\241103-T013.WAV"
    csv_path = r"C:\Users\dror.e\Documents\WAVFileInfo4.csv"

    y,sr,csv_df ,channel_list= loading_file(audio_path,csv_path)
    start_time, end_time = wav_real_time(audio_path,csv_df)
    y_reduced_noise = {}
    for i, channel in enumerate(channel_list):
        y_reduced_chanel = reduce_noise(channel, sr)
        y_reduced_noise[f"y_reduced_noise_{i+1}"] = np.asarray(y_reduced_chanel, dtype=np.float32)
    plot_spectrogram_or_mel(y_reduced_noise, sr, start_time, end_time, start_min=0, end_min=40, cmap='magma', mel=True)

###################################################################################################220125
import librosa
import noisereduce as nr
import time
import pickle
import multiprocessing as mp


def process_and_save_audio(file_path):
    """
    Process a single WAV file: reduce noise, split channels, and save results in a pickle file.

    Args:
        file_path (str): Path to the WAV file.
    
    Saves:
        A `.pkl` file for the WAV file with processed data.
    """
    try:
        print(f"[Worker] Started processing: {file_path}", flush=True)

        # Load audio file
        print(f"[Worker] Loading file: {file_path}", flush=True)
        y, sr = librosa.load(file_path, sr=None, mono=False)
        print(f"[Worker] Loaded file: {file_path} (sample rate: {sr})", flush=True)

        if y.ndim == 1:
            # Mono audio processing
            print(f"[Worker] Processing mono audio: {file_path}", flush=True)
            noise_section = y[:sr]  # First 1 second as noise
            y_reduced_noise = nr.reduce_noise(y=y, sr=sr, y_noise=noise_section)
            processed_data = {'channels': [y_reduced_noise], 'sr': sr}
        else:
            # Stereo or multi-channel audio processing
            print(f"[Worker] Processing multi-channel audio: {file_path}", flush=True)
            channels = []
            for ch in range(y.shape[0]):
                noise_section = y[ch, :sr]  # First 1 second as noise
                reduced_channel = nr.reduce_noise(y=y[ch], sr=sr, y_noise=noise_section)
                channels.append(reduced_channel)
            processed_data = {'channels': channels, 'sr': sr}

        print(f"[Worker] Finished processing: {file_path}", flush=True)

        # Save results to a pickle file
        file_name = file_path.split('/')[-1].split('.')[0]  # Extract file name without extension
        pkl_file_name = f"{file_name}.pkl"
        print(f"[Worker] Saving data to: {pkl_file_name}", flush=True)
        with open(pkl_file_name, 'wb') as pkl_file:
            pickle.dump(processed_data, pkl_file)
        print(f"[Worker] Saved processed data to {pkl_file_name}", flush=True)
    except Exception as e:
        print(f"[Worker] Error in file {file_path}: {e}", flush=True)
        raise RuntimeError(f"Error processing {file_path}: {e}")


def process_and_save_audio_files_concurrently(wav_list):
    """
    Process a list of WAV files concurrently: reduce noise, split channels, and save results in pickle files.

    Args:
        wav_list (list): List of paths to WAV files.
    """
    try:
        print("[Main] Starting multiprocessing", flush=True)
        print(f"[Main] Files to process: {wav_list}", flush=True)
        with mp.Pool(processes=mp.cpu_count()) as pool:
            print(f"[Main] Pool initialized with {mp.cpu_count()} workers", flush=True)
            pool.map(process_and_save_audio, wav_list)
        print("[Main] Finished multiprocessing", flush=True)
    except Exception as e:
        print(f"[Main] Exception in multiprocessing: {e}", flush=True)
        raise

#################################################################################### cross-corelation 
################CPU PER FILE####################################
import pickle
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_and_save_cross_correlation(file_path, template_path):
    """
    Compute cross-correlation for all channels in an audio file and save results in a pickle file.

    Args:
        file_path (str): Path to the input pickle file containing audio data.
        template_path (str): Path to the template file for cross-correlation.

    Saves:
        A `.pkl` file with the cross-correlation results for the audio file.
    """
    try:
        print(f"[Worker] Started processing: {file_path}", flush=True)

        # Load audio data from pickle file
        print(f"[Worker] Loading pickle file: {file_path}", flush=True)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        channels = data['channels']
        sr = data['sr']
        print(f"[Worker] Loaded data from {file_path} (sample rate: {sr}, channels: {len(channels)})", flush=True)

        # Load the template for cross-correlation
        print(f"[Worker] Loading template file: {template_path}", flush=True)
        template = np.loadtxt(template_path, dtype=float)

        # Perform cross-correlation for each channel
        print(f"[Worker] Starting cross-correlation for channels: {file_path}", flush=True)
        cross_correlations = []
        for idx, channel in enumerate(tqdm(channels, desc="[Worker] Channels")):
            signal_array = np.array(channel)  # Convert memmap to NumPy array if needed
            cross_correlation = np.correlate(signal_array, template, mode='full')
            cross_correlations.append(cross_correlation)

        # Save results to a pickle file
        file_name = file_path.split('/')[-1].split('.')[0]  # Extract file name without extension
        pkl_file_name = f"{file_name}_cross_correlation.pkl"
        print(f"[Worker] Saving cross-correlation results to: {pkl_file_name}", flush=True)
        with open(pkl_file_name, 'wb') as pkl_file:
            pickle.dump({'cross_correlations': cross_correlations, 'sr': sr}, pkl_file)

        print(f"[Worker] Saved cross-correlation results to {pkl_file_name}", flush=True)
    except Exception as e:
        print(f"[Worker] Error processing file {file_path}: {e}", flush=True)
        raise RuntimeError(f"Error processing {file_path}: {e}")


def process_and_save_cross_correlation_concurrently(file_list, template_path):
    """
    Process a list of audio files concurrently for cross-correlation.

    Args:
        file_list (list): List of paths to input pickle files.
        template_path (str): Path to the template file for cross-correlation.
    """
    try:
        print("[Main] Starting multiprocessing for cross-correlation", flush=True)
        print(f"[Main] Files to process: {file_list}", flush=True)
        with Pool(processes=cpu_count()) as pool:
            print(f"[Main] Pool initialized with {cpu_count()} workers", flush=True)
            pool.starmap(process_and_save_cross_correlation, [(file, template_path) for file in file_list])
        print("[Main] Finished multiprocessing for cross-correlation", flush=True)
    except Exception as e:
        print(f"[Main] Exception in multiprocessing: {e}", flush=True)
        raise

##################CPU PER CHANNEL##############################
def compute_cross_correlation_per_channel(channel, template):
    """
    Perform cross-correlation for a single channel.

    Args:
        channel (numpy.array): The audio data for one channel.
        template (numpy.array): The template for cross-correlation.

    Returns:
        numpy.array: Cross-correlation result for the channel.
    """
    signal_array = np.array(channel)
    return np.correlate(signal_array, template, mode='full')


def process_and_save_cross_correlation_by_channel(file_path, template_path):
    """
    Compute cross-correlation for all channels in an audio file and save results in a pickle file,
    distributing the work across CPUs at the channel level.

    Args:
        file_path (str): Path to the input pickle file containing audio data.
        template_path (str): Path to the template file for cross-correlation.

    Saves:
        A `.pkl` file with the cross-correlation results for the audio file.
    """
    try:
        print(f"[Worker] Started processing: {file_path}", flush=True)

        # Load audio data from pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        channels = data['channels']
        sr = data['sr']
        print(f"[Worker] Loaded data from {file_path} (sample rate: {sr}, channels: {len(channels)})", flush=True)

        # Load the template for cross-correlation
        template = np.loadtxt(template_path, dtype=float)

        # Parallel processing of channels
        print(f"[Worker] Starting cross-correlation for channels: {file_path}", flush=True)
        with Pool(processes=cpu_count()) as pool:
            cross_correlations = pool.starmap(
                compute_cross_correlation_per_channel, [(channel, template) for channel in channels]
            )

        # Save results to a pickle file
        file_name = file_path.split('/')[-1].split('.')[0]  # Extract file name without extension
        pkl_file_name = f"{file_name}_cross_correlation.pkl"
        print(f"[Worker] Saving cross-correlation results to: {pkl_file_name}", flush=True)
        with open(pkl_file_name, 'wb') as pkl_file:
            pickle.dump({'cross_correlations': cross_correlations, 'sr': sr}, pkl_file)

        print(f"[Worker] Saved cross-correlation results to {pkl_file_name}", flush=True)
    except Exception as e:
        print(f"[Worker] Error processing file {file_path}: {e}", flush=True)
        raise RuntimeError(f"Error processing {file_path}: {e}")


def process_and_save_cross_correlation_by_channel_concurrently(file_list, template_path):
    """
    Process a list of audio files, performing cross-correlation at the channel level.

    Args:
        file_list (list): List of paths to input pickle files.
        template_path (str): Path to the template file for cross-correlation.
    """
    try:
        print("[Main] Starting multiprocessing for cross-correlation", flush=True)
        print(f"[Main] Files to process: {file_list}", flush=True)
        for file_path in file_list:
            process_and_save_cross_correlation_by_channel(file_path, template_path)
        print("[Main] Finished multiprocessing for cross-correlation", flush=True)
    except Exception as e:
        print(f"[Main] Exception in multiprocessing: {e}", flush=True)
        raise
###########################################power##############################
import numpy as np

import numpy as np

def calculate_results(signals, threshold, sample_rate=48000):
    """
    Calculate results for overlapping 10-minute windows for each signal, processing only points above the threshold.

    Parameters:
    - signals (list of np.ndarray): List of signal arrays.
    - threshold (float): Threshold value for processing.
    - sample_rate (int): Sampling rate in Hz.

    Returns:
    - dict: Dictionary with signal index as keys and results as values.
    """
    results_dict = {}
    window_size = 10 * 60 * sample_rate  # 10-minute window in samples

    for i, signal in enumerate(signals):
        results = []
        for start in range(0, len(signal) - window_size + 1, sample_rate * 60):
            end = start + window_size
            window = signal[start:end]
            
            # Filter points above the threshold
            filtered_window = window[window > threshold]
            if len(filtered_window) > 0:  # Only process if any points exceed the threshold
                squared_diff = (filtered_window - threshold) ** 2
                results.append(np.sum(squared_diff))
        
        # Save results for the current signal
        results_dict[f'signal_{i}'] = results
    
    return results_dict

###GRAPH
import matplotlib.pyplot as plt

def plot_signals(results):
    
    """
    Plots signals from a dictionary in a 3x2 grid of subplots.
    
    Args:
        results (dict): A dictionary with keys 'signal_0' to 'signal_5', where each value is a signal (list or array).
    """
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))  # Create a 3x2 grid
    axes = axes.ravel()  # Flatten the 2D array of axes for easy iteration
    
    for i in range(6):  # Loop through each signal
        signal_key = f'signal_{i}'
        if signal_key in results:
            axes[i].plot(results[signal_key])  # Plot the signal
            axes[i].set_title(f'Signal {i}')  # Add title
        else:
            axes[i].text(0.5, 0.5, 'No Data', fontsize=12, ha='center')  # Handle missing signals
            axes[i].set_title(f'Signal {i}')
        axes[i].grid(True)  # Add grid for better readability
    
    plt.tight_layout()  # Adjust spacing between plots
    plt.show()  # Display the plot
