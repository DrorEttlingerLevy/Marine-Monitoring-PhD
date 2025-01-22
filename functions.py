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

def plot_spectrogram_or_mel(y, sr, start_time, end_time, start_min=0, end_min=None, cmap='magma', mel=False):
    """
    Plots spectrograms or mel spectrograms for each channel in `y` if it is a dictionary of arrays.

    Parameters:
        y (dict): Dictionary of audio time series, where each key is a channel name and each value is a 1D array.
        sr (int): Sampling rate of the audio.
        start_time (pd.Timestamp): Real-world start time of the audio segment.
        end_time (pd.Timestamp): Real-world end time of the audio segment.
        start_min (int): Start time in minutes relative to the beginning of the audio.
        end_min (int): End time in minutes relative to the beginning of the audio.
        cmap (str): Colormap to use for the spectrogram.
        mel (bool): Whether to plot a mel spectrogram. Default is False.
    """
    

    num_channels = len(y)  # Number of channels

    # Create subplots for multiple channels
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))  # For up to 6 channels, 2 columns and 3 rows
    axes = axes.flatten()

    for idx, (channel_name, channel_data) in enumerate(y.items()):
        # Convert start and end time to sample indices
        start_sample = int(start_min * 60 * sr)
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

        # Generate time axis in real-world format
        segment_duration = librosa.get_duration(y=y_segment, sr=sr)
        segment_start_time = pd.to_datetime(start_time) + pd.to_timedelta(start_min, unit="m")
        segment_end_time = segment_start_time + pd.to_timedelta(segment_duration / 60, unit="m")
        num_minutes = int(segment_duration // 60) + 1
        minute_ticks = np.linspace(0, segment_duration, num=num_minutes)
        time_labels = pd.date_range(start=segment_start_time, periods=num_minutes, freq='min')

        # Plot the spectrogram or mel spectrogram
        ax = axes[idx]
        if mel:
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', cmap=cmap, fmax=sr, ax=ax)
            ax.set_ylabel('Frequency (Mel)')
        else:
            frequencies = librosa.fft_frequencies(sr=sr)
            ax.pcolormesh(np.linspace(0, segment_duration, S_db.shape[1]), frequencies, S_db, cmap=cmap)
            ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'Channel {channel_name[-1:]} Signal on date {start_time.date()}')
        ax.set_xticks(minute_ticks)
        ax.set_xticklabels(time_labels.strftime('%H:%M'), rotation=45)
        ax.set_xlabel('Time')

    # Hide any unused subplots
    for ax in axes[num_channels:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def reduce_noise(y,sr):
    noise_section = y[:sr]  # First 1 second

    # Reduce noise
    y_reduced_noise = nr.reduce_noise(y=y, sr=sr, y_noise=noise_section)
    
    return y_reduced_noise

def channels_split(audio_path, csv_path):
    y,sr,csv_df ,channel_list= loading_file(audio_path,csv_path)
    start_time, end_time = wav_real_time(audio_path,csv_df)
    y_reduced_noise = {}
    for i, channel in enumerate(channel_list):
        y_reduced_chanel = reduce_noise(channel, sr)
        y_reduced_noise[f"y_reduced_noise_{i+1}"] = np.asarray(y_reduced_chanel, dtype=np.float32)

def hey():
    print("hey")

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
import numpy as np
import pickle
from multiprocessing import Pool
import logging
import time

# Setup logging
def configure_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_cross_correlation(args):
    """
    Helper function to calculate cross-correlation for a single channel.
    
    Parameters:
        args (tuple): A tuple containing the channel data, template, and channel index.

    Returns:
        tuple: Channel index and its cross-correlation result.
    """
    channel_data, template, index = args
    logging.info(f"Processing channel {index}...")
    logging.getLogger().handlers[0].flush()  # Force log flushing
    cross_correlation = np.correlate(channel_data, template, mode='full')
    logging.info(f"Channel {index}: Cross-correlation values: {cross_correlation[:10]}... (showing first 10 values)")
    logging.getLogger().handlers[0].flush()  # Force log flushing
    # Save full cross-correlation values to a file
    with open(f"channel_{index}_cross_correlation.txt", "w") as f:
        f.write(", ".join(map(str, cross_correlation)))
    logging.info(f"Channel {index}: Cross-correlation values saved to channel_{index}_cross_correlation.txt.")
    logging.getLogger().handlers[0].flush()  # Force log flushing
    return index, cross_correlation


def perform_cross_correlation_for_channels(template_path, pkl_file_path, channel_indices):
    """
    Perform cross-correlation between a template and multiple channels from a .pkl file in parallel.

    Parameters:
        template_path (str): Path to the .txt file containing the template.
        pkl_file_path (str): Path to the .pkl file containing signal data.
        channel_indices (slice): Slice object to select specific channels (e.g., slice(0, 6)).

    Returns:
        dict: Dictionary with channel indices as keys and cross-correlation results as values.
    """
    logging.info("Loading template...")
    template = np.loadtxt(template_path, dtype=float)
    logging.info("Template loaded.")

    logging.info("Loading signal data from .pkl file...")
    with open(pkl_file_path, 'rb') as file:
        y_reduced_noise = pickle.load(file)
        channels = y_reduced_noise['channels']
    logging.info("Signal data loaded.")

    # Extract the name of the .pkl file (without extension)
    wav_name = pkl_file_path.split('\\')[-1].split('.')[0]
    logging.info(f"Processing file: {wav_name}")

    # Prepare arguments for parallel processing
    args = [(channels[i], template, i) for i in range(channel_indices.start, channel_indices.stop)]
    logging.info(f"Prepared arguments for channels {channel_indices.start} to {channel_indices.stop - 1}.")

    # Perform cross-correlation in parallel
    logging.info("Starting parallel processing...")
    with Pool(initializer=configure_logging) as pool:
        results = pool.map(calculate_cross_correlation, args)
    logging.info("Parallel processing completed.")

    # Compile results into a dictionary
    cross_correlation_results = {index: result for index, result in results}
    logging.info("Cross-correlation results compiled.")

    return {f"cross_correlation_{wav_name}": cross_correlation_results}