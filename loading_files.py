import numpy as np
import matplotlib.pyplot as plt
import os
import time

start_time = time.time()
# List of file prefixes (e.g., "241030-T006", "241030-T007", etc.)
file_prefixes = file_paths = [ r"C:\Users\dror.e\Documents\wav_files\241104-T012.WAV",
    r"C:\Users\dror.e\Documents\wav_files\241104-T036.WAV",
    r"C:\Users\dror.e\Documents\wav_files\241105-T006.WAV",
    r"C:\Users\dror.e\Documents\wav_files\241105-T007.WAV",
    #r"C:\Users\dror.e\Documents\wav_files\241105-T012.WAV",
    r"C:\Users\dror.e\Documents\wav_files\241105-T034.WAV",
    r"C:\Users\dror.e\Documents\wav_files\241105-T035.WAV",
    r"C:\Users\dror.e\Documents\wav_files\241105-T0607.WAV",
    r"C:\Users\dror.e\Documents\wav_files\241105-T3435.WAV"]

file_prefixes = [os.path.splitext(os.path.basename(path))[0] for path in file_paths]

# Loop through each file prefix
for prefix in file_prefixes:
    try:
        # Paths for clean signal and matched filter results
        path_clean_signal = f"{prefix}_clean_signal.npz"
        path_matched_filter = f"{prefix}_matched_filter_results.npz"

        print(f"Processing: {prefix}")

        # Load clean signal
        if os.path.exists(path_clean_signal):
            data = np.load(path_clean_signal)
            filtered_signal = data["arr1"]

            # Plot the filtered signal
            plt.figure(figsize=(10, 4))
            plt.plot(filtered_signal, label="Matched Filter Output")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.title(f'{prefix} Clean Signal')
            #plt.legend(loc="upper right")
            plt.grid()
            plt.savefig(f"{prefix}_clean_signal.png", dpi=300, bbox_inches='tight')
            #plt.show()

        # Load matched filter results
        if os.path.exists(path_matched_filter):
            matched = np.load(path_matched_filter)
            filtered_signal = matched["arr2"]

            # Plot the matched filter output
            plt.figure(figsize=(10, 4))
            plt.plot(filtered_signal, label="Matched Filter Output")
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.title(f'{prefix} Matched Filter Response')
            #plt.legend(loc="upper right")
            plt.grid()
            plt.savefig(f"{prefix}_mag.png", dpi=300, bbox_inches='tight')
            #plt.show()

            # Energy Graph
            def process_array(arr, threshold, sampling_rate=48000):
                window_size = 10 * 60 * sampling_rate  # 10-minute window size
                step_size = 1 * 60 * sampling_rate  # 1-minute step size
                num_windows = (len(arr) - window_size) // step_size + 1  

                result = np.zeros(num_windows)
                for i in range(num_windows):
                    start = i * step_size
                    window = arr[start:start + window_size]
                    squared_values = np.where(window > threshold, window**2, 0)  
                    result[i] = np.sum(squared_values)
                return result

            energy = process_array(matched['arr2'], 0.1)
            plt.figure(figsize=(10, 4))
            plt.plot(energy, label="Energy Graph")
            plt.xlabel("Time - Windows Sliding")
            plt.ylabel("Amplitude")
            plt.title(f'{prefix} Energy Window Graph')
            plt.ylim(0,5000)
            #plt.legend(loc="upper right")
            plt.grid()
            plt.savefig(f"{prefix}_energy.png", dpi=300, bbox_inches='tight')
            #plt.show()


            # Sum of points above threshold
            def sum_points_above_threshold(arr, threshold, sampling_rate=48000):
                window_size = 10 * 60 * sampling_rate  
                step_size = 1 * 60 * sampling_rate  
                num_windows = (len(arr) - window_size) // step_size + 1  

                result = np.zeros(num_windows)
                for i in range(num_windows):
                    start = i * step_size
                    window = arr[start:start + window_size]
                    result[i] = np.sum(window > threshold)  # Count samples above threshold
                return result

            above_threshold = sum_points_above_threshold(matched['arr2'], 0.1)
            plt.figure(figsize=(10, 4))
            plt.plot(above_threshold , label="Samples Above Threshold")
            plt.xlabel("Time - Windows Sliding")
            plt.ylabel("Amount of samples above threshold")
            plt.title(f'{prefix} Sum of Samples Above Threshold')
            plt.ylim(0,50000)
            #plt.legend(loc="upper right")
            plt.grid()
            plt.savefig(f"{prefix}_amount_samples.png", dpi=300, bbox_inches='tight')
            #plt.show()

        print(f"✅ Done processing {prefix}\n")

    except Exception as e:
        print(f"⚠️ Error processing {prefix}: {e}\n")

end_time = time.time()  # End timing
print(f"Time taken for {prefix}: {end_time - start_time:.4f} seconds\n")