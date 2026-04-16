import os
from pathlib import Path
from scipy.signal import butter, filtfilt
import pandas as pd
import numpy as np
from scipy.stats import mode


def data_segmentation(x, y, ws=50, sr=100, op=0, use_mode=False, timesteps=2):
    """
    Segments the data and labels into windows with the option to specify overlap.

    Parameters:
    - x: Input data, expected to be in the shape of (n_samples, n_features).
    - y: Labels for the data, expected to be in the shape of (n_samples,).
    - window_size: Size of each window in milliseconds (default is 50ms).
    - sampling_rate: Sampling frequency of the data in Hz (default is 100Hz).
    - overlap_percentage: Desired percentage of overlap between windows (0-100).

    Returns:
    - segmented_x: List of segmented windows from the data.
    - segmented_y: List of labels for each window, based on the label of the first sample in the window.
    """
    # Calculate the number of samples per window
    samples_per_window = int(ws * sr)

    # Calculate step size based on overlap
    step_size = int(samples_per_window * (1 - (op / 100)))

    # Initialize the lists to store segmented data and labels
    segmented_x = []
    segmented_y = []

    # Segment the data based on calculated step size and window size
    for start in range(0, len(x) - samples_per_window + 1, step_size):
        end = start + samples_per_window
        if end > len(x):  # Ensure the last window doesn't exceed data length
            break
        window = x[start:end]

        if timesteps > 1:
            window = window.values.reshape(timesteps, 1, window.shape[0]//timesteps, window.shape[1])

        if use_mode:
            label = mode(y.values[start:end])
        else:
            label = y.values[start:end][0]

        segmented_x.append(window)
        segmented_y.append(label)

    return np.array(segmented_x), np.array(segmented_y)


def data_segmentation_forecasting(x, y, sr, segment_length, forecast_length, timesteps=2):
    segmented_x, segmented_y = [], []
    n_snapshot_per_segment = int(segment_length * sr)
    n_snapshot_per_forecast = int(forecast_length * sr)

    for i in range(len(x) - n_snapshot_per_segment - n_snapshot_per_forecast + 1):
        window = x[i:i+n_snapshot_per_segment].values

        if timesteps > 1:
            window = window.reshape(timesteps, 1, window.shape[0]//timesteps, window.shape[1])

        segmented_x.append(window)
        segmented_y.append(y[i+n_snapshot_per_segment: i+n_snapshot_per_segment+n_snapshot_per_forecast].values)
    return np.array(segmented_x), np.array(segmented_y)

def low_pass_filter(dataframe, cutoff_frequency, sr, order=5):
    """
    Applies a low-pass Butterworth filter to each column of a pandas DataFrame.

    Parameters:
    - dataframe: pandas DataFrame where rows are snapshots and columns are features.
    - cutoff_frequency: The cutoff frequency of the filter in Hz.
    - sr: The sampling rate of the data in Hz.
    - order: The order of the filter (default is 5).

    Returns:
    - filtered_df: pandas DataFrame of the same shape as `df`, with the filter applied.
    """
    # Calculate the Nyquist frequency
    nyquist = 0.5 * sr

    # Normalize the cutoff frequency with respect to the Nyquist frequency
    normal_cutoff = cutoff_frequency / nyquist

    # Design the Butterworth low-pass filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Apply the filter to each column of the DataFrame
    filtered_data = dataframe.apply(lambda x: filtfilt(b, a, x), axis=0)

    # Return a new DataFrame to ensure the index and columns are preserved
    filtered_df = pd.DataFrame(filtered_data, index=dataframe.index, columns=dataframe.columns)

    return filtered_df


def main_classification():
    subjects = sorted([s for s in sorted(os.listdir('dataset')) if os.path.isdir(f'dataset/{s}')])
    print(subjects)

    # Parameters
    window_size = 1
    sampling_rate = 60
    overlap_percentage = 20

    # Preprocessing settings
    apply_low_pass_filter = True
    pre_fog = True
    post_fog = True
    cutoff = 10

    # Stats
    fog_instances = 0
    non_fog_instances = 0

    for subject in subjects:
        print(f'Processing subject: {subject}')

        subject_segments = []
        subject_labels = []

        for file in sorted([f for f in os.listdir(f'dataset/{subject}') if f.endswith('.csv')]):
            print(f'Processing file: {file}')
            df = pd.read_csv(f'dataset/{subject}/{file}', index_col=0, parse_dates=True)
            df.dropna(inplace=True)
            data = df.iloc[:, :-8]
            data.dropna(inplace=True)

            if apply_low_pass_filter:
                data = low_pass_filter(data, cutoff, sampling_rate)

            # Convert No labels to "Non-Fog" and all the type of fog to a generic "Fog"
            labels = df["fog-Agree"].apply(
                lambda x: "Non-Fog" if x.lower().startswith("n") else "Fog" if x.lower().startswith("f") else x
            )

            # Check if the file contains fog
            contains_fog = labels.unique().shape[0] > 1

            # Perform segmentation
            if contains_fog:
                try:
                    start_fog = labels[labels == "Fog"].index[0]
                    end_fog = labels[labels == "Fog"].index[-1]
                except IndexError:
                    print(f'No fog found in file: {file}')
                    continue

                # Split the data into 3 segments: before, during and after the fog
                before_fog = data.loc[:start_fog]
                during_fog = data.loc[start_fog:end_fog]
                after_fog = data.loc[end_fog:]

                # Segment in this way to avoid the overlap of the fog
                segments_before, labels_before = data_segmentation(
                    x=before_fog, y=labels.loc[:start_fog], ws=window_size, sr=sampling_rate, op=overlap_percentage
                )
                segments_during, labels_during = data_segmentation(
                    x=during_fog, y=labels.loc[start_fog:end_fog], ws=window_size, sr=sampling_rate, op=overlap_percentage
                )
                segments_after, labels_after = data_segmentation(
                    x=after_fog, y=labels.loc[end_fog:], ws=window_size, sr=sampling_rate, op=overlap_percentage
                )

                # Check if the segments are missing one or more sensors
                if not (
                        segments_before.shape[-1] == 24 or
                        segments_during.shape[-1] == 24 or
                        segments_after.shape[-1] == 24
                ):
                    print(f'Error in file: {file}, shape: {segments_before.shape} is missing one or more sensor')
                    continue

                # Append the segments and labels to the lists
                if len(segments_before.shape) == 3:
                    subject_segments.extend(segments_before)
                    if pre_fog:
                        labels_before[-1] = "Pre-Fog"

                    subject_labels.extend(labels_before)
                    non_fog_instances += labels_during.shape[0]

                if len(segments_during.shape) == 3:
                    subject_segments.extend(segments_during)
                    subject_labels.extend(labels_during)
                    fog_instances += labels_during.shape[0]

                if len(segments_after.shape) == 3:
                    subject_segments.extend(segments_after)
                    if post_fog:
                        labels_after[0] = "Post-Fog"

                    subject_labels.extend(labels_after)
                    non_fog_instances += labels_after.shape[0]

            else:
                # Segment the data
                segments_no_fog, labels_no_fog = data_segmentation(
                    x=data, y=labels, ws=window_size, sr=sampling_rate, op=overlap_percentage
                )

                # Check if the segments are missing one or more sensors
                if not segments_no_fog.shape[-1] == 24:
                    print(f'Error in file: {file}, shape: {segments_no_fog.shape} is missing one or more sensor')
                    continue

                # Append the segments and labels to the lists
                if len(segments_no_fog.shape) == 3:
                    subject_segments.extend(segments_no_fog)
                    subject_labels.extend(labels_no_fog)
                    non_fog_instances += labels_no_fog.shape[0]

        # Save the segments and labels to file
        output_dir = f"segments/{window_size}s_{overlap_percentage}perc_overlap{'_filtered' if apply_low_pass_filter else ''}{'_pre_fog' if pre_fog else ''}{'_post_fog' if post_fog else ''}/{subject}"
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        try:
            subject_segments = np.array(subject_segments)
            subject_labels = np.array(subject_labels)
            np.save(f"{output_dir}/segments.npy", subject_segments)
            np.save(f"{output_dir}/labels.npy", subject_labels)
        except Exception as e:
            print(f"Error saving segments: {e}")

    print("Stats:")
    print(f"Fog instances: {fog_instances}")
    print(f"Non-fog instances: {non_fog_instances}")

def main_forecasting():
    subjects = sorted([s for s in sorted(os.listdir('dataset')) if os.path.isdir(f'dataset/{s}')])
    print(subjects)

    # Parameters
    window_size = 3
    forecast_length = 1
    sampling_rate = 60

    # Preprocessing settings
    apply_low_pass_filter = True
    pre_fog = True
    post_fog = True
    cutoff = 10

    # Stats
    fog_instances = 0
    non_fog_instances = 0

    for subject in subjects:
        print(f'Processing subject: {subject}')

        subject_segments = []
        subject_labels = []

        for file in sorted([f for f in os.listdir(f'dataset/{subject}') if f.endswith('.csv')]):
            print(f'Processing file: {file}')
            df = pd.read_csv(f'dataset/{subject}/{file}', index_col=0, parse_dates=True)
            df.dropna(inplace=True)
            data = df.iloc[:, :-8]
            data.dropna(inplace=True)

            if apply_low_pass_filter:
                data = low_pass_filter(data, cutoff, sampling_rate)

            # Convert No labels to "Non-Fog" and all the type of fog to a generic "Fog"
            labels = df["fog-Agree"].apply(
                lambda x: "Non-Fog" if x.lower().startswith("n") else "Fog" if x.lower().startswith("f") else x
            )

            # Check if the file contains fog
            contains_fog = labels.unique().shape[0] > 1

            # Perform segmentation
            if contains_fog:
                try:
                    start_fog = labels[labels == "Fog"].index[0]
                    end_fog = labels[labels == "Fog"].index[-1]
                except IndexError:
                    print(f'No fog found in file: {file}')
                    continue

                # Split the data into 3 segments: before, during and after the fog
                before_fog = data.loc[:start_fog]
                during_fog = data.loc[start_fog:end_fog]
                after_fog = data.loc[end_fog:]

                # Add pre- post-fog if required
                if pre_fog:
                    labels[start_fog-pd.to_timedelta(60, 'ms'):start_fog] = "Pre-Fog"
                    labels[end_fog:end_fog + pd.to_timedelta(60, 'ms')] = "Post-Fog"

                # Segment in this way to avoid the overlap of the fog
                segments_before, labels_before = data_segmentation_forecasting(
                    x=before_fog, y=labels.loc[:start_fog], segment_length=window_size, forecast_length=forecast_length, sr=sampling_rate
                )
                segments_during, labels_during = data_segmentation_forecasting(
                    x=during_fog, y=labels.loc[start_fog:end_fog], segment_length=window_size, forecast_length=forecast_length, sr=sampling_rate
                )
                segments_after, labels_after = data_segmentation_forecasting(
                    x=after_fog, y=labels.loc[end_fog:], segment_length=window_size, forecast_length=forecast_length, sr=sampling_rate
                )

                # Check if the segments are missing one or more sensors
                if not (
                        segments_before.shape[-1] == 24 or
                        segments_during.shape[-1] == 24 or
                        segments_after.shape[-1] == 24
                ):
                    print(f'Error in file: {file}, shape: {segments_before.shape} is missing one or more sensor')
                    continue

                # Append the segments and labels to the lists
                if len(segments_before.shape) == 3 or len(segments_before.shape) == 5:
                    subject_segments.extend(segments_before)
                    subject_labels.extend(labels_before)
                    non_fog_instances += labels_during.shape[0]

                if len(segments_during.shape) == 3 or len(segments_during.shape) == 5:
                    subject_segments.extend(segments_during)
                    subject_labels.extend(labels_during)
                    fog_instances += labels_during.shape[0]

                if len(segments_after.shape) == 3 or len(segments_after.shape) == 5:
                    subject_segments.extend(segments_after)
                    subject_labels.extend(labels_after)
                    non_fog_instances += labels_after.shape[0]

            else:
                # Segment the data
                segments_no_fog, labels_no_fog = data_segmentation_forecasting(
                    x=data, y=labels, segment_length=window_size, forecast_length=forecast_length, sr=sampling_rate
                )

                # Check if the segments are missing one or more sensors
                if not segments_no_fog.shape[-1] == 24:
                    print(f'Error in file: {file}, shape: {segments_no_fog.shape} is missing one or more sensor')
                    continue

                # Append the segments and labels to the lists
                if len(segments_no_fog.shape) == 3:
                    subject_segments.extend(segments_no_fog)
                    subject_labels.extend(labels_no_fog)
                    non_fog_instances += labels_no_fog.shape[0]

        # Save the segments and labels to file
        output_dir = f"segments/forecasting_{window_size}s_{'_filtered' if apply_low_pass_filter else ''}{'_pre_fog' if pre_fog else ''}{'_post_fog' if post_fog else ''}/{subject}"
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        try:
            subject_segments = np.array(subject_segments)
            subject_labels = np.array(subject_labels)
            np.save(f"{output_dir}/segments.npy", subject_segments)
            np.save(f"{output_dir}/labels.npy", subject_labels)
        except Exception as e:
            print(f"Error saving segments: {e}")

    print("Stats:")
    print(f"Fog instances: {fog_instances}")
    print(f"Non-fog instances: {non_fog_instances}")


if __name__ == '__main__':
    main_forecasting()
