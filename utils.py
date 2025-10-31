import numpy as np
import os

def rename_files(folder_path, old_char, new_char):
    ''' 
    Replace 'old_char' with 'new_char' in all files in the folder
    '''
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Construct the old and new filenames
        old_filename = os.path.join(folder_path, filename)
        new_filename = os.path.join(folder_path, filename.replace(old_char, new_char))
        
        # Rename the file
        os.rename(old_filename, new_filename)
        print(f"Renamed '{old_filename}' to '{new_filename}'")


def unwrap_with_nans(phases, unit='deg'):
    ''' Adaptation of the numpy.unwrap function to handling multiple consecutive NaNs, from ChatGPT.
    input: phases - 1D array of phase angles in degrees or radians, as specified by unit; unit - default deg
    output: 1D array of unwrapped phase angles in degrees or radians
    '''
    # Create a boolean mask for NaNs
    nan_mask = np.isnan(phases)
    
    # Split the array into contiguous non-NaN segments
    segments = []
    current_segment = []
    for i, val in enumerate(phases):
        if not np.isnan(val):
            current_segment.append(val)
        else:
            if current_segment:
                segments.append(np.array(current_segment))
                current_segment = []
            segments.append(np.array([np.nan]))
    if current_segment:
        segments.append(np.array(current_segment))
    
    # Apply unwrap to non-NaN segments
    unwrapped_segments = []
    for segment in segments:
        if np.isnan(segment).all():  # Skip the NaN segments
            unwrapped_segments.append(segment)
        else:
            # Convert degrees to radians for correct unwrapping, then back to degrees
            if unit == 'deg':
                unwrapped_segment = np.unwrap(np.deg2rad(segment))
                unwrapped_segments.append(np.rad2deg(unwrapped_segment))
            else:
                unwrapped_segments.append(np.unwrap(segment))
    
    # Merge the unwrapped segments back into one array
    unwrapped_phases = np.concatenate(unwrapped_segments)
    
    return unwrapped_phases


def compute_rmse(signal1, signal2):
    """
    Computes the Root Mean Square Error (RMSE) between two signals.
    """
    if len(signal1) != len(signal2):
        raise ValueError("Signals must have the same length.")
    mse = np.nanmean((signal1 - signal2) ** 2)  # Mean Squared Error, ignoring NaNs
    rmse = np.sqrt(mse)  # Root Mean Square Error
    return rmse


def compute_abs_area_between_signals(signal1, signal2):
    """
    Computes the area between two signals, ignoring NaNs (area is computed only where both signals are valid).
    """
    signal1 = np.asarray(signal1)
    signal2 = np.asarray(signal2)
    if signal1.shape != signal2.shape:
        raise ValueError("Signals must have the same length.")
    # Only keep indices where both signals are not NaN
    valid = ~np.isnan(signal1) & ~np.isnan(signal2)
    if not np.any(valid):
        return np.nan  # No valid overlap
    area = np.trapz(signal1[valid] - signal2[valid])
    return np.abs(area)
