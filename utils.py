import numpy as np
import os
from scipy.interpolate import CubicSpline
def get_baseline_scatter_ylim(param_name, use_uniform_ranges, bars_ranges):
    if not use_uniform_ranges:
        return None
    max_abs_limit = max(abs(value) for value in bars_ranges[param_name])
    scaled_limit = 1.5 * max_abs_limit
    return [-scaled_limit, scaled_limit]


def get_baseline_export_directory(path, export_folder_name):
    normalized_path = os.path.normpath(path)
    current_dir = normalized_path

    while True:
        folder_name = os.path.basename(current_dir)
        if folder_name.startswith('HISTO_CHECKED_ANIMALS'):
            return os.path.join(os.path.dirname(current_dir), export_folder_name)

        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir

    return os.path.join(os.path.dirname(normalized_path), export_folder_name)


def get_baseline_export_tag(path):
    normalized_path = os.path.normpath(path)
    session_name = os.path.basename(normalized_path).replace('CL-Ali', '').strip()
    session_name_lower = session_name.lower()
    tag_parts = []

    for cohort_name in ['WT', 'LATinj', 'RLinj', 'Linj', 'Rinj']:
        if cohort_name.lower() in normalized_path.lower():
            tag_parts.append(cohort_name)
            break

    if 'contra' in session_name_lower and 'fast' in session_name_lower:
        tag_parts.append('contra-fast')
    elif 'ipsi' in session_name_lower and 'fast' in session_name_lower:
        tag_parts.append('ipsi-fast')
    elif 'left' in session_name_lower and 'fast' in session_name_lower:
        tag_parts.append('left-fast')
    elif 'right' in session_name_lower and 'fast' in session_name_lower:
        tag_parts.append('right-fast')

    if 'contra' in session_name_lower and 'right' in session_name_lower:
        tag_parts.append('right')
    elif 'ipsi' in session_name_lower and 'left' in session_name_lower:
        tag_parts.append('left')

    if not tag_parts:
        tag_parts.append(session_name.replace(' ', '_'))

    return '_'.join(tag_parts)


def save_baseline_scatter_values(param_name, param_label, path, baseline_values, animal_names, reference_mode, export_folder_name):
    export_dir = get_baseline_export_directory(path, export_folder_name)
    os.makedirs(export_dir, exist_ok=True)

    baseline_values = np.asarray(baseline_values, dtype=float)
    animal_names = list(animal_names)
    if len(animal_names) < len(baseline_values):
        animal_names.extend([f'animal_{idx+1}' for idx in range(len(animal_names), len(baseline_values))])
    else:
        animal_names = animal_names[:len(baseline_values)]

    baseline_table = pd.DataFrame({
        'animal': animal_names,
        'baseline_value': baseline_values,
        'source_path': path,
    })
    baseline_table = baseline_table[np.isfinite(baseline_table['baseline_value'])]

    output_name = f'{param_label}_{get_baseline_export_tag(path)}.csv'
    if param_name == 'phase_st':
        output_name = get_phase_st_output_filename(output_name, reference_mode)
    baseline_table.to_csv(os.path.join(export_dir, output_name), index=False)

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


def count_zeros(arr):
    """Return (n_zeros, n_finite) for any array-like input (supports ragged arrays).
    Handles:
    - None or empty → (0, 0)
    - Homogeneous ndarrays → direct vectorized ops
    - Ragged/nested lists of arrays → recursively flatten numeric values
    """
    if arr is None:
        return 0, 0
    # Fast path: proper ndarray with numeric dtype
    if isinstance(arr, np.ndarray) and arr.dtype != object:
        a = arr
        if a.size == 0:
            return 0, 0
        finite = np.isfinite(a)
        zeros = np.sum((a == 0) & finite)
        return int(zeros), int(np.sum(finite))

    # Slow path: ragged or nested sequences → flatten numeric scalars
    def _iter_numeric(x):
        if isinstance(x, (list, tuple)):
            for xi in x:
                yield from _iter_numeric(xi)
        elif isinstance(x, np.ndarray):
            if x.dtype == object:
                # iterate elements (could be nested)
                for xi in x:
                    yield from _iter_numeric(xi)
            else:
                for xi in x.ravel():
                    yield xi
        else:
            # Try to treat as numeric scalar
            try:
                yield float(x)
            except Exception:
                # skip non-numeric
                return

    flat_vals = np.array(list(_iter_numeric(arr)), dtype=float)
    if flat_vals.size == 0:
        return 0, 0
    finite = np.isfinite(flat_vals)
    zeros = np.sum((flat_vals == 0.0) & finite)
    return int(zeros), int(np.sum(finite))


def correct_signal_cubic_spline(signal, dt, thr_sig='speed'):
    """Correct a 1D signal using cubic spline interpolation after outlier detection.
    thr_sig: 'speed' (default) or 'acceleration' for thresholding.
    Returns (speed, acceleration, signal_without_outliers, corrected_signal).
    """
    time = np.arange(len(signal)) * dt
    speed = np.diff(signal, prepend=signal[0])
    acceleration = np.diff(speed, prepend=speed[0]) / dt

    if thr_sig == 'speed':
        s_std = np.nanstd(speed)
        s_thresh = 3 * s_std
        outliers_speed = np.abs(speed) > s_thresh
        constant_speed = np.abs(np.diff(speed, prepend=speed[0])) == 0
        constant_speed_outliers = np.convolve(constant_speed, np.ones(5, dtype=int), mode='same') >= 5
        outliers = outliers_speed | constant_speed_outliers
    elif thr_sig == 'acceleration':
        a_std = np.nanstd(acceleration)
        a_thresh = 3 * a_std
        outliers = np.abs(acceleration) > a_thresh
    else:
        outliers = np.zeros_like(speed, dtype=bool)

    buffer = 5
    outlier_idx = np.where(outliers)[0]
    for idx in outlier_idx:
        outliers[max(0, idx - buffer):min(len(outliers), idx + buffer + 1)] = True

    good_idx = ~outliers
    signal_without_outliers = signal.copy()
    signal_without_outliers[~good_idx] = np.nan

    valid_idx = ~np.isnan(signal[good_idx])
    valid_time = time[good_idx][valid_idx]
    valid_signal = signal[good_idx][valid_idx]

    if len(valid_signal) > 0:
        cs = CubicSpline(valid_time, valid_signal, bc_type='clamped')
        corrected_signal = cs(time)
    else:
        corrected_signal = signal.copy()

    return speed, acceleration, signal_without_outliers, corrected_signal


def extract_paw_vec(ds_arr, idx):
    """Extract paw vector at index idx from ds_arr supporting 2D or 1D(object) shapes."""
    try:
        a = np.asarray(ds_arr)
        if a.ndim == 2:
            vec = np.asarray(a[idx, :], dtype=float)
        else:
            vec = np.asarray(ds_arr[idx], dtype=float)
    except Exception:
        try:
            vec = np.asarray(ds_arr[idx], dtype=float)
        except Exception:
            vec = np.array([], dtype=float)
    vec = np.where(np.isfinite(vec), vec, np.nan).astype(float)
    return vec


def stride_bounds(mat, paw_idx):
    """Return stride start/end frame indices for a given paw from stride matrices."""
    try:
        arr = np.asarray(mat[paw_idx])
        starts = arr[:, 0, 4].astype(float)
        ends = arr[:, 1, 4].astype(float)
        mask = np.isfinite(starts) & np.isfinite(ends)
        return starts[mask].astype(int), ends[mask].astype(int)
    except Exception:
        return np.array([], dtype=int), np.array([], dtype=int)
