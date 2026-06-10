import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.interpolate import CubicSpline


def save_figure_multi_format(figure, directory, filename, dpi=128, bbox_inches=None):
    """Save a figure as PNG plus EPS/SVG copies in dedicated subfolders."""
    if not os.path.exists(directory):
        os.mkdir(directory)

    eps_path = os.path.join(directory, 'eps')
    svg_path = os.path.join(directory, 'svg')
    if not os.path.exists(eps_path):
        os.mkdir(eps_path)
    if not os.path.exists(svg_path):
        os.mkdir(svg_path)

    filename_root = os.path.splitext(filename)[0]
    save_kwargs = {'dpi': dpi}
    if bbox_inches is not None:
        save_kwargs['bbox_inches'] = bbox_inches

    figure.savefig(os.path.join(directory, filename_root + '.png'), **save_kwargs)
    figure.savefig(os.path.join(eps_path, filename_root + '.eps'), **save_kwargs)
    figure.savefig(os.path.join(svg_path, filename_root + '.svg'), **save_kwargs)


def add_output_suffix(filename, suffix):
    if not suffix:
        return filename
    filename_root, filename_ext = os.path.splitext(filename)
    return f'{filename_root}{suffix}{filename_ext}'


def get_stance_phase_reference_suffix(reference_mode):
    if reference_mode == 'slow_hind':
        return ''
    if reference_mode == 'ipsi_hind':
        return '_ipsi_hind_ref'
    raise ValueError(f'Unknown stance_phase_reference_mode: {reference_mode}')


def get_phase_st_output_filename(filename, reference_mode):
    return add_output_suffix(filename, get_stance_phase_reference_suffix(reference_mode))


def sanitize_filename_label(label):
    return label.replace(' ', '_')


def get_stance_phase_reference_paw(experiment_name, animal, reference_mode, left_animals, right_animals, bilateral_animals):
    if reference_mode == 'slow_hind':
        ref_paw = 3
        if ('contra' in experiment_name and animal in right_animals) or ('ipsi' in experiment_name and animal in left_animals) or ('ipsi' in experiment_name and animal in bilateral_animals):
            ref_paw = 1
        return ref_paw

    if reference_mode == 'ipsi_hind':
        if experiment_name == 'WT':
            return 1
        if animal in left_animals:
            return 3
        if animal in right_animals or animal in bilateral_animals:
            return 1
        return get_stance_phase_reference_paw(experiment_name, animal, 'slow_hind', left_animals, right_animals, bilateral_animals)

    raise ValueError(f'Unknown stance_phase_reference_mode: {reference_mode}')


def get_paw_plot_labels(experiment_name, experiment_names, paws):
    """Return display and filename labels for the current paw ordering."""
    if 'contra' in experiment_name or 'ipsi' in experiment_name:
        return ['FF', 'HF', 'FS', 'HS'], ['FF', 'HF', 'FS', 'HS']
    if any('right' in element for element in experiment_names) and any('left' in element for element in experiment_names):
        return ['FF', 'HF', 'FS', 'HS'], ['FF', 'HF', 'FS', 'HS']
    if 'right' in experiment_name:
        return ['FF', 'HF', 'FS', 'HS'], ['FF', 'HF', 'FS', 'HS']
    if 'left' in experiment_name:
        return ['FS', 'HS', 'FF', 'HF'], ['FS', 'HS', 'FF', 'HF']
    return list(paws), list(paws)


def get_experiment_name_for_path(path, experiment_names):
    for experiment_name in experiment_names:
        if experiment_name in path:
            return experiment_name
    return os.path.basename(os.path.normpath(path))


def get_path_color(experiment_name, path_index, experiment_colors_dict, color_cycle=None):
    if color_cycle is None:
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if experiment_name in experiment_colors_dict:
        return experiment_colors_dict[experiment_name]
    for label, color in experiment_colors_dict.items():
        if label in experiment_name or experiment_name in label:
            return color
    return color_cycle[path_index % len(color_cycle)]


def get_summary_limb_order(experiment_names, paws):
    if any(('contra' in name) or ('ipsi' in name) for name in experiment_names):
        return [0, 2, 1, 3], ['FF', 'FS', 'HF', 'HS']
    if any('right' in name for name in experiment_names) and any('left' in name for name in experiment_names):
        return [0, 2, 1, 3], ['FF', 'FS', 'HF', 'HS']
    if any('right' in name for name in experiment_names):
        return [0, 2, 1, 3], ['FF', 'FS', 'HF', 'HS']
    if any('left' in name for name in experiment_names):
        return [2, 0, 3, 1], ['FF', 'FS', 'HF', 'HS']
    return list(range(len(paws))), list(paws)


def normalize_summary_scatter_paw_label(label):
    alias_map = {
        'FR': 'FR',
        'FL': 'FL',
        'HR': 'HR',
        'HL': 'HL',
        'FF': 'FR',
        'FS': 'FL',
        'HF': 'HR',
        'HS': 'HL',
    }
    return alias_map.get(label, label)


def compare_metric_groups(reference_values, comparison_values, statistics_test, paired=None, reference_names=None, comparison_names=None):
    reference_values = np.asarray(reference_values, dtype=float)
    comparison_values = np.asarray(comparison_values, dtype=float)

    if paired is None:
        paired = (
            reference_names is not None
            and comparison_names is not None
            and list(reference_names) == list(comparison_names)
            and len(reference_values) == len(comparison_values)
        )

    if paired:
        valid_mask = np.isfinite(reference_values) & np.isfinite(comparison_values)
        reference_clean = reference_values[valid_mask]
        comparison_clean = comparison_values[valid_mask]
        if len(reference_clean) == 0:
            return np.nan, 'Wilcoxon' if statistics_test != 'ttest' else 'paired t-test'
        try:
            if statistics_test == 'ttest':
                return st.ttest_rel(reference_clean, comparison_clean).pvalue, 'paired t-test'
            return st.wilcoxon(reference_clean, comparison_clean).pvalue, 'Wilcoxon'
        except ValueError:
            return np.nan, 'Wilcoxon' if statistics_test != 'ttest' else 'paired t-test'

    reference_clean = reference_values[np.isfinite(reference_values)]
    comparison_clean = comparison_values[np.isfinite(comparison_values)]
    if len(reference_clean) == 0 or len(comparison_clean) == 0:
        return np.nan, 'Mann-Whitney U' if statistics_test != 'ttest' else 't-test'
    if statistics_test == 'ttest':
        return st.ttest_ind(reference_clean, comparison_clean, equal_var=False).pvalue, 't-test'
    return st.mannwhitneyu(reference_clean, comparison_clean, alternative='two-sided').pvalue, 'Mann-Whitney U'


def compare_metric_against_zero(values, statistics_test):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, 'Wilcoxon vs 0' if statistics_test != 'ttest' else 'one-sample t-test'
    try:
        if statistics_test == 'ttest':
            return st.ttest_1samp(values, 0.0, nan_policy='omit').pvalue, 'one-sample t-test'
        return st.wilcoxon(values, zero_method='wilcox').pvalue, 'Wilcoxon vs 0'
    except ValueError:
        return np.nan, 'Wilcoxon vs 0' if statistics_test != 'ttest' else 'one-sample t-test'


def pvalue_to_label(pvalue):
    if pvalue is None or not np.isfinite(pvalue):
        return 'n.s.'
    if pvalue < 0.001:
        return '**'
    if pvalue < 0.05:
        return '*'
    return 'n.s.'


def pvalue_to_annotation(pvalue):
    if pvalue is None or not np.isfinite(pvalue):
        return 'n.s.'
    if pvalue < 0.001:
        return '**'
    if pvalue < 0.05:
        return '*'
    return f'p={pvalue:.2f}'


def annotation_fontsize(annotation_text):
    if annotation_text in {'*', '**'}:
        return 14
    if annotation_text.startswith('p='):
        return 9
    return 10


def plot_limb_metric_scatter(metric_by_path, path_display_names, path_colors, summary_colors_by_path, limb_labels, limb_colors, ylabel, stat_results=None, stat_mode='between_paths', fig_size=(5, 3)):
    scatter_width = max(3.0, fig_size[0] * (len(limb_labels) / 4))
    fig, ax = plt.subplots(figsize=(scatter_width, fig_size[1]), tight_layout=True)
    n_paths = len(path_display_names)
    x_positions = np.arange(1, len(limb_labels) + 1)
    offsets = np.linspace(-0.25, 0.25, n_paths) if n_paths > 1 else np.array([0.0])

    for limb_idx, limb_label in enumerate(limb_labels):
        for path_idx, path_name in enumerate(path_display_names):
            values = np.asarray(metric_by_path[path_idx][limb_idx], dtype=float)
            valid_mask = np.isfinite(values)
            values = values[valid_mask]
            x_coord = x_positions[limb_idx] + offsets[path_idx]
            if len(values) == 0:
                continue
            ax.scatter(
                np.full(len(values), x_coord),
                values,
                s=30,
                c=summary_colors_by_path[path_idx][limb_idx],
                edgecolors=summary_colors_by_path[path_idx][limb_idx],
                linewidth=0.8,
                alpha=0.9,
                zorder=3,
            )
            ax.plot(
                [x_coord - 0.10, x_coord + 0.10],
                [np.nanmean(values), np.nanmean(values)],
                color=summary_colors_by_path[path_idx][limb_idx],
                linewidth=2.5,
                zorder=4,
            )

    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(limb_labels)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)

    all_values = []
    for path_metrics in metric_by_path:
        for limb_metrics in path_metrics:
            valid_values = np.asarray(limb_metrics, dtype=float)
            valid_values = valid_values[np.isfinite(valid_values)]
            if len(valid_values) > 0:
                all_values.append(valid_values)
    if all_values:
        global_min = min(np.min(values) for values in all_values)
        global_max = max(np.max(values) for values in all_values)
    else:
        global_min, global_max = -1, 1
    y_range = max(global_max - global_min, 1)

    if stat_results is not None and len(stat_results) > 0 and stat_mode == 'between_paths':
        for limb_idx in range(len(limb_labels)):
            limb_values = []
            for path_metrics in metric_by_path:
                values = np.asarray(path_metrics[limb_idx], dtype=float)
                values = values[np.isfinite(values)]
                if len(values) > 0:
                    limb_values.append(values)
            limb_max = max((np.max(values) for values in limb_values), default=global_max)
            for comparison_idx, pvalues in enumerate(stat_results):
                if limb_idx >= len(pvalues):
                    continue
                pvalue = pvalues[limb_idx]
                label = pvalue_to_annotation(pvalue)
                x1 = x_positions[limb_idx] + offsets[0]
                x2 = x_positions[limb_idx] + offsets[comparison_idx + 1]
                y = limb_max + y_range * (0.08 + 0.10 * comparison_idx)
                ax.plot([x1, x2], [y, y], color='k', linewidth=0.7)
                ax.text((x1 + x2) / 2, y + y_range * 0.02, label, ha='center', va='bottom', fontsize=annotation_fontsize(label))
        top_padding = y_range * (0.22 + 0.10 * max(len(stat_results) - 1, 0))
        ax.set_ylim(global_min - 0.08 * y_range, global_max + top_padding)
    elif stat_results is not None and len(stat_results) > 0 and stat_mode == 'vs_zero':
        for limb_idx in range(len(limb_labels)):
            limb_values = []
            for path_metrics in metric_by_path:
                values = np.asarray(path_metrics[limb_idx], dtype=float)
                values = values[np.isfinite(values)]
                if len(values) > 0:
                    limb_values.append(values)
            limb_max = max((np.max(values) for values in limb_values), default=global_max)
            for path_idx, pvalues in enumerate(stat_results):
                if limb_idx >= len(pvalues):
                    continue
                x = x_positions[limb_idx] + offsets[path_idx]
                y = limb_max + y_range * (0.08 + 0.08 * path_idx)
                label = pvalue_to_annotation(pvalues[limb_idx])
                ax.text(x, y, label, ha='center', va='bottom', fontsize=annotation_fontsize(label))
        top_padding = y_range * (0.18 + 0.08 * max(len(stat_results) - 1, 0))
        ax.set_ylim(global_min - 0.08 * y_range, global_max + top_padding)

    return fig


def plot_front_paws_average(param_values, animal_ids, ylabel, title, fig_size, split_start, split_duration, stim_start, stim_duration, baseline_centered, n_trials, paw_colors, paws, y_limits=None):
    fig, ax = plt.subplots(figsize=fig_size, tight_layout=True)
    if y_limits is not None:
        ax.set_ylim(y_limits)
        rectangle_y = y_limits[0]
        rectangle_height = y_limits[1] - y_limits[0]
    else:
        selected_values = param_values[animal_ids, [0, 2], :]
        rectangle_y = np.nanmin(selected_values)
        rectangle_height = np.nanmax(selected_values) - rectangle_y
    rectangle = plt.Rectangle(
        (split_start - 0.5, rectangle_y),
        split_duration,
        rectangle_height,
        fc='lightgray',
        alpha=0.3,
    )
    ax.add_patch(rectangle)
    ax.axvline(x=stim_start - 0.5, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=stim_start + stim_duration - 0.5, color='k', linestyle='-', linewidth=0.5)
    if baseline_centered:
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    x_values = np.arange(1, n_trials + 1)
    for paw_idx in [0, 2]:
        paw_data = param_values[animal_ids, paw_idx, :]
        paw_mean = np.nanmean(paw_data, axis=0)
        paw_sem = np.nanstd(paw_data, axis=0) / np.sqrt(len(animal_ids))
        ax.plot(x_values, paw_mean, color=paw_colors[paw_idx], linewidth=3, label=paws[paw_idx])
        ax.fill_between(x_values, paw_mean - paw_sem, paw_mean + paw_sem, color=paw_colors[paw_idx], alpha=0.35)

    ax.set_xlabel('Trial', fontsize=28)
    ax.set_ylabel(ylabel, fontsize=28)
    ax.set_title(title, fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(frameon=False)
    return fig


def get_param_output_name(param_name, reference_mode):
    if param_name == 'phase_st':
        return get_phase_st_output_filename(param_name, reference_mode)
    return param_name


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
