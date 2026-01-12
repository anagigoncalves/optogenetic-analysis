"""
check_tracking.py - Analyze and validate DLC tracking data for locomotion experiments.
"""

import locomotion_class
import plotly.graph_objects as go
from scipy.signal import savgol_filter, find_peaks, medfilt
import numpy as np
import os
from copy import deepcopy
from plotly.subplots import make_subplots
from utils import (
    compute_abs_area_between_signals,
    count_zeros,
    correct_signal_cubic_spline,
    extract_paw_vec,
    stride_bounds,
)
try:
    import cv2
except Exception:
    cv2 = None


# =============================================================================
# HELPER FUNCTIONS - Reduce code redundancy in plotting
# =============================================================================

def add_line_trace(fig, y, name, color, dash=None, secondary_y=False):
    """Add a line trace to a plotly figure."""
    line_style = dict(color=color)
    if dash:
        line_style['dash'] = dash
    trace = go.Scatter(y=y, name=name, line=line_style)
    if secondary_y:
        fig.add_trace(trace, secondary_y=True)
    else:
        fig.add_trace(trace)


def add_marker_trace(fig, x, y, name, symbol, color, size=8, row=None, col=None, showlegend=True):
    """Add a marker trace to a plotly figure."""
    marker_style = dict(symbol=symbol, color=color, size=size)
    trace = go.Scatter(x=x, y=y, mode='markers', name=name, marker=marker_style, showlegend=showlegend)
    if row is not None and col is not None:
        fig.add_trace(trace, row=row, col=col)
    else:
        fig.add_trace(trace)


def extract_stride_indices(stride_mat, paw_idx):
    """Extract finite stride indices from a stride matrix for a given paw.
    
    Returns:
        np.ndarray: Array of valid (finite) stride frame indices as int64.
    """
    if stride_mat is None:
        return np.array([], dtype=np.int64)
    raw = np.asarray(stride_mat[paw_idx][:, 0, -1])
    return raw[np.isfinite(raw)].astype(np.int64)


def add_stride_markers(fig, data_signal, st_all, sw_all, st_excl, sw_excl, paw, 
                       window_label, st_color, sw_color, symbol_open, symbol_filled):
    """Add stance/swing markers for both pre-exclusion and post-exclusion strides."""
    # Pre-exclusion markers (open symbols)
    if len(st_all) > 0:
        add_marker_trace(fig, st_all, data_signal[st_all], 
                        f'{paw} ST {window_label} pre-excl', symbol_open, st_color)
    if len(sw_all) > 0:
        add_marker_trace(fig, sw_all, data_signal[sw_all], 
                        f'{paw} SW {window_label} pre-excl', symbol_open, sw_color)
    # Post-exclusion markers (filled symbols)
    if len(st_excl) > 0:
        add_marker_trace(fig, st_excl, data_signal[st_excl], 
                        f'{paw} ST {window_label} excl', symbol_filled, st_color)
    if len(sw_excl) > 0:
        add_marker_trace(fig, sw_excl, data_signal[sw_excl], 
                        f'{paw} SW {window_label} excl', symbol_filled, sw_color)


def compute_max_abs_safe(arrays, default=1.0):
    """Compute the maximum absolute value across multiple arrays, handling NaNs and edge cases."""
    try:
        max_val = np.nanmax([np.nanmax(np.abs(arr)) for arr in arrays])
    except Exception:
        max_val = default
    if not np.isfinite(max_val) or max_val == 0:
        max_val = default
    return max_val


def print_rmse_stats(label, signal1, signal2, compute_func):
    """Print RMSE and NaN count for a signal comparison."""
    rmse = compute_func(signal1, signal2)
    nan_count = np.sum(np.isnan(signal2))
    print(f"RMSE and #nan for {label}: {rmse}  {nan_count}")


def add_vertical_lines(fig, x_indices, y_min, y_max, name, color, dash=None, width=1):
    """Add vertical lines at specified x positions using a single trace with None separators."""
    if len(x_indices) == 0:
        return
    x_lines = []
    y_lines = []
    for x0 in x_indices:
        x_lines.extend([x0, x0, None])
        y_lines.extend([y_min, y_max, None])
    line_style = dict(color=color, width=width)
    if dash:
        line_style['dash'] = dash
    fig.add_trace(go.Scatter(x=x_lines, y=y_lines, mode='lines', name=name, line=line_style))


# =============================================================================
# CONFIGURATION - Modify these values to configure the analysis
# =============================================================================

# --- Paths ---
path = 'D:\\AliG\\climbing-opto-treadmill\\WT split-belt learning\\'
#path = 'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\tied stance stim retracked with ClosedLoop-AliceG-2025-10-06\\'
path_retracked = None  # Optional: path to retracked data for comparison

# --- Session Selection ---
animal = 'MC2586'          # 'MC16848'     #
session = 1
trial = 9

# --- Multi-Animal Double Support Analysis ---
# If animals_ds and trials_ds are defined, run batch double support analysis
animals_ds = ['MC16848', 'MC16851', 'MC17319', 'MC17665', 'MC17666', 'MC17669', 'MC17670', 'MC19082', 'MC19124', 
                       'MC19214', 'VIV41329', 'VIV41330', 'VIV42375', 'VIV42428', 'VIV42429', 'VIV42430', 'VIV42376', 'MC19107']       #     ['MC2586']  # List of animals, e.g., ['MC2586', 'MC2587']
trials_ds = list(np.arange(9, 29))          # List of trials, e.g., [1, 2, 3] or range(1, 10)

# --- Tracking Parameters ---
confidence = 0.9      # DLC confidence threshold (0-1)
floor_offset = 4      # Offset added to Z signal to set floor at zero
paws = ['FR', 'FL']   # Paws to analyze: 'FR', 'FL', 'HR', 'HL'

# --- Analysis Options ---
manual_track = False  # If True, load and compare with manual tracking data

# --- Visualization ---
show_plots = True     # If True, display interactive plots; if False, only save HTML

# --- Video Overlay Options ---
overlay_manual_on_video = False                 # Overlay manual track points on video
overlay_signal_choice = 'filtered'              # 'manual' | 'filtered' | 'corrected'
overlay_paw = 'FR'                              # Paw for filtered/corrected overlay
overlay_tag = 'filtered'                        # 'dlc raw' | 'filtered' | 'corrected'

# =============================================================================
# CONSTANTS
# =============================================================================

PAWS_INDEX = {'FR': 0, 'HR': 1, 'FL': 2, 'HL': 3}
paws_ind = PAWS_INDEX  # Backward compatibility alias

# Pixel-to-mm conversion (depends on setup type)
PIXEL_TO_MM_REALTIME = 1 / 3.3
PIXEL_TO_MM_WT = 1 / 1.955
pixel_to_mm = PIXEL_TO_MM_WT if 'WT' in path else PIXEL_TO_MM_REALTIME

# =============================================================================
# INITIALIZATION
# =============================================================================

loco = locomotion_class.loco_class(path, pixel_to_mm=pixel_to_mm)

animal_session_list = loco.animals_within_session()
animals_ds = []
for a in range(len(animal_session_list)):
    animals_ds.append(animal_session_list[a][0])
if path_retracked:
    loco_retracked = locomotion_class.loco_class(path_retracked)
    filelist_retracked = loco_retracked.get_track_files(animal, session)

dt = 1 / loco.sr  # Time step in seconds (e.g., 0.003s for 333Hz)

filelist = loco.get_track_files(animal, session)

# Load manual tracking data if enabled
if manual_track:
    file_path_FRbottom = os.path.join(path, 'MC16848_159_22_0.275_0.275_tied_1_1_FRbottom_points.npy')
    manual_track_FRbottom = np.load(file_path_FRbottom)
    file_path_STonset = os.path.join(path, 'MC16848_159_22_0.275_0.275_tied_1_1_STonset_points.npy')
    manual_track_STonset = np.load(file_path_STonset)

start_trial = int(filelist[0].split('DLC')[0].split('_')[-1])
current_filelist = filelist[trial-start_trial:trial-start_trial+2]
count_trial = int(current_filelist[0].split('DLC')[0].split('_')[-1])
f = filelist[trial-1]

[final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, confidence, 0)
[st_strides_mat_w11, sw_pts_mat_w11, st_strides_mat_all_w11, sw_pts_mat_all_w11] = loco.get_sw_st_matrices(final_tracks, 1, wl=11, return_all=True)
[st_strides_mat_w5, sw_pts_mat_w5, st_strides_mat_all_w5, sw_pts_mat_all_w5] = loco.get_sw_st_matrices(final_tracks, 1, wl=5, return_all=True)
paws_rel = {'x': loco.get_paws_rel(final_tracks, 'X'), 'y': loco.get_paws_rel(final_tracks, 'Y'), 'z': loco.get_paws_rel(final_tracks, 'Z')}

if path_retracked:
    f_retracked = filelist_retracked[filelist.index(f)]
    [final_tracks_retracked, tracks_tail_retracked, joints_wrist_retracked, joints_elbow_retracked, ear_retracked, bodycenter_retracked] = loco_retracked.read_h5(f_retracked, 0.9, 0)
    paws_rel_retracked = {'x': loco_retracked.get_paws_rel(final_tracks_retracked, 'X'), 
                            'y': loco_retracked.get_paws_rel(final_tracks_retracked, 'Y'), 
                            'z': loco_retracked.get_paws_rel(final_tracks_retracked, 'Z')}

# =============================================================================
# PRECOMPUTED VARIABLES - Computed once, used throughout the script
# =============================================================================

# --- Raw position arrays (all paws) ---
X_raw = final_tracks[0, :, :] * loco.pixel_to_mm
Y_raw = final_tracks[1, :, :] * loco.pixel_to_mm
Z_raw = -final_tracks[3, :, :] * loco.pixel_to_mm  # Inverted for visualization

# --- Interpolated position arrays (linear interpolation of NaNs) ---
X_interp = loco.inpaint_nans(deepcopy(X_raw))
Y_interp = loco.inpaint_nans(deepcopy(Y_raw))
Z_interp = loco.inpaint_nans(deepcopy(Z_raw))

# --- Interpolated with cubic spline ---
X_interp_spline = loco.inpaint_nans_cubic_spline(deepcopy(X_raw))
Y_interp_spline = loco.inpaint_nans_cubic_spline(deepcopy(Y_raw))
Z_interp_spline = loco.inpaint_nans_cubic_spline(deepcopy(Z_raw))

# --- Per-paw precomputed data ---
paw_data = {}
for paw in paws:
    pidx = paws_ind[paw]
    
    # Filtered signals
    filt_w5 = savgol_filter(X_interp[pidx, :], window_length=5, polyorder=1)
    filt_w11 = savgol_filter(X_interp[pidx, :], window_length=11, polyorder=1)
    filt_w5_raw = savgol_filter(X_raw[pidx, :], window_length=5, polyorder=1)
    filt_w11_raw = savgol_filter(X_raw[pidx, :], window_length=11, polyorder=1)
    
    # Peak detection (stance/swing)
    peaks_w5 = find_peaks(filt_w5)
    troughs_w5 = find_peaks(-filt_w5)
    stance_frames = peaks_w5[0]
    swing_frames = troughs_w5[0]
    
    # Stride matrices extraction (w11)
    if st_strides_mat_w11 is not None and sw_pts_mat_w11 is not None:
        st_raw_w11 = np.asarray(st_strides_mat_w11[pidx][:, 0, -1])
        sw_raw_w11 = np.asarray(sw_pts_mat_w11[pidx][:, 0, -1])
        st_all_raw_w11 = np.asarray(st_strides_mat_all_w11[pidx][:, 0, -1])
        sw_all_raw_w11 = np.asarray(sw_pts_mat_all_w11[pidx][:, 0, -1])
        
        st_strides_w11 = st_raw_w11[np.isfinite(st_raw_w11)].astype(np.int64)
        sw_pts_w11 = sw_raw_w11[np.isfinite(sw_raw_w11)].astype(np.int64)
        st_strides_all_w11 = st_all_raw_w11[np.isfinite(st_all_raw_w11)].astype(np.int64)
        sw_pts_all_w11 = sw_all_raw_w11[np.isfinite(sw_all_raw_w11)].astype(np.int64)
    else:
        st_strides_w11 = sw_pts_w11 = st_strides_all_w11 = sw_pts_all_w11 = np.array([], dtype=np.int64)
    
    # Stride matrices extraction (w5)
    if st_strides_mat_w5 is not None and sw_pts_mat_w5 is not None:
        st_raw_w5 = np.asarray(st_strides_mat_w5[pidx][:, 0, -1])
        sw_raw_w5 = np.asarray(sw_pts_mat_w5[pidx][:, 0, -1])
        st_all_raw_w5 = np.asarray(st_strides_mat_all_w5[pidx][:, 0, -1])
        sw_all_raw_w5 = np.asarray(sw_pts_mat_all_w5[pidx][:, 0, -1])
        
        st_strides_w5 = st_raw_w5[np.isfinite(st_raw_w5)].astype(np.int64)
        sw_pts_w5 = sw_raw_w5[np.isfinite(sw_raw_w5)].astype(np.int64)
        st_strides_all_w5 = st_all_raw_w5[np.isfinite(st_all_raw_w5)].astype(np.int64)
        sw_pts_all_w5 = sw_all_raw_w5[np.isfinite(sw_all_raw_w5)].astype(np.int64)
    else:
        st_strides_w5 = sw_pts_w5 = st_strides_all_w5 = sw_pts_all_w5 = np.array([], dtype=np.int64)
    
    paw_data[paw] = {
        'idx': pidx,
        'filt_w5': filt_w5,
        'filt_w11': filt_w11,
        'filt_w5_raw': filt_w5_raw,
        'filt_w11_raw': filt_w11_raw,
        'stance_frames': stance_frames,
        'swing_frames': swing_frames,
        'st_strides_w11': st_strides_w11,
        'sw_pts_w11': sw_pts_w11,
        'st_strides_all_w11': st_strides_all_w11,
        'sw_pts_all_w11': sw_pts_all_w11,
        'st_strides_w5': st_strides_w5,
        'sw_pts_w5': sw_pts_w5,
        'st_strides_all_w5': st_strides_all_w5,
        'sw_pts_all_w5': sw_pts_all_w5,
    }

# --- Retracked data (if available) ---
if path_retracked:
    X_retracked = final_tracks_retracked[0, :, :] * loco.pixel_to_mm
    Y_retracked = final_tracks_retracked[1, :, :] * loco.pixel_to_mm
    Z_retracked = -final_tracks_retracked[3, :, :] * loco.pixel_to_mm
    X_interp_retracked = loco_retracked.inpaint_nans(deepcopy(X_retracked))
    Y_interp_retracked = loco_retracked.inpaint_nans(deepcopy(Y_retracked))
    Z_interp_retracked = loco_retracked.inpaint_nans(deepcopy(Z_retracked))

# =============================================================================
# PER-PAW ANALYSIS LOOP
# =============================================================================
    
for p in range(len(paws)):
    paw = paws[p]
    pidx = paw_data[paw]['idx']
    
    # Get precomputed data for this paw
    data_filt = paw_data[paw]['filt_w5']
    data_filt_w11 = paw_data[paw]['filt_w11_raw']
    data_filt_w5 = paw_data[paw]['filt_w5_raw']
    stance = paw_data[paw]['stance_frames']
    swing = paw_data[paw]['swing_frames']
    st_strides = paw_data[paw]['st_strides_w11']
    sw_pts = paw_data[paw]['sw_pts_w11']
    st_strides_all = paw_data[paw]['st_strides_all_w11']
    sw_pts_all = paw_data[paw]['sw_pts_all_w11']
    
    # Represent all signals in a plotly figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # --- Manual tracking signals (if enabled) ---
    if manual_track:
        add_line_trace(fig, manual_track_FRbottom[:, 0] * loco.pixel_to_mm, f'{paw} X bottom manual track', 'blue')
        X_manual_corrected = deepcopy(X_raw[pidx, :])
        ind_manual_track = np.where(~np.isnan(manual_track_FRbottom[:, 0]))[0]
        X_manual_corrected[ind_manual_track] = manual_track_FRbottom[ind_manual_track, 0] * loco.pixel_to_mm
        X_interp_manual_corrected = loco.inpaint_nans(X_manual_corrected)
        add_line_trace(fig, X_interp_manual_corrected, f'{paw} X bottom manual track corrected', 'purple')
    else:
        X_interp_manual_corrected = X_raw[pidx, :]
    
    # --- Raw and interpolated signals ---
    add_line_trace(fig, X_raw[pidx, :], f'{paw} Xraw', 'gray')
    if manual_track:
        print_rmse_stats("raw signal", X_raw[pidx, :], manual_track_FRbottom[:, 0] * loco.pixel_to_mm, compute_abs_area_between_signals)
    
    add_line_trace(fig, X_interp[pidx, :], f'{paw} Xraw_interp', 'black')
    add_line_trace(fig, X_interp_spline[pidx, :], f'{paw} Xraw_interp_spline', 'green')
    if manual_track:
        print_rmse_stats("raw signal with cubic spline interpolation of NaN", X_interp_spline[pidx, 600:], manual_track_FRbottom[600:, 0] * loco.pixel_to_mm, compute_abs_area_between_signals)

    # --- Retracked signals (if available) ---
    if path_retracked:
        add_line_trace(fig, X_retracked[pidx, :], f'{paw} Xraw retracked', 'orange')
        if manual_track:
            print_rmse_stats("retracked raw signal", X_retracked[pidx, :], manual_track_FRbottom[:, 0] * loco.pixel_to_mm, compute_abs_area_between_signals)
        add_line_trace(fig, X_interp_retracked[pidx, :], f'{paw} Xraw_interp retracked', 'darkorange')
    
    # VALIDATED: the manual tracking is correct and matches the DLC tracks, so we can use it to compare with the automatic tracking
    #fig_validation = go.Figure()
    #fig_validation.add_trace(go.Scatter(y=manual_track_FRbottom_validation[:,0]*loco.pixel_to_mm, name=paws[p]+' X bottom validation track', line=dict(color='green')))
    #fig_validation.add_trace(go.Scatter(y=X[p,:], name=paws[p]+' Xraw', line=dict(color='gray')))
    #fig_validation.show()
    # --- Filtered signals and error metrics ---
    add_line_trace(fig, data_filt, 'X_filtered_swst w5', '#FFB6C1')
    if manual_track:
        print_rmse_stats("method 1 (savgol filter, window 5, order 1)", X_interp_manual_corrected, data_filt, compute_abs_area_between_signals)

    add_line_trace(fig, data_filt_w11, 'X_filtered w11', 'red')
    if manual_track:
        print_rmse_stats("method 2 (savgol filter, window 11, order 1)", X_interp_manual_corrected, paw_data[paw]['filt_w11'], compute_abs_area_between_signals)

    # --- Retracked filtered signals (if available) ---
    if path_retracked:
        filt_retracked_w5 = savgol_filter(X_interp_retracked[pidx,:], window_length=5, polyorder=1)
        add_line_trace(fig, filt_retracked_w5, 'X_retracked_filtered ord1', 'pink')
        print_rmse_stats("method 1 (retracked savgol filter, window 5, order 1)", X_interp_manual_corrected, filt_retracked_w5, compute_abs_area_between_signals)

        filt_retracked_w11 = savgol_filter(X_retracked[pidx,:], window_length=11, polyorder=1)
        add_line_trace(fig, filt_retracked_w11, 'X_retracked_filtered_swst ord1', 'pink')
        print_rmse_stats("method 2 (retracked savgol filter, window 11, order 1)", X_interp_manual_corrected, filt_retracked_w11, compute_abs_area_between_signals)

    # --- Stance/Swing detection markers from peak finding ---
    add_marker_trace(fig, stance, data_filt[stance], 'Stance Onset peak w5', 'square', 'orange')
    add_marker_trace(fig, swing, data_filt[swing], 'Swing Onset trough w5', 'square', 'green')

    # --- Stride markers from matrices (w11) ---
    if st_strides_mat_w11 is not None and sw_pts_mat_w11 is not None:
        print(f"Number of stance onsets w11: before exclusion={len(st_strides_all)}, after exclusion={len(st_strides)}")
        add_stride_markers(fig, data_filt_w11, st_strides_all, sw_pts_all, st_strides, sw_pts,
                          '', 'w11', 'orange', 'green', 'circle-open', 'circle')

    # --- Stride markers from matrices (w5) ---
    if st_strides_mat_w5 is not None and sw_pts_mat_w5 is not None:
        # Extract w5 stride indices using helper function
        st_strides_w5 = extract_stride_indices(st_strides_mat_w5, pidx)
        sw_pts_w5 = extract_stride_indices(sw_pts_mat_w5, pidx)
        st_strides_all_w5 = extract_stride_indices(st_strides_mat_all_w5, pidx)
        sw_pts_all_w5 = extract_stride_indices(sw_pts_mat_all_w5, pidx)
        
        add_stride_markers(fig, data_filt_w5, st_strides_all_w5, sw_pts_all_w5, st_strides_w5, sw_pts_w5,
                          '', 'w5', 'magenta', 'darkgreen', 'diamond-open', 'diamond')

    # --- Spline-based markers (same as stance/swing since computed on same data_filt) ---
    add_marker_trace(fig, stance, data_filt[stance], 'Stance Onset on filtered interp spline', 'star-open', 'orange', size=10)
    add_marker_trace(fig, swing, data_filt[swing], 'Swing Onset on filtered interp spline', 'star-open', 'green', size=10)

    # --- Additional filtering methods for comparison ---
    filt_w50 = savgol_filter(X_interp[pidx,:], window_length=50, polyorder=3)
    add_line_trace(fig, filt_w50, 'X_filtered_swst w50 ord3', 'lightblue')
    if manual_track:
        print_rmse_stats("method 3 (savgol filter, window 50, order 3)", X_interp_manual_corrected, filt_w50, compute_abs_area_between_signals)

    medfilt_result = medfilt(X_interp[pidx,:], kernel_size=5)
    add_line_trace(fig, medfilt_result, 'X_filtered_medianfil', 'darkblue')
    if manual_track:
        print_rmse_stats("method 4 (median filter, kernel size 5)", X_interp_manual_corrected, medfilt_result, compute_abs_area_between_signals)

    # --- Method 5: outlier detection and interpolation with cubic spline ---
    x_speed_raw, x_acceleration_raw, x_without_outliers, corrected_x = correct_signal_cubic_spline(X_interp[pidx,:], dt, thr_sig='speed')
    y_speed_raw, y_acceleration_raw, y_without_outliers, corrected_y = correct_signal_cubic_spline(Y_interp[pidx,:], dt, thr_sig='speed')
    z_speed_raw, z_acceleration_raw, z_without_outliers, corrected_z = correct_signal_cubic_spline(Z_interp[pidx,:], dt, thr_sig='speed')

    add_line_trace(fig, x_without_outliers, f'{paw} Xraw with removed outliers speed/acceleration', 'salmon')
    if manual_track:
        print_rmse_stats("method 5 (removal of outliers)", X_interp_manual_corrected, x_without_outliers, compute_abs_area_between_signals)
    
    add_line_trace(fig, corrected_x, f'{paw} X_corrected spline', '#8B0000')
    if manual_track:
        print_rmse_stats("method 6 (cubic spline interpolation)", X_interp_manual_corrected[100:], corrected_x[100:], compute_abs_area_between_signals)

    # Speed and acceleration signals on secondary axis
    add_line_trace(fig, x_speed_raw, f'{paw} X speed raw', 'green', secondary_y=True)
    add_line_trace(fig, x_acceleration_raw, f'{paw} X acceleration raw', 'lightgreen', secondary_y=True)

    # --- Manual stance onset points (if available) ---
    if manual_track:
        st_arr = np.array(manual_track_STonset, dtype=float)
        valid_mask = ~np.isnan(st_arr).all(axis=1)
        manual_stance_onset_frames = np.where(valid_mask)[0]
        manual_stance_onset_x = st_arr[manual_stance_onset_frames, 0] * loco.pixel_to_mm
        if manual_stance_onset_frames.size > 0:
            print(f"Manual stance onset points: {manual_stance_onset_frames.size} (frames {manual_stance_onset_frames[0]} to {manual_stance_onset_frames[-1]})")
        else:
            print("Manual stance onset points: none found")
        add_marker_trace(fig, manual_stance_onset_frames, manual_stance_onset_x, 'Manual Stance Onset', 'x', 'black', size=6)

    # --- Figure layout and export ---
    fig.update_layout(
        title=f"{animal} trial {count_trial} - {paw} Tracking X Signals confidence {confidence}",
        xaxis_title='Frame',
        yaxis_title='Position (mm)',
        xaxis=dict(autorange=True),
        yaxis=dict(autorange=True),
        showlegend=True
    )
    
    if show_plots:
        fig.show()

    # Save as HTML
    fig.write_html(os.path.join(loco.path, f"{animal}_{paw}_trial{count_trial}_tracking_Xsignals.html"))

    # Preserve tracks for overlay
    if paw == overlay_paw:
        raw_track = deepcopy(X_raw[pidx, :])
        filtered_x_for_overlay = deepcopy(data_filt)
        corrected_x_for_overlay = deepcopy(X_interp_spline[pidx,:]) if 'corrected_x' in locals() else None

    # =========================================================================
    # FIG2: Position and Speed Signals
    # =========================================================================
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Filtered position signals
    x_filt = data_filt
    y_filt = savgol_filter(Y_interp[pidx,:], window_length=5, polyorder=1)
    z_filt = savgol_filter(Z_interp[pidx,:], window_length=5, polyorder=1)
    rel_z_filt = z_filt + loco.floor
    
    # Speed signals
    speed_x = np.diff(x_filt, prepend=x_filt[0])
    speed_y = np.diff(y_filt, prepend=y_filt[0])
    speed_z = np.diff(z_filt, prepend=z_filt[0])
    speed_xz = np.sqrt(speed_x**2 + speed_z**2)
    speed_xyz = np.sqrt(speed_x**2 + speed_y**2 + speed_z**2)
    
    # Add position traces
    add_line_trace(fig2, X_interp[pidx,:], f'{paw} Xraw_interp', 'black')
    add_line_trace(fig2, data_filt, 'X_filtered_swst w5', '#FFB6C1')
    add_line_trace(fig2, savgol_filter(Z_interp[pidx,:], window_length=5, polyorder=1), 'Z_filtered w5', '#209ECF')
    add_line_trace(fig2, rel_z_filt, 'Z relative', '#209ECF', dash='dot')
    
    # Add speed traces on secondary axis
    add_line_trace(fig2, speed_x, f'{paw} X speed', 'purple', secondary_y=True)
    add_line_trace(fig2, speed_z, f'{paw} Z speed', 'green', secondary_y=True)
    add_line_trace(fig2, speed_xz, f'{paw} XZ speed', 'darkblue', secondary_y=True)
    add_line_trace(fig2, speed_xyz, f'{paw} XYZ speed', 'orange', secondary_y=True)
    
    fig2.update_yaxes(title_text="Speed (mm/s)", secondary_y=True)
    fig2.add_hline(y=-loco.floor, line=dict(color='black', dash='dash'), name='Floor level')
    
    # Stance/Swing markers
    add_marker_trace(fig2, stance, x_filt[stance], 'Stance Onset peak', 'circle-open', 'orange')
    add_marker_trace(fig2, swing, x_filt[swing], 'Swing Onset trough', 'circle-open', 'green')

    # Prepare indices for vertical lines
    rel_z_filt = rel_z_filt - floor_offset
    s_z = np.sign(rel_z_filt)
    s_z[~np.isfinite(s_z)] = 0
    zc_idx = np.where((s_z[:-1] * s_z[1:]) < 0)[0] + 1
    
    speed_diff = speed_x - speed_z
    equal_zero = np.where(np.isfinite(speed_diff) & (np.abs(speed_diff) <= 1e-6))[0]
    s_v = np.sign(speed_diff)
    s_v[~np.isfinite(s_v)] = 0
    eq_cross = np.where((s_v[:-1] * s_v[1:]) < 0)[0] + 1
    eq_idx = np.unique(np.concatenate([equal_zero, eq_cross]))
    
    # Compute axis ranges using helper function
    pos_max_abs = compute_max_abs_safe([x_filt, rel_z_filt])
    speed_max_abs = compute_max_abs_safe([speed_x, speed_z, speed_xz, speed_xyz])
    pos_margin = 1.05
    speed_margin = 1.05

    # Loop over automatic stance onset frames from valid strides (after exclusion)
    n_stance = st_strides.size
    if n_stance > 0:
        closest_zc = np.full(n_stance, np.nan, dtype=float)
        closest_eq = np.full(n_stance, np.nan, dtype=float)
        dist_zc = np.full(n_stance, np.nan, dtype=float)
        dist_eq = np.full(n_stance, np.nan, dtype=float)
        # Distance from automatic stance peak to nearest manual stance onset (initialized later)
        dist_st_to_manual = np.full(n_stance, np.nan, dtype=float)
        # Distances from matched zcross / speed equality to nearest manual stance onset
        dist_zc_to_manual = np.full(n_stance, np.nan, dtype=float)
        dist_eq_to_manual = np.full(n_stance, np.nan, dtype=float)
        # Prepare manual stance frames if available
        if manual_track and 'manual_stance_onset_frames' in locals() and manual_stance_onset_frames.size > 0:
            manual_frames_sorted = np.sort(manual_stance_onset_frames.astype(int))
            def _nearest_manual(frame):
                """Return distance (signed?) We'll use absolute distance only for summaries. Provide signed difference to closest manual (manual - frame)."""
                # binary search for insertion point
                ip = np.searchsorted(manual_frames_sorted, frame, side='left')
                candidates = []
                if ip < manual_frames_sorted.size:
                    candidates.append(manual_frames_sorted[ip])
                if ip > 0:
                    candidates.append(manual_frames_sorted[ip-1])
                if not candidates:
                    return np.nan, np.nan
                # choose candidate with minimum absolute distance
                cand_arr = np.array(candidates)
                abs_d = np.abs(cand_arr - frame)
                best_idx = int(np.argmin(abs_d))
                best_frame = int(cand_arr[best_idx])
                signed = best_frame - frame
                return signed, abs_d[best_idx]
        else:
            manual_frames_sorted = np.array([], dtype=int)
            def _nearest_manual(frame):
                return np.nan, np.nan
        for i, stf in enumerate(st_strides.astype(int)):
            # choose the first zero-crossing at or after the stance frame
            if zc_idx.size:
                ip = np.searchsorted(zc_idx, stf, side='left')
                if ip < zc_idx.size:
                    cand = int(zc_idx[ip])
                    closest_zc[i] = cand
                    dist_zc[i] = cand - stf
            # choose the first speed-equality at or after the stance frame
            if eq_idx.size:
                ip2 = np.searchsorted(eq_idx, stf, side='left')
                if ip2 < eq_idx.size:
                    cand2 = int(eq_idx[ip2])
                    closest_eq[i] = cand2
                    dist_eq[i] = cand2 - stf
            # Distance from stance peak to nearest manual stance onset
            signed_d, abs_d = _nearest_manual(stf)
            dist_st_to_manual[i] = signed_d  # signed difference (manual - stance)
            # Distances for matched events to nearest manual stance
            if np.isfinite(closest_zc[i]):
                signed_zc, abs_zc = _nearest_manual(int(closest_zc[i]))
                dist_zc_to_manual[i] = signed_zc
            if np.isfinite(closest_eq[i]):
                signed_eq, abs_eq = _nearest_manual(int(closest_eq[i]))
                dist_eq_to_manual[i] = signed_eq
        # Plot vertical lines for closest zero-crossings and speed equalities using helper
        y_range = pos_max_abs * pos_margin
        valid_zc = np.isfinite(closest_zc)
        if np.any(valid_zc):
            add_vertical_lines(fig2, closest_zc[valid_zc].astype(int), -y_range, y_range, 'ST Zcross', 'red')
        valid_eq = np.isfinite(closest_eq)
        if np.any(valid_eq):
            add_vertical_lines(fig2, closest_eq[valid_eq].astype(int), -y_range, y_range, 'ST XZSpeedEq', 'darkblue', dash='dot')
        
        print(
            f"[INFO] stance_peaks={n_stance} zc_found={valid_zc.sum()} eq_found={valid_eq.sum()} | "
            f"zc_dist mean={np.nanmean(dist_zc):.2f} median={np.nanmedian(dist_zc):.2f}; "
            f"eq_dist mean={np.nanmean(dist_eq):.2f} median={np.nanmedian(dist_eq):.2f}"
        )
        if manual_frames_sorted.size > 0:
            print(
                f"[INFO] Distances to manual stance (signed manual - event frame): "
                f"stance->manual mean={np.nanmean(dist_st_to_manual):.2f} median={np.nanmedian(dist_st_to_manual):.2f}; "
                f"zc->manual mean={np.nanmean(dist_zc_to_manual):.2f} median={np.nanmedian(dist_zc_to_manual):.2f}; "
                f"eq->manual mean={np.nanmean(dist_eq_to_manual):.2f} median={np.nanmedian(dist_eq_to_manual):.2f}"
            )
        else:
            print("[INFO] No manual stance frames available for distance comparison.")
    else:
        print("[INFO] No automatic stance peaks available for event alignment.")
    
    # Constrain y-axis ranges for zoomed view
    x_min_view, x_max_view = 5000, 5300
    x_min_idx = max(0, int(x_min_view))
    x_max_idx = min(x_filt.size-1, int(x_max_view))
    
    # Compute ranges on visible window using helper
    pos_max_abs_window = compute_max_abs_safe([x_filt[x_min_idx:x_max_idx+1], rel_z_filt[x_min_idx:x_max_idx+1]], default=pos_max_abs)
    speed_max_abs_window = compute_max_abs_safe([
        speed_x[x_min_idx:x_max_idx+1], speed_z[x_min_idx:x_max_idx+1],
        speed_xz[x_min_idx:x_max_idx+1], speed_xyz[x_min_idx:x_max_idx+1]
    ], default=speed_max_abs)
    
    fig2.update_layout(
        title=f"{animal} trial {count_trial} - {paw} Relative position and speed signals",
        xaxis=dict(range=[x_min_view, x_max_view], title='Frame'),
        yaxis=dict(range=[-pos_max_abs_window*pos_margin, pos_max_abs_window*pos_margin], title="Position (mm)", zeroline=True, zerolinecolor='rgba(0,0,0,0.3)'),
        yaxis2=dict(range=[-speed_max_abs_window*speed_margin, speed_max_abs_window*speed_margin], title="Speed (mm/s)", zeroline=True, zerolinecolor='rgba(0,0,0,0.3)'),
        showlegend=True
    )

    if manual_track:
        manual_stance_onset_x_rel = st_arr[manual_stance_onset_frames, 0] * loco.pixel_to_mm
        add_marker_trace(fig2, manual_stance_onset_frames, manual_stance_onset_x_rel, 'Manual Stance Onset', 'x', 'black', size=10)
    
    if show_plots:
        fig2.show()
    fig2.write_html(os.path.join(loco.path, f"{animal}_{paw}_trial{count_trial}_zoomed_pos_speed_signals.html"))

    # Build a single figure with subplots for multiple strides (near-square layout)
    # Define s_start as first stride whose starting frame (swing onset frame sw_pts[s]) is >= x_min_view
    # Fallback to 0 if none found.
    if 'sw_pts_mat_all_w5' in locals() and len(sw_pts_mat_all_w5[pidx]) > 1:
        # compare using last column (frame index) of sw_pts_mat_all for the current paw
        sw_all_frames = sw_pts_mat_all_w5[pidx][:, 0, -1]
        candidate_indices = np.where(sw_all_frames[:-1] >= x_min_view)[0]
        if candidate_indices.size > 0:
            s_start = int(candidate_indices[0])
        else:
            s_start = 0
        # choose a window of strides (e.g., next 6 strides) but ensure we don't exceed length-1 (since we index s+1)
        stride_window = 6
        s_end = min(s_start + stride_window, len(sw_pts) - 1)
    else:
        s_start, s_end = 0, 0
    # Build candidate stride indices and skip ones with NaN start/end frames.
    # Extend selection forward to maintain stride_window count of valid strides.
    s_list = []
    s_candidate = s_start
    while len(s_list) < stride_window and s_candidate < (len(sw_all_frames) - 1):
        if np.isfinite(sw_all_frames[s_candidate]) and np.isfinite(sw_all_frames[s_candidate+1]):
            s_list.append(s_candidate)
        s_candidate += 1
    n = len(s_list)
    if n > 0:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        fig3 = make_subplots(rows=rows, cols=cols, subplot_titles=[f"str start={int(sw_all_frames[s])}" for s in s_list])
        for idx, s in enumerate(s_list):
            r = idx // cols + 1
            c = idx % cols + 1
            # Main trajectory
            fig3.add_trace(
                go.Scatter(
                    x=x_filt[int(sw_all_frames[s]):int(sw_all_frames[s+1])],
                    y=z_filt[int(sw_all_frames[s]):int(sw_all_frames[s+1])],
                    name=paw+' XZ trajectory',
                    line=dict(color='red'),
                    showlegend=(idx == 0)
                ),
                row=r,
                col=c
            )
            # Velocity vectors
            scale = 1.0
            for t in range(int(sw_all_frames[s]), int(sw_all_frames[s+1])):
                fig3.add_trace(
                    go.Scatter(
                        x=[x_filt[t], x_filt[t] + speed_x[t]*scale],
                        y=[z_filt[t], z_filt[t] + speed_z[t]*scale],
                        mode='lines',
                        line=dict(color='blue'),
                        showlegend=False
                    ),
                    row=r,
                    col=c
                )
                fig3.add_trace(
                    go.Scatter(
                        x=[x_filt[t], x_filt[t] + speed_x[t]*scale],
                        y=[z_filt[t], z_filt[t]],
                        mode='lines',
                        line=dict(color='purple'),
                        showlegend=False
                    ),
                    row=r,
                    col=c
                )
                fig3.add_trace(
                    go.Scatter(
                        x=[x_filt[t], x_filt[t]],
                        y=[z_filt[t], z_filt[t] + speed_z[t]*scale],
                        mode='lines',
                        line=dict(color='green'),
                        showlegend=False
                    ),
                    row=r,
                    col=c
                )
            # Add markers for manual and automatic stance-related events within this stride window
            stride_start = int(sw_all_frames[s])
            stride_end = int(sw_all_frames[s+1]) - 1  # inclusive end frame
            # Manual stance onset frames in stride
            if manual_track and 'manual_stance_onset_frames' in locals():
                m_mask = (manual_stance_onset_frames >= stride_start) & (manual_stance_onset_frames <= stride_end)
                m_frames_stride = manual_stance_onset_frames[m_mask]
                if m_frames_stride.size:
                    fig3.add_trace(
                        go.Scatter(
                            x=x_filt[m_frames_stride],
                            y=z_filt[m_frames_stride],
                            mode='markers',
                            marker=dict(symbol='x', color='black', size=12),
                            name='Manual ST',
                            showlegend=(idx == 0)
                        ), row=r, col=c
                    )
            # Plot closest zero-crossing and speed equality matched to stance peak if inside stride
            if 'closest_zc' in locals():
                zc_in_stride = closest_zc[np.isfinite(closest_zc) & (closest_zc >= stride_start) & (closest_zc <= stride_end)]
                if zc_in_stride.size:
                    fig3.add_trace(
                        go.Scatter(
                            x=x_filt[zc_in_stride.astype(int)],
                            y=z_filt[zc_in_stride.astype(int)],
                            mode='markers',
                            marker=dict(symbol='diamond', color='red', size=10),
                            name='ST Zcross',
                            showlegend=(idx == 0)
                        ), row=r, col=c
                    )
            if 'closest_eq' in locals():
                eq_in_stride = closest_eq[np.isfinite(closest_eq) & (closest_eq >= stride_start) & (closest_eq <= stride_end)]
                if eq_in_stride.size:
                    fig3.add_trace(
                        go.Scatter(
                            x=x_filt[eq_in_stride.astype(int)],
                            y=z_filt[eq_in_stride.astype(int)],
                            mode='markers',
                            marker=dict(symbol='triangle-up', color='darkblue', size=8, line=dict(width=2, color='navy')),
                            name='ST SpeedEq',
                            showlegend=(idx == 0)
                        ), row=r, col=c
                    )
            # Automatic stance peaks inside stride
            stance_in_stride = stance[(stance >= stride_start) & (stance <= stride_end)]
            if stance_in_stride.size:
                fig3.add_trace(
                    go.Scatter(
                        x=x_filt[stance_in_stride],
                        y=z_filt[stance_in_stride],
                        mode='markers',
                        marker=dict(symbol='circle-open', color='orange', size=10, line=dict(width=2, color='orange')),
                        name='ST Peak',
                        showlegend=(idx == 0)
                    ), row=r, col=c
                )
        fig3.update_layout(
            title=animal + ' trial '+str(count_trial)+' - '+paw+' Trajectories with Velocity Vectors (grid)',
            showlegend=True
        )
        if show_plots:
            fig3.show()
        # Save as HTML combined grid
        fig3.write_html(os.path.join(loco.path, f"{animal}_{paw}_trial{count_trial}_trajectories_grid.html"))



    # Plot each stride separately    - TO VERIFY CONSISTENCY WITH kinematic_analysis.py code

# -------------------- Optional: overlay manual tracks on labeled video --------------------
if overlay_manual_on_video:
    if cv2 is None:
        print("[WARN] OpenCV (cv2) not available. Skipping manual overlay video export.")
    else:
        # Determine ORIGINAL video path from the .h5 filename `f`
        # Use the portion of the filename up to (and excluding) 'DLC' and assume .mp4 extension
        orig_stem = f.split('DLC')[0].rstrip('_-. ')
        source_video_path = os.path.join(path, orig_stem + '.mp4')

        if not os.path.exists(source_video_path):
            print(f"[WARN] Original video not found: {source_video_path}\n       Skipping overlay export.")
        else:
            # Output path next to the source video; suffix depends on overlay_tag
            _overlay_tag = overlay_tag.replace(' ', '_') if isinstance(overlay_tag, str) else 'filtered'
            out_video_base = os.path.splitext(source_video_path)[0] + f'_{_overlay_tag}_overlay'
            out_video_path = out_video_base + '.mp4'
            out_dir = os.path.dirname(out_video_path) or '.'
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                print(f"[WARN] Could not ensure output directory exists ('{out_dir}'): {e}")
            cap = cv2.VideoCapture(source_video_path)
            if not cap.isOpened():
                print(f"[WARN] Could not open video: {source_video_path}. Skipping overlay export.")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Prepare writer (mp4v works cross-platform in most cases)
                target_fps = fps if fps and fps > 0 else 30.0
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(out_video_path, fourcc, target_fps, (width, height))
                use_imageio = False
                if not writer.isOpened():
                    # Try imageio for MP4 if available (import lazily)
                    try:
                        _imageio = __import__('imageio')
                        # Prefer H.264 with good quality and broad compatibility
                        # Use CRF 18 (visually lossless-ish) and yuv420p pixel format
                        imageio_writer = _imageio.get_writer(
                            out_video_path,
                            fps=target_fps,
                            codec='libx264',
                            ffmpeg_params=['-crf','18','-pix_fmt','yuv420p']
                        )
                        use_imageio = True
                        print(f"[INFO] Using imageio/ffmpeg writer for MP4: {out_video_path}")
                    except Exception as e:
                        print(f"[WARN] imageio MP4 writer failed or not available: {e}")
                    if not use_imageio:
                        # Fallback to AVI with XVID via OpenCV
                        alt_out = os.path.splitext(out_video_path)[0] + '.avi'
                        alt_fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        writer = cv2.VideoWriter(alt_out, alt_fourcc, target_fps, (width, height))
                        if writer.isOpened():
                            print(f"[WARN] mp4v writer failed; using fallback AVI: {alt_out}")
                            out_video_path = alt_out
                        else:
                            print("[ERROR] Could not open any video writer (mp4v/XVID) and imageio unavailable. Aborting overlay export.")
                            cap.release()
                            writer.release()
                            writer = None
                if (not use_imageio and (writer is None or not writer.isOpened())):
                    # Could not proceed
                    print("[ERROR] Video writer not available. Skipping overlay export.")
                else:
                    # Manual arrays may contain NaNs; overlay only finite points
                    # Guard: ensure arrays exist
                    fr_points = 'manual_track_FRbottom' in locals()
                    st_points = 'manual_track_STonset' in locals()
                    max_idx = total_frames
                    if fr_points:
                        max_idx = min(max_idx, manual_track_FRbottom.shape[0])
                    if st_points:
                        max_idx = min(max_idx, manual_track_STonset.shape[0])
                    if max_idx == 0:
                        print("[WARN] No frames to process for overlay (no manual arrays or empty video).")
                    else:
                        print(f"[INFO] Writing overlay video: {out_video_path} ({max_idx} frames) | source={overlay_tag}")
                    # Precompute Y interpolation for the selected paw once (pixels)
                    try:
                        Y_full = final_tracks[1, :, :]
                        Y_interp_full = loco.inpaint_nans(Y_full)
                    except Exception:
                        Y_interp_full = None
                    paw_idx_video = paws_ind.get(overlay_paw, 0)

                    # Build quick-lookup sets for event frames within bounds
                    def _safe_set(arr, upper):
                        try:
                            a = np.asarray(arr)
                            if a.size == 0:
                                return set()
                            a = a[np.isfinite(a)]
                            a = a.astype(int)
                            a = a[(a >= 0) & (a < upper)]
                            return set(a.tolist())
                        except Exception:
                            return set()

                    stance_frames = _safe_set(stance if 'stance' in locals() else [], max_idx)
                    swing_frames = _safe_set(swing if 'swing' in locals() else [], max_idx)
                    zc_frames    = _safe_set(closest_zc if 'closest_zc' in locals() else [], max_idx)
                    eq_frames    = _safe_set(closest_eq if 'closest_eq' in locals() else [], max_idx)

                    frame_idx = 0
                    while frame_idx < max_idx:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # paw_idx_video defined above
                        # Always draw manual FRbottom as a blue dot
                        if fr_points and frame_idx < manual_track_FRbottom.shape[0]:
                            pt = manual_track_FRbottom[frame_idx]
                            if np.isfinite(pt).all():
                                x, y = int(round(pt[0])), int(round(pt[1]))
                                if 0 <= x < width and 0 <= y < height:
                                    cv2.circle(frame, (x, y), 3, (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                        # Draw ST onset manual point as white cross (width 2)
                        if st_points and frame_idx < manual_track_STonset.shape[0]:
                            pt2 = manual_track_STonset[frame_idx]
                            if np.isfinite(pt2).all():
                                x2, y2 = int(round(pt2[0])), int(round(pt2[1]))
                                if 0 <= x2 < width and 0 <= y2 < height:
                                    cv2.drawMarker(frame, (x2, y2), (255,255,255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)
                        # Overlay DLC track based on overlay_tag: raw (pixels), filtered/corrected (mm -> px)
                        if Y_interp_full is not None:
                            try:
                                y_pix = int(round(Y_interp_full[paw_idx_video, frame_idx]))
                            except Exception:
                                y_pix = None
                        else:
                            y_pix = None
                        if y_pix is not None and 0 <= y_pix < height:
                            if isinstance(overlay_tag, str) and overlay_tag.lower() == 'dlc raw':
                                if 'raw_track' in globals() and raw_track is not None and frame_idx < raw_track.size:
                                    if np.isfinite(raw_track[frame_idx]):
                                        x_pix = int(round(raw_track[frame_idx] / loco.pixel_to_mm))
                                        if 0 <= x_pix < width:
                                            # Gray empty circle (BGR)
                                            cv2.circle(frame, (x_pix, y_pix), 5, (128, 128, 128), thickness=1, lineType=cv2.LINE_AA)
                            elif isinstance(overlay_tag, str) and overlay_tag.lower() == 'filtered':
                                if 'filtered_x_for_overlay' in globals() and filtered_x_for_overlay is not None and frame_idx < filtered_x_for_overlay.size:
                                    x_mm_val = filtered_x_for_overlay[frame_idx]
                                    if np.isfinite(x_mm_val):
                                        x_pix = int(round(x_mm_val / loco.pixel_to_mm))
                                        if 0 <= x_pix < width:
                                            # Pink empty circle matching '#FFB6C1' (BGR approx (193,182,255))
                                            cv2.circle(frame, (x_pix, y_pix), 5, (193, 182, 255), thickness=1, lineType=cv2.LINE_AA)
                            elif isinstance(overlay_tag, str) and overlay_tag.lower() == 'corrected':
                                if 'corrected_x_for_overlay' in globals() and corrected_x_for_overlay is not None and frame_idx < corrected_x_for_overlay.size:
                                    x_mm_val = corrected_x_for_overlay[frame_idx]
                                    if np.isfinite(x_mm_val):
                                        x_pix = int(round(x_mm_val / loco.pixel_to_mm))
                                        if 0 <= x_pix < width:
                                            # Dark green empty circle (BGR approx)
                                            cv2.circle(frame, (x_pix, y_pix), 5, (0, 100, 0), thickness=1, lineType=cv2.LINE_AA)

                            # Also overlay event markers (empty circles) at the same paw position for this frame
                            # Determine x position consistent with overlay_tag for event highlighting
                            x_pix_event = None
                            if isinstance(overlay_tag, str) and overlay_tag.lower() == 'dlc raw':
                                if 'raw_track' in globals() and raw_track is not None and frame_idx < raw_track.size and np.isfinite(raw_track[frame_idx]):
                                    x_pix_event = int(round(raw_track[frame_idx] / loco.pixel_to_mm))
                            elif isinstance(overlay_tag, str) and overlay_tag.lower() == 'filtered':
                                if 'filtered_x_for_overlay' in globals() and filtered_x_for_overlay is not None and frame_idx < filtered_x_for_overlay.size:
                                    val = filtered_x_for_overlay[frame_idx]
                                    if np.isfinite(val):
                                        x_pix_event = int(round(val / loco.pixel_to_mm))
                            elif isinstance(overlay_tag, str) and overlay_tag.lower() == 'corrected':
                                if 'corrected_x_for_overlay' in globals() and corrected_x_for_overlay is not None and frame_idx < corrected_x_for_overlay.size:
                                    val = corrected_x_for_overlay[frame_idx]
                                    if np.isfinite(val):
                                        x_pix_event = int(round(val / loco.pixel_to_mm))

                            if x_pix_event is not None and 0 <= x_pix_event < width:
                                # Stance peak (yellow) and Swing trough (green)
                                if frame_idx in stance_frames:
                                    cv2.circle(frame, (x_pix_event, y_pix), 6, (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)  # yellow
                                if frame_idx in swing_frames:
                                    cv2.circle(frame, (x_pix_event, y_pix), 6, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)    # green
                                # Stance Z-crossing (red)
                                if frame_idx in zc_frames:
                                    cv2.circle(frame, (x_pix_event, y_pix), 7, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)    # red
                                # ST XZ-Speed equality (dark blue)
                                if frame_idx in eq_frames:
                                    cv2.circle(frame, (x_pix_event, y_pix), 7, (128, 0, 0), thickness=2, lineType=cv2.LINE_AA)    # dark blue
                                    
                        # Optional: overlay frame index
                        cv2.putText(frame, f"frame {frame_idx}", (12, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"frame {frame_idx}", (12, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

                        if use_imageio:
                            # Convert BGR (cv2) to RGB (imageio expects RGB)
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            imageio_writer.append_data(rgb)
                        else:
                            writer.write(frame)
                        frame_idx += 1

                    if use_imageio:
                        imageio_writer.close()
                    else:
                        writer.release()
                    cap.release()
                    # Verify output file was created and has content
                    try:
                        if os.path.exists(out_video_path):
                            size_bytes = os.path.getsize(out_video_path)
                            if size_bytes > 0:
                                print(f"[INFO] Overlay video saved: {out_video_path} ({size_bytes} bytes)")
                                # Try reopening to ensure it's readable
                                vtest = cv2.VideoCapture(out_video_path)
                                if vtest.isOpened():
                                    ok, _ = vtest.read()
                                    if ok:
                                        print("[INFO] Overlay video verified: first frame readable.")
                                    else:
                                        print("[WARN] Overlay video opened but first frame not readable.")
                                    vtest.release()
                                else:
                                    print("[WARN] Overlay video could not be opened for verification.")
                            else:
                                print(f"[WARN] Overlay video file exists but is empty: {out_video_path}")
                        else:
                            print(f"[WARN] Overlay video was not found after writing: {out_video_path}")
                    except Exception as e:
                        print(f"[WARN] Could not verify overlay video file: {e}")


# =============================================================================
# COMBINED FIGURE - All paws X tracking signals
# =============================================================================

combined_fig = go.Figure()

for p in range(len(paws)):
    paw = paws[p]
    pidx = paw_data[paw]['idx']
    
    # Use precomputed data
    data_filt = paw_data[paw]['filt_w5']
    data_filt_w11 = paw_data[paw]['filt_w11_raw']
    data_filt_w5 = paw_data[paw]['filt_w5_raw']
    stance = paw_data[paw]['stance_frames']
    swing = paw_data[paw]['swing_frames']
    st_strides = paw_data[paw]['st_strides_w11']
    sw_pts = paw_data[paw]['sw_pts_w11']
    st_strides_all = paw_data[paw]['st_strides_all_w11']
    sw_pts_all = paw_data[paw]['sw_pts_all_w11']
    
    # Raw and interpolated signals
    add_line_trace(combined_fig, X_raw[pidx,:], f'{paw} Xraw', 'gray')
    add_line_trace(combined_fig, X_interp[pidx,:], f'{paw} Xraw_interp', 'black')
    add_line_trace(combined_fig, X_interp_spline[pidx,:], f'{paw} Xraw_interp_spline', 'green')
    
    # Filtered signals
    add_line_trace(combined_fig, data_filt, f'{paw} X_filtered_swst w5', '#FFB6C1')
    add_line_trace(combined_fig, paw_data[paw]['filt_w11'], f'{paw} X_filtered w11', 'red')
    
    # Stance and swing detection markers
    add_marker_trace(combined_fig, stance, data_filt[stance], f'{paw} ST peak w5', 'square', 'orange')
    add_marker_trace(combined_fig, swing, data_filt[swing], f'{paw} SW trough w5', 'square', 'green')
    
    # Stride matrix markers (w11)
    if st_strides_mat_w11 is not None and sw_pts_mat_w11 is not None:
        add_stride_markers(combined_fig, data_filt_w11, st_strides_all, sw_pts_all, st_strides, sw_pts,
                          paw, 'w11', 'orange', 'green', 'circle-open', 'circle')
    
    # Stride matrix markers (w5)
    if st_strides_mat_w5 is not None and sw_pts_mat_w5 is not None:
        st_strides_w5 = extract_stride_indices(st_strides_mat_w5, pidx)
        sw_pts_w5 = extract_stride_indices(sw_pts_mat_w5, pidx)
        st_strides_all_w5 = extract_stride_indices(st_strides_mat_all_w5, pidx)
        sw_pts_all_w5 = extract_stride_indices(sw_pts_mat_all_w5, pidx)
        
        add_stride_markers(combined_fig, data_filt_w5, st_strides_all_w5, sw_pts_all_w5, st_strides_w5, sw_pts_w5,
                          paw, 'w5', 'magenta', 'darkgreen', 'diamond-open', 'diamond')
    
    # Spline-based markers
    add_marker_trace(combined_fig, stance, data_filt[stance], f'{paw} ST on filt spline', 'star-open', 'orange', size=10)
    add_marker_trace(combined_fig, swing, data_filt[swing], f'{paw} SW on filt spline', 'star-open', 'green', size=10)

combined_fig.update_layout(
    title=f"{animal} trial {count_trial} - All paws Tracking X Signals",
    xaxis_title='Frame',
    yaxis_title='Position (mm)',
    xaxis=dict(autorange=True),
    yaxis=dict(autorange=True),
    showlegend=True
)

if show_plots:
    combined_fig.show()
combined_fig.write_html(os.path.join(loco.path, f"{animal}_trial{count_trial}_allpaws_tracking_Xsignals.html"))


# =============================================================================
# DOUBLE SUPPORT ANALYSIS - Multi-Animal/Multi-Trial Batch Analysis
# =============================================================================

def analyze_double_support_trial(loco_obj, animal_id, trial_num, paws_to_analyze, session_num=1, conf=0.9, save_plot=True, show_plot=False):
    """
    Analyze double support for a single trial.
    
    Args:
        loco_obj: locomotion_class instance
        animal_id: animal identifier string
        trial_num: trial number
        paws_to_analyze: list of paw names to analyze, e.g. ['FR', 'FL']
        session_num: session number (default 1)
        conf: confidence threshold (default 0.9)
        save_plot: if True, save HTML plot for this trial
        show_plot: if True, display the plot interactively
    
    Returns:
        dict with analysis results, or None if failed
    """
    try:
        filelist = loco_obj.get_track_files(animal_id, session_num)
        start_trial = int(filelist[0].split('DLC')[0].split('_')[-1])
        f = filelist[trial_num - start_trial]
        
        [final_tracks_t, tracks_tail_t, joints_wrist_t, joints_elbow_t, ear_t, bodycenter_t] = loco_obj.read_h5(f, conf, 0)
        [st_mat_w11, sw_mat_w11, st_mat_all_w11, sw_mat_all_w11] = loco_obj.get_sw_st_matrices(final_tracks_t, 1, wl=11, return_all=True)
        [st_mat_w5, sw_mat_w5, st_mat_all_w5, sw_mat_all_w5] = loco_obj.get_sw_st_matrices(final_tracks_t, 1, wl=5, return_all=True)
        paws_rel_t = {'x': loco_obj.get_paws_rel(final_tracks_t, 'X'), 
                      'y': loco_obj.get_paws_rel(final_tracks_t, 'Y'), 
                      'z': loco_obj.get_paws_rel(final_tracks_t, 'Z')}
        
        ds_w11 = loco_obj.compute_gait_param(bodycenter_t, final_tracks_t, paws_rel_t, st_mat_w11, sw_mat_w11, 'double_support')
        ds_w5 = loco_obj.compute_gait_param(bodycenter_t, final_tracks_t, paws_rel_t, st_mat_w5, sw_mat_w5, 'double_support')
        
        # Count zeros
        zs11, tot11 = count_zeros(ds_w11)
        zs5, tot5 = count_zeros(ds_w5)
        
        # Extract paw vectors for all 4 paws
        _ds_w11 = np.asarray(ds_w11, dtype=object)
        _ds_w5 = np.asarray(ds_w5, dtype=object)
        
        paw_vectors = {}
        paw_names = paws_to_analyze
        paw_indices = [PAWS_INDEX[p] for p in paws_to_analyze]
        
        for paw_name, paw_idx in zip(paw_names, paw_indices):
            paw_vectors[f'{paw_name}_w11'] = extract_paw_vec(_ds_w11, paw_idx)
            paw_vectors[f'{paw_name}_w5'] = extract_paw_vec(_ds_w5, paw_idx)
        
        # Compute means and count zeros for each paw
        paw_means = {}
        paw_zeros = {}
        for key, vec in paw_vectors.items():
            paw_means[f'{key}_mean'] = np.nanmean(vec) if vec.size > 0 else np.nan
            # Count zeros in this paw's vector
            finite_vals = vec[np.isfinite(vec)]
            paw_zeros[f'{key}_zeros'] = int(np.sum(finite_vals == 0)) if finite_vals.size > 0 else 0
        
        # Create and save plot for this trial
        if save_plot or show_plot:
            fig_ds = go.Figure()
            
            # Color scheme: FR=red, HR=orange, FL=blue, HL=cyan
            paw_colors = {'FR': 'red', 'HR': 'orange', 'FL': 'blue', 'HL': 'cyan'}
            
            for paw_name in paw_names:
                color = paw_colors[paw_name]
                vec_w11 = paw_vectors[f'{paw_name}_w11']
                vec_w5 = paw_vectors[f'{paw_name}_w5']
                mean_w11 = paw_means[f'{paw_name}_w11_mean']
                mean_w5 = paw_means[f'{paw_name}_w5_mean']
                
                # W11 trace (solid line)
                if vec_w11.size > 0:
                    fig_ds.add_trace(go.Scatter(
                        x=np.arange(vec_w11.size), y=vec_w11,
                        name=f'{paw_name} w11', line=dict(color=color, dash='solid')
                    ))
                
                # W5 trace (dashed line)
                if vec_w5.size > 0:
                    fig_ds.add_trace(go.Scatter(
                        x=np.arange(vec_w5.size), y=vec_w5,
                        name=f'{paw_name} w5', line=dict(color=color, dash='dash')
                    ))
                
                # Mean lines
                if np.isfinite(mean_w11):
                    fig_ds.add_hline(y=mean_w11, line=dict(color=color, dash='solid', width=1),
                                     annotation_text=f"{paw_name} w11={mean_w11:.2f}", 
                                     annotation_position="left")
                if np.isfinite(mean_w5):
                    fig_ds.add_hline(y=mean_w5, line=dict(color=color, dash='dot', width=1),
                                     annotation_text=f"{paw_name} w5={mean_w5:.2f}", 
                                     annotation_position="right")
            
            fig_ds.update_layout(
                title=f"{animal_id} trial {trial_num} - Double Support: w11 vs w5 (all paws)",
                xaxis_title="Stride index",
                yaxis_title="Double support (frames)",
                showlegend=True,
                legend=dict(x=1.02, y=1)
            )
            
            if show_plot:
                fig_ds.show()
            
            if save_plot:
                plot_path = os.path.join(loco_obj.path, f"{animal_id}_trial{trial_num}_double_support_w11_vs_w5.html")
                fig_ds.write_html(plot_path)
        
        # Count strides (before and after exclusion)
        n_strides_all_w11_fr = len(extract_stride_indices(st_mat_all_w11, 0)) if st_mat_all_w11 is not None else 0
        n_strides_all_w11_fl = len(extract_stride_indices(st_mat_all_w11, 2)) if st_mat_all_w11 is not None else 0
        n_strides_w11_fr = len(extract_stride_indices(st_mat_w11, 0)) if st_mat_w11 is not None else 0
        n_strides_w11_fl = len(extract_stride_indices(st_mat_w11, 2)) if st_mat_w11 is not None else 0
        
        n_strides_all_w5_fr = len(extract_stride_indices(st_mat_all_w5, 0)) if st_mat_all_w5 is not None else 0
        n_strides_all_w5_fl = len(extract_stride_indices(st_mat_all_w5, 2)) if st_mat_all_w5 is not None else 0
        n_strides_w5_fr = len(extract_stride_indices(st_mat_w5, 0)) if st_mat_w5 is not None else 0
        n_strides_w5_fl = len(extract_stride_indices(st_mat_w5, 2)) if st_mat_w5 is not None else 0
        
        # Build result dict with all paw means
        result = {
            'animal': animal_id,
            'trial': trial_num,
            'zeros_w11': zs11,
            'zeros_w5': zs5,
            'total_w11': tot11,
            'total_w5': tot5,
            'n_strides_all_w11_fr': n_strides_all_w11_fr,
            'n_strides_all_w11_fl': n_strides_all_w11_fl,
            'n_strides_w11_fr': n_strides_w11_fr,
            'n_strides_w11_fl': n_strides_w11_fl,
            'n_excluded_w11_fr': n_strides_all_w11_fr - n_strides_w11_fr,
            'n_excluded_w11_fl': n_strides_all_w11_fl - n_strides_w11_fl,
            'n_strides_all_w5_fr': n_strides_all_w5_fr,
            'n_strides_all_w5_fl': n_strides_all_w5_fl,
            'n_strides_w5_fr': n_strides_w5_fr,
            'n_strides_w5_fl': n_strides_w5_fl,
            'n_excluded_w5_fr': n_strides_all_w5_fr - n_strides_w5_fr,
            'n_excluded_w5_fl': n_strides_all_w5_fl - n_strides_w5_fl,
        }
        
        # Add all paw means and zeros to result
        for paw_name in paw_names:
            result[f'{paw_name.lower()}_mean_w11'] = paw_means[f'{paw_name}_w11_mean']
            result[f'{paw_name.lower()}_mean_w5'] = paw_means[f'{paw_name}_w5_mean']
            result[f'{paw_name.lower()}_zeros_w11'] = paw_zeros[f'{paw_name}_w11_zeros']
            result[f'{paw_name.lower()}_zeros_w5'] = paw_zeros[f'{paw_name}_w5_zeros']
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Failed to analyze {animal_id} trial {trial_num}: {e}")
        return None


# Run batch analysis if animals_ds and trials_ds are defined
if 'animals_ds' in dir() and 'trials_ds' in dir() and len(animals_ds) > 0 and len(trials_ds) > 0:
    print("\n" + "="*80)
    print("BATCH DOUBLE SUPPORT ANALYSIS")
    print("="*80)
    
    ds_results = []
    
    is_first_trial = True  # Flag to track first trial for plotting
    for animal_ds in animals_ds:
        print(f"\n--- Analyzing animal: {animal_ds} ---")
        for trial_ds in trials_ds:
            # Only save/show plot for first animal and first trial
            save_this_plot = is_first_trial
            show_this_plot = show_plots and is_first_trial
            
            result = analyze_double_support_trial(loco, animal_ds, trial_ds, paws,
                                                   session_num=session, conf=confidence, 
                                                   save_plot=save_this_plot, show_plot=show_this_plot)
            is_first_trial = False  # After first iteration, no more plots
            
            if result is not None:
                ds_results.append(result)
                print(f"  Trial {trial_ds}: zeros_w11={result['zeros_w11']}/{result['total_w11']}, "
                      f"zeros_w5={result['zeros_w5']}/{result['total_w5']}")
                # Print means for selected paws only
                means_w11_str = ", ".join([f"{p}={result[f'{p.lower()}_mean_w11']:.3f}" for p in paws])
                means_w5_str = ", ".join([f"{p}={result[f'{p.lower()}_mean_w5']:.3f}" for p in paws])
                print(f"    Means w11: {means_w11_str}")
                print(f"    Means w5:  {means_w5_str}")
    
    # Save results to CSV
    if ds_results:
        import csv
        csv_path = os.path.join(loco.path, 'double_support_batch_results.csv')
        fieldnames = list(ds_results[0].keys())
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ds_results)
        
        print(f"\n[INFO] Batch results saved to: {csv_path}")
        
        # Print summary statistics
        print("\n" + "-"*80)
        print("SUMMARY STATISTICS")
        print("-"*80)
        
        for animal_ds in animals_ds:
            animal_results = [r for r in ds_results if r['animal'] == animal_ds]
            if not animal_results:
                continue
            
            print(f"\n{animal_ds} ({len(animal_results)} trials):")
            
            # Aggregate stats
            total_zeros_w11 = sum(r['zeros_w11'] for r in animal_results)
            total_total_w11 = sum(r['total_w11'] for r in animal_results)
            total_zeros_w5 = sum(r['zeros_w5'] for r in animal_results)
            total_total_w5 = sum(r['total_w5'] for r in animal_results)
            
            # Compute averages for selected paws only
            paw_names_lower = [p.lower() for p in paws]
            avg_means_w11 = {p: np.nanmean([r[f'{p}_mean_w11'] for r in animal_results]) for p in paw_names_lower}
            avg_means_w5 = {p: np.nanmean([r[f'{p}_mean_w5'] for r in animal_results]) for p in paw_names_lower}
            
            total_excluded_w11_fr = sum(r['n_excluded_w11_fr'] for r in animal_results)
            total_excluded_w11_fl = sum(r['n_excluded_w11_fl'] for r in animal_results)
            total_excluded_w5_fr = sum(r['n_excluded_w5_fr'] for r in animal_results)
            total_excluded_w5_fl = sum(r['n_excluded_w5_fl'] for r in animal_results)
            
            pct_zeros_w11 = (total_zeros_w11 / total_total_w11 * 100) if total_total_w11 > 0 else np.nan
            pct_zeros_w5 = (total_zeros_w5 / total_total_w5 * 100) if total_total_w5 > 0 else np.nan
            
            # Build dynamic means string
            means_w11_str = ", ".join([f"{p.upper()}={avg_means_w11[p]:.3f}" for p in paw_names_lower])
            means_w5_str = ", ".join([f"{p.upper()}={avg_means_w5[p]:.3f}" for p in paw_names_lower])
            
            print(f"  W11: zeros={total_zeros_w11}/{total_total_w11} ({pct_zeros_w11:.2f}%)")
            print(f"       Means: {means_w11_str}")
            print(f"       Excluded: FR={total_excluded_w11_fr}, FL={total_excluded_w11_fl}")
            print(f"  W5:  zeros={total_zeros_w5}/{total_total_w5} ({pct_zeros_w5:.2f}%)")
            print(f"       Means: {means_w5_str}")
            print(f"       Excluded: FR={total_excluded_w5_fr}, FL={total_excluded_w5_fl}")
    
        # --- Plot batch results: w5 vs w11 comparison ---
        print("\n" + "-"*80)
        print("GENERATING BATCH COMPARISON PLOTS")
        print("-"*80)
        
        paw_colors = {'FR': 'red', 'HR': 'orange', 'FL': 'blue', 'HL': 'cyan'}
        
        # Plot 1: Scatter plot of means w5 vs w11 for each paw (all animals, all trials)
        fig_scatter = go.Figure()
        
        for paw in paws:
            paw_lower = paw.lower()
            w11_vals = [r[f'{paw_lower}_mean_w11'] for r in ds_results]
            w5_vals = [r[f'{paw_lower}_mean_w5'] for r in ds_results]
            labels = [f"{r['animal']} T{r['trial']}" for r in ds_results]
            
            fig_scatter.add_trace(go.Scatter(
                x=w11_vals, y=w5_vals,
                mode='markers',
                name=paw,
                marker=dict(color=paw_colors.get(paw, 'gray'), size=10),
                text=labels,
                hovertemplate='%{text}<br>w11: %{x:.3f}<br>w5: %{y:.3f}<extra></extra>'
            ))
        
        # Add identity line
        all_w11 = [r[f'{p.lower()}_mean_w11'] for r in ds_results for p in paws]
        all_w5 = [r[f'{p.lower()}_mean_w5'] for r in ds_results for p in paws]
        min_val = min(min(all_w11), min(all_w5)) if all_w11 and all_w5 else 0
        max_val = max(max(all_w11), max(all_w5)) if all_w11 and all_w5 else 1
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', name='Identity (y=x)',
            line=dict(color='black', dash='dash')
        ))
        
        fig_scatter.update_layout(
            title='Double Support: w5 vs w11 Mean Comparison (all animals, all trials)',
            xaxis_title='Mean Double Support w11',
            yaxis_title='Mean Double Support w5',
            showlegend=True
        )
        
        scatter_path = os.path.join(loco.path, 'batch_double_support_w5_vs_w11_scatter.html')
        fig_scatter.write_html(scatter_path)
        print(f"[INFO] Scatter plot saved to: {scatter_path}")
        if show_plots:
            fig_scatter.show()
        
        # --- Line plots across trials with mean ± std across animals ---
        from plotly.subplots import make_subplots
        
        # Organize data by trial (aggregate across animals)
        trial_data = {}
        for trial_ds in trials_ds:
            trial_data[trial_ds] = {
                'fr_fl_diff_w11': [], 'fr_fl_diff_w5': [],
                'animals': [],  # Track which animal each data point belongs to
                'excluded_w11_fr': [], 'excluded_w5_fr': [],
                'excluded_w11_fl': [], 'excluded_w5_fl': [],
                'zeros_w11': [], 'zeros_w5': []
            }
        
        for r in ds_results:
            t = r['trial']
            if t in trial_data:
                trial_data[t]['fr_fl_diff_w11'].append(r['fr_mean_w11'] - r['fl_mean_w11'])
                trial_data[t]['fr_fl_diff_w5'].append(r['fr_mean_w5'] - r['fl_mean_w5'])
                trial_data[t]['animals'].append(r['animal'])
                trial_data[t]['excluded_w11_fr'].append(r.get('n_excluded_w11_fr', 0))
                trial_data[t]['excluded_w5_fr'].append(r.get('n_excluded_w5_fr', 0))
                trial_data[t]['excluded_w11_fl'].append(r.get('n_excluded_w11_fl', 0))
                trial_data[t]['excluded_w5_fl'].append(r.get('n_excluded_w5_fl', 0))
                trial_data[t]['zeros_w11'].append(r['zeros_w11'])
                trial_data[t]['zeros_w5'].append(r['zeros_w5'])
        
        # Compute mean and std per trial
        trial_nums = sorted(trial_data.keys())
        
        def compute_mean_std(data_list):
            return np.nanmean(data_list), np.nanstd(data_list)
        
        # FR-FL difference: for each trial, compute mean of (animal means across strides)
        # First, compute per-animal mean across all their strides in each trial
        # trial_data[t]['fr_fl_diff_w11'] already contains FR-FL diff per animal (mean across strides for that trial)
        # So we compute mean ± std across animals for each trial
        diff_w11_mean = [compute_mean_std(trial_data[t]['fr_fl_diff_w11'])[0] for t in trial_nums]
        diff_w11_std = [compute_mean_std(trial_data[t]['fr_fl_diff_w11'])[1] for t in trial_nums]
        diff_w5_mean = [compute_mean_std(trial_data[t]['fr_fl_diff_w5'])[0] for t in trial_nums]
        diff_w5_std = [compute_mean_std(trial_data[t]['fr_fl_diff_w5'])[1] for t in trial_nums]
        
        # Excluded strides (sum FR + FL)
        excl_w11_mean = [compute_mean_std([a + b for a, b in zip(trial_data[t]['excluded_w11_fr'], trial_data[t]['excluded_w11_fl'])])[0] for t in trial_nums]
        excl_w11_std = [compute_mean_std([a + b for a, b in zip(trial_data[t]['excluded_w11_fr'], trial_data[t]['excluded_w11_fl'])])[1] for t in trial_nums]
        excl_w5_mean = [compute_mean_std([a + b for a, b in zip(trial_data[t]['excluded_w5_fr'], trial_data[t]['excluded_w5_fl'])])[0] for t in trial_nums]
        excl_w5_std = [compute_mean_std([a + b for a, b in zip(trial_data[t]['excluded_w5_fr'], trial_data[t]['excluded_w5_fl'])])[1] for t in trial_nums]
        
        # Zero values
        zeros_w11_mean = [compute_mean_std(trial_data[t]['zeros_w11'])[0] for t in trial_nums]
        zeros_w11_std = [compute_mean_std(trial_data[t]['zeros_w11'])[1] for t in trial_nums]
        zeros_w5_mean = [compute_mean_std(trial_data[t]['zeros_w5'])[0] for t in trial_nums]
        zeros_w5_std = [compute_mean_std(trial_data[t]['zeros_w5'])[1] for t in trial_nums]
        
        # Plot 2: Line plot - FR-FL DS difference across trials
        # With shaded standard error areas and individual animal lines
        fig_diff = go.Figure()
        
        # Prepare individual animal data across trials
        animal_trial_data_w11 = {a: {} for a in animals_ds}
        animal_trial_data_w5 = {a: {} for a in animals_ds}
        
        for t in trial_nums:
            for i, animal in enumerate(trial_data[t]['animals']):
                animal_trial_data_w11[animal][t] = trial_data[t]['fr_fl_diff_w11'][i]
                animal_trial_data_w5[animal][t] = trial_data[t]['fr_fl_diff_w5'][i]
        
        # Add individual animal lines (thin, semi-transparent)
        for animal in animals_ds:
            animal_trials_w11 = sorted(animal_trial_data_w11[animal].keys())
            animal_values_w11 = [animal_trial_data_w11[animal][t] for t in animal_trials_w11]
            animal_trials_w5 = sorted(animal_trial_data_w5[animal].keys())
            animal_values_w5 = [animal_trial_data_w5[animal][t] for t in animal_trials_w5]
            
            # w11 individual lines
            fig_diff.add_trace(go.Scatter(
                x=animal_trials_w11, y=animal_values_w11,
                mode='lines+markers', name=f'{animal} (w11)',
                line=dict(color='darkcyan', width=1),
                marker=dict(size=4),
                opacity=0.3,
                legendgroup='w11_individual',
                showlegend=(animal == animals_ds[0])  # Only show legend for first animal
            ))
            
            # w5 individual lines
            fig_diff.add_trace(go.Scatter(
                x=animal_trials_w5, y=animal_values_w5,
                mode='lines+markers', name=f'{animal} (w5)',
                line=dict(color='cyan', width=1),
                marker=dict(size=4),
                opacity=0.3,
                legendgroup='w5_individual',
                showlegend=(animal == animals_ds[0])  # Only show legend for first animal
            ))
        
        # Add shaded area for standard error (w11)
        diff_w11_upper = [m + s for m, s in zip(diff_w11_mean, diff_w11_std)]
        diff_w11_lower = [m - s for m, s in zip(diff_w11_mean, diff_w11_std)]
        
        fig_diff.add_trace(go.Scatter(
            x=list(trial_nums) + list(trial_nums)[::-1],
            y=diff_w11_upper + diff_w11_lower[::-1],
            fill='toself',
            fillcolor='rgba(0, 139, 139, 0.2)',  # darkcyan with alpha
            line=dict(color='rgba(255,255,255,0)'),
            name='w11 ± std',
            showlegend=True,
            legendgroup='w11_mean'
        ))
        
        # Add shaded area for standard error (w5)
        diff_w5_upper = [m + s for m, s in zip(diff_w5_mean, diff_w5_std)]
        diff_w5_lower = [m - s for m, s in zip(diff_w5_mean, diff_w5_std)]
        
        fig_diff.add_trace(go.Scatter(
            x=list(trial_nums) + list(trial_nums)[::-1],
            y=diff_w5_upper + diff_w5_lower[::-1],
            fill='toself',
            fillcolor='rgba(0, 255, 255, 0.2)',  # cyan with alpha
            line=dict(color='rgba(255,255,255,0)'),
            name='w5 ± std',
            showlegend=True,
            legendgroup='w5_mean'
        ))
        
        # Add mean lines (thick, on top)
        fig_diff.add_trace(go.Scatter(
            x=trial_nums, y=diff_w11_mean,
            mode='lines+markers', name='w11 mean',
            line=dict(color='darkcyan', width=3),
            marker=dict(size=8),
            legendgroup='w11_mean'
        ))
        fig_diff.add_trace(go.Scatter(
            x=trial_nums, y=diff_w5_mean,
            mode='lines+markers', name='w5 mean',
            line=dict(color='cyan', width=3),
            marker=dict(size=8),
            legendgroup='w5_mean'
        ))
        
        fig_diff.add_hline(y=0, line=dict(color='black', dash='dash'))
        
        fig_diff.update_layout(
            title='FR-FL Double Support Difference across Trials (mean ± std across animals)',
            xaxis_title='Trial',
            yaxis_title='FR-FL Difference (frames)',
            showlegend=True
        )
        
        diff_path = os.path.join(loco.path, 'batch_double_support_FR_FL_diff_lines.html')
        fig_diff.write_html(diff_path)
        print(f"[INFO] FR-FL difference line plot saved to: {diff_path}")
        if show_plots:
            fig_diff.show()
        
        # Plot 3: Combined bar plot - Excluded strides and Zero values (2 subplots)
        from plotly.subplots import make_subplots
        
        # Compute per-animal totals for excluded strides (FR and FL)
        excl_per_animal_w11_fr = [sum(r.get('n_excluded_w11_fr', 0) for r in ds_results if r['animal'] == a) for a in animals_ds]
        excl_per_animal_w11_fl = [sum(r.get('n_excluded_w11_fl', 0) for r in ds_results if r['animal'] == a) for a in animals_ds]
        excl_per_animal_w5_fr = [sum(r.get('n_excluded_w5_fr', 0) for r in ds_results if r['animal'] == a) for a in animals_ds]
        excl_per_animal_w5_fl = [sum(r.get('n_excluded_w5_fl', 0) for r in ds_results if r['animal'] == a) for a in animals_ds]
        
        mean_excl_w11_fr = np.mean(excl_per_animal_w11_fr)
        std_excl_w11_fr = np.std(excl_per_animal_w11_fr)
        mean_excl_w11_fl = np.mean(excl_per_animal_w11_fl)
        std_excl_w11_fl = np.std(excl_per_animal_w11_fl)
        mean_excl_w5_fr = np.mean(excl_per_animal_w5_fr)
        std_excl_w5_fr = np.std(excl_per_animal_w5_fr)
        mean_excl_w5_fl = np.mean(excl_per_animal_w5_fl)
        std_excl_w5_fl = np.std(excl_per_animal_w5_fl)
        
        # Compute per-animal totals for zeros per paw
        zeros_w11_means = []
        zeros_w11_stds = []
        zeros_w5_means = []
        zeros_w5_stds = []
        
        for paw in paws:
            paw_lower = paw.lower()
            zeros_per_animal_w11 = [sum(r.get(f'{paw_lower}_zeros_w11', 0) for r in ds_results if r['animal'] == a) for a in animals_ds]
            zeros_per_animal_w5 = [sum(r.get(f'{paw_lower}_zeros_w5', 0) for r in ds_results if r['animal'] == a) for a in animals_ds]
            
            zeros_w11_means.append(np.mean(zeros_per_animal_w11))
            zeros_w11_stds.append(np.std(zeros_per_animal_w11))
            zeros_w5_means.append(np.mean(zeros_per_animal_w5))
            zeros_w5_stds.append(np.std(zeros_per_animal_w5))
        
        # Create figure with 2 subplots
        fig_combined = make_subplots(rows=1, cols=2, 
                                      subplot_titles=('Excluded Strides', 'Zero DS Values'))
        
        # Subplot 1: Excluded strides
        fig_combined.add_trace(go.Bar(
            name='w11', x=['FR', 'FL'], y=[mean_excl_w11_fr, mean_excl_w11_fl],
            marker_color='darkcyan',
            error_y=dict(type='data', array=[std_excl_w11_fr, std_excl_w11_fl], visible=True),
            legendgroup='w11'
        ), row=1, col=1)
        fig_combined.add_trace(go.Bar(
            name='w5', x=['FR', 'FL'], y=[mean_excl_w5_fr, mean_excl_w5_fl],
            marker_color='cyan',
            error_y=dict(type='data', array=[std_excl_w5_fr, std_excl_w5_fl], visible=True),
            legendgroup='w5'
        ), row=1, col=1)
        
        # Subplot 2: Zero values per paw
        fig_combined.add_trace(go.Bar(
            name='w11', x=paws, y=zeros_w11_means,
            marker_color='darkcyan',
            error_y=dict(type='data', array=zeros_w11_stds, visible=True),
            legendgroup='w11', showlegend=False
        ), row=1, col=2)
        fig_combined.add_trace(go.Bar(
            name='w5', x=paws, y=zeros_w5_means,
            marker_color='cyan',
            error_y=dict(type='data', array=zeros_w5_stds, visible=True),
            legendgroup='w5', showlegend=False
        ), row=1, col=2)
        
        fig_combined.update_layout(
            title='Excluded Strides and Zero DS Values: w11 vs w5 (mean ± std across animals)',
            barmode='group',
            showlegend=True
        )
        fig_combined.update_xaxes(title_text='Paw', row=1, col=1)
        fig_combined.update_xaxes(title_text='Paw', row=1, col=2)
        fig_combined.update_yaxes(title_text='Count', row=1, col=1)
        fig_combined.update_yaxes(title_text='Count', row=1, col=2)
        
        combined_path = os.path.join(loco.path, 'batch_excluded_zeros_combined.html')
        fig_combined.write_html(combined_path)
        print(f"[INFO] Combined excluded/zeros bar plot saved to: {combined_path}")
        if show_plots:
            fig_combined.show()

else:
    # Original single-trial analysis (backward compatibility)
    double_support_w11 = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat_w11, sw_pts_mat_w11, 'double_support')
    double_support_w5 = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat_w5, sw_pts_mat_w5, 'double_support')

    # Count number of finite elements equal to 0
    zs11, tot11 = count_zeros(double_support_w11)
    zs5, tot5 = count_zeros(double_support_w5)

    pct11 = (zs11 / tot11 * 100) if tot11 > 0 else np.nan
    pct5 = (zs5 / tot5 * 100) if tot5 > 0 else np.nan

    print(f"double_support_w11: zeros = {zs11} / finite = {tot11} ({pct11:.2f}%)")
    print(f"double_support_w5 : zeros = {zs5}  / finite = {tot5}  ({pct5:.2f}%)")

    # Safely convert to object arrays
    _ds_w11 = np.asarray(double_support_w11, dtype=object)
    _ds_w5 = np.asarray(double_support_w5, dtype=object)

    fr_w11 = extract_paw_vec(_ds_w11, 0)
    fl_w11 = extract_paw_vec(_ds_w11, 2)
    fr_w5 = extract_paw_vec(_ds_w5, 0)
    fl_w5 = extract_paw_vec(_ds_w5, 2)

    fr_w11_mean = np.nanmean(fr_w11)
    fl_w11_mean = np.nanmean(fl_w11)
    fr_w5_mean = np.nanmean(fr_w5)
    fl_w5_mean = np.nanmean(fl_w5)

    fig_ds_comp = go.Figure()

    # FR traces (blue)
    if fr_w11.size:
        add_line_trace(fig_ds_comp, fr_w11, 'FR double_support w11', 'blue')
    if fr_w5.size:
        add_line_trace(fig_ds_comp, fr_w5, 'FR double_support w5', 'blue', dash='dash')

    # FL traces (red)
    if fl_w11.size:
        add_line_trace(fig_ds_comp, fl_w11, 'FL double_support w11', 'red')
    if fl_w5.size:
        add_line_trace(fig_ds_comp, fl_w5, 'FL double_support w5', 'red', dash='dash')

    # Horizontal mean lines with annotations
    if np.isfinite(fr_w11_mean):
        fig_ds_comp.add_hline(y=fr_w11_mean, line=dict(color='blue', dash='solid'),
                              annotation_text=f"FR w11 mean={fr_w11_mean:.3f}", annotation_position="top left")
    if np.isfinite(fr_w5_mean):
        fig_ds_comp.add_hline(y=fr_w5_mean, line=dict(color='blue', dash='dash'),
                              annotation_text=f"FR w5 mean={fr_w5_mean:.3f}", annotation_position="bottom left")
    if np.isfinite(fl_w11_mean):
        fig_ds_comp.add_hline(y=fl_w11_mean, line=dict(color='red', dash='solid'),
                              annotation_text=f"FL w11 mean={fl_w11_mean:.3f}", annotation_position="top right")
    if np.isfinite(fl_w5_mean):
        fig_ds_comp.add_hline(y=fl_w5_mean, line=dict(color='red', dash='dash'),
                              annotation_text=f"FL w5 mean={fl_w5_mean:.3f}", annotation_position="bottom right")

    fig_ds_comp.update_layout(
        title=f"{animal} trial {count_trial} - Double support: w11 vs w5",
        xaxis_title="Frame index",
        yaxis_title="Double support (arb. units)",
        showlegend=True
    )

    if show_plots:
        fig_ds_comp.show()

    out_comp = os.path.join(loco.path, f"{animal}_trial{count_trial}_double_support_w11_vs_w5.html")
    fig_ds_comp.write_html(out_comp)
    print(f"[INFO] Double-support w11 vs w5 comparison saved to: {out_comp}")

    # Print nanmeans and FR-FL differences
    print(f"[STATS] FR w11 nanmean: {fr_w11_mean:.6g}" if np.isfinite(fr_w11_mean) else "[STATS] FR w11 nanmean: NaN")
    print(f"[STATS] FL w11 nanmean: {fl_w11_mean:.6g}" if np.isfinite(fl_w11_mean) else "[STATS] FL w11 nanmean: NaN")
    print(f"[STATS] FR w5  nanmean: {fr_w5_mean:.6g}" if np.isfinite(fr_w5_mean) else "[STATS] FR w5  nanmean: NaN")
    print(f"[STATS] FL w5  nanmean: {fl_w5_mean:.6g}" if np.isfinite(fl_w5_mean) else "[STATS] FL w5  nanmean: NaN")

    diff_w11 = fr_w11_mean - fl_w11_mean if (np.isfinite(fr_w11_mean) and np.isfinite(fl_w11_mean)) else np.nan
    diff_w5 = fr_w5_mean - fl_w5_mean if (np.isfinite(fr_w5_mean) and np.isfinite(fl_w5_mean)) else np.nan

    print(f"[STATS] FR-FL nanmean difference w11: {diff_w11:.6g}" if np.isfinite(diff_w11) else "[STATS] FR-FL nanmean difference w11: NaN")
    print(f"[STATS] FR-FL nanmean difference w5 : {diff_w5:.6g}" if np.isfinite(diff_w5) else "[STATS] FR-FL nanmean difference w5 : NaN")

    # Compare double-support vectors (w5 vs w11) for FR and FL
    for paw_name, paw_idx, ds_w11_vec, ds_w5_vec in (
        ('FR', 0, fr_w11, fr_w5),
        ('FL', 2, fl_w11, fl_w5),
    ):
        starts_w11, ends_w11 = stride_bounds(st_strides_mat_w11, paw_idx)
        starts_w5, ends_w5 = stride_bounds(st_strides_mat_w5, paw_idx)
        starts_use = starts_w11 if starts_w11.size else starts_w5
        ends_use = ends_w11 if ends_w11.size else ends_w5
        n = min(ds_w11_vec.size, ds_w5_vec.size, starts_use.size, ends_use.size)
        if n == 0:
            continue
        diff_vec = ds_w5_vec[:n] - ds_w11_vec[:n]
        mask = np.abs(diff_vec) > 4
        if np.any(mask):
            print(f"[WARN] {paw_name} double_support |w5-w11| > 4 at {mask.sum()} stride(s):")
            for i in np.where(mask)[0]:
                print(f"  stride {i}: diff={diff_vec[i]:.3f}, start={int(starts_use[i])}, end={int(ends_use[i])}")

