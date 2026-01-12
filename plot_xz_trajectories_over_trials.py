"""
Generates interactive Plotly figures for locomotion trials:
- X–Z paw trajectories or velocities overlapped across trials (per animal, chosen paw).
- Contralateral paw distances (FR–HL, FL–HR), their difference, and PSD.
- Hilbert-based relative phase of contralateral signals.
- Across animals: per-trial lag (ms) between FR–HL and FL–HR with average.

Notes: NaNs are inpainted; Savitzky–Golay filter (5,1) applied; Z is inverted (−Z); trial colors indicate tied, stim, and late phases.
Outputs saved under `<PATH>/kinematics/` as HTML.

Authors: Alice Geminiani (AliceGem), GitHub Copilot
Model: GitHub Copilot (ChatGPT-5)
"""

import os
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter, welch, correlate, hilbert
from scipy.stats import circmean
from locomotion_class import loco_class



# Configuration: update these for your dataset
PATH = 'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\tied stance stim retracked with ClosedLoop-AliceG-2025-10-06\\'
PATH = 'D:\\AliG\\climbing-opto-treadmill\\WT split-belt learning\\'
ANIMALS = ['MC2585','MC2592']        # ['MC16848', 'VIV42428', 'MC19124']           # # list of animal IDs to include  
SESSION = 1
PAW = 'FR'  # which paw to plot
PAW_IDX = {'FR':0, 'HR':1, 'FL':2, 'HL':3}

# Signal type: 'position' or 'velocity'
SIGNAL_TYPE = 'velocity'  # Options: 'position', 'velocity'
FRAME_RATE = 330  # frames per second, used for velocity calculation

if 'WT' in PATH:
    pixel_to_mm = 1/1.955
else:
    pixel_to_mm = 1/3.3


# Trial color configuration
NTRIALS = 28
STIM_START = 9
SPLIT_START = 9  # kept for compatibility; equals STIM_START
STIM_DURATION = 10  # trials STIM_START .. STIM_START+STIM_DURATION-1 are stim

def plot_animal_xz_trajectories(path, animal, session, paw_name, signal_type='position', frame_rate=330):
    """
    Plot overlapped X-Z trajectories or velocities across trials for an animal.
    
    Parameters:
    -----------
    signal_type : str
        'position' for X-Z position trajectories, 'velocity' for X-Z velocity
    frame_rate : float
        Frame rate in Hz, used for velocity calculation (mm/s)
    """
    loco = loco_class(path, pixel_to_mm=pixel_to_mm)
    filelist = loco.get_track_files(animal, session)
    paw_idx = PAW_IDX.get(paw_name, 0)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def parse_trial_number(filename: str) -> int:
        try:
            # Example: MC16848_159_22_0.275_0.275_tied_1_1DLC... → trial is the part before 'DLC', last token may contain session/trial
            base = filename.split('DLC')[0].strip('_')
            parts = base.split('_')
            # Common pattern: ..._tied_SESSION_TRIAL
            trial_num = int(parts[-1])
            return trial_num
        except Exception:
            return -1

    def shade_color(hex_color: str, factor: float) -> str:
        """Scale color darkness/lightness.
        factor < 1.0: darken by multiplying RGB.
        factor = 1.0: original.
        factor > 1.0: lighten by blending towards white (255).
        """
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        if factor <= 1.0:
            r = int(r * factor)
            g = int(g * factor)
            b = int(b * factor)
        else:
            # Blend towards white; factor in (1, 1.5] recommended
            alpha = min(factor - 1.0, 0.5)  # 0..0.5
            r = int(r + (255 - r) * alpha)
            g = int(g + (255 - g) * alpha)
            b = int(b + (255 - b) * alpha)
        return f"#{r:02x}{g:02x}{b:02x}"

    def color_for_trial(trial_num: int) -> str:
        tied_end = STIM_START - 1
        stim_end = STIM_START + STIM_DURATION - 1
        last_start = stim_end + 1
        # Tied trials: same blue
        if 1 <= trial_num <= tied_end:
            return '#1f77b4'
        # Stim trials: same red, increasing lightness (dark to light)
        if STIM_START <= trial_num <= stim_end:
            # Start dark, fade to lighter
            steps = stim_end - STIM_START
            idx = trial_num - STIM_START
            start_factor = 0.50  # dark
            end_factor = 1.50    # lighten towards white
            factor = start_factor + ((end_factor - start_factor) * (idx / max(steps, 1)))
            return shade_color('#d62728', factor)
        # Last trials: green with increasing lightness (dark to light)
        if last_start <= trial_num <= NTRIALS:
            steps = NTRIALS - last_start
            idx = trial_num - last_start
            # Start dark and go to lighter color
            start_factor = 0.50  # dark
            end_factor = 1.50    # lighten
            factor = start_factor + ((end_factor - start_factor) * (idx / max(steps, 1)))
            return shade_color('#2ca02c', factor)
        # Fallback
        return '#7f7f7f'
    fig = go.Figure()

    for t_idx, f in enumerate(filelist):
        try:
            final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter = loco.read_h5(f, 0.9, 0)
        except Exception as e:
            print(f"[WARN] Skipping trial '{f}' due to read error: {e}")
            continue
        # Use paws_rel (relative to body center), like kinematic_analysis
        paws_rel = {
            'x': loco.get_paws_rel(final_tracks, 'X'),
            'z': loco.get_paws_rel(final_tracks, 'Z')
        }
        # Ensure array shape supports [paw_idx, :]
        x_rel_arr = np.asarray(paws_rel['x'])
        z_rel_arr = np.asarray(paws_rel['z'])
        try:
            x_rel = x_rel_arr[paw_idx, :]
            z_rel = z_rel_arr[paw_idx, :]
        except Exception:
            # Fallback if paws_rel returns a list per paw
            x_rel = np.asarray(paws_rel['x'][paw_idx])
            z_rel = np.asarray(paws_rel['z'][paw_idx])
        # Filter
        x_filt = savgol_filter(x_rel, window_length=5, polyorder=1)
        z_filt = savgol_filter(z_rel, window_length=5, polyorder=1)
        # Invert Z to match expected orientation in the plots
        z_filt = -z_filt

        # Compute velocity if requested
        if signal_type == 'velocity':
            # Velocity = derivative of position * frame_rate (mm/s)
            x_vel = np.gradient(x_filt) * frame_rate
            z_vel = np.gradient(z_filt) * frame_rate
            # Use velocity instead of position
            x_filt = x_vel
            z_filt = z_vel

        # Get strides and swing points from locomotion_class
        st_strides_mat, sw_pts_mat, st_all, sw_all = loco.get_sw_st_matrices(final_tracks, 1, return_all=True)
        # Extract swing onset frames for this paw
        if sw_pts_mat is None:
            print(f"[WARN] No swing points for trial '{f}' paw '{paw_name}'")
            continue
        # Handle list or array container
        try:
            n_paws = sw_pts_mat.shape[0]
        except Exception:
            n_paws = len(sw_pts_mat)
        if n_paws <= paw_idx:
            print(f"[WARN] No swing points for trial '{f}' paw '{paw_name}' (index out of range)")
            continue
        sw_paw = sw_pts_mat[paw_idx]
        sw_arr = np.asarray(sw_paw)
        # Expect shape [N_strides, 1, 2] or similar; last column holds frame index
        try:
            sw_frames = np.int64(sw_arr[:, 0, -1])
        except Exception:
            # Fallback: if sw_paw already holds frame indices list/1D
            sw_frames = np.int64(np.asarray(sw_paw).ravel())
        # Remove non-finite and sort
        sw_frames = sw_frames[np.isfinite(sw_frames)]
        sw_frames = sw_frames[(sw_frames >= 0) & (sw_frames < x_filt.size)]
        sw_frames = np.unique(np.sort(sw_frames))
        if sw_frames.size < 2:
            print(f"[WARN] Not enough swing cycles for trial '{f}' paw '{paw_name}'")
            continue

        # Resample each stride segment swing->next swing to common length
        target_n = 200
        resampled_strides = []
        for s in range(sw_frames.size - 1):
            i0 = sw_frames[s]
            i1 = sw_frames[s + 1]
            if i1 - i0 < 5:
                continue
            xs = x_filt[i0:i1]
            zs = z_filt[i0:i1]
            # Param along segment
            t = np.linspace(0.0, 1.0, xs.size)
            t_new = np.linspace(0.0, 1.0, target_n)
            # Linear interpolation
            x_res = np.interp(t_new, t, xs)
            z_res = np.interp(t_new, t, zs)
            resampled_strides.append((x_res, z_res))

        if not resampled_strides:
            print(f"[WARN] No valid stride segments after resampling for trial '{f}' paw '{paw_name}'")
            continue

        trial_num = parse_trial_number(f)
        c = color_for_trial(trial_num)
        # Combine all resampled strides into a single trace with NaN separators
        x_concat = []
        z_concat = []
        sep_x = np.array([np.nan])
        sep_z = np.array([np.nan])
        for (x_res, z_res) in resampled_strides:
            x_concat.append(x_res)
            z_concat.append(z_res)
            # separator between segments
            x_concat.append(sep_x)
            z_concat.append(sep_z)
        x_concat = np.concatenate(x_concat) if x_concat else np.array([])
        z_concat = np.concatenate(z_concat) if z_concat else np.array([])
        fig.add_trace(go.Scatter(
            x=x_concat,
            y=z_concat,
            mode='lines',
                line=dict(color=c, width=1),
                opacity=0.3,
            name=f"trial {t_idx+1}"
        ))

        # Average stride per trial (width 2)
        X_mat = np.stack([x for (x, _) in resampled_strides], axis=0)
        Z_mat = np.stack([z for (_, z) in resampled_strides], axis=0)
        x_avg = np.nanmean(X_mat, axis=0)
        z_avg = np.nanmean(Z_mat, axis=0)
        fig.add_trace(go.Scatter(
            x=x_avg,
            y=z_avg,
            mode='lines',
            line=dict(color=c, width=2),
            name=f"trial {t_idx+1} avg"
        ))

    # Set axis labels based on signal type
    if signal_type == 'velocity':
        title_suffix = 'Velocities'
        x_label = 'Vx (mm/s)'
        y_label = 'Vz (mm/s)'
        file_suffix = 'velocities'
    else:
        title_suffix = 'Trajectories'
        x_label = 'X (mm)'
        y_label = 'Z rel to body (mm)'
        file_suffix = 'trajectories'

    fig.update_layout(
        title=f"{animal} - Overlapped X-Z {title_suffix} ({paw_name})",
        xaxis_title=x_label,
        yaxis_title=y_label,
        showlegend=True,
        width=900,
        height=700
    )
    # Show and save
    fig.show()
    # Ensure kinematics directory exists
    save_dir = os.path.join(path, 'kinematics')
    os.makedirs(save_dir, exist_ok=True)
    out_name = os.path.join(save_dir, f"{animal}_{paw_name}_xz_{file_suffix}_over_trials.html")
    fig.write_html(out_name)
    print(f"[INFO] Saved: {out_name}")


def plot_contralateral_distances(path, animal, session):
    """Plot continuous frame-by-frame distances between contralateral paws (FR–HL, FL–HR) across all trials for an animal.
    Saves HTML to the kinematics folder.
    """
    loco = loco_class(path)
    filelist = loco.get_track_files(animal, session)
    # Local helpers for trial parsing and coloring (mirror logic used above)
    def parse_trial_number(filename: str) -> int:
        try:
            base = str(filename).split('DLC')[0].strip('_')
            parts = base.split('_')
            return int(parts[-1])
        except Exception:
            return -1
    def shade_color(hex_color: str, factor: float) -> str:
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        if factor <= 1.0:
            r = int(r * factor); g = int(g * factor); b = int(b * factor)
        else:
            alpha = min(factor - 1.0, 0.5)
            r = int(r + (255 - r) * alpha)
            g = int(g + (255 - g) * alpha)
            b = int(b + (255 - b) * alpha)
        return f"#{r:02x}{g:02x}{b:02x}"
    def color_for_trial(trial_num: int) -> str:
        tied_end = STIM_START - 1
        stim_end = STIM_START + STIM_DURATION - 1
        last_start = stim_end + 1
        if 1 <= trial_num <= tied_end:
            return '#1f77b4'
        if STIM_START <= trial_num <= stim_end:
            steps = stim_end - STIM_START
            idx = trial_num - STIM_START
            start_factor = 1.50
            end_factor = 0.50
            factor = start_factor - ((start_factor - end_factor) * (idx / max(steps, 1)))
            return shade_color('#d62728', factor)
        if last_start <= trial_num <= NTRIALS:
            steps = NTRIALS - last_start
            idx = trial_num - last_start
            start_factor = 1.50
            end_factor = 0.50
            factor = start_factor - ((start_factor - end_factor) * (idx / max(steps, 1)))
            return shade_color('#2ca02c', factor)
        return '#7f7f7f'
    fr_hl_concat = []
    fl_hr_concat = []
    diff_concat = []
    frame_concat = []
    sep = np.array([np.nan], dtype=float)
    frame_sep = np.array([np.nan], dtype=float)
    frame_offset = 0
    trial_start_frames = {}
    trial_lengths = {}
    # Figure for power spectrum of per-trial difference signal
    fig_psd = go.Figure()
    for f in filelist:
        try:
            final_tracks, *_ = loco.read_h5(f, 0.9, 0)
        except Exception as e:
            print(f"[WARN] Skipping trial '{f}' for distance plot due to read error: {e}")
            continue
        # Record start frame (cumulative index) keyed by trial number parsed from filename
        trial_num = None
        try:
            trial_num = int(str(f).split('DLC')[0].strip('_').split('_')[-1])
        except Exception:
            # Fallback: leave None; we'll still record sequentially if needed
            pass
        if trial_num is not None:
            trial_start_frames[trial_num] = frame_offset
        # Use positions relative to body center for robustness
        paws_rel_x = loco.get_paws_rel(final_tracks, 'X')
        paws_rel_y = loco.get_paws_rel(final_tracks, 'Y')
        X_arr = np.asarray(paws_rel_x)
        Y_arr = np.asarray(paws_rel_y)
        i_FR = PAW_IDX['FR']; i_HL = PAW_IDX['HL']; i_FL = PAW_IDX['FL']; i_HR = PAW_IDX['HR']
        try:
            x_FR = X_arr[i_FR, :]; y_FR = Y_arr[i_FR, :]
            x_HL = X_arr[i_HL, :]; y_HL = Y_arr[i_HL, :]
            x_FL = X_arr[i_FL, :]; y_FL = Y_arr[i_FL, :]
            x_HR = X_arr[i_HR, :]; y_HR = Y_arr[i_HR, :]
        except Exception:
            x_FR = np.asarray(paws_rel_x[i_FR]); y_FR = np.asarray(paws_rel_y[i_FR])
            x_HL = np.asarray(paws_rel_x[i_HL]); y_HL = np.asarray(paws_rel_y[i_HL])
            x_FL = np.asarray(paws_rel_x[i_FL]); y_FL = np.asarray(paws_rel_y[i_FL])
            x_HR = np.asarray(paws_rel_x[i_HR]); y_HR = np.asarray(paws_rel_y[i_HR])
        # Inpaint NaNs
        x_FR = loco.inpaint_nans(x_FR); y_FR = loco.inpaint_nans(y_FR)
        x_HL = loco.inpaint_nans(x_HL); y_HL = loco.inpaint_nans(y_HL)
        x_FL = loco.inpaint_nans(x_FL); y_FL = loco.inpaint_nans(y_FL)
        x_HR = loco.inpaint_nans(x_HR); y_HR = loco.inpaint_nans(y_HR)
        # Savitzky–Golay filter (window=5, polyorder=1) like above
        x_FR = savgol_filter(x_FR, window_length=5, polyorder=1); y_FR = savgol_filter(y_FR, window_length=5, polyorder=1)
        x_HL = savgol_filter(x_HL, window_length=5, polyorder=1); y_HL = savgol_filter(y_HL, window_length=5, polyorder=1)
        x_FL = savgol_filter(x_FL, window_length=5, polyorder=1); y_FL = savgol_filter(y_FL, window_length=5, polyorder=1)
        x_HR = savgol_filter(x_HR, window_length=5, polyorder=1); y_HR = savgol_filter(y_HR, window_length=5, polyorder=1)
        # Distances
        d_FR_HL = np.sqrt((x_FR - x_HL)**2 + (y_FR - y_HL)**2)
        d_FL_HR = np.sqrt((x_FL - x_HR)**2 + (y_FL - y_HR)**2)
        d_DIFF = d_FR_HL - d_FL_HR
        # Add power spectrum of difference for this trial
        try:
            tnum_psd = parse_trial_number(f)
        except Exception:
            tnum_psd = -1
        color_psd = color_for_trial(tnum_psd)
        n_psd = d_DIFF.size
        if n_psd > 1:
            # Remove DC component
            d_DIFF_demean = d_DIFF - np.nanmean(d_DIFF)
            # Welch PSD in Hz using loco.sr; target ~1 Hz resolution using nfft ~= sr
            sr = float(loco.sr)
            nfft = int(round(sr))  # one bin per Hz
            nperseg = min(max(256, nfft), n_psd)  # ensure segment length is reasonable and <= signal length
            noverlap = nperseg // 2
            freqs, power = welch(d_DIFF_demean, fs=sr, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=False)
            # Limit to 0–50 Hz and ensure one sample per Hz (bins already ~1 Hz)
            mask = (freqs >= 0) & (freqs <= 50)
            freqs_50 = freqs[mask]
            power_50 = power[mask]
            fig_psd.add_trace(go.Scatter(x=freqs_50, y=power_50, mode='lines', line=dict(color=color_psd, width=1), name=f"trial {tnum_psd if tnum_psd!=-1 else '?'}"))
        # Per-trial range and concatenation
        n = int(d_FR_HL.size)
        if trial_num is not None:
            trial_lengths[trial_num] = n
        # Use cumulative frame index to avoid overlapping x values across trials
        frame_range = np.arange(frame_offset, frame_offset + n, dtype=float)
        fr_hl_concat.append(d_FR_HL); fr_hl_concat.append(sep)
        fl_hr_concat.append(d_FL_HR); fl_hr_concat.append(sep)
        diff_concat.append(d_DIFF); diff_concat.append(sep)
        frame_concat.append(frame_range); frame_concat.append(frame_sep)
        # Advance offset (add +1 for the NaN separator as a gap)
        frame_offset += n + 1
    # Build traces
    fr_hl_concat = np.concatenate(fr_hl_concat) if fr_hl_concat else np.array([])
    fl_hr_concat = np.concatenate(fl_hr_concat) if fl_hr_concat else np.array([])
    diff_concat = np.concatenate(diff_concat) if diff_concat else np.array([])
    frame_concat = np.concatenate(frame_concat) if frame_concat else np.array([])
    fig = go.Figure()
    # Colors: red for FR–HL, blue for FL–HR, black for difference
    fig.add_trace(go.Scatter(x=frame_concat, y=fr_hl_concat, mode='lines', name='FR–HL distance', line=dict(color='#d62728')))
    fig.add_trace(go.Scatter(x=frame_concat, y=fl_hr_concat, mode='lines', name='FL–HR distance', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=frame_concat, y=diff_concat, mode='lines', name='Difference (FR–HL minus FL–HR)', line=dict(color='black')))
    # Add vertical lines for trial starts (thin gray)
    # Draw thin gray lines at each recorded trial start
    for x0 in sorted(trial_start_frames.values()):
        fig.add_vline(x=x0, line=dict(color='gray', width=1))
    # Add stimulation start and end (thick black) based on trial indices
    tied_end = STIM_START - 1
    stim_end = STIM_START + STIM_DURATION - 1
    if STIM_START in trial_start_frames:
        stim_start_x = trial_start_frames[STIM_START]
        fig.add_vline(x=stim_start_x, line=dict(color='black', width=3))
    # For end of stimulation, mark the end within the last stim trial
    if stim_end in trial_start_frames:
        last_stim_start = trial_start_frames[stim_end]
        # Find length of last stim trial (fallback to 0 if unavailable)
        last_stim_len = trial_lengths.get(stim_end, 0)
        stim_end_x = last_stim_start + max(last_stim_len - 1, 0)
        fig.add_vline(x=stim_end_x, line=dict(color='black', width=3))
    fig.update_layout(
        title=f"{animal} - Contralateral Paw Distances (frame-by-frame)",
        xaxis_title='Frame',
        yaxis_title='Distance (mm)',
        showlegend=True,
        width=900,
        height=500
    )
    fig.show()
    save_dir = os.path.join(path, 'kinematics')
    os.makedirs(save_dir, exist_ok=True)
    out_name = os.path.join(save_dir, f"{animal}_contralateral_paw_distances.html")
    fig.write_html(out_name)
    print(f"[INFO] Saved: {out_name}")

    # Finalize and save the power spectrum figure
    fig_psd.update_layout(
        title=f"{animal} - Power Spectrum of Difference (FR–HL minus FL–HR)",
        xaxis_title='Frequency (Hz)',
        yaxis_title='Power (log scale)',
        yaxis_type='log',
        showlegend=True,
        width=900,
        height=500
    )
    fig_psd.show()
    out_name_psd = os.path.join(save_dir, f"{animal}_contralateral_diff_power_spectrum.html")
    fig_psd.write_html(out_name_psd)
    print(f"[INFO] Saved: {out_name_psd}")


def plot_hilbert_relative_phase(path, animal, session):
    """Hilbert transform analysis using FR–HL and FL–HR distance signals.
    For each trial: compute instantaneous phase via Hilbert, relative phase Δφ(t) = wrap(phi_FRHL - phi_FLHR),
    and circular statistics (mean phase, coherence R). Plot signals and Δφ(t). Save per-animal figure."""
    loco = loco_class(path)
    filelist = loco.get_track_files(animal, session)
    sr = float(loco.sr)
    # Containers for plotting across trials concatenated
    t_concat = []
    FRHL_concat = []
    FLHR_concat = []
    dphi_concat = []
    sep = np.array([np.nan], dtype=float)
    t_sep = np.array([np.nan], dtype=float)
    t_offset = 0.0

    def wrap(phi):
        return (phi + np.pi) % (2*np.pi) - np.pi

    stats_lines = []  # text summaries per trial
    for f in filelist:
        try:
            final_tracks, *_ = loco.read_h5(f, 0.9, 0)
        except Exception as e:
            print(f"[WARN] Skipping trial '{f}' for Hilbert due to read error: {e}")
            continue
        # Relative positions
        paws_rel_x = loco.get_paws_rel(final_tracks, 'X')
        paws_rel_y = loco.get_paws_rel(final_tracks, 'Y')
        X_arr = np.asarray(paws_rel_x)
        Y_arr = np.asarray(paws_rel_y)
        i_FR = PAW_IDX['FR']; i_HL = PAW_IDX['HL']; i_FL = PAW_IDX['FL']; i_HR = PAW_IDX['HR']
        try:
            x_FR = X_arr[i_FR, :]; y_FR = Y_arr[i_FR, :]
            x_HL = X_arr[i_HL, :]; y_HL = Y_arr[i_HL, :]
            x_FL = X_arr[i_FL, :]; y_FL = Y_arr[i_FL, :]
            x_HR = X_arr[i_HR, :]; y_HR = Y_arr[i_HR, :]
        except Exception:
            x_FR = np.asarray(paws_rel_x[i_FR]); y_FR = np.asarray(paws_rel_y[i_FR])
            x_HL = np.asarray(paws_rel_x[i_HL]); y_HL = np.asarray(paws_rel_y[i_HL])
            x_FL = np.asarray(paws_rel_x[i_FL]); y_FL = np.asarray(paws_rel_y[i_FL])
            x_HR = np.asarray(paws_rel_x[i_HR]); y_HR = np.asarray(paws_rel_y[i_HR])
        # Inpaint + filter
        x_FR = loco.inpaint_nans(x_FR); y_FR = loco.inpaint_nans(y_FR)
        x_HL = loco.inpaint_nans(x_HL); y_HL = loco.inpaint_nans(y_HL)
        x_FL = loco.inpaint_nans(x_FL); y_FL = loco.inpaint_nans(y_FL)
        x_HR = loco.inpaint_nans(x_HR); y_HR = loco.inpaint_nans(y_HR)
        x_FR = savgol_filter(x_FR, window_length=5, polyorder=1); y_FR = savgol_filter(y_FR, window_length=5, polyorder=1)
        x_HL = savgol_filter(x_HL, window_length=5, polyorder=1); y_HL = savgol_filter(y_HL, window_length=5, polyorder=1)
        x_FL = savgol_filter(x_FL, window_length=5, polyorder=1); y_FL = savgol_filter(y_FL, window_length=5, polyorder=1)
        x_HR = savgol_filter(x_HR, window_length=5, polyorder=1); y_HR = savgol_filter(y_HR, window_length=5, polyorder=1)
        # Distances
        d_FR_HL = np.sqrt((x_FR - x_HL)**2 + (y_FR - y_HL)**2)
        d_FL_HR = np.sqrt((x_FL - x_HR)**2 + (y_FL - y_HR)**2)
        # Demean and normalize (optional: amplitude normalization to improve phase stability)
        a = d_FR_HL - np.nanmean(d_FR_HL)
        b = d_FL_HR - np.nanmean(d_FL_HR)
        n = min(a.size, b.size)
        a = a[:n]; b = b[:n]
        if n < 5:
            print(f"[WARN] Trial '{f}' too short for Hilbert analysis")
            continue
        # Time axis for this trial
        t_local = np.arange(n) / sr
        # Hilbert transform to get analytic signals and unwrapped phases
        phi_a = np.unwrap(np.angle(hilbert(a)))
        phi_b = np.unwrap(np.angle(hilbert(b)))
        # Relative phase wrapped to [-pi, pi]
        dphi = wrap(phi_a - phi_b)
        # Circular statistics
        mean_rel_phase = circmean(dphi, high=np.pi, low=-np.pi)
        R = np.abs(np.mean(np.exp(1j*dphi)))
        # Record for concatenated plotting
        FRHL_concat.append(a); FRHL_concat.append(sep)
        FLHR_concat.append(b); FLHR_concat.append(sep)
        dphi_concat.append(dphi); dphi_concat.append(sep)
        t_concat.append(t_local + t_offset); t_concat.append(t_sep)
        # Advance offset
        t_offset += (n / sr) + (1.0 / sr)
        # Stats line
        try:
            tnum = int(str(f).split('DLC')[0].strip('_').split('_')[-1])
        except Exception:
            tnum = -1
        stats_lines.append(f"trial {tnum}: mean_phase={mean_rel_phase:.3f} rad ({np.degrees(mean_rel_phase):.1f} deg), R={R:.3f}")

    # Build concatenated arrays
    FRHL_concat = np.concatenate(FRHL_concat) if FRHL_concat else np.array([])
    FLHR_concat = np.concatenate(FLHR_concat) if FLHR_concat else np.array([])
    dphi_concat = np.concatenate(dphi_concat) if dphi_concat else np.array([])
    t_concat = np.concatenate(t_concat) if t_concat else np.array([])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_concat, y=FRHL_concat, mode='lines', name='FR–HL (demeaned)', line=dict(color='#d62728', width=1)))
    fig.add_trace(go.Scatter(x=t_concat, y=FLHR_concat, mode='lines', name='FL–HR (demeaned)', line=dict(color='#1f77b4', width=1)))
    fig.add_trace(go.Scatter(x=t_concat, y=dphi_concat, mode='lines', name='Δφ(t) = φ_FRHL − φ_FLHR', line=dict(color='black', width=2)))
    fig.update_layout(
        title=f"{animal} - Hilbert Relative Phase (concatenated trials)",
        xaxis_title='Time (s)',
        yaxis_title='Amplitude / Phase (rad)',
        showlegend=True,
        width=950,
        height=600,
        legend=dict(itemsizing='trace')
    )
    fig.show()
    save_dir = os.path.join(path, 'kinematics')
    os.makedirs(save_dir, exist_ok=True)
    out_name = os.path.join(save_dir, f"{animal}_hilbert_relative_phase.html")
    fig.write_html(out_name)
    print(f"[INFO] Saved: {out_name}")
    # Log stats
    for s in stats_lines:
        print("[STATS] ", s)

def compute_trial_lag_ms(path, animal, session):
    """Compute per-trial lag (in ms) between FR–HL and FL–HR distance signals for a given animal.
    Returns (trial_numbers, lag_ms_array). Uses cross-correlation on demeaned, filtered signals.
    """
    loco = loco_class(path)
    filelist = loco.get_track_files(animal, session)
    trial_nums = []
    lags_ms = []

    def parse_trial_number(filename: str) -> int:
        try:
            base = str(filename).split('DLC')[0].strip('_')
            parts = base.split('_')
            return int(parts[-1])
        except Exception:
            return -1

    sr = float(loco.sr)
    max_lag_seconds = 2.0  # search up to +/- 2 seconds
    max_lag = int(round(max_lag_seconds * sr))

    for f in filelist:
        try:
            final_tracks, *_ = loco.read_h5(f, 0.9, 0)
        except Exception as e:
            print(f"[WARN] Skipping trial '{f}' for lag due to read error: {e}")
            continue
        tnum = parse_trial_number(f)
        # Relative positions
        paws_rel_x = loco.get_paws_rel(final_tracks, 'X')
        paws_rel_y = loco.get_paws_rel(final_tracks, 'Y')
        X_arr = np.asarray(paws_rel_x)
        Y_arr = np.asarray(paws_rel_y)
        i_FR = PAW_IDX['FR']; i_HL = PAW_IDX['HL']; i_FL = PAW_IDX['FL']; i_HR = PAW_IDX['HR']
        try:
            x_FR = X_arr[i_FR, :]; y_FR = Y_arr[i_FR, :]
            x_HL = X_arr[i_HL, :]; y_HL = Y_arr[i_HL, :]
            x_FL = X_arr[i_FL, :]; y_FL = Y_arr[i_FL, :]
            x_HR = X_arr[i_HR, :]; y_HR = Y_arr[i_HR, :]
        except Exception:
            x_FR = np.asarray(paws_rel_x[i_FR]); y_FR = np.asarray(paws_rel_y[i_FR])
            x_HL = np.asarray(paws_rel_x[i_HL]); y_HL = np.asarray(paws_rel_y[i_HL])
            x_FL = np.asarray(paws_rel_x[i_FL]); y_FL = np.asarray(paws_rel_y[i_FL])
            x_HR = np.asarray(paws_rel_x[i_HR]); y_HR = np.asarray(paws_rel_y[i_HR])
        # Inpaint NaNs
        x_FR = loco.inpaint_nans(x_FR); y_FR = loco.inpaint_nans(y_FR)
        x_HL = loco.inpaint_nans(x_HL); y_HL = loco.inpaint_nans(y_HL)
        x_FL = loco.inpaint_nans(x_FL); y_FL = loco.inpaint_nans(y_FL)
        x_HR = loco.inpaint_nans(x_HR); y_HR = loco.inpaint_nans(y_HR)
        # Filter
        x_FR = savgol_filter(x_FR, window_length=5, polyorder=1); y_FR = savgol_filter(y_FR, window_length=5, polyorder=1)
        x_HL = savgol_filter(x_HL, window_length=5, polyorder=1); y_HL = savgol_filter(y_HL, window_length=5, polyorder=1)
        x_FL = savgol_filter(x_FL, window_length=5, polyorder=1); y_FL = savgol_filter(y_FL, window_length=5, polyorder=1)
        x_HR = savgol_filter(x_HR, window_length=5, polyorder=1); y_HR = savgol_filter(y_HR, window_length=5, polyorder=1)
        # Distances
        d_FR_HL = np.sqrt((x_FR - x_HL)**2 + (y_FR - y_HL)**2)
        d_FL_HR = np.sqrt((x_FL - x_HR)**2 + (y_FL - y_HR)**2)
        # Demean
        a = d_FR_HL - np.nanmean(d_FR_HL)
        b = d_FL_HR - np.nanmean(d_FL_HR)
        n = min(a.size, b.size)
        if n < 3:
            print(f"[WARN] Trial '{f}' too short for lag computation")
            continue
        # Truncate to same length
        a = a[:n]; b = b[:n]
        # Limit lag search
        L = min(max_lag, n - 1)
        # Compute cross-correlation for lags in [-L, L]
        # Use FFT-based correlate for speed if available; else fallback to slice-based
        corr_full = correlate(a, b, mode='full')
        lags = np.arange(-n + 1, n)
        # Mask to desired lag window
        mask = (lags >= -L) & (lags <= L)
        lags_win = lags[mask]
        corr_win = corr_full[mask]
        # Find lag with max correlation
        k = int(lags_win[np.nanargmax(corr_win)])
        lag_ms = (k / sr) * 1000.0
        trial_nums.append(tnum)
        lags_ms.append(lag_ms)

    # Sort by trial number if available
    if trial_nums:
        order = np.argsort(np.array(trial_nums))
        trial_nums = list(np.array(trial_nums)[order])
        lags_ms = list(np.array(lags_ms)[order])
    return trial_nums, lags_ms


def plot_overlapped_lag_per_animal(path, animals, session):
    """Plot lag (ms) between FR–HL and FL–HR per trial, one line per animal overlapped,
    plus a thicker average line across animals per trial."""
    fig = go.Figure()
    # Collect per-animal lag by trial mapping
    per_animal = []  # list of dicts {trial_num: lag_ms}
    all_trials_set = set()
    for animal in animals:
        trials, lag_ms = compute_trial_lag_ms(path, animal, session)
        if not trials:
            print(f"[WARN] No lag data for animal {animal}")
            continue
        d = {t: l for t, l in zip(trials, lag_ms)}
        per_animal.append((animal, d))
        all_trials_set.update(trials)
        # Plot individual animal line (thinner)
        fig.add_trace(go.Scatter(
            x=trials,
            y=lag_ms,
            mode='lines+markers',
            line=dict(width=1),
            name=animal
        ))
    # Compute average across animals per trial (only where values exist)
    if per_animal:
        all_trials = sorted(all_trials_set)
        avg_vals = []
        for t in all_trials:
            vals = []
            for _, d in per_animal:
                if t in d and np.isfinite(d[t]):
                    vals.append(d[t])
            avg_vals.append(np.nan if len(vals) == 0 else float(np.mean(vals)))
        # Remove trials with NaN average
        avg_x = []
        avg_y = []
        for t, v in zip(all_trials, avg_vals):
            if np.isfinite(v):
                avg_x.append(t)
                avg_y.append(v)
        if avg_x:
            fig.add_trace(go.Scatter(
                x=avg_x,
                y=avg_y,
                mode='lines+markers',
                line=dict(color='black', width=3),
                name='Average across animals'
            ))
    fig.update_layout(
        title='Lag between FR–HL and FL–HR per trial (one line per animal)',
        xaxis_title='Trial number',
        yaxis_title='Lag (ms; positive means FR–HL leads)',
        showlegend=True,
        width=900,
        height=500
    )
    fig.show()
    # Save
    save_dir = os.path.join(path, 'kinematics')
    os.makedirs(save_dir, exist_ok=True)
    out_name = os.path.join(save_dir, f"lag_FRHL_vs_FLHR_over_trials_all_animals.html")
    fig.write_html(out_name)
    print(f"[INFO] Saved: {out_name}")

if __name__ == '__main__':
    for animal in ANIMALS:
        plot_animal_xz_trajectories(PATH, animal, SESSION, PAW, signal_type=SIGNAL_TYPE, frame_rate=FRAME_RATE)
        plot_contralateral_distances(PATH, animal, SESSION)
        plot_hilbert_relative_phase(PATH, animal, SESSION)
    # Plot overlapped lag across animals
    plot_overlapped_lag_per_animal(PATH, ANIMALS, SESSION)
