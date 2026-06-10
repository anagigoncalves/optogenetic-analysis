"""
Kinematic and laser timing analysis for optogenetic treadmill sessions.

This script loads processed tracking data per animal/session, extracts paw trajectories,
resamples strides centered on swing/stance, computes per-trial/animal averages, and
generates outputs:
- Stacked stride position images with laser-on intervals
- Histograms of stride and laser onset/offset timings
- Pickled average trajectory/time data for downstream analysis

Created on Tue Oct 15 14:42:12 2024
Author: Alice Geminiani
"""

# Kinematic analysis of individual limbs
import online_tracking_class
import locomotion_class
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend to avoid Tk pixmap allocation errors
import matplotlib.pyplot as plt
import kinematic_functions
import utils
import gc
from itertools import chain
import pickle



path='D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\tied swing stim CL-Ali\\'
path='D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT\\LOW expression\\ALL_ANIMALS\\tied th200st IO 50ms CL-Ali\\'
#path='D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT\\LOW expression\\ALL_ANIMALS\\tied th100sw IO 50ms CL-Ali\\'

# Fill these in to enable the convolution-based expected CSpk plots.
cspk_histogram_folder = 'C:\\Users\\Utilizador\\Carey Lab Dropbox\\Alice Geminiani\\LocoCF-Data\\Analysis\\Optogenetics\\Ephys\\'
cspk_histogram_dataset = 'CAMKII_CHR2_60s_PULSED_50ms_laser_IO_5.0_ms_bins_wrapped'           #  '_CAMKII_Jaws_60sPULSED75ms_40mw_laser_5.0_ms_bins_wrapped'        # 

# Expected external files:
#   <cspk_histogram_folder>/<cspk_histogram_dataset>_data.csv
#   <cspk_histogram_folder>/<cspk_histogram_dataset>_metadata.json
# Histogram CSV columns: bin_time_sec,pkj_cs_probability_per_trial
# Metadata JSON keys: bin_size_sec,laser_pulse_duration_sec
CSPK_HISTOGRAM_SUFFIX = '_data.csv'
CSPK_METADATA_SUFFIX = '_metadata.json'
CSPK_HISTOGRAM_TIME_COLUMN = 'bin_time_sec'
CSPK_HISTOGRAM_COUNT_COLUMN = 'pkj_cs_probability_per_trial'
CSPK_METADATA_BIN_SIZE_COLUMN = 'bin_size_sec'
CSPK_METADATA_LASER_DURATION_COLUMN = 'laser_pulse_duration_sec'

paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']
paw_names = ['FR']      #, 'HR', 'FL', 'HL']

to_plot = ['x']
center = 'st'
force_center = False        # 0 no forcing, 0.5 forcing in the middle, 0.33 or 0.66 forcing at 33 or 66  %     
hist_all = False            # If True, plot histograms of stride laser onsets and offsets for all stim trials, otherwise of only the first 3 stimulated trials
plot_off_to_on = False              # TODO!!!!!!!!!!!!!!!!!!!!!!!! Manage all cases
plot_only_off = False
stim_start = 9
stim_duration = 10
num_trials = 28
num_resamples = 360
sf = 330        # [Hz] sampling frequency of the camera
time_range = [-200, 200]        # [ms] time range we are going to look at (single stride plots, histograms, etc.)
pixel_to_mm = 1/3.3            # real-time setup
if 'WT' in path:
    pixel_to_mm = 1/1.955   # Dana's setup  

# Laser color scheme based on experiment type
if 'JAWS' in path:
    laser_color_stride_plot = 'yellow'
    onset_face, onset_edge = 'lightyellow', 'yellow'
    offset_face, offset_edge = 'orange', 'darkorange'
else:
    laser_color_stride_plot = 'blue'
    onset_face, onset_edge = 'lightskyblue', 'royalblue'
    offset_face, offset_edge = 'blue', 'darkblue'

# DELAYS from EPHYS
med_airpuff_delay_ms = 44      # [ms] median delay between the airpuff to the paw and the recorded CSpk
if 'JAWS' in path:
    med_cspk_delay_ms = 11      # [ms] median delay between the yellow laser offset and the recorded CSpk (with JAWS)
else:
    med_cspk_delay_ms = 35       # [ms] median delay between the laser onset and the recorded CSpk (with ChR2)

otrack_class = online_tracking_class.otrack_class(path)
loco = locomotion_class.loco_class(path, pixel_to_mm=pixel_to_mm)
folder_name = 'kinematics_laser_timing'

base_output_dir = os.path.join(path, folder_name)
path_save = os.path.join(
    base_output_dir,
    f"{center}_centered_force_center{force_center}_OFFtoON{plot_off_to_on}"
)
os.makedirs(utils.win_safe_path(path_save), exist_ok=True)
print("Analysing..........................", path)



# GET THE NUMBER OF ANIMALS AND THE SESSION ID
animal_session_list = loco.animals_within_session()
animal_list = []
for a in range(len(animal_session_list)):
    animal_list.append(animal_session_list[a][0])

session_list = []
for a in range(len(animal_session_list)):
    session_list.append(animal_session_list[a][1])


# All included animals
included_animal_list = ['MC16848', 'MC16851', 'MC17319', 'MC17665', 'MC17666', 'MC17669', 'MC17670', 'MC19082',
                      'MC19124', 'MC19214', 'VIV41329', 'VIV41330', 'VIV42375', 'VIV42428', 'VIV42429',
              'VIV42430', 'VIV42376', 'MC19107']

included_animal_list = ['VIV42906', 'VIV42908', 'VIV42974', 'VIV42985','VIV42987','VIV44766', 'VIV45372']  # ChR2

avg_traj_resampled_all_trials_all_animals = {}

for paw in paw_names:
    avg_traj_resampled_all_trials_all_animals[paw] = {}
    for axis in to_plot:        avg_traj_resampled_all_trials_all_animals[paw][axis] = np.full((len(animal_list), num_trials, num_resamples), np.nan) 
        
# FOR EACH SESSION AND ANIMAL EXTRACT PAW POSITIONS in 3D
stride_onsets_all_trials_all_animals = []        # List to store stride onsets for each trial for all animals
stride_offsets_all_trials_all_animals = []        # List to store stride offsets for each trial for all animals
stride_laser_onsets_all_trials_all_animals = []        # List to store stride laser onsets for each trial for all animals
stride_laser_offsets_all_trials_all_animals = []        # List to store stride laser offsets for each trial for all animals

# Create per-trial figures for all-animals comparison (one figure per stimulated trial)
import math
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
n_animals = len(included_animal_list)
n_rows = 3
n_cols = math.ceil(n_animals / n_rows)
figs_per_trial = {}
axes_per_trial = {}
for trial_idx in range(stim_start, stim_start + stim_duration):
    fig = Figure(figsize=(3*n_cols, 3*n_rows))
    FigureCanvasAgg(fig)
    axes = fig.subplots(n_rows, n_cols, squeeze=False)
    fig.suptitle(f'Stacked Stride Positions - Trial {trial_idx} - All Animals ({paw_names[0]}, {to_plot[0]})', fontsize=14)
    figs_per_trial[trial_idx] = fig
    axes_per_trial[trial_idx] = axes

for count_animal, animal in enumerate(included_animal_list):
    print('Processing ' + animal)
    # Look up the correct session for this animal (included list can be a subset in different order)
    session_matches = [sess for (a, sess) in animal_session_list if a == animal]
    if len(session_matches) == 0:
        print(f"No session found for {animal}. Skipping...")
        continue
    session = int(session_matches[0])
    trials = otrack_class.get_trials(animal)
    # Initialize dictionaries to store the resampled trajectories for each paw and axis
    traj_resampled_all_trials = {}
    for paw in paw_names:
        traj_resampled_all_trials[paw] = {}
        for axis in to_plot:
            traj_resampled_all_trials[paw][axis] = []

    #TODO: check if this filelist needs to be emptied first!
    filelist = loco.get_track_files(animal, session)

    # LOAD PROCESSED DATA
    # Check first if the current animal has processed files
    print(os.path.join(path, 'processed files', animal))
    if not os.path.exists(os.path.join(path, 'processed files', animal)):
       print(f"No processed files found for {animal}. Skipping...")
       continue

    [otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, laser_on] = otrack_class.load_processed_files(animal)

    for paw in range(len(paw_names)):       # For each paw
        for axis in to_plot:        # For each axis to plot
            stride_onsets_all_trials = []        # List to store stride onsets for each trial
            stride_offsets_all_trials = []        # List to store stride offsets for each trial
            stride_laser_onsets_all_trials = []        # List to store stride laser onsets for each trial
            stride_laser_offsets_all_trials = []        # List to store stride laser offsets for each trial
            positions_all_stim_trials = []        # List to store stride positions for each trial
            positions_metadata_all_stim_trials = []  # Store (padded_positions, min_time, max_time, laser_onsets, laser_offsets) per trial
            trimmed_positions_all_stim_trials = []        # List to store stride positions for each trial
            trimmed_times_all_stim_trials = []
            for f in filelist:          # For each trial
                count_trial = int(f.split('DLC')[0].split('_')[-1])-1      # Get trial number from file name, to spot any missing trial; parameters for remaining ones will stay to NaN
                print( "Processing animal", animal, "trial", count_trial) 
                
                [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, 0)
                [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
                paws_rel = {'x': loco.get_paws_rel(final_tracks, 'X'), 'y': loco.get_paws_rel(final_tracks, 'Y'), 'z': loco.get_paws_rel(final_tracks, 'Z')}

                # Resample strides for each paw and plot them
                strides_traj_resampled = kinematic_functions.resample_strides_position(paws_rel[axis], st_strides_mat, sw_pts_mat, paw, num_resamples, center=center, force_center=force_center)
                #kinematic_functions.plot_resampled_position(traj_resampled, axis, paw_colors[paw], paw_names[paw], animal+'_trial'+str(count_trial), path_save, center=center, force_center=force_center)
                # Save the resampled stride trajectories for each trial
                traj_resampled_all_trials[paw_names[paw]][axis].append(strides_traj_resampled)
                # Save the average resampled trajectory for each trial and each animal
                avg_traj_resampled_all_trials_all_animals[paw_names[paw]][axis][count_animal, count_trial, :] = np.nanmean(strides_traj_resampled, axis=0)

                # PLOT in TIME
                fig_all_trajectories, ax_all_trajectories = plt.subplots()
                # Add vertical zero line
                plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
                step_y = 50
                stride_positions = []
                stride_times = []
                trimmed_stride_times = []
                trimmed_stride_positions = []
                stride_onsets = []
                stride_offsets = []
                stride_laser_onsets = []
                stride_laser_offsets = []
                # Extract current trial laser onset and offset times
                if count_trial + 1 >= stim_start and count_trial + 1 < stim_start + stim_duration:
                    # Select current trial
                    current_trial_laser = laser_on.loc[laser_on['trial'] == count_trial + 1]
                    current_onset_times = np.array(current_trial_laser['time_on'])*1000        # [ms]
                    current_offset_times = np.array(current_trial_laser['time_off'])*1000        # [ms]
                else:
                    current_onset_times = np.array(np.nan)
                    current_offset_times = np.array(np.nan)
                for s in range(len(st_strides_mat[paw])-1):
                    if center == 'st':
                        
                        current_stride_position = paws_rel[axis][paw][int(sw_pts_mat[paw][s,0,-1]):int(sw_pts_mat[paw][s+1,0,-1])]
                        current_stride_time = np.linspace(sw_pts_mat[paw][s,0,0],sw_pts_mat[paw][s+1,0,0], len(current_stride_position))
                        
                        # Put stance moment at 0
                        stance_onset = st_strides_mat[paw][s,1,0]
                        current_stride_time = current_stride_time - stance_onset

                        # Store stride onsets and offsets
                        current_stride_onset = sw_pts_mat[paw][s,0,0] - stance_onset
                        current_stride_offset = sw_pts_mat[paw][s+1,0,0] - stance_onset

                        # Initialize default values for current stride laser onset and offset
                        current_stride_laser_onset = np.nan
                        current_stride_laser_offset = np.nan

                        # Check if any onset and offset times are within the current stride interval and center around stance
                        if count_trial + 1 >= stim_start and count_trial + 1 < stim_start + stim_duration:
                            if plot_off_to_on:
                                for l in range(len(current_onset_times)-1):
                                    if sw_pts_mat[paw][s, 0, 0] <= current_offset_times[l] < current_onset_times[l+1] <= sw_pts_mat[paw][s + 1, 0, 0]:
                                        current_stride_laser_onset = current_offset_times[l] - stance_onset
                                        current_stride_laser_offset = current_onset_times[l+1] - stance_onset
                                        break   # Found the first valid onset and offset, no need to check further
                                    # Just offset within stride bounds
                                    if sw_pts_mat[paw][s, 0, 0] <= current_offset_times[l] <= sw_pts_mat[paw][s + 1, 0, 0] and current_onset_times[l+1] > sw_pts_mat[paw][s + 1, 0, 0]:
                                        current_stride_laser_offset = current_offset_times[l] - stance_onset
                                        current_stride_laser_onset = sw_pts_mat[paw][s + 1, 0, 0] - stance_onset
                                        break
                                    # Just onset within stride bounds
                                    if sw_pts_mat[paw][s, 0, 0] <= current_onset_times[l+1] <= sw_pts_mat[paw][s + 1, 0, 0] and current_offset_times[l] < sw_pts_mat[paw][s, 0, 0]:
                                        current_stride_laser_onset = sw_pts_mat[paw][s, 0, 0] - stance_onset
                                        current_stride_laser_offset = current_onset_times[l+1] - stance_onset
                                        break
                            else:
                                for onset_time, offset_time in zip(current_onset_times, current_offset_times):
                                    # Onset and offset within stride bounds
                                    if sw_pts_mat[paw][s, 0, 0] <= onset_time < offset_time <= sw_pts_mat[paw][s + 1, 0, 0]:
                                        current_stride_laser_onset = onset_time - stance_onset
                                        current_stride_laser_offset = offset_time - stance_onset
                                        break   # Found the first valid onset and offset, no need to check further
                                    # Just onset within stride bounds
                                    if sw_pts_mat[paw][s, 0, 0] <= onset_time <= sw_pts_mat[paw][s + 1, 0, 0] and offset_time > sw_pts_mat[paw][s + 1, 0, 0]:
                                        current_stride_laser_onset = onset_time - stance_onset
                                        current_stride_laser_offset = sw_pts_mat[paw][s + 1, 0, 0] - stance_onset
                                        break
                                    # Just offset within stride bounds
                                    if sw_pts_mat[paw][s, 0, 0] <= offset_time <= sw_pts_mat[paw][s + 1, 0, 0] and onset_time < sw_pts_mat[paw][s, 0, 0]:
                                        current_stride_laser_onset = sw_pts_mat[paw][s, 0, 0] - stance_onset
                                        current_stride_laser_offset = offset_time - stance_onset
                                        break
                    elif center == 'sw':
                        current_stride_position = paws_rel[axis][paw][int(st_strides_mat[paw][s,0,-1]):int(st_strides_mat[paw][s,1,-1])]
                        current_stride_time = np.linspace(st_strides_mat[paw][s,0,0],st_strides_mat[paw][s,1,0], len(current_stride_position))
                        
                        # Put swing moment at 0
                        swing_onset = sw_pts_mat[paw][s,0,0]
                        current_stride_time = current_stride_time - swing_onset

                        # Store stride onsets and offsets
                        current_stride_onset = st_strides_mat[paw][s,0,0] - swing_onset
                        current_stride_offset = st_strides_mat[paw][s,1,0] - swing_onset

                        # Initialize default values for current stride laser onset and offset
                        current_stride_laser_onset = np.nan
                        current_stride_laser_offset = np.nan

                        # Check if any onset and offset times are within the current stride interval and center around swing
                        if count_trial + 1 >= stim_start and count_trial + 1 < stim_start + stim_duration:
                            if plot_off_to_on:
                                for l in range(len(current_onset_times)-1):
                                    if st_strides_mat[paw][s,0,0] <= current_offset_times[l] < current_onset_times[l+1] <= st_strides_mat[paw][s,1,0]:
                                        current_stride_laser_onset = current_offset_times[l] - st_strides_mat[paw][s,1,0]
                                        current_stride_laser_offset = current_onset_times[l+1] - st_strides_mat[paw][s,1,0]
                                        break   # Found the first valid onset and offset, no need to check further
                                    # Only offset within stride bounds
                                    if st_strides_mat[paw][s,0,0] <= current_offset_times[l] <= st_strides_mat[paw][s,1,0] and current_onset_times[l+1] > st_strides_mat[paw][s,1,0]:
                                        current_stride_laser_onset = current_offset_times[l] - swing_onset
                                        current_stride_laser_offset = st_strides_mat[paw][s,1,0] - swing_onset
                                        break
                                    # Only onset within stride bounds
                                    if st_strides_mat[paw][s,0,0] <= current_onset_times[l+1] <= st_strides_mat[paw][s,1,0] and current_offset_times[l] < st_strides_mat[paw][s,0,0]:
                                        current_stride_laser_onset = st_strides_mat[paw][s][0][0] - swing_onset
                                        current_stride_laser_offset = current_onset_times[l+1] - swing_onset
                                        break
                            else:
                                for onset_time, offset_time in zip(current_onset_times, current_offset_times):
                                    if st_strides_mat[paw][s,0,0] <= onset_time < offset_time <= st_strides_mat[paw][s,1,0]:
                                        current_stride_laser_onset = onset_time - swing_onset
                                        current_stride_laser_offset = offset_time - swing_onset
                                        break   # Found the first valid onset and offset, no need to check further
                                    if st_strides_mat[paw][s,0,0] <= onset_time <= st_strides_mat[paw][s,1,0] and offset_time > st_strides_mat[paw][s,1,0]:
                                        current_stride_laser_onset = onset_time - swing_onset
                                        current_stride_laser_offset = st_strides_mat[paw][s,1,0] - swing_onset
                                        break
                                    if st_strides_mat[paw][s,0,0] <= offset_time <= st_strides_mat[paw][s,1,0] and onset_time < st_strides_mat[paw][s,0,0]:
                                        current_stride_laser_onset = st_strides_mat[paw][s,0,0] - swing_onset
                                        current_stride_laser_offset = offset_time - swing_onset
                                        break

                    #ax_all_trajectories.plot(current_stride_time, current_stride_position, color=paw_colors[paw], linewidth=1, alpha=0.2)


                    # Append the position to the list
                    stride_positions.append(current_stride_position)
                    stride_times.append(current_stride_time)

                    stride_onsets.append(current_stride_onset)
                    stride_offsets.append(current_stride_offset)
                    stride_laser_onsets.append(current_stride_laser_onset)
                    stride_laser_offsets.append(current_stride_laser_offset)

                # Catch if no good strides were found
                if len(stride_positions) == 0:
                    print("No good strides found for trial", count_trial + 1, "in animal", animal, "for paw", paw_names[paw], "and axis", axis, ". Skipping trial.")
                    continue
                
                # Plot as 2D image
                # Determine the maximum length of all strides
                max_length = max(len(pos) for pos in stride_positions)
                min_time = np.min([np.min(times) for times in stride_times])
                max_time = np.max([np.max(times) for times in stride_times])

                # Expected number of samples in the output window
                expected_samples = int((time_range[1] - time_range[0]) / (1000/sf))

                for s in range(len(stride_times)):
                    # Pad to align all strides to the same global time range
                    to_add_before = int(np.ceil((np.min(stride_times[s]) - min_time) / (1000/sf)))
                    to_add_after = int(np.ceil((max_time - np.max(stride_times[s])) / (1000/sf)))
                    
                    stride_times[s] = np.pad(stride_times[s], (to_add_before, to_add_after), constant_values=np.nan)
                    stride_positions[s] = np.pad(stride_positions[s], (to_add_before, to_add_after), constant_values=np.nan)
                    
                    # Find index of time=0 (center point)
                    idx_zero = np.nanargmin(np.abs(stride_times[s]))
                    
                    # Calculate window indices
                    start_idx = idx_zero + int(time_range[0] / (1000/sf))
                    end_idx = idx_zero + int(time_range[1] / (1000/sf))
                    
                    # Extract window with proper boundary handling
                    n = len(stride_times[s])
                    pad_before = max(0, -start_idx)
                    pad_after = max(0, end_idx - n)
                    actual_start = max(0, start_idx)
                    actual_end = min(n, end_idx)
                    
                    times_slice = stride_times[s][actual_start:actual_end]
                    positions_slice = stride_positions[s][actual_start:actual_end]
                    
                    times_padded_trimmed = np.pad(times_slice, (pad_before, pad_after), constant_values=np.nan)
                    positions_padded_trimmed = np.pad(positions_slice, (pad_before, pad_after), constant_values=np.nan)
                    
                    trimmed_stride_times.append(times_padded_trimmed)
                    trimmed_stride_positions.append(positions_padded_trimmed)
                
                # Convert trimmed positions to array (consistent time_range for all strides)
                trimmed_positions_array = np.array(trimmed_stride_positions)
                total_strides = len(stride_laser_onsets)
                stimulated_strides = 0
                for onset in stride_laser_onsets:
                    if onset is None:
                        continue
                    try:
                        if np.isnan(onset):
                            continue
                    except TypeError:
                        pass
                    stimulated_strides += 1
                if total_strides > 0:
                    stim_pct = int(round(100 * stimulated_strides / total_strides))
                else:
                    stim_pct = 0
                stim_title = f"{animal} - {stim_pct}% str stim"
                # Plot the stacked positions as a grayscale image
                fig_stacked, ax_stacked = plt.subplots(figsize=(10, 8))
                print(animal, "trial", count_trial + 1, "paw", paw_names[paw], "axis", axis)
                kinematic_functions.plot_stacked_stride_positions(
                    ax=ax_stacked,
                    padded_positions=trimmed_positions_array,
                    stride_laser_onsets=stride_laser_onsets,
                    stride_laser_offsets=stride_laser_offsets,
                    min_time=time_range[0],
                    max_time=time_range[1],
                    time_range=time_range,
                    axis=axis,
                    paw_name=paw_names[paw],
                    paw_color=paw_colors[paw],
                    laser_color=laser_color_stride_plot,
                    center=center,
                    title='Trial ' + str(count_trial + 1) + ' - ' + paw_names[paw] + ' - ' + axis,
                    show_colorbar=False,
                    show_labels=True
                )
               # fig_stacked.show()
                plt.colorbar(ax_stacked.images[0], ax=ax_stacked, label=axis+' Position')
                save_file = os.path.join(path_save, f"{animal}_trial_{count_trial+1}_stacked_stride_positions.png")
                plt.savefig(utils.win_safe_path(save_file), bbox_inches='tight', dpi=300)
                plt.close(fig_stacked)
                # Plot to per-trial all-animals figure (only for stimulated trials)
                if count_trial + 1 >= stim_start and count_trial + 1 < stim_start + stim_duration:
                    trial_num = count_trial + 1
                    row = count_animal % n_rows
                    col = count_animal // n_rows
                    print("Drawing on all-animals figure: trial", trial_num, "animal", animal, "row", row, "col", col)
                    ax_animal = axes_per_trial[trial_num][row, col]
                    kinematic_functions.plot_stacked_stride_positions(
                        ax=ax_animal,
                        padded_positions=trimmed_positions_array,
                        stride_laser_onsets=stride_laser_onsets,
                        stride_laser_offsets=stride_laser_offsets,
                        min_time=time_range[0],
                        max_time=time_range[1],
                        time_range=time_range,
                        axis=axis,
                        paw_name=paw_names[paw],
                        paw_color=paw_colors[paw],
                        laser_color=laser_color_stride_plot,
                        center=center,
                        title=stim_title,
                        show_colorbar=False,
                        show_xlabel=(row == n_rows - 1),
                        show_ylabel=(row == 1 and col == 0),
                        show_legend=False,
                        legend_loc='upper right',
                        legend_bbox=(1.02, 1.02),
                    )
                    if col != 0:
                        ax_animal.set_ylabel('')
                  #  figs_per_trial[trial_num].show() 
               # plt.show()

                stride_onsets_all_trials.append(stride_onsets)
                stride_offsets_all_trials.append(stride_offsets)
                if count_trial + 1 >= stim_start and count_trial + 1 < stim_start + stim_duration:
                    stride_laser_onsets_all_trials.append(stride_laser_onsets)
                    stride_laser_offsets_all_trials.append(stride_laser_offsets)
                    positions_all_stim_trials.append(trimmed_positions_array)
                    # Store full data needed to reproduce single-trial plot in all-animals figure
                    positions_metadata_all_stim_trials.append({
                        'trimmed_positions': trimmed_positions_array,
                        'laser_onsets': stride_laser_onsets.copy(),
                        'laser_offsets': stride_laser_offsets.copy()
                    })
                    trimmed_positions_all_stim_trials.append(trimmed_stride_positions)
                    trimmed_times_all_stim_trials.append(trimmed_stride_times)
            gc.collect()


            # Histograms of stride laser onsets and offsets
            if hist_all:
                selected_onsets = list(chain.from_iterable(stride_laser_onsets_all_trials))   
                selected_offsets =  list(chain.from_iterable(stride_laser_offsets_all_trials)) 
                selected_positions = list(chain.from_iterable(positions_all_stim_trials)) 
                selected_stride_onsets = list(chain.from_iterable(stride_onsets_all_trials))   
                selected_stride_offsets =  list(chain.from_iterable(stride_offsets_all_trials)) 
            else:
                selected_onsets = list(chain.from_iterable(stride_laser_onsets_all_trials[:3]))  # Select the first 3 trials
                selected_offsets = list(chain.from_iterable(stride_laser_offsets_all_trials[:3]))  # Select the first 3 trials
                selected_positions = list(chain.from_iterable(positions_all_stim_trials[:3]))  # Select the first 3 trials
                selected_stride_onsets = list(chain.from_iterable(stride_onsets_all_trials[:3]))  # Select the first 3 trials
                selected_stride_offsets =  list(chain.from_iterable(stride_offsets_all_trials[:3])) 

            fig, ax1 = plt.subplots(figsize=(14, 8))
            bin_width = 5
            min_edge = min(np.nanmin(selected_onsets), np.nanmin(selected_offsets), np.nanmin(selected_stride_onsets), np.nanmin(selected_stride_offsets))
            max_edge = max(np.nanmax(selected_onsets), np.nanmax(selected_offsets), np.nanmax(selected_stride_onsets), np.nanmax(selected_stride_offsets))
            nbins = np.arange(min_edge, max_edge + bin_width, bin_width)

            ax1.hist(selected_stride_onsets, bins=nbins, alpha=0.3, label='Stride Onsets', color=paw_colors[paw], edgecolor=paw_colors[paw])
            ax1.hist(selected_stride_offsets, bins=nbins, alpha=0.6, label='Stride Offsets', color=paw_colors[paw], edgecolor=paw_colors[paw])
            if not plot_only_off:
                ax1.hist(selected_onsets, bins=nbins, alpha=0.6, label='Laser Onsets', color=onset_face, edgecolor=onset_edge)
                ax1.axvline(x=np.nanmedian(selected_onsets), color=onset_edge, linestyle='-', linewidth=2, label='Med Onset')
            ax1.hist(selected_offsets, bins=nbins, alpha=0.6, label='Laser Offsets', color=offset_face, edgecolor=offset_edge)
            ax1.axvline(x=0, color=paw_colors[paw], linestyle='--', linewidth=1, label=center + ' onset')
            ax1.axvline(x=np.nanmedian(selected_offsets), color=offset_edge, linestyle='-', linewidth=2, label='Med Offset')
            ax1.axvline(x=np.nanmedian(selected_stride_onsets), color=paw_colors[paw], linestyle='-', linewidth=2)
            ax1.axvline(x=np.nanmedian(selected_stride_offsets), color='darkred', linestyle='-', linewidth=2)
            ax1.set_xlabel('Time (ms)', fontsize=14)
            ax1.set_ylabel('Frequency', fontsize=14)
            ax1.set_xlim(time_range[0], time_range[1])
            ax1.legend(fontsize=12, bbox_to_anchor=(1.1, 1), loc='upper left')
            

            # Add avg trajectory
            if hist_all:
                selected_trajectories = np.array(list(chain.from_iterable(trimmed_positions_all_stim_trials)))
                selected_times = np.array(list(chain.from_iterable(trimmed_times_all_stim_trials)))
            else:
                selected_trajectories = np.array(list(chain.from_iterable(trimmed_positions_all_stim_trials[:3])))
                selected_times = np.array(list(chain.from_iterable(trimmed_times_all_stim_trials[:3])))

            avg_trajectory = np.nanmean(selected_trajectories,axis=0) 
            std_trajectory = np.nanstd(selected_trajectories,axis=0)
            avg_time = np.nanmean(selected_times,axis=0)
            with open(os.path.join(path_save, animal+"_avg_trajectory_data.pkl"), "wb") as f:
                pickle.dump({
                    "avg_trajectory": avg_trajectory,
                    "std_trajectory": std_trajectory,
                    "avg_time": avg_time
                }, f)
            
            
            ax2 = ax1.twinx()
            ax2.plot(avg_time, avg_trajectory, color=paw_colors[paw], linewidth=2, label='Avg')
            ax2.fill_between(avg_time, avg_trajectory + std_trajectory, 
                            avg_trajectory - std_trajectory, 
                            color=paw_colors[paw], alpha=0.1, label='Std')
            ax2.set_ylim([np.nanmin(avg_trajectory-std_trajectory), np.nanmax(avg_trajectory+std_trajectory)])
            ax2.set_ylabel('Position (mm)', color=paw_colors[paw], fontsize=14)
            ax2.tick_params(axis='y', labelcolor=paw_colors[paw])
            fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to ensure the title is not cut off
            ax1.tick_params(axis='both', which='major', labelsize=12)
            ax2.tick_params(axis='both', which='major', labelsize=12)
            
            if hist_all:
                plt.title('Histogram of Laser Onsets and Offsets (All Trials)', fontsize=16)
            else:
                plt.title('Histogram of Laser Onsets and Offsets (First 3 Trials)', fontsize=16)
            ax2.legend(fontsize=12)
            hist_file = os.path.join(path_save, f"{animal}_laser_onset_offset_histogram_all_trials_{hist_all}.png")
            plt.savefig(utils.win_safe_path(hist_file), dpi=150)
            plt.close(fig)
            #plt.show()

            # Plot all single trajectories and also the average trajectory
            for t in range(selected_trajectories.shape[0]):
                # Compute velocity
                velocity = np.gradient(selected_trajectories[t], selected_times[t])
                if np.nanquantile(velocity,0.85) > 0.65 or np.nanquantile(velocity,0.15)<-0.65:  # Avoid division by zero
                    ax_all_trajectories.plot(selected_times[t], selected_trajectories[t], color='darkred', linewidth=0.5, alpha=0.5)
                    #ax_all_trajectories.plot(selected_times[t], velocity, color='darkred', linewidth=0.5, alpha=0.5)
                else:
                    ax_all_trajectories.plot(selected_times[t], selected_trajectories[t], color='gray', linewidth=0.5, alpha=0.2)
                    #ax_all_trajectories.plot(selected_times[t], velocity, color='gray', linewidth=0.5, alpha=0.2)
                #ax_all_trajectories.plot(velocity, color='gray', linewidth=0.5, alpha=0.2)
                #ax_all_trajectories.plot(selected_times[t], selected_trajectories[t], color='gray', linewidth=0.5, alpha=0.2)
            ax_all_trajectories.plot(avg_time, avg_trajectory, color=paw_colors[paw], linewidth=2, label='Avg')
            ax_all_trajectories.set_xlim(time_range[0], time_range[1])
            ax_all_trajectories.set_ylim([np.nanmin(avg_trajectory-std_trajectory), np.nanmax(avg_trajectory+std_trajectory)])
            ax_all_trajectories.set_ylabel('Position (mm)', fontsize=14)
            ax_all_trajectories.set_xlabel('Time (ms)', fontsize=14)
           # plt.show()
            hv_file = os.path.join(path_save, f"{animal}_stacked_stride_positions_with_high_peak_velocity_all_trials{hist_all}first_tied.png")
            fig_all_trajectories.savefig(utils.win_safe_path(hv_file), bbox_inches='tight', dpi=150)
            plt.close(fig_all_trajectories)
            # Close only per-trial figures we created here; keep shared multi-animal figures alive
            for _fig in [locals().get('fig_stacked'), locals().get('fig_all_trajectories'), locals().get('fig')]:
                try:
                    plt.close(_fig)
                except Exception:
                    pass
            # Plot the average of all trials for each animal, paw and axis
#            kinematic_functions.plot_resampled_position_all_trials(traj_resampled_all_trials[paw_names[paw]][axis], axis, paw_names[paw], animal, path_save, center=center, force_center=force_center)


     #       kinematic_functions.plot_resampled_position_avg_all(avg_traj_resampled_all_trials_all_animals[paw_names[paw]][axis], axis, paw_names[paw], list(range(28)), path_save, center=center, force_center=force_center)
     #       kinematic_functions.plot_resampled_position_avg_all(avg_traj_resampled_all_trials_all_animals[paw_names[paw]][axis], axis, paw_names[paw], list(range(9)), path_save, center=center, force_center=force_center)
     #       kinematic_functions.plot_resampled_position_avg_all(avg_traj_resampled_all_trials_all_animals[paw_names[paw]][axis], axis, paw_names[paw], list(range(9,19)), path_save, center=center, force_center=force_center)
     #       kinematic_functions.plot_resampled_position_avg_all(avg_traj_resampled_all_trials_all_animals[paw_names[paw]][axis], axis, paw_names[paw], list(range(19,28)), path_save, center=center, force_center=force_center)

    stride_onsets_all_trials_all_animals.append(selected_stride_onsets)        # Append stride onsets for the current animal
    stride_offsets_all_trials_all_animals.append(selected_stride_offsets)        # Append stride offsets for the current animal
    stride_laser_onsets_all_trials_all_animals.append(selected_onsets)        # Append stride laser onsets for the current animal
    stride_laser_offsets_all_trials_all_animals.append(selected_offsets)        # Append stride laser offsets for the current animal
    print("Finished analysing animal", animal)
    gc.collect()  # Clear memory after each animal to avoid memory issues
    

# Do histograms of stride and laser onsets and offsets for all animals
paw=0               # To make flexible for managing multiple paws, but currently only one paw is used
all_onsets = list(chain.from_iterable(stride_laser_onsets_all_trials_all_animals))   
all_offsets =  list(chain.from_iterable(stride_laser_offsets_all_trials_all_animals)) 
all_stride_onsets = list(chain.from_iterable(stride_onsets_all_trials_all_animals))   
all_stride_offsets =  list(chain.from_iterable(stride_offsets_all_trials_all_animals)) 

all_onsets_finite = utils.finite_array(all_onsets)
all_offsets_finite = utils.finite_array(all_offsets)
all_stride_onsets_finite = utils.finite_array(all_stride_onsets)
all_stride_offsets_finite = utils.finite_array(all_stride_offsets)

fig, ax = plt.subplots(figsize=(14, 8))
bin_width = 5
min_edge = min(
    np.nanmin(all_onsets_finite),
    np.nanmin(all_offsets_finite),
    np.nanmin(all_stride_onsets_finite),
    np.nanmin(all_stride_offsets_finite),
)
max_edge = max(
    np.nanmax(all_onsets_finite),
    np.nanmax(all_offsets_finite),
    np.nanmax(all_stride_onsets_finite),
    np.nanmax(all_stride_offsets_finite),
)
nbins = np.arange(min_edge, max_edge + bin_width, bin_width)

convolution_histogram_data = None
if cspk_histogram_folder and cspk_histogram_dataset:
    convolution_histogram_data = utils.load_cspk_histogram_data(
        cspk_histogram_folder,
        cspk_histogram_dataset,
        CSPK_HISTOGRAM_SUFFIX,
        CSPK_METADATA_SUFFIX,
        CSPK_HISTOGRAM_TIME_COLUMN,
        CSPK_HISTOGRAM_COUNT_COLUMN,
        CSPK_METADATA_BIN_SIZE_COLUMN,
        CSPK_METADATA_LASER_DURATION_COLUMN,
    )

ax.hist(all_stride_onsets, bins=nbins, alpha=0.3, label='Stride Onsets', color=paw_colors[paw], edgecolor=paw_colors[paw])
ax.hist(all_stride_offsets, bins=nbins, alpha=0.6, label='Stride Offsets', color=paw_colors[paw], edgecolor=paw_colors[paw])
if not plot_only_off:
    ax.hist(all_onsets, bins=nbins, alpha=0.6, label='Laser Onsets', color=onset_face, edgecolor=onset_edge)
ax.hist(all_offsets, bins=nbins, alpha=0.6, label='Laser Offsets', color=offset_face, edgecolor=offset_edge)
ax.axvline(x=0, color=paw_colors[paw], linestyle='--', linewidth=1, label=center + ' onset')
if not plot_only_off:
    ax.axvline(x=np.nanmedian(all_onsets), color=onset_edge, linestyle='-', linewidth=2, label='Med Onset')
ax.axvline(x=np.nanmedian(all_offsets), color=offset_edge, linestyle='-', linewidth=2, label='Med Offset')
ax.axvline(x=np.nanmedian(all_stride_onsets), color=paw_colors[paw], linestyle='-', linewidth=2)
ax.axvline(x=np.nanmedian(all_stride_offsets), color='darkred', linestyle='-', linewidth=2)
# Add the ephys delays
if 'JAWS' in path:
    ax.axvline(x=np.nanmedian(all_offsets) + med_cspk_delay_ms - med_airpuff_delay_ms, color='black', linestyle=':', linewidth=2, label='Med Airpuff')
    ax.axvline(x=np.nanmedian(all_offsets) + med_cspk_delay_ms, color='darkgray', linestyle='-', linewidth=2, label='Med Cspk')
else:
    ax.axvline(x=np.nanmedian(all_onsets) + med_cspk_delay_ms - med_airpuff_delay_ms, color='black', linestyle=':', linewidth=2, label='Med Airpuff')
    ax.axvline(x=np.nanmedian(all_onsets) + med_cspk_delay_ms, color='darkgray', linestyle='-', linewidth=2, label='Med Cspk')

ax.set_xlabel('Time (ms)', fontsize=18)
ax.set_ylabel('Frequency', fontsize=18)
ax.set_xlim(time_range[0], time_range[1])
ax.legend(fontsize=14, bbox_to_anchor=(1.1, 1), loc='upper left')
fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to ensure the title is not cut off
ax.tick_params(axis='both', which='major', labelsize=16)
all_hist_file = os.path.join(path_save, f"ALLanimals_laser_onset_offset_histogram_all_trials_{hist_all}.png")
plt.savefig(utils.win_safe_path(all_hist_file))
plt.close(fig)

if convolution_histogram_data is not None:
    use_laser_offset = 'JAWS' in path
    driver_times = all_offsets if use_laser_offset else all_onsets
    driver_times_finite = utils.finite_array(driver_times)
    relevant_laser_median_ms = np.nanmedian(driver_times_finite)
    relevant_laser_color = offset_edge if use_laser_offset else onset_edge
    relevant_laser_label = 'Median laser offset' if use_laser_offset else 'Median laser onset'
    expected_cspk_histogram = utils.build_expected_cspk_histogram(
        driver_times,
        nbins,
        convolution_histogram_data,
        use_laser_offset=use_laser_offset,
    )
    expected_cspk_convolved_median_ms = utils.weighted_median_from_histogram(
        expected_cspk_histogram['expected_bin_centers'],
        expected_cspk_histogram['expected_counts'],
    )
    expected_cspk_line_median_ms = np.nanmedian(driver_times_finite) + med_cspk_delay_ms
    print(
        'Expected CSpk median comparison [ms] | '
        f"line={expected_cspk_line_median_ms:.3f}, "
        f"convolved={expected_cspk_convolved_median_ms:.3f}, "
        f"difference={expected_cspk_convolved_median_ms - expected_cspk_line_median_ms:.3f}"
    )

    fig_conv, ax_conv = plt.subplots(figsize=(14, 8))
    ax_conv.hist(all_stride_onsets, bins=nbins, alpha=0.3, label='Stride Onsets', color=paw_colors[paw], edgecolor=paw_colors[paw])
    ax_conv.hist(all_stride_offsets, bins=nbins, alpha=0.6, label='Stride Offsets', color=paw_colors[paw], edgecolor=paw_colors[paw])
    if not plot_only_off:
        ax_conv.hist(all_onsets, bins=nbins, alpha=0.6, label='Laser Onsets', color=onset_face, edgecolor=onset_edge)
    ax_conv.hist(all_offsets, bins=nbins, alpha=0.6, label='Laser Offsets', color=offset_face, edgecolor=offset_edge)
    ax_conv.axvline(x=0, color=paw_colors[paw], linestyle='--', linewidth=1, label=center + ' onset')
    if not plot_only_off:
        ax_conv.axvline(x=np.nanmedian(all_onsets_finite), color=onset_edge, linestyle='-', linewidth=2, label='Med Onset')
    ax_conv.axvline(x=np.nanmedian(all_offsets_finite), color=offset_edge, linestyle='-', linewidth=2, label='Med Offset')
    ax_conv.axvline(x=np.nanmedian(all_stride_onsets_finite), color=paw_colors[paw], linestyle='-', linewidth=2)
    ax_conv.axvline(x=np.nanmedian(all_stride_offsets_finite), color='darkred', linestyle='-', linewidth=2)
    ax_conv.step(
        expected_cspk_histogram['expected_bin_centers'],
        expected_cspk_histogram['expected_counts'],
        where='mid',
        color='darkgray',
        linewidth=2.5,
        label='Expected CSpk',
    )
    ax_conv.axvline(
        x=relevant_laser_median_ms,
        color=relevant_laser_color,
        linestyle='-',
        linewidth=2,
        label=relevant_laser_label,
    )
    ax_conv.axvline(
        x=expected_cspk_convolved_median_ms,
        color='darkgray',
        linestyle='-',
        linewidth=2.5,
        label='Median expected CSpk',
    )
    ax_conv.axvline(
        x=expected_cspk_convolved_median_ms - med_airpuff_delay_ms,
        color='black',
        linestyle='--',
        linewidth=2,
        label='Med Airpuff',
    )
    ax_conv.set_xlabel('Time (ms)', fontsize=18)
    ax_conv.set_ylabel('Frequency', fontsize=18)
    ax_conv.set_xlim(time_range[0], time_range[1])
    ax_conv.legend(fontsize=14, bbox_to_anchor=(1.1, 1), loc='upper left')
    ax_conv.tick_params(axis='both', which='major', labelsize=16)
    fig_conv.tight_layout(rect=[0, 0, 1, 0.95])
    all_hist_conv_file = os.path.join(
        path_save,
        f"ALLanimals_laser_onset_offset_histogram_convolved_cspk_all_trials_{hist_all}.png",
    )
    fig_conv.savefig(utils.win_safe_path(all_hist_conv_file), dpi=150)
    plt.close(fig_conv)
else:
    print('Skipping convolution-based all-animals histograms: fill cspk_histogram_folder and cspk_histogram_dataset.')

# --- Simplified histogram: laser events + median stride/CSpk lines ---
fig_simple, ax_simple = plt.subplots(figsize=(12, 8))

# Phase labels depending on centering: sw-centered stride goes st→sw→st; st-centered goes sw→st→sw
if center == 'sw':
    lbl_onset, lbl_center, lbl_offset = 'st', 'sw', 'st'
else:
    lbl_onset, lbl_center, lbl_offset = 'sw', 'st', 'sw'

if 'JAWS' in path:
    # JAWS: show laser offsets histogram
    ax_simple.hist(all_offsets, bins=nbins, alpha=0.8,
                   color='lightyellow', edgecolor=offset_edge)
    med_laser = np.nanmedian(all_offsets)
    ax_simple.axvline(x=med_laser, color='gold', linestyle='-', linewidth=3)
    laser_label = 'laser offset'
    laser_label_color = 'gold'
    med_cspk_x = med_laser + med_cspk_delay_ms
else:
    # ChR2: show laser onsets histogram
    ax_simple.hist(all_onsets, bins=nbins, alpha=0.8,
                   color='lightblue', edgecolor=onset_edge)
    med_laser = np.nanmedian(all_onsets)
    ax_simple.axvline(x=med_laser, color='steelblue', linestyle='-', linewidth=3)
    laser_label = 'laser onset'
    laser_label_color = 'steelblue'
    med_cspk_x = med_laser + med_cspk_delay_ms

# Center line (sw or st onset at 0)
ax_simple.axvline(x=0, color=paw_colors[paw], linestyle='--', linewidth=4)
# Median stride onset & offset
med_stride_onset_x = np.nanmedian(all_stride_onsets)
med_stride_offset_x = np.nanmedian(all_stride_offsets)
ax_simple.axvline(x=med_stride_onset_x, color='red', linestyle='-', linewidth=1.5)
ax_simple.axvline(x=med_stride_offset_x, color='red', linestyle='-', linewidth=1.5)
# Median expected CSpk
ax_simple.axvline(x=med_cspk_x, color='darkgray', linestyle='-', linewidth=3)
# Median airpuff
ax_simple.axvline(x=med_cspk_x - med_airpuff_delay_ms, color='black', linestyle='--', linewidth=3)

ax_simple.set_xlabel('Time (ms)', fontsize=36)
ax_simple.set_ylabel('Counts', fontsize=36)
ax_simple.set_xlim(-150, 150)
ax_simple.tick_params(axis='both', which='major', labelsize=34)

# Place stride labels at the bottom, laser/cspk labels at the top
_top = ax_simple.get_ylim()[1]
_bot = ax_simple.get_ylim()[0]
_txt_top = dict(ha='center', va='bottom', fontsize=31, fontweight='bold', rotation=0)
_txt_bot = dict(ha='left', va='bottom', fontsize=36, fontweight='normal', rotation=0)
_nudge = 3  # ms to nudge stride labels away from the line
# Stride labels at bottom inside axes (red, not bold, bigger)
ax_simple.text(med_stride_onset_x + _nudge, _bot + (_top - _bot) * 0.01, lbl_onset, color='red', **_txt_bot)
ax_simple.text(0 + _nudge, _bot + (_top - _bot) * 0.01, lbl_center, color=paw_colors[paw], **_txt_bot)
ax_simple.text(med_stride_offset_x + _nudge, _bot + (_top - _bot) * 0.01, lbl_offset, color='red', **_txt_bot)
# Laser and predicted CSpk labels at top, same height, spread apart horizontally
_label_y = _top * 1.02
_spread = 55  # ms offset to push labels apart from each other
ax_simple.text(med_laser - _spread, _label_y, laser_label, color=laser_label_color, **_txt_top)
ax_simple.text(med_cspk_x + _spread, _label_y, 'predicted Cspk', color='darkgray', **_txt_top)

fig_simple.tight_layout()
simple_hist_file = os.path.join(path_save, f"ALLanimals_laser_cspk_simplified_histogram_{hist_all}.png")
fig_simple.savefig(utils.win_safe_path(simple_hist_file), dpi=150)
plt.close(fig_simple)

if convolution_histogram_data is not None:
    fig_simple_conv, ax_simple_conv = plt.subplots(figsize=(12, 8))
    if 'JAWS' in path:
        ax_simple_conv.hist(all_offsets, bins=nbins, alpha=0.8, color='lightyellow', edgecolor=offset_edge, label='laser offset')
        laser_label = 'laser offset'
        laser_label_color = 'gold'
    else:
        ax_simple_conv.hist(all_onsets, bins=nbins, alpha=0.8, color='lightblue', edgecolor=onset_edge, label='laser onset')
        laser_label = 'laser onset'
        laser_label_color = 'steelblue'

    ax_simple_conv.axvline(x=0, color=paw_colors[paw], linestyle='--', linewidth=4)
    ax_simple_conv.axvline(x=med_stride_onset_x, color='red', linestyle='-', linewidth=1.5)
    ax_simple_conv.axvline(x=med_stride_offset_x, color='red', linestyle='-', linewidth=1.5)
    ax_simple_conv.axvline(x=relevant_laser_median_ms, color=relevant_laser_color, linestyle='-', linewidth=3)
    ax_simple_conv.axvline(x=expected_cspk_convolved_median_ms, color='darkgray', linestyle='-', linewidth=3)
    ax_simple_conv.axvline(x=expected_cspk_convolved_median_ms - med_airpuff_delay_ms, color='black', linestyle='--', linewidth=3, label='Med Airpuff')
    ax_simple_conv.step(
        expected_cspk_histogram['expected_bin_centers'],
        expected_cspk_histogram['expected_counts'],
        where='mid',
        color='darkgray',
        linewidth=3,
        label='Expected CSpk',
    )

    ax_simple_conv.set_xlabel('Time (ms)', fontsize=36)
    ax_simple_conv.set_ylabel('Counts', fontsize=36)
    ax_simple_conv.set_xlim(-150, 150)
    ax_simple_conv.tick_params(axis='both', which='major', labelsize=34)

    top_simple_conv = ax_simple_conv.get_ylim()[1]
    bottom_simple_conv = ax_simple_conv.get_ylim()[0]
    ax_simple_conv.text(med_stride_onset_x + _nudge, bottom_simple_conv + (top_simple_conv - bottom_simple_conv) * 0.01, lbl_onset, color='red', **_txt_bot)
    ax_simple_conv.text(0 + _nudge, bottom_simple_conv + (top_simple_conv - bottom_simple_conv) * 0.01, lbl_center, color=paw_colors[paw], **_txt_bot)
    ax_simple_conv.text(med_stride_offset_x + _nudge, bottom_simple_conv + (top_simple_conv - bottom_simple_conv) * 0.01, lbl_offset, color='red', **_txt_bot)
    simple_conv_legend_loc = 'upper right' if 'JAWS' in path else 'upper left'
    ax_simple_conv.legend(fontsize=24, loc=simple_conv_legend_loc)

    fig_simple_conv.tight_layout()
    simple_hist_conv_file = os.path.join(
        path_save,
        f"ALLanimals_laser_cspk_simplified_convolved_histogram_{hist_all}.png",
    )
    fig_simple_conv.savefig(utils.win_safe_path(simple_hist_conv_file), dpi=150)
    plt.close(fig_simple_conv)

# Finalize and save per-trial all-animals figures
for trial_num, fig in figs_per_trial.items():
    axes = axes_per_trial[trial_num]
    # Hide empty subplots
    for idx in range(n_animals, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    # Also hide any subplot that ended up unused (no image/lines/collections)
    for ax in axes.flatten():
        if not ax.images and len(ax.collections) == 0 and len(ax.lines) == 0:
            ax.axis('off')
    # Figure-level legend (center onset line), anchored near the title
    legend_handles = [Line2D([], [], color=paw_colors[0], linestyle='--', linewidth=1, label=center + ' onset')]
    fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.98, 0.98), frameon=False)
    # Add a single colorbar for all subplots
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    sm = ScalarMappable(cmap='gray', norm=Normalize(vmin=0, vmax=40))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label=f'{to_plot[0]} Position')
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    all_animals_file = os.path.join(path_save, f"ALL_animals_trial_{trial_num}_stacked_stride_positions_{paw_names[0]}_{to_plot[0]}.png")
    fig.savefig(utils.win_safe_path(all_animals_file), bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved trial {trial_num} all-animals plot to {all_animals_file}")