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
import matplotlib.pyplot as plt
import kinematic_functions
import gc
from itertools import chain
import pickle



path='D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\tied stance stim\\'
#path = 'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\tied swing stim\\'
path='D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT\\LOW expression\\ALL_ANIMALS\\tied th200st IO 50ms\\'
#path = 'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Tied belt sessions\\20240222 tied stance stim'

paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']
paw_names = ['FR']      #, 'HR', 'FL', 'HL']

to_plot = ['x']
center = 'st'
force_center = 0        # 0 no forcing, 0.5 forcing in the middle, 0.33 or 0.66 forcing at 33 or 66  %     
hist_all = False            # If True, plot histograms of stride laser onsets and offsets for all stim trials, otherwise of only the first 3 stimulated trials
plot_off_to_on = False
stim_start = 9
stim_duration = 10
num_trials = 28
num_resamples = 360
sf = 330        # [Hz] sampling frequency of the camera
time_range = [-200, 200]        # [ms] time range we are going to look at (single stride plots, histograms, etc.)

# Laser color scheme based on experiment type
if 'JAWS' in path:
    laser_color_stride_plot = 'yellow'
    onset_face, onset_edge = 'lightyellow', 'yellow'
    offset_face, offset_edge = 'orange', 'darkorange'
else:
    laser_color_stride_plot = 'blue'
    onset_face, onset_edge = 'lightskyblue', 'royalblue'
    offset_face, offset_edge = 'blue', 'darkblue'

otrack_class = online_tracking_class.otrack_class(path)
loco = locomotion_class.loco_class(path)
folder_name = 'kinematics_laser_timing'
# Build paths with os.path.join and ensure directories exist
def _win_safe_path(p: str) -> str:
    """Prefix Windows paths with \\?\ to avoid MAX_PATH issues when length is large."""
    try:
        # Only apply on Windows
        if os.name == 'nt':
            p_norm = os.path.normpath(p)
            if len(p_norm) > 240 and not p_norm.startswith('\\\\?\\'):
                return '\\\\?\\' + p_norm
            return p_norm
        return p
    except Exception:
        return p

base_output_dir = os.path.join(path, folder_name)
path_save = os.path.join(
    base_output_dir,
    f"{center}_centered_force_center{force_center}_OFFtoON{plot_off_to_on}"
)
os.makedirs(_win_safe_path(path_save), exist_ok=True)
print("Analysing..........................", path)



# GET THE NUMBER OF ANIMALS AND THE SESSION ID
animal_session_list = loco.animals_within_session()
animal_list = []
for a in range(len(animal_session_list)):
    animal_list.append(animal_session_list[a][0])

session_list = []
for a in range(len(animal_session_list)):
    session_list.append(animal_session_list[a][1])

included_animal_list = ['VIV42908']    # ['VIV42375']
#[ 'VIV42908']  
#['VIV42375' ]
#'MC16848', 'MC16851', 'MC17319', 'MC17665', 'MC17666', 'MC17669', 'MC17670', 'MC19082', 'MC19124', 
                       #'MC19214', 'VIV41329', 'VIV41330',
                     #    'VIV42375', 'VIV42428', 'VIV42429', 'VIV42430', 'VIV42376'] 

avg_traj_resampled_all_trials_all_animals = {}

for paw in paw_names:
    avg_traj_resampled_all_trials_all_animals[paw] = {}
    for axis in to_plot:        avg_traj_resampled_all_trials_all_animals[paw][axis] = np.full((len(animal_list), num_trials, num_resamples), np.nan) 
        
# FOR EACH SESSION AND ANIMAL EXTRACT PAW POSITIONS in 3D
stride_onsets_all_trials_all_animals = []        # List to store stride onsets for each trial for all animals
stride_offsets_all_trials_all_animals = []        # List to store stride offsets for each trial for all animals
stride_laser_onsets_all_trials_all_animals = []        # List to store stride laser onsets for each trial for all animals
stride_laser_offsets_all_trials_all_animals = []        # List to store stride laser offsets for each trial for all animals
for count_animal, animal in enumerate(included_animal_list):
    print('Processing ' + animal)
    session = int(session_list[count_animal])
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
                                    if sw_pts_mat[paw][s, 0, 0] <= current_offset_times[l+1] < current_onset_times[l] <= sw_pts_mat[paw][s + 1, 0, 0]:
                                        current_stride_laser_onset = current_offset_times[l+1] - stance_onset
                                        current_stride_laser_offset = current_onset_times[l] - stance_onset
                                        break   # Found the first valid onset and offset, no need to check further
                                    # Just offset within stride bounds
                                    if sw_pts_mat[paw][s, 0, 0] <= offset_time <= sw_pts_mat[paw][s + 1, 0, 0] and onset_time > sw_pts_mat[paw][s + 1, 0, 0]:
                                        current_stride_laser_offset = offset_time - stance_onset
                                        current_stride_laser_onset = sw_pts_mat[paw][s + 1, 0, 0] - stance_onset
                                        break
                                    # Just onset within stride bounds
                                    if sw_pts_mat[paw][s, 0, 0] <= onset_time <= sw_pts_mat[paw][s + 1, 0, 0] and onset_time < sw_pts_mat[paw][s, 0, 0]:
                                        current_stride_laser_onset = sw_pts_mat[paw][s, 0, 0] - stance_onset
                                        current_stride_laser_offset = offset_time - stance_onset
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
                                    if st_strides_mat[paw][s,0,0] <= current_offset_times[l+1] < current_onset_times[l] <= st_strides_mat[paw][s,1,0]:
                                        current_stride_laser_onset = current_offset_times[l+1] - st_strides_mat[paw][s,1,0]
                                        current_stride_laser_offset = current_onset_times[l] - st_strides_mat[paw][s,1,0]
                                        break   # Found the first valid onset and offset, no need to check further
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

                for s in range(len(stride_times)):
                    to_add_before = (np.min(stride_times[s]) - min_time)/(1000/sf)
                    to_add_after = (max_time - np.max(stride_times[s]))/(1000/sf)
                    stride_times[s] = np.pad(stride_times[s], (int(np.ceil(to_add_before)), int(np.ceil(to_add_after))), constant_values=np.nan)
                    idx_zero = np.nanargmin(np.abs(stride_times[s]))
                    start_idx = idx_zero + int(time_range[0]/(1000/sf))
                    end_idx = idx_zero + int(time_range[1]/(1000/sf))
                    stride_positions[s] = np.pad(stride_positions[s], (int(np.ceil(to_add_before)), int(np.ceil(to_add_after))), constant_values=np.nan)
                    # Get slice with padding and trimming
                    positions_padded_trimmed = np.pad(stride_positions[s], 
                                        (abs(min(0, start_idx)), max(0, end_idx - len(stride_positions[s]))),
                                        constant_values=np.nan)[max(0, start_idx):end_idx+abs(min(0, start_idx))]
                    times_padded_trimmed = np.pad(stride_times[s], 
                                        (abs(min(0, start_idx)), max(0, end_idx - len(stride_positions[s]))),
                                        constant_values=np.nan)[max(0, start_idx):end_idx+abs(min(0, start_idx))]
                    trimmed_stride_times.append(times_padded_trimmed)
                    trimmed_stride_positions.append(positions_padded_trimmed)
                # Determine the maximum length of all strides
                max_length = max(len(pos) for pos in stride_positions)

                # Pad each stride with NaN to make them the same length
                padded_positions = np.array([
                    np.pad(pos, (0, max_length - len(pos)), constant_values=np.nan) for pos in stride_positions
                ])


                # Plot the stacked positions as a grayscale image
                plt.figure()
                manager = plt.get_current_fig_manager()
                manager.full_screen_toggle()
                plt.imshow(padded_positions, aspect='auto', cmap='gray', 
                           extent=[min_time, max_time, 0, len(padded_positions)], origin='lower', vmin=0, vmax=40)
                # Add shaded areas between onset and offset times of laser
                for t in range(len(stride_times)):
                    if stride_laser_onsets[t] is not None and stride_laser_offsets[t] is not None:
                        plt.fill_betweenx(
                            y=[t,t+1],                            
                            x1=stride_laser_onsets[t],
                            x2=stride_laser_offsets[t],
                            color=laser_color_stride_plot,
                            alpha=0.3
                        )
                plt.colorbar(label=axis+' Position')
                
                plt.xlabel('Time (ms)')
                plt.ylabel('Stride Index')
                plt.xlim(time_range[0], time_range[1])
                plt.ylim(0, len(padded_positions))
                plt.title('Trial ' + str(count_trial + 1) + ' - ' + paw_names[paw] + ' - ' + axis)
                plt.axvline(x=0, color=paw_colors[paw], linestyle='--', linewidth=1, label=center+' onset')
                plt.legend(fontsize=8) #text(0, len(padded_positions) + 1, center+' onset', color=paw_colors[paw], fontsize=12, ha='center')
               # mng = plt.get_current_fig_manager()
               # mng.window.state('zoomed')  # for Windows
                save_file = os.path.join(path_save, f"{animal}_trial_{count_trial+1}_stacked_stride_positions.png")
                plt.savefig(_win_safe_path(save_file), bbox_inches='tight', dpi=300)
               # plt.show()

                stride_onsets_all_trials.append(stride_onsets)
                stride_offsets_all_trials.append(stride_offsets)
                if count_trial + 1 >= stim_start and count_trial + 1 < stim_start + stim_duration:
                    stride_laser_onsets_all_trials.append(stride_laser_onsets)
                    stride_laser_offsets_all_trials.append(stride_laser_offsets)
                    positions_all_stim_trials.append(padded_positions)
                    trimmed_positions_all_stim_trials.append(trimmed_stride_positions)
                    trimmed_times_all_stim_trials.append(trimmed_stride_times)
            plt.close('all')
            gc.collect()
    

            # Histograms of stride laser onsets and offsets
            from itertools import chain
            if hist_all:
                selected_onsets = list(chain.from_iterable(stride_laser_onsets_all_trials))   
                selected_offsets =  list(chain.from_iterable(stride_laser_offsets_all_trials)) 
                selected_positions = list(chain.from_iterable(positions_all_stim_trials)) 
            else:
                selected_onsets = list(chain.from_iterable(stride_laser_onsets_all_trials[:3]))  # Select the first 3 trials
                selected_offsets = list(chain.from_iterable(stride_laser_offsets_all_trials[:3]))  # Select the first 3 trials
                selected_positions = list(chain.from_iterable(positions_all_stim_trials[:3]))  # Select the first 3 trials

            fig, ax1 = plt.subplots(figsize=(14, 8))
            bin_width = 5
            min_edge = min(np.nanmin(selected_onsets), np.nanmin(selected_offsets))
            max_edge = max(np.nanmax(selected_onsets), np.nanmax(selected_offsets))
            nbins = np.arange(min_edge, max_edge + bin_width, bin_width)

            ax1.hist(selected_stride_onsets, bins=nbins, alpha=0.3, label='Stride Onsets', color=paw_colors[paw], edgecolor=paw_colors[paw])
            ax1.axvline(x=0, color=paw_colors[paw], linestyle='--', linewidth=1, label=center + ' onset')
            ax1.axvline(x=np.nanmedian(selected_onsets), color=color_laser, linestyle='-', linewidth=2, label='Med Onset')
            ax1.axvline(x=np.nanmedian(selected_offsets), color='dark'+color_laser, linestyle='-', linewidth=2, label='Med Offset')
            ax1.axvline(x=np.nanmedian(stride_onsets), color=paw_colors[paw], linestyle='-', linewidth=2)
            ax1.axvline(x=np.nanmedian(stride_offsets), color='darkred', linestyle='-', linewidth=2)
            ax1.set_xlabel('Time (ms)', fontsize=14)
            ax1.set_ylabel('Frequency', fontsize=14)
            ax1.set_xlim(time_range[0], time_range[1])
            ax1.legend(fontsize=12, bbox_to_anchor=(1.1, 1), loc='upper left')
            

            # Add avg trajectory
            if hist_all:
                avg_trajectory = np.nanmean(np.array(list(chain.from_iterable(trimmed_positions_all_stim_trials))),axis=0) 
                std_trajectory = np.nanstd(np.array(list(chain.from_iterable(trimmed_positions_all_stim_trials))),axis=0)
                avg_time = np.nanmean(np.array(list(chain.from_iterable(trimmed_times_all_stim_trials))),axis=0)
            else:
            with open(_win_safe_path(animal+"_avg_trajectory_data.pkl"), "wb") as f:
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
            plt.savefig(_win_safe_path(hist_file))
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
            fig_all_trajectories.savefig(_win_safe_path(hv_file), bbox_inches='tight', dpi=300)
            # Plot the average of all trials for each animal, paw and axis
#            kinematic_functions.plot_resampled_position_all_trials(traj_resampled_all_trials[paw_names[paw]][axis], axis, paw_names[paw], animal, path_save, center=center, force_center=force_center)


     #       kinematic_functions.plot_resampled_position_avg_all(avg_traj_resampled_all_trials_all_animals[paw_names[paw]][axis], axis, paw_names[paw], list(range(28)), path_save, center=center, force_center=force_center)
     #       kinematic_functions.plot_resampled_position_avg_all(avg_traj_resampled_all_trials_all_animals[paw_names[paw]][axis], axis, paw_names[paw], list(range(9)), path_save, center=center, force_center=force_center)
     #       kinematic_functions.plot_resampled_position_avg_all(avg_traj_resampled_all_trials_all_animals[paw_names[paw]][axis], axis, paw_names[paw], list(range(9,19)), path_save, center=center, force_center=force_center)
       #     kinematic_functions.plot_resampled_position_avg_all(avg_traj_resampled_all_trials_all_animals[paw_names[paw]][axis], axis, paw_names[paw], list(range(19,28)), path_save, center=center, force_center=force_center)

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

fig, ax = plt.subplots(figsize=(14, 8))
bin_width = 5
min_edge = min(np.nanmin(all_onsets), np.nanmin(all_offsets), np.nanmin(all_stride_onsets), np.nanmin(all_stride_offsets))
max_edge = max(np.nanmax(all_onsets), np.nanmax(all_offsets), np.nanmax(all_stride_onsets), np.nanmax(all_stride_offsets))
nbins = np.arange(min_edge, max_edge + bin_width, bin_width)

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
ax.set_xlabel('Time (ms)', fontsize=18)
ax.set_ylabel('Frequency', fontsize=18)
ax.set_xlim(time_range[0], time_range[1])
ax.legend(fontsize=14, bbox_to_anchor=(1.1, 1), loc='upper left')
fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to ensure the title is not cut off
ax2.tick_params(axis='both', which='major', labelsize=16)
all_hist_file = os.path.join(path_save, f"ALLanimals_laser_onset_offset_histogram_all_trials_{hist_all}.png")
plt.savefig(_win_safe_path(all_hist_file))