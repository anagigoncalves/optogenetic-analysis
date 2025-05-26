"""
Created on Tue Oct 15 14:42:12 2024

@author: Alice Geminiani
"""

# Kinematic analysis of individual limbs
import online_tracking_class
import locomotion_class
import os
import numpy as np
import matplotlib.pyplot as plt
import kinematic_functions
import gc

#path='D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\20240722 power checks CTX\\stance stim\\1mW\\'

path='D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\tied stance stim\\'

paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']
paw_names = ['FR']      #, 'HR', 'FL', 'HL']
num_resamples = 360
to_plot = ['x']
center = 'st'
force_center = False

otrack_class = online_tracking_class.otrack_class(path)
loco = locomotion_class.loco_class(path)
path_save = path + '\\kinematics\\'+center+'_centered\\'
if not os.path.exists(path_save):
    os.mkdir(path_save)
print("Analysing..........................", path)

# GET THE NUMBER OF ANIMALS AND THE SESSION ID
animal_session_list = loco.animals_within_session()
animal_list = []
for a in range(len(animal_session_list)):
    animal_list.append(animal_session_list[a][0])

session_list = []
for a in range(len(animal_session_list)):
    session_list.append(animal_session_list[a][1])


# FOR EACH SESSION AND ANIMAL EXTRACT PAW POSITIONS in 3D
for count_animal, animal in enumerate(animal_list):
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

    for paw in range(len(paw_names)):       # For each paw
        for axis in to_plot:        # For each axis to plot
            for f in filelist:          # For each trial
                count_trial = int(f.split('DLC')[0].split('_')[-1])-1      # Get trial number from file name, to spot any missing trial; parameters for remaining ones will stay to NaN
                [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, 0)
                [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
                paws_rel = {'x': loco.get_paws_rel(final_tracks, 'X'), 'y': loco.get_paws_rel(final_tracks, 'Y'), 'z': loco.get_paws_rel(final_tracks, 'Z')}

                # Divide in strides for each paw and plot them
                traj_resampled = kinematic_functions.resample_strides_position(paws_rel[axis], st_strides_mat, sw_pts_mat, paw, num_resamples, center=center, force_center=force_center)
                kinematic_functions.plot_resampled_position(traj_resampled, axis, paw_colors[paw], paw_names[paw], animal+'_trial'+str(count_trial), path_save, center=center, force_center=force_center)
                # Save the average resampled trajectory for each trial
                traj_resampled_all_trials[paw_names[paw]][axis].append(traj_resampled)
            
            plt.close('all')
            gc.collect()
    

            # Plot the average of all trials for each animal, paw and axis
            # kinematic_functions.plot_resampled_position_all_trials(traj_resampled_all_trials[paw_names[paw]][axis], axis, paw_names[paw], animal, path_save, center=center, force_center=force_center)


            kinematic_functions.plot_resampled_position_avg_all(avg_traj_resampled_all_trials_all_animals[paw_names[paw]][axis], axis, paw_names[paw], list(range(28)), path_save, center=center, force_center=force_center)
            kinematic_functions.plot_resampled_position_avg_all(avg_traj_resampled_all_trials_all_animals[paw_names[paw]][axis], axis, paw_names[paw], list(range(9)), path_save, center=center, force_center=force_center)
            kinematic_functions.plot_resampled_position_avg_all(avg_traj_resampled_all_trials_all_animals[paw_names[paw]][axis], axis, paw_names[paw], list(range(9,19)), path_save, center=center, force_center=force_center)
            kinematic_functions.plot_resampled_position_avg_all(avg_traj_resampled_all_trials_all_animals[paw_names[paw]][axis], axis, paw_names[paw], list(range(19,28)), path_save, center=center, force_center=force_center)
