import os
import matplotlib.pyplot as plt
import numpy as np

path_st = 'J:\\Opto JAWS Data\\split right fast stance stim\\'
path_sw = 'J:\\Opto JAWS Data\\split right fast swing stim\\'

animal = 'MC16851'
Ntrials = 28 
rec_size = 10 
stim_trials = np.arange(9, 19) 
experiment_type = 'split'
save_path = 'J:\\Opto JAWS Data\\animal viz split\\'

if not os.path.exists(os.path.join(save_path, animal)):
    os.mkdir(os.path.join(save_path, animal))

#import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\optogenetic-analysis\\')
import online_tracking_class
import locomotion_class
param_sym_name = ['coo', 'step_length', 'double_support', 'duty_factor', 'swing_length', 'stance_speed']
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']
def get_param_sym(path_name, animal, stim_trials):
    print('Getting symmetry info for ' + path_name)
    loco = locomotion_class.loco_class(path_name)
    animal_session_list = loco.animals_within_session()
    animal_list = []
    for a in range(len(animal_session_list)):
        animal_list.append(animal_session_list[a][0])
    animal_list_plot_idx = np.array([count_a for count_a, a in enumerate(animal_list) if a in animal])
    session_list = []
    for a in range(len(animal_session_list)):
        session_list.append(animal_session_list[a][1])
    session_list_plot = np.array(session_list)[animal_list_plot_idx]
    #summary gait parameters
    param_sym = np.zeros((len(param_sym_name), Ntrials))
    param_sym[:] = np.nan
    stance_speed = np.zeros((4, Ntrials))
    stance_speed[:] = np.nan
    session = int(session_list_plot[0])
    filelist = loco.get_track_files(animal, session)
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, 0)
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        paws_rel = loco.get_paws_rel(final_tracks, 'X')
        for count_p, param in enumerate(param_sym_name):
            param_mat = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param)
            if param == 'stance_speed':
                for p in range(4):
                    stance_speed[p, count_trial] = np.nanmean(param_mat[p])
            else:
                param_sym[count_p, count_trial] = np.nanmean(param_mat[0]) - np.nanmean(param_mat[2])
    param_sym_bs = np.zeros(np.shape(param_sym))
    param_sym_bs[:] = np.nan
    for p in range(len(param_sym)):
        bs_mean = np.nanmean(param_sym[p, :stim_trials[0]-1])
        param_sym_bs[p, :] = param_sym[p, :] - bs_mean
    return param_sym_bs, stance_speed

[param_sym_bs_st, stance_speed_st] = get_param_sym(path_st, animal, stim_trials)
[param_sym_bs_sw, stance_speed_sw] = get_param_sym(path_sw, animal, stim_trials)
for p in range(len(param_sym_name)-1):
    data_st = np.squeeze(param_sym_bs_st[p, :])
    data_sw = np.squeeze(param_sym_bs_sw[p, :])
    fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
    rectangle = plt.Rectangle((stim_trials[0]-0.5, np.nanmin([data_st, data_sw])), rec_size, np.nanmax([data_st, data_sw])-np.nanmin([data_st, data_sw]), fc='lightblue', zorder=0, alpha=0.3)
    # rectangle = plt.Rectangle((stim_trials[0]-0.5, np.nanmin(data_sw)), rec_size, np.nanmax(data_sw)-np.nanmin(data_sw), fc='lightblue', zorder=0, alpha=0.3)
    plt.gca().add_patch(rectangle)
    plt.hlines(0, 1, Ntrials, colors='grey', linestyles='--')
    plt.plot(np.arange(1, Ntrials+1), data_st, linewidth=2, marker='o', color='orange')
    plt.plot(np.arange(1, Ntrials+1), data_sw, linewidth=2, marker='o', color='green')
    ax.legend(['stance \nstim.', 'swing \nstim.', '', ''], frameon=False, fontsize=10)
    ax.set_xlabel('Trial', fontsize=14)
    ax.set_ylabel(param_sym_name[p], fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(save_path, animal, experiment_type + '_' + param_sym_name[p] + '_' + animal + '.png'))

fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
for p in range(4):
    ax.axvline(x=stim_trials[0]-0.5, color='dimgray', linestyle='--')
    ax.axvline(x=stim_trials[0]+10-0.5, color='dimgray', linestyle='--')
    ax.plot(np.arange(1, Ntrials+1), stance_speed_st[p, :], color=paw_colors[p], linewidth=2)
    ax.plot(np.arange(1, Ntrials+1), stance_speed_sw[p, :], color=paw_colors[p], linestyle='dashed', linewidth=2)
    ax.axvline(x=8.5, color='black')
    ax.axvline(x=18.5, color='black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(os.path.join(save_path, animal, experiment_type + '_stance_speed_' + animal + '.png'))

def get_laser_timing(path_name, laser_event, animal):
    print('Getting timing info for ' + path_name)
    otrack_class = online_tracking_class.otrack_class(path_name)
    loco = locomotion_class.loco_class(path_name)
    trials = otrack_class.get_trials(animal)
    # LOAD PROCESSED DATA
    [otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session,
     laser_on] = otrack_class.load_processed_files(animal)
    animal_session_list = loco.animals_within_session()
    animal_list = []
    for a in range(len(animal_session_list)):
        animal_list.append(animal_session_list[a][0])
    animal_list_plot_idx = np.array([count_a for count_a, a in enumerate(animal_list) if a in animal])
    session_list = []
    for a in range(len(animal_session_list)):
        session_list.append(animal_session_list[a][1])
    session_list_plot = np.array(session_list)[animal_list_plot_idx]
    # READ OFFLINE PAW EXCURSIONS
    [final_tracks_trials, st_strides_trials, sw_strides_trials] = otrack_class.get_offtrack_paws(loco, animal, np.int64(
        session_list_plot[0]))
    final_tracks_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials,
                                                 'st-sw-st')
    # LASER ONSET AND OFFSET PHASE
    onset = []
    offset = []
    for count_t, trial in enumerate(stim_trials):
        [light_onset_phase, light_offset_phase, stim_nr, stride_nr] = \
            otrack_class.laser_presentation_phase_all(trial, trials, laser_event, offtracks_st, offtracks_sw, laser_on,
                                                      timestamps_session, final_tracks_phase, "FR")
        onset.extend(light_onset_phase)
        offset.extend(light_offset_phase)
    hist_onset = np.histogram(onset, range=(np.min(onset), np.max(onset)))
    hist_offset = np.histogram(offset, range=(np.min(offset), np.max(offset)))
    weights_onset = np.ones_like(onset) / np.max(hist_onset[0])
    weights_offset = np.ones_like(offset) / np.max(hist_offset[0])
    return onset, offset, weights_onset, weights_offset
[onset_st, offset_st, weights_onset_st, weights_offset_st] = get_laser_timing(path_st, 'stance', animal)
[onset_sw, offset_sw, weights_onset_sw, weights_offset_sw] = get_laser_timing(path_sw, 'swing', animal)
amp_plot = 0.5
time = np.arange(-1, 2, np.round(1 / 330, 3))
FR = amp_plot * np.sin(2 * np.pi * time + (np.pi / 2)) + amp_plot
fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
ax.plot(time, FR, color='lightgray', zorder=0)
ax.hist(onset_st, histtype='step', color='gold', linewidth=4, weights=weights_onset_st)
ax.hist(offset_st, histtype='step', color='darkorange', linewidth=4, weights=weights_offset_st)
ax.hist(onset_sw, histtype='step', color='lightgreen', linewidth=4, weights=weights_onset_sw)
ax.hist(offset_sw, histtype='step', color='darkgreen', linewidth=4, weights=weights_offset_sw)
ax.set_xticks([-1, -0.5, 0, 0.5, 1, 1.5, 2])
ax.set_xticklabels(['-100', '-50', '0', '50', '100', '150', '200'])
ax.set_xlabel('Phase (%)', fontsize=12)
ax.set_ylabel('Laser-on counts', fontsize=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)
plt.savefig(os.path.join(save_path, animal, experiment_type + '_timing_' + animal + '.png'))
