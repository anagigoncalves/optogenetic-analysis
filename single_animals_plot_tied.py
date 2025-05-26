import os
import matplotlib.pyplot as plt
import numpy as np

path_st = 'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\ALL_ANIMALS\\tied stance stim\\'
path_sw = 'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\ALL_ANIMALS\\tied swing stim\\'

animal = 'VIV42430'
Ntrials = 28 #28
stim_duration = 10 #10
stim_trials = np.arange(9, 19) #np.arange(9, 19)
experiment_type = 'tied'
save_path = 'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\ALL_ANIMALS\\single_animal_analysis\\'
bars_ranges = {'coo': [-3, 3], 'step_length': [-5, 5], 'double_support': [-8, 8], 'coo_stance': [-5, 5], 'swing_length': [-5, 5], 'stance_speed': [-0.4,-0.4], 'duty_factor': [-3, 3],}

if not os.path.exists(os.path.join(save_path, animal)):
    os.mkdir(os.path.join(save_path, animal))

#import classes
#os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\optogenetic-analysis\\')
import online_tracking_class
import locomotion_class
param_sym_name = ['coo', 'step_length', 'double_support', 'duty_factor', 'swing_length']

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
    session = int(session_list_plot[0])
    filelist = loco.get_track_files(animal, session)
    for f in filelist:
        count_trial = int(f.split('DLC')[0].split('_')[-1])-1      # Get trial number from file name, to spot any missing trial; parameters for remaining ones will stay to NaN
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, 0)
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        paws_rel = loco.get_paws_rel(final_tracks, 'X')
        for count_p, param in enumerate(param_sym_name):
            param_mat = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param)
            param_sym[count_p, count_trial] = np.nanmean(param_mat[0]) - np.nanmean(param_mat[2])
    param_sym_bs = np.zeros(np.shape(param_sym))
    param_sym_bs[:] = np.nan
    for p in range(len(param_sym)):
        bs_mean = np.nanmean(param_sym[p, :stim_trials[0]-1])
        param_sym_bs[p, :] = param_sym[p, :] - bs_mean
    return param_sym_bs

param_sym_bs_st = get_param_sym(path_st, animal, stim_trials)
param_sym_bs_sw = get_param_sym(path_sw, animal, stim_trials)
for p in range(len(param_sym_name)):
    data_st = np.squeeze(param_sym_bs_st[p, :])
    data_sw = np.squeeze(param_sym_bs_sw[p, :])
    fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
    plt.plot(np.arange(1, Ntrials+1), data_st, linewidth=2, marker='o', color='orange')
    plt.plot(np.arange(1, Ntrials+1), data_sw, linewidth=2, marker='o', color='green')
    ax.legend(['stance \nstim.', 'swing \nstim.', '', ''], frameon=False, fontsize=10)
    rectangle = plt.Rectangle((stim_trials[0]-0.5, np.nanmin([data_st, data_sw])), stim_duration, np.nanmax([data_st, data_sw])-np.nanmin([data_st, data_sw]), fc='lightblue', zorder=0, alpha=0.3)
    plt.gca().add_patch(rectangle)
    plt.hlines(0, 1, Ntrials, colors='grey', linestyles='--')
    ax.set_xlabel('Trial', fontsize=20)
    ax.set_ylabel(param_sym_name[p], fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(save_path, animal, experiment_type + '_' + param_sym_name[p] + '_' + animal + '.png'))

    # After-effect scatter/bar plot
    ae_st = np.nanmean(data_st[stim_trials[0]+stim_duration-1:stim_trials[0]+stim_duration+1])
    ae_sw = np.nanmean(data_sw[stim_trials[0]+stim_duration-1:stim_trials[0]+stim_duration+1])
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    plt.plot([1], ae_st, '-o', c='orange', linewidth=2, markersize=10)
    plt.plot([3], ae_sw, '-o', c='green',  linewidth=2, markersize=10)
    plt.plot([1,3], [ae_st, ae_sw], '--', c='black',  linewidth=1)
    plt.hlines(0, -0.5, 3.5, colors='grey', linestyles='-')
    plt.xticks(ticks=[1,3], labels=['Stance', 'Swing'])
    ax.set_ylabel(param_sym_name[p]+ ' after effect', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim([0.5, 3.5])
    plt.ylim(bars_ranges[param_sym_name[p]])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(save_path, animal, experiment_type + '_' + param_sym_name[p] + '_aftereffect_' + animal + '.png'))
'''
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
plt.savefig(os.path.join(save_path, animal, experiment_type + '_timing_' + animal + '.png'))'''