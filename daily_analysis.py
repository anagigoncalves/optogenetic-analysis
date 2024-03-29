import numpy as np
import matplotlib.pyplot as plt
import os

# Inputs
laser_event = 'swing'
single_animal_analysis = 1
path = 'C:\\Users\\Ana\\Documents\\PhD\\Projects\\Online Stimulation Treadmill\\Experiments\\24042023 split right fast swing large stim\\MC16851\\'
animal = 'MC16851'
th_st = 100
th_sw = 100
session = 7
Ntrials = 28
stim_start = 4
stim_duration = 14
plot_rig_signals = 0
print_plots = 1
bs_bool = 0
control_bool = 0
control_ses = 'left'
control_path = 'C:\\Users\\Ana\\Documents\\PhD\\Projects\\Online Stimulation Treadmill\\Experiments\\'
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)
path_save = path + 'grouped output\\'
if not os.path.exists(path + 'grouped output'):
    os.mkdir(path + 'grouped output')
paw_colors = ['red', 'magenta', 'blue', 'cyan']
paw_otrack = 'FR'

# GET THE NUMBER OF ANIMALS AND THE SESSION ID
animal_session_list = loco.animals_within_session()
animal_list = []
for a in range(len(animal_session_list)):
    animal_list.append(animal_session_list[a][0])
session_list = []
for a in range(len(animal_session_list)):
    session_list.append(animal_session_list[a][1])

if single_animal_analysis:
    trials = otrack_class.get_trials()
    # READ CAMERA TIMESTAMPS AND FRAME COUNTER
    [camera_timestamps_session, camera_frames_kept, camera_frame_counter_session] = otrack_class.get_session_metadata(plot_rig_signals)

    # READ SYNCHRONIZER SIGNALS
    [timestamps_session, frame_counter_session, trial_signal_session, sync_signal_session, laser_signal_session, laser_trial_signal_session] = otrack_class.get_synchronizer_data(camera_frames_kept, plot_rig_signals)

    # READ ONLINE DLC TRACKS
    otracks = otrack_class.get_otrack_excursion_data(timestamps_session)
    [otracks_st, otracks_sw] = otrack_class.get_otrack_event_data(timestamps_session)

    # READ OFFLINE DLC TRACKS
    [offtracks_st, offtracks_sw] = otrack_class.get_offtrack_event_data(paw_otrack, loco, animal, session, timestamps_session)

    # READ OFFLINE PAW EXCURSIONS
    final_tracks_trials = otrack_class.get_offtrack_paws(loco, animal, session)

    # PROCESS SYNCHRONIZER LASER SIGNALS
    laser_on = otrack_class.get_laser_on(laser_signal_session, timestamps_session)

    # LASER ACCURACY
    tp_laser = np.zeros(len(trials))
    fp_laser = np.zeros(len(trials))
    tn_laser = np.zeros(len(trials))
    fn_laser = np.zeros(len(trials))
    precision_laser = np.zeros(len(trials))
    recall_laser = np.zeros(len(trials))
    f1_laser = np.zeros(len(trials))
    for count_t, trial in enumerate(trials):
        [tp_trial, fp_trial, tn_trial, fn_trial, precision_trial, recall_trial, f1_trial] = otrack_class.accuracy_laser_sync(trial, laser_event, offtracks_st, offtracks_sw, laser_on, final_tracks_trials, timestamps_session, 0)
        tp_laser[count_t] = tp_trial
        fp_laser[count_t] = fp_trial
        tn_laser[count_t] = tn_trial
        fn_laser[count_t] = fn_trial
        precision_laser[count_t] = precision_trial
        recall_laser[count_t] = recall_trial
        f1_laser[count_t] = f1_trial
    fig, ax = plt.subplots(tight_layout=True, figsize=(10, 7))
    for count_t, trial in enumerate(trials):
        ax.bar(trial, tp_laser[count_t], color='green')
        ax.bar(trial, tn_laser[count_t], bottom=tp_laser[count_t], color='darkgreen')
        ax.bar(trial, fp_laser[count_t], bottom=(tp_laser[count_t] + tn_laser[count_t]), color='red')
        ax.bar(trial, fn_laser[count_t], bottom=(tp_laser[count_t] + tn_laser[count_t] + fp_laser[count_t]), color='crimson')
    ax.legend(['true positive', 'true negative', 'false positive', 'false negative'], frameon=False, fontsize=12)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(path_save + animal + '_laser_performance.png')
    fig, ax = plt.subplots(tight_layout=True, figsize=(10, 7))
    for count_t, trial in enumerate(trials):
        ax.bar(trial, f1_laser[count_t], color='darkgrey')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(path_save + animal + '_laser_performance_f1.png')
    fig, ax = plt.subplots(tight_layout=True, figsize=(10, 7))
    for count_t, trial in enumerate(trials):
        ax.bar(trial, tp_laser[count_t]+tn_laser[count_t], color='darkgrey')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(path_save + animal + '_laser_performance_accuracy.png')

    # SETUP ACCURACY
    st_correct_setup = np.zeros(len(trials))
    sw_correct_setup = np.zeros(len(trials))
    for count_t, trial in enumerate(trials):
        [st_correct_trial, sw_correct_trial] = otrack_class.setup_accuracy(otracks, otracks_st, otracks_sw, th_st, th_sw, 0)
        st_correct_setup[count_t] = st_correct_trial
        sw_correct_setup[count_t] = st_correct_trial
    fig, ax = plt.subplots(2, 1, tight_layout=True, figsize=(10, 10))
    ax = ax.ravel()
    ax[0].bar(trials, st_correct_setup, color='orange')
    ax[0].tick_params(axis='both', which='major', labelsize=14)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[1].bar(trials, sw_correct_setup, color='green')
    ax[1].tick_params(axis='both', which='major', labelsize=14)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    plt.savefig(path_save + animal + '_setup_accuracy.png')

if single_animal_analysis == 0:
    # GAIT PARAMETERS ACROSS TRIALS
    param_sym_name = ['coo', 'step_length', 'double_support', 'coo_stance', 'swing_length', 'stance_speed']
    param_sym = np.zeros((len(param_sym_name), len(animal_list), Ntrials))
    stance_speed = np.zeros((4, len(animal_list), Ntrials))
    st_strides_trials = []
    for count_animal, animal in enumerate(animal_list):
        session = int(session_list[count_animal])
        filelist = loco.get_track_files(animal, session)
        for count_trial, f in enumerate(filelist):
            [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, 0)
            [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
            st_strides_trials.append(st_strides_mat)
            paws_rel = loco.get_paws_rel(final_tracks, 'X')
            for count_p, param in enumerate(param_sym_name):
                param_mat = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param)
                if param == 'stance_speed':
                    for p in range(4):
                        stance_speed[p, count_animal, count_trial] = np.nanmean(param_mat[p])
                elif param == 'step_length':
                    param_sym[count_p, count_animal, count_trial] = np.nanmean(param_mat[0]) - np.nanmean(param_mat[2])
                else:
                    param_sym[count_p, count_animal, count_trial] = np.nanmean(param_mat[0])-np.nanmean(param_mat[2])

    # BASELINE SUBTRACTION OF GAIT PARAMETERS
    if bs_bool:
        param_sym_bs = np.zeros(np.shape(param_sym))
        for p in range(np.shape(param_sym)[0]-1):
            for a in range(np.shape(param_sym)[1]):
                if stim_start == 9:
                    bs_mean = np.nanmean(param_sym[p, a, :stim_start-1])
                if stim_start == 5:
                    bs_mean = np.nanmean(param_sym[p, a, stim_start-1:8])
                param_sym_bs[p, a, :] = param_sym[p, a, :] - bs_mean
    else:
        param_sym_bs = param_sym

    # PLOT GAIT PARAMETERS - BASELINE SUBTRACTED
    for p in range(np.shape(param_sym)[0] - 1):
        fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
        rectangle = plt.Rectangle((stim_start - 0.5, np.min(param_sym_bs[p, :, :].flatten())), stim_duration,
                                  np.max(param_sym_bs[p, :, :].flatten()) - np.min(param_sym_bs[p, :, :].flatten()),
                                  fc='dimgrey', alpha=0.3)
        plt.gca().add_patch(rectangle)
        plt.hlines(0, 1, len(param_sym_bs[p, a, :]), colors='grey', linestyles='--')
        for a in range(np.shape(param_sym)[1]):
            plt.plot(np.linspace(1, len(param_sym_bs[p, a, :]), len(param_sym_bs[p, a, :])), param_sym_bs[p, a, :],
                     label=animal_list[a], linewidth=2)
        ax.set_xlabel('Trial', fontsize=20)
        ax.legend(frameon=False)
        ax.set_ylabel(param_sym_name[p].replace('_', ' '), fontsize=20)
        if p == 2:
            plt.gca().invert_yaxis()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if print_plots:
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            plt.savefig(path_save + param_sym_name[p] + '_sym_bs', dpi=128)
    plt.close('all')

    # PLOT GAIT PARAMETERS - NOT-BASELINE SUBTRACTED
    for p in range(np.shape(param_sym)[0] - 1):
        fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
        rectangle = plt.Rectangle((stim_start - 0.5, np.min(param_sym[p, :, :].flatten())), stim_duration,
                                  np.max(param_sym[p, :, :].flatten()) - np.min(param_sym[p, :, :].flatten()),
                                  fc='dimgrey', alpha=0.3)
        plt.gca().add_patch(rectangle)
        plt.hlines(0, 1, len(param_sym[p, a, :]), colors='grey', linestyles='--')
        for a in range(np.shape(param_sym)[1]):
            plt.plot(np.linspace(1, len(param_sym[p, a, :]), len(param_sym[p, a, :])), param_sym[p, a, :],
                     label=animal_list[a], linewidth=2)
        ax.set_xlabel('Trial', fontsize=20)
        ax.legend(frameon=False)
        ax.set_ylabel(param_sym_name[p].replace('_', ' '), fontsize=20)
        if p == 2:
            plt.gca().invert_yaxis()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if print_plots:
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            plt.savefig(path_save + param_sym_name[p] + '_sym', dpi=128)
    plt.close('all')

    # PLOT ANIMAL AVERAGE
    for p in range(np.shape(param_sym)[0] - 1):
        # param_sym_bs_ave = param_sym_bs[p, np.array([0, 2, 3, 4]), :]
        param_sym_bs_ave = param_sym_bs[p, :, :]
        fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
        rectangle = plt.Rectangle((stim_start - 0.5, np.min(param_sym_bs_ave[:, :].flatten())), stim_duration,
                                  np.max(param_sym_bs_ave[:, :].flatten()) - np.min(param_sym_bs_ave[:, :].flatten()),
                                  fc='dimgrey', alpha=0.3)
        plt.gca().add_patch(rectangle)
        plt.hlines(0, 1, len(param_sym_bs_ave[0, :]), colors='grey', linestyles='--')
        for a in range(np.shape(param_sym_bs_ave)[0]):
            plt.plot(np.linspace(1, len(param_sym_bs_ave[a, :]), len(param_sym_bs_ave[a, :])), param_sym_bs_ave[a, :], color='darkgray', linewidth=1)
        plt.plot(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])), np.nanmean(param_sym_bs_ave, axis=0), color='black', linewidth=2)
        ax.set_xlabel('Trial', fontsize=20)
        ax.set_ylabel(param_sym_name[p].replace('_', ' '), fontsize=20)
        if p == 2:
            plt.gca().invert_yaxis()
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if print_plots:
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            plt.savefig(path_save + param_sym_name[p] + '_sym_bs_average', dpi=128)
    plt.close('all')

    if control_bool:
        param_sym_bs_plot = param_sym_bs[:, :, :]
        param_sym_bs_control = np.load(control_path + 'split_' + control_ses + '_fast_control_params_sym_bs_noMC16846.npy')
        for p in range(np.shape(param_sym)[0] - 1):
            param_sym_bs_ave = param_sym_bs_plot[p, :, :]
            fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
            rectangle = plt.Rectangle((stim_start - 0.5, np.min(param_sym_bs_ave[:, :].flatten())), stim_duration,
                                      np.max(param_sym_bs_ave[:, :].flatten()) - np.min(
                                          param_sym_bs_ave[:, :].flatten()),
                                      fc='dimgrey', alpha=0.3)
            plt.gca().add_patch(rectangle)
            plt.hlines(0, 1, len(param_sym_bs_ave[0, :]), colors='grey', linestyles='--')
            for a in range(np.shape(param_sym_bs_ave)[0]):
                plt.plot(np.linspace(1, len(param_sym_bs_ave[a, :]), len(param_sym_bs_ave[a, :])),
                         param_sym_bs_ave[a, :], color='darkgray', linewidth=1)
            plt.plot(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])),
                     np.nanmean(param_sym_bs_ave, axis=0), color='orange', linewidth=2)
            plt.plot(np.linspace(1, len(param_sym_bs_ave[0, :]), len(param_sym_bs_ave[0, :])),
                     np.nanmean(param_sym_bs_control[p, :, :], axis=0), color='black', linewidth=2)
            ax.set_xlabel('Trial', fontsize=20)
            ax.set_ylabel(param_sym_name[p].replace('_', ' '), fontsize=20)
            if p == 2:
                plt.gca().invert_yaxis()
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if print_plots:
                if not os.path.exists(path_save):
                    os.mkdir(path_save)
                plt.savefig(path_save + param_sym_name[p] + '_sym_bs_average_with_control', dpi=128)
        plt.close('all')

    # PLOT STANCE SPEED
    for a in range(np.shape(stance_speed)[1]):
        data = stance_speed[:, a, :]
        fig, ax = plt.subplots(figsize=(7,10), tight_layout=True)
        rectangle = plt.Rectangle((stim_start-0.5, stim_duration), stim_duration, 210-(-10), fc='dimgrey',alpha=0.3)
        for p in range(4):
            ax.axvline(x = stim_start + 0.5, color='dimgray', linestyle='--')
            ax.axvline(x = stim_start + 0.5 + stim_duration, color='dimgray', linestyle='--')
            ax.plot(np.linspace(1,len(data[p,:]),len(data[p,:])), data[p,:], color = paw_colors[p], linewidth = 2)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('Trial', fontsize = 20)
            ax.set_ylabel('Stance speed', fontsize = 20)
            ax.tick_params(axis='x',labelsize = 16)
            ax.tick_params(axis='y',labelsize = 16)
            ax.set_title(animal_list[a],fontsize=18)
        if print_plots:
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            plt.savefig(path_save + animal_list[a] + '_stancespeed', dpi=96)
    plt.close('all')

    # CONTINUOUS STEP LENGTH WITH LASER ON
    trials = np.arange(1, 28)
    for count_animal, animal in enumerate(animal_list):
        param_sl = []
        for count_trial, f in enumerate(filelist):
            [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, 0)
            [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
            paws_rel = loco.get_paws_rel(final_tracks, 'X')
            for count_p, param in enumerate(param_sym_name):
                param_mat = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, 'step_length')
        param_sl.append(param_mat)
        [stride_idx, trial_continuous, sl_time, sl_values] = loco.param_continuous_sym(param_sl, st_strides_trials, trials, 'FR', 'FL', 1, 1)
        fig, ax = plt.subplots(tight_layout=True, figsize=(25,10))
        sl_time_start = sl_time[np.where(np.array(trial_continuous) == stim_start)[0][0]]
        sl_time_duration = sl_time[np.where(np.array(trial_continuous) == stim_start)[0][0]]+(loco.trial_time*stim_duration)
        rectangle = plt.Rectangle((sl_time_start, np.nanmin(sl_values)), sl_time_duration-sl_time_start, np.nanmax(sl_values)+np.abs(np.nanmin(sl_values)), fc='dimgrey', alpha=0.3)
        plt.gca().add_patch(rectangle)
        ax.plot(sl_time, sl_values, color='black')
        ax.set_xlabel('time (s)')
        ax.set_title('continuous step length')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if print_plots:
            if not os.path.exists(path_save):
                os.mkdir(path_save)
            plt.savefig(path_save + animal_list[a] + '_sl_sym_continuous', dpi=96)
