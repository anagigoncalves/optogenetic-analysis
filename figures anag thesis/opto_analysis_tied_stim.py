import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
import pandas as pd

#path inputs
path_loco = 'J:\\Opto JAWS Data\\tied stance stim\\'
event_stim = path_loco.split('\\')[-2].split(' ')[1]
experiment = path_loco.split('\\')[-2].replace(' ', '_')
if event_stim == 'stance':
    color_cond = 'orange'
if event_stim == 'swing':
    color_cond = 'green'
paws = ['FR', 'HR', 'FL', 'HL']
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']
# # stance
# animals = ['MC16851', 'MC17665', 'MC17666', 'MC17670', 'MC19082', 'MC19130', 'MC19214', 'MC19107']
# animals_triggers = ['MC16851', 'MC17665', 'MC17666', 'MC17670', 'MC19082', 'MC19130', 'MC19214']
# animals_long_session = ['MC19082', 'MC19130', 'MC19214', 'MC19107']
# #swing
# animals = ['MC17319', 'MC17665', 'MC17666', 'MC17670', 'MC19082', 'MC19124', 'MC19130', 'MC19214', 'MC19107']
# animals_triggers = ['MC17319', 'MC17665', 'MC17666', 'MC17670', 'MC19082', 'MC19130']
# animals_long_session = ['MC19082', 'MC19124', 'MC19130', 'MC19214', 'MC19107']
##animals matched with high accuracy for both sessions
animals = ['MC17665', 'MC17666', 'MC17670', 'MC19082', 'MC19130', 'MC19214']
animals_triggers = ['MC17665', 'MC17666', 'MC17670', 'MC19082', 'MC19130'] #add MC19214 here for stance stim
animals_long_session = ['MC19082', 'MC19130', 'MC19214']
#BAD RT TRACKING IN STANCE FOR MC17319 AND MC19124
#BAD RT TRACKING IN SWING FOR MC16851
stim_trials = np.arange(9, 17)

#import classes
os.chdir('C:\\Users\\Ana\\Documents\\PhD\\Dev\\optogenetic-analysis\\')
import locomotion_class
loco = locomotion_class.loco_class(path_loco)
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path_loco)
path_save = path_loco+'grouped output\\'
if not os.path.exists(path_save):
    os.mkdir(path_save)

animal_session_list = loco.animals_within_session()
animal_list = []
for a in range(len(animal_session_list)):
    animal_list.append(animal_session_list[a][0])
animal_list_plot = [a for count_a, a in enumerate(animal_list) if a in animals]
animal_list_plot_idx = np.array([count_a for count_a, a in enumerate(animal_list) if a in animals])
session_list = []
for a in range(len(animal_session_list)):
    session_list.append(animal_session_list[a][1])
session_list_plot = np.array(session_list)[animal_list_plot_idx]
Ntrials = 24
animals_triggers_idx = []
for count_i, i in enumerate(animals):
    if i in animals_triggers:
        animals_triggers_idx.append(count_i)

#summary gait parameters
param_sym_name = ['coo', 'step_length', 'double_support', 'coo_stance', 'coo_swing', 'swing_length', 'phase_st', 'stance_speed']
param_sym_label = ['Center of oscillation\nsymmetry (mm)', 'Step length\nsymmetry (mm)',
    'Percentage of double\nsupport symmetry', 'Spatial motor output\nsymmetry (mm)',
        'Temporal motor output\nsymmetry (mm)', 'Swing length\nsymmetry (mm)']
param_label = ['Center of\noscillation (mm)', 'Step length (mm)',
    'Percentage of\ndouble support', 'Spatial motor\noutput (mm)', 'Temporal motor\noutput (mm)', 'Swing length(mm)']
phase_label = 'Stance phasing\n(degrees)'
stance_speed_label = 'Stance speed (m/s)'
param_sym = np.zeros((len(param_sym_name), len(animal_list_plot), Ntrials))
param_sym[:] = np.nan
param_paw = np.zeros((len(param_sym_name), len(animal_list_plot), 4, Ntrials))
param_paw[:] = np.nan
param_phase = np.zeros((4, len(animal_list_plot), Ntrials))
param_phase[:] = np.nan
stance_speed = np.zeros((4, len(animal_list_plot), Ntrials))
stance_speed[:] = np.nan
for count_animal, animal in enumerate(animal_list_plot):
    session = int(session_list_plot[count_animal])
    filelist = loco.get_track_files(animal, session)
    # get trial list in filelist
    trial_filelist = np.zeros(len(filelist))
    for count_f, f in enumerate(filelist):
        trial_filelist[count_f] = np.int64(f.split('_')[7][:f.split('_')[7].find('D')])
    #for the first animals the session was shorter was 8-8-8 instead of 8-10-10
    if animal in animals_long_session:
        #remove the extra trials
        trial_filelist_corr = np.delete(trial_filelist, [np.where(trial_filelist == 17)[0][0],
            np.where(trial_filelist == 18)[0][0], np.where(trial_filelist == 27)[0][0], np.where(trial_filelist == 28)[0][0]])
        trial_filelist_corr[np.where(trial_filelist_corr<=16)[0][-1]+1:] = trial_filelist_corr[np.where(trial_filelist_corr<=16)[0][-1]+1:]-2
        filelist_corr = np.delete(filelist, [np.where(trial_filelist == 17)[0][0],
            np.where(trial_filelist == 18)[0][0], np.where(trial_filelist == 27)[0][0], np.where(trial_filelist == 28)[0][0]])
    else:
        trial_filelist_corr = trial_filelist
        filelist_corr = filelist
    #account for missing trials and sessions of different lengths
    trials_idx = np.arange(0, 24)
    trials_ses = np.arange(1, 25)
    trials_idx_corr = np.zeros(len(trial_filelist_corr))
    for count_t, t in enumerate(trial_filelist_corr):
        trials_idx_corr[count_t] = trials_idx[np.where(t == trials_ses)[0][0]]
    trials_idx_corr = np.int64(trials_idx_corr)
    for count_trial, f in enumerate(filelist_corr):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, 0)
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        paws_rel = loco.get_paws_rel(final_tracks, 'X')
        for count_p, param in enumerate(param_sym_name):
            param_mat = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param)
            if param == 'phase_st':
                for p in range(4): #reference HL
                    param_phase[p, count_animal, trials_idx_corr[count_trial]] = sp.circmean(param_mat[3][p], nan_policy='omit')
            elif param == 'stance_speed':
                for p in range(4):
                    stance_speed[p, count_animal, trials_idx_corr[count_trial]] = np.nanmean(param_mat[p])
            else:
                param_sym[count_p, count_animal, trials_idx_corr[count_trial]] = np.nanmean(param_mat[0])-np.nanmean(param_mat[2])
                for count_paw, paw in enumerate(paws):
                    param_paw[count_p, count_animal, count_paw, trials_idx_corr[count_trial]] = np.nanmean(param_mat[count_paw])

#Plot
#baseline subtracion of parameters
param_sym_bs = np.zeros(np.shape(param_sym))
param_sym_bs[:] = np.nan
param_paw_bs = np.zeros(np.shape(param_paw))
param_paw_bs[:] = np.nan
for p in range(np.shape(param_sym)[0]-2):
    for a in range(np.shape(param_sym)[1]):
        bs_mean = np.nanmean(param_sym[p, a, :stim_trials[0]-1])
        param_sym_bs[p, a, :] = param_sym[p, a, :] - bs_mean
        for count_paw in range(4):
            bs_paw_mean = np.nanmean(param_paw[p, a, count_paw, :stim_trials[0]-1])
            param_paw_bs[p, a, count_paw, :] = param_paw[p, a, count_paw, :] - bs_paw_mean

np.save(os.path.join(path_loco, path_save, 'param_sym_bs.npy'), param_sym_bs)
np.save(os.path.join(path_loco, path_save, 'param_paw_bs.npy'), param_paw_bs)
np.save(os.path.join(path_loco, path_save, 'param_paw.npy'), param_paw)
np.save(os.path.join(path_loco, path_save, 'param_phase.npy'), param_phase)
np.save(os.path.join(path_loco, path_save, 'animal_order.npy'), animal_list_plot)

# #plot symmetry baseline subtracted - mean animals
# for p in range(np.shape(param_sym)[0]-2):
#     fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
#     mean_data = np.nanmean(param_sym_bs[p, :, :], axis=0)
#     std_data = (np.nanstd(param_sym_bs[p, :, :], axis=0)/np.sqrt(len(animals)))
#     rectangle = plt.Rectangle((stim_trials[0]-0.5, np.nanmin(mean_data-std_data)), 8, np.nanmax(mean_data+std_data)-np.nanmin(mean_data-std_data), fc='lightblue', alpha=0.3)
#     plt.gca().add_patch(rectangle)
#     plt.hlines(0, 1, Ntrials, colors='grey', linestyles='--')
#     plt.plot(np.arange(1, Ntrials+1), mean_data, linewidth=2, marker='o', color=color_cond)
#     plt.fill_between(np.arange(1, Ntrials+1), mean_data-std_data, mean_data+std_data, color=color_cond, alpha=0.5)
#     ax.set_xlabel('Trial', fontsize=20)
#     ax.set_ylabel(param_sym_label[p], fontsize=20)
#     if p == 2:
#         plt.gca().invert_yaxis()
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.savefig(os.path.join(path_save, 'mean_animals_symmetry_' + param_sym_name[p] + '.png'), dpi=128)
#     plt.savefig(os.path.join(path_save, 'mean_animals_symmetry_' + param_sym_name[p] + '.svg'), dpi=128)
# plt.close('all')
#
# #plot individual limbs baseline subtracted - mean animals DO NOT BS AND SEE IF BETTER
# for p in range(np.shape(param_sym)[0]-2):
#     fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
#     mean_data = np.vstack((np.nanmean(param_paw_bs[p, :, 0, :], axis=0), np.nanmean(param_paw_bs[p, :, 1, :], axis=0),
#         np.nanmean(param_paw_bs[p, :, 2, :], axis=0), np.nanmean(param_paw_bs[p, :, 3, :], axis=0)))
#     std_data = np.vstack((np.nanstd(param_paw_bs[p, :, 0, :], axis=0)/np.sqrt(len(animals)),
#         np.nanstd(param_paw_bs[p, :, 1, :], axis=0)/np.sqrt(len(animals)),
#         np.nanstd(param_paw_bs[p, :, 2, :], axis=0)/np.sqrt(len(animals)),
#         np.nanstd(param_paw_bs[p, :, 3, :], axis=0)/np.sqrt(len(animals))))
#     rectangle = plt.Rectangle((stim_trials[0]-0.5, np.nanmin(mean_data-std_data)), 10, np.nanmax(mean_data+std_data)-np.nanmin(mean_data-std_data), fc='lightgray', alpha=0.3)
#     plt.gca().add_patch(rectangle)
#     plt.hlines(0, 1, Ntrials, colors='grey', linestyles='--')
#     for paw in range(4):
#         plt.plot(np.arange(1, Ntrials+1), np.nanmean(param_paw_bs[p, :, paw, :], axis=0), linewidth=2, color=paw_colors[paw])
#         plt.fill_between(np.arange(1, Ntrials+1),
#             np.nanmean(param_paw_bs[p, :, paw, :], axis=0)-(np.nanstd(param_paw_bs[p, :, paw, :], axis=0)/np.sqrt(len(animals))),
#             np.nanmean(param_paw_bs[p, :, paw, :], axis=0)+(np.nanstd(param_paw_bs[p, :, paw, :], axis=0)/np.sqrt(len(animals))), color=paw_colors[paw], alpha=0.5)
#     ax.set_xlabel('Trial', fontsize=20)
#     ax.set_ylabel(param_label[p], fontsize=20)
#     # if p == 2:
#     #     plt.gca().invert_yaxis()
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.savefig(os.path.join(path_save, 'mean_animals_paws_' + param_sym_name[p] + '.png'), dpi=128)
#     plt.savefig(os.path.join(path_save, 'mean_animals_paws_' + param_sym_name[p] + '.svg'), dpi=128)
# plt.close('all')
#
# #plot symmetry baseline subtracted - individual animals
# for p in range(np.shape(param_sym)[0]-2):
#     fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
#     rectangle = plt.Rectangle((stim_trials[0]-0.5, np.min(param_sym_bs[p, :, :].flatten())), 8, np.max(param_sym_bs[p, :, :].flatten())-np.min(param_sym_bs[p, :, :].flatten()), fc='lightblue', alpha=0.3)
#     plt.gca().add_patch(rectangle)
#     plt.hlines(0, 1, len(param_sym_bs[p, 0, :]), colors='grey', linestyles='--')
#     for a in range(np.shape(param_sym)[1]):
#         plt.plot(np.arange(1, Ntrials+1), param_sym_bs[p, a, :], label=animal_list_plot[a], linewidth=2)
#     ax.set_xlabel('Trial', fontsize=20)
#     ax.legend(frameon=False)
#     ax.set_ylabel(param_sym_label[p], fontsize=20)
#     if p == 2:
#         plt.gca().invert_yaxis()
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.savefig(os.path.join(path_save, 'ind_animals_symmetry_' + param_sym_name[p] + '.png'), dpi=128)
# plt.close('all')
#
# # plot stance phase - group mean
# #error bars in polar plot don't rotate well
# fig = plt.figure(figsize=(10, 10), tight_layout=True)
# ax = fig.add_subplot(111, projection='polar')
# for paw in range(3):
#     data_mean = sp.circmean(param_phase[paw, :, :], axis=0)
#     ax.scatter(data_mean, np.arange(1, Ntrials + 1), c=paw_colors[paw], s=30)
# ax.set_yticks([8.5, 16.5])
# ax.set_yticklabels(['', ''])
# ax.tick_params(axis='both', which='major', labelsize=20)
# plt.savefig(os.path.join(path_save, 'mean_animals_stance_phase.png'), dpi=256)
# plt.savefig(os.path.join(path_save, 'mean_animals_stance_phase.svg'), dpi=256)
#
# #plot stance phase - individual animals
# fig, ax = plt.subplots(4, 3, figsize=(20, 20), tight_layout=True, sharey=True, sharex=True)
# ax = ax.ravel()
# for count_a in range(len(animals)):
#     for p in range(4):
#         ax[count_a].axvline(x=stim_trials[0]-0.5, color='dimgray', linestyle='--')
#         ax[count_a].axvline(x=stim_trials[0]+10-0.5, color='dimgray', linestyle='--')
#         ax[count_a].plot(np.arange(1, Ntrials+1), np.rad2deg(param_phase[p, count_a, :]), color=paw_colors[p], linewidth=2)
#         ax[count_a].spines['right'].set_visible(False)
#         ax[count_a].spines['top'].set_visible(False)
#         ax[count_a].tick_params(axis='x')
#         # ax[count_a].set_ylim([70, 230])
#         ax[count_a].tick_params(axis='y')
#         ax[count_a].set_title(animals[count_a])
# plt.savefig(os.path.join(path_save, 'ind_animals_stance_phase.png'), dpi=128)
# plt.close('all')

# accuracy_animals = np.zeros((len(animals_triggers), Ntrials))
# accuracy_animals[:] = np.nan
# light_onset_phase_animals = []
# light_offset_phase_animals = []
# stim_nr_animals = []
# stride_nr_animals = []
# fraction_strides_stim_on_animals = np.zeros((len(animals), Ntrials))
# fraction_strides_stim_on_animals[:] = np.nan
# fraction_strides_stim_off_animals = np.zeros((len(animals), Ntrials))
# fraction_strides_stim_off_animals[:] = np.nan
# offtracks_phase_stim_animals = []
# for count_a, animal in enumerate(animals_triggers):
#     trials = otrack_class.get_trials(animal)
#     # LOAD PROCESSED DATA
#     [otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, laser_on] = otrack_class.load_processed_files(animal)
#     # READ OFFLINE PAW EXCURSIONS
#     [final_tracks_trials, st_strides_trials, sw_strides_trials] = otrack_class.get_offtrack_paws(loco, animal, np.int64(session_list_plot[[i for i, s in enumerate(animal_list_plot) if s == animal][0]]))
#     final_tracks_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials,
#                                                  'st-sw-st')
#     # LASER ACCURACY
#     tp_laser = np.zeros(len(trials))
#     fp_laser = np.zeros(len(trials))
#     tn_laser = np.zeros(len(trials))
#     fn_laser = np.zeros(len(trials))
#     precision_laser = np.zeros(len(trials))
#     recall_laser = np.zeros(len(trials))
#     f1_laser = np.zeros(len(trials))
#     for count_t, trial in enumerate(trials):
#         [tp_trial, fp_trial, tn_trial, fn_trial, precision_trial, recall_trial, f1_trial] = otrack_class.accuracy_laser_sync(trial, event_stim, offtracks_st, offtracks_sw, laser_on, final_tracks_trials, timestamps_session, 0)
#         tp_laser[count_t] = tp_trial
#         fp_laser[count_t] = fp_trial
#         tn_laser[count_t] = tn_trial
#         fn_laser[count_t] = fn_trial
#         precision_laser[count_t] = precision_trial
#         recall_laser[count_t] = recall_trial
#         f1_laser[count_t] = f1_trial
#     fig, ax = plt.subplots(tight_layout=True, figsize=(10, 7))
#     rectangle = plt.Rectangle((stim_trials[0]-0.5, 0), stim_trials[-1]-stim_trials[0], 1, fc='lightblue',alpha=0.3)
#     plt.gca().add_patch(rectangle)
#     ax.plot(trials, tp_laser+tn_laser, marker='o', color='black', linewidth=2)
#     ax.set_ylim([0, 1])
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.set_title(animal, fontsize=16)
#     ax.set_ylabel('Accuracy', fontsize=14)
#     plt.savefig(path_save + 'ind_animals_' + animal + '_laser_performance_accuracy.png')
#
#     # get trial list in filelist
#     trial_filelist = np.copy(trials)
#     #for the first animals the session was shorter was 8-8-8 instead of 8-10-10
#     if animal in animals_long_session:
#         #remove the extra trials
#         trial_filelist_corr = np.delete(trial_filelist, [np.where(trial_filelist == 17)[0][0],
#             np.where(trial_filelist == 18)[0][0], np.where(trial_filelist == 27)[0][0], np.where(trial_filelist == 28)[0][0]])
#         trial_filelist_corr[np.where(trial_filelist_corr<=16)[0][-1]+1:] = trial_filelist_corr[np.where(trial_filelist_corr<=16)[0][-1]+1:]-2
#         stim_trials_in = trials[np.where(trials == 9)[0][0]:np.where(trials == 16)[0][0]+1]
#     else:
#         trial_filelist_corr = trial_filelist
#         stim_trials_in = trials[np.where(trials == 9)[0][0]:np.where(trials == 16)[0][0]+1]
#     #account for missing trials and sessions of different lengths
#     trials_idx = np.arange(0, 24)
#     trials_ses = np.arange(1, 25)
#     trials_idx_corr = np.zeros(len(trial_filelist_corr))
#     for count_t, t in enumerate(trial_filelist_corr):
#         trials_idx_corr[count_t] = trials_idx[np.where(t == trials_ses)[0][0]]
#     trials_idx_corr = np.int64(trials_idx_corr)
#     stim_trials_in_idx = np.zeros(len(stim_trials_in))
#     for count_t, t in enumerate(stim_trials_in):
#         stim_trials_in_idx[count_t] = trials_idx[np.where(t == trials_ses)[0][0]]
#     stim_trials_in_idx = np.int64(stim_trials_in_idx)
#     #save accuracy
#     if animal in animals_long_session:
#         tp_corr = np.delete(tp_laser, [np.where(trial_filelist == 17)[0][0],
#             np.where(trial_filelist == 18)[0][0], np.where(trial_filelist == 27)[0][0], np.where(trial_filelist == 28)[0][0]])
#         tn_corr = np.delete(tn_laser, [np.where(trial_filelist == 17)[0][0],
#             np.where(trial_filelist == 18)[0][0], np.where(trial_filelist == 27)[0][0], np.where(trial_filelist == 28)[0][0]])
#         accuracy_animals[count_a, trials_idx_corr] = tp_corr + tn_corr
#     else:
#         accuracy_animals[count_a, trials_idx_corr] = tp_laser+tn_laser
#
#     #LASER ONSET AND OFFSET PHASE
#     light_onset_phase_all = []
#     light_offset_phase_all = []
#     stim_nr_trials = np.zeros(len(stim_trials_in))
#     stride_nr_trials = np.zeros(len(stim_trials_in))
#     for count_t, trial in enumerate(stim_trials_in):
#         [light_onset_phase, light_offset_phase, stim_nr, stride_nr] = \
#             otrack_class.laser_presentation_phase_all(trial, trials, event_stim, offtracks_st, offtracks_sw, laser_on,
#                                                   timestamps_session, final_tracks_phase, 'FR')
#         stim_nr_trials[count_t] = stim_nr
#         stride_nr_trials[count_t] = stride_nr
#         light_onset_phase_all.extend(light_onset_phase)
#         light_offset_phase_all.extend(light_offset_phase)
#     #Plot laser timing for all trials
#     #normalize by number of stimulations 1, 0
#     [fraction_strides_stim_on, fraction_strides_stim_off] = otrack_class.plot_laser_presentation_phase(light_onset_phase_all,
#                     light_offset_phase_all, event_stim, 16, np.sum(stim_nr_trials), np.sum(stride_nr_trials), 1, 0,
#                     path_save, 'ind_animals_' + animal + '_' + event_stim, 1)
#     otrack_class.plot_laser_presentation_phase_hist(light_onset_phase_all, light_offset_phase_all,
#                                                     20, path_save, 'ind_animals_hist_' + animal + '_' + event_stim, 1)
#     light_onset_phase_animals.extend(light_onset_phase_all)
#     light_offset_phase_animals.extend(light_offset_phase_all)
#     stim_nr_animals.append(np.sum(stim_nr_trials))
#     stride_nr_animals.append(np.sum(stride_nr_trials))
#     # Get fraction strides stimulated for each trial
#     frac_on = np.zeros(len(trials))
#     frac_off = np.zeros(len(trials))
#     for count_t, trial in enumerate(trials):
#         [light_onset_phase, light_offset_phase, stim_nr, stride_nr] = \
#             otrack_class.laser_presentation_phase_all(trial, trials, event_stim, offtracks_st, offtracks_sw, laser_on,
#                                                   timestamps_session, final_tracks_phase, 'FR')
#         [fraction_strides_stim_on, fraction_strides_stim_off] = otrack_class.plot_laser_presentation_phase(
#             light_onset_phase, light_onset_phase, event_stim, 16, stim_nr, stride_nr, 1, 0,
#             path_save, 'ind_animals_' + animal + '_' + event_stim, 0)
#         plt.close('all')
#         frac_on[count_t] = fraction_strides_stim_on
#         frac_off[count_t] = fraction_strides_stim_off
#
#     if animal in animals_long_session:
#         frac_on_corr = np.delete(frac_on, [np.where(trial_filelist == 17)[0][0],
#             np.where(trial_filelist == 18)[0][0], np.where(trial_filelist == 27)[0][0], np.where(trial_filelist == 28)[0][0]])
#         frac_off_corr = np.delete(frac_off, [np.where(trial_filelist == 17)[0][0],
#             np.where(trial_filelist == 18)[0][0], np.where(trial_filelist == 27)[0][0], np.where(trial_filelist == 28)[0][0]])
#         fraction_strides_stim_on_animals[count_a, trials_idx_corr] = frac_on_corr
#         fraction_strides_stim_off_animals[count_a, trials_idx_corr] = frac_off_corr
#     else:
#         fraction_strides_stim_on_animals[count_a, trials_idx_corr] = frac_on
#         fraction_strides_stim_off_animals[count_a, trials_idx_corr] = frac_off
#
#     # Laser timing with symmetry
#     offtracks_phase = loco.get_symmetry_laser_phase_offtracks_df(animal, np.int64(session_list_plot[[i for i, s in enumerate(animal_list_plot) if s == animal][0]]), trials, final_tracks_phase, event_stim, laser_on,
#                 timestamps_session, offtracks_st, offtracks_sw, ['coo', 'step_length', 'double_support', 'coo_stance', 'swing_length'])
#     offtracks_phase_stim = offtracks_phase.loc[(offtracks_phase['trial']>stim_trials[0]-1) & (offtracks_phase['trial']<stim_trials[-1]+1)]
#     offtracks_phase_stim.to_csv(
#         os.path.join(path_loco, path_save, 'offtracks_phase_stim_' + experiment + '_' + animal + '.csv'), sep=',',
#         index=False)
#     offtracks_phase_stim_animals.append(offtracks_phase_stim)
#
# # MEAN LASER ACCURACY
# fig, ax = plt.subplots(tight_layout=True, figsize=(10, 7))
# mean_data = np.nanmean(accuracy_animals, axis=0)
# std_data = (np.nanstd(accuracy_animals, axis=0) / np.sqrt(np.shape(accuracy_animals)[0]))
# rectangle = plt.Rectangle((stim_trials[0] - 0.5, 0), 8, 1, fc='lightblue', zorder=-1, alpha=0.3)
# plt.gca().add_patch(rectangle)
# ax.plot(np.arange(1, Ntrials+1), mean_data, marker='o', color='black', linewidth=2)
# ax.fill_between(np.arange(1, Ntrials+1), mean_data - std_data, mean_data + std_data, color='black', alpha=0.5)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# ax.set_ylim([0, 1])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_ylabel('Accuracy', fontsize=20)
# plt.savefig(path_save + 'mean_animals_laser_' + event_stim + '_performance_accuracy.png')
# plt.savefig(path_save + 'mean_animals_laser_' + event_stim + '_performance_accuracy.svg')
#
# # MEAN LASER TIMING - HISTOGRAM WITH DURATION
# [fraction_strides_stim_on_all, fraction_strides_stim_off_all] = \
#     otrack_class.plot_laser_presentation_phase(light_onset_phase_animals, light_offset_phase_animals, event_stim,
#     20, np.sum(stim_nr_animals), np.sum(stride_nr_animals), 1, 0, path_save, 'mean_animals_' + event_stim, 1)
#
# # MEAN LASER TIMING - HISTOGRAM
# otrack_class.plot_laser_presentation_phase_hist(light_onset_phase_animals, light_offset_phase_animals,
#                                           20, path_save, 'mean_animals_hist_' + event_stim, 1)
#
# # FRACTION OF STIMULATED STRIDES - MEAN
# fig, ax = plt.subplots(tight_layout=True, figsize=(10, 7))
# mean_data = np.nanmean(fraction_strides_stim_on_animals, axis=0)
# std_data = (np.nanstd(fraction_strides_stim_on_animals, axis=0) / np.sqrt(np.shape(fraction_strides_stim_on_animals)[0]))
# rectangle = plt.Rectangle((stim_trials[0] - 0.5, 0), 8, 1, fc='lightblue', zorder=0, alpha=0.3)
# plt.gca().add_patch(rectangle)
# ax.plot(np.arange(1, Ntrials+1), mean_data, marker='o', color='black', linewidth=2)
# ax.fill_between(np.arange(1, Ntrials+1), mean_data - std_data, mean_data + std_data, color='black', alpha=0.5)
# ax.set_ylim([0, 1])
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_ylabel('Fraction of stimulated\nstrides in the onset', fontsize=20)
# plt.savefig(path_save + 'mean_animals_laser_' + event_stim + '_fraction_stim_strides_onset.png')
# plt.savefig(path_save + 'mean_animals_laser_' + event_stim + '_fraction_stim_strides_onset.svg')
# fig, ax = plt.subplots(tight_layout=True, figsize=(10, 7))
# mean_data = np.nanmean(fraction_strides_stim_off_animals, axis=0)
# std_data = (np.nanstd(fraction_strides_stim_off_animals, axis=0) / np.sqrt(np.shape(fraction_strides_stim_off_animals)[0]))
# rectangle = plt.Rectangle((stim_trials[0] - 0.5, 0), 8, 1, fc='lightblue', zorder=0, alpha=0.3)
# plt.gca().add_patch(rectangle)
# ax.plot(np.arange(1, Ntrials+1), mean_data, marker='o', color='black', linewidth=2)
# ax.fill_between(np.arange(1, Ntrials+1), mean_data - std_data, mean_data + std_data, color='black', alpha=0.5)
# ax.set_ylim([0, 1])
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_ylabel('Fraction of stimulated\nstrides in the offset', fontsize=20)
# plt.savefig(path_save + 'mean_animals_laser_' + event_stim + '_fraction_stim_strides_offset.png')
# plt.savefig(path_save + 'mean_animals_laser_' + event_stim + '_fraction_stim_strides_offset.svg')
#
# # LASER TIMING WITH SYMMETRY
# for p in range(3):
#     fig, ax = plt.subplots(2, 1, figsize=(12, 7), tight_layout=True, sharex=True, sharey=True)
#     ax = ax.ravel()
#     for count_animal, animal in enumerate(animals_triggers):
#         ax[0].scatter(offtracks_phase_stim_animals[count_animal]['onset'], offtracks_phase_stim_animals[count_animal][param_sym_name[p]]*100, s=5, color=color_cond)
#         ax[1].scatter(offtracks_phase_stim_animals[count_animal]['offset'], offtracks_phase_stim_animals[count_animal][param_sym_name[p]]*100, s=5, color=color_cond)
#         ax[1].set_xlabel('stride phase (%)', fontsize=20)
#         ax[1].set_ylabel(param_sym_label[p] + '\n for stim offset', fontsize=20)
#         ax[0].set_ylabel(param_sym_label[p] + '\n for stim onset', fontsize=20)
#         ax[0].tick_params(axis='both', labelsize=20)
#         ax[1].tick_params(axis='both', labelsize=20)
#         ax[0].spines['right'].set_visible(False)
#         ax[0].spines['top'].set_visible(False)
#         ax[1].spines['right'].set_visible(False)
#         ax[1].spines['top'].set_visible(False)
#     plt.savefig(path_save + 'mean_animals_laser_phase_sym_' + event_stim + '_' + param_sym_name[p] + '.png')
#
# # FRACTION OF STIMULATED STRIDES - INDIVIDUAL ANIMALS
# fig, ax = plt.subplots(4, 3, figsize=(20, 20), tight_layout=True, sharey=True, sharex=True)
# ax = ax.ravel()
# for count_a in range(len(animals_triggers)):
#     ax[count_a].axvline(x=stim_trials[0]-0.5, color='dimgray')
#     ax[count_a].axvline(x=stim_trials[0]+10-0.5, color='dimgray')
#     ax[count_a].plot(np.arange(1, Ntrials+1), fraction_strides_stim_on_animals[count_a, :], color='black', linewidth=2)
#     ax[count_a].plot(np.arange(1, Ntrials+1), fraction_strides_stim_off_animals[count_a, :], color='black', linewidth=2,
#     linestyle='dashed')
#     ax[count_a].spines['right'].set_visible(False)
#     ax[count_a].spines['top'].set_visible(False)
#     ax[count_a].tick_params(axis='x')
#     ax[count_a].tick_params(axis='y')
#     ax[count_a].set_title(animals[count_a])
# plt.savefig(path_save + 'ind_animals_laser_' + event_stim + '_fraction_stim_strides_onset.png')
# plt.close('all')
#
# # ACCURACY VERSUS STIMULATION EFFECT
# param_sym_label_ae = ['Center of oscillation\nafter-effect symmetry (mm)', 'Step length\nafter-effect symmetry(mm)', 'Percentage of double support\nafter-effect symmetry', 'Center of oscillation\n stance after-effect symmetry (mm)',
#         'Swing length\nafter-effect symmetry (mm)']
# for count_p in range(len(param_sym_label_ae)):
#     fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
#     ax.scatter(np.nanmean(accuracy_animals[:, stim_trials-1], axis=1), np.abs(param_sym_bs[count_p, animals_triggers_idx, stim_trials[-1]+1]), color='black')
#     ax.set_xlabel('Accuracy', fontsize=16)
#     ax.set_ylabel(param_sym_label_ae[count_p], fontsize=16)
#     ax.tick_params(axis='both', which='major', labelsize=16)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.savefig(path_save + param_sym_name[count_p] + 'after_effect_accuracy_quantification', dpi=256)
# param_sym_label_delta = ['Change over stim. of\ncenter of oscillation symmetry (mm)', 'Change over stim. of\nstep length symmetry (mm)', 'Change over stim. of\npercentage of double support symmetry', 'Change over stim. of\ncenter of oscillation stance symmetry (mm)',
#         'Change over stim. of\nswing length symmetry (mm)']
# for count_p in range(len(param_sym_label_delta)):
#     fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
#     ax.scatter(np.nanmean(accuracy_animals[:, stim_trials-1], axis=1), param_sym_bs[count_p, animals_triggers_idx, stim_trials[-1]-3]-param_sym_bs[count_p, animals_triggers_idx, stim_trials[0]-1], color='black')
#     ax.set_xlabel('Accuracy', fontsize=16)
#     ax.set_ylabel(param_sym_label_delta[count_p], fontsize=16)
#     ax.tick_params(axis='both', which='major', labelsize=16)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.savefig(path_save + param_sym_name[count_p] + 'delta_stim_accuracy_quantification', dpi=256)
# plt.close('all')
#
# light_onset_phase_animals_hist = []
# light_offset_phase_animals_hist = []
# for count_a, animal in enumerate(animals_triggers):
#     trials = otrack_class.get_trials(animal)
#     # LOAD PROCESSED DATA
#     [otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, laser_on] = otrack_class.load_processed_files(animal)
#     # READ OFFLINE PAW EXCURSIONS
#     [final_tracks_trials, st_strides_trials, sw_strides_trials] = otrack_class.get_offtrack_paws(loco, animal, np.int64(session_list_plot[[i for i, s in enumerate(animal_list_plot) if s == animal][0]]))
#     final_tracks_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials,
#                                                  'st-sw-st')
#
#     # get trial list in filelist
#     trial_filelist = np.copy(trials)
#     #for the first animals the session was shorter was 8-8-8 instead of 8-10-10
#     if animal in animals_long_session:
#         #remove the extra trials
#         trial_filelist_corr = np.delete(trial_filelist, [np.where(trial_filelist == 17)[0][0],
#             np.where(trial_filelist == 18)[0][0], np.where(trial_filelist == 27)[0][0], np.where(trial_filelist == 28)[0][0]])
#         trial_filelist_corr[np.where(trial_filelist_corr<=16)[0][-1]+1:] = trial_filelist_corr[np.where(trial_filelist_corr<=16)[0][-1]+1:]-2
#         stim_trials_in = trials[np.where(trials == 9)[0][0]:np.where(trials == 16)[0][0]]
#     else:
#         trial_filelist_corr = trial_filelist
#         stim_trials_in = trials[np.where(trials == 9)[0][0]:np.where(trials == 16)[0][0]]
#     #account for missing trials and sessions of different lengths
#     trials_idx = np.arange(0, 24)
#     trials_ses = np.arange(1, 25)
#     trials_idx_corr = np.zeros(len(trial_filelist_corr))
#     for count_t, t in enumerate(trial_filelist_corr):
#         trials_idx_corr[count_t] = trials_idx[np.where(t == trials_ses)[0][0]]
#     trials_idx_corr = np.int64(trials_idx_corr)
#     stim_trials_in_idx = np.zeros(len(stim_trials_in))
#     for count_t, t in enumerate(stim_trials_in):
#         stim_trials_in_idx[count_t] = trials_idx[np.where(t == trials_ses)[0][0]]
#     stim_trials_in_idx = np.int64(stim_trials_in_idx)
#
#     #LASER ONSET AND OFFSET PHASE
#     light_onset_phase_all = []
#     light_offset_phase_all = []
#     stim_nr_trials = np.zeros(len(stim_trials_in))
#     stride_nr_trials = np.zeros(len(stim_trials_in))
#     for count_t, trial in enumerate(stim_trials_in):
#         [light_onset_phase, light_offset_phase, stim_nr, stride_nr] = \
#             otrack_class.laser_presentation_phase_all(trial, trials, event_stim, offtracks_st, offtracks_sw, laser_on,
#                                                   timestamps_session, final_tracks_phase, 'FR')
#         light_onset_phase_all.extend(light_onset_phase)
#         light_offset_phase_all.extend(light_offset_phase)
#     light_onset_phase_animals_hist.append(light_onset_phase_all)
#     light_offset_phase_animals_hist.append(light_offset_phase_all)
#
# if color_cond == 'green':
#     color_onset = 'green'
#     color_offset = 'lightgreen'
# if color_cond == 'orange':
#     color_onset = 'orange'
#     color_offset = 'gold'
# amp_plot = 400
# time = np.arange(-1, 2, np.round(1 / loco.sr, 3))
# FR = amp_plot * np.sin(2 * np.pi * time + (np.pi / 2)) + amp_plot
# fontsize_plot = 22
# fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
# ax.plot(time, FR, color='lightgray', zorder=0)
# for count_a in range(len(light_onset_phase_animals_hist)):
#     hist_onset = np.histogram(light_onset_phase_animals_hist[count_a], range=(
#         np.min(light_onset_phase_animals_hist[count_a]), np.max(light_onset_phase_animals_hist[count_a])), bins=20)
#     hist_offset = np.histogram(light_offset_phase_animals_hist[count_a], range=(
#         np.min(light_offset_phase_animals_hist[count_a]), np.max(light_offset_phase_animals_hist[count_a])), bins=20)
#     # weights_onset = np.ones_like(light_onset_phase_animals[count_a]) / np.max(hist_onset[0])
#     # weights_offset = np.ones_like(light_offset_phase_animals[count_a]) / np.max(hist_offset[0])
#     # ax.hist(light_onset_phase_animals[count_a], histtype='step', color=color_onset, alpha=1-(count_a)*0.2, linewidth=2, weights=weights_onset)
#     # ax.hist(light_offset_phase_animals[count_a], histtype='step', color=color_offset, alpha=1-(count_a)*0.2, linewidth=2, weights=weights_offset)
#     ax.hist(light_onset_phase_animals_hist[count_a], histtype='step', color=color_onset, alpha=1-(count_a)*0.1, linewidth=2)
#     ax.hist(light_offset_phase_animals_hist[count_a], histtype='step', color=color_offset, alpha=1-(count_a)*0.1, linewidth=2)
# ax.set_xticks([-1, -0.5, 0, 0.5, 1, 1.5, 2])
# ax.set_xticklabels(['-100', '-50', '0', '50', '100', '150', '200'])
# ax.set_xlabel('Stride phase (%)', fontsize=fontsize_plot)
# ax.set_ylabel('Laser presentation\ncounts', fontsize=fontsize_plot)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.tick_params(axis='both', which='major', labelsize=fontsize_plot - 2)
# plt.savefig(path_save + 'all_animals_hist_' + event_stim + '.png', dpi=256)
# plt.savefig(path_save + 'all_animals_hist_' + event_stim + '.svg', dpi=256)