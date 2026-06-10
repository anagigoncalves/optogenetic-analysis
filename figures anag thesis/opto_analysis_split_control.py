import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp

#path inputs
path_loco = 'J:\\Opto JAWS Data\\split right fast control\\'
paws = ['FR', 'HR', 'FL', 'HL']
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']
animals = ['MC16851', 'MC17319', 'MC17665', 'MC17670', 'MC19082', 'MC19124', 'MC19130', 'MC19214', 'MC19107']
stim_trials = np.arange(9, 19)

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
Ntrials = 28

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
param_sym_hind = np.zeros((len(param_sym_name), len(animal_list_plot), Ntrials))
param_sym_hind[:] = np.nan
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
    trials_idx = np.arange(0, 28)
    trials_ses = np.arange(1, 29)
    trials_idx_corr = np.zeros(len(trial_filelist))
    for count_t, t in enumerate(trial_filelist):
        trials_idx_corr[count_t] = trials_idx[np.where(t == trials_ses)[0][0]]
    trials_idx_corr = np.int64(trials_idx_corr)
    for count_trial, f in enumerate(filelist):
        [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, 0)
        [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
        paws_rel = loco.get_paws_rel(final_tracks, 'X')
        for count_p, param in enumerate(param_sym_name):
            param_mat = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param)
            if param == 'phase_st':
                for p in range(4): #HL as reference
                    param_phase[p, count_animal, trials_idx_corr[count_trial]] = sp.circmean(param_mat[3][p], nan_policy='omit')
            elif param == 'stance_speed':
                for p in range(4):
                    stance_speed[p, count_animal, trials_idx_corr[count_trial]] = np.nanmean(param_mat[p])
            else:
                param_sym[count_p, count_animal, trials_idx_corr[count_trial]] = np.nanmean(param_mat[0])-np.nanmean(param_mat[2])
                param_sym_hind[count_p, count_animal, trials_idx_corr[count_trial]] = np.nanmean(param_mat[1]) - np.nanmean(
                    param_mat[3])
                for count_paw, paw in enumerate(paws):
                    param_paw[count_p, count_animal, count_paw, trials_idx_corr[count_trial]] = np.nanmean(param_mat[count_paw])

#Plot
#baseline subtracion of parameters
param_sym_bs = np.zeros(np.shape(param_sym))
param_sym_hind_bs = np.zeros(np.shape(param_sym_hind))
param_paw_bs = np.zeros(np.shape(param_paw))
for p in range(np.shape(param_sym)[0]-2):
    for a in range(np.shape(param_sym)[1]):
        bs_mean = np.nanmean(param_sym[p, a, :stim_trials[0]-1])
        bs_mean_hind = np.nanmean(param_sym_hind[p, a, :stim_trials[0] - 1])
        param_sym_bs[p, a, :] = param_sym[p, a, :] - bs_mean
        param_sym_hind_bs[p, a, :] = param_sym_hind[p, a, :] - bs_mean_hind
        for count_paw in range(4):
            bs_paw_mean = np.nanmean(param_paw[p, a, count_paw, :stim_trials[0]-1])
            param_paw_bs[p, a, count_paw, :] = param_paw[p, a, count_paw, :] - bs_paw_mean
np.save(os.path.join(path_loco, path_save, 'param_sym_bs.npy'), param_sym_bs)
np.save(os.path.join(path_loco, path_save, 'param_phase.npy'), param_phase)
np.save(os.path.join(path_loco, path_save, 'param_sym_hind_bs.npy'), param_sym_hind_bs)
np.save(os.path.join(path_loco, path_save, 'animal_order.npy'), animal_list_plot)

#plot symmetry baseline subtracted - mean animals
for p in range(np.shape(param_sym)[0]-2):
    fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
    mean_data = np.nanmean(param_sym_bs[p, :, :], axis=0)
    std_data = (np.nanstd(param_sym_bs[p, :, :], axis=0)/np.sqrt(len(animals)))
    rectangle = plt.Rectangle((stim_trials[0]-0.5, np.nanmin(mean_data-std_data)), 10, np.nanmax(mean_data+std_data)-np.nanmin(mean_data-std_data), fc='lightblue', alpha=0.3)
    plt.gca().add_patch(rectangle)
    plt.hlines(0, 1, Ntrials, colors='grey', linestyles='--')
    plt.plot(np.arange(1, Ntrials+1), mean_data, linewidth=2, marker='o', color='black')
    plt.fill_between(np.arange(1, Ntrials+1), mean_data-std_data, mean_data+std_data, color='black', alpha=0.5)
    ax.set_xlabel('Trial', fontsize=20)
    ax.set_ylabel(param_sym_label[p], fontsize=20)
    if p == 2:
        plt.gca().invert_yaxis()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(path_save, 'mean_animals_symmetry_' + param_sym_name[p] + '.png'), dpi=128)
plt.close('all')

for p in range(np.shape(param_sym)[0]-2):
    fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
    mean_data = np.nanmean(param_sym_hind_bs[p, :, :], axis=0)
    std_data = (np.nanstd(param_sym_hind_bs[p, :, :], axis=0)/np.sqrt(len(animals)))
    rectangle = plt.Rectangle((stim_trials[0]-0.5, np.nanmin(mean_data-std_data)), 10, np.nanmax(mean_data+std_data)-np.nanmin(mean_data-std_data), fc='lightblue', zorder=0, alpha=0.3)
    plt.gca().add_patch(rectangle)
    plt.hlines(0, 1, Ntrials, colors='grey', linestyles='--')
    plt.plot(np.arange(1, Ntrials+1), mean_data, linewidth=2, marker='o', color='black', zorder=-1)
    plt.fill_between(np.arange(1, Ntrials+1), mean_data-std_data, mean_data+std_data, color='black', alpha=0.5)
    ax.set_xlabel('Trial', fontsize=28)
    ax.set_ylabel(param_sym_label[p], fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(path_save, 'mean_animals_symmetry_hind_' + param_sym_name[p] + '.png'), dpi=128)
    plt.savefig(os.path.join(path_save, 'mean_animals_symmetry_hind_' + param_sym_name[p] + '.svg'), dpi=128)
plt.close('all')

#plot individual limbs baseline subtracted - mean animals
for p in range(np.shape(param_sym)[0]-2):
    fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
    mean_data = np.vstack((np.nanmean(param_paw_bs[p, :, 0, :], axis=0), np.nanmean(param_paw_bs[p, :, 1, :], axis=0),
        np.nanmean(param_paw_bs[p, :, 2, :], axis=0), np.nanmean(param_paw_bs[p, :, 3, :], axis=0)))
    std_data = np.vstack((np.nanstd(param_paw_bs[p, :, 0, :], axis=0)/np.sqrt(len(animals)),
        np.nanstd(param_paw_bs[p, :, 1, :], axis=0)/np.sqrt(len(animals)),
        np.nanstd(param_paw_bs[p, :, 2, :], axis=0)/np.sqrt(len(animals)),
        np.nanstd(param_paw_bs[p, :, 3, :], axis=0)/np.sqrt(len(animals))))
    rectangle = plt.Rectangle((stim_trials[0]-0.5, np.nanmin(mean_data-std_data)), 10, np.nanmax(mean_data+std_data)-np.nanmin(mean_data-std_data), fc='lightblue', alpha=0.3)
    plt.gca().add_patch(rectangle)
    plt.hlines(0, 1, Ntrials, colors='grey', linestyles='--')
    for paw in range(4):
        plt.plot(np.arange(1, Ntrials+1), np.nanmean(param_paw_bs[p, :, paw, :], axis=0), linewidth=2, color=paw_colors[paw])
        plt.fill_between(np.arange(1, Ntrials+1),
            np.nanmean(param_paw_bs[p, :, paw, :], axis=0)-(np.nanstd(param_paw_bs[p, :, paw, :], axis=0)/np.sqrt(len(animals))),
            np.nanmean(param_paw_bs[p, :, paw, :], axis=0)+(np.nanstd(param_paw_bs[p, :, paw, :], axis=0)/np.sqrt(len(animals))), color=paw_colors[paw], alpha=0.5)
    ax.set_xlabel('Trial', fontsize=20)
    ax.set_ylabel(param_label[p], fontsize=20)
    if p == 2:
        plt.gca().invert_yaxis()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(path_save, 'mean_animals_paws_' + param_sym_name[p] + '.png'), dpi=128)
plt.close('all')

#plot stance phase - group mean polar plot
#error bars in polar plot don't rotate well
fig = plt.figure(figsize=(10, 10), tight_layout=True)
ax = fig.add_subplot(111, projection='polar')
for paw in range(4):
    data_mean = sp.circmean(param_phase[paw, :, :], axis=0)
    ax.scatter(data_mean, np.arange(1, Ntrials + 1), c=paw_colors[paw], s=30)
ax.set_yticks([8.5, 18.5])
ax.set_yticklabels(['', ''])
ax.tick_params(axis='both', which='major', labelsize=20)
plt.savefig(os.path.join(path_save, 'mean_animals_stance_phase_polar.png'), dpi=256)
plt.savefig(os.path.join(path_save, 'mean_animals_stance_phase_polar.svg'), dpi=256)

#plot stance phase - group mean
fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
mean_data = np.vstack((np.nanmean(param_phase[0, :, :], axis=0), np.nanmean(param_phase[1, :, :], axis=0),
                       np.nanmean(param_phase[2, :, :], axis=0), np.nanmean(param_phase[3, :, :], axis=0)))
std_data = np.vstack((np.nanstd(param_phase[0, :, :], axis=0) / np.sqrt(len(animals)),
                      np.nanstd(param_phase[1, :, :], axis=0) / np.sqrt(len(animals)),
                      np.nanstd(param_phase[2, :, :], axis=0) / np.sqrt(len(animals)),
                      np.nanstd(param_phase[3, :, :], axis=0) / np.sqrt(len(animals))))
rectangle = plt.Rectangle((stim_trials[0] - 0.5, np.rad2deg(np.nanmin(mean_data - std_data))), 10,
                          np.rad2deg(np.nanmax(mean_data + std_data) - np.nanmin(mean_data - std_data)), fc='lightblue', alpha=0.3)
plt.gca().add_patch(rectangle)
for paw in range(4):
    plt.plot(np.arange(1, Ntrials + 1), np.rad2deg(np.nanmean(param_phase[paw, :, :], axis=0)), linewidth=2,
             color=paw_colors[paw])
    plt.fill_between(np.arange(1, Ntrials + 1),
                     np.rad2deg(np.nanmean(param_phase[paw, :, :], axis=0) - (
                                 np.nanstd(param_phase[paw, :, :], axis=0) / np.sqrt(len(animals)))),
                     np.rad2deg(np.nanmean(param_phase[paw, :, :], axis=0) + (
                                 np.nanstd(param_phase[paw, :, :], axis=0) / np.sqrt(len(animals)))),
                     color=paw_colors[paw], alpha=0.5)
ax.set_xlabel('Trial', fontsize=20)
ax.set_ylabel(phase_label, fontsize=20)
ax.set_ylim([70, 230])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path_save, 'mean_animals_stance_phase.png'), dpi=128)
plt.close('all')

#plot stance speed - individual animals
fig, ax = plt.subplots(3, 3, figsize=(20, 20), tight_layout=True, sharey=True, sharex=True)
ax = ax.ravel()
for count_a in range(len(animals)):
    for p in range(4):
        ax[count_a].axvline(x=stim_trials[0]-0.5, color='dimgray', linestyle='--')
        ax[count_a].axvline(x=stim_trials[0]+10-0.5, color='dimgray', linestyle='--')
        ax[count_a].plot(np.arange(1, Ntrials+1), stance_speed[p, count_a, :], color=paw_colors[p], linewidth=2)
        ax[count_a].axvline(x=8.5, color='black')
        ax[count_a].axvline(x=18.5, color='black')
        ax[count_a].spines['right'].set_visible(False)
        ax[count_a].spines['top'].set_visible(False)
        ax[count_a].tick_params(axis='x')
        ax[count_a].tick_params(axis='y')
        ax[count_a].set_title(animals[count_a])
plt.savefig(os.path.join(path_save, 'ind_animals_stance_speed.png'), dpi=128)
