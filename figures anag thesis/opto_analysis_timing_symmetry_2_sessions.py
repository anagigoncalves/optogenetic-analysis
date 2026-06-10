import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy

#path inputs
path_st = 'J:\\Opto JAWS Data\\tied stance stim\\'
path_sw = 'J:\\Opto JAWS Data\\tied swing stim\\'
experiment_type = 'tied'
save_path = 'J:\\Thesis\\for figures\\fig tied opto\\'
experiment_st = path_st.split('\\')[-2].replace(' ', '_')
experiment_sw = path_sw.split('\\')[-2].replace(' ', '_')
param_sym_name = ['coo', 'step_length', 'double_support', 'coo_stance', 'coo_swing', 'swing_length']
param_sym_label = ['Center of oscillation\nsymmetry (mm)', 'Step length\nsymmetry (mm)',
    'Percentage of double\nsupport symmetry', 'Spatial motor output\nsymmetry (mm)',
        'Temporal motor output\nsymmetry (mm)', 'Swing length\nsymmetry (mm)']
paws = ['FR', 'HR', 'FL', 'HL']
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']

if experiment_type == 'split':
    stim_trials = np.arange(9, 19)
    Ntrials = 28
    rec_size = 10
if experiment_type == 'tied':
    stim_trials = np.arange(9, 17)
    Ntrials = 24
    rec_size = 8

param_sym_bs_st = np.load(
    path_st + '\\grouped output\\param_sym_bs.npy')
param_sym_bs_sw = np.load(
    path_sw + '\\grouped output\\param_sym_bs.npy')
param_paw_bs_st = np.load(
    path_st + '\\grouped output\\param_paw_bs.npy')
param_paw_bs_sw = np.load(
    path_sw + '\\grouped output\\param_paw_bs.npy')
param_phase_st = np.load(
    path_st + '\\grouped output\\param_phase.npy')
param_phase_sw = np.load(
    path_sw + '\\grouped output\\param_phase.npy')
if experiment_type == 'tied':
    animals_st = ['MC19214', 'MC19130', 'MC17665', 'MC19082', 'MC17670', 'MC17666']
    animals_sw = ['MC19130', 'MC19082', 'MC17670', 'MC17666', 'MC17665', 'MC19214']
    stim_trial_end_st = np.repeat(16, len(animals_st))
    stim_trial_end_sw = np.repeat(16, len(animals_sw))
else:
    animals_st = ['MC16851', 'MC17319', 'MC17665', 'MC17670', 'MC19082', 'MC19124', 'MC19130', 'MC19214', 'MC19107']
    animals_sw = ['MC16851', 'MC17319', 'MC17665', 'MC17670', 'MC19082', 'MC19124', 'MC19130', 'MC19214', 'MC19107']
    stim_trial_end_st = np.repeat(18, len(animals_st))
    stim_trial_end_sw = np.repeat(18, len(animals_sw))
Nanimals = len(animals_st)

# min_plot = [-3, -4, -6, -4, -4, -4] #tied
# max_plot = [3, 4, 6, 4, 4, 4] #tied
# # min_plot = [-5, -7, -7, -2, -2, -3] #split right fast
# # max_plot = [2, 4, 11, 9, 9, 8] #split right fast
# # min_plot = [-2, -3, -11, -6, -6, -9] #split left fast
# # max_plot = [4, 6, 8, 2, 2, 3] #split left fast
# for p in range(np.shape(param_sym_name)[0]):
#     mean_data_st = np.nanmean(param_sym_bs_st[p, :, :], axis=0)
#     std_data_st = np.nanstd(param_sym_bs_st[p, :, :], axis=0) / np.sqrt(Nanimals)
#     mean_data_sw = np.nanmean(param_sym_bs_sw[p, :, :], axis=0)
#     std_data_sw = np.nanstd(param_sym_bs_sw[p, :, :], axis=0) / np.sqrt(Nanimals)
#     # mean_data_control = np.nanmean(param_sym_bs_control[p, :, :], axis=0)
#     # std_data_control = np.nanstd(param_sym_bs_control[p, :, :], axis=0)
#     fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
#     rectangle = plt.Rectangle((stim_trials[0]-0.5, min_plot[p]), rec_size, max_plot[p]-min_plot[p], fc='lightblue', zorder=0, alpha=0.3)
#     plt.gca().add_patch(rectangle)
#     plt.hlines(0, 1, Ntrials, colors='grey', linestyles='--')
#     # plt.plot(np.arange(1, Ntrials+1), mean_data_control, linewidth=2, marker='o', color='black')
#     # plt.fill_between(np.arange(1, Ntrials+1), mean_data_control-std_data_control, mean_data_control+std_data_control, color='black', alpha=0.5)
#     plt.plot(np.arange(1, Ntrials+1), mean_data_st, linewidth=2, marker='o', color='orange')
#     plt.fill_between(np.arange(1, Ntrials+1), mean_data_st-std_data_st, mean_data_st+std_data_st, color='orange', alpha=0.5)
#     plt.plot(np.arange(1, Ntrials+1), mean_data_sw, linewidth=2, marker='o', color='green')
#     plt.fill_between(np.arange(1, Ntrials+1), mean_data_sw-std_data_sw, mean_data_sw+std_data_sw, color='green', alpha=0.5)
#     ax.legend(['stance \nstim.', 'swing \nstim.', '', ''], frameon=False, fontsize=24)
#     ax.set_xlabel('Trial', fontsize=28)
#     ax.set_ylabel(param_sym_label[p], fontsize=28)
#     # if p == 2:
#     #     plt.gca().invert_yaxis()
#     plt.xticks(fontsize=28)
#     plt.yticks(fontsize=28)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.savefig(os.path.join(save_path, experiment_type + '_mean_animals_sym' + param_sym_name[p] + '.png'))
#     plt.savefig(os.path.join(save_path, experiment_type + '_mean_animals_sym' + param_sym_name[p] + '.svg'))
# plt.close('all')

# #For thesis presentation plots
# param_sym_bs_control = np.load('J:\\Opto JAWS Data\\split right fast control\\grouped output\\param_sym_bs.npy')
# fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
# mean_data_control = np.nanmean(param_sym_bs_control[2, :, :], axis=0)
# std_data_control = (np.nanstd(param_sym_bs_control[2, :, :], axis=0) / np.sqrt(Nanimals))
# mean_data_st = np.nanmean(param_sym_bs_st[2, :, :], axis=0)
# std_data_st = (np.nanstd(param_sym_bs_st[2, :, :], axis=0) / np.sqrt(Nanimals))
# mean_data_sw = np.nanmean(param_sym_bs_sw[2, :, :], axis=0)
# std_data_sw = (np.nanstd(param_sym_bs_sw[2, :, :], axis=0) / np.sqrt(Nanimals))
# # plt.vlines(8.5, -6, 10, colors='grey', linestyle='--')
# # plt.vlines(18.5, -6, 10, colors='grey', linestyle='--')
# rectangle = plt.Rectangle((stim_trials[0] - 0.5, -6), rec_size, 16, fc='lightblue',
#                           zorder=0, alpha=0.3)
# plt.gca().add_patch(rectangle)
# plt.hlines(0, 1, Ntrials, colors='grey', linestyles='--')
# plt.plot(np.arange(1, Ntrials + 1), mean_data_control, linewidth=2, marker='o', color='black')
# plt.fill_between(np.arange(1, Ntrials + 1), mean_data_control - std_data_control, mean_data_control + std_data_control, color='black', alpha=0.5)
# # plt.plot(np.arange(1, Ntrials + 1), mean_data_st, linewidth=2, marker='o', color='orange')
# # plt.fill_between(np.arange(1, Ntrials + 1), mean_data_st - std_data_st, mean_data_st + std_data_st, color='orange', alpha=0.5)
# plt.plot(np.arange(1, Ntrials + 1), mean_data_sw, linewidth=2, marker='o', color='green')
# plt.fill_between(np.arange(1, Ntrials + 1), mean_data_sw - std_data_sw, mean_data_sw + std_data_sw, color='green', alpha=0.5)
# ax.set_xlabel('Trial', fontsize=20)
# ax.set_ylabel('Temporal asymmetry\n(% stride cycle)', fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.ylim([-6, 10])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig(os.path.join('J:\\Thesis\\Presentation\\split_right_fast_control_sw.png'), dpi=256)
fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
mean_data_st = np.nanmean(param_sym_bs_st[2, :, :], axis=0)
std_data_st = (np.nanstd(param_sym_bs_st[2, :, :], axis=0) / np.sqrt(Nanimals))
mean_data_sw = np.nanmean(param_sym_bs_sw[2, :, :], axis=0)
std_data_sw = (np.nanstd(param_sym_bs_sw[2, :, :], axis=0) / np.sqrt(Nanimals))
rectangle = plt.Rectangle((stim_trials[0] - 0.5, -6), rec_size, 12, fc='lightblue',
                          zorder=0, alpha=0.3)
plt.gca().add_patch(rectangle)
plt.hlines(0, 1, Ntrials, colors='grey', linestyles='--')
plt.plot(np.arange(1, Ntrials + 1), mean_data_st, linewidth=2, marker='o', color='orange')
plt.fill_between(np.arange(1, Ntrials + 1), mean_data_st - std_data_st, mean_data_st + std_data_st, color='orange', alpha=0.5)
# plt.plot(np.arange(1, Ntrials + 1), mean_data_sw, linewidth=2, marker='o', color='green')
# plt.fill_between(np.arange(1, Ntrials + 1), mean_data_sw - std_data_sw, mean_data_sw + std_data_sw, color='green', alpha=0.5)
ax.set_xlabel('Trial', fontsize=20)
ax.set_ylabel('Temporal asymmetry\n(% stride cycle)', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim([-6, 6])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join('J:\\Thesis\\Presentation\\tied_stim_st.png'), dpi=256)

# # Quantifications - seaborn
# #TODO delta split is the difference between mean of the last 2 stim trials and the 2 last baseline (not beginning of stim)
# group_id = []
# group_name = []
# ae_amp = []
# delta_split = []
# param = []
# for count_p, p in enumerate(param_sym_name):
#     group_st_len = np.shape(param_sym_bs_st)[1]
#     group_id.extend(np.repeat(0, group_st_len))
#     ae_st_data = np.vstack((param_sym_bs_st[count_p, :, stim_trials[-1]], param_sym_bs_st[count_p, :, stim_trials[-1]+1]))
#     ae_amp.extend(np.nanmean(ae_st_data, axis=0))
#     for a in range(param_sym_bs_st.shape[1]):
#         delta_split.append(param_sym_bs_st[count_p, a, stim_trial_end_st[a]-1]-param_sym_bs_st[count_p, a, stim_trials[0]-1])
#     param.extend(np.repeat(p, group_st_len))
#     group_sw_len = np.shape(param_sym_bs_sw)[1]
#     group_id.extend(np.repeat(1, group_sw_len))
#     ae_sw_data = np.vstack((param_sym_bs_sw[count_p, :, stim_trials[-1]], param_sym_bs_sw[count_p, :, stim_trials[-1]+1]))
#     ae_amp.extend(np.nanmean(ae_sw_data, axis=0))
#     for a in range(param_sym_bs_sw.shape[1]):
#         delta_split.append(param_sym_bs_sw[count_p, a, stim_trial_end_sw[a]-1]-param_sym_bs_sw[count_p, a, stim_trials[0]-1])
#     param.extend(np.repeat(p, group_sw_len))
# split_quant_df = pd.DataFrame({'param': param, 'group': group_id, 'after-effect': ae_amp, 'delta-split': delta_split})
# param_sym_label_ae = ['Center of oscillation\nafter-effect symmetry (mm)', 'Step length\nafter-effect symmetry(mm)', 'Percentage of double support\nafter-effect symmetry', 'Center of oscillation\n stance after-effect symmetry (mm)',
#         'Center of oscillation\n swing after-effect symmetry (mm)', 'Swing length\nafter-effect symmetry (mm)']
# colors_boxplot = ['orange', 'green']
# for p in range(np.shape(param_sym_name)[0]):
#     fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
#     h = sns.boxplot(x='group', y='after-effect', data=split_quant_df.loc[split_quant_df['param'] == param_sym_name[p]],
#               showmeans=True, meanline=True, ax=ax, medianprops={'visible': False}, meanprops={'ls': '-', 'lw': 3},
#                 whiskerprops={'visible': False}, showfliers=False, showbox=False, showcaps=False, palette={0: 'orange', 1: 'green'})
#     sns.stripplot(x='group', y='after-effect', data=split_quant_df.loc[split_quant_df['param'] == param_sym_name[p]],
#               dodge=True, s=10, ax=ax, palette={0: 'orange', 1: 'green'}, edgecolor='None')
#     ax.set_xticklabels(['Stance', 'Swing'])
#     ax.set_xlabel('')
#     ax.set_ylabel(param_sym_label_ae[p], fontsize=16)
#     ax.tick_params(axis='both', which='major', labelsize=16)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.savefig(save_path + param_sym_name[p] + 'after_effect_quantification', dpi=256)
#     plt.savefig(save_path + param_sym_name[p] + 'after_effect_quantification.svg', dpi=256)
#     ###### STATS #######
#     # Mann-Whitney on param means
#     print('after-effect')
#     print(param_sym_name[p])
#     data_stats = split_quant_df.loc[split_quant_df['param'] == param_sym_name[p]]
#     stats_mannwhitney_ae = scipy.stats.mannwhitneyu(data_stats.loc[data_stats['group']==0, 'after-effect'], data_stats.loc[data_stats['group']==1, 'after-effect'], method='exact')
#     print(stats_mannwhitney_ae)
# # param_sym_label_delta = ['Change over split of\ncenter of oscillation symmetry (mm)', 'Change over split of\nstep length symmetry (mm)', 'Change over split of\npercentage of double support symmetry', 'Change over split of\ncenter of oscillation stance symmetry (mm)',
# #         'Change over split of\nswing length symmetry (mm)']
# param_sym_label_delta = ['Change over stim. of\ncenter of oscillation symmetry (mm)', 'Change over stim. of\nstep length symmetry (mm)', 'Change over stim. of\npercentage of double support symmetry', 'Change over stim. of\ncenter of oscillation stance symmetry (mm)',
#         'Center of oscillation\n swing after-effect symmetry (mm)', 'Change over stim. of\nswing length symmetry (mm)']
# for p in range(np.shape(param_sym_name)[0]):
#     fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
#     h = sns.boxplot(x='group', y='delta-split', data=split_quant_df.loc[split_quant_df['param'] == param_sym_name[p]],
#               showmeans=True, meanline=True, ax=ax, medianprops={'visible': False}, meanprops={'ls': '-', 'lw': 3},
#                 whiskerprops={'visible': False}, showfliers=False, showbox=False, showcaps=False, palette={0: 'orange', 1: 'green'})
#     sns.stripplot(x='group', y='delta-split', data=split_quant_df.loc[split_quant_df['param'] == param_sym_name[p]],
#                   dodge=True, s=10, ax=ax, palette={0: 'orange', 1: 'green'}, edgecolor='None')
#     ax.set_xticklabels(['Stance', 'Swing'])
#     ax.set_xlabel('')
#     ax.set_ylabel(param_sym_label_delta[p], fontsize=16)
#     ax.tick_params(axis='both', which='major', labelsize=16)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.savefig(save_path + param_sym_name[p] + 'delta_stim_quantification', dpi=256)
#     plt.savefig(save_path + param_sym_name[p] + 'delta_stim_quantification.svg', dpi=256)
#     ###### STATS #######
#     # Mann-Whitney on param means
#     print('delta-split')
#     print(param_sym_name[p])
#     data_stats = split_quant_df.loc[split_quant_df['param'] == param_sym_name[p]]
#     stats_mannwhitney_ds = scipy.stats.mannwhitneyu(data_stats.loc[data_stats['group']==0, 'delta-split'], data_stats.loc[data_stats['group']==1, 'delta-split'], method='exact')
#     print(stats_mannwhitney_ds)
# plt.close('all')
#
# # Individual limbs - st
# for p in range(np.shape(param_sym_name)[0]):
#     fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
#     rectangle = plt.Rectangle((stim_trials[0] - 0.5, min_plot[p]), rec_size, max_plot[p] - min_plot[p], fc='lightgray',
#                               zorder=0, alpha=0.3)
#     plt.gca().add_patch(rectangle)
#     for paw in range(4):
#         param_paw_st_bs_mean = np.nanmean(param_paw_bs_st[p, :, paw, :], axis=0)
#         param_paw_st_bs_std = np.nanstd(param_paw_bs_st[p, :, paw, :], axis=0) / np.sqrt(
#             np.shape(param_paw_bs_st)[1])
#         plt.plot(np.arange(1, Ntrials+1), param_paw_st_bs_mean, linewidth=2, marker='o', color=paw_colors[paw])
#         plt.fill_between(np.arange(1, Ntrials+1), param_paw_st_bs_mean-param_paw_st_bs_std,
#                     param_paw_st_bs_mean+param_paw_st_bs_std, color=paw_colors[paw], alpha=0.5)
#     ax.set_xlabel('Trial', fontsize=20)
#     ax.set_ylabel(param_sym_label[p], fontsize=20)
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.savefig(os.path.join(save_path, 'mean_animals_paws_bs_' + param_sym_name[p] + '_st.png'), dpi=128)
#     plt.savefig(os.path.join(save_path, 'mean_animals_paws_bs_' + param_sym_name[p] + '_st.svg'), dpi=128)
#
# # Individual limbs - sw
# for p in range(np.shape(param_sym_name)[0]):
#     fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
#     rectangle = plt.Rectangle((stim_trials[0] - 0.5, min_plot[p]), rec_size, max_plot[p] - min_plot[p], fc='lightgray',
#                               zorder=0, alpha=0.3)
#     plt.gca().add_patch(rectangle)
#     for paw in range(4):
#         param_paw_sw_bs_mean = np.nanmean(param_paw_bs_sw[p, :, paw, :], axis=0)
#         param_paw_sw_bs_std = np.nanstd(param_paw_bs_sw[p, :, paw, :], axis=0) / np.sqrt(
#             np.shape(param_paw_bs_sw)[1])
#         plt.plot(np.arange(1, Ntrials+1), param_paw_sw_bs_mean, linewidth=2, marker='o', color=paw_colors[paw])
#         plt.fill_between(np.arange(1, Ntrials+1), param_paw_sw_bs_mean-param_paw_st_bs_std,
#                     param_paw_sw_bs_mean+param_paw_st_bs_std, color=paw_colors[paw], alpha=0.5)
#     ax.set_xlabel('Trial', fontsize=20)
#     ax.set_ylabel(param_sym_label[p], fontsize=20)
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.savefig(os.path.join(save_path, 'mean_animals_paws_bs_' + param_sym_name[p] + '_sw.png'), dpi=128)
#     plt.savefig(os.path.join(save_path, 'mean_animals_paws_bs_' + param_sym_name[p] + '_sw.svg'), dpi=128)
#
# # Stance phase - st
# fig = plt.figure(figsize=(10, 10), tight_layout=True)
# ax = fig.add_subplot(111, projection='polar')
# for paw in range(3):
#     data_mean = scipy.stats.circmean(param_phase_st[paw, :, :], axis=0, nan_policy='omit')
#     ax.scatter(data_mean[~np.isnan(data_mean)], np.arange(1, Ntrials+1), c=paw_colors[paw], s=30)
# ax.set_yticks([8.5, 8.5+rec_size])
# ax.tick_params(axis='both', which='major', labelsize=28)
# plt.savefig(os.path.join(save_path, 'mean_animals_stance_phase_polar_st.png'), dpi=256)
# plt.savefig(os.path.join(save_path, 'mean_animals_stance_phase_polar_st.svg'), dpi=256)
#
# # Stance phase - sw
# fig = plt.figure(figsize=(10, 10), tight_layout=True)
# ax = fig.add_subplot(111, projection='polar')
# for paw in range(3):
#     data_mean = scipy.stats.circmean(param_phase_sw[paw, :, :], axis=0, nan_policy='omit')
#     ax.scatter(data_mean[~np.isnan(data_mean)], np.arange(1, Ntrials+1), c=paw_colors[paw], s=30)
# ax.set_yticks([8.5, 8.5+rec_size])
# ax.tick_params(axis='both', which='major', labelsize=28)
# plt.savefig(os.path.join(save_path, 'mean_animals_stance_phase_polar_sw.png'), dpi=256)
# plt.savefig(os.path.join(save_path, 'mean_animals_stance_phase_polar_sw.svg'), dpi=256)
#
#
#
