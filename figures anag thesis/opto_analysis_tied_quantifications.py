import os
import matplotlib.pyplot as plt
import numpy as np

#path inputs
path_st = 'J:\\Opto JAWS Data\\split left fast stance stim\\'
path_sw = 'J:\\Opto JAWS Data\\split left fast swing stim\\'
experiment_type = 'split'
save_path = 'J:\\Thesis\\for figures\\fig split left fast opto\\'
experiment_st = path_st.split('\\')[-2].replace(' ', '_')
experiment_sw = path_sw.split('\\')[-2].replace(' ', '_')
param_sym_name = ['coo', 'step_length', 'double_support', 'coo_stance', 'coo_swing', 'swing_length']
param_sym_label = ['Center of oscillation\nsymmetry (mm)', 'Step length\nsymmetry (mm)',
    'Percentage of double\nsupport symmetry', 'Spatial motor output\nsymmetry (mm)',
        'Temporal motor output\nsymmetry (mm)', 'Swing length\nsymmetry (mm)']
paws = ['FR', 'HR', 'FL', 'HL']
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']

stim_trials = np.arange(9, 19)
Ntrials = 28
rec_size = 10

param_sym_bs_st = np.load(
    path_st + '\\grouped output\\param_sym_bs.npy')
animal_order_st = np.load(
    path_st + '\\grouped output\\animal_order.npy')
param_sym_bs_sw = np.load(
    path_sw + '\\grouped output\\param_sym_bs.npy')
animal_order_sw = np.load(
    path_sw + '\\grouped output\\animal_order.npy')

#Quantification after-effect and % over split DS
#after-effect
param_sym_label_ae = ['Center of oscillation\nafter-effect symmetry (mm)', 'Step length\nafter-effect symmetry(mm)', 'Percentage of double support\nafter-effect symmetry', 'Center of oscillation\n stance after-effect symmetry (mm)',
        'Center of oscillation\n swing after-effect symmetry (mm)', 'Swing length\nafter-effect symmetry (mm)']
param_name = ['coo', 'step_length', 'double_support', 'coo_stance', 'coo_swing', 'swing_length']
for p in np.array([0, 2]):
    param_sym_bs_ae_st = np.nanmean(param_sym_bs_st[p, :, [stim_trials[-1]+1, stim_trials[-1]]], axis=0)
    param_sym_bs_ae_sw = np.nanmean(param_sym_bs_sw[p, :, [stim_trials[-1]+1, stim_trials[-1]]], axis=0)
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    ax.scatter(np.repeat(1, len(param_sym_bs_ae_st)), param_sym_bs_ae_st, color='orange', s=40)
    ax.scatter(np.repeat(2, len(param_sym_bs_ae_sw)), param_sym_bs_ae_sw, color='green', s=40)
    ax.scatter(1, np.nanmean(param_sym_bs_ae_st), marker='_', color='orange', s=5000, linewidth=3)
    ax.scatter(2, np.nanmean(param_sym_bs_ae_sw), marker='_', color='green', s=5000, linewidth=3)
    for count_a1, a1 in enumerate(animal_order_st):
        for count_a2, a2 in enumerate(animal_order_sw):
            if a2 == a1:
                ax.plot([2, 1], [param_sym_bs_ae_sw[count_a2], param_sym_bs_ae_st[count_a1]], color='darkgray', linewidth=0.5)
    ax.set_xlim([0.6, 2.4])
    ax.set_xticks([1, 2])
    ax.hlines(0, 0.6, 2.4, colors='grey', linestyles='--')
    ax.set_xticklabels(['stance\nstim.', 'swing\nstim.'])
    ax.set_xlabel('')
    ax.set_ylabel(param_sym_label_ae[p], fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(save_path, param_name[p] + '_animals_symmetry_after_effect_quantification.png'), dpi=128)
    plt.savefig(os.path.join(save_path, param_name[p] + '_animals_symmetry_after_effect_quantification.svg'), dpi=128)
#delta split
param_sym_label_delta = ['Change over stim. of\ncenter of oscillation\nsymmetry (mm)', 'Change over stim. of\nstep length symmetry (mm)', 'Change over stim. of\npercentage of double support\nsymmetry', 'Change over stim. of\ncenter of oscillation stance symmetry (mm)',
        'Center of oscillation\n swing after-effect symmetry (mm)', 'Change over stim. of\nswing length symmetry (mm)']
for p in np.array([0, 2]):
    param_sym_bs_delta_st = -np.abs(
        np.nanmean(param_sym_bs_st[p, :, [stim_trials[-1] - 1, stim_trials[-1] - 2]], axis=0)) - np.abs(
        np.nanmean(param_sym_bs_st[p, :, [stim_trials[0] - 1, stim_trials[-1] - 2]], axis=0))
    param_sym_bs_delta_control = np.nanmean(param_sym_bs_sw[p, :, [stim_trials[-1] - 1, stim_trials[-1] - 2]],
                                            axis=0) - np.nanmean(
        param_sym_bs_sw[p, :, [stim_trials[0] - 1, stim_trials[-1] - 2]], axis=0)
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    ax.scatter(np.repeat(1, len(param_sym_bs_delta_st)), param_sym_bs_delta_st, color='orange', s=40)
    ax.scatter(np.repeat(2, len(param_sym_bs_delta_control)), param_sym_bs_delta_control, color='green', s=40)
    ax.scatter(1, np.nanmean(param_sym_bs_delta_st), marker='_', color='orange', s=5000, linewidth=3)
    ax.scatter(2, np.nanmean(param_sym_bs_delta_control), marker='_', color='green', s=5000, linewidth=3)
    for count_a1, a1 in enumerate(animal_order_st):
        for count_a2, a2 in enumerate(animal_order_sw):
            if a2 == a1:
                ax.plot([2, 1], [param_sym_bs_delta_control[count_a2], param_sym_bs_delta_st[count_a1]], color='darkgray', linewidth=0.5)
    ax.set_xticks([1, 2])
    ax.set_xlim([0.6, 2.4])
    ax.set_xticklabels(['stance\nstim.', 'swing\nstim.'])
    ax.set_xlabel('')
    ax.set_ylabel(param_sym_label_delta[p], fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(save_path, param_name[p] + '_animals_symmetry_delta_split_quantification.png'), dpi=128)
    plt.savefig(os.path.join(save_path, param_name[p] + '_animals_symmetry_delta_split_quantification.svg'), dpi=128)
