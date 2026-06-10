"""Created on Fri May 9 10:24:34 2025

@author: anagigoncalves and AliceGem

It compares the locomouse parameters for different experiments, after having computed them for each animal and each experiment with the "tied_behavior_analysis_single_animal.py" script.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotting_functions as pf
import tied_behavior_analysis_multi_session_utils as multi_session_utils
from matplotlib.lines import Line2D

paths = [
    'D:\\AliG\\climbing-opto-treadmill\\WT split-belt learning',
    'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\HISTO_CHECKED_ANIMALS_Linj\\split right fast and split left fast CL-Ali\\',
    'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\HISTO_CHECKED_ANIMALS_Rinj\\split right fast and split left fast CL-Ali\\',
    'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\HISTO_CHECKED_ANIMALS_RLinj\\split contra fast right and split ipsi fast left CL-Ali\\',
# 'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT\\LOW expression\\ALL_ANIMALS\\tied th100sw IO 50ms',
#'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT\\LOW expression\\ALL_ANIMALS\\tied th200st IO 50ms and tied th100sw IO 50ms',
]   

markers = ['o','*', 's', '^']           # '*', '^'
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']
phase_paws = ['FR', 'HR', 'FL', 'HL']
phase_paw_index = {paw: index for index, paw in enumerate(phase_paws)}
intralimb_paws = ['FR', 'FL', 'HR', 'HL']
paws = ['FR', 'FL']     #, 'HR', 'HL']  # Front right, front left, hind right, hind left
param_tied = ['stance_duration', 'swing_duration', 'cadence', 'swing_length']
speed_range = np.arange(0.1,0.4,0.05)
speed_bins = np.array([1, 2, 3])
speed_bin_side = 2
stride_count_source_param = 'stance_duration'
stride_count_source_paw = 'FR'
intralimb_ylim = {
    'stance_duration': (50, 300),
    'swing_duration': (50, 150),
    'cadence': (0, 7),
}

swing_velocity_ylim = (-0.4, 1)
swing_amplitude_ylim = (-1, 6)
support_ylim = {
    'diagonal_fl': (0, 50),
    'diagonal_fr': (0, 50),
    'homolateral': (0, 10),
    'three_paws': (0, 40),
}
support_plot_config = {
    'diagonal_fl': {'column': 2, 'title': 'diagonal FL'},
    'diagonal_fr': {'column': 3, 'title': 'diagonal FR'},
    'homolateral': {'column': 4, 'title': 'homolateral'},
    'three_paws': {'column': 1, 'title': '3 paws'},
}
phase_summary_plot_config = {
    'front_hind': {
        'title': 'front-hind phase',
        'pairs': {
            'right': {'first_paw': 'FR', 'second_paw': 'HR'},
            'left': {'first_paw': 'FL', 'second_paw': 'HL'},
        },
    },
    'left_right': {
        'title': 'left-right phase',
        'pairs': {
            'front': {'first_paw': 'FR', 'second_paw': 'FL'},
            'hind': {'first_paw': 'HR', 'second_paw': 'HL'},
        },
    },
}
phase_pair_styles = {
    'right': {'filled': True},
    'left': {'filled': False},
    'front': {'filled': True},
    'hind': {'filled': False},
}

lightseagreen_light = pf.lighten_color('lightseagreen', 0.55)
lightseagreen_dark = pf.darken_color('lightseagreen', 0.4)
cohort_colors = {
    'WT': 'gray',
    'Linj': lightseagreen_dark,
    'Rinj': lightseagreen_light,
    'RLinj': 'darkblue',
}

# For each folder path, we want to load the data and parameters, and plot them

# Initialize figures with different handlers
fig_intra = [plt.figure(num=i+1, figsize=(12, 10)) for i in range(len(param_tied))]            #Intralimb parameters
ax_intra = {
    'FR': [fig.add_subplot(221) for fig in fig_intra],
    'FL': [fig.add_subplot(222) for fig in fig_intra],
    'HR': [fig.add_subplot(223) for fig in fig_intra],
    'HL': [fig.add_subplot(224) for fig in fig_intra],
}
for fig in fig_intra:
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
for paw in intralimb_paws:
    for ax in ax_intra[paw]:
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(f'{paw} paw', fontsize=20, pad=12)

fig_phase_st = plt.figure(num=len(param_tied)+1, figsize=(5 * len(paths), 5))            # Phase stance
ax_phase_st = [fig_phase_st.add_subplot(1, len(paths), count_p + 1, projection='polar') for count_p in range(len(paths))]
fig_phase_st.subplots_adjust(bottom=0.18, top=0.92, wspace=0.45)
for path, phase_ax in zip(paths, ax_phase_st):
    multi_session_utils.configure_phase_polar_axis(
        phase_ax,
        speed_range[:-1],
        multi_session_utils.get_path_label(path, cohort_colors),
    )

fig_phase_st_summary = plt.figure(num=len(param_tied)+2, figsize=(8, 5))
ax_phase_st_summary = fig_phase_st_summary.add_subplot(111, projection='polar')
fig_phase_st_summary.subplots_adjust(left=0.06, right=0.7, bottom=0.08, top=0.96)
multi_session_utils.configure_phase_polar_axis(ax_phase_st_summary, speed_range[:-1])

fig_phase_summary = plt.figure(num=len(param_tied)+3, figsize=(12, 5), tight_layout=True)
ax_phase_summary = {
    'front_hind': fig_phase_summary.add_subplot(121),
    'left_right': fig_phase_summary.add_subplot(122),
}
for phase_name, ax in ax_phase_summary.items():
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('speed (m/s)', fontsize=16)
    ax.set_ylabel(f"{phase_summary_plot_config[phase_name]['title']} (%)", fontsize=16)
    ax.set_ylim(0, 100)

fig_swing_vel = plt.figure(num=len(param_tied)+4, figsize=(12, 10))            # Swing velocity
ax_swing_vel = {
    'FR': fig_swing_vel.add_subplot(221),
    'FL': fig_swing_vel.add_subplot(222),
    'HR': fig_swing_vel.add_subplot(223),
    'HL': fig_swing_vel.add_subplot(224),
}
fig_swing_vel.subplots_adjust(hspace=0.4, wspace=0.3)
for paw in intralimb_paws:
    ax_swing_vel[paw].set_xlabel('% swing (norm)', fontsize=16)
    ax_swing_vel[paw].set_ylabel('Swing instantaneous\nvelocity (m/s)', fontsize=16)
    ax_swing_vel[paw].tick_params(axis='both', which='major', labelsize=14)
    ax_swing_vel[paw].spines['right'].set_visible(False)
    ax_swing_vel[paw].spines['top'].set_visible(False)
    ax_swing_vel[paw].set_title(f'{paw} paw', fontsize=20, pad=12)
    ax_swing_vel[paw].set_ylim(swing_velocity_ylim)

fig_swing_z = plt.figure(num=len(param_tied)+5, figsize=(12, 10))            # Swing amplitude
ax_swing_z = {
    'FR': fig_swing_z.add_subplot(221),
    'FL': fig_swing_z.add_subplot(222),
    'HR': fig_swing_z.add_subplot(223),
    'HL': fig_swing_z.add_subplot(224),
}
fig_swing_z.subplots_adjust(hspace=0.4, wspace=0.3)
for paw in intralimb_paws:
    ax_swing_z[paw].set_xlabel('% swing (norm)', fontsize=16)
    ax_swing_z[paw].set_ylabel('swing amplitude (mm)', fontsize=16)
    ax_swing_z[paw].tick_params(axis='both', which='major', labelsize=14)
    ax_swing_z[paw].spines['right'].set_visible(False)
    ax_swing_z[paw].spines['top'].set_visible(False)
    ax_swing_z[paw].set_title(f'{paw} paw', fontsize=20, pad=12)
    ax_swing_z[paw].set_ylim(swing_amplitude_ylim)

fig_swing_z_peak = plt.figure(num=len(param_tied)+6, figsize=(10, 8))            # Peak swing amplitude
ax_swing_z_peak = {
    'FR': fig_swing_z_peak.add_subplot(221),
    'FL': fig_swing_z_peak.add_subplot(222),
    'HR': fig_swing_z_peak.add_subplot(223),
    'HL': fig_swing_z_peak.add_subplot(224),
}
fig_swing_z_peak.subplots_adjust(hspace=0.45, wspace=0.35)
for paw in intralimb_paws:
    ax_swing_z_peak[paw].tick_params(axis='both', which='major', labelsize=14)
    ax_swing_z_peak[paw].spines['right'].set_visible(False)
    ax_swing_z_peak[paw].spines['top'].set_visible(False)
    ax_swing_z_peak[paw].set_title(f'{paw} paw', fontsize=20, pad=12)
    ax_swing_z_peak[paw].set_ylabel('Peak swing amplitude (mm)', fontsize=16)

peak_swing_amplitude_by_path = {paw: [] for paw in intralimb_paws}

fig_bos = plt.figure(num=len(param_tied)+7, figsize=(6, 5), tight_layout=True)            # Base of support
ax_bos = fig_bos.add_subplot(111)

fig_supports = plt.figure(num=len(param_tied)+8, figsize=(20, 5), tight_layout=True)            # Supports by category
ax_supports = {
    support_name: fig_supports.add_subplot(1, len(support_plot_config), count_s + 1)
    for count_s, support_name in enumerate(support_plot_config)
}
for support_name, ax in ax_supports.items():
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('speed (m/s)', fontsize=16)
    ax.set_ylabel('% of support', fontsize=16)
    ax.set_ylim(support_ylim[support_name])

fig_stride_count = plt.figure(num=len(param_tied)+9, figsize=(7, 5), tight_layout=True)
ax_stride_count = fig_stride_count.add_subplot(111)
ax_stride_count.set_xlabel('speed (m/s)', fontsize=16)
ax_stride_count.set_ylabel('stride density', fontsize=16)
ax_stride_count.tick_params(axis='both', which='major', labelsize=14)
ax_stride_count.spines['right'].set_visible(False)
ax_stride_count.spines['top'].set_visible(False)

path_index = 0

# LOAD the DATA
for path in paths:
    current_color = multi_session_utils.get_path_color(path, cohort_colors)
    current_label = multi_session_utils.get_path_label(path, cohort_colors)
    gait_parameters_dir = multi_session_utils.get_gait_parameters_dir(path)
    folders_animals, animal_session_folders = multi_session_utils.get_animal_session_folders(gait_parameters_dir)
    animal_ids = list(animal_session_folders.keys())
    print("folders_animals: ", folders_animals)
    print("animals: ", animal_ids)
    # Intralimb parameters for FR and FL paws
    param_tied_current_values = {}
    param_tied_current_speed = {}
    full_speed_values = speed_range[:-1]
    stride_count_current = np.zeros(len(full_speed_values), dtype=float)
    for paw in intralimb_paws:
        param_tied_current_values[paw] = np.full((len(param_tied), len(animal_ids), len(speed_range)-1), np.nan)
        param_tied_current_speed[paw] = np.tile(full_speed_values, (len(param_tied), len(animal_ids), 1))
    for count_p, g in enumerate(param_tied):
        for count_a, animal_id in enumerate(animal_ids):
            session_paths = animal_session_folders[animal_id]
            for paw in intralimb_paws:
                param_tied_current_file = multi_session_utils.concatenate_session_rows(session_paths, param_tied[count_p] + '_'+paw+'.npy')
                param_tied_current_df = pd.DataFrame({'values': param_tied_current_file[:, 0], 'speed': param_tied_current_file[:, 1]})
                param_tied_current_mean = param_tied_current_df.groupby(['speed'])['values'].mean().reindex(full_speed_values)
                param_tied_current_values[paw][count_p, count_a, :] = param_tied_current_mean.to_numpy()
            stride_count_current += multi_session_utils.count_session_rows_by_speed(
                session_paths,
                stride_count_source_param + '_' + stride_count_source_paw + '.npy',
                full_speed_values,
            )
    # Intralimb parameters PLOT
    ylabel = ['Stance duration (ms)', 'Swing duration (ms)', 'Cadence ($\mathregular{ms^{-1}}$)', 'Swing length (mm)']
    for paw in intralimb_paws:
        for count_p, g in enumerate(param_tied):
            if ylabel[count_p] == 'Cadence ($\mathregular{ms^{-1}}$)':
                scale = 1000  # Convert cadence to s^-1
            else:
                scale = 1  # No conversion needed for other parameters
            line_label = current_label if paw == 'FR' else None
            ax_intra[paw][count_p].plot(param_tied_current_speed[paw][0, 0, :], scale*np.nanmean(param_tied_current_values[paw][count_p, :, :], axis=0), color=current_color, linewidth=2, label=line_label)
            ax_intra[paw][count_p].fill_between(param_tied_current_speed[paw][0, 0, :],
                            scale*np.nanmean(param_tied_current_values[paw][count_p, :, :], axis=0)-scale*np.nanstd(param_tied_current_values[paw][count_p, :, :], axis=0),
            scale*np.nanmean(param_tied_current_values[paw][count_p, :, :], axis=0)+scale*np.nanstd(param_tied_current_values[paw][count_p, :, :], axis=0), color=current_color, alpha=0.3)
            ax_intra[paw][count_p].set_xlabel('Speed (m/s)', fontsize=16)
            # ax.set_title(g.replace('_', ' ') + ' FR paw', fontsize=20)
            ax_intra[paw][count_p].set_ylabel(ylabel[count_p], fontsize=16)
            if g in intralimb_ylim:
                ax_intra[paw][count_p].set_ylim(intralimb_ylim[g])

    
    # Stance phase and trajectories
    phase_bin_paw_current_mat = np.full((len(animal_ids), 4, len(speed_range)-1), np.nan)
    swing_z_current_mat = {}
    swing_inst_vel_current_mat = {}
    
    for paw in intralimb_paws:
        swing_z_current_mat[paw] = np.full((len(animal_ids), 100), np.nan)
        swing_inst_vel_current_mat[paw] = np.full((len(animal_ids), 100), np.nan)
        
    for count_a, animal_id in enumerate(animal_ids):
        session_paths = animal_session_folders[animal_id]
        phase_st_animal = multi_session_utils.average_session_phase_arrays(session_paths, 'phase_st.npy')
        phase_bin_paw_current_mat[count_a, :, :] = phase_st_animal
        for paw in intralimb_paws:
            swing_z_animal = multi_session_utils.average_session_arrays(session_paths, 'swing_z'+'_'+paw+'.npy')
            swing_z_current_mat[paw][count_a, :] = swing_z_animal[speed_bin_side, :]
            swing_inst_vel_animal = multi_session_utils.average_session_arrays(session_paths, 'swing_inst_vel'+'_'+paw+'.npy')
            swing_inst_vel_current_mat[paw][count_a, :] = swing_inst_vel_animal[speed_bin_side, :]
            

    # Stance phase plot
    phase_ax = ax_phase_st[path_index]
    for p, paw_name in enumerate(phase_paws):
        paw_phase_values = phase_bin_paw_current_mat[:, p, :]
        for animal_phase_values in paw_phase_values:
            phase_ax.scatter(
                animal_phase_values,
                full_speed_values,
                color=paw_colors[p],
                s=18,
                alpha=0.25,
                edgecolors='none',
            )
        mean_values = multi_session_utils.circular_mean_angles(paw_phase_values, axis=0)
        phase_ax.scatter(mean_values, full_speed_values, color=paw_colors[p], s=64, alpha=0.95, edgecolors='none')
        if path_index == 0:
            ax_phase_st_summary.scatter(
                mean_values,
                full_speed_values,
                color=paw_colors[p],
                marker=markers[path_index],
                s=120,
                alpha=0.5,
                edgecolors='none',
            )
        elif path_index == 1:
            ax_phase_st_summary.scatter(
                mean_values,
                full_speed_values,
                color=paw_colors[p],
                marker=markers[path_index],
                s=120,
                edgecolors='none',
            )
        else:
            ax_phase_st_summary.scatter(
                mean_values,
                full_speed_values,
                color=paw_colors[p],
                marker=markers[path_index],
                s=60,
                edgecolors='none',
            )

    phase_prediction_speeds = np.linspace(speed_range[0], speed_range[-2], 100)
    for phase_name, phase_config in phase_summary_plot_config.items():
        phase_panel_dfs = []
        for pair_label, pair_config in phase_config['pairs'].items():
            phase_df = multi_session_utils.build_phase_summary_dataframe(
                phase_bin_paw_current_mat,
                animal_ids,
                current_label,
                pair_label,
                pair_config['first_paw'],
                pair_config['second_paw'],
                phase_paw_index,
                full_speed_values,
            )
            if phase_df.empty:
                continue
            phase_panel_dfs.append(phase_df)
            pair_style = phase_pair_styles[pair_label]
            if pair_style['filled']:
                ax_phase_summary[phase_name].scatter(
                    phase_df['speed'],
                    phase_df['phase_value'],
                    color=current_color,
                    s=24,
                    alpha=0.75,
                    edgecolors='none',
                )
            else:
                ax_phase_summary[phase_name].scatter(
                    phase_df['speed'],
                    phase_df['phase_value'],
                    facecolors='none',
                    edgecolors=current_color,
                    s=24,
                    alpha=0.55,
                    linewidths=1,
                )
        if phase_panel_dfs:
            combined_phase_df = multi_session_utils.build_phase_fit_dataframe(pd.concat(phase_panel_dfs, ignore_index=True))
            phase_fit = multi_session_utils.fit_phase_mixedlm(
                combined_phase_df,
                phase_prediction_speeds,
                phase_config['title'],
                current_label,
                'combined',
            )
            if phase_fit is not None:
                ax_phase_summary[phase_name].plot(
                    phase_prediction_speeds,
                    phase_fit,
                    color=current_color,
                    linewidth=2.5,
                )

    # Swing inst velocity plot
    for paw in intralimb_paws:
        line_label = current_label if paw == 'FR' else None
        ax_swing_vel[paw].plot(np.linspace(0, 100, 100), np.nanmean(swing_inst_vel_current_mat[paw], axis=0), color=current_color, linewidth=2, label=line_label)
        ax_swing_vel[paw].fill_between(np.linspace(0, 100, 100),
                        np.nanmean(swing_inst_vel_current_mat[paw], axis=0)-np.nanstd(swing_inst_vel_current_mat[paw], axis=0),
                np.nanmean(swing_inst_vel_current_mat[paw], axis=0)+np.nanstd(swing_inst_vel_current_mat[paw], axis=0), color=current_color, alpha=0.3)
    
    # Swing amplitude plot
    for paw in intralimb_paws:
        line_label = current_label if paw == 'FR' else None
        ax_swing_z[paw].plot(np.linspace(0, 100, 100), np.nanmean(swing_z_current_mat[paw], axis=0), color=current_color, linewidth=2, label=line_label)
        ax_swing_z[paw].fill_between(np.linspace(0, 100, 100),
                        np.nanmean(swing_z_current_mat[paw], axis=0)-np.nanstd(swing_z_current_mat[paw], axis=0),
        np.nanmean(swing_z_current_mat[paw], axis=0)+np.nanstd(swing_z_current_mat[paw], axis=0), color=current_color, alpha=0.3)
        peak_swing_amplitude_by_path[paw].append(np.nanmax(swing_z_current_mat[paw], axis=1))


    # Supports
    # BOS
    swing_x_rel_current_mat = np.full((len(animal_ids), 4, len(speed_bins), 100), np.nan)
    swing_y_rel_current_mat= np.full((len(animal_ids), 4, len(speed_bins), 100), np.nan)
    for count_a, animal_id in enumerate(animal_ids):
        session_paths = animal_session_folders[animal_id]
        swing_x_rel_animal = multi_session_utils.average_session_arrays(session_paths, 'swing_x_rel.npy')
        swing_x_rel_current_mat[count_a, :, :, :] = swing_x_rel_animal
        swing_y_rel_animal = multi_session_utils.average_session_arrays(session_paths, 'swing_y_rel.npy')
        swing_y_rel_current_mat[count_a, :, :, :] = swing_y_rel_animal

    
    for p in range(4):
        b_count = 0
        for b in speed_bins:
            line_label = current_label if p == 0 and b_count == 0 else None
            ax_bos.plot(np.mean(swing_y_rel_current_mat[:,p,b_count,:], axis=0),np.mean(swing_x_rel_current_mat[:,p,b_count,:], axis=0),linewidth=speed_bins[b_count]/2,color=current_color, alpha=1, label=line_label)
            ax_bos.set_title('base of support', fontsize=20)
            ax_bos.set_ylabel('swing x rel', fontsize=16)
            ax_bos.set_xlabel('swing y rel', fontsize=16)
            ax_bos.tick_params(axis='x', labelsize=14)
            ax_bos.tick_params(axis='y', labelsize=14)
            ax_bos.spines['right'].set_visible(False)
            ax_bos.spines['top'].set_visible(False)
            ax_bos.set_xlim([-15, 15])       
            ax_bos.set_ylim([-40, 40]) 
            b_count += 1


    # % Supports
    supports_current_mat = np.full((len(animal_ids),len(speed_range)-1,7), np.nan)
    for count_a, animal_id in enumerate(animal_ids):
        session_paths = animal_session_folders[animal_id]
        supports_animal = multi_session_utils.average_session_arrays(session_paths, 'supports.npy')
        supports_current_mat[count_a, :, :] = supports_animal
    support_prediction_speeds = np.linspace(speed_range[0], speed_range[-2], 100)
    for support_name, support_config in support_plot_config.items():
        support_df = multi_session_utils.build_support_dataframe(
            supports_current_mat,
            animal_ids,
            current_label,
            full_speed_values,
            support_config['column'],
        )
        ax_supports[support_name].scatter(
            support_df['speed'],
            support_df['support_value'],
            color=current_color,
            s=16,
            alpha=0.55,
            edgecolors='none',
        )
        support_fit = multi_session_utils.fit_support_mixedlm(
            support_df,
            support_prediction_speeds,
            support_config['title'],
            current_label,
        )
        if support_fit is not None:
            ax_supports[support_name].plot(
                support_prediction_speeds,
                support_fit,
                color=current_color,
                linewidth=2.5,
            )

    bar_width = 0.04 / len(paths)
    bar_offset = (path_index - (len(paths) - 1) / 2) * bar_width
    stride_count_total = np.sum(stride_count_current)
    if stride_count_total > 0:
        stride_count_density = stride_count_current / stride_count_total
    else:
        stride_count_density = stride_count_current
    ax_stride_count.bar(full_speed_values + bar_offset, stride_count_density, width=bar_width, color=current_color, alpha=0.8, label=current_label, align='center')

    path_index += 1


phase_paw_handles = [
    Line2D(
        [0],
        [0],
        color=paw_colors[count_p],
        marker='o',
        linestyle='-',
        linewidth=2,
        markersize=6,
        label=paw_name,
    )
    for count_p, paw_name in enumerate(phase_paws)
]
fig_phase_st.legend(handles=phase_paw_handles, frameon=False, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02), fontsize=11)

phase_experiment_handles = [
    Line2D(
        [0],
        [0],
        markerfacecolor='dimgray',
        markeredgecolor='dimgray',
        marker=markers[path_index],
        linestyle='None',
        markersize=8,
        label=multi_session_utils.get_path_label(path, cohort_colors),
    )
    for path_index, path in enumerate(paths)
]
phase_summary_paw_legend = ax_phase_st_summary.legend(handles=phase_paw_handles, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11)
ax_phase_st_summary.add_artist(phase_summary_paw_legend)
ax_phase_st_summary.legend(handles=phase_experiment_handles, frameon=False, loc='lower left', bbox_to_anchor=(1.02, 0), fontsize=11)

phase_group_handles = [
    Line2D(
        [0],
        [0],
        color=multi_session_utils.get_path_color(path, cohort_colors),
        linewidth=2.5,
        marker='o',
        markersize=5,
        markerfacecolor=multi_session_utils.get_path_color(path, cohort_colors),
        markeredgewidth=0,
        label=multi_session_utils.get_path_label(path, cohort_colors),
    )
    for path in paths
]
front_hind_pair_handles = [
    Line2D(
        [0],
        [0],
        color='black',
        linewidth=0,
        marker='o',
        markersize=6,
        markerfacecolor='black' if phase_pair_styles[pair_label]['filled'] else 'white',
        markeredgecolor='black',
        label=pair_label,
    )
    for pair_label in ['right', 'left']
]
left_right_pair_handles = [
    Line2D(
        [0],
        [0],
        color='black',
        linewidth=0,
        marker='o',
        markersize=6,
        markerfacecolor='black' if phase_pair_styles[pair_label]['filled'] else 'white',
        markeredgecolor='black',
        label=pair_label,
    )
    for pair_label in ['front', 'hind']
]
front_hind_group_legend = ax_phase_summary['front_hind'].legend(handles=phase_group_handles, frameon=False, loc='upper right', fontsize=11)
ax_phase_summary['left_right'].legend(handles=left_right_pair_handles, frameon=False, loc='lower right', fontsize=11)
front_hind_pair_legend = ax_phase_summary['front_hind'].legend(handles=front_hind_pair_handles, frameon=False, loc='lower right', fontsize=11)
ax_phase_summary['front_hind'].add_artist(front_hind_group_legend)
ax_phase_summary['front_hind'].add_artist(front_hind_pair_legend)

bos_legend_handles = [
    Line2D(
        [0],
        [0],
        color=multi_session_utils.get_path_color(path, cohort_colors),
        linewidth=3.0,
        alpha=1.0,
        label=multi_session_utils.get_path_label(path, cohort_colors),
    )
    for path in paths
]
ax_bos.legend(handles=bos_legend_handles, frameon=False, loc='upper left', bbox_to_anchor=(1.16, 1), borderaxespad=0, fontsize=11)
support_legend_handles = [
    Line2D(
        [0],
        [0],
        color=multi_session_utils.get_path_color(path, cohort_colors),
        linewidth=2.5,
        marker='o',
        markersize=5,
        markerfacecolor=multi_session_utils.get_path_color(path, cohort_colors),
        markeredgewidth=0,
        alpha=0.8,
        label=multi_session_utils.get_path_label(path, cohort_colors),
    )
    for path in paths
]
for support_name, support_config in support_plot_config.items():
    ax_supports[support_name].set_title(support_config['title'], fontsize=20)
ax_supports['three_paws'].legend(handles=support_legend_handles, frameon=False, loc='lower right', fontsize=11)

path_positions = np.arange(1, len(paths) + 1)
path_labels = [multi_session_utils.get_path_label(path, cohort_colors) for path in paths]
for paw in intralimb_paws:
    boxplot = ax_swing_z_peak[paw].boxplot(
        peak_swing_amplitude_by_path[paw],
        positions=path_positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={'color': 'black', 'linewidth': 1.5},
        whiskerprops={'color': 'black', 'linewidth': 1.2},
        capprops={'color': 'black', 'linewidth': 1.2},
        boxprops={'edgecolor': 'black', 'linewidth': 1.2},
    )
    for patch, path in zip(boxplot['boxes'], paths):
        patch.set_facecolor(multi_session_utils.get_path_color(path, cohort_colors))
        patch.set_alpha(0.7)
    ax_swing_z_peak[paw].set_xticks(path_positions)
    ax_swing_z_peak[paw].set_xticklabels(path_labels, rotation=20)
    ax_swing_z_peak[paw].set_ylim(0, swing_amplitude_ylim[1])




# Define the directory where the file will be saved
save_dir = os.path.join(paths[0], 'gait parameters combined pixel_to_mm Jovin\\')

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

for count_p, g in enumerate(param_tied):
    fig_intra[count_p].savefig(os.path.join(save_dir, f'loco_analysis_param_intralimb_{g}.png'), dpi=256)
    fig_intra[count_p].savefig(os.path.join(save_dir, f'loco_analysis_param_intralimb_{g}.svg'), dpi=256)

fig_phase_st.savefig(os.path.join(save_dir, 'loco_analysis_phase_st'), dpi=256, bbox_inches='tight')
fig_phase_st.savefig(os.path.join(save_dir,'loco_analysis_phase_st.svg'), dpi=256, bbox_inches='tight')

fig_phase_st_summary.savefig(os.path.join(save_dir, 'loco_analysis_phase_st_summary'), dpi=256, bbox_inches='tight')
fig_phase_st_summary.savefig(os.path.join(save_dir, 'loco_analysis_phase_st_summary.svg'), dpi=256, bbox_inches='tight')

fig_phase_summary.savefig(os.path.join(save_dir, 'loco_analysis_phase_summary'), dpi=256)
fig_phase_summary.savefig(os.path.join(save_dir, 'loco_analysis_phase_summary.svg'), dpi=256)

fig_swing_vel.savefig(os.path.join(save_dir,'loco_analysis_swing_inst_vel'), dpi=256)
fig_swing_vel.savefig(os.path.join(save_dir,'loco_analysis_swing_inst_vel'), dpi=256)

fig_swing_z.savefig(os.path.join(save_dir,'loco_analysis_swing_z'), dpi=256)
fig_swing_z.savefig(os.path.join(save_dir,'loco_analysis_swing_z.svg'), dpi=256)

fig_swing_z_peak.savefig(os.path.join(save_dir,'loco_analysis_swing_z_peak_boxplot'), dpi=256)
fig_swing_z_peak.savefig(os.path.join(save_dir,'loco_analysis_swing_z_peak_boxplot.svg'), dpi=256)


fig_bos.savefig(os.path.join(save_dir,'loco_analysis_bos'), dpi=256)
fig_bos.savefig(os.path.join(save_dir,'loco_analysis_bos.svg'), dpi=256)

fig_supports.savefig(os.path.join(save_dir,'loco_analysis_supports'), dpi=256)
fig_supports.savefig(os.path.join(save_dir,'loco_analysis_supports.svg'), dpi=256)

fig_stride_count.savefig(os.path.join(save_dir,'loco_analysis_stride_count_histogram'), dpi=256)
fig_stride_count.savefig(os.path.join(save_dir,'loco_analysis_stride_count_histogram.svg'), dpi=256)


print(f"Saving figure to: {save_dir}")

'''
# Plot phases and trajectories
phase_bin_paw_control_mat = np.zeros((len(folders_animals), 4, len(speed_range)-1))
swing_z_control_mat = np.zeros((len(folders_animals), 100))
swing_inst_vel_control_mat = np.zeros((len(folders_animals), 100))
swing_x_rel_control_mat = np.zeros((len(folders_animals), 4, 100))
swing_y_rel_control_mat = np.zeros((len(folders_animals), 4, 100))
for count_a, a in enumerate(folders_animals):
    control_path = main_control_path + a
    phase_st_animal = np.load(control_path + '\\phase_st.npy')
    phase_bin_paw_control_mat[count_a, :, :] = phase_st_animal
    swing_z_animal = np.load(control_path + '\\swing_z.npy')
    swing_z_control_mat[count_a, :] = swing_z_animal[speed_bin_side, :]
    swing_inst_vel_animal = np.load(control_path + '\\swing_inst_vel.npy')
    swing_inst_vel_control_mat[count_a, :] = swing_inst_vel_animal[speed_bin_side, :]
    swing_x_rel_animal = np.load(control_path + '\\swing_x_rel.npy')
    swing_x_rel_control_mat[count_a, :, :] = swing_x_rel_animal[:, speed_bin_side, :]
    swing_y_rel_animal = np.load(control_path + '\\swing_y_rel.npy')
    swing_y_rel_control_mat[count_a, :, :] = swing_y_rel_animal[:, speed_bin_side, :]
phase_bin_paw_all_mat = np.zeros((len(phase_bin_paw_all), 4, len(speed_range)-1))
swing_z_all_mat = np.zeros((len(phase_bin_paw_all), 100))
swing_inst_vel_all_mat = np.zeros((len(phase_bin_paw_all), 100))
swing_x_rel_all_mat = np.zeros((len(phase_bin_paw_all), 4, 100))
swing_y_rel_all_mat = np.zeros((len(phase_bin_paw_all), 4, 100))
for a in range(len(phase_bin_paw_all)):
    phase_bin_paw_all_mat[a, :, :] = phase_bin_paw_all[a]
    swing_z_all_mat[a, :] = swing_z_all[a][speed_bin_side]
    swing_inst_vel_all_mat[a, :] = swing_inst_vel_all[a][speed_bin_side]
    swing_x_rel_all_mat[a, :, ] = swing_x_rel_all[a][:, speed_bin_side, :]
    swing_y_rel_all_mat[a, :, ] = swing_y_rel_all[a][:, speed_bin_side, :]






# quantification of maximum z amplitude
swing_z_max_control = np.nanmax(swing_z_control_mat, axis=1)
swing_z_max_miniscope = np.nanmax(swing_z_all_mat, axis=1)
swing_z_max_df = pd.DataFrame({'max': np.concatenate((swing_z_max_control, swing_z_max_miniscope)),
'group': np.concatenate((np.repeat(0, len(swing_z_max_control)), np.repeat(1, len(swing_z_max_miniscope))))})
fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
sns.boxplot(x='group', y='max', data=swing_z_max_df,
            medianprops=dict(color='black'), palette={0: 'darkgrey', 1: 'darkviolet'}, showfliers=False)
ax.set_xticklabels(['Without\nminiscopes', 'With\nminiscopes'])
ax.set_xlabel('')
ax.set_ylabel('Peak swing\namplitude', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_ylim([3, 4.5])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('J:\\Thesis\\figuresChapter2\\loco_analysis_swing_z_max', dpi=256)
plt.savefig('J:\\Thesis\\figuresChapter2\\loco_analysis_swing_z_max.svg', dpi=256)
fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
for p in range(4):
    ax.plot(np.nanmean(swing_y_rel_control_mat[:, p, :], axis=0), np.nanmean(swing_x_rel_control_mat[:, p, :], axis=0), color='black', linewidth=2)
    ax.fill_between(np.nanmean(swing_y_rel_control_mat[:, p, :], axis=0),
                       np.nanmean(swing_x_rel_control_mat[:, p, :], axis=0) - np.nanstd(swing_x_rel_control_mat[:, p, :], axis=0),
                       np.nanmean(swing_x_rel_control_mat[:, p, :], axis=0) + np.nanstd(swing_x_rel_control_mat[:, p, :], axis=0), color='black',
                       alpha=0.3)
    ax.plot(np.nanmean(swing_y_rel_all_mat[:, p, :], axis=0), np.nanmean(swing_x_rel_all_mat[:, p, :], axis=0), color='darkviolet', linewidth=2)
    ax.fill_between(np.nanmean(swing_y_rel_all_mat[:, p, :], axis=0),
                       np.nanmean(swing_x_rel_all_mat[:, p, :], axis=0) - np.nanstd(swing_x_rel_all_mat[:, p, :], axis=0),
                       np.nanmean(swing_x_rel_all_mat[:, p, :], axis=0) + np.nanstd(swing_x_rel_all_mat[:, p, :], axis=0), color='darkviolet',
                       alpha=0.3)
    ax.set_xlabel('Y relative to bodycenter (mm)', fontsize=20)
    ax.set_ylabel('X relative to bodycenter (mm)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
plt.savefig('J:\\Thesis\\figuresChapter2\\loco_analysis_bos', dpi=256)
plt.savefig('J:\\Thesis\\figuresChapter2\\loco_analysis_bos.svg', dpi=256)
# plt.savefig('J:\\Miniscope processed files\\tied belt locomotion analysis\\trajectories.png')'''