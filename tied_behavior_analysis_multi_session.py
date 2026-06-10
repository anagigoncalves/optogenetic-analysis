"""Created on Fri May 9 10:24:34 2025

@author: Ana Goncalves and Alice Geminiani

It compares the locomouse parameters for different experiments, after having computed them for each animal and each experiment with the "tied_behavior_analysis_single_animal.py" script.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

paths = ['D:\\AliG\\climbing-opto-treadmill\\WT split-belt learning\\',
    'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\HISTO_CHECKED_ANIMALS_LATinj\\split contra fast\\',
'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\HISTO_CHECKED_ANIMALS_RLinj\\split contra fast right\\'
]   

colors = ['darkgray', 'lightseagreen', 'darkblue']
markers = ['o', '*', '^']
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']
paw_names = ['FR', 'HR', 'FL', 'HL']
paws = ['FR', 'FL']
param_tied = ['stance_duration', 'swing_duration', 'cadence', 'swing_length']
speed_range = np.arange(0.1,0.4,0.05)
speed_bins = np.array([1, 2, 3])
speed_bin_side = 2

# For each folder path, we want to load the data and parameters, and plot them

# Initialize figures with different handlers
fig_intra = [plt.figure(num=i+1, figsize=(10, 5), tight_layout=True) for i in range(len(param_tied))]            #Intralimb parameters
ax_intra = {'FR': [fig.add_subplot(121) for fig in fig_intra], 'FL': [fig.add_subplot(122) for fig in fig_intra]}
for paw in paws:
    for ax in ax_intra[paw]:
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

fig_phase_st = plt.figure(num=len(param_tied)+1)            # Phase stance
ax_phase_st = fig_phase_st.add_subplot(111, projection='polar')

fig_swing_vel = plt.figure(num=len(param_tied)+2, figsize=(10, 5))            # Swing velocity
ax_swing_vel = {'FR': fig_swing_vel.add_subplot(121), 'FL': fig_swing_vel.add_subplot(122)}
for paw in paws:
    ax_swing_vel[paw].set_xlabel('% swing (norm)', fontsize=20)
    ax_swing_vel[paw].set_ylabel('Swing instantaneous\nvelocity (m/s)', fontsize=20)
    ax_swing_vel[paw].tick_params(axis='both', which='major', labelsize=20)
    ax_swing_vel[paw].spines['right'].set_visible(False)
    ax_swing_vel[paw].spines['top'].set_visible(False)

fig_swing_z = plt.figure(num=len(param_tied)+3, figsize=(10, 5))            # Swing amplitude
ax_swing_z = {'FR': fig_swing_z.add_subplot(121), 'FL': fig_swing_z.add_subplot(122)}
for paw in paws:
    ax_swing_z[paw].set_xlabel('% swing (norm)', fontsize=20)
    ax_swing_z[paw].set_ylabel('swing amplitude (mm)', fontsize=20)
    ax_swing_z[paw].tick_params(axis='both', which='major', labelsize=20)
    ax_swing_z[paw].spines['right'].set_visible(False)
    ax_swing_z[paw].spines['top'].set_visible(False)

fig_bos = plt.figure(num=len(param_tied)+5, figsize=(10, 5))            # Body center of mass
ax_bos = fig_bos.add_subplot(121)
ax_supp = fig_bos.add_subplot(122)

path_index = 0

# LOAD the DATA
for path in paths:
    folders_animals = os.listdir(path+'\\gait parameters\\')
    # Intralimb parameters for FR and FL paws
    param_tied_current_values = {}
    param_tied_current_speed = {}
    for paw in paws:
        param_tied_current_values[paw] = np.zeros((len(param_tied), len(folders_animals), len(speed_range)-1))
        param_tied_current_speed[paw] = np.zeros((len(param_tied), len(folders_animals), len(speed_range)-1))
    for count_p, g in enumerate(param_tied):
        for count_a, a in enumerate(folders_animals):
            current_path = path + a
            for paw in paws:  
                param_tied_current_file = np.load(path + '\\gait parameters\\' + a + '\\' + param_tied[count_p] + '_'+paw+'.npy')
                param_tied_current_df = pd.DataFrame({'values': param_tied_current_file[:, 0], 'speed': param_tied_current_file[:, 1]})
                param_tied_current_mean = param_tied_current_df.groupby(['speed'])['values'].mean()
                param_tied_current_speed[paw][count_p, count_a, :] = param_tied_current_mean.index
                param_tied_current_values[paw][count_p, count_a, :] = param_tied_current_mean.values
    # Intralimb parameters PLOT
    ylabel = ['Stance duration (ms)', 'Swing duration (ms)', 'Cadence ($\mathregular{ms^{-1}}$)', 'Swing length (mm)']
    for paw in paws:
        for count_p, g in enumerate(param_tied):
            ax_intra[paw][count_p].plot(param_tied_current_speed[paw][0, 0, :], np.nanmean(param_tied_current_values[paw][count_p, :, :], axis=0), color=colors[path_index], linewidth=2)
            ax_intra[paw][count_p].fill_between(param_tied_current_speed[paw][0, 0, :],
                            np.nanmean(param_tied_current_values[paw][count_p, :, :], axis=0)-np.nanstd(param_tied_current_values[paw][count_p, :, :], axis=0),
            np.nanmean(param_tied_current_values[paw][count_p, :, :], axis=0)+np.nanstd(param_tied_current_values[paw][count_p, :, :], axis=0), color=colors[path_index], alpha=0.3)
            ax_intra[paw][count_p].set_xlabel('Speed (m/s)', fontsize=20)
            # ax.set_title(g.replace('_', ' ') + ' FR paw', fontsize=20)
            ax_intra[paw][count_p].set_ylabel(ylabel[count_p], fontsize=20)

    
    # Stance phase and trajectories
    phase_bin_paw_current_mat = np.zeros((len(folders_animals), 4, len(speed_range)-1))
    swing_z_current_mat = {}
    swing_inst_vel_current_mat = {}
    
    for paw in paws:   
        swing_z_current_mat[paw] = np.zeros((len(folders_animals), 100))
        swing_inst_vel_current_mat[paw] = np.zeros((len(folders_animals), 100))
        
    for count_a, a in enumerate(folders_animals):
        current_path = path + '\\gait parameters\\' + a
        phase_st_animal = np.load(current_path + '\\phase_st.npy')
        phase_bin_paw_current_mat[count_a, :, :] = phase_st_animal
        for paw in paws:
            swing_z_animal = np.load(current_path + '\\swing_z'+'_'+paw+'.npy')
            swing_z_current_mat[paw][count_a, :] = swing_z_animal[speed_bin_side, :]
            swing_inst_vel_animal = np.load(current_path + '\\swing_inst_vel'+'_'+paw+'.npy')
            swing_inst_vel_current_mat[paw][count_a, :] = swing_inst_vel_animal[speed_bin_side, :]
            

    # Stance phase plot
    for p in range(4):
        ax_phase_st.scatter(np.nanmean(phase_bin_paw_current_mat[:, p, :], axis=0), speed_range[:-1], c=paw_colors[p], marker=markers[path_index], s=120)
        ax_phase_st.tick_params(axis='both', which='major', labelsize=20)
        ax_phase_st.set_yticks(speed_range[:-1:2])

    # Swing inst velocity plot
    for paw in paws:
        ax_swing_vel[paw].plot(np.linspace(0, 100, 100), np.nanmean(swing_inst_vel_current_mat[paw], axis=0), color=colors[path_index], linewidth=2)
        ax_swing_vel[paw].fill_between(np.linspace(0, 100, 100),
                        np.nanmean(swing_inst_vel_current_mat[paw], axis=0)-np.nanstd(swing_inst_vel_current_mat[paw], axis=0),
                        np.nanmean(swing_inst_vel_current_mat[paw], axis=0)+np.nanstd(swing_inst_vel_current_mat[paw], axis=0), color=colors[path_index], alpha=0.3)
    
    # Swing amplitude plot
    for paw in paws:
        ax_swing_z[paw].plot(np.linspace(0, 100, 100), np.nanmean(swing_z_current_mat[paw], axis=0), color=colors[path_index], linewidth=2)
        ax_swing_z[paw].fill_between(np.linspace(0, 100, 100),
                        np.nanmean(swing_z_current_mat[paw], axis=0)-np.nanstd(swing_z_current_mat[paw], axis=0),
        np.nanmean(swing_z_current_mat[paw], axis=0)+np.nanstd(swing_z_current_mat[paw], axis=0), color=colors[path_index], alpha=0.3)


    # Supports
    # BOS
    swing_x_rel_current_mat = np.zeros((len(folders_animals), 4, len(speed_bins), 100))
    swing_y_rel_current_mat= np.zeros((len(folders_animals), 4, len(speed_bins), 100))
    for count_a, a in enumerate(folders_animals):
        current_path = path + '\\gait parameters\\' + a
        swing_x_rel_animal = np.load(current_path + '\\swing_x_rel.npy')
        swing_x_rel_current_mat[count_a, :, :, :] = swing_x_rel_animal
        swing_y_rel_animal = np.load(current_path + '\\swing_y_rel.npy')
        swing_y_rel_current_mat[count_a, :, :, :] = swing_y_rel_animal

    
    for p in range(4):
        b_count = 0
        for b in speed_bins:
            ax_bos.plot(np.mean(swing_y_rel_current_mat[:,p,b_count,:], axis=0),np.mean(swing_x_rel_current_mat[:,p,b_count,:], axis=0),linewidth=speed_bins[b_count]/2,color=colors[path_index], alpha=0.5)
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
    supports_current_mat = np.zeros((len(folders_animals),len(speed_range)-1,7))
    for count_a, a in enumerate(folders_animals):
        current_path = path + '\\gait parameters\\' + a
        supports_animal = np.load(current_path + '\\supports.npy')
        supports_current_mat[count_a, :, :] = supports_animal
    ax_supp.plot(speed_range[:-1],np.mean(supports_current_mat[:,:,2], axis=0),label='diagonals FL',color=colors[path_index])
    ax_supp.plot(speed_range[:-1],np.mean(supports_current_mat[:,:,3], axis=0),label='diagonals FR',linestyle = 'dashed', color=colors[path_index])
    ax_supp.plot(speed_range[:-1],np.mean(supports_current_mat[:,:,4], axis=0),label='homolateral', linestyle = ':', color=colors[path_index])
    ax_supp.plot(speed_range[:-1],np.mean(supports_current_mat[:,:,1], axis=0),label='3 paws',linestyle = '-.', color=colors[path_index])
    ax_supp.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.3, 1))
    ax_supp.set_title('supports', fontsize=20)
    ax_supp.set_ylabel('% of support', fontsize=16)
    ax_supp.set_xlabel('speed', fontsize=16)
    ax_supp.tick_params(axis='x', labelsize=14)
    ax_supp.tick_params(axis='y', labelsize=14)
    ax_supp.spines['right'].set_visible(False)
    ax_supp.spines['top'].set_visible(False)

    path_index += 1




# Define the directory where the file will be saved
save_dir = os.path.join(paths[0], 'gait parameters combined\\')

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

for count_p, g in enumerate(param_tied):
    fig_intra[count_p].savefig(os.path.join(save_dir, f'loco_analysis_param_intralimb_{g}.png'), dpi=256)
    fig_intra[count_p].savefig(os.path.join(save_dir, f'loco_analysis_param_intralimb_{g}.svg'), dpi=256)

fig_phase_st.savefig(os.path.join(save_dir, 'loco_analysis_phase_st'), dpi=256)
fig_phase_st.savefig(os.path.join(save_dir,'loco_analysis_phase_st.svg'), dpi=256)

fig_swing_vel.savefig(os.path.join(save_dir,'loco_analysis_swing_inst_vel'), dpi=256)
fig_swing_vel.savefig(os.path.join(save_dir,'loco_analysis_swing_inst_vel'), dpi=256)

fig_swing_z.savefig(os.path.join(save_dir,'loco_analysis_swing_z'), dpi=256)
fig_swing_z.savefig(os.path.join(save_dir,'loco_analysis_swing_z.svg'), dpi=256)


fig_bos.savefig(os.path.join(save_dir,'loco_analysis_bos'), dpi=256)
fig_bos.savefig(os.path.join(save_dir,'loco_analysis_bos.svg'), dpi=256)


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