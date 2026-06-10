import os
import numpy as np
import matplotlib.pyplot as plt

#path inputs
path_loco = 'C:\\Users\\Ana\\Desktop\\Opto JAWS RT\\tied stance stim\\'
perc_division = 10 #percentage of the trial to plot in each bin
paws = ['FR', 'HR', 'FL', 'HL']
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']
# JAWS histology-confirmed (18 animals)
animals = ['MC16848', 'MC16851', 'MC17319', 'MC17665', 'MC17666', 'MC17669', 'MC17670', 'MC19082', 'MC19124',
                         'MC19214', 'VIV41329', 'VIV41330', 'VIV42375', 'VIV42428', 'VIV42429', 'VIV42430', 'VIV42376', 'MC19107']
# Animals with tied session of 24 trials instead of 28 - to adjust bins
tied_session_short = ['MC16846', 'MC16848', 'MC16850', 'MC16851', 'MC17319', 'MC17665', 'MC17666', 'MC17668', 'MC17669', 'MC17670']
stim_trials = np.arange(9, 18)
Ntrials = 28
color_cond = 'orange'

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

#summary gait parameters
param_sym_name = ['double_support']
param_sym_label = ['Percentage of double\nsupport symmetry']
param_label = ['Percentage of\ndouble support']
param_sym = np.zeros((len(param_sym_name), len(animal_list_plot), Ntrials*np.int64(100/perc_division)))
param_sym[:] = np.nan
for count_animal, animal in enumerate(animal_list_plot):
    session = int(session_list_plot[count_animal])
    filelist = loco.get_track_files(animal, session)
    # get trial list in filelist
    trial_filelist = np.zeros(len(filelist))
    for count_f, f in enumerate(filelist):
        trial_filelist[count_f] = np.int64(f.split('_')[7][:f.split('_')[7].find('D')])
    perc_times = np.int64(100 / perc_division)
    bin_range = np.arange(0, Ntrials*np.int64(100/perc_division))
    bin_range_split = np.split(bin_range, len(bin_range)/perc_times)
    bin_range_split_trials_in_ses = []
    for t in np.int64(trial_filelist):
        bin_range_split_trials_in_ses.append(bin_range_split[t-1])
    bin_range_trials_in_ses = np.hstack(bin_range_split_trials_in_ses)
    for count_p, param in enumerate(param_sym_name):
        param_sym_session = []
        for count_trial, f in enumerate(filelist):
            [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, 0)
            # function to separate the final_tracks into bins
            [final_tracks_perctrial, bodycenter_perctrial] = loco.final_tracks_perctrial(final_tracks, bodycenter, perc_division)
            param_sym_single_trial = []
            for i in range(len(final_tracks_perctrial)):
                #can happen that there are no tracks if bins is too small
                if not final_tracks_perctrial[1].size > 0:
                    #empty final tracks
                    param_sym_single_trial.append(np.nan)
                else:
                    #normal gait param computation on final tracks perc trial
                    [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks_perctrial[i], 1)
                    paws_rel = loco.get_paws_rel(final_tracks_perctrial[i], 'X')
                    param_mat = loco.compute_gait_param(bodycenter_perctrial[i], final_tracks_perctrial[i], paws_rel, st_strides_mat, sw_pts_mat, param)
                    param_sym_single_trial.append(np.nanmean(param_mat[0])-np.nanmean(param_mat[2]))
            param_sym_session.extend(param_sym_single_trial)
        #matching number of trials in final array of gait params
        if len(param_sym_session) < np.shape(param_sym)[2]:
            param_sym[count_p, count_animal, bin_range_trials_in_ses] = param_sym_session
        elif len(param_sym_session) == np.shape(param_sym)[2]:
            param_sym[count_p, count_animal, :] = param_sym_session
        else:
            print('NUMBER OF TRIALS IS HIGHER THAN THE DEFINED NTRIALS')

#Plot
#baseline subtraction of parameters
param_sym_bs = np.zeros(np.shape(param_sym))
param_sym_bs[:] = np.nan
for p in range(np.shape(param_sym)[0]):
    for a in range(np.shape(param_sym)[1]):
        bs_mean = np.nanmean(param_sym[p, a, :stim_trials[0]-1])
        param_sym_bs[p, a, :] = param_sym[p, a, :] - bs_mean

perc_times = np.int64(100/perc_division)
bin_size = np.int64((perc_division/100)*60)
#plot symmetry baseline subtracted - mean animals
for p in range(np.shape(param_sym)[0]):
    fig, ax = plt.subplots(figsize=(10, 15), tight_layout=True)
    mean_data = np.nanmean(param_sym_bs[p, :, :], axis=0)
    std_data = (np.nanstd(param_sym_bs[p, :, :], axis=0)/np.sqrt(len(animals)))
    rectangle = plt.Rectangle((stim_trials[0]*perc_times-0.5, np.nanmin(mean_data-std_data)), 8*perc_times, np.nanmax(mean_data+std_data)-np.nanmin(mean_data-std_data), fc='lightblue', alpha=0.3)
    plt.gca().add_patch(rectangle)
    plt.hlines(0, 1, Ntrials*perc_times, colors='grey', linestyles='--')
    plt.plot(np.arange(1, Ntrials*perc_times+1), mean_data, linewidth=2, marker='o', color=color_cond)
    plt.fill_between(np.arange(1, Ntrials*perc_times+1), mean_data-std_data, mean_data+std_data, color=color_cond, alpha=0.5)
    ax.set_xlabel('Bins', fontsize=20)
    ax.set_ylabel(param_sym_label[p], fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(path_save, 'mean_animals_symmetry_' + param_sym_name[p] + '_perc_trial.png'), dpi=128)
    plt.savefig(os.path.join(path_save, 'mean_animals_symmetry_' + param_sym_name[p] + '_perc_trial.svg'), dpi=128)

#quantification of change within trial - mean data
cmap = plt.get_cmap('Reds', len(stim_trials))
for p in range(np.shape(param_sym)[0]):
    mean_data = np.nanmean(param_sym_bs[p, :, :], axis=0)
    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
    for count_t, t in enumerate(stim_trials):
        plt.plot(np.arange(bin_size, bin_size*perc_times+1, bin_size),
                 mean_data[(t-1)*perc_times:(t-1)*perc_times+perc_times], color=cmap(count_t), label='Trial '+str(t))
    ax.set_xlabel('Time (s)', fontsize=20)
    ax.set_ylabel(param_sym_label[p], fontsize=20)
    ax.set_title('Change over trial - average across animals')
    ax.legend(frameon=False)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(path_save, 'mean_animals_symmetry_' + param_sym_name[p] + '_perc_trial_quantification.png'), dpi=128)
    plt.savefig(os.path.join(path_save, 'mean_animals_symmetry_' + param_sym_name[p] + '_perc_trial_quantification.svg'), dpi=128)

# #quantification of change within trial - single animals
# for p in range(np.shape(param_sym)[0]):
#     for a in range(len(animal_list_plot)):
#         fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
#         for count_t, t in enumerate(stim_trials):
#             plt.plot(np.arange(bin_size, bin_size*perc_times+1, bin_size),
#                      param_sym_bs[p, a, (t-1)*perc_times:(t-1)*perc_times+perc_times], color=cmap(count_t),
#             label='Trial '+str(t))
#         ax.set_xlabel('Time (s)', fontsize=20)
#         ax.set_ylabel(param_sym_label[p], fontsize=20)
#         ax.set_title('Change over trial - ' + animal_list_plot[a])
#         ax.legend(frameon=False)
#         plt.xticks(fontsize=20)
#         plt.yticks(fontsize=20)
#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_visible(False)
#         plt.savefig(os.path.join(path_save, 'ind_animals_symmetry_' + param_sym_name[p] + '_' + animal_list_plot[a] + '_perc_trial_quantification.png'), dpi=128)
#         plt.savefig(os.path.join(path_save, 'ind_animals_symmetry_' + param_sym_name[p] + '_' + animal_list_plot[a] + '_perc_trial_quantification.svg'), dpi=128)
#     plt.close('all')