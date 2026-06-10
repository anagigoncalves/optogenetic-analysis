import numpy as np
import matplotlib.pyplot as plt
import os
import online_tracking_class

from scipy.stats import linregress
import locomotion_class

# Inputs
laser_event = 'stance'
trials_plot = np.arange(9, 19) #trials with stimulation to check phase of laser
path = 'D:\\AliG\\climbing-opto-treadmill\\Experiments\\singletrial\\20230606 tied stance stim\\'
included_animal_list = ['MC17319','MC17665','MC17666', 'MC17669','MC17670']          #'MC17666', 
Ntrials = 24   
stim_start = 9
split_start = 9    
stim_duration = 8  
split_duration = 8 
shift_catch_trial = False           # True if you want to take strides after one with or without complex spike; False otherwise
subtract_baseline = True

param_sym_name = ['coo', 'step_length', 'double_support']

#axes_ranges = {'coo': [-3, 5], 'step_length': [-5, 12], 'double_support': [-13, 7]}
axes_ranges = {'coo': [-5, 3], 'step_length': [-12, 5], 'double_support': [-7, 13]}
scatter_ranges = {'coo': [-20, 20], 'step_length': [-60, 60], 'double_support': [-100, 100]}

colors = {'stance': 'orange', 'swing': 'green'}
animal_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']              # Use the default matplotlib colours
animal_colors_dict = {'MC16851': animal_colors[0], 'MC17319': animal_colors[1],'MC17665': animal_colors[2],'MC17670': animal_colors[3],'MC17666': animal_colors[4],'MC17668': animal_colors[5],'MC17669': animal_colors[6],
                      'MC19022': animal_colors[7],'MC19082': animal_colors[8],'MC19123': animal_colors[9],'MC19124': '#FF00FF', 'MC19130': '#00FFFF','MC19132': '#0000FF','MC19214': '#00FF00',
                      'MC18737': animal_colors[0], 'MC19107': animal_colors[1]}

otrack_class = online_tracking_class.otrack_class(path)
loco = locomotion_class.loco_class(path)
path_save = path + 'grouped output\\'
if not os.path.exists(path + 'grouped output'):
    os.mkdir(path + 'grouped output')

# GET THE NUMBER OF ANIMALS AND THE SESSION ID
animal_session_list = loco.animals_within_session()
animal_list = []
for a in range(len(animal_session_list)):
    animal_list.append(animal_session_list[a][0])
if len(included_animal_list) == 0:              # All animals included
    included_animal_list = animal_list
included_animals_id = [animal_list.index(i) for i in included_animal_list]
session_list = []
for a in range(len(animal_session_list)):
    session_list.append(animal_session_list[a][1])

param_sym_avg_stim_all_animals = {}
param_sym_avg_no_stim_all_animals = {}
param_sym_avg_all_animals = {}
fig_avg_all = {}
ax_avg_all = {}
scatter0_0 = {} 
scatter0_1 = {} 
scatter1_0 = {}
scatter1_1 = {} 

# Initialize dictionaries with all parameters and plot all all animals
for ps in param_sym_name:
    param_sym_avg_stim_all_animals[ps]=[]
    param_sym_avg_no_stim_all_animals[ps]=[]
    param_sym_avg_all_animals[ps]=[]
    fig_avg_all[ps], ax_avg_all[ps] = plt.subplots(tight_layout=True, figsize=(7,10))
    scatter0_0[ps] = plt.figure()
    scatter1_0[ps] = plt.figure()
    scatter0_1[ps] = plt.figure()
    scatter1_1[ps] = plt.figure()

for count_a, animal in enumerate(included_animal_list):
    trials = otrack_class.get_trials(animal)
    # LOAD PROCESSED DATA
    [otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, laser_on] = otrack_class.load_processed_files(animal)

    # EXTRACT REBOUND TIMES for EACH TRIAL in the stimulation phase
    rebound_times = []          # List of rebound times in each trial - len(rebound_times)=stim_duration
    for trial in range(stim_start,stim_start+stim_duration):
        # Select current trial
        current_trial_laser = laser_on.loc[laser_on['trial'] == trial]
        rebound_times.append(np.array(current_trial_laser['time_off'])*1000)        # [ms]

    # READ OFFLINE PAW EXCURSIONS
    [final_tracks_trials, st_strides_trials, sw_strides_trials] = otrack_class.get_offtrack_paws(loco, animal, np.int64(session_list[count_a]))
    final_tracks_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials,
                                                 'st-sw-st')
    
    # CATCH STRIDE ANALYSIS
    # Compute continuous asymmetry measures in each stride
    
    
   
    trials = np.arange(1, Ntrials+1)
    
    for ps in range(len(param_sym_name)):
        st_strides_trials = []          # For each trial, contains the st_strides_mat with stridesx2x5 matrices for each paw; columns are: st/sw in ms; x(st/sw); y(st/sw); z(st/sw); st idx/sw idx. 2 middle columns for beginning and end of stride
        rebound_strides_trials = []     # For each trial, contains a boolean array saying which strides have rebound and which do not        
        current_param_sym = []
        session = int(session_list[count_a])
        filelist = loco.get_track_files(animal, session)
        

        for count_trial, f in enumerate(filelist):
            [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, 0)
            [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
            paws_rel = loco.get_paws_rel(final_tracks, 'X')
            param_mat = loco.compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param_sym_name[ps])
            current_param_sym.append(param_mat)
            st_strides_trials.append(st_strides_mat)
            # If we are in stim trials, check for strides containing a rebound spike (at laser offset) 
            if count_trial>=stim_start-1 and count_trial<stim_start+stim_duration-1:
                within_interval = (np.array(st_strides_mat[0])[:,0,0][:,np.newaxis] <= rebound_times[count_trial-stim_start+1]) & (rebound_times[count_trial-stim_start+1] <= np.array(st_strides_mat[0])[:,1,0][:,np.newaxis])
                rebound_strides_array = np.any(within_interval, axis=1)         # boolean array with shape = stridesx1
                rebound_strides_trials.append(rebound_strides_array)
        [stride_idx, trial_continuous, param_sym_time, param_sym_values] = loco.param_continuous_sym(current_param_sym, st_strides_trials, trials, 'FR', 'FL', 1, 0)

        sl_time_start = param_sym_time[np.where(np.array(trial_continuous) == stim_start)[0][0]]
        sl_time_duration = param_sym_time[np.where(np.array(trial_continuous) == stim_start)[0][0]]+(loco.trial_time*stim_duration)

        rebound_strides_trials_continuous = np.zeros(param_sym_time.shape,dtype=bool)
        offset_index = np.where(np.array(trial_continuous)==stim_start)[0][0]
        if shift_catch_trial:
            offset_index+=1
        indices = np.where(np.concatenate(rebound_strides_trials))
        rebound_strides_trials_continuous[indices[0]+offset_index]=True
        param_sym_values_stim = np.copy(param_sym_values)
        param_sym_values_stim[np.where(~rebound_strides_trials_continuous)[0]] = np.nan
        param_sym_values_no_stim = np.copy(param_sym_values)
        param_sym_values_no_stim[np.where(rebound_strides_trials_continuous)[0]] = np.nan
        if subtract_baseline:
            bs = np.nanmean(param_sym_values[param_sym_time<sl_time_start])
            param_sym_values_stim = param_sym_values_stim-bs
            param_sym_values_no_stim = param_sym_values_no_stim-bs
            param_sym_values = param_sym_values - bs
        
        negative_indices = np.where(~np.concatenate(rebound_strides_trials))
        # If strides without rebound are preceeded by strides with rebound, we puth them in the matrix of pairs_1_0 (2 rows, first is asymmetry measure at trial with rebound, second at trial without)
        pair1_0_indices = [i-1 for i in negative_indices[0] if (i-1) in indices[0]]
        param_sym_values_pair1_0 = np.vstack((param_sym_values[pair1_0_indices+offset_index], param_sym_values[np.add(pair1_0_indices,1)]))
        pair0_0_indices = [i-1 for i in negative_indices[0] if (i-1) in negative_indices[0]]
        param_sym_values_pair0_0 = np.vstack((param_sym_values[pair0_0_indices+offset_index], param_sym_values[np.add(pair0_0_indices,1)+offset_index]))

        # If strides with rebound are preceeded by strides with rebound, we puth them in the matrix of pairs_1_1 (2 rows, first is asymmetry measure at trial with rebound, second at trial with)
        pair1_1_indices = [i-1 for i in indices[0] if (i-1) in indices[0]]
        param_sym_values_pair1_1 = np.vstack((param_sym_values[pair1_1_indices+offset_index], param_sym_values[np.add(pair1_1_indices,1)]))
        pair0_1_indices = [i-1 for i in indices[0] if (i-1) in negative_indices[0]]
        param_sym_values_pair0_1 = np.vstack((param_sym_values[pair0_1_indices+offset_index], param_sym_values[np.add(pair0_1_indices,1)+offset_index]))

        fig, ax = plt.subplots(tight_layout=True, figsize=(25,10))
        
        ax.plot(param_sym_time, param_sym_values, color='black')
        plt.axvline(sl_time_start)
        plt.axvline(sl_time_duration)
        ax.set_xlabel('time (s)')
        ax.set_title('continuous '+param_sym_name[ps])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel(param_sym_name[ps].replace('_', ' '), fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        if param_sym_name[ps]=='double_support':
            ax.invert_yaxis()
        if not os.path.exists(path_save):
            os.mkdir(path_save)
        plt.savefig(path_save + included_animal_list[count_a] + param_sym_name[ps]+'_continuous_shift'+str(shift_catch_trial), dpi=96)

        fig_split, ax_split = plt.subplots(tight_layout=True, figsize=(25,10))
        ax_split.plot(param_sym_time, param_sym_values_stim, marker='o',linestyle='None', color=colors[laser_event], markersize=4)
        ax_split.plot(param_sym_time, param_sym_values_no_stim, marker='o',linestyle='None', color='gray', markersize=4)
        plt.axvline(sl_time_start)
        plt.axvline(sl_time_duration)
        ax_split.set_xlabel('time (s)')
        ax_split.set_title('continuous '+param_sym_name[ps]+' stim no stim')
        ax_split.spines['right'].set_visible(False)
        ax_split.spines['top'].set_visible(False)
        ax_split.set_ylabel(param_sym_name[ps].replace('_', ' '), fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        if param_sym_name[ps]=='double_support':
            ax_split.invert_yaxis()
        if not os.path.exists(path_save):
            os.mkdir(path_save)
        plt.savefig(path_save + included_animal_list[count_a] + param_sym_name[ps]+'_continuous_stim_no_stim_shift'+str(shift_catch_trial), dpi=96)


        # Averages
        # by trial
        fig_avg, ax_avg = plt.subplots(tight_layout=True, figsize=(7,10))
        param_sym_avg_stim = [np.nanmean(param_sym_values_stim[np.array(trial_continuous) == val]) for val in range(1,Ntrials+1)]
        param_sym_avg_no_stim = [np.nanmean(param_sym_values_no_stim[np.array(trial_continuous) == val]) for val in range(1,Ntrials+1)]
        param_sym_avg = [np.nanmean(param_sym_values[np.array(trial_continuous) == val]) for val in range(1,Ntrials+1)]
        param_sym_std_stim = [np.nanstd(param_sym_values_stim[np.array(trial_continuous) == val]) for val in range(1,Ntrials+1)]
        param_sym_std_no_stim = [np.nanstd(param_sym_values_no_stim[np.array(trial_continuous) == val]) for val in range(1,Ntrials+1)]
        param_sym_std = [np.nanstd(param_sym_values[np.array(trial_continuous) == val]) for val in range(1,Ntrials+1)]
        param_sym_avg_stim_all_animals[param_sym_name[ps]].append(param_sym_avg_stim)
        param_sym_avg_no_stim_all_animals[param_sym_name[ps]].append(param_sym_avg_no_stim)
        param_sym_avg_all_animals[param_sym_name[ps]].append(param_sym_avg)

        #ax_avg.fill_between(list(range(1,Ntrials+1)), 
        #                np.array(param_sym_avg_stim)+np.array(param_sym_std_stim), 
        #                np.array(param_sym_avg_stim)-np.array(param_sym_std_stim), 
        #                facecolor='green', alpha=0.4)
        ax_avg.plot(list(range(1,Ntrials+1)), param_sym_avg_stim, linestyle='-', color=colors[laser_event], linewidth=2)
        ax_avg.plot(list(range(1,Ntrials+1)), param_sym_avg_no_stim, linestyle='-', color='gray', linewidth=2)
        ax_avg.plot(list(range(1,Ntrials+1)), param_sym_avg, linestyle='-', color='black', linewidth=1)
        plt.axvline(stim_start-0.5)
        plt.axvline(stim_start+stim_duration-0.5)
        ax_avg.set_xlabel('trials')
        ax_avg.set_title('avg trials '+param_sym_name[ps]+' stim no stim')
        ax_avg.set_ylabel(param_sym_name[ps].replace('_', ' '), fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        ax_avg.set(ylim=axes_ranges[param_sym_name[ps]])
        if param_sym_name[ps]=='double_support':
            ax_avg.invert_yaxis()
        if not os.path.exists(path_save):
            os.mkdir(path_save)
        plt.savefig(path_save + included_animal_list[count_a] + param_sym_name[ps]+'_avg_trials_stim_no_stim_shift'+str(shift_catch_trial), dpi=96)
 
        # Update the all animal plot with single animal traces
        plt.figure(fig_avg_all[param_sym_name[ps]])
        ax_avg_all[param_sym_name[ps]].plot(list(range(1,Ntrials+1)), param_sym_avg_stim, linestyle='-', color=animal_colors_dict[animal], linewidth=2)
        ax_avg_all[param_sym_name[ps]].plot(list(range(1,Ntrials+1)), param_sym_avg_no_stim, linestyle='-',color=animal_colors_dict[animal], linewidth=1)
        

        # Pairing strides like in Fig. 6 of Medina and Liesberger paper: 0-0 if both subsequent strides without CoSp; 1-0, stride with CoSp followed by one without
        asymmetry_change0_0 = param_sym_values_pair0_0[1,:]-param_sym_values_pair0_0[0,:]
        asymmetry_change1_0 = param_sym_values_pair1_0[1,:]-param_sym_values_pair1_0[0,:]

        plt.figure()
        plt.plot(param_sym_time[pair0_0_indices+offset_index], asymmetry_change0_0, 'bo', label='0-0')
        plt.plot(param_sym_time[pair1_0_indices+offset_index], asymmetry_change1_0, 'ro', label='1-0')
        plt.legend()
        plt.title('Pairs during stim strides/trials '+param_sym_name[ps])
        plt.axvline(sl_time_start)
        plt.axvline(sl_time_duration)
        plt.savefig(path_save + included_animal_list[count_a] + param_sym_name[ps]+'_pairs_shift'+str(shift_catch_trial), dpi=96)


        # Averages of pairs
        fig_avg, ax_avg = plt.subplots(tight_layout=True, figsize=(7,10))
        asymmetry_change0_0_avg = [np.nanmean(asymmetry_change0_0[np.where(np.array(trial_continuous)[pair0_0_indices+offset_index] == val)]) for val in range(stim_start,stim_start+stim_duration+1)]
        asymmetry_change1_0_avg = [np.nanmean(asymmetry_change1_0[np.where(np.array(trial_continuous)[pair0_0_indices+offset_index] == val)]) for val in range(stim_start,stim_start+stim_duration+1)]
        ax_avg.plot(list(range(stim_start,stim_start+stim_duration+1)),asymmetry_change0_0_avg, 'b', label='0-0')
        ax_avg.plot(list(range(stim_start,stim_start+stim_duration+1)),asymmetry_change1_0_avg, 'r', label='1-0')
        plt.legend()
        ax_avg.set_title('Pairs during stim strides/trials averaged '+param_sym_name[ps])
        plt.savefig(path_save + included_animal_list[count_a] + param_sym_name[ps]+'_pairs_avg_shift'+str(shift_catch_trial), dpi=96)

        # Scatterplot of all pairs (Fig. S2 b and d of Medina and Liesberger paper)
        x_bisector = np.linspace(scatter_ranges[param_sym_name[ps]][0], scatter_ranges[param_sym_name[ps]][1], 100)
        y_bisector = x_bisector
        mask = ~np.isnan(param_sym_values_pair0_0[0,:]) & ~np.isnan(param_sym_values_pair0_0[1,:])
        slope, intercept, r, p, se = linregress(param_sym_values_pair0_0[0,:][mask], param_sym_values_pair0_0[1,:][mask])
        
        plt.figure(scatter0_0[param_sym_name[ps]])
        color_scat = np.linspace(0,1,len(param_sym_values_pair0_0[0,:]))
        plt.scatter(param_sym_values_pair0_0[0,:], param_sym_values_pair0_0[1,:], c=color_scat, cmap='winter')
        #plt.plot(param_sym_values_pair0_0[0,:], intercept + slope*param_sym_values_pair0_0[0,:], 'r', label='fitted line')  
        plt.xlabel(param_sym_name[ps]+' first stride (0)')
        plt.ylabel(param_sym_name[ps]+' second stride (0)')
        plt.title('0-0 stride pairs')
        plt.axis('equal')
        plt.xlim(scatter_ranges[param_sym_name[ps]])
        plt.ylim(scatter_ranges[param_sym_name[ps]])
        plt.plot(x_bisector, y_bisector,'k-')
        plt.grid(True, linestyle='--', alpha=0.6)
       # plt.savefig(path_save + included_animal_list[count_a] + param_sym_name[ps]+'_0-0pairs_scatter_shift'+str(shift_catch_trial), dpi=96)
        plt.savefig(path_save + 'all_animals_' + param_sym_name[ps]+'_0-0pairs_scatter_shift'+str(shift_catch_trial), dpi=96)

        mask = ~np.isnan(param_sym_values_pair1_0[0,:]) & ~np.isnan(param_sym_values_pair1_0[1,:])
        slope, intercept, r, p, se = linregress(param_sym_values_pair1_0[0,:][mask], param_sym_values_pair1_0[1,:][mask])
        plt.figure(scatter1_0[param_sym_name[ps]])
        color_scat = np.linspace(0,1,len(param_sym_values_pair1_0[0,:]))
        plt.scatter(param_sym_values_pair1_0[0,:], param_sym_values_pair1_0[1,:], c=color_scat, cmap='winter')
        #plt.plot(param_sym_values_pair1_0[0,:], intercept + slope*param_sym_values_pair1_0[0,:], 'r', label='fitted line')  
        plt.xlabel(param_sym_name[ps]+' first stride (1)')
        plt.ylabel(param_sym_name[ps]+' second stride (0)')
        plt.title('1-0 stride pairs')
        plt.axis('equal')
        plt.xlim(scatter_ranges[param_sym_name[ps]])
        plt.ylim(scatter_ranges[param_sym_name[ps]])
        plt.plot(x_bisector, y_bisector,'k-')
        plt.grid(True, linestyle='--', alpha=0.6)
        #plt.savefig(path_save + included_animal_list[count_a] + param_sym_name[ps]+'_1-0pairs_scatter_shift'+str(shift_catch_trial), dpi=96)
        plt.savefig(path_save + 'all_animals_' + param_sym_name[ps]+'_1-0pairs_scatter_shift'+str(shift_catch_trial), dpi=96)

        mask = ~np.isnan(param_sym_values_pair0_1[0,:]) & ~np.isnan(param_sym_values_pair0_1[1,:])
        slope, intercept, r, p, se = linregress(param_sym_values_pair0_1[0,:][mask], param_sym_values_pair0_1[1,:][mask])
        plt.figure(scatter0_1[param_sym_name[ps]])
        color_scat = np.linspace(0,1,len(param_sym_values_pair0_1[0,:]))
        plt.scatter(param_sym_values_pair0_1[0,:], param_sym_values_pair0_1[1,:], c=color_scat, cmap='winter')
        #plt.plot(param_sym_values_pair0_1[0,:], intercept + slope*param_sym_values_pair0_1[0,:], 'r', label='fitted line')  
        plt.xlabel(param_sym_name[ps]+' first stride (0)')
        plt.ylabel(param_sym_name[ps]+' second stride (1)')
        plt.title('0-1 stride pairs')
        plt.axis('equal')
        plt.xlim(scatter_ranges[param_sym_name[ps]])
        plt.ylim(scatter_ranges[param_sym_name[ps]])
        plt.plot(x_bisector, y_bisector,'k-')
        plt.grid(True, linestyle='--', alpha=0.6)
       # plt.savefig(path_save + included_animal_list[count_a] + param_sym_name[ps]+'_0-1pairs_scatter_shift'+str(shift_catch_trial), dpi=96)
        plt.savefig(path_save + 'all_animals_' + param_sym_name[ps]+'_0-1pairs_scatter_shift'+str(shift_catch_trial), dpi=96)


        mask = ~np.isnan(param_sym_values_pair1_1[0,:]) & ~np.isnan(param_sym_values_pair1_1[1,:])
        slope, intercept, r, p, se = linregress(param_sym_values_pair1_1[0,:][mask], param_sym_values_pair1_1[1,:][mask])
        plt.figure(scatter1_1[param_sym_name[ps]])
        color_scat = np.linspace(0,1,len(param_sym_values_pair1_1[0,:]))
        plt.scatter(param_sym_values_pair1_1[0,:], param_sym_values_pair1_1[1,:], c=color_scat, cmap='winter')
        #plt.plot(param_sym_values_pair1_1[0,:], intercept + slope*param_sym_values_pair1_1[0,:], 'r', label='fitted line') 
        plt.xlabel(param_sym_name[ps]+' first stride (1)')
        plt.ylabel(param_sym_name[ps]+' second stride (1)')
        plt.title('1-1 stride pairs')
        plt.axis('equal')
        plt.xlim(scatter_ranges[param_sym_name[ps]])
        plt.ylim(scatter_ranges[param_sym_name[ps]])
        plt.plot(x_bisector, y_bisector,'k-')
        plt.grid(True, linestyle='--', alpha=0.6)
        #plt.savefig(path_save + included_animal_list[count_a] + param_sym_name[ps]+'_1-1pairs_scatter_shift'+str(shift_catch_trial), dpi=96)
        plt.savefig(path_save + 'all_animals_' +param_sym_name[ps]+'_1-1pairs_scatter_shift'+str(shift_catch_trial), dpi=96)
        print('ciao')



# Plot all animals
for ps in param_sym_avg_all_animals.keys():
    # Update the all animal plot with single animal traces
    plt.figure(fig_avg_all[ps])
    ax_avg_all[ps].plot(list(range(1,Ntrials+1)), np.nanmean(param_sym_avg_stim_all_animals[ps], axis=0), linestyle='-', color=colors[laser_event], linewidth=4)
    ax_avg_all[ps].plot(list(range(1,Ntrials+1)), np.nanmean(param_sym_avg_no_stim_all_animals[ps], axis=0), linestyle='-', color='gray', linewidth=4)
    plt.axvline(stim_start-0.5)
    plt.axvline(stim_start+stim_duration-0.5)
    ax_avg_all[ps].set_xlabel('trials')
    ax_avg_all[ps].set_title('avg trials '+ps+' stim no stim')
    ax_avg_all[ps].set_ylabel(ps.replace('_', ' '), fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax_avg_all[ps].set(ylim=axes_ranges[ps])
    if ps=='double_support':
        ax_avg_all[ps].invert_yaxis()
    if not os.path.exists(path_save):
            os.mkdir(path_save)
    fig_avg_all[ps].savefig(path_save + 'all_animals_' + ps+'_avg_trials_stim_no_stim_shift'+str(shift_catch_trial), dpi=96)

    # Avg and std without single animals
    fig_avg_std, ax_avg_std = plt.subplots(tight_layout=True, figsize=(7,10))
    plt.plot(list(range(1,Ntrials+1)), np.nanmean(param_sym_avg_stim_all_animals[ps], axis=0), linestyle='-', color=colors[laser_event], linewidth=4)
    plt.plot(list(range(1,Ntrials+1)), np.nanmean(param_sym_avg_no_stim_all_animals[ps], axis=0), linestyle='-', color='gray', linewidth=4)
    # Add control SE
    ax_avg_std.fill_between(np.linspace(1, Ntrials, Ntrials), 
            np.nanmean(param_sym_avg_stim_all_animals[ps], axis=0)+np.nanstd(param_sym_avg_stim_all_animals[ps], axis=0)/np.sqrt(len(included_animal_list)), 
            np.nanmean(param_sym_avg_stim_all_animals[ps], axis=0)-np.nanstd(param_sym_avg_stim_all_animals[ps], axis=0)/np.sqrt(len(included_animal_list)), 
            facecolor=colors[laser_event], alpha=0.5)
    ax_avg_std.fill_between(np.linspace(1, Ntrials, Ntrials), 
            np.nanmean(param_sym_avg_no_stim_all_animals[ps], axis=0)+np.nanstd(param_sym_avg_no_stim_all_animals[ps], axis=0)/np.sqrt(len(included_animal_list)), 
            np.nanmean(param_sym_avg_no_stim_all_animals[ps], axis=0)-np.nanstd(param_sym_avg_no_stim_all_animals[ps], axis=0)/np.sqrt(len(included_animal_list)), 
            facecolor='gray', alpha=0.5)
    plt.axvline(stim_start-0.5)
    plt.axvline(stim_start+stim_duration-0.5)
    plt.xlabel('trials')
    plt.title('avg trials '+ps+' stim no stim')
    plt.ylabel(ps.replace('_', ' '), fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax_avg_std.set(ylim=axes_ranges[ps])
    if ps=='double_support':
        ax_avg_std.invert_yaxis()
    if not os.path.exists(path_save):
            os.mkdir(path_save)
    fig_avg_std.savefig(path_save + 'all_animals_avg_std_' + ps+'_avg_trials_stim_no_stim_shift'+str(shift_catch_trial), dpi=96)
