import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as st
import math
import online_tracking_class
import locomotion_class
import utils
import pickle

# Set the default font
plt.rcParams['font.family'] = 'Arial'

# Opto
axes_ranges = {'coo': {'FR':[-5, 2], 'HR':[-5, 2], 'FL':[-3, 4], 'HL':[-3, 4]}, 
               'duty_factor': {'FR':[-4, 4], 'HR':[-4, 4], 'FL':[-4, 4], 'HL':[-4, 4]}, 
               'double_support': {'FR':[-5, 8], 'HR':[-3, 11], 'FL':[-8, 5], 'HL':[-11, 3]},
               'phase_st': {'FR':[-10, 10], 'HR':[-15, 15], 'FL':[-10, 10], 'HL':[-10, 10]}}    # split Rfast
              # 'phase_st': {'FR':[0, 100], 'HR':[0, 100], 'FL':[0, 100], 'HL':[0, 100]}}    # split Rfast
#axes_ranges = {'coo': {'FR':[-2, 5], 'HR':[-2, 5], 'FL':[-4, 3], 'HL':[-7, 3]}, 'duty_factor': {'FR':[-2, 8], 'HR':[-2, 8], 'FL':[-5, 2], 'HL':[-5, 2]}, 
#              'double_support': {'FR':[-8, 5], 'HR':[-11, 3], 'FL':[-5, 8], 'HL':[-3, 11]},
 #             'phase_st': {'FR':[-7, 5], 'HR':[-10, 5], 'FL':[-10, 5], 'HL':[-10, 5]}}    # split Lfast

uniform_ranges = 1

# List of paths for each experiment - it is possible to have only one element
experiment_names = ['stance stim', 'swing stim' ]           #'left fast no-stim','left fast perturb']   #'right fast', 'left fast' ]   'split left fast stim',    # 'control'] #         #'trial stim', 'stance stim', swing stim    'chr2'


paths = [
 #'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\tied stance stim\\',
  'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\tied swing stim\\',
 #'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Split belt sessions\\ALL_ANIMALS\\split left fast control\\',
 #'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Split belt sessions\\ALL_ANIMALS\\split left fast swing stim\\'
]



experiment_colors_dict = {'trial stim':'purple', 'stance stim':'orange','swing stim': 'green', 'control':'black', 'chr2': 'cyan',
                          'right fast no-stim': 'gray',     # 'blue', 
                          'left fast no-stim': 'gray', 
                          'right fast stim': 'green', 
                          'left fast stim': 'cyan',
                          'right fast perturb': 'cyan',     #'red', 
                          'left fast perturb': 'lightgreen'}      # stim on: trial stance swing    'trial stim':'purple', 
animal_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']              # Use the default matplotlib colours
animal_colors_dict = {'MC16846': "#FFD700",'MC16848':"#BBF90F",'MC16850': "#15B01A",'MC16851': animal_colors[0], 'MC17319': animal_colors[1],
                      'MC17665': '#CCCCFF','MC17670': '#660033','MC17666': animal_colors[4], 'MC17668': animal_colors[5],'MC17669': animal_colors[6], 
                      'MC19022': animal_colors[7],'MC19082': animal_colors[8],'MC19123': animal_colors[9], 'MC19124': '#FF00FF', 'MC19130': '#00FFFF',
                      'MC19132': '#0000FF','MC19214': '#00FF00', 'MC18737': '#F08080', 'MC19107': '#FA8072', 'VIV41330': animal_colors[2], 
                      'VIV41329': animal_colors[3], 'VIV41375': '#5C62D6', 'VIV41376': '#FF0000', 'VIV41428': '#BC8F8F', 'VIV41429': '#A9932CC',
                      'VIV41430': '#FF4500',
                      #IO fiber control
                      'VIV40958':animal_colors[4], 'VIV41344':animal_colors[5], 'VIV41345':animal_colors[6], 
                      #ChR2
                      'VIV42375': animal_colors[4],'VIV42376': animal_colors[5],'VIV42428': animal_colors[7],'VIV42429': animal_colors[8],
                      'VIV42430': animal_colors[9], 'VIV42906': animal_colors[2], 'VIV42907': animal_colors[3],'VIV42908':animal_colors[4], 'VIV42974':animal_colors[5],
                      'VIV42985':animal_colors[6], 'VIV42992': animal_colors[7],'VIV42987': animal_colors[8],
                      'VIV44771': '#CCCCFF', 'VIV44765': '#00FF00', 'VIV44766': '#FF4500', 'VIV45372': '#BC8F8F', 'VIV45373': '#F08080', 
                      #HGM
                      'MC11231': "#FFD700",'MC11232':"#BBF90F",'MC11234': "#15B01A",'MC11235': animal_colors[0], 'MC24409': animal_colors[1],
                      'MC24410': '#CCCCFF','MC24411': '#660033','MC24412': animal_colors[4], 'MC24413': animal_colors[5],
                      'MC1262': animal_colors[0],'MC1263':  animal_colors[1],'MC1328': animal_colors[2],'MC1329': animal_colors[3],'MC1330':  animal_colors[4],
                      'A1': "#FFD700",'A2':"#BBF90F",'A3': "#15B01A",'A4': '#0000FF','A5': '#00FF00', 'MC1705': '#F08080', 'V1': '#FA8072',
                      'V2': '#5C62D6', 'V3': '#FF0000', 'V4': '#BC8F8F', 'MC1659': '#BC8F8F', 'MC1660': '#FF4500','MC1661': '#CCCCFF','MC1663': '#660033','MC1664': '#00FFFF',
                      }

#included_animal_list = []
#included_animal_list = [ 'MC17319','MC17665','MC17666','MC17668','MC17669','MC17670']

#included_animal_list = ['VIV44771', 'VIV44766', 'VIV45372', 'VIV45373']
#'MC11231','MC11234','MC11235','MC24410','MC24413'] # ChR2 LE
            #['MC1262','MC1263','MC1328','MC1329','MC1330']     # ChR2 HE                #'VIV42906', 'VIV42974', 'VIV42908','VIV42985','VIV42987']  


included_animal_list =  ['MC16848','MC16851', 'MC17319','MC17665','MC17666','MC17669','MC17670', 'MC19082','MC19124', 'MC19214', 
                      'VIV41329', 'VIV41330', 'VIV42375',  'VIV42428', 'VIV42429', 'VIV42430', 'VIV42376', 'MC19107']         # jaws, histo confirmed              'VIV42376' only tied   'MC19107', only tied and Rfast

session = 1
Ntrials = 28 
stim_start = 9
split_start = 9 
stim_duration = 10
split_duration = 10
print_plots = 1
print_plots_multi_session = 1
bs_bool = 1
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']
paw_otrack = 'FR'
paws = ['FR', 'HR', 'FL', 'HL']
otrack_classes =  []
locos = []
paths_save = []

path_index = 0

param_paw_bs_multi = {}
for path in paths:
    to_save_param_mat = False
    # If param_mat_saved exists in the path, load it
    if os.path.exists(path + 'param_mat_saved.pkl'):
        with open(path + 'param_mat_saved.pkl', 'rb') as f:
            param_mat_saved = pickle.load(f)
    else: # Create it
        param_mat_saved = {}
        to_save_param_mat = True

    print("Analysing..........................", path)
    otrack_classes.append(online_tracking_class.otrack_class(path))
    locos.append(locomotion_class.loco_class(path))
    paths_save.append(path + 'grouped output\\individual limbs\\')
    if not os.path.exists(path + 'grouped output\\individual limbs'):
        os.mkdir(path + 'grouped output\\individual limbs')

    for exp in experiment_names:
        if exp in path:
            experiment_name = exp
    

    # GET THE NUMBER OF ANIMALS AND THE SESSION ID
    animal_session_list = locos[path_index].animals_within_session()
    animal_list = []
    for a in range(len(animal_session_list)):
        animal_list.append(animal_session_list[a][0])
    if len(included_animal_list) == 0:              # All animals included
        included_animal_list = animal_list
    included_animals_id = [animal_list.index(i) for i in included_animal_list]
    session_list = []
    for a in range(len(animal_session_list)):
        session_list.append(animal_session_list[a][1])

    
    # FOR EACH SESSION SEPARATE CALCULATION AND PLOT SAVING
    # GAIT PARAMETERS ACROSS TRIALS
    param_paw_name = ['phase_st']         #['phase_st']         #['coo', 'duty_factor', 'double_support']
    param_label = ['Stance phase (%)']    #['Stance phase (°)']            #['Center of\noscillation (mm)', 'Duty factor (%)', '% Double support']
    param_paw = np.zeros((len(param_paw_name), len(animal_list), 4, Ntrials))
    param_paw[:] = np.nan
    fig_animals = {}
    ax_animals = {}
    for count_animal, animal in enumerate(animal_list):     # Loop on animals
        if not animal in param_mat_saved.keys():
            param_mat_saved[animal] = {}
            to_save_param_mat = True
        session = int(session_list[count_animal])
        #TODO: check if this filelist needs to be emptied first!
        filelist = locos[path_index].get_track_files(animal, session)
        for count_p, param in enumerate(param_paw_name):            # Loop on parameters
            if not param in param_mat_saved[animal].keys() or param=='phase_st':
                    param_mat_saved[animal][param] = {}
                    to_save_param_mat = True
            for f in filelist:          # Loop on trials
                print('Analysing trial:', f)
                count_trial = int(f.split('DLC')[0].split('_')[-1])-1      # Get trial number from file name, to spot any missing trial; parameters for remaining ones will stay to NaN
                # if we have param_mat from the saved file, load it, otherwise do analysis
                if count_trial+1 in param_mat_saved[animal][param]:
                    param_mat = param_mat_saved[animal][param][count_trial+1]
                else:
                    [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = locos[path_index].read_h5(f, 0.9, 0)
                    [st_strides_mat, sw_pts_mat] = locos[path_index].get_sw_st_matrices(final_tracks, 1)
                    paws_rel = locos[path_index].get_paws_rel(final_tracks, 'X')
                
                    param_mat = locos[path_index].compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param)
                    # Add to param_mat_saved variable
                    param_mat_saved[animal][param][count_trial+1] = param_mat
                    to_save_param_mat = True

                for count_paw, paw in enumerate(paws):          # Loop on paws
                    if param == 'phase_st':
                        #HL - 3 - as reference
                       # if paw=='FR':
                       #     param_paw[count_p, count_animal, count_paw, count_trial] = (st.circmean(param_mat[3][count_paw], low=-np.pi, high=np.pi, nan_policy='omit'))
                       # else:
                       #     param_paw[count_p, count_animal, count_paw, count_trial] = (st.circmean(param_mat[3][count_paw], low=0, high=2*np.pi, nan_policy='omit'))
                        # Unwrap
                        #param_mat[3][count_paw] = utils.unwrap_with_nans(param_mat[3][count_paw], unit='rad')
                        #param_paw[count_p, count_animal, count_paw, count_trial] = math.degrees(np.nanmean(param_mat[3][count_paw]))
                        #if paw=='FR':
                        #    param_paw[count_p, count_animal, count_paw, count_trial] = math.degrees(st.circmean(param_mat[3][count_paw], low=-np.pi, high=np.pi, nan_policy='omit'))
                        #else:
                        # Check for invalid strides: if more than 50% of the values are NaN, discard the trial
                        if np.sum(np.isnan(param_mat[3][count_paw])) > 0.5*len(param_mat[3][count_paw]):
                            param_paw[count_p, count_animal, count_paw, count_trial] = np.nan
                        else:
                            param_paw[count_p, count_animal, count_paw, count_trial] = (st.circmean(param_mat[3][count_paw], low=0, high=2*np.pi, nan_policy='omit'))
                        
                        # Put between -pi and pi
                        #param_paw[count_p, count_animal, count_paw, count_trial] = np.mod(param_paw[count_p, count_animal, count_paw, count_trial] + np.pi, 2*np.pi) - np.pi            # 

                        #param_paw[count_p, count_animal, count_paw, count_trial] = math.degrees(st.circmean(param_mat[3][count_paw], low=0, high=2*np.pi, nan_policy='omit'))
                        #param_paw[count_p, count_animal, count_paw, count_trial] = (100/(2*math.pi))*st.circmean(param_mat[3][count_paw],low=0, high=2*np.pi, nan_policy='omit')
                    else:
                        param_paw[count_p, count_animal, count_paw, count_trial] = np.nanmean(param_mat[count_paw])
    
      
        #if 'phase_st' in param_paw_name:        # avoid discontinuity along trials
        #    param_ind = param_paw_name.index('phase_st')
        #    # Unwrap each paw stance phase array:
        #    for count_paw in range(len(paws)):
        #        param_paw[param_ind, count_animal, count_paw, :] = utils.unwrap_with_nans(param_paw[param_ind, count_animal, count_paw, :])

        # Plot stance phase for FR and FL paw
        fig_animals[animal_list[count_animal]], ax_animals[animal_list[count_animal]] = plt.subplots(nrows=2, ncols=4, figsize=(14, 10), tight_layout=True)
        stance_phase_index = param_paw_name.index('phase_st')
        ax_animals[animal_list[count_animal]][0, 0].plot(param_paw[stance_phase_index, count_animal, 0, :], color='red', label='FR')
        ax_animals[animal_list[count_animal]][0,0].set_title('Stance phase (rad)')
        ax_animals[animal_list[count_animal]][1, 0].plot(param_paw[stance_phase_index, count_animal, 2, :], color='blue', label='FL')


    # Save the param_mat_saved variable
    if to_save_param_mat:
        with open(path + 'param_mat_saved.pkl', 'wb') as f:
            pickle.dump(param_mat_saved, f)
     

    # BASELINE SUBTRACTION OF PARAMETERS
    if bs_bool:
        param_paw_bs = np.zeros(np.shape(param_paw))
        for p in range(np.shape(param_paw)[0]):
            if param_paw_name[p] != 'phase_st':
                for a in range(np.shape(param_paw)[1]):
                    # Compute baseline and subtract
                    for count_paw in range(4):
                        if stim_start == split_start:
                            bs_paw_mean = np.nanmean(param_paw[p, a, count_paw, :stim_start-1])
                        if stim_start < split_start:
                            bs_paw_mean = np.nanmean(param_paw[p, a, count_paw, stim_start-1:split_start-1])
                        param_paw_bs[p, a, count_paw, :] = param_paw[p, a, count_paw, :] - bs_paw_mean
            else:
                for a in range(np.shape(param_paw)[1]):
                    # Compute baseline and subtract
                    for count_paw in range(4):
                        if stim_start == split_start:
                            bs_paw_mean = st.circmean(param_paw[p, a, count_paw, :stim_start-1], nan_policy='omit')
                        if stim_start < split_start:
                            bs_paw_mean = st.circmean(param_paw[p, a, count_paw, stim_start-1:split_start-1], nan_policy='omit')
                        param_paw_bs[p, a, count_paw, :] = param_paw[p, a, count_paw, :] - bs_paw_mean

                        # Update the single animal and single paw plot
                        if count_paw == 0:
                            ax_animals[animal_list[a]][0,1].plot(param_paw_bs[p, a, count_paw, :], color='red', label='FR')
                            ax_animals[animal_list[a]][0,1].set_title('Stance phase (rad) bs')
                        if count_paw == 2:
                            ax_animals[animal_list[a]][1,1].plot(param_paw_bs[p, a, count_paw, :], color='blue', label='FL')

                        # Put between -pi and pi
                        param_paw_bs[p, a, count_paw, :] = np.mod(param_paw_bs[p, a, count_paw, :] + np.pi, 2*np.pi) - np.pi

                        # Update the single animal and single paw plot
                        if count_paw == 0:
                            ax_animals[animal_list[a]][0,2].plot(param_paw_bs[p, a, count_paw, :], color='red', label='FR')
                            ax_animals[animal_list[a]][0,2].set_title('Stance phase (rad) bs -pi to pi')
                        if count_paw == 2:
                            ax_animals[animal_list[a]][1,2].plot(param_paw_bs[p, a, count_paw, :], color='blue', label='FL')

                        # Remove outliers for each animal
                       # circ_mean_limb = st.circmean(param_paw_bs[p, a, count_paw, :], nan_policy='omit')
                       # circ_mean_limb = np.mod(circ_mean_limb + np.pi, 2*np.pi) - np.pi    # Put back between pi and -pi, otherwise get wrong values around 2*pi
                      #  circ_std_limb = st.circstd(param_paw_bs[p, a, count_paw, :], nan_policy='omit')
                       # z_scores = abs(param_paw_bs[p, a, count_paw, :] - circ_mean_limb) / circ_std_limb
                       # z_threshold = 3
                       # outlier_indices = z_scores > z_threshold                
                       # param_paw_bs[p, a, count_paw][outlier_indices] = np.nan
    else:
        param_paw_bs = param_paw
    

    # TODO: adapt this to the other parameters
    # Convert to %
    param_paw_rad = param_paw
    param_paw_bs = (param_paw_bs/(2*np.pi))*100

    for count_animal, animal in enumerate(animal_list):
        # Update the single animal and single paw plot
        ax_animals[animal_list[count_animal]][0,3].plot(param_paw_bs[stance_phase_index, count_animal, 0, :], color='red', label='FR')
        ax_animals[animal_list[count_animal]][1,3].plot(param_paw_bs[stance_phase_index, count_animal, 2, :], color='blue', label='FL')
        ax_animals[animal_list[count_animal]][0,3].axhline(y = 0, linestyle='--', color='k')
        ax_animals[animal_list[count_animal]][1,3].axhline(y = 0, linestyle='--', color='k')
        ax_animals[animal_list[count_animal]][0,3].set_title('Stance phase (%)')
        fig_animals[animal_list[count_animal]].legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
        fig_animals[animal_list[count_animal]].savefig(paths_save[path_index]  + animal_list[count_animal] + 'stance_phase_processing', dpi=128)

    if any('right' in element for element in experiment_names) and any('left' in element for element in experiment_names) and 'left' in experiment_name:      # If we are comparing left and right we will have them both in experiment names
        param_paw_bs = -param_paw_bs

    param_paw_bs_multi[path] = param_paw_bs
    
    ##### PLOTS #####
    # Linear plot untransformed animals
    for p in range(np.shape(param_paw)[0]):
        for paw in range(len(paws)):
            fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
            rectangle = plt.Rectangle((split_start - 0.5, np.nanmin(param_paw[p, included_animals_id, paw, :].flatten())), split_duration,
                                    np.nanmax(param_paw[p, included_animals_id, paw, :].flatten()) - np.nanmin(param_paw[p, included_animals_id, paw, :].flatten()),
                                    fc='lightblue', alpha=0.3)
            plt.gca().add_patch(rectangle)
            if bs_bool:
                ax.axhline(y = 0, color = 'gray', linestyle = '--', linewidth=0.5)
            ax.axvline(x = stim_start-0.5, color = 'k', linestyle = '-', linewidth=0.5)
            ax.axvline(x = stim_start+stim_duration-0.5, color = 'k', linestyle = '-', linewidth=0.5)
            for a in included_animals_id:   # Loop on included animals            
                plt.plot(np.linspace(1, len(param_paw[p, a, paw, :]), len(param_paw[p, a, paw, :])), param_paw[p, a, paw, :], color= animal_colors_dict[animal_list[a]],
                        label=animal_list[a], linewidth=1)
            ax.set_xlabel('1-min trial', fontsize=34)
            ax.set_ylabel('Stance phase [rad]', fontsize=34)
            plt.xticks(fontsize=28)
            plt.yticks(fontsize=28)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
            if print_plots:
                if not os.path.exists(paths_save[path_index]):
                    os.mkdir(paths_save[path_index])
                plt.savefig(paths_save[path_index] + param_paw_name[p] + paws[paw] + '_ind_animals', dpi=128)
                
    # Linear plot
    for p in range(np.shape(param_paw)[0]):
        for paw in range(len(paws)):
            fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
            rectangle = plt.Rectangle((split_start - 0.5, np.nanmin(param_paw_bs[p, included_animals_id, paw, :].flatten())), split_duration,
                                    np.nanmax(param_paw_bs[p, included_animals_id, paw, :].flatten()) - np.nanmin(param_paw_bs[p, included_animals_id, paw, :].flatten()),
                                    fc='lightblue', alpha=0.3)
            plt.gca().add_patch(rectangle)
            if bs_bool:
                ax.axhline(y = 0, color = 'gray', linestyle = '--', linewidth=0.5)
            ax.axvline(x = stim_start-0.5, color = 'k', linestyle = '-', linewidth=0.5)
            ax.axvline(x = stim_start+stim_duration-0.5, color = 'k', linestyle = '-', linewidth=0.5)
            #plt.hlines(0, 1, param_paw_bs.shape[3], colors='grey', linestyles='--')
            
            for a in included_animals_id:   # Loop on included animals            
                plt.plot(np.linspace(1, len(param_paw_bs[p, a, paw, :]), len(param_paw_bs[p, a, paw, :])), param_paw_bs[p, a, paw, :], color= animal_colors_dict[animal_list[a]],
                        label=animal_list[a], linewidth=1)
            # Add average and SE
            plt.plot(np.linspace(1, param_paw_bs[p,included_animals_id,paw,:].shape[1], param_paw_bs[p,included_animals_id,paw,:].shape[1]), np.nanmean(param_paw_bs[p,included_animals_id,paw,:], axis=0), color=paw_colors[paw], linewidth=3)
            plt.fill_between(np.arange(1, Ntrials+1), np.nanmean(param_paw_bs[p,included_animals_id,paw,:], axis=0)-np.nanstd(param_paw_bs[p,included_animals_id,paw,:], axis=0)/np.sqrt(len(included_animals_id)), np.nanmean(param_paw_bs[p,included_animals_id,paw,:], axis=0)+np.nanstd(param_paw_bs[p,included_animals_id,paw,:], axis=0)/np.sqrt(len(included_animals_id)), color=paw_colors[paw], alpha=0.5)
            #plt.fill_between(np.arange(1, Ntrials+1), np.nanquantile(param_paw_bs[p,included_animals_id,paw,:], q=0.25, axis=0), np.nanquantile(param_paw_bs[p,included_animals_id,paw,:], q=0.75, axis=0), color=paw_colors[paw], alpha=0.5)
            
            ax.set_xlabel('1-min trial', fontsize=34)
            #ax.legend(frameon=False,loc='center left', bbox_to_anchor=(1, 0.5)) 
            ax.set_ylabel(param_label[p], fontsize=34)
            plt.xticks(fontsize=28)
            plt.yticks(fontsize=28)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set(ylim=axes_ranges[param_paw_name[p]][paws[paw]])
            if print_plots:
                if not os.path.exists(paths_save[path_index]):
                    os.mkdir(paths_save[path_index])
                if bs_bool:
                    plt.savefig(paths_save[path_index] + param_paw_name[p] + paws[paw] + '_aritmean_circmeanbs_allcentered0', dpi=128)
                else:
                    plt.savefig(paths_save[path_index] + param_paw_name[p] + paws[paw] + '_aritmean_non_bs_allcentered0', dpi=128)

        # Polar plot for stance phase
        if param_paw_name[p] == 'phase_st':
            fig = plt.figure(figsize=(10, 10), tight_layout=True)
            ax = fig.add_subplot(111, projection='polar')
            for paw in range(4):
                data_mean = st.circmean(param_paw_rad[p,included_animals_id,paw,:], axis=0, nan_policy='omit')
                ax.scatter(data_mean, np.arange(1, Ntrials + 1), c=paw_colors[paw], s=30)
            ax.set_yticks([8.5, 18.5])
            ax.set_yticklabels(['', ''])
            ax.tick_params(axis='both', which='major', labelsize=20)
            plt.savefig(os.path.join(paths_save[path_index], 'phase_st_non_bs_circmean_animals_polar.png'), dpi=256)
        plt.close('all')
    path_index = path_index+1


# PLOT ANIMAL AVERAGE alone 
for p in range(np.shape(param_paw)[0]):
    for paw in range(len(paws)): 
        fig_avg, ax_avg = plt.subplots(figsize=(7, 10), tight_layout=True)
        path_index = 0
        for path in paths: 
            for exp in experiment_names:
                if exp in path:
                    experiment_name = exp
            if uniform_ranges:
                if path_index == 0:
                    axes_ranges[param_paw_name[p]][paws[paw]] = np.divide(axes_ranges[param_paw_name[p]][paws[paw]], 2)
                ax_avg.set(ylim=axes_ranges[param_paw_name[p]][paws[paw]])
                rectangle = plt.Rectangle((split_start - 0.5, axes_ranges[param_paw_name[p]][paws[paw]][0]), split_duration,
                                            axes_ranges[param_paw_name[p]][paws[paw]][1] - axes_ranges[param_paw_name[p]][paws[paw]][0],
                                            fc='lightblue', alpha=0.3)
            else:
                rectangle = plt.Rectangle((split_start - 0.5, np.nanmin(param_paw_bs[p,included_animals_id,paw,:].flatten())), split_duration,
                                            np.nanmax(param_paw_bs[p,included_animals_id,paw,:].flatten()) - np.nanmin(param_paw_bs[p,included_animals_id,paw,:].flatten()),
                                            fc='lightblue', alpha=0.3)
            plt.gca().add_patch(rectangle)
            ax_avg.axvline(x = stim_start-0.5, color = 'k', linestyle = '-', linewidth=0.5)
            ax_avg.axvline(x = stim_start+stim_duration-0.5, color = 'k', linestyle = '-', linewidth=0.5)
            if bs_bool:
                ax_avg.axhline(y = 0, color = 'gray', linestyle = '--', linewidth=0.5)
            #plt.hlines(0, 1, param_paw_bs.shape[3], colors='grey', linestyles='--')
            if experiment_name == 'control':
                exp_color = 'black'
            else:
                exp_color = paw_colors[paw]
            if max(included_animals_id)==param_paw_bs_multi[path].shape[1]:
                included_animals_id = included_animals_id[:-1]
            plt.plot(np.linspace(1, param_paw_bs[p,included_animals_id,paw,:].shape[1], param_paw_bs_multi[path][p,included_animals_id,paw,:].shape[1]), np.nanmean(param_paw_bs_multi[path][p,included_animals_id,paw,:], axis=0), color=exp_color, linewidth=3)
            plt.fill_between(np.arange(1, Ntrials+1), np.nanmean(param_paw_bs_multi[path][p,included_animals_id,paw,:], axis=0)-np.nanstd(param_paw_bs_multi[path][p,included_animals_id,paw,:], axis=0)/np.sqrt(len(included_animals_id)), np.nanmean(param_paw_bs_multi[path][p,included_animals_id,paw,:], axis=0)+np.nanstd(param_paw_bs_multi[path][p,included_animals_id,paw,:], axis=0)/np.sqrt(len(included_animals_id)), color=exp_color, alpha=0.5)
            ax_avg.set_xlabel('Trial', fontsize=28)
            ax_avg.set_ylabel(param_label[p], fontsize=28)          
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            ax_avg.spines['right'].set_visible(False)
            ax_avg.spines['top'].set_visible(False)
            
            if print_plots and path_index==len(paths)-1:
                if not os.path.exists(paths_save[path_index]):
                    os.mkdir(paths_save[path_index])
                if bs_bool:
                    plt.savefig(paths_save[path_index] + param_paw_name[p] + paws[paw] + experiment_names[0] +'_'+experiment_names[-1]+ '_compare_bs_average', dpi=128)
                else:
                    plt.savefig(paths_save[path_index] + param_paw_name[p] + paws[paw] + experiment_names[0] +'_'+experiment_names[-1]+ '_compare_non_bs_average', dpi=128)
            path_index = path_index+1
