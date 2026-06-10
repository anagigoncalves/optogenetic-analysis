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
merge_RLfast_sessions = 0
# Figure sizes for easy adjustment
FIGSIZE_LC = (7, 8)   # learning curves
FIGSIZE_SP = (5, 3)   # summary scatterplots (reference size for 4 paws)
# Leave empty to keep all paws from the summary order; set labels like ['FF', 'FS'] or ['FR', 'FL'] to filter.
SUMMARY_SCATTER_PAWS = ['FR', 'FL']

# Opto
axes_ranges = {'coo': {'FR':[-3.5, 1.5], 'HR':[-6, 2], 'FL':[-1.5,3.5], 'HL':[-2, 6]}, 
               'duty_factor': {'FR':[-6, 4], 'HR':[-6, 4], 'FL':[-4, 6], 'HL':[-4, 6]}, 
               'double_support': {'FR':[-3, 11], 'HR':[-3, 13], 'FL':[-11, 3], 'HL':[-13, 3]},
               'step_length': {'FR':[-10, 5], 'HR':[-15, 5], 'FL':[-5, 10], 'HL':[-5, 10]},
               'phase_st': {'FR':[-6, 6], 'HR':[-6, 6], 'FL':[-6, 6], 'HL':[-6, 6]}}    # split Rfast
              # 'phase_st': {'FR':[0, 100], 'HR':[0, 100], 'FL':[0, 100], 'HL':[0, 100]}}    # split Rfast
#axes_ranges = {'coo': {'FR':[-2, 5], 'HR':[-2, 5], 'FL':[-4, 3], 'HL':[-7, 3]}, 'duty_factor': {'FR':[-2, 8], 'HR':[-2, 8], 'FL':[-5, 2], 'HL':[-5, 2]}, 
#              'double_support': {'FR':[-8, 5], 'HR':[-11, 3], 'FL':[-5, 8], 'HL':[-3, 11]},
 #             'phase_st': {'FR':[-7, 5], 'HR':[-10, 5], 'FL':[-10, 5], 'HL':[-10, 5]}}    # split Lfast

uniform_ranges = 1
compute_statistics = 1
significance_threshold = 0.05
statistics_test = 'nonparametric'  # 'nonparametric' or 'ttest'
statistics_comparison = 'vs_zero'  # 'between_paths' or 'vs_zero'
stance_phase_reference_mode = 'slow_hind'  # 'slow_hind' or 'ipsi_hind'

# List of paths for each experiment - it is possible to have only one element
experiment_names = ['stance stim', 'swing stim']       #'th200st', 'th100sw' ]  # 'WT', 'contra fast right', 'ipsi fast left']   # 'contra fast right',        #'left fast no-stim','left fast perturb']   #'right fast', 'left fast' ]   'split left fast stim',    # 'control'] #         #'trial stim', 'stance stim', swing stim    'chr2'

data_path = 'D:\\AliG\\climbing-opto-treadmill\\'

paths = [

 #'D:\\AliG\\climbing-opto-treadmill\\WT split-belt learning\\',
 #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\HISTO_CHECKED_ANIMALS_RLinj\\split contra fast right CL-Ali\\',
 #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\HISTO_CHECKED_ANIMALS_RLinj\\split ipsi fast left CL-Ali\\', 
 #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\HISTO_CHECKED_ANIMALS_LATinj\\split ipsi fast CL-Ali\\',
 data_path+'Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\tied stance stim CL-Ali\\',
data_path+'Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\tied swing stim CL-Ali\\',
# data_path+'Experiments ChR2 RT\\LOW expression\\ALL_ANIMALS\\tied th200st IO 50ms CL-Ali\\',
#data_path+'Experiments ChR2 RT\\LOW expression\\ALL_ANIMALS\\tied th100sw IO 50ms CL-Ali\\',
]



experiment_colors_dict = {'trial stim':'purple', 'stance stim':'orange','swing stim': 'green', 'control':'black', 'chr2': 'cyan', 'WT': 'gray',
                          'right fast no-stim': 'gray',     # 'blue', 
                          'left fast no-stim': 'gray', 
                          'right fast stim': 'green', 
                          'left fast stim': 'cyan',
                          'right fast perturb': 'cyan',     #'red', 
                          'left fast perturb': 'lightgreen',
                          'ipsi fast': 'black',
                          'contra fast': 'lightseagreen',
                          'ipsi fast left': 'black',
                          'contra fast right': 'lightseagreen',
                          'th200st': 'darkorange',   #'royalblue',
                          'th100sw': 'green',   #'skyblue' 
                          }      # stim on: trial stance swing    'trial stim':'purple', 
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
                       # extra-zombies
                      # Linj
                      'VIV47094': "#FFD700",'VIV47095':"#BBF90F",'VIV47147': "#0000FF",'VIV47116': animal_colors[0], 'VIV47212': animal_colors[1],
                      'VIV49409': animal_colors[2], 'VIV49410': animal_colors[3], 'VIV49411': animal_colors[4], 'VIV49412': animal_colors[5], 
                      # Rinj
                      'VIV49574':animal_colors[6], 'VIV49931': animal_colors[7],'VIV49933': animal_colors[8],
                      'VIV49934': '#CCCCFF', 'VIV49939': '#00FF00', 'VIV49940': '#FF4500', 'VIV49935': '#BC8F8F', 'VIV49941': '#F08080', 
                      'VIV50033': "#FFD700",'VIV50034':"#BBF90F",'VIV50051': "#15B01A",'VIV50052': '#660033',
                      # WT
                      'MC2166': "#FFD700",'MC2168':"#BBF90F",'MC2585': "#15B01A",'MC2586': animal_colors[0], 'MC2587': animal_colors[1],
                      'MC2588': '#CCCCFF','MC2589': '#660033','MC2590': animal_colors[4], 'MC2591': animal_colors[5],'MC2592': animal_colors[6],
                      }

included_animal_list = []
#included_animal_list = [ 'MC17319','MC17665','MC17666','MC17668','MC17669','MC17670']

#included_animal_list = ['VIV44771', 'VIV44766', 'VIV45372', 'VIV45373']
#'MC11231','MC11234','MC11235','MC24410','MC24413'] # ChR2 LE
            #['MC1262','MC1263','MC1328','MC1329','MC1330']     # ChR2 HE                #'VIV42906', 'VIV42974', 'VIV42908','VIV42985','VIV42987']  

#included_animal_list =  ['VIV42906', 'VIV42908', 'VIV42974', 'VIV42985','VIV42987','VIV44766', 'VIV45372'] 
included_animal_list = ['MC16848', 'MC16851', 'MC17319', 'MC17665', 'MC17666', 'MC17669', 'MC17670', 'MC19082', 'MC19124', 
                       'MC19214', 'VIV41329', 'VIV41330', 'VIV42375', 'VIV42428', 'VIV42429', 'VIV42430', 'VIV42376', 'MC19107']         # jaws, histo confirmed              'VIV42376' only tied   'MC19107', only tied and Rfast

# Extra-zombies histology-checked UNIlateral injections
included_animal_list_Linj = ['VIV47094', 'VIV47095', 'VIV47147', 'VIV47212', 'VIV49409', 'VIV49410', 'VIV49411', 'VIV49412']

included_animal_list_EZ = ['VIV47094', 'VIV47095', 'VIV47147', 'VIV47212', 'VIV49409', 'VIV49410', 'VIV49411', 'VIV49412',
                       'VIV49574', 'VIV49939', 'VIV49940', 'VIV49931', 'VIV49933', 'VIV49934', 'VIV50051']

included_animal_list_EZ_double = ['VIV49935', 'VIV49941', 'VIV50033', 'VIV50034', 'VIV50052']

# Extra-zombie animals info
LexpEZ = ['VIV47094', 'VIV47095', 'VIV47147', 'VIV47212', 'VIV49409', 'VIV49410', 'VIV49411', 'VIV49412']
RexpEZ = ['VIV49574', 'VIV49939', 'VIV49940', 'VIV49931', 'VIV49933', 'VIV49934', 'VIV50051']

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


param_abbreviations = {
    'coo': 'COO',
    'duty_factor': 'DF',
    'double_support': 'DS',
    'step_length': 'SL',
    'phase_st': 'phase',
}


def get_front_paw_y_limits(param_name):
    front_ranges = [axes_ranges[param_name]['FR'], axes_ranges[param_name]['FL']]
    return [min(axis_range[0] for axis_range in front_ranges), max(axis_range[1] for axis_range in front_ranges)]


otrack_classes =  []
locos = []
paths_save = []

path_index = 0

animal_list_multi = {}
experiment_name_multi = {}
included_animals_multi = {}
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

    pixel_to_mm = 1/3.3            # real-time setup
    if 'WT' in path:
        pixel_to_mm = 1/1.98          # Jovin's   1/1.955   # Dana's setup                         
        included_animal_list = []
    elif 'LATinj' in path:
        included_animal_list = included_animal_list_EZ
        merge_RLfast_sessions = 0
    elif 'RLinj' in path:
        included_animal_list = included_animal_list_EZ_double
        merge_RLfast_sessions = 1             # This will merge the right and left fast sessions for the same animal, that should be element 2nd and 3rd in the list of paths
    elif 'Linj' in path:
        included_animal_list = included_animal_list_Linj
        merge_RLfast_sessions = 0
        
    print("Analysing..........................", path)
    otrack_classes.append(online_tracking_class.otrack_class(path))
    locos.append(locomotion_class.loco_class(path, pixel_to_mm))
    if 'WT' in path:
        paths_save.append(path + 'grouped output temp - individual limbs\\')
        if not os.path.exists(path + 'grouped output temp - individual limbs'):
            os.mkdir(path + 'grouped output temp - individual limbs')
    else:
        paths_save.append(path + 'grouped output temp - individual limbs\\')
        if not os.path.exists(path + 'grouped output temp - individual limbs'):
            os.mkdir(path + 'grouped output temp - individual limbs')

    for exp in experiment_names:
        if exp in path:
            experiment_name = exp
    experiment_name_multi[path] = experiment_name
    paw_plot_labels, paw_file_labels = utils.get_paw_plot_labels(experiment_name, experiment_names, paws)
    

    # GET THE NUMBER OF ANIMALS AND THE SESSION ID
    animal_session_list = locos[path_index].animals_within_session()
    animal_list = []
    for a in range(len(animal_session_list)):
        animal_list.append(animal_session_list[a][0])
    if len(included_animal_list) == 0:              # All animals included
        included_animal_list = animal_list
    included_animals_id = [animal_list.index(i) for i in included_animal_list]
    included_animals_multi[path] = [animal_list[i] for i in included_animals_id]
    session_list = []
    for a in range(len(animal_session_list)):
        session_list.append(animal_session_list[a][1])

    animal_list_multi[path] = animal_list

    # FOR EACH SESSION SEPARATE CALCULATION AND PLOT SAVING
    # GAIT PARAMETERS ACROSS TRIALS
    param_paw_name = ['coo', 'duty_factor', 'double_support','step_length','phase_st']                #['coo', 'duty_factor', 'double_support']
    param_label = ['Center of\noscillation (mm)', 'Duty factor (%)', '% Double support', 'Step length (mm)', 'Stance phase (%)']    #['Stance phase (°)']            #['Center of\noscillation (mm)', 'Duty factor (%)', '% Double support']
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
                        ref_paw = utils.get_stance_phase_reference_paw(
                            experiment_name,
                            animal,
                            stance_phase_reference_mode,
                            LexpEZ,
                            RexpEZ,
                            included_animal_list_EZ_double,
                        )
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
                        if np.sum(np.isnan(param_mat[ref_paw][count_paw])) > 0.5*len(param_mat[ref_paw][count_paw]):
                            param_paw[count_p, count_animal, count_paw, count_trial] = np.nan
                        else:
                            param_paw[count_p, count_animal, count_paw, count_trial] = (st.circmean(param_mat[ref_paw][count_paw], low=0, high=2*np.pi, nan_policy='omit'))
                        
                        # Put between -pi and pi
                        #param_paw[count_p, count_animal, count_paw, count_trial] = np.mod(param_paw[count_p, count_animal, count_paw, count_trial] + np.pi, 2*np.pi) - np.pi            # 

                        #param_paw[count_p, count_animal, count_paw, count_trial] = math.degrees(st.circmean(param_mat[3][count_paw], low=0, high=2*np.pi, nan_policy='omit'))
                        #param_paw[count_p, count_animal, count_paw, count_trial] = (100/(2*math.pi))*st.circmean(param_mat[3][count_paw],low=0, high=2*np.pi, nan_policy='omit')
                    else:
                        param_paw[count_p, count_animal, count_paw, count_trial] = np.nanmean(param_mat[count_paw])
        
        if ('contra' in experiment_name and animal in RexpEZ) or ('ipsi' in experiment_name and animal in LexpEZ):  # To take the proper fast limb   
            print('Swapping limbs for animal ' + animal + ' in ' + experiment_name + ' experiment!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            front_fast_limb = param_paw[:,count_animal,2,:].copy()     # FL
            front_slow_limb = param_paw[:,count_animal,0,:].copy()     # FR
            hind_fast_limb = param_paw[:,count_animal,3,:].copy()      # HL
            hind_slow_limb = param_paw[:,count_animal,1,:].copy()     # HR
            param_paw[:,count_animal,0,:] = front_fast_limb
            param_paw[:,count_animal,1,:] = hind_fast_limb
            param_paw[:,count_animal,2,:] = front_slow_limb
            param_paw[:,count_animal,3,:] = hind_slow_limb 
        

        #if 'phase_st' in param_paw_name:        # avoid discontinuity along trials
        #    param_ind = param_paw_name.index('phase_st')
        #    # Unwrap each paw stance phase array:
        #    for count_paw in range(len(paws)):
        #        param_paw[param_ind, count_animal, count_paw, :] = utils.unwrap_with_nans(param_paw[param_ind, count_animal, count_paw, :])

        # Plot stance phase for FR and FL paw
        fig_animals[animal_list[count_animal]], ax_animals[animal_list[count_animal]] = plt.subplots(nrows=2, ncols=4, figsize=(14, 10), tight_layout=True)
        stance_phase_index = param_paw_name.index('phase_st')
        ax_animals[animal_list[count_animal]][0, 0].plot(param_paw[stance_phase_index, count_animal, 0, :], color='red', label=paw_plot_labels[0])
        ax_animals[animal_list[count_animal]][0,0].set_title('Stance phase (rad)')
        ax_animals[animal_list[count_animal]][1, 0].plot(param_paw[stance_phase_index, count_animal, 2, :], color='blue', label=paw_plot_labels[2])


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
                            ax_animals[animal_list[a]][0,1].plot(param_paw_bs[p, a, count_paw, :], color='red', label=paw_plot_labels[0])
                            ax_animals[animal_list[a]][0,1].set_title('Stance phase (rad) bs')
                        if count_paw == 2:
                            ax_animals[animal_list[a]][1,1].plot(param_paw_bs[p, a, count_paw, :], color='blue', label=paw_plot_labels[2])

                        # Put between -pi and pi
                        param_paw_bs[p, a, count_paw, :] = np.mod(param_paw_bs[p, a, count_paw, :] + np.pi, 2*np.pi) - np.pi

                        # Update the single animal and single paw plot
                        if count_paw == 0:
                            ax_animals[animal_list[a]][0,2].plot(param_paw_bs[p, a, count_paw, :], color='red', label=paw_plot_labels[0])
                            ax_animals[animal_list[a]][0,2].set_title('Stance phase (rad) bs -pi to pi')
                        if count_paw == 2:
                            ax_animals[animal_list[a]][1,2].plot(param_paw_bs[p, a, count_paw, :], color='blue', label=paw_plot_labels[2])

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
    

    # Convert to % for stance phase
    param_paw_rad = param_paw
    param_paw_bs[stance_phase_index,:,:] = (param_paw_bs[stance_phase_index,:,:]/(2*np.pi))*100

    for count_animal, animal in enumerate(animal_list):
        # Update the single animal and single paw plot
        ax_animals[animal_list[count_animal]][0,3].plot(param_paw_bs[stance_phase_index, count_animal, 0, :], color='red', label=paw_plot_labels[0])
        ax_animals[animal_list[count_animal]][1,3].plot(param_paw_bs[stance_phase_index, count_animal, 2, :], color='blue', label=paw_plot_labels[2])
        ax_animals[animal_list[count_animal]][0,3].axhline(y = 0, linestyle='--', color='k')
        ax_animals[animal_list[count_animal]][1,3].axhline(y = 0, linestyle='--', color='k')
        ax_animals[animal_list[count_animal]][0,3].set_title('Stance phase (%)')
        fig_animals[animal_list[count_animal]].legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
        utils.save_figure_multi_format(
            fig_animals[animal_list[count_animal]],
            paths_save[path_index],
            utils.get_phase_st_output_filename(animal_list[count_animal] + 'stance_phase_processing', stance_phase_reference_mode),
            dpi=128,
        )

    if any('right' in element for element in experiment_names) and any('left' in element for element in experiment_names) and 'left' in experiment_name:     
         # If we are comparing left and right we will have them both in experiment names
         # We need to merge fast and slow limbs, so in the left fast experiment, the fast limb is FL (index 2) that will go into position 0 
         # (index of FR - which we assume is also the position of the fast limb)
        print('Merging left and right fast limbs!!!!!!!!!!!!!!!!!! ', experiment_name, ' ', path)
        front_fast_limb = param_paw_bs[:,:,2,:].copy()     # FL
        front_slow_limb = param_paw_bs[:,:,0,:].copy()     # FR
        hind_fast_limb = param_paw_bs[:,:,3,:].copy()      # HL
        hind_slow_limb = param_paw_bs[:,:,1,:].copy()     # HR
        param_paw_bs[:,:,0,:] = front_fast_limb
        param_paw_bs[:,:,1,:] = hind_fast_limb
        param_paw_bs[:,:,2,:] = front_slow_limb
        param_paw_bs[:,:,3,:] = hind_slow_limb

    param_paw_bs_multi[path] = param_paw_bs
    
    ##### PLOTS #####
    # Linear plot untransformed animals
    for p in range(np.shape(param_paw)[0]):
        for paw in range(len(paws)):
            fig, ax = plt.subplots(figsize=FIGSIZE_LC, tight_layout=True)
            rectangle = plt.Rectangle((split_start - 0.5, np.nanmin(param_paw[p, included_animals_id, paw, :].flatten())), split_duration,
                                    np.nanmax(param_paw[p, included_animals_id, paw, :].flatten()) - np.nanmin(param_paw[p, included_animals_id, paw, :].flatten()),
                                    fc='lightgray', alpha=0.3)
            plt.gca().add_patch(rectangle)
            if bs_bool:
                ax.axhline(y = 0, color = 'gray', linestyle = '--', linewidth=0.5)
            ax.axvline(x = stim_start-0.5, color = 'k', linestyle = '-', linewidth=0.5)
            ax.axvline(x = stim_start+stim_duration-0.5, color = 'k', linestyle = '-', linewidth=0.5)
            for a in included_animals_id:   # Loop on included animals            
                plt.plot(np.linspace(1, len(param_paw[p, a, paw, :]), len(param_paw[p, a, paw, :])), param_paw[p, a, paw, :], color= animal_colors_dict[animal_list[a]],
                        label=animal_list[a], linewidth=1)
            ax.set_xlabel('1-min trial', fontsize=34)
            ax.set_ylabel(param_paw_name[p], fontsize=34)
            ax.set_title(paw_plot_labels[paw], fontsize=28)
            plt.xticks(fontsize=28)
            plt.yticks(fontsize=28)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
            if print_plots:
                output_name = param_paw_name[p] + '_' + paw_file_labels[paw] + '_ind_animals'
                if param_paw_name[p] == 'phase_st':
                    output_name = utils.get_phase_st_output_filename(output_name, stance_phase_reference_mode)
                utils.save_figure_multi_format(fig, paths_save[path_index], output_name, dpi=128)
                
    # Linear plot
    for p in range(np.shape(param_paw)[0]):
        for paw in range(len(paws)):
            fig, ax = plt.subplots(figsize=FIGSIZE_LC, tight_layout=True)
            rectangle = plt.Rectangle((split_start - 0.5, np.nanmin(param_paw_bs[p, included_animals_id, paw, :].flatten())), split_duration,
                                    np.nanmax(param_paw_bs[p, included_animals_id, paw, :].flatten()) - np.nanmin(param_paw_bs[p, included_animals_id, paw, :].flatten()),
                                    fc='lightgray', alpha=0.3)
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
            ax.set_title(paw_plot_labels[paw], fontsize=28)
            plt.xticks(fontsize=28)
            plt.yticks(fontsize=28)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set(ylim=axes_ranges[param_paw_name[p]][paws[paw]])
            if print_plots:
                if bs_bool:
                    if param_paw_name[p] == 'phase_st':
                        utils.save_figure_multi_format(
                            fig,
                            paths_save[path_index],
                            utils.get_phase_st_output_filename(param_paw_name[p] + '_' + paw_file_labels[paw] + '_aritmean_circmeanbs_allcentered0', stance_phase_reference_mode),
                            dpi=128,
                        )
                    else:
                        utils.save_figure_multi_format(fig, paths_save[path_index], param_paw_name[p] + '_' + paw_file_labels[paw] + '_bs', dpi=128)
                else:
                    if param_paw_name[p] == 'phase_st':
                        utils.save_figure_multi_format(
                            fig,
                            paths_save[path_index],
                            utils.get_phase_st_output_filename(param_paw_name[p] + '_' + paw_file_labels[paw] + '_aritmean_non_bs_allcentered0', stance_phase_reference_mode),
                            dpi=128,
                        )
                    else:
                        utils.save_figure_multi_format(fig, paths_save[path_index], param_paw_name[p] + '_' + paw_file_labels[paw] + '_non_bs', dpi=128)

        front_paws_fig = utils.plot_front_paws_average(
            param_paw_bs[p],
            included_animals_id,
            param_label[p],
            f'{experiment_name}: FR and FL',
            FIGSIZE_LC,
            split_start,
            split_duration,
            stim_start,
            stim_duration,
            bs_bool,
            Ntrials,
            paw_colors,
            paws,
            y_limits=get_front_paw_y_limits(param_paw_name[p]) if uniform_ranges else None,
        )
        if print_plots:
            if bs_bool:
                output_name = param_paw_name[p] + '_FR_FL_bs_average'
            else:
                output_name = param_paw_name[p] + '_FR_FL_non_bs_average'
            if param_paw_name[p] == 'phase_st':
                output_name = utils.get_phase_st_output_filename(output_name, stance_phase_reference_mode)
            utils.save_figure_multi_format(front_paws_fig, paths_save[path_index], output_name, dpi=128)
        plt.close(front_paws_fig)

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
            utils.save_figure_multi_format(
                fig,
                paths_save[path_index],
                utils.get_phase_st_output_filename('phase_st_non_bs_circmean_animals_polar.png', stance_phase_reference_mode),
                dpi=256,
            )
        plt.close('all')
    path_index = path_index+1

# Close all figures to free up memory
plt.close('all')
# PLOT ANIMAL AVERAGE alone 
if merge_RLfast_sessions:
    print('Merging RLinj sessions!!!!!!!!!!!!!!!!!!')
    # Get the path keys (should be 0, 1, 2)
    paths_keys = list(param_paw_bs_multi.keys())
    
    # For each parameter, concatenate data from paths[2] into paths[1]
    merged = []
    for p in range(np.shape(param_paw_bs)[0]):
        merged.append(np.concatenate((param_paw_bs_multi[paths_keys[1]][p], 
                                    param_paw_bs_multi[paths_keys[2]][p]), axis=0))         
        
    del param_paw_bs_multi[paths_keys[1]]
    # Delete the third element
    del param_paw_bs_multi[paths_keys[2]]
    # Add the merged data back to the dictionary with the key of the second path
    param_paw_bs_multi[paths_keys[1]] = np.array(merged)
    included_animal_list_EZ_merged = included_animal_list_EZ_double + included_animal_list_EZ_double
    included_animals_id_EZ_merged = included_animals_id + [i + len(included_animals_id) for i in included_animals_id]
    animal_list_multi[paths_keys[1]] = animal_list + animal_list    
    included_animals_multi[paths_keys[1]] = included_animal_list_EZ_merged
    experiment_colors_dict['contra fast right'] = 'darkblue'
for p in range(np.shape(param_paw)[0]):
    for paw in range(len(paws)): 
        fig_avg, ax_avg = plt.subplots(figsize=FIGSIZE_LC, tight_layout=True)
        path_index = 0
        for path in param_paw_bs_multi.keys():
            if 'WT' in path:
                included_animal_list = []
            elif 'LATinj' in path:
                included_animal_list = included_animal_list_EZ
                included_animals_id = [animal_list_multi[path].index(i) for i in included_animal_list]
            elif 'RLinj' in path:
                included_animal_list = included_animal_list_EZ_merged
                included_animals_id = included_animals_id_EZ_merged
            elif 'Linj' in path:
                included_animal_list = included_animal_list_Linj
                included_animals_id = [animal_list_multi[path].index(i) for i in included_animal_list]
            if len(included_animal_list) == 0:              # All animals included
                included_animal_list = animal_list_multi[path]
                included_animals_id = [animal_list_multi[path].index(i) for i in included_animal_list]
                
            for exp in experiment_names:
                if exp in path:
                    experiment_name = exp
            paw_plot_labels, paw_file_labels = utils.get_paw_plot_labels(experiment_name, experiment_names, paws)
            if uniform_ranges:
                #if path_index == 0:
                #    axes_ranges[param_paw_name[p]][paws[paw]] = np.divide(axes_ranges[param_paw_name[p]][paws[paw]], 2)
                ax_avg.set(ylim=axes_ranges[param_paw_name[p]][paws[paw]])
                rectangle = plt.Rectangle((split_start - 0.5, axes_ranges[param_paw_name[p]][paws[paw]][0]), split_duration,
                                            axes_ranges[param_paw_name[p]][paws[paw]][1] - axes_ranges[param_paw_name[p]][paws[paw]][0],
                                            fc='lightgray', alpha=0.3)
            else:
                rectangle = plt.Rectangle((split_start - 0.5, np.nanmin(param_paw_bs[p,included_animals_id,paw,:].flatten())), split_duration,
                                            np.nanmax(param_paw_bs[p,included_animals_id,paw,:].flatten()) - np.nanmin(param_paw_bs[p,included_animals_id,paw,:].flatten()),
                                            fc='lightgray', alpha=0.3)
            plt.gca().add_patch(rectangle)
            ax_avg.axvline(x = stim_start-0.5, color = 'k', linestyle = '-', linewidth=0.5)
            ax_avg.axvline(x = stim_start+stim_duration-0.5, color = 'k', linestyle = '-', linewidth=0.5)
            if bs_bool:
                ax_avg.axhline(y = 0, color = 'gray', linestyle = '--', linewidth=0.5)
            #plt.hlines(0, 1, param_paw_bs.shape[3], colors='grey', linestyles='--')
            exp_color = utils.get_path_color(experiment_name, path_index, experiment_colors_dict)
            plot_animals_id = included_animals_id.copy()
            if max(plot_animals_id)==param_paw_bs_multi[path].shape[1]:
                plot_animals_id = plot_animals_id[:-1]
            plt.plot(np.linspace(1, param_paw_bs_multi[path][p,plot_animals_id,paw,:].shape[1], param_paw_bs_multi[path][p,plot_animals_id,paw,:].shape[1]), np.nanmean(param_paw_bs_multi[path][p,plot_animals_id,paw,:], axis=0), color=exp_color, linewidth=3)
            plt.fill_between(np.arange(1, Ntrials+1), np.nanmean(param_paw_bs_multi[path][p,plot_animals_id,paw,:], axis=0)-np.nanstd(param_paw_bs_multi[path][p,plot_animals_id,paw,:], axis=0)/np.sqrt(len(plot_animals_id)), np.nanmean(param_paw_bs_multi[path][p,plot_animals_id,paw,:], axis=0)+np.nanstd(param_paw_bs_multi[path][p,plot_animals_id,paw,:], axis=0)/np.sqrt(len(plot_animals_id)), color=exp_color, alpha=0.5)
            ax_avg.set_xlabel('Trial', fontsize=28)
            ax_avg.set_ylabel(param_label[p], fontsize=28)          
            ax_avg.set_title(paw_plot_labels[paw], fontsize=24)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            ax_avg.spines['right'].set_visible(False)
            ax_avg.spines['top'].set_visible(False)
            
            if print_plots and ((path_index==len(paths)-2 and merge_RLfast_sessions) or (path_index==len(paths)-1 and not merge_RLfast_sessions)):
                if bs_bool:
                    output_name = param_paw_name[p] + '_' + paw_file_labels[paw] + '_' + experiment_names[0] +'_'+experiment_names[-1]+ '_compare_bs_average'
                else:
                    output_name = param_paw_name[p] + '_' + paw_file_labels[paw] + '_' + experiment_names[0] +'_'+experiment_names[-1]+ '_compare_non_bs_average'
                if param_paw_name[p] == 'phase_st':
                    output_name = utils.get_phase_st_output_filename(output_name, stance_phase_reference_mode)
                utils.save_figure_multi_format(fig_avg, paths_save[path_index], output_name, dpi=128)
                print("Saved figures in ", paths_save[path_index]) 
            
            if path_index == 1:
                for a in included_animals_id:   # Loop on included animals
                    plt.plot(np.linspace(1, len(param_paw_bs_multi[path][p, a, paw, :]), len(param_paw_bs_multi[path][p, a, paw, :])), param_paw_bs_multi[path][p, a, paw, :], 
                    
                    color= animal_colors_dict[animal_list_multi[path][a]],
                            linewidth=1)
                # Add average and SE
                plt.plot(np.linspace(1, param_paw_bs_multi[path][p,plot_animals_id,paw,:].shape[1], param_paw_bs_multi[path][p,plot_animals_id,paw,:].shape[1]), np.nanmean(param_paw_bs_multi[path][p,plot_animals_id,paw,:], axis=0), color=exp_color, linewidth=3)
                plt.fill_between(np.arange(1, Ntrials+1), np.nanmean(param_paw_bs_multi[path][p,plot_animals_id,paw,:], axis=0)-np.nanstd(param_paw_bs_multi[path][p,plot_animals_id,paw,:], axis=0)/np.sqrt(len(plot_animals_id)), np.nanmean(param_paw_bs_multi[path][p,plot_animals_id,paw,:], axis=0)+np.nanstd(param_paw_bs_multi[path][p,plot_animals_id,paw,:], axis=0)/np.sqrt(len(plot_animals_id)), color=exp_color, alpha=0.5)
                ax_avg.set_xlabel('Trial', fontsize=28)
                ax_avg.set_ylabel(param_label[p], fontsize=28) 
                output_name = param_paw_name[p] + '_' + paw_file_labels[paw] + '_' + experiment_name + '_merged_with_ind_animals'
                if param_paw_name[p] == 'phase_st':
                    output_name = utils.get_phase_st_output_filename(output_name, stance_phase_reference_mode)
                utils.save_figure_multi_format(fig_avg, paths_save[path_index], output_name, dpi=128)
            
            path_index = path_index+1

analysis_paths = list(param_paw_bs_multi.keys())
summary_limb_order, summary_limb_labels = utils.get_summary_limb_order(experiment_names, paws)
if SUMMARY_SCATTER_PAWS:
    normalized_summary_scatter_paws = {utils.normalize_summary_scatter_paw_label(paw_label) for paw_label in SUMMARY_SCATTER_PAWS}
    selected_limb_positions = [
        limb_pos for limb_pos, limb_label in enumerate(summary_limb_labels)
        if utils.normalize_summary_scatter_paw_label(limb_label) in normalized_summary_scatter_paws
    ]
    if len(selected_limb_positions) == 0:
        raise ValueError(
            f'No paws from SUMMARY_SCATTER_PAWS={SUMMARY_SCATTER_PAWS} are available in summary_limb_labels={summary_limb_labels}. '
            f'Use one of {summary_limb_labels} or the aliases FR/FL/HR/HL.'
        )
    summary_limb_order_filtered = [summary_limb_order[limb_pos] for limb_pos in selected_limb_positions]
    summary_limb_labels_filtered = [summary_limb_labels[limb_pos] for limb_pos in selected_limb_positions]
else:
    summary_limb_order_filtered = summary_limb_order
    summary_limb_labels_filtered = summary_limb_labels

analysis_display_names = [utils.get_experiment_name_for_path(path, experiment_names) for path in analysis_paths]
analysis_path_colors = [utils.get_path_color(analysis_display_names[idx], idx, experiment_colors_dict) for idx in range(len(analysis_paths))]
analysis_has_wt = any(experiment_name == 'WT' for experiment_name in analysis_display_names)
analysis_summary_colors = []
for idx, path in enumerate(analysis_paths):
    experiment_name = analysis_display_names[idx]
    if experiment_name == 'WT':
        analysis_summary_colors.append(['gray'] * len(summary_limb_labels_filtered))
    else:
        analysis_summary_colors.append([paw_colors[limb_idx] for limb_idx in summary_limb_order_filtered])

compare_output_index = len(paths) - 2 if merge_RLfast_sessions else len(paths) - 1

if compute_statistics:
    stats_file = os.path.join(
        paths_save[compare_output_index],
        utils.add_output_suffix(
            f'limb_learning_statistics_{statistics_test}_{statistics_comparison}.txt',
            utils.get_stance_phase_reference_suffix(stance_phase_reference_mode),
        ),
    )
    with open(stats_file, 'w') as stats_f:
        stats_f.write('='*60 + '\n')
        stats_f.write('Per-limb learning statistics\n')
        stats_f.write(f'Statistical test family: {statistics_test}\n')
        stats_f.write(f'Statistics comparison: {statistics_comparison}\n')
        stats_f.write(f'Stance phase reference mode: {stance_phase_reference_mode}\n')
        stats_f.write(f'Significance threshold: {significance_threshold}\n')
        stats_f.write(f'Paths: {analysis_display_names}\n')
        stats_f.write('='*60 + '\n\n')

        for p, param_name in enumerate(param_paw_name):
            learning_by_path = []
            aftereffect_by_path = []
            analysis_names_by_path = []

            for path in analysis_paths:
                animal_names = included_animals_multi[path]
                animal_ids = [animal_list_multi[path].index(name) for name in animal_names]
                current_values = param_paw_bs_multi[path][p, animal_ids, :, :]
                learning_by_limb = []
                aftereffect_by_limb = []
                for limb_idx in summary_limb_order_filtered:
                    initial_error = current_values[:, limb_idx, split_start-1]
                    learning = np.nanmean(current_values[:, limb_idx, split_start+split_duration-3:split_start+split_duration-1], axis=1) - initial_error
                    # Aftereffect: average of first two trials after split ends (trials 19 and 20 when split_start=9, split_duration=10)
                    aftereffect = np.nanmean(current_values[:, limb_idx, split_start+split_duration-1:split_start+split_duration+1], axis=1)
                    learning_by_limb.append(learning)
                    aftereffect_by_limb.append(aftereffect)
                learning_by_path.append(learning_by_limb)
                aftereffect_by_path.append(aftereffect_by_limb)
                analysis_names_by_path.append(animal_names)

            learning_pvalues = []
            aftereffect_pvalues = []
            stat_test_names = []
            if statistics_comparison == 'between_paths':
                for comparison_idx in range(1, len(analysis_paths)):
                    comparison_learning = []
                    comparison_aftereffect = []
                    comparison_tests = []
                    for limb_pos, limb_label in enumerate(summary_limb_labels_filtered):
                        learning_pvalue, test_name = utils.compare_metric_groups(
                            learning_by_path[0][limb_pos],
                            learning_by_path[comparison_idx][limb_pos],
                            statistics_test,
                            reference_names=analysis_names_by_path[0],
                            comparison_names=analysis_names_by_path[comparison_idx],
                        )
                        aftereffect_pvalue, _ = utils.compare_metric_groups(
                            aftereffect_by_path[0][limb_pos],
                            aftereffect_by_path[comparison_idx][limb_pos],
                            statistics_test,
                            reference_names=analysis_names_by_path[0],
                            comparison_names=analysis_names_by_path[comparison_idx],
                        )
                        comparison_learning.append(learning_pvalue)
                        comparison_aftereffect.append(aftereffect_pvalue)
                        comparison_tests.append(test_name)
                    learning_pvalues.append(comparison_learning)
                    aftereffect_pvalues.append(comparison_aftereffect)
                    stat_test_names.append(comparison_tests)
            elif statistics_comparison == 'vs_zero':
                for path_idx in range(len(analysis_paths)):
                    path_learning = []
                    path_aftereffect = []
                    path_tests = []
                    for limb_pos, limb_label in enumerate(summary_limb_labels_filtered):
                        learning_pvalue, test_name = utils.compare_metric_against_zero(
                            learning_by_path[path_idx][limb_pos],
                            statistics_test,
                        )
                        aftereffect_pvalue, _ = utils.compare_metric_against_zero(
                            aftereffect_by_path[path_idx][limb_pos],
                            statistics_test,
                        )
                        path_learning.append(learning_pvalue)
                        path_aftereffect.append(aftereffect_pvalue)
                        path_tests.append(test_name)
                    learning_pvalues.append(path_learning)
                    aftereffect_pvalues.append(path_aftereffect)
                    stat_test_names.append(path_tests)
            else:
                raise ValueError(f'Unknown statistics_comparison mode: {statistics_comparison}')

            if statistics_comparison == 'vs_zero' and not analysis_has_wt:
                for path_idx, path_name in enumerate(analysis_display_names):
                    learning_fig = utils.plot_limb_metric_scatter(
                        [learning_by_path[path_idx]],
                        [path_name],
                        [analysis_path_colors[path_idx]],
                        [analysis_summary_colors[path_idx]],
                        summary_limb_labels_filtered,
                        [paw_colors[idx] for idx in summary_limb_order_filtered],
                        f'{param_abbreviations[param_name]} learning',
                        stat_results=[learning_pvalues[path_idx]] if compute_statistics else None,
                        stat_mode=statistics_comparison,
                        fig_size=FIGSIZE_SP,
                    )
                    learning_output_name = f'{param_name}_learning_scatterplot_limbs_{utils.sanitize_filename_label(path_name)}_{statistics_test}_{statistics_comparison}.png'
                    utils.save_figure_multi_format(
                        learning_fig,
                        paths_save[compare_output_index],
                        utils.get_phase_st_output_filename(learning_output_name, stance_phase_reference_mode) if param_name == 'phase_st' else learning_output_name,
                        dpi=150,
                        bbox_inches='tight',
                    )
                    plt.close(learning_fig)

                    aftereffect_fig = utils.plot_limb_metric_scatter(
                        [aftereffect_by_path[path_idx]],
                        [path_name],
                        [analysis_path_colors[path_idx]],
                        [analysis_summary_colors[path_idx]],
                        summary_limb_labels_filtered,
                        [paw_colors[idx] for idx in summary_limb_order_filtered],
                        f'{param_abbreviations[param_name]} after effect',
                        stat_results=[aftereffect_pvalues[path_idx]] if compute_statistics else None,
                        stat_mode=statistics_comparison,
                        fig_size=FIGSIZE_SP,
                    )
                    aftereffect_output_name = f'{param_name}_aftereffect_scatterplot_limbs_{utils.sanitize_filename_label(path_name)}_{statistics_test}_{statistics_comparison}.png'
                    utils.save_figure_multi_format(
                        aftereffect_fig,
                        paths_save[compare_output_index],
                        utils.get_phase_st_output_filename(aftereffect_output_name, stance_phase_reference_mode) if param_name == 'phase_st' else aftereffect_output_name,
                        dpi=150,
                        bbox_inches='tight',
                    )
                    plt.close(aftereffect_fig)
            else:
                learning_fig = utils.plot_limb_metric_scatter(
                    learning_by_path,
                    analysis_display_names,
                    analysis_path_colors,
                    analysis_summary_colors,
                    summary_limb_labels_filtered,
                    [paw_colors[idx] for idx in summary_limb_order_filtered],
                    f'{param_abbreviations[param_name]} learning',
                    stat_results=learning_pvalues if compute_statistics else None,
                    stat_mode=statistics_comparison,
                    fig_size=FIGSIZE_SP,
                )
                utils.save_figure_multi_format(
                    learning_fig,
                    paths_save[compare_output_index],
                    utils.get_phase_st_output_filename(f'{param_name}_learning_scatterplot_limbs_{statistics_test}_{statistics_comparison}.png', stance_phase_reference_mode) if param_name == 'phase_st' else f'{param_name}_learning_scatterplot_limbs_{statistics_test}_{statistics_comparison}.png',
                    dpi=150,
                    bbox_inches='tight',
                )
                plt.close(learning_fig)

                aftereffect_fig = utils.plot_limb_metric_scatter(
                    aftereffect_by_path,
                    analysis_display_names,
                    analysis_path_colors,
                    analysis_summary_colors,
                    summary_limb_labels_filtered,
                    [paw_colors[idx] for idx in summary_limb_order_filtered],
                    f'{param_abbreviations[param_name]} after effect',
                    stat_results=aftereffect_pvalues if compute_statistics else None,
                    stat_mode=statistics_comparison,
                    fig_size=FIGSIZE_SP,
                )
                utils.save_figure_multi_format(
                    aftereffect_fig,
                    paths_save[compare_output_index],
                    utils.get_phase_st_output_filename(f'{param_name}_aftereffect_scatterplot_limbs_{statistics_test}_{statistics_comparison}.png', stance_phase_reference_mode) if param_name == 'phase_st' else f'{param_name}_aftereffect_scatterplot_limbs_{statistics_test}_{statistics_comparison}.png',
                    dpi=150,
                    bbox_inches='tight',
                )
                plt.close(aftereffect_fig)

            stats_f.write(f'{"#"*60}\n')
            stats_f.write(f'Parameter: {param_name}\n')
            stats_f.write(f'{"#"*60}\n')
            if statistics_comparison == 'between_paths':
                for comparison_idx in range(1, len(analysis_paths)):
                    stats_f.write(f'  {analysis_display_names[0]} vs {analysis_display_names[comparison_idx]}\n')
                    for limb_pos, limb_label in enumerate(summary_limb_labels_filtered):
                        reference_learning = np.asarray(learning_by_path[0][limb_pos], dtype=float)
                        comparison_learning = np.asarray(learning_by_path[comparison_idx][limb_pos], dtype=float)
                        reference_aftereffect = np.asarray(aftereffect_by_path[0][limb_pos], dtype=float)
                        comparison_aftereffect = np.asarray(aftereffect_by_path[comparison_idx][limb_pos], dtype=float)
                        reference_learning = reference_learning[np.isfinite(reference_learning)]
                        comparison_learning = comparison_learning[np.isfinite(comparison_learning)]
                        reference_aftereffect = reference_aftereffect[np.isfinite(reference_aftereffect)]
                        comparison_aftereffect = comparison_aftereffect[np.isfinite(comparison_aftereffect)]
                        stats_f.write(f'    Limb {limb_label} ({stat_test_names[comparison_idx-1][limb_pos]})\n')
                        stats_f.write(
                            f'      Learning: p = {learning_pvalues[comparison_idx-1][limb_pos]:.6f}  {utils.pvalue_to_label(learning_pvalues[comparison_idx-1][limb_pos])}\n'
                        )
                        stats_f.write(
                            f'        {analysis_display_names[0]} mean +/- SEM: {np.nanmean(reference_learning):.4f} +/- {np.nanstd(reference_learning)/np.sqrt(len(reference_learning)) if len(reference_learning) > 0 else np.nan:.4f}\n'
                        )
                        stats_f.write(
                            f'        {analysis_display_names[comparison_idx]} mean +/- SEM: {np.nanmean(comparison_learning):.4f} +/- {np.nanstd(comparison_learning)/np.sqrt(len(comparison_learning)) if len(comparison_learning) > 0 else np.nan:.4f}\n'
                        )
                        stats_f.write(
                            f'      After effect: p = {aftereffect_pvalues[comparison_idx-1][limb_pos]:.6f}  {utils.pvalue_to_label(aftereffect_pvalues[comparison_idx-1][limb_pos])}\n'
                        )
                        stats_f.write(
                            f'        {analysis_display_names[0]} mean +/- SEM: {np.nanmean(reference_aftereffect):.4f} +/- {np.nanstd(reference_aftereffect)/np.sqrt(len(reference_aftereffect)) if len(reference_aftereffect) > 0 else np.nan:.4f}\n'
                        )
                        stats_f.write(
                            f'        {analysis_display_names[comparison_idx]} mean +/- SEM: {np.nanmean(comparison_aftereffect):.4f} +/- {np.nanstd(comparison_aftereffect)/np.sqrt(len(comparison_aftereffect)) if len(comparison_aftereffect) > 0 else np.nan:.4f}\n'
                        )
                    stats_f.write('\n')
            else:
                for path_idx, path_name in enumerate(analysis_display_names):
                    stats_f.write(f'  {path_name} vs 0\n')
                    for limb_pos, limb_label in enumerate(summary_limb_labels_filtered):
                        current_learning = np.asarray(learning_by_path[path_idx][limb_pos], dtype=float)
                        current_aftereffect = np.asarray(aftereffect_by_path[path_idx][limb_pos], dtype=float)
                        current_learning = current_learning[np.isfinite(current_learning)]
                        current_aftereffect = current_aftereffect[np.isfinite(current_aftereffect)]
                        stats_f.write(f'    Limb {limb_label} ({stat_test_names[path_idx][limb_pos]})\n')
                        stats_f.write(
                            f'      Learning: p = {learning_pvalues[path_idx][limb_pos]:.6f}  {utils.pvalue_to_label(learning_pvalues[path_idx][limb_pos])}\n'
                        )
                        stats_f.write(
                            f'        {path_name} mean +/- SEM: {np.nanmean(current_learning):.4f} +/- {np.nanstd(current_learning)/np.sqrt(len(current_learning)) if len(current_learning) > 0 else np.nan:.4f}\n'
                        )
                        stats_f.write(
                            f'      After effect: p = {aftereffect_pvalues[path_idx][limb_pos]:.6f}  {utils.pvalue_to_label(aftereffect_pvalues[path_idx][limb_pos])}\n'
                        )
                        stats_f.write(
                            f'        {path_name} mean +/- SEM: {np.nanmean(current_aftereffect):.4f} +/- {np.nanstd(current_aftereffect)/np.sqrt(len(current_aftereffect)) if len(current_aftereffect) > 0 else np.nan:.4f}\n'
                        )
                    stats_f.write('\n')

    print(f'Statistics saved to {stats_file}')

