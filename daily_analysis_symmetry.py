# From daily_analysis cleaned; focus only on symmetry; for paper LocoCF
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as st
import random
import plotting_functions as pf
import utils

# Set the default font
plt.rcParams['font.family'] = 'Arial'


# Inputs
bs_bool = 1
plot_continuous = 0
compute_statistics = 1
scatter_single_animals = 0
significance_threshold = 0.05
statistics_test = 'nonparametric'  # 'nonparametric' or 'ttest'
stance_phase_reference_mode = 'slow_hind'  # 'slow_hind' or 'ipsi_hind'
data_path = 'D:\\AliG\\climbing-opto-treadmill\\'


# Axes ranges
# Tied
axes_ranges = {'coo': [-3, 3], 'step_length': [-9, 9], 'double_support': [-10, 10], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2],'phase_st':[-4, 7]}
bars_ranges = {'coo': [-3, 3], 'step_length': [-9, 9], 'double_support': [-10, 10], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2],'phase_st':[-4, 7]}   # tied
# Split                                         # DS [-7, 11]
axes_ranges = {'coo': [-6, 2], 'step_length': [-10, 5], 'double_support': [-7, 16], 'coo_stance': [-2, 7], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2],'phase_st':[-3, 8]}   #Rfast
bars_ranges = {'coo': [-2, 4], 'step_length': [-5, 9], 'double_support': [-12, 5], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2],'phase_st':[-4, 7]}     #Rfast
#axes_ranges = {'coo': [-4, 4], 'step_length': [-7, 8], 'double_support': [-16, 7], 'coo_stance': [-7, 2], 'swing_length': [-12, 5], 'stance_speed': [-0.4,-0.2],'phase_st':[-1,1]}     #Lfast
#bars_ranges = {'coo': [-4, 2], 'step_length': [-9, 5], 'double_support': [-5, 10], 'coo_stance': [-5, 5], 'swing_length': [-12, 5], 'stance_speed': [-0.4,-0.2],'phase_st':[-1,1]}     #Lfast

uniform_ranges = 1

# List of paths for each experiment - it is possible to have only one element
# If there is a control with different sample size, it should be the first!!!
experiment_names = ['WT', 'right fast']       #'th200st' ,'th100sw']   #'WT', 'contra fast', 'ipsi fast']   #'th200st' ,'th100sw']           #'th200st' ,'th100sw']               #'WT']       #'stance stim', 'swing stim']       # ['WT', 'contra fast right', 'ipsi fast left']           # ,'th100sw' ['control', 'stance onset', 'swing onset']             #'ChR2']           #'right fast', 'left fast']          #,'stance stim', 'swing stim']           #'left fast no-stim','left fast perturb']   #'right fast', 'left fast' ]   'split left fast stim',    # 'control'] #         #'trial stim', 'stance stim', swing stim    'chr2'


paths = [  
data_path+'WT split-belt learning\\',        # Non injected control

#data_path+'Experiments ChR2 RT extra-zombies\\HISTO_CHECKED_ANIMALS_LATinj\\split contra fast CL-Ali\\',
#data_path+'Experiments ChR2 RT extra-zombies\\HISTO_CHECKED_ANIMALS_LATinj\\split ipsi fast CL-Ali\\',
#data_path+'Experiments ChR2 RT extra-zombies\\HISTO_CHECKED_ANIMALS_RLinj\\split contra fast right CL-Ali\\',
#data_path+'Experiments ChR2 RT extra-zombies\\HISTO_CHECKED_ANIMALS_RLinj\\split ipsi fast left CL-Ali\\', 
data_path+'Experiments ChR2 RT extra-zombies\\HISTO_CHECKED_ANIMALS_Linj\\split right fast CL-Ali\\', 

#data_path+'Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\tied stance stim CL-Ali\\',
#data_path+'Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\tied swing stim CL-Ali\\',
# data_path+'Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\all REPLAY stim\\',
# data_path+'Experiments JAWS RT\\Split belt sessions\\ALL_ANIMALS\\split right fast control\\',
# data_path+'Experiments JAWS RT\\Split belt sessions\\ALL_ANIMALS\\split right fast stance stim\\',
# data_path+'Experiments JAWS RT\\Split belt sessions\\ALL_ANIMALS\\split right fast swing stim\\',

#data_path+'Experiments ChR2 RT\\LOW expression\\ALL_ANIMALS\\tied th200st IO 50ms CL-Ali\\',
#data_path+'Experiments ChR2 RT\\LOW expression\\ALL_ANIMALS\\tied th100sw IO 50ms CL-Ali\\',
]

experiment_colors_dict = {'trial stim':'purple', 'stance stim':'darkorange','swing stim': 'green', 'control':'black', 'ChR2': 'cyan',
                          'stance onset':'green','swing onset': 'darkorange',
                          'right fast no-stim': 'gray',  
                          'left fast no-stim': 'gray', 
                          'right fast stim': 'green', 
                          'left fast stim': 'cyan',
                          'right fast perturb': 'cyan',     #'red', 
                          'left fast perturb': 'lightgreen',
                          'right fast': 'blue',
                          'left fast': 'black',
                          'WT': 'gray',
                          'th200st': 'darkorange',   #'royalblue',
                          'th100sw': 'green',   #'skyblue'        
                          'ipsi fast': 'black',
                          'contra fast': 'lightseagreen',             # '#b892e9ff',         # 'magenta'
                          'ipsi fast left': 'black',
                          'contra fast right': 'darkblue',
                          'REPLAY stim': 'skyblue',
                          'data split': 'black'
                          }

animal_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']              # Use the default matplotlib colours
animal_colors_dict = {'MC16846': "#FFD700",'MC16848':"#BBF90F",'MC16850': "#15B01A",'MC16851': animal_colors[0], 'MC17319': animal_colors[1],
                      'MC17665': '#CCCCFF','MC17670': '#660033','MC17666': animal_colors[4], 'MC17668': animal_colors[5],'MC17669': animal_colors[6], 
                      'MC19022': animal_colors[7],'MC19082': animal_colors[8],'MC19123': animal_colors[9], 'MC19124': '#FF00FF', 'MC19130': '#00FFFF',
                      'MC19132': '#0000FF','MC19214': '#00FF00', 'MC18737': '#F08080', 'MC19107': '#FA8072', 'VIV41330': animal_colors[2], 
                      'VIV41329': animal_colors[3], 'VIV42375': '#5C62D6', 'VIV42376': '#FF0000', 'VIV42428': '#BC8F8F', 'VIV42429': animal_colors[8],
                      'VIV42430': '#FF4500','VIV41330a': animal_colors[2], 'VIV41329a': animal_colors[3], 'VIV42375a': '#5C62D6', 'VIV42376a': '#FF0000', 
                      'VIV42428a': '#BC8F8F', 'VIV42429a': animal_colors[8], 'VIV42430a': '#FF4500',
                      #IO fiber control
                      'VIV40958':animal_colors[4], 'VIV41344':animal_colors[5], 'VIV41345':animal_colors[6], 
                      #ChR2         wrong!!! FIX names
                      'VIV42375': animal_colors[4],'VIV42376': animal_colors[5],'VIV42428': animal_colors[7],'VIV42429': animal_colors[8],
                      'VIV42430': animal_colors[9], 'VIV42906': animal_colors[2], 'VIV42907': animal_colors[3],'VIV42908':animal_colors[4], 'VIV42974':animal_colors[5],
                      'VIV42985':animal_colors[6], 'VIV42992': animal_colors[7],'VIV42987': animal_colors[8],
                      'VIV44771': '#CCCCFF', 'VIV44765': '#00FF00', 'VIV44766': '#FF4500', 'VIV45372': '#BC8F8F', 'VIV45373': '#F08080', 
                      'VIV49571': "#FFD700",'VIV49572':"#BBF90F",'VIV49604': "#15B01A",'VIV49605': animal_colors[0],
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

# All animals
included_animal_list = []

# JAWS histology-confirmed (18 animals)
included_animal_list_jaws =  ['MC16848', 'MC16851', 'MC17319', 'MC17665', 'MC17666', 'MC17669', 'MC17670', 'MC19082', 'MC19124', 
                        'MC19214', 'VIV41329', 'VIV41330', 'VIV42375', 'VIV42428', 'VIV42429', 'VIV42430', 'VIV42376', 'MC19107']         # 'VIV42376' only tied, 'MC19107' only tied and Rfast

# ChR2 LE histology-confirmed (7 animals)
included_animal_list_chr2 =  ['VIV42906', 'VIV42908', 'VIV42974', 'VIV42985','VIV42987','VIV44766', 'VIV45372']  

# Extra-zombies histology-checked UNIlateral injections
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
if any('split' in path for path in paths):
    intervals_split_stim = {'split': [split_start, split_duration]}
elif any('stim' in path or 'th' in path for path in paths):
    intervals_split_stim = {'stim': [stim_start, stim_duration]}
print_plots = 1
print_plots_multi_session = 1
paw_colors = ['red', 'magenta', 'blue', 'cyan']
paws = ['FR', 'HR', 'FL', 'HL']
import online_tracking_class
import locomotion_class
otrack_classes =  []
locos = []
paths_save = []
param_sym_multi = {}
baseline_multi = {}
included_animals_multi = {}
path_index = 0
merge_RLfast_sessions = 0 
for path in paths:
    included_animal_list = []
    print("Analysing..........................", path)
    pixel_to_mm = 1/3.3            # real-time setup
    floor_factor = 268
    if 'WT' in path or 'Miniscopes' in path:
        pixel_to_mm = 1/1.955   # Dana's setup              1/1.98          # Jovin's     
        floor_factor = 152         
        included_animal_list = []
    elif 'JAWS' in path:
        included_animal_list = included_animal_list_jaws
        if 'split' in path:
            included_animal_list.remove('VIV42376')     # Only tied, not split
            if 'left' in path:
                included_animal_list.remove('MC19107')     # Only tied and right fast, not left fast
    elif 'ChR2' in path:
        included_animal_list = included_animal_list_chr2
    elif 'LATinj' in path:
        included_animal_list = included_animal_list_EZ
    elif 'RLinj' in path:
        included_animal_list = included_animal_list_EZ_double
        merge_RLfast_sessions = 1             # This will merge the right and left fast sessions for the same animal, that should be element 2nd and 3rd in the list of paths
    
        

    otrack_classes.append(online_tracking_class.otrack_class(path))
    locos.append(locomotion_class.loco_class(path, pixel_to_mm, floor_factor))
    paths_save.append(path + 'grouped output temp\\')   
    if not os.path.exists(path + 'grouped output temp\\'): 
        os.mkdir(path + 'grouped output temp\\') 

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
    included_animals_multi[path] = list(included_animal_list)

    # Assign random colors to animals not already in the dictionary
    for animal in animal_list:
        if animal not in animal_colors_dict:
            animal_colors_dict[animal] = "#%06x" % random.randint(0, 0xFFFFFF)

    session_list = []
    for a in range(len(animal_session_list)):
        session_list.append(animal_session_list[a][1])

  
    # FOR EACH SESSION SEPARATE CALCULATION AND PLOT SAVING
    param_sym_labels = {'coo': 'Center of\noscillation (mm)', 'step_length': 'Step length (mm)', 'double_support': '% double support', 
                    'coo_stance': 'Spatial motor\noutput (mm)', 'swing_length': 'Swing length(mm)', 'phase_st': 'Stance phase (%)', 'stance_speed': 'Stance speed'}
    param_sym_abbreviations = {'coo': 'COO', 'step_length': 'SL', 'double_support': 'DS', 'coo_stance': 'CS', 'swing_length': 'SW', 'phase_st': 'phase', 'stance_speed': 'SS'}
    param_sym_label = list(param_sym_labels.values())
    param_sym_name = list(param_sym_labels.keys())
    param_sym = np.zeros((len(param_sym_name), len(animal_list), Ntrials))
    param_sym[:] = np.NaN
    param_phase = np.zeros((4, len(animal_list), Ntrials))
    param_phase[:] = np.nan
    param_phase_bs = np.zeros((4, len(animal_list), Ntrials))
    param_phase_bs[:] = np.nan
    stance_speed = np.zeros((4, len(animal_list), Ntrials))
    stance_speed[:] = np.NaN
    st_strides_trials = []
    for count_animal, animal in enumerate(animal_list):
        session = int(session_list[count_animal])
        #TODO: check if this filelist needs to be emptied first!
        filelist = locos[path_index].get_track_files(animal, session)
        for f in filelist:
            count_trial = int(f.split('DLC')[0].split('_')[-1])-1      # Get trial number from file name, to spot any missing trial; parameters for remaining ones will stay to NaN
            [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = locos[path_index].read_h5(f, 0.9, 0)
            print("Processing animal ", animal, " trial ", str(count_trial+1))
            [st_strides_mat, sw_pts_mat] = locos[path_index].get_sw_st_matrices(final_tracks, 1)
            st_strides_trials.append(st_strides_mat)
            paws_rel = locos[path_index].get_paws_rel(final_tracks, 'X')
            for count_p, param in enumerate(param_sym_name):
                if np.sum(~np.isnan(np.concatenate([np.ravel(x) for x in st_strides_mat if isinstance(x, (np.ndarray, list))]))) == 0:
                    print("No good strides found for trial", count_trial + 1, "in animal", animal, ". Skipping trial.")
                    continue
                param_mat = locos[path_index].compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param)
                if param == 'stance_speed':
                    for count_paw in range(4):
                        stance_speed[count_paw, count_animal, count_trial] = np.nanmean(param_mat[count_paw])
                elif param == 'phase_st':
                    ref_paw = utils.get_stance_phase_reference_paw(
                        experiment_name,
                        animal,
                        stance_phase_reference_mode,
                        LexpEZ,
                        RexpEZ,
                        included_animal_list_EZ_double,
                    )
                    for count_paw in range(4):
                        phase_values = np.asarray(param_mat[ref_paw][count_paw], dtype=float)
                        if len(phase_values) == 0 or np.sum(np.isnan(phase_values)) > 0.5*len(phase_values):
                            param_phase[count_paw, count_animal, count_trial] = np.nan
                        else:
                            param_phase[count_paw, count_animal, count_trial] = st.circmean(phase_values, low=0, high=2*np.pi, nan_policy='omit')
                    front_phase = param_phase[0, count_animal, count_trial]
                    fast_front_phase = param_phase[2, count_animal, count_trial]
                    if np.isfinite(front_phase) and np.isfinite(fast_front_phase):
                        param_sym[count_p, count_animal, count_trial] = np.angle(np.exp(1j*(front_phase-fast_front_phase)))
                else:
                    param_sym[count_p, count_animal, count_trial] = np.nanmean(param_mat[0])-np.nanmean(param_mat[2])

        if ('contra' in experiment_name and animal in RexpEZ) or ('ipsi' in experiment_name and animal in LexpEZ):  # To invert Lfast sessions for being comparable to Rfast sessions    
            param_sym[:, count_animal, :] = -param_sym[:, count_animal, :]
    
    
    # BASELINE SUBTRACTION OF PARAMETERS
    if bs_bool:
        param_sym_bs = np.full(np.shape(param_sym), np.nan)
        baseline_values = np.full((np.shape(param_sym)[0], np.shape(param_sym)[1]), np.nan)
        for p in range(np.shape(param_sym)[0]-1):
            for a in range(np.shape(param_sym)[1]):
                animal_name = animal_list[a]
                if stim_start == split_start:
                    baseline_window = param_sym[p, a, :stim_start-1]
                else:
                    baseline_window = param_sym[p, a, stim_start-1:split_start-1]

                if param_sym_name[p] == 'phase_st':
                    for count_paw in range(4):
                        if stim_start == split_start:
                            paw_baseline_window = param_phase[count_paw, a, :stim_start-1]
                        else:
                            paw_baseline_window = param_phase[count_paw, a, stim_start-1:split_start-1]

                        if np.any(np.isfinite(paw_baseline_window)):
                            bs_paw_mean = st.circmean(paw_baseline_window, low=0, high=2*np.pi, nan_policy='omit')
                            param_phase_bs[count_paw, a, :] = np.angle(np.exp(1j*(param_phase[count_paw, a, :]-bs_paw_mean)))
                        else:
                            param_phase_bs[count_paw, a, :] = np.nan

                    phase_sym_bs = np.angle(np.exp(1j*(param_phase_bs[0, a, :]-param_phase_bs[2, a, :])))
                    param_sym_bs[p, a, :] = (phase_sym_bs/(2*np.pi))*100
                    if ('contra' in experiment_name and animal_name in RexpEZ) or ('ipsi' in experiment_name and animal_name in LexpEZ):
                        param_sym_bs[p, a, :] = -param_sym_bs[p, a, :]

                    if np.any(np.isfinite(baseline_window)):
                        bs_mean = (st.circmean(baseline_window, low=-np.pi, high=np.pi, nan_policy='omit')/(2*np.pi))*100
                    else:
                        bs_mean = np.nan
                else:
                    bs_mean = np.nanmean(baseline_window)
                    param_sym_bs[p, a, :] = param_sym[p, a, :] - bs_mean

                baseline_values[p, a] = bs_mean
                if ('contra' in experiment_name and animal_name in RexpEZ) or ('ipsi' in experiment_name and animal_name in LexpEZ):
                    baseline_values[p, a] = -baseline_values[p, a]              # To recover original value before inversion, so that we compare signed baselines
    else:
        param_sym_bs = param_sym.copy()
        phase_index = param_sym_name.index('phase_st')
        param_sym_bs[phase_index, :, :] = (param_sym_bs[phase_index, :, :]/(2*np.pi))*100

    if any('right' in element for element in experiment_names) and any('left' in element for element in experiment_names) and 'left' in experiment_name:      # If we are comparing left and right we will have them both in experiment names
        param_sym_bs = -param_sym_bs

           

    for p in range(np.shape(param_sym)[0] - 1):
        # Plot learning curve for individual animals
        fig = pf.plot_learning_curve_ind_animals(param_sym_bs, p, param_sym_labels, animal_list, animal_colors_dict, intervals=intervals_split_stim)
        # Save plot
        if print_plots:
            pf.save_plot(fig, paths_save[path_index], utils.get_param_output_name(param_sym_name[p], stance_phase_reference_mode), plot_name='ind_animals', bs_bool=bs_bool)
            
    plt.close('all')

    # PLOT ANIMAL AVERAGE with INDIVIDUAL ANIMALS FOR EACH SESSION
    param_sym_multi[path] = {}
    baseline_multi[path] = {}
    for p in range(np.shape(param_sym)[0]):
        param_sym_bs_ave = param_sym_bs[p, included_animals_id, :]
        fig = pf.plot_learning_curve_ind_animals_avg(param_sym_bs_ave, p, param_sym_labels, animal_list, [included_animal_list, included_animals_id],
                                                        [animal_colors_dict, experiment_colors_dict], experiment_name, intervals=intervals_split_stim, 
                                                        ranges=[uniform_ranges, {k: [v*1.5 for v in vals] for k, vals in axes_ranges.items()}])
        # Save plot
        if print_plots:
            pf.save_plot(fig, paths_save[path_index], utils.get_param_output_name(param_sym_name[p], stance_phase_reference_mode), plot_name='average', bs_bool=bs_bool)
            
        # Save param_sym for multi-session plot (in case we have multiple sessions to analyse/plot)
        param_sym_multi[path][p] = param_sym_bs_ave
        # Store baseline (pre-subtraction) values for baseline scatterplot
        if bs_bool:
            baseline_multi[path][p] = baseline_values[p, included_animals_id]
    plt.close('all')


    # PLOT STANCE SPEED for ALL ANIMALS
    for a in range(np.shape(stance_speed)[1]):
        data = stance_speed[:, a, :]
        fig, ax = plt.subplots(figsize=(7,10), tight_layout=True)
        rectangle = plt.Rectangle((split_start-0.5, -0.5), split_duration, 1, fc='lightblue',alpha=0.3)
        for p in range(4):
            ax.axvline(x = split_start, color='dimgray', linestyle='--')
            ax.axvline(x = split_start+ split_duration, color='dimgray', linestyle='--')
            ax.plot(np.linspace(1,len(data[p,:]),len(data[p,:])), data[p,:], color = paw_colors[p], linewidth = 2)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('Trial', fontsize = 24)
            ax.set_ylabel('Stance speed', fontsize = 24)
            ax.tick_params(axis='x',labelsize = 20)
            ax.tick_params(axis='y',labelsize = 20)
            ax.set_title(animal_list[a],fontsize=18)
        
        if print_plots:
            if not os.path.exists(paths_save[path_index]):
                os.mkdir(paths_save[path_index])
            utils.save_figure_multi_format(fig, paths_save[path_index], animal_list[a] + '_stancespeed', dpi=96)
    plt.close('all')


    # CONTINUOUS STEP LENGTH WITH LASER ON
    if plot_continuous:
        trials = np.arange(1, Ntrials+1)
        for count_animal, animal in enumerate(animal_list):
            param_sl = []
            st_strides_trials = []
            session = int(session_list[count_animal])
            filelist = locos[path_index].get_track_files(animal, session)
            for count_trial, f in enumerate(filelist):
                [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = locos[path_index].read_h5(f, 0.9, 0)
                [st_strides_mat, sw_pts_mat] = locos[path_index].get_sw_st_matrices(final_tracks, 1)
                paws_rel = locos[path_index].get_paws_rel(final_tracks, 'X')
                param_mat = locos[path_index].compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, 'step_length')
                param_sl.append(param_mat)
                st_strides_trials.append(st_strides_mat)
            [stride_idx, trial_continuous, sl_time, sl_values] = locos[path_index].param_continuous_sym(param_sl, st_strides_trials, trials, 'FR', 'FL', 1, 1)
            fig, ax = plt.subplots(tight_layout=True, figsize=(25,10))
            sl_time_start = sl_time[np.where(np.array(trial_continuous) == stim_start)[0][0]]
            sl_time_duration = sl_time[np.where(np.array(trial_continuous) == stim_start)[0][0]]+(locos[path_index].trial_time*stim_duration)
            rectangle = plt.Rectangle((sl_time_start, np.nanmin(sl_values)), sl_time_duration-sl_time_start, np.nanmax(sl_values)+np.abs(np.nanmin(sl_values)), fc=experiment_colors_dict[experiment_name], alpha=0.3)
            plt.gca().add_patch(rectangle)
            ax.plot(sl_time, sl_values, color='black')
            ax.set_xlabel('time (s)')
            ax.set_title('continuous step length')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if print_plots:
                if not os.path.exists(paths_save[path_index]):
                    os.mkdir(paths_save[path_index])
                utils.save_figure_multi_format(fig, paths_save[path_index], animal_list[a] + '_sl_sym_continuous', dpi=96)

    path_index = path_index+1
    

# MULTI-SESSION PLOT
if merge_RLfast_sessions:
    # Get the path keys (should be 0, 1, 2)
    paths_keys = list(param_sym_multi.keys())
    
    # For each parameter, concatenate data from paths[2] into paths[1]
    for p in range(np.shape(param_sym)[0]):
        param_sym_multi[paths_keys[1]][p] = np.concatenate((param_sym_multi[paths_keys[1]][p], 
                                                                   param_sym_multi[paths_keys[2]][p]), axis=0)
    
    # Delete the third element
    del param_sym_multi[paths_keys[2]]
    # Merge baseline_multi similarly
    for p in range(np.shape(param_sym)[0]):
        baseline_multi[paths_keys[1]][p] = np.concatenate((baseline_multi[paths_keys[1]][p],
                                                           baseline_multi[paths_keys[2]][p]), axis=0)
    del baseline_multi[paths_keys[2]]
    experiment_colors_dict['contra fast right'] = 'darkblue'
    included_animal_list = included_animal_list*2  # We have the same animals in both sessions
    included_animals_id = included_animals_id + [i + len(included_animals_id) for i in included_animals_id]  # We have the same animals in both sessions, so we just need to offset the indices for the second session


# Determine the maximum number of animals across all paths, to handle the case of different number of animals in each path
max_animals = max(param_sym_multi[path][0].shape[0] for path in param_sym_multi.keys())

# Open stats file once for all parameters
if compute_statistics:
    if not os.path.exists(paths_save[0]):
        os.mkdir(paths_save[0])
    stats_file = os.path.join(paths_save[0], f'statistics_{statistics_test}.txt')
    stats_f = open(stats_file, 'w')
    stats_f.write('='*60 + '\n')
    stats_f.write(f'Statistical results\n')
    stats_f.write(f'Statistical test family: {statistics_test}\n')
    stats_f.write(f'Significance threshold: {significance_threshold}\n')
    stats_f.write(f'Experiments: {experiment_names}\n')
    stats_f.write('='*60 + '\n\n')

for p in range(np.shape(param_sym)[0] - 1):
    fig_multi = pf.plot_learning_curve_avg_compared(param_sym_multi, p, param_sym_labels, [included_animal_list, included_animals_id], experiment_colors_dict, experiment_names, intervals=intervals_split_stim, ranges=[uniform_ranges, axes_ranges], use_median_iqr=False)
    
    if print_plots:
        pf.save_plot(fig_multi, paths_save[0], utils.get_param_output_name(param_sym_name[p], stance_phase_reference_mode), plot_name='average_multi_session', bs_bool=bs_bool)

    # LEARNING PARAMETERS - each one will be num_experiments x num_animals
    initial_error = []                      
    learning = []
    aftereffect = []
    learning_sym_change = []
    aftereffect_sym_change = []
    stat_initial_error = []
    stat_learning = []
    stat_aftereffect = []
    stat_learning_sym_change = []
    stat_aftereffect_sym_change = []
    pval_initial_error = []
    pval_learning = []
    pval_aftereffect = []
    pval_learning_sym_change = []
    pval_aftereffect_sym_change = []
    stat_test_names = []
    if split_duration==0:
        split_start = stim_start
        split_duration = stim_duration


    path_index = 0  
    
    if merge_RLfast_sessions:
        paths = param_sym_multi.keys()  # If we merge RL fast sessions, we have less paths to consider
        experiment_names = [name for name in experiment_names if any(name in key for key in param_sym_multi.keys())]
    current_experiment_colors = [experiment_colors_dict[key] for key in experiment_names if key in experiment_names]
    scatter_display_names = ['non-inj' if name == 'WT' else name for name in experiment_names]
    if merge_RLfast_sessions:
        scatter_display_names = ['RL-inj' if name != 'WT' else 'non-inj' for name in experiment_names]
    
    for path in paths:

        # Get the current array
        current_initial_error = param_sym_multi[path][p][:, split_start-1]      # First trial of split
        current_learning = np.nanmean(param_sym_multi[path][p][:, split_start+split_duration-3:split_start+split_duration-1], axis=1) - param_sym_multi[path][p][:, split_start-1]          # Last 2 trials of split - first trial of split
        current_aftereffect = np.nanmean(param_sym_multi[path][p][:, split_start+split_duration-1:split_start+split_duration+1], axis=1)            # First 2 trials of after effect

        # Pad arrays with NaN to match the maximum number of animals
        padded_initial_error = np.full(max_animals, np.nan)
        padded_learning = np.full(max_animals, np.nan)
        padded_aftereffect = np.full(max_animals, np.nan)

        padded_initial_error[:len(current_initial_error)] = current_initial_error
        padded_learning[:len(current_learning)] = current_learning
        padded_aftereffect[:len(current_aftereffect)] = current_aftereffect
        
        # Append the padded arrays
        initial_error.append(padded_initial_error)
        learning.append(padded_learning)
        aftereffect.append(padded_aftereffect)


        # Compare to first column (if there is control, it should go to first column)
        if compute_statistics and path_index>0:
            print(learning[0], learning[path_index])
            print(aftereffect[0], aftereffect[path_index])
            if max_animals!=len(current_initial_error) or max_animals!=np.sum(~np.isnan(initial_error[0])):     # Samples have different size, so unpaired stats with the control (that is first row)
                if statistics_test == 'ttest':
                    test_name = 't-test'
                    pv_ie = st.ttest_ind(initial_error[0], initial_error[path_index], equal_var=False, nan_policy='omit').pvalue
                    pv_lr = st.ttest_ind(learning[0], learning[path_index], equal_var=False, nan_policy='omit').pvalue
                    pv_ae = st.ttest_ind(aftereffect[0], aftereffect[path_index], equal_var=False, nan_policy='omit').pvalue
                else:
                    test_name = 'Mann-Whitney U'
                    pv_ie = st.mannwhitneyu(initial_error[0], initial_error[path_index], alternative='two-sided', nan_policy='omit').pvalue
                    pv_lr = st.mannwhitneyu(learning[0], learning[path_index], alternative='two-sided', nan_policy='omit').pvalue
                    pv_ae = st.mannwhitneyu(aftereffect[0], aftereffect[path_index], alternative='two-sided', nan_policy='omit').pvalue
            else:
                if statistics_test == 'ttest':
                    test_name = 'paired t-test'
                    pv_ie = st.ttest_rel(initial_error[0], initial_error[path_index], nan_policy='omit').pvalue
                    pv_lr = st.ttest_rel(learning[0], learning[path_index], nan_policy='omit').pvalue
                    pv_ae = st.ttest_rel(aftereffect[0], aftereffect[path_index], nan_policy='omit').pvalue
                else:
                    test_name = 'Wilcoxon'
                    pv_ie = st.wilcoxon(initial_error[0], initial_error[path_index], nan_policy='omit').pvalue
                    pv_lr = st.wilcoxon(learning[0], learning[path_index], nan_policy='omit').pvalue
                    pv_ae = st.wilcoxon(aftereffect[0], aftereffect[path_index], nan_policy='omit').pvalue
            print(['param ', param_sym_name[p], 'learning p=', pv_lr])
            print(['param ', param_sym_name[p], 'aftereffect p=', pv_ae])
            stat_initial_error.append(pv_ie < significance_threshold)
            stat_learning.append(pv_lr < significance_threshold)
            stat_aftereffect.append(pv_ae < significance_threshold)
            pval_initial_error.append(pv_ie)
            pval_learning.append(pv_lr)
            pval_aftereffect.append(pv_ae)
            stat_test_names.append(test_name)
        path_index+=1

    learning_sym_change=100*np.divide(np.array(learning),np.array(initial_error))
    initial_error_array = np.array(initial_error)
    initial_error_array[np.absolute(initial_error_array)<1] = np.nan
    aftereffect_sym_change=100*np.divide(np.array(aftereffect),np.absolute(initial_error_array))
    if compute_statistics and path_index>0:
        for path_index in range(1,len(paths)):
            if max_animals!=len(current_initial_error):     # Samples have different size, so unpaired stats
                if statistics_test == 'ttest':
                    pv_lsc = st.ttest_ind(learning_sym_change[0], learning_sym_change[path_index], equal_var=False, nan_policy='omit').pvalue
                    pv_asc = st.ttest_ind(aftereffect_sym_change[0], aftereffect_sym_change[path_index], equal_var=False, nan_policy='omit').pvalue
                else:
                    pv_lsc = st.mannwhitneyu(learning_sym_change[0], learning_sym_change[path_index], alternative='two-sided', nan_policy='omit').pvalue
                    pv_asc = st.mannwhitneyu(aftereffect_sym_change[0], aftereffect_sym_change[path_index], alternative='two-sided', nan_policy='omit').pvalue
            else:
                if statistics_test == 'ttest':
                    pv_lsc = st.ttest_rel(learning_sym_change[0], learning_sym_change[path_index], nan_policy='omit').pvalue
                    pv_asc = st.ttest_rel(aftereffect_sym_change[0], aftereffect_sym_change[path_index], nan_policy='omit').pvalue
                else:
                    pv_lsc = st.wilcoxon(learning_sym_change[0], learning_sym_change[path_index], nan_policy='omit').pvalue
                    pv_asc = st.wilcoxon(aftereffect_sym_change[0], aftereffect_sym_change[path_index], nan_policy='omit').pvalue
            stat_learning_sym_change.append(pv_lsc < significance_threshold)
            stat_aftereffect_sym_change.append(pv_asc < significance_threshold)
            pval_learning_sym_change.append(pv_lsc)
            pval_aftereffect_sym_change.append(pv_asc)
 

    # Plot selected parameters alone
    to_plot_separately = [aftereffect, learning]
    name_to_plot_separately = ['after effect', 'learning']
    pvals_to_plot_separately = [pval_aftereffect, pval_learning]

    for s in range(len(to_plot_separately)):
        fig_separate_onlyscatter = pf.plot_symmetry_scatterplot(
            to_plot_separately[s], experiment_names, current_experiment_colors,
            param_sym_abbreviations[param_sym_name[p]] + ' asymm ' + name_to_plot_separately[s],
            stat_results=pvals_to_plot_separately[s] if compute_statistics else None,
            ylim=bars_ranges[param_sym_name[p]] if uniform_ranges else None,
            display_names=scatter_display_names)
        if not os.path.exists(paths_save[0]):
                os.mkdir(paths_save[0])
        output_name = param_sym_name[p] + '_scatterplot_'+name_to_plot_separately[s] + (('_' + statistics_test) if compute_statistics else '')
        if param_sym_name[p] == 'phase_st':
            output_name = utils.get_phase_st_output_filename(output_name, stance_phase_reference_mode)
        utils.save_figure_multi_format(fig_separate_onlyscatter, paths_save[0], output_name, dpi=120)

    # LATinj-specific scatterplot: colour LexpEZ (dark) and RexpEZ (light) animals differently
    if any('LATinj' in p for p in list(paths)):
        paths_list = list(paths)
        for s in range(len(to_plot_separately)):
            point_colors_lr = []
            for ei in range(len(experiment_names)):
                n = len(to_plot_separately[s][ei])
                if ei < len(paths_list) and 'LATinj' in paths_list[ei]:
                    path_animals = included_animals_multi.get(paths_list[ei], [])
                    colors_arr = pf.get_latinj_point_colors(path_animals, n, current_experiment_colors[ei], LexpEZ, RexpEZ)
                    point_colors_lr.append(colors_arr)
                else:
                    point_colors_lr.append(None)
            fig_lr = pf.plot_symmetry_scatterplot(
                to_plot_separately[s], experiment_names, current_experiment_colors,
                param_sym_abbreviations[param_sym_name[p]] + ' asymm ' + name_to_plot_separately[s],
                stat_results=pvals_to_plot_separately[s] if compute_statistics else None,
                ylim=bars_ranges[param_sym_name[p]] if uniform_ranges else None,
                point_colors=point_colors_lr)
            output_name = param_sym_name[p] + '_scatterplot_' + name_to_plot_separately[s] + '_LRcolored' + (('_' + statistics_test) if compute_statistics else '')
            if param_sym_name[p] == 'phase_st':
                output_name = utils.get_phase_st_output_filename(output_name, stance_phase_reference_mode)
            utils.save_figure_multi_format(fig_lr, paths_save[0], output_name, dpi=120)
    if bs_bool:
        baseline = []
        stat_baseline = []
        pval_baseline = []
        path_idx = 0
        iter_paths = param_sym_multi.keys() if merge_RLfast_sessions else paths
        for path in iter_paths:
            current_baseline = baseline_multi[path][p]
            padded_baseline = np.full(max_animals, np.nan)
            padded_baseline[:len(current_baseline)] = current_baseline
            baseline.append(padded_baseline)
            if compute_statistics and path_idx > 0:
                if max_animals != len(current_baseline) or max_animals != np.sum(~np.isnan(baseline[0])):
                    if statistics_test == 'ttest':
                        pv_baseline = st.ttest_ind(baseline[0], baseline[path_idx], equal_var=False, nan_policy='omit').pvalue
                    else:
                        pv_baseline = st.mannwhitneyu(baseline[0], baseline[path_idx], alternative='two-sided', nan_policy='omit').pvalue
                else:
                    if statistics_test == 'ttest':
                        pv_baseline = st.ttest_rel(baseline[0], baseline[path_idx], nan_policy='omit').pvalue
                    else:
                        pv_baseline = st.wilcoxon(baseline[0], baseline[path_idx], nan_policy='omit').pvalue
                pval_baseline.append(pv_baseline)
                stat_baseline.append(pv_baseline < significance_threshold)
            path_idx += 1

        fig_baseline = pf.plot_symmetry_scatterplot(
            baseline, experiment_names, current_experiment_colors,
            param_sym_abbreviations[param_sym_name[p]] + ' baseline',
            stat_results=pval_baseline if compute_statistics else None,
            ylim=utils.get_baseline_scatter_ylim(param_sym_name[p], uniform_ranges, bars_ranges),
            display_names=scatter_display_names)
        if not os.path.exists(paths_save[0]):
            os.mkdir(paths_save[0])
        output_name = param_sym_name[p] + '_scatterplot_baseline' + (('_' + statistics_test) if compute_statistics else '')
        if param_sym_name[p] == 'phase_st':
            output_name = utils.get_phase_st_output_filename(output_name, stance_phase_reference_mode)
        utils.save_figure_multi_format(fig_baseline, paths_save[0], output_name, dpi=120)

        # LATinj-specific baseline scatterplot with L/R colors
        if any('LATinj' in p for p in list(paths)):
            paths_list = list(paths)
            point_colors_lr_bs = []
            for ei in range(len(experiment_names)):
                n = len(baseline[ei])
                if ei < len(paths_list) and 'LATinj' in paths_list[ei]:
                    path_animals = included_animals_multi.get(paths_list[ei], [])
                    colors_arr = pf.get_latinj_point_colors(path_animals, n, current_experiment_colors[ei], LexpEZ, RexpEZ)
                    point_colors_lr_bs.append(colors_arr)
                else:
                    point_colors_lr_bs.append(None)
            fig_baseline_lr = pf.plot_symmetry_scatterplot(
                baseline, experiment_names, current_experiment_colors,
                param_sym_abbreviations[param_sym_name[p]] + ' baseline',
                stat_results=pval_baseline if compute_statistics else None,
                ylim=utils.get_baseline_scatter_ylim(param_sym_name[p], uniform_ranges, bars_ranges),
                point_colors=point_colors_lr_bs)
            output_name = param_sym_name[p] + '_scatterplot_baseline_LRcolored' + (('_' + statistics_test) if compute_statistics else '')
            if param_sym_name[p] == 'phase_st':
                output_name = utils.get_phase_st_output_filename(output_name, stance_phase_reference_mode)
            utils.save_figure_multi_format(fig_baseline_lr, paths_save[0], output_name, dpi=120)
    if compute_statistics:
        stats_f.write(f'\n{"#"*60}\n')
        stats_f.write(f'  Parameter: {param_sym_name[p]} ({param_sym_abbreviations[param_sym_name[p]]})\n')
        stats_f.write(f'{"#"*60}\n\n')

        for ci in range(len(pval_initial_error)):
            stats_f.write(f'  {experiment_names[0]} vs {experiment_names[ci+1]}  ({stat_test_names[ci]})\n')
            stats_f.write(f'    Initial error:          p = {pval_initial_error[ci]:.6f}  {"*" if stat_initial_error[ci] else "n.s."}\n')
            stats_f.write(f'    Learning:               p = {pval_learning[ci]:.6f}  {"*" if stat_learning[ci] else "n.s."}\n')
            stats_f.write(f'    After effect:           p = {pval_aftereffect[ci]:.6f}  {"*" if stat_aftereffect[ci] else "n.s."}\n')
            if len(pval_learning_sym_change) > ci:
                stats_f.write(f'    Learning % change:      p = {pval_learning_sym_change[ci]:.6f}  {"*" if stat_learning_sym_change[ci] else "n.s."}\n')
            if len(pval_aftereffect_sym_change) > ci:
                stats_f.write(f'    After effect % change:  p = {pval_aftereffect_sym_change[ci]:.6f}  {"*" if stat_aftereffect_sym_change[ci] else "n.s."}\n')
            stats_f.write('\n')

        if bs_bool and len(stat_baseline) > 0:
            stats_f.write('  Baseline symmetry:\n')
            for ci in range(len(stat_baseline)):
                stats_f.write(f'    {experiment_names[0]} vs {experiment_names[ci+1]}: significant = {stat_baseline[ci]}\n')
            stats_f.write('\n')

        # Descriptive statistics
        stats_f.write('  Descriptive statistics (mean +/- SEM):\n')
        for ei, exp in enumerate(experiment_names):
            n_valid = np.sum(~np.isnan(initial_error[ei]))
            stats_f.write(f'    {exp} (n={int(n_valid)}):\n')
            stats_f.write(f'      Initial error:  {np.nanmean(initial_error[ei]):.4f} +/- {np.nanstd(initial_error[ei])/np.sqrt(n_valid):.4f}\n')
            stats_f.write(f'      Learning:       {np.nanmean(learning[ei]):.4f} +/- {np.nanstd(learning[ei])/np.sqrt(n_valid):.4f}\n')
            stats_f.write(f'      After effect:   {np.nanmean(aftereffect[ei]):.4f} +/- {np.nanstd(aftereffect[ei])/np.sqrt(n_valid):.4f}\n')

# Close stats file after all parameters
if compute_statistics:
    stats_f.close()
    print(f'Statistics saved to {stats_file}')

