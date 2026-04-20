import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as st
import math
import pandas as pd
import plotting_functions as pf

# Set the default font
#plt.rcParams['font.family'] = 'sans-serif'  # Or 'sans-serif', 'monospace', etc.
#plt.rcParams['font.serif'] = ['Helvetica']  # Example font family
plt.rcParams['font.family'] = 'Arial'


# Inputs
laser_event = 'swing'
single_animal_analysis = 0
if single_animal_analysis:
    animal = 'MC19022'
plot_continuous = 0
compare_baselines = 0
compute_statistics = 0
scatter_single_animals = 1
significance_threshold = 0.05

#axes_ranges = {'coo': [-5, 3], 'step_length': [-12, 5], 'double_support': [-7, 13], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2]}
#bars_ranges = {'coo': [-2, 5], 'step_length': [-3, 12], 'double_support': [-5, 13], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2]}
# Opto
axes_ranges = {'coo': [-3, 3], 'step_length': [-9, 9], 'double_support': [-8, 8], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2],'phase_st':[-1,1]}
bars_ranges = {'coo': [-3, 3], 'step_length': [-9, 9], 'double_support': [-8, 8], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2],'phase_st':[-1,1]}   # tied
#axes_ranges = {'coo': [-6, 2], 'step_length': [-12, 5], 'double_support': [-5, 10], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2],'phase_st':[-1,1]}       #Rfast
#bars_ranges = {'coo': [-2, 4], 'step_length': [-5, 9], 'double_support': [-9, 5], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2],'phase_st':[-1,1]}       #Rfast
#axes_ranges = {'coo': [-2, 4], 'step_length': [-3, 9], 'double_support': [-10, 5], 'coo_stance': [-5, 5], 'swing_length': [-12, 5], 'stance_speed': [-0.4,-0.2],'phase_st':[-1,1]}       #Lfast
#bars_ranges = {'coo': [-4, 2], 'step_length': [-9, 5], 'double_support': [-5, 10], 'coo_stance': [-5, 5], 'swing_length': [-12, 5], 'stance_speed': [-0.4,-0.2],'phase_st':[-1,1]}       #Lfast

# ChR2 right
axes_ranges = {'coo': [-6, 2], 'step_length': [-11, 5], 'double_support': [-6, 12], 'coo_stance': [-2, 6], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2], 'phase_st':[-1,1]}
bars_ranges = {'coo': [-2, 3], 'step_length': [-2, 5], 'double_support': [-8, 6], 'coo_stance': [-2, 4], 'swing_length': [-4, 6], 'stance_speed': [-0.4,-0.2], 'phase_st':[-1,1]}
# ChR2 left
#axes_ranges = {'coo': [-4, 6], 'step_length': [-7, 10], 'double_support': [-12, 8], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2], 'phase_st':[-1,1]}
#bars_ranges = {'coo': [-2, 2], 'step_length': [-5, 2], 'double_support': [-1, 6], 'coo_stance': [-5, 5], 'swing_length': [-5, 12], 'stance_speed': [-0.4,-0.2], 'phase_st':[-1,1]}
uniform_ranges = 1

# List of paths for each experiment - it is possible to have only one element
# If there is a control with different sample size, it should be the first!!!
experiment_names = [ 'data split']          #'WT', 'contra fast right', 'ipsi fast left']           # ,'th100sw' ['control', 'stance onset', 'swing onset']             #'ChR2']           #'right fast', 'left fast']          #,'stance stim', 'swing stim']           #'left fast no-stim','left fast perturb']   #'right fast', 'left fast' ]   'split left fast stim',    # 'control'] #         #'trial stim', 'stance stim', swing stim    'chr2'


paths = [
    'C:\\Users\\Utilizador\\Downloads\\data split 250514\\'
  #  'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\all REPLAY\\',
  #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT\\LOW expression\\ALL_ANIMALS\\tied th200st IO 50ms\\',
 # 'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT\\LOW expression\\ALL_ANIMALS\\tied th100sw IO 50ms\\'
   #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT\\LOW expression\\Split belt experiments\\20241111 split right fast control batch#4C\\',
  # 'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT\\LOW expression\\Split belt experiments\\20241112 split right fast stance onset stim 200st IO batch#4C\\',
  # 'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT\\LOW expression\\Split belt experiments\\20241113 split right fast swing onset stim 100sw IO batch#4C\\',
   #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT\\LOW expression\\Split belt experiments\\20241114 split left fast control batch#4C\\',
   #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT\\LOW expression\\Split belt experiments\\20241115 split left fast stance onset stim 200st IO batch#4C\\',
   #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT\\LOW expression\\Split belt experiments\\20241116 split left fast swing onset stim 100sw IO batch#4C\\'
  # 'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT\\LOW expression\\20241112 split right fast stance onset stim 200st IO batch#4C\\'
   #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\20240708 split right fast\\',
  # 'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\20241022 split right fast VIV49574 repeated\\'
   #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\20240709 split left fast\\'
   #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\20240716 tied stance stim 100sw CTXchr2\\',
  # 'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\20240729 tied swing stim 100st CTXchr2 1mW\\',
  #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT\\LOW expression\\20241025 tied stance stim 100sw batch #4C\\'
   #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\20241028 split right fast batch#4EZb\\'
  #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\20241007 split right fast batch#2EZ\\',
  #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\20241008 split left fast batch#2EZ\\',
  #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\20241007 split right fast batch#2EZ\\',
  #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\20241021 split right fast batch#3EZ\\',
  #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\20241021 split left fast batch#3EZ\\'
 #'D:\\AliG\\climbing-opto-treadmill\\WT split-belt learning\\',
  #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\HISTO_CHECKED_ANIMALS_RLinj\\split right fast\\',
  #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\HISTO_CHECKED_ANIMALS_RLinj\\split left fast\\',
   # 'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\HISTO_CHECKED_ANIMALS_Linj_CTXstim\\stance stim 100sw\\',
 # 'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\HISTO_CHECKED_ANIMALS_Linj_CTXstim\\swing stim 100st\\',
  #'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\HISTO_CHECKED_ANIMALS_LATinj\\split contra fast\\',
  #  'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\HISTO_CHECKED_ANIMALS_LATinj\\split ipsi fast\\',
   # 'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\HISTO_CHECKED_ANIMALS_RLinj\\split contra fast right\\',
   #  'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT extra-zombies\\HISTO_CHECKED_ANIMALS_RLinj\\split ipsi fast left\\', 
   # 'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Tied belt sessions\\20240311 tied swing stim redone\\'
   # 'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\tied stance stim\\',
 #  'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\tied swing stim\\',
  #  'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT\\LOW expression\\ALL_ANIMALS\\tied th200st IO 50ms\\',
  # 'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\tied stance stim REPLAY\\',
# 'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\tied swing stim REPLAY\\',
 # 'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Split belt sessions\\ALL_ANIMALS\\split left fast stance stim\\',
 #  'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Split belt sessions\\ALL_ANIMALS\\split right fast control\\',
 #'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Split belt sessions\\ALL_ANIMALS\\split right fast stance stim\\',
 #'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Split belt sessions\\ALL_ANIMALS\\split right fast swing stim\\',
    #'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Tied belt sessions\\20240130 tied stance stim IOcontrol\\',
   # 'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Tied belt sessions\\20240129 tied swing stim IOcontrol\\',
   # 'C:\\Users\\Utilizador\\Carey Lab Dropbox\\Alice Geminiani\\LocoCF-internal\\Tout_data\\20230606 tied stance stim\\'
    #'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Split belt sessions\\20230608 split right fast control\\'
 #'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\20240409 tied stance stim CTXchr2\\',
   # 'D:\ #\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\20240307 tied swing stim IOchr2\\'
       # 'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\20240202 split left fast stance stim\\',
   # 'C:\\Users\\Utilizador\\Carey Lab Dropbox\\Alice Geminiani\\Susd4KO project\\20240212 20240221 20240304 split right fast Susd4KO\\',
   # 'C:\\Users\\Utilizador\\Carey Lab Dropbox\\Alice Geminiani\\Susd4KO project\\20240216 20240226 20240308 split left fast Susd4KO\\'
     #"D:\\AliG\\climbing-opto-treadmill\\Experiments HGM\\LE\\split left fast no-stim S3\\",
    # "D:\\AliG\\climbing-opto-treadmill\\Experiments HGM\\LE\\split right fast no-stim S2\\",
      #"D:\\AliG\\climbing-opto-treadmill\\Experiments HGM\\LE\\split right fast stim S4\\",
     # "D:\\AliG\\climbing-opto-treadmill\\Experiments HGM\\LE\\split right fast perturb S5\\",
     #"D:\\AliG\\climbing-opto-treadmill\\Experiments HGM\\LE\\split left fast perturb S6\\"
    # "D:\\AliG\\climbing-opto-treadmill\\Experiments HGM\\HE\\split left fast no-stim\\",
    # "D:\\AliG\\climbing-opto-treadmill\\Experiments HGM\\HE\\split left fast perturb\\"
     #"D:\\AliG\\climbing-opto-treadmill\\Experiments HGM\\HE\\split right fast perturb\\original protocol\\"
     #"D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\20240531 tied swing stim IOchr2 50ms\\"
      #"D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\20240503 tied stance stim IOchr2 50ms\\"
      #'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Split belt sessions\\20240227 split right fast control\\'
     ]

experiment_colors_dict = {'trial stim':'purple', 'stance stim':'darkorange','swing stim': 'green', 'control':'black', 'ChR2': 'cyan',
                          'stance onset':'green','swing onset': 'orange',
                          'right fast no-stim': 'gray',     # 'blue', 
                          'left fast no-stim': 'gray', 
                          'right fast stim': 'green', 
                          'left fast stim': 'cyan',
                          'right fast perturb': 'cyan',     #'red', 
                          'left fast perturb': 'lightgreen',
                          'right fast': 'blue',
                          'left fast': 'black',
                          'WT': 'gray',     #'green',
                          'th200st': 'darkorange',   #'royalblue',
                          'th100sw': 'green',   #'skyblue',          #  ['control', 'stance onset', 'swing onset']             #'ChR2']           #'right fast', 'left fast']          #,'stance stim', 'swing stim']           #'left fast no-stim','left fast perturb']   #'right fast', 'left fast' ]   'split left fast stim',    # 'control'] #         #'trial stim', 'stance stim', swing stim    'chr2'
                          'ipsi fast': 'black',
                          'contra fast': 'lightseagreen',
                          'ipsi fast left': 'black',
                          'contra fast right': 'lightseagreen',
                          'REPLAY': 'skyblue',
                          'data split': 'black'
                          }      # stim on: trial stance swing    'trial stim':'purple', 
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
                      # Coco RN
                      'RNBIP1': animal_colors[0], 'RNBIP2': animal_colors[1], 'RNBIP3': '#CCCCFF',
                      'RNBIP4': animal_colors[4], 'RNBIP5': animal_colors[5]
                      }
'''
# Not histo confirmed in gray
animal_colors_dict = {'MC16846': "#BBBBBB",'MC16848':"#BBBBBB",'MC16850': "#BBBBBB",'MC16851': animal_colors[0], 'MC17319': animal_colors[1],
                      'MC17665': '#CCCCFF','MC17670': '#660033','MC17666': animal_colors[4], 'MC17668': animal_colors[5],'MC17669': '#BBBBBB', 
                      'MC19022': '#BBBBBB','MC19082': animal_colors[8],'MC19123': '#BBBBBB', 'MC19124': '#FF00FF', 'MC19130': '#00FFFF',
                      'MC19132': '#BBBBBB','MC19214': '#00FF00', 'MC18737': '#BBBBBB', 'MC19107': '#FA8072', 'VIV41330': '#777777', 
                      'VIV41329': '#777777', 'VIV41375': '#777777', 'VIV41376': '#777777', 'VIV41428': '#777777', 'VIV41429': '#777777',
                      'VIV41430': '#777777',
                      #IO fiber control
                      'VIV40958':animal_colors[4], 'VIV41344':animal_colors[5], 'VIV41345':animal_colors[6], 
                      #ChR2
                      'VIV42375': animal_colors[4],'VIV42376': animal_colors[5],'VIV42428': animal_colors[7],'VIV42429': animal_colors[8],
                      'VIV42430': animal_colors[9], 'VIV42906': animal_colors[2], 'VIV42907': animal_colors[3],'VIV42908':animal_colors[4], 'VIV42974':animal_colors[5],
                      'VIV42985':animal_colors[6], 'VIV42992': animal_colors[7],'VIV42987': animal_colors[8]}

'''
# All animals
included_animal_list = []

# JAWS histology-confirmed (18 animals)
#included_animal_list =  ['MC16848', 'MC16851', 'MC17319', 'MC17665', 'MC17666', 'MC17669', 'MC17670', 'MC19082', 'MC19124', 
 #                        'MC19214', 'VIV41329', 'VIV41330', 'VIV42375', 'VIV42428', 'VIV42429', 'VIV42430', 'VIV42376', 'MC19107']         # 'VIV42376' only tied, 'MC19107' only tied and Rfast

# ChR2 LE histology-confirmed (7 animals)
#included_animal_list =  ['VIV42906', 'VIV42908', 'VIV42974', 'VIV42985','VIV42987','VIV44766', 'VIV45372']  

#                        #]           # 'VIV42907',, 'VIV44765' only 100st    # 'VIV42906', 'VIV42908', 'VIV42974', 'VIV42985','VIV42987', only100st and 100sw

# Extra-zombies histology-checked UNIlateral injections
#included_animal_list_EZ = ['VIV47094', 'VIV47095', 'VIV47147', 'VIV47212', 'VIV49409', 'VIV49410', 'VIV49411', 'VIV49412', 
 #                       'VIV49574', 'VIV49939', 'VIV49940', 'VIV49931', 'VIV49933', 'VIV49934', 'VIV50051']

#included_animal_list_EZ_double = ['VIV49935', 'VIV49941', 'VIV50033', 'VIV50034', 'VIV50052']

# Extra-zombie animals info
LexpEZ = ['VIV47094', 'VIV47095', 'VIV47147', 'VIV47212', 'VIV49409', 'VIV49410', 'VIV49411', 'VIV49412']
RexpEZ = ['VIV49574', 'VIV49939', 'VIV49940', 'VIV49931', 'VIV49933', 'VIV49934', 'VIV50051']

session = 1
Ntrials = 17  #28    #56       # 28
stim_start = 5 #9  #18 #9
split_start = 5   #9 #18        #9
stim_duration = 10  #10  #20      #8
split_duration = 12 #10 #20         #8
if any('split' in path for path in paths):
    intervals_split_stim = {'split': [split_start, split_duration]}
elif any('stim' in path or 'th' in path for path in paths):
    intervals_split_stim = {'stim': [stim_start, stim_duration]}
plot_rig_signals = 0
print_plots = 1
print_plots_multi_session = 1
bs_bool = 1
control_ses = 'left'
control_path = []       #'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\15092023 split left fast control\\']      #'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\14092023 split right fast control\\'] #  'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Split belt sessions\\15092023 split left fast control\\']   #]         # This should be a list; if empty, we have no control (e.g. in tied sessions)
control_filename = 'split_'+control_ses+'_fast_control_params_sym_bs.npy'
paw_colors = ['red', 'magenta', 'blue', 'cyan']
paw_otrack = 'FR'
paws = ['FR', 'HR', 'FL', 'HL']
import online_tracking_class
import locomotion_class
otrack_classes =  []
locos = []
paths_save = []
param_sym_multi = {}
path_index = 0
merge_RLfast_sessions = 0 
for path in paths:
    #included_animal_list = []
    print("Analysing..........................", path)
    pixel_to_mm = 1/3.3            # real-time setup
    floor_factor = 268
    if 'WT' in path or 'Miniscopes' in path:
        pixel_to_mm = 1/1.955   # Dana's setup              1/1.98          # Jovin's     
        floor_factor = 152         
        included_animal_list = []
    elif 'LATinj' in path:
        included_animal_list = included_animal_list_EZ
    elif 'RLinj' in path:
        included_animal_list = included_animal_list_EZ_double
        merge_RLfast_sessions = 1             # This will merge the right and left fast sessions for the same animal, that should be element 2nd and 3rd in the list of paths
    
        
    # pixel_to_mm = 1/3.3            # real-time setup   
    #self.pixel_to_mm = 1/1.955            # Dana's setup
    otrack_classes.append(online_tracking_class.otrack_class(path))
    locos.append(locomotion_class.loco_class(path, pixel_to_mm, floor_factor))
    paths_save.append(path + 'grouped output\\')       #\\histo_filtered\\')                #')       #\\
    if not os.path.exists(path + 'grouped output\\'):     #\\histo_filtered\\'):
        os.mkdir(path + 'grouped output\\')       #\\histo_filtered\\')

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

    # Run the single animal analysis for each of the sessions in paths
    if single_animal_analysis:
        trials = otrack_classes[path_index].get_trials()
        # READ CAMERA TIMESTAMPS AND FRAME COUNTER
        [camera_timestamps_session, camera_frames_kept, camera_frame_counter_session] = otrack_classes[path_index].get_session_metadata(plot_rig_signals)

        # READ SYNCHRONIZER SIGNALS
        [timestamps_session, frame_counter_session, trial_signal_session, sync_signal_session, laser_signal_session, laser_trial_signal_session] = otrack_classes[path_index].get_synchronizer_data(camera_frames_kept, plot_rig_signals)

        # READ ONLINE DLC TRACKS
        otracks = otrack_classes[path_index].get_otrack_excursion_data(timestamps_session)
        [otracks_st, otracks_sw] = otrack_classes[path_index].get_otrack_event_data(timestamps_session)

        # READ OFFLINE DLC TRACKS
        [offtracks_st, offtracks_sw] = otrack_classes[path_index].get_offtrack_event_data(paw_otrack, locos[path_index], animal, session, timestamps_session)

        # READ OFFLINE PAW EXCURSIONS
        final_tracks_trials = otrack_classes[path_index].get_offtrack_paws(locos[path_index], animal, session)

        # PROCESS SYNCHRONIZER LASER SIGNALS
        laser_on = otrack_classes[path_index].get_laser_on(laser_signal_session, timestamps_session)

        # ACCURACY OF LIGHT ON
        laser_hits = np.zeros(len(trials))
        laser_incomplete = np.zeros(len(trials))
        laser_misses = np.zeros(len(trials))
        for count_t, trial in enumerate(trials):
            [full_hits, incomplete_hits, misses] = otrack_classes[path_index].get_hit_laser_synch(trial, laser_event, offtracks_st, offtracks_sw, laser_on, final_tracks_trials, timestamps_session, 0)
            laser_hits[count_t] = full_hits
            laser_incomplete[count_t] = incomplete_hits
            laser_misses[count_t] = misses
        # plot summaries
        fig, ax = plt.subplots(tight_layout=True, figsize=(5,3))
        ax.bar(trials, laser_hits, color='green')
        ax.bar(trials, laser_incomplete, bottom = laser_hits, color='orange')
        ax.bar(trials, laser_misses, bottom = laser_hits + laser_incomplete, color='red')
        ax.set_title(laser_event + ' misses')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(os.path.join(paths_save[path_index], 'laser_on_accuracy_' + laser_event + '.png'))
        fig, ax = plt.subplots(tight_layout=True, figsize=(5,3))
        rectangle = plt.Rectangle((split_start - 0.5, 0), split_duration, 50, fc='lightblue', alpha=0.3)
        ax.axvline(x = stim_start-0.5, color = 'k', linestyle = '-', linewidth=0.5)
        ax.axvline(x = stim_start+stim_duration+0.5, color = 'k', linestyle = '-', linewidth=0.5)
        plt.gca().add_patch(rectangle)
        ax.plot(trials, (laser_hits/(laser_hits+laser_misses+laser_incomplete))*100, '-o', color='green')
        ax.plot(trials, (laser_incomplete/(laser_hits+laser_misses+laser_incomplete))*100, '-o', color='orange')
        ax.set_title(laser_event + ' accuracy')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(os.path.join(paths_save[path_index], 'laser_on_accuracy_values_' + laser_event + '.png'))
    

    if single_animal_analysis == 0:
        # FOR EACH SESSION SEPARATE CALCULATION AND PLOT SAVING
        # GAIT PARAMETERS ACROSS TRIALS
        param_gait_name = ['coo', 'step_length', 'double_support', 'coo_stance', 'swing_length', 'stride_duration', 'swing_duration', 'stance_duration', 'swing_velocity','stance_speed','body_center_x_stride','body_speed_x','duty_factor','candence','phase_st']
        param_sym_labels = {'coo': 'Center of\noscillation (mm)', 'step_length': 'Step length (mm)', 'double_support': '% double support', 
                        'coo_stance': 'Spatial motor\noutput (mm)', 'swing_length': 'Swing length(mm)', 'phase_st': 'Stance phase', 'stance_speed': 'Stance speed'}
        param_sym_label = list(param_sym_labels.values())
        param_sym_name = list(param_sym_labels.keys())
        param_sym = np.zeros((len(param_sym_name), len(animal_list), Ntrials))
        param_sym[:] = np.NaN
        param_paw = np.zeros((len(param_sym_name), len(animal_list), 4, Ntrials))
        param_paw[:] = np.nan
        param_phase = np.zeros((4, len(animal_list), Ntrials))
        param_phase[:] = np.nan
        stance_speed = np.zeros((4, len(animal_list), Ntrials))
        stance_speed[:] = np.NaN
        st_strides_trials = []
        param_gait = np.zeros((len(param_gait_name), len(animal_list), Ntrials))
        param_gait[:] = np.NaN
        for count_animal, animal in enumerate(animal_list):
            session = int(session_list[count_animal])
            #TODO: check if this filelist needs to be emptied first!
            filelist = locos[path_index].get_track_files(animal, session)
            for f in filelist:
                count_trial = int(f.split('DLC')[0].split('_')[-1])-1      # Get trial number from file name, to spot any missing trial; parameters for remaining ones will stay to NaN
                [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = locos[path_index].read_h5(f, 0.9, 0)
                [st_strides_mat, sw_pts_mat] = locos[path_index].get_sw_st_matrices(final_tracks, 1)
                st_strides_trials.append(st_strides_mat)
                paws_rel = locos[path_index].get_paws_rel(final_tracks, 'X')
                for count_p, param in enumerate(param_sym_name):
                    param_mat = locos[path_index].compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param)
                    if param == 'stance_speed':
                        for p in range(4):
                            stance_speed[p, count_animal, count_trial] = np.nanmean(param_mat[p])
                    elif param == 'step_length':
                        param_sym[count_p, count_animal, count_trial] = np.nanmean(param_mat[0]) - np.nanmean(param_mat[2])
                    else:
                        param_sym[count_p, count_animal, count_trial] = np.nanmean(param_mat[0])-np.nanmean(param_mat[2])

                    if param == 'phase_st':
                        for p in range(4):
                            # Check for invalid strides: if more than 50% of the values are NaN, discard the trial
                            if np.sum(np.isnan(param_mat[3][count_paw])) > 0.5*len(param_mat[3][count_paw]):
                                param_phase[p, count_animal, count_trial] = np.nan
                            else:
                                param_phase[p, count_animal, count_trial] = (st.circmean(param_mat[3][count_paw], low=0, high=2*np.pi, nan_policy='omit'))

                    elif param == 'stance_speed':
                        for p in range(4):
                            stance_speed[p, count_animal,count_trial] = np.nanmean(param_mat[p])
                    else:
                        param_sym[count_p, count_animal, count_trial] = np.nanmean(param_mat[0])-np.nanmean(param_mat[2])
                        '''
                        param_mat_sym = locos[path_index].compute_continuous_sym_gaitparam(param_mat, st_strides_mat, 'FR', 'FL')
                        
                        if (count_trial==7 or count_trial==8) and param_sym_name[count_p]=='double_support':
                            
                            fig1, ax1 = plt.subplots(figsize=(7, 10), tight_layout=True)
                            plt.plot(param_mat_sym, linewidth=2)
                            plt.title([param_sym_name[count_p], ' trial ', str(count_trial)])
                            plt.show()
                            # With moving avg
                            series = pd.Series(param_mat_sym)
                            # Compute a moving average with a window of 3, ignoring NaNs
                            window_size = 2
                            moving_avg = series.rolling(window=window_size, min_periods=1).mean()
                            if count_trial==7:
                                fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)
                            ax.plot(moving_avg, linewidth=2)
                            ax.set_title([param_sym_name[count_p], ' trial ', str(count_trial), ' moving avg'])
                            ax.set_ylim([-25,30])
                            print("avg DS!!!!!!!!!!!!!!!!!",np.nanmedian(param_mat_sym))
                            plt.show()
                            '''
                                           
                    for count_paw, paw in enumerate(paws):
                        param_paw[count_p, count_animal, count_paw,count_trial] = np.nanmean(param_mat[count_paw])

                if compare_baselines:
                    for count_p, param in enumerate(param_gait_name):
                        param_mat = locos[path_index].compute_gait_param(bodycenter, final_tracks, paws_rel, st_strides_mat, sw_pts_mat, param)
                        param_gait[count_p, count_animal, count_trial] = np.nanmedian(param_mat[0])

            if ('contra' in experiment_name and animal in RexpEZ) or ('ipsi' in experiment_name and animal in LexpEZ):  # To invert Lfast sessions for being comparable to Rfast sessions    
                param_sym[:, count_animal, :] = -param_sym[:, count_animal, :]
       
       
        # BASELINE SUBTRACTION OF PARAMETERS
        if bs_bool:
            param_sym_bs = np.zeros(np.shape(param_sym))
            param_paw_bs = np.zeros(np.shape(param_paw))
            for p in range(np.shape(param_sym)[0]-1):
                if param_sym_name[p] == 'phase_st':
                    for a in range(np.shape(param_sym)[1]):
                        # Compute baseline and subtract
                        for count_paw in range(4):
                            if stim_start == split_start:
                                bs_paw_mean = st.circmean(param_paw[p, a, count_paw, :stim_start-1], nan_policy='omit')
                            if stim_start < split_start:
                                bs_paw_mean = st.circmean(param_paw[p, a, count_paw, stim_start-1:split_start-1], nan_policy='omit')
                            param_paw_bs[p, a, count_paw, :] = param_paw[p, a, count_paw, :] - bs_paw_mean

                            

                            # Put between -pi and pi
                            param_paw_bs[p, a, count_paw, :] = np.mod(param_paw_bs[p, a, count_paw, :] + np.pi, 2*np.pi) - np.pi


                else:
                    for a in range(np.shape(param_sym)[1]):
                        # Compute baseline
                        if stim_start == split_start:
                            bs_mean = np.nanmean(param_sym[p, a, :stim_start-1])
                            bs_paw_mean = np.nanmean(param_paw[p, a, count_paw, :stim_start-1])
                        if stim_start < split_start:
                            bs_mean = np.nanmean(param_sym[p, a, stim_start-1:split_start-1])
                            bs_paw_mean = np.nanmean(param_paw[p, a, count_paw, stim_start-1:split_start-1])
                        # Subtract
                        param_sym_bs[p, a, :] = param_sym[p, a, :] - bs_mean
                        for count_paw in range(4):
                            param_paw_bs[p, a, count_paw, :] = param_paw[p, a, count_paw, :] - bs_paw_mean
        else:
            if param_sym_name[p] == 'phase_st':
                #TODO: check if this is correct!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                param_sym_bs = st.circmean(param_phase[0,:,:], low=0, high=2*np.pi, nan_policy='omit') - st.circmean(param_phase[2,:,:], low=0, high=2*np.pi, nan_policy='omit')
            else:
                param_sym_bs = param_sym

        if any('right' in element for element in experiment_names) and any('left' in element for element in experiment_names) and 'left' in experiment_name:      # If we are comparing left and right we will have them both in experiment names
           param_sym_bs = -param_sym_bs

            
        # Compare baseline symmetry with and without stim
        if compare_baselines:
            param_no_stim = np.zeros((len(param_sym_name)+len(param_gait_name)-1, len(animal_list), stim_start-1))          # Stance speed is not considered in the params
            param_no_stim[:] = np.NaN
            param_stim = np.zeros((len(param_sym_name)+len(param_gait_name)-1, len(animal_list), stim_start-1))
            param_stim[:] = np.NaN
            # Symmetry parameters
            for p in range(np.shape(param_sym)[0]-1):
                for a in included_animals_id:            #range(np.shape(param_sym)[1]):
                    param_no_stim[p, a, :] = param_sym[p, a, :stim_start-1]
                    param_stim[p, a, :] = param_sym[p, a, stim_start-1:stim_start+stim_duration-1]        # :8]
            # Gait parameters
            start_ind=np.shape(param_sym)[0]-1
            for p in range(np.shape(param_gait)[0]):
                for a in included_animals_id:           #range(np.shape(param_gait)[1]):
                    param_no_stim[start_ind+p, a, :] = param_gait[p, a, :stim_start-1]
                    param_stim[start_ind+p, a, :] = param_gait[p, a, stim_start-1:stim_start+stim_duration-1]           # :8]
            

            # Plot and compare
            import math
            param_names = param_sym_name[:-1]+param_gait_name
            fig_stim_cmp, ax_stim_cmp = plt.subplots(3,5)           # len(param_sym_name)+len(param_gait_name))
            rc = [[0,0],[0,1],[0,2],[0,3],[0,4],[1,0],[1,1],[1,2],[1,3],[1,4],[2,0],[2,1],[2,2],[2,3],[2,4]]
            for p in range(np.shape(param_stim)[0]):
                ax_stim_cmp[rc[p][0],rc[p][1]].bar([0,1], [np.nanmean(np.nanmean(param_no_stim[p,:,:], axis=1)), np.nanmean(np.nanmean(param_stim[p,:,:], axis=1))], yerr=[np.nanstd(np.nanmean(param_no_stim[p,:,:], axis=1)),np.nanstd(np.nanmean(param_stim[p,:,:], axis=1))], align='center', color=['gray', experiment_colors_dict[experiment_name]], alpha=0.5, ecolor='black', capsize=6)
                ax_stim_cmp[rc[p][0],rc[p][1]].plot([np.nanmean(param_no_stim[p,:,:], axis=1), np.nanmean(param_stim[p,:,:], axis=1)],'-o', markersize=2, markeredgecolor='black', color='black', linewidth=0.5, markerfacecolor='none')
                ax_stim_cmp[rc[p][0],rc[p][1]].set_title(param_names[p], size=9)
                ax_stim_cmp[rc[p][0],rc[p][1]].set_xticks([])
            fig_stim_cmp.tight_layout()
            if print_plots:
                if not os.path.exists(paths_save[path_index]):
                    os.mkdir(paths_save[path_index])
                
                plt.savefig(paths_save[path_index] + 'compare_baselines', dpi=128)
                

        for p in range(np.shape(param_sym)[0] - 1):
            # Plot learning curve for individual animals
            fig = pf.plot_learning_curve_ind_animals(param_sym_bs, p, param_sym_labels, animal_list, animal_colors_dict, intervals=intervals_split_stim)
            # Save plot
            if print_plots:
                pf.save_plot(fig, paths_save[path_index], param_sym_name[p], plot_name='ind_animals', bs_bool=bs_bool)
                
        plt.close('all')

    # PLOT ANIMAL AVERAGE with INDIVIDUAL ANIMALS FOR EACH SESSION
    param_sym_multi[path] = {}
    if single_animal_analysis == 0:
        for p in range(np.shape(param_sym)[0]):
            param_sym_bs_ave = param_sym_bs[p, included_animals_id, :]
            fig = pf.plot_learning_curve_ind_animals_avg(param_sym_bs_ave, p, param_sym_labels, animal_list, [included_animal_list, included_animals_id],
                                                         [animal_colors_dict, experiment_colors_dict], experiment_name, intervals=intervals_split_stim, 
                                                         ranges=[uniform_ranges, axes_ranges])
            # Save plot
            if print_plots:
                pf.save_plot(fig, paths_save[path_index], param_sym_name[p], plot_name='average', bs_bool=bs_bool)
                
            # Save param_sym for multi-session plot (in case we have multiple sessions to analyse/plot)
            param_sym_multi[path][p] = param_sym_bs_ave
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
                plt.savefig(paths_save[path_index] + animal_list[a] + '_stancespeed', dpi=96)
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
                    plt.savefig(paths_save[path_index] + animal_list[a] + '_sl_sym_continuous', dpi=96)

    path_index = path_index+1
    
#included_animal_list = ['VIV41330', 'VIV41329']   #               'MC18737','MC19107']         #[ 'MC19022','MC19082','MC19123','MC19124','MC19214']  

#included_animals_id = [animal_list.index(i) for i in included_animal_list]
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
    experiment_colors_dict['contra fast right'] = 'darkblue'
    included_animal_list = included_animal_list*2  # We have the same animals in both sessions
    included_animals_id = included_animals_id + [i + len(included_animals_id) for i in included_animals_id]  # We have the same animals in both sessions, so we just need to offset the indices for the second session
if single_animal_analysis==0:
    # Determine the maximum number of animals across all paths, to handle the case of different number of animals in each path
    max_animals = max(param_sym_multi[path][p].shape[0] for path in paths)

    for p in range(np.shape(param_sym)[0] - 1):
        fig_multi = pf.plot_learning_curve_avg_compared(param_sym_multi, p, param_sym_labels, [included_animal_list, included_animals_id], experiment_colors_dict, experiment_names, intervals=intervals_split_stim, ranges=[uniform_ranges, axes_ranges])
        
        if print_plots:
            pf.save_plot(fig_multi, paths_save[0], param_sym_name[p], plot_name='average_multi_session', bs_bool=bs_bool)



        # LEARNING PARAMETERS (bar plots) - each one will be num_experiments x num_animals
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
        if split_duration==0:
            split_start = stim_start
            split_duration = stim_duration


        path_index = 0  
        
        if merge_RLfast_sessions:
            paths = param_sym_multi.keys()  # If we merge RL fast sessions, we have less paths to consider
            experiment_names = [name for name in experiment_names if any(name in key for key in param_sym_multi.keys())]
        current_experiment_colors = [experiment_colors_dict[key] for key in experiment_names if key in experiment_names]
        
        for path in paths:
            
            # Flip signs to have good learning always positive
            #if (param_sym_name[p] == 'double_support' and control_ses == 'right') or ((param_sym_name[p] == 'step_length' or param_sym_name[p] == 'coo') and control_ses == 'left'):
            #    initial_error.append(np.nanmean(param_sym_multi[path][p][:,split_start-1:split_start+1], axis=1))  
            #    learning.append(-(np.nanmean(param_sym_multi[path][p][:,split_start+split_duration-3:split_start+split_duration-1], axis=1)-np.nanmean(param_sym_multi[path][p][:,split_start-1:split_start+1], axis=1)))
            #    aftereffect.append(-np.nanmean(param_sym_multi[path][p][:,split_start+split_duration-1:split_start+split_duration+1], axis=1))
            #elif ((param_sym_name[p] == 'step_length' or param_sym_name[p] == 'coo') and control_ses == 'right') or (param_sym_name[p] == 'double_support' and control_ses == 'left'):
            #    initial_error.append(-np.nanmean(param_sym_multi[path][p][:,split_start-1:split_start+1], axis=1))         
            #    learning.append(np.nanmean(param_sym_multi[path][p][:,split_start+split_duration-3:split_start+split_duration-1], axis=1)-np.nanmean(param_sym_multi[path][p][:,split_start-1:split_start+1], axis=1))
            #    aftereffect.append(np.nanmean(param_sym_multi[path][p][:,split_start+split_duration-1:split_start+split_duration+1], axis=1))
           
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
                print(aftereffect[0], aftereffect[path_index])
                if max_animals!=len(current_initial_error) or max_animals!=np.sum(~np.isnan(initial_error[0])):     # Samples have different size, so unpaired stats with the control (that is first row)
                    stat_initial_error.append(st.mannwhitneyu(initial_error[0], initial_error[path_index], alternative='two-sided').pvalue<significance_threshold)
                    stat_learning.append(st.mannwhitneyu(learning[0], learning[path_index], alternative='two-sided').pvalue<significance_threshold) 
                    stat_aftereffect.append(st.mannwhitneyu(aftereffect[0], aftereffect[path_index], alternative='two-sided', nan_policy='omit').pvalue<significance_threshold)
                    print(['param ', param_sym_name[p], st.mannwhitneyu(aftereffect[0], aftereffect[path_index], alternative='two-sided', nan_policy='omit')])
                else:
                    stat_initial_error.append(st.wilcoxon(initial_error[0], initial_error[path_index], nan_policy='omit').pvalue<significance_threshold)
                    stat_learning.append(st.wilcoxon(learning[0], learning[path_index], nan_policy='omit').pvalue<significance_threshold)
                    stat_aftereffect.append(st.wilcoxon(aftereffect[0], aftereffect[path_index], nan_policy='omit').pvalue<significance_threshold)
                    print(['param ', param_sym_name[p], st.wilcoxon(aftereffect[0], aftereffect[path_index], nan_policy='omit')])
            path_index+=1

        learning_sym_change=100*np.divide(np.array(learning),np.array(initial_error))
        initial_error_array = np.array(initial_error)
        initial_error_array[np.absolute(initial_error_array)<1] = np.nan
        aftereffect_sym_change=100*np.divide(np.array(aftereffect),np.absolute(initial_error_array))
        if compute_statistics and path_index>0:
           for path_index in range(1,len(paths)):
                if max_animals!=len(current_initial_error):     # Samples have different size, so unpaired stats
                    stat_learning_sym_change.append(st.mannwhitneyu(learning_sym_change[0], learning_sym_change[path_index], alternative='two-sided').pvalue<significance_threshold)
                    stat_aftereffect_sym_change.append(st.mannwhitneyu(aftereffect_sym_change[0], aftereffect_sym_change[path_index], alternative='two-sided').pvalue<significance_threshold)
                else:
                    stat_learning_sym_change.append(st.wilcoxon(learning_sym_change[0], learning_sym_change[path_index], nan_policy='omit').pvalue<significance_threshold)
                    stat_aftereffect_sym_change.append(st.wilcoxon(aftereffect_sym_change[0], aftereffect_sym_change[path_index], nan_policy='omit').pvalue<significance_threshold)
       
        
        # Bar plot
        fig_bar, ax_bar = plt.subplots(2,3)
        bars=ax_bar[0,0].bar([0]+list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)), np.nanmean(initial_error, axis=1), yerr=np.nanstd(initial_error, axis=1)/np.sqrt(len(learning)), align='center', alpha=0.5, color=current_experiment_colors, ecolor='black', capsize=6)
        #ax_bar[0,0].plot(initial_error,'-o', markersize=2, markeredgecolor='black', color='black', linewidth=0.5, markerfacecolor='none')
        ax_bar[0,0].set_ylabel(param_sym_name[p]+' (mm)')
        ax_bar[0,0].set_title('init. error', size=9)
        ax_bar[0,1].bar([0]+list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)), np.nanmean(learning, axis=1), yerr=np.nanstd(learning, axis=1)/np.sqrt(len(learning)), align='center', alpha=0.5, color=current_experiment_colors, ecolor='black', capsize=6)
        ax_bar[0,1].set_title('late-early', size=9)
        ax_bar[0,2].bar([0]+list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)), np.nanmean(aftereffect, axis=1), yerr=np.nanstd(aftereffect, axis=1)/np.sqrt(len(learning)), align='center', alpha=0.5, color=current_experiment_colors, ecolor='black', capsize=6)
        #ax_bar[0,2].plot(aftereffect,'-o', markersize=2, markeredgecolor='black', color='black', linewidth=0.5, markerfacecolor='none')
        ax_bar[0,2].set_title('aftereffect', size=9)
        ax_bar[1,1].bar([0]+list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)), np.nanmean(learning_sym_change, axis=1), yerr=np.nanstd(learning_sym_change, axis=1)/np.sqrt(len(learning)), align='center', alpha=0.5, color=current_experiment_colors, ecolor='black', capsize=6)
        #ax_bar[1,1].plot(learning_sym_change,'-o', markersize=2, markeredgecolor='black', color='black', linewidth=0.5, markerfacecolor='none')
        ax_bar[1,1].set_title('% change late-early', size=9)
        ax_bar[1,1].set_ylabel('% sym change')
        if not np.isnan(aftereffect_sym_change).all():
            ax_bar[1,2].bar([0]+list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)), np.nanmean(aftereffect_sym_change, axis=1), yerr=np.nanstd(aftereffect_sym_change, axis=1)/np.sqrt(len(learning)), align='center', alpha=0.5, color=current_experiment_colors, ecolor='black', capsize=6)
        #ax_bar[1,2].plot(aftereffect_sym_change,'-o', markersize=2, markeredgecolor='black', color='black', linewidth=0.5, markerfacecolor='none')
        ax_bar[1,2].set_title('% change aftereffect', size=9)
        if scatter_single_animals:
            # Add single animal data
            for a in range(np.sum(~np.isnan(initial_error[1]))):
                ax_bar[0,0].plot(list(range(1,len(control_path)+len(paths)+1)),np.array(initial_error)[:,a],'-o', markersize=4, markerfacecolor=animal_colors_dict[animal_list[included_animals_id[a]]], color=animal_colors_dict[animal_list[included_animals_id[a]]], linewidth=1)
                ax_bar[0,1].plot(list(range(1,len(control_path)+len(paths)+1)),np.array(learning)[:,a],'-o', markersize=4, markerfacecolor=animal_colors_dict[animal_list[included_animals_id[a]]], color=animal_colors_dict[animal_list[included_animals_id[a]]], linewidth=1)
                ax_bar[0,2].plot(list(range(1,len(control_path)+len(paths)+1)),np.array(aftereffect)[:,a],'-o', markersize=4, markerfacecolor=animal_colors_dict[animal_list[included_animals_id[a]]], color=animal_colors_dict[animal_list[included_animals_id[a]]], linewidth=1)
                ax_bar[1,1].plot(list(range(1,len(control_path)+len(paths)+1)),np.array(learning_sym_change)[:,a],'-o', markersize=4, markerfacecolor=animal_colors_dict[animal_list[included_animals_id[a]]], color=animal_colors_dict[animal_list[included_animals_id[a]]], linewidth=1)
                ax_bar[1,2].plot(list(range(1,len(control_path)+len(paths)+1)),np.array(aftereffect_sym_change)[:,a],'-o', markersize=4, markerfacecolor=animal_colors_dict[animal_list[included_animals_id[a]]], color=animal_colors_dict[animal_list[included_animals_id[a]]], linewidth=1)
                #ax_bar[0,0].plot(list(range(1,len(control_path)+len(paths)+1)),np.array(initial_error)[:,a],'-', color='gray', linewidth=1)
                #ax_bar[0,1].plot(list(range(1,len(control_path)+len(paths)+1)),np.array(learning)[:,a],'-', color='gray', linewidth=1)
                #ax_bar[0,2].plot(list(range(1,len(control_path)+len(paths)+1)),np.array(aftereffect)[:,a],'-', color='gray', linewidth=1)
                #ax_bar[1,1].plot(list(range(1,len(control_path)+len(paths)+1)),np.array(learning_sym_change)[:,a],'-', color='gray', linewidth=1)
                #ax_bar[1,2].plot(list(range(1,len(control_path)+len(paths)+1)),np.array(aftereffect_sym_change)[:,a],'-', color='gray', linewidth=1)
        
        # Add zero line and set ranges
        for ax in ax_bar.flatten():
            ax.axhline(y = 0, color = 'k', linestyle = '--', linewidth=0.5)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if uniform_ranges:
                ax.set(ylim= bars_ranges[param_sym_name[p]])      #   [-4.5,max(abs(np.array(axes_ranges[param_sym_name[p]])))])
        if uniform_ranges:      # Increased ranges for _sym_change parameters
            ax_bar[1,1].set(ylim= list(30*np.array(bars_ranges[param_sym_name[p]])))

        # Add plots Statistics
        if compute_statistics and path_index>0:
            #ax_bar[0,0].plot(list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)),[(bars_ranges[param_sym_name[p]][1]-0.5)*i if i==1 else math.nan*i for i in stat_initial_error ],'*', color='black')
            
            ax_bar[0,0].plot(list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)),[max(max(np.nanmean(initial_error, axis=1)+np.nanstd(initial_error, axis=1)),0)*i if i==1 else math.nan*i for i in stat_initial_error ],'*', color='black')
            ax_bar[0,1].plot(list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)),[max(max(np.nanmean(learning, axis=1)+np.nanstd(learning, axis=1)),0)*i if i==1 else math.nan*i for i in stat_learning ],'*', color='black')
            ax_bar[0,2].plot(list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)),[max(max(np.nanmean(aftereffect, axis=1)+np.nanstd(aftereffect, axis=1)),0)*i if i==1 else math.nan*i for i in stat_aftereffect ],'*', color='black')
            ax_bar[1,1].plot(list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)),[max(max(np.nanmean(learning_sym_change, axis=1)+np.nanstd(learning_sym_change, axis=1)),0)*i if i==1 else math.nan*i for i in stat_learning_sym_change ],'*', color='black')
            ax_bar[1,2].plot(list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)),[max(max(np.nanmean(aftereffect_sym_change, axis=1)+np.nanstd(aftereffect_sym_change, axis=1)),0)*i if i==1 else math.nan*i for i in stat_aftereffect_sym_change ],'*', color='black')
            print('stat '+param_sym_name[p]+': '+str(stat_aftereffect))
            
        for ax in ax_bar.flatten():
            ax.set_xticks([])
        fig_bar.delaxes(ax_bar[1,0])
        fig_bar.suptitle(param_sym_name[p])
        fig_bar.tight_layout()
        fig_bar.legend(bars, experiment_names,
           loc="lower left",   
           borderaxespad=3
           )
        

        if print_plots_multi_session:
            if not os.path.exists(paths_save[0]):
                os.mkdir(paths_save[0])
            if scatter_single_animals:
                plt.savefig(paths_save[0] + param_sym_name[p] + '_sym_bs_average_with_control_multi_session_barplot_scatterplot', dpi=96)      
            else:
                plt.savefig(paths_save[0] + param_sym_name[p] + '_sym_bs_average_with_control_multi_session_onlybarplot', dpi=96)     
    
        # Plot aftereffect alone
        to_plot_separately = aftereffect
        name_to_plot_separately = 'after effect'
        stat_to_plot_separately = stat_aftereffect
        fig_separate, ax_separate = plt.subplots(figsize=(7, 10), tight_layout=True)
        ax_separate.bar([0]+list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)), np.nanmean(to_plot_separately, axis=1), yerr=np.nanstd(to_plot_separately, axis=1)/np.sqrt(len(to_plot_separately)), align='center', alpha=0.5, color=current_experiment_colors, ecolor='black', capsize=6)
        if scatter_single_animals:
            # Add single animal data
            for a in range(np.sum(~np.isnan(to_plot_separately[1]))):
                ax_separate.plot(list(range(1,len(control_path)+len(paths)+1)),np.array(to_plot_separately)[:,a],'-o', markersize=8, markerfacecolor=animal_colors_dict[animal_list[included_animals_id[a]]], color=animal_colors_dict[animal_list[included_animals_id[a]]], linewidth=1)
        ax_separate.axhline(y = 0, color = 'k', linestyle = '--', linewidth=0.5)
        ax_separate.spines['right'].set_visible(False)
        ax_separate.spines['top'].set_visible(False)
        ax_separate.set_ylabel(param_sym_label[p]+' asymmetry '+name_to_plot_separately, fontsize = 28)
        if uniform_ranges:
            ax_separate.set(ylim= bars_ranges[param_sym_name[p]])  
        ax_separate.set_xticks([0]+list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)))
        ax_separate.set_xticklabels(experiment_names)

        # Plot aftereffect alone - only scatter and avg value
        to_plot_separately_onlyscatter = aftereffect
        name_to_plot_separately_onlyscatter = 'after effect'
        stat_to_plot_separately_onlyscatter = stat_aftereffect
        fig_separate_onlyscatter, ax_separate_onlyscatter = plt.subplots(figsize=(7, 10), tight_layout=True)
        x_aftereffect = list(range(1,len(control_path)+len(paths)+1))
        count_exp = 0
        # Add single animal data
        if 'WT' in experiment_names:            # If we compare to WT not injected, we do not connect those animals cos they are not the same
            for a in range(len(to_plot_separately_onlyscatter[1])):
                ax_separate_onlyscatter.plot(x_aftereffect[1:], np.array(to_plot_separately_onlyscatter)[1:, a], '-', color='lightgray', linewidth=1)
        else:
            for a in range(len(to_plot_separately_onlyscatter[0])):
                ax_separate_onlyscatter.plot(x_aftereffect, np.array(to_plot_separately_onlyscatter)[:, a], '-', color='lightgray', linewidth=1)
        
        for x in x_aftereffect:
            ax_separate_onlyscatter.scatter([x] * len(to_plot_separately_onlyscatter[count_exp][:]), to_plot_separately_onlyscatter[count_exp][:], s=60, c=current_experiment_colors[count_exp])
            # Add avg value
            ax_separate_onlyscatter.plot([x-0.15, x+0.15], [np.nanmean(to_plot_separately_onlyscatter[count_exp][:]), np.nanmean(to_plot_separately_onlyscatter[count_exp][:])], color=current_experiment_colors[count_exp], linewidth=4)
            count_exp+=1
        
            
          
        ax_separate_onlyscatter.axhline(y = 0, color = 'k', linestyle = '--', linewidth=0.5)
        ax_separate_onlyscatter.spines['right'].set_visible(False)
        ax_separate_onlyscatter.spines['top'].set_visible(False)
        ax_separate_onlyscatter.set_ylabel(param_sym_label[p]+' asymmetry '+name_to_plot_separately, fontsize = 28)
        if uniform_ranges:
            ax_separate_onlyscatter.set(ylim= bars_ranges[param_sym_name[p]])  
        ax_separate_onlyscatter.set(xlim=[x_aftereffect[0]-0.5, x_aftereffect[-1]+0.5])
        ax_separate_onlyscatter.set_xticks(x_aftereffect)
#        ax_separate_onlyscatter.set_xticklabels(['control','stance','swing'])
        
        # Add plots Statistics
        if compute_statistics:
            ax_separate_onlyscatter.plot(list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)),[max(max(np.nanmean(to_plot_separately, axis=1)+np.nanstd(to_plot_separately, axis=1)),0)*i if i==1 else math.nan*i for i in stat_to_plot_separately ],'*', color='black')
            
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        if not os.path.exists(paths_save[0]):
                os.mkdir(paths_save[0])
        plt.savefig(paths_save[0] + param_sym_name[p] + '_bar_scatterplot_same_color_'+name_to_plot_separately, dpi=120)      

        # Plot learning parameter alone
        to_plot_separately = learning
        name_to_plot_separately = 'adaptation'
        stat_to_plot_separately = stat_learning
        fig_separate, ax_separate = plt.subplots(figsize=(7, 10), tight_layout=True)
        ax_separate.bar([0]+list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)), np.nanmean(to_plot_separately, axis=1), yerr=np.nanstd(to_plot_separately, axis=1)/np.sqrt(len(to_plot_separately)), align='center', alpha=0.5, color=current_experiment_colors, ecolor='black', capsize=6)
        if scatter_single_animals:
            # Add single animal data
            for a in range(np.sum(~np.isnan((to_plot_separately[1])))):
                ax_separate.plot(list(range(1,len(control_path)+len(paths)+1)),np.array(to_plot_separately)[:,a],'-o', markersize=8, markerfacecolor=animal_colors_dict[animal_list[included_animals_id[a]]], color=animal_colors_dict[animal_list[included_animals_id[a]]], linewidth=1)
        ax_separate.axhline(y = 0, color = 'k', linestyle = '--', linewidth=0.5)
        ax_separate.spines['right'].set_visible(False)
        ax_separate.spines['top'].set_visible(False)
        ax_separate.set_ylabel(param_sym_label[p]+' asymmetry '+name_to_plot_separately, fontsize = 28)
        if uniform_ranges:
            ax_separate.set(ylim= bars_ranges[param_sym_name[p]])  
        ax_separate.set_xticks([0]+list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)))
        ax_separate.set_xticklabels(experiment_names)

        # Add plots Statistics
        if compute_statistics:
            ax_separate.plot(list(range(len(paths)+len(control_path)+1,(len(paths)+len(control_path))*2)),[max(max(np.nanmean(to_plot_separately, axis=1)+np.nanstd(to_plot_separately, axis=1)),0)*i if i==1 else math.nan*i for i in stat_to_plot_separately ],'*', color='black')
         
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        if not os.path.exists(paths_save[0]):
                os.mkdir(paths_save[0])
        plt.savefig(paths_save[0] + param_sym_name[p] + '_bar_scatterplot_'+name_to_plot_separately, dpi=1200)   
                  
     
