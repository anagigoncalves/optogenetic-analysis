import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as sig
from matplotlib.cm import ScalarMappable

# Inputs
laser_event = 'stance'  # 'swing' or 'stance'
if laser_event == 'stance':
    color_cond = 'orange'
    color_onset = 'orange'
    color_offset = 'darkorange'
if laser_event == 'swing':
    color_cond = 'green'
    color_onset = 'lightgreen'
    color_offset = 'green'

window_time = 0.05

#path = 'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\tied stance stim\\'
path='D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT\\LOW expression\\ALL_ANIMALS\\tied th200st IO 50ms\\'
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)
path_save = path + 'laser_phase\\'
if not os.path.exists(path_save):
    os.mkdir(path_save)

# GET THE NUMBER OF ANIMALS AND THE SESSION ID
animal_session_list = loco.animals_within_session()
animal_list = []
for a in range(len(animal_session_list)):
    animal_list.append(animal_session_list[a][0])
session_list = []
for a in range(len(animal_session_list)):
    session_list.append(animal_session_list[a][1])

#included_animal_list = ['MC16848', 'MC16851', 'MC17319', 'MC17665', 'MC17666', 'MC17669', 'MC17670', 'MC19082',
#                       'MC19124', 'MC19214', 'VIV41329', 'VIV41330', 'VIV42375', 'VIV42428', 'VIV42429',
#                      'VIV42430']   #, 'VIV42376', 'MC19107']   # jaws


# swing ['MC16848', 'MC16851', 'MC17319', 'MC17665', 'MC17666', 'MC17669', 'MC17670', 'MC19082',
    #                     'VIV41329', 'VIV41330', 'VIV42375', 'VIV42428', 'VIV42429',
   #                   'VIV42430', 'VIV42376']


# stance ['MC16848','MC17319', 'MC17665', 'MC17666', 'MC17669', 'MC17670', 'MC19082',
                #        'MC19124', 'MC19214', 'VIV41329', 'VIV41330', 'VIV42375', 'VIV42428', 'VIV42429',
              #          'VIV42430', 'VIV42376']          # 'MC16851','MC19107' missing to DO!!!  
included_animal_list = ['VIV42906', 'VIV42908', 'VIV42974', 'VIV42985','VIV42987','VIV44766', 'VIV45372']  # ChR2
#['MC16848', 'MC16851', 'MC17319', 'MC17665', 'MC17666', 'MC17669', 'MC17670', 'MC19082',
 #                       'MC19124', 'MC19214', 'VIV41329', 'VIV41330', 'VIV42375', 'VIV42428', 'VIV42429',
  #                      'VIV42430', 'VIV42376', 'MC19107']   # jaws

if len(included_animal_list) > 0:
    animal_list = included_animal_list

light_onset_phase_animals_hist = []
light_offset_phase_animals_hist = []
light_onset_time_animals_hist = []
light_offset_time_animals_hist = []
for count_a, animal in enumerate(animal_list):
    print('Processing ' + animal)
    trials = otrack_class.get_trials(animal)

    if len(trials) <= 24:
        trials_plot = np.arange(9, 17)     #trials with stimulation to check phase of laser
    else:
        trials_plot = np.arange(9, 19)
    # LOAD PROCESSED DATA
    # Check first if the current animal has processed files
    if not os.path.exists(os.path.join(path, 'processed files', animal)):
        print(f"No processed files found for {animal}. Skipping...")
        continue
    
    [otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, laser_on] = otrack_class.load_processed_files(animal)

    # READ OFFLINE PAW EXCURSIONS
    [final_tracks_trials, st_strides_trials, sw_strides_trials] = otrack_class.get_offtrack_paws(loco, animal, np.int64(session_list[count_a]))
    final_tracks_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials,
                                                'st-sw-st')
    # # LASER ACCURACY
    tp_laser = np.zeros(len(trials_plot))
    fp_laser = np.zeros(len(trials_plot))
    tn_laser = np.zeros(len(trials_plot))
    fn_laser = np.zeros(len(trials_plot))
    precision_laser = np.zeros(len(trials_plot))
    recall_laser = np.zeros(len(trials_plot))
    f1_laser = np.zeros(len(trials_plot))
    for count_t, trial in enumerate(trials_plot):
        [tp_trial, fp_trial, tn_trial, fn_trial, precision_trial, recall_trial, f1_trial] = otrack_class.accuracy_laser_sync(trial, laser_event, offtracks_st, offtracks_sw, laser_on, final_tracks_trials, timestamps_session, 0)
        tp_laser[count_t] = tp_trial
        fp_laser[count_t] = fp_trial
        tn_laser[count_t] = tn_trial
        fn_laser[count_t] = fn_trial
        precision_laser[count_t] = precision_trial
        recall_laser[count_t] = recall_trial
        f1_laser[count_t] = f1_trial
    #
    # fig, ax = plt.subplots(tight_layout=True, figsize=(10, 7))
    # ax.plot(trials_plot, tp_laser+tn_laser, marker='o', color='black', linewidth=2)
    # ax.set_ylim([0, 1])
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.set_title(animal, fontsize=16)
    # ax.set_ylabel('Accuracy', fontsize=14)
    # ax.set_ylabel('Accuracy', fontsize=14)
    # plt.savefig(path_save + animal + '_laser_performance_accuracy.png')

    fig, ax = plt.subplots(tight_layout=True, figsize=(10, 7))
    ax.plot(trials_plot, tp_laser+tn_laser, marker='o', color='black', linewidth=2)
    ax.set_ylim([0, 1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(animal, fontsize=16)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    plt.savefig(path_save + animal + '_'+laser_event +'_laser_performance_accuracy.png')

    #LASER ONSET AND OFFSET PHASE
    light_onset_phase_all = []
    light_offset_phase_all = []
    predicted_cspk_phase_all = []
    stim_nr_trials = np.zeros(len(trials_plot))
    stride_nr_trials = np.zeros(len(trials_plot))
    for count_t, trial in enumerate(trials_plot):
        [light_onset_phase, light_offset_phase, stim_nr, stride_nr] = \
            otrack_class.laser_presentation_phase_all(trial, trials, laser_event, offtracks_st, offtracks_sw, laser_on,
                                                timestamps_session, final_tracks_phase, "FR")
        [light_onset_phase, predicted_cspk_phase, stim_nr, stride_nr] = \
            otrack_class.predicted_cspk_phase_all(trial, trials, laser_event, offtracks_st, offtracks_sw, laser_on,
                                                timestamps_session, final_tracks_phase, "FR", cspk_mu=0.04, cspk_std=0.024)
        stim_nr_trials[count_t] = stim_nr
        stride_nr_trials[count_t] = stride_nr
        light_onset_phase_all.extend(light_onset_phase)
        light_offset_phase_all.extend(light_offset_phase)
    # Step-like histogram of stimulation phases
    otrack_class.plot_laser_presentation_phase_hist(light_onset_phase_all, light_offset_phase_all,
                    16, path_save, animal+'_session_'+session_list[count_a], 1, laser_event, hist_type='bar')
    # Heatmap histograms of stimulation phases
    otrack_class.plot_laser_presentation_phase_hist_heatmap(light_onset_phase_all, light_offset_phase_all, 16, color_cond,
                                path_save, animal+'_session_'+session_list[count_a]+'_heatmap', 1)
    light_onset_phase_animals_hist.append(light_onset_phase_all)
    light_offset_phase_animals_hist.append(light_offset_phase_all)

    #LASER ONSET AND OFFSET TIMES
    [light_onset_time, light_offset_time] = otrack_class.laser_presentation_time_hist(trials_plot, trials, laser_on, st_strides_trials,
                            sw_strides_trials, laser_event, window_time, 'FR', 0, color_onset, color_offset, 16, path_save, '')
    light_onset_time_animals_hist.append(light_onset_time)
    light_offset_time_animals_hist.append(light_offset_time)
    plt.close('all')

# Step-like histograms of stimulation phases for all animals
otrack_class.plot_laser_presentation_phase_hist_allanimals(light_onset_phase_animals_hist, light_offset_phase_animals_hist, 16, 1,
                                                           color_onset, color_offset, path_save,
                                                           'all_animals_hist', 1, laser_event)
# # Heatmap histograms of stimulation phases for all animals
# otrack_class.plot_laser_presentation_phase_hist_heatmap(light_onset_phase_all, light_offset_phase_all, 16, color_cond,
#                                                         path_save, 'stim_phase_session_' + session_list[
#                                                             count_a] + '_heatmap', 1)

# Step-like histograms of stimulation times for all animals
fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
ax.axvline(x=0, color='darkgray', linewidth=2)
for count_a in range(len(light_onset_time_animals_hist)):
    ax.hist(light_onset_time_animals_hist[count_a], histtype='step', color=color_onset, alpha=min(abs(1-(count_a)*0.1),1), linewidth=2)
    ax.hist(light_offset_time_animals_hist[count_a], histtype='step', color=color_offset, alpha=min(abs(1-(count_a)*0.1),1), linewidth=2)
ax.set_xticks([-0.1, -0.05, 0, 0.05, 0.1])
ax.set_xticklabels(['-100', '-50', '0', '50', '100'])
ax.set_xlabel('Time from stride event (ms)', fontsize=16)
ax.set_ylabel('Laser presentation\ncounts', fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim(-window_time-window_time*(1/3), window_time+window_time*(1/3))
ax.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(path_save + 'all_animals_time_hist')
plt.savefig(path_save + 'all_animals_time_hist.svg')

