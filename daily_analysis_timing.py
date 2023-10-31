import numpy as np
import matplotlib.pyplot as plt
import os

# Inputs
laser_event = 'stance'
trials_plot = np.arange(9, 19) #trials with stimulation to check phase of laser
path = 'C:\\Users\\Ana\\Documents\\PhD\\Projects\\Online Stimulation Treadmill\\Experiments\\16062023 split right fast stance stim only split\\'
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)
path_save = path + 'grouped output\\'
if not os.path.exists(path + 'grouped output'):
    os.mkdir(path + 'grouped output')

# GET THE NUMBER OF ANIMALS AND THE SESSION ID
animal_session_list = loco.animals_within_session()
animal_list = []
for a in range(len(animal_session_list)):
    animal_list.append(animal_session_list[a][0])
session_list = []
for a in range(len(animal_session_list)):
    session_list.append(animal_session_list[a][1])

for count_a, animal in enumerate(animal_list):
    trials = otrack_class.get_trials(animal)
    # LOAD PROCESSED DATA
    [otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, laser_on] = otrack_class.load_processed_files(animal)
    # READ OFFLINE PAW EXCURSIONS
    [final_tracks_trials, st_strides_trials, sw_strides_trials] = otrack_class.get_offtrack_paws(loco, animal, np.int64(session_list[count_a]))
    final_tracks_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials,
                                                 'st-sw-st')
    # LASER ACCURACY
    tp_laser = np.zeros(len(trials))
    fp_laser = np.zeros(len(trials))
    tn_laser = np.zeros(len(trials))
    fn_laser = np.zeros(len(trials))
    precision_laser = np.zeros(len(trials))
    recall_laser = np.zeros(len(trials))
    f1_laser = np.zeros(len(trials))
    for count_t, trial in enumerate(trials):
        [tp_trial, fp_trial, tn_trial, fn_trial, precision_trial, recall_trial, f1_trial] = otrack_class.accuracy_laser_sync(trial, laser_event, offtracks_st, offtracks_sw, laser_on, final_tracks_trials, timestamps_session, 0)
        tp_laser[count_t] = tp_trial
        fp_laser[count_t] = fp_trial
        tn_laser[count_t] = tn_trial
        fn_laser[count_t] = fn_trial
        precision_laser[count_t] = precision_trial
        recall_laser[count_t] = recall_trial
        f1_laser[count_t] = f1_trial

    fig, ax = plt.subplots(tight_layout=True, figsize=(10, 7))
    rectangle = plt.Rectangle((trials_plot[0]+0.5, 0), trials_plot[-1]-trials_plot[0],
            1, fc='dimgrey',alpha=0.3)
    plt.gca().add_patch(rectangle)
    ax.plot(trials, tp_laser+tn_laser, marker='o', color='black', linewidth=2)
    ax.set_ylim([0, 1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(animal, fontsize=16)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    plt.savefig(path_save + animal + '_laser_performance_accuracy.png')

    #LASER ONSET AND OFFSET PHASE
    light_onset_phase_all = []
    light_offset_phase_all = []
    stim_nr_trials = np.zeros(len(trials_plot))
    stride_nr_trials = np.zeros(len(trials_plot))
    for count_t, trial in enumerate(trials_plot):
        [light_onset_phase, light_offset_phase, stim_nr, stride_nr] = \
            otrack_class.laser_presentation_phase(trial, trials, laser_event, offtracks_st, offtracks_sw, laser_on,
                                                  timestamps_session, final_tracks_phase, 0)
        stim_nr_trials[count_t] = stim_nr
        stride_nr_trials[count_t] = stride_nr
        light_onset_phase_all.extend(light_onset_phase)
        light_offset_phase_all.extend(light_offset_phase)
    otrack_class.plot_laser_presentation_phase(light_onset_phase_all, light_offset_phase_all, laser_event,
                    16, np.sum(stim_nr_trials), np.sum(stride_nr_trials), 0, 1,
                    path_save, animal+'_'+laser_event+'_'+session_list[count_a])
    plt.close('all')


