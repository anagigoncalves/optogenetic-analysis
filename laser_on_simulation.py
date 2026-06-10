import numpy as np
import matplotlib.pyplot as plt
import os

# Inputs
laser_event = 'stance'
trials_plot = np.arange(9, 19)  # trials with stimulation to check phase of laser
path = 'J:\\Experiments ChR2 RT\\20240521 chr2 rt ctx fiber low expression stance stim (swing cable th 200 75percent) 20mW\\'
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
    [otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session,
     laser_on] = otrack_class.load_processed_files(animal)

    laser_on['time_off'] = laser_on['time_on']+0.05 #add 50ms to laser on
    laser_on['frames_off'] = laser_on['frames_on'] + 17 #add 17 frames ~50ms

    # READ OFFLINE PAW EXCURSIONS
    [final_tracks_trials, st_strides_trials, sw_strides_trials] = otrack_class.get_offtrack_paws(loco, animal, np.int64(
        session_list[count_a]))
    final_tracks_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials,
                                                 'st-sw-st')
    # LASER ONSET AND OFFSET PHASE
    light_onset_phase_all = []
    light_offset_phase_all = []
    stim_nr_trials = np.zeros(len(trials_plot))
    stride_nr_trials = np.zeros(len(trials_plot))
    for count_t, trial in enumerate(trials_plot):
        [light_onset_phase, light_offset_phase, stim_nr, stride_nr] = \
            otrack_class.laser_presentation_phase_all(trial, trials, laser_event, offtracks_st, offtracks_sw, laser_on,
                                                      timestamps_session, final_tracks_phase, "FR")
        stim_nr_trials[count_t] = stim_nr
        stride_nr_trials[count_t] = stride_nr
        light_onset_phase_all.extend(light_onset_phase)
        light_offset_phase_all.extend(light_offset_phase)
    otrack_class.plot_laser_presentation_phase_hist(light_onset_phase_all, light_offset_phase_all,
                                                    16, path_save,
                                                    animal + '_' + laser_event + '_session_' + session_list[count_a],
                                                    True)
    plt.close('all')


