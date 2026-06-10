import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.warnings.filterwarnings('ignore')

path = 'C:\\Users\\Ana\\Desktop\\\Data OPTO\\CM tests\\75percent\\'
condition = path.split('\\')[-2]
animals = ['MC18089', 'MC18090', 'MC18091', 'VIV40922', 'VIV40923', 'VIV40924']
colors_animals = ['blue', 'orange', 'darkgreen', 'crimson', 'purple', 'gray']
session = 1
if not os.path.exists(os.path.join(path, 'plots')):
    os.mkdir(os.path.join(path, 'plots'))
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)

path_save = 'C:\\Users\\Ana\\Desktop\\temp plots\\'

frac_strides_st_all = np.zeros((len(animals), 5, 2))
frac_strides_sw_all = np.zeros((len(animals), 5, 2))
for count_a, animal in enumerate(animals):
    trials = otrack_class.get_trials(animal)
    # LOAD PROCESSED DATA
    [otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, laser_on] = otrack_class.load_processed_files(animal)
    # LOAD DATA FOR BENCHMARK ANALYSIS
    [st_led_on, sw_led_on, frame_counter_session] = otrack_class.load_benchmark_files(animal)
    # READ OFFLINE PAW EXCURSIONS
    [final_tracks_trials, st_strides_trials, sw_strides_trials] = otrack_class.get_offtrack_paws(loco, animal, session)
    final_tracks_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials,
                                                 'st-sw-st')

    # STIMULATION DURATION OFFSETS AND ONSETS
    trials_reshape = np.reshape(np.arange(1, 11), (5, 2))
    for i in range(len(trials_reshape)):
        for count_t, trial in enumerate(trials_reshape[i, :]):
            #stim phase
            if animal == 'MC18089' or animal == 'MC18090' or animal == 'MC18091': #use laser
                [light_onset_phase_st_trial, light_offset_phase_st_trial, stim_nr_st, stride_nr_st] = \
                    otrack_class.laser_presentation_phase(trial, trials, 'stance', offtracks_st, offtracks_sw, laser_on,
                                                          timestamps_session, final_tracks_phase, 0)
            else: #use LED
                [light_onset_phase_st_trial, light_offset_phase_st_trial, stim_nr_st, stride_nr_st] = \
                    otrack_class.light_presentation_phase(trial, trials, 'stance', offtracks_st, offtracks_sw, st_led_on,
                                                          sw_led_on,
                                                          timestamps_session, final_tracks_phase, 0)
            [fraction_strides_stim_st_on, fraction_strides_stim_st_off] = \
                    otrack_class.plot_laser_presentation_phase_benchmark(light_onset_phase_st_trial,
                                                                         light_offset_phase_st_trial, 'stance',
                                                                         16, stim_nr_st,
                                                                         stride_nr_st, 'Greys',
                                                                         path_save, 'test')
            if animal == 'MC18089' or animal == 'MC18090' or animal == 'MC18091': #use LED
                [light_onset_phase_sw_trial, light_offset_phase_sw_trial, stim_nr_sw, stride_nr_sw] = \
                    otrack_class.light_presentation_phase(trial, trials, 'swing', offtracks_st, offtracks_sw, st_led_on,  sw_led_on,
                                                          timestamps_session, final_tracks_phase, 0)
            else: #use laser
                [light_onset_phase_sw_trial, light_offset_phase_sw_trial, stim_nr_sw, stride_nr_sw] = \
                    otrack_class.light_presentation_phase(trial, trials, 'swing', offtracks_st, offtracks_sw,
                                                          st_led_on,
                                                          sw_led_on,
                                                          timestamps_session, final_tracks_phase, 0)
            [fraction_strides_stim_sw_on, fraction_strides_stim_sw_off] = \
                otrack_class.plot_laser_presentation_phase_benchmark(light_onset_phase_sw_trial,
                                                                     light_offset_phase_sw_trial, 'swing',
                                                                     16, stim_nr_st,
                                                                     stride_nr_st, 'Greys',
                                                                     path_save, 'test')
            plt.close('all')
            frac_strides_st_all[count_a, i, count_t] = fraction_strides_stim_st_on
            frac_strides_sw_all[count_a, i, count_t] = fraction_strides_stim_sw_on

np.save(os.path.join(otrack_class.path, 'processed files', 'frac_strides_st.npy'), frac_strides_st_all, allow_pickle=True)
np.save(os.path.join(otrack_class.path, 'processed files', 'frac_strides_sw.npy'), frac_strides_sw_all, allow_pickle=True)

