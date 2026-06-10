import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.warnings.filterwarnings('ignore')

path = 'J:\\Opto Benchmarks\\HR tests\\50percent\\'
condition = path.split('\\')[-2]
network = path.split('\\')[-3]
animals = ['MC18089', 'MC18090', 'MC18091', 'VIV40922', 'VIV40923', 'VIV40924']
paws = ['FR', 'HR', 'FL', 'HL']
session = 1
save_path = 'J:\\Opto Benchmarks\\Benchmark plots\\HR network-JAWS threshold-other paws-phase stimulation\\'
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)

animal_session_list = loco.animals_within_session()
animal_list = []
for a in range(len(animal_session_list)):
    animal_list.append(animal_session_list[a][0])
session_list = []
for a in range(len(animal_session_list)):
    session_list.append(animal_session_list[a][1])

for paw in paws:
    light_onset_phase_st = []
    light_offset_phase_st = []
    light_onset_phase_sw = []
    light_offset_phase_sw = []
    for count_a, animal in enumerate(animals):
        print('Processing ' + animal)
        trials = otrack_class.get_trials(animal)
        # LOAD PROCESSED DATA
        [otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, laser_on] = otrack_class.load_processed_files(animal)
        #GET RIGHT OFFTRACK_ST FOR EACH PAW
        [offtracks_st, offtracks_sw] = otrack_class.get_offtrack_event_data('FR', loco, animal, np.int64(session_list[count_a]), timestamps_session, 0)

        # LOAD DATA FOR BENCHMARK ANALYSIS
        [st_led_on, sw_led_on, frame_counter_session] = otrack_class.load_benchmark_files(animal)
        # READ OFFLINE PAW EXCURSIONS
        [final_tracks_trials, st_strides_trials, sw_strides_trials] = otrack_class.get_offtrack_paws(loco, animal, session)
        final_tracks_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials,
                                                     'st-sw-st')

        # STIMULATION DURATION OFFSETS AND ONSETS
        trials_reshape = np.reshape(np.arange(1, 11), (5, 2))
        light_onset_phase_st_animal = []
        light_offset_phase_st_animal = []
        light_onset_phase_sw_animal = []
        light_offset_phase_sw_animal = []
        for i in range(len(trials_reshape)):
            stim_duration_st_trialtype = []
            stim_duration_sw_trialtype = []
            light_onset_phase_st_trialtype = []
            light_offset_phase_st_trialtype = []
            light_onset_phase_sw_trialtype = []
            light_offset_phase_sw_trialtype = []
            for count_t, trial in enumerate(trials_reshape[i, :]):
                #stim phase
                if animal[0] == 'M':
                    if paw == 'FR' or paw == 'HL':
                        [light_onset_phase_st_trial, light_offset_phase_st_trial, stim_nr_st, stride_nr_st] = \
                            otrack_class.laser_presentation_phase_all(trial, trials, 'stance', offtracks_st,
                                                                      offtracks_sw,
                                                                      laser_on,
                                                                      timestamps_session, final_tracks_phase, paw)
                    if paw == 'FL' or paw == 'HR':
                        [light_onset_phase_st_trial, light_offset_phase_st_trial, stim_nr_st, stride_nr_st] = \
                            otrack_class.laser_presentation_phase_contralateral_all(trial, trials, 'stance',
                                                                                    offtracks_st,
                                                                                    offtracks_sw, laser_on,
                                                                                    timestamps_session,
                                                                                    final_tracks_phase,
                                                                                    paw)
                if animal[0] == 'V':
                    if paw == 'FR' or paw == 'HL':
                        [light_onset_phase_st_trial, light_offset_phase_st_trial, stim_nr_st, stride_nr_st] = \
                            otrack_class.light_presentation_phase_all(trial, trials, 'stance', offtracks_st,
                                                                      offtracks_sw,
                                                                      st_led_on, sw_led_on,
                                                                      timestamps_session, final_tracks_phase, paw)
                    if paw == 'FL' or paw == 'HR':
                        [light_onset_phase_st_trial, light_offset_phase_st_trial, stim_nr_st, stride_nr_st] = \
                            otrack_class.light_presentation_phase_contralateral_all(trial, trials, 'stance',
                                                                                    offtracks_st,
                                                                                    offtracks_sw, st_led_on, sw_led_on,
                                                                                    timestamps_session,
                                                                                    final_tracks_phase,
                                                                                    paw)
                if paw == 'FR' or paw == 'HL':
                    [light_onset_phase_sw_trial, light_offset_phase_sw_trial, stim_nr_sw, stride_nr_sw] = \
                        otrack_class.light_presentation_phase_all(trial, trials, 'swing', offtracks_st, offtracks_sw,
                                                                  st_led_on, sw_led_on,
                                                                  timestamps_session, final_tracks_phase, paw)
                if paw == 'FL' or paw == 'HR':
                    [light_onset_phase_sw_trial, light_offset_phase_sw_trial, stim_nr_sw, stride_nr_sw] = \
                        otrack_class.light_presentation_phase_contralateral_all(trial, trials, 'swing', offtracks_st,
                                                                                offtracks_sw,
                                                                                st_led_on, sw_led_on,
                                                                                timestamps_session, final_tracks_phase,
                                                                                paw)
                light_onset_phase_st_trialtype.extend(light_onset_phase_st_trial)
                light_offset_phase_st_trialtype.extend(light_offset_phase_st_trial)
                light_onset_phase_sw_trialtype.extend(light_onset_phase_sw_trial)
                light_offset_phase_sw_trialtype.extend(light_offset_phase_sw_trial)
            light_onset_phase_st_animal.append(light_onset_phase_st_trialtype)
            light_offset_phase_st_animal.append(light_offset_phase_st_trialtype)
            light_onset_phase_sw_animal.append(light_onset_phase_sw_trialtype)
            light_offset_phase_sw_animal.append(light_offset_phase_sw_trialtype)
        light_onset_phase_st.append(light_onset_phase_st_animal)
        light_offset_phase_st.append(light_offset_phase_st_animal)
        light_onset_phase_sw.append(light_onset_phase_sw_animal)
        light_offset_phase_sw.append(light_offset_phase_sw_animal)

    # STIMULATION ONSET AND OFFSET IN %STRIDE - HISTOGRAM
    speeds = ['0,175', '0,275', '0,375', 'split_ipsi_fast', 'split_contra_fast']
    for i in range(len(trials_reshape)):
        light_onset_phase_st_plot_animals = []
        light_offset_phase_st_plot_animals = []
        light_onset_phase_sw_plot_animals = []
        light_offset_phase_sw_plot_animals = []
        for count_a in range(len(animals)):
            light_onset_phase_st_plot_animals.extend(light_onset_phase_st[count_a][i])
            light_offset_phase_st_plot_animals.extend(light_offset_phase_st[count_a][i])
            light_onset_phase_sw_plot_animals.extend(light_onset_phase_sw[count_a][i])
            light_offset_phase_sw_plot_animals.extend(light_offset_phase_sw[count_a][i])
        otrack_class.plot_laser_presentation_phase_hist(light_onset_phase_st_plot_animals, light_offset_phase_st_plot_animals,
                                              20, save_path, 'stance_time_hist_'+condition+'_'+network+'_'+speeds[i]+'_'+paw, 1)
        otrack_class.plot_laser_presentation_phase_hist(light_onset_phase_sw_plot_animals, light_offset_phase_sw_plot_animals,
                                              20, save_path, 'swing_time_hist_'+condition+'_'+network+'_'+speeds[i]+'_'+paw, 1)
    plt.close('all')

