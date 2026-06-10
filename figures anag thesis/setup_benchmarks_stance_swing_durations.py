# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
np.warnings.filterwarnings('ignore')


conditions = ['25percent', '50percent', '75percent']
networks = ['Tailbase tests', 'CM tests', 'HR tests']
animals = ['MC18089', 'MC18090', 'MC18091', 'VIV40922', 'VIV40923', 'VIV40924']
session = 1

st_duration_trials = np.zeros(10)
sw_duration_trials = np.zeros(10)
for count_n, network in enumerate(networks):
    for count_c, condition in enumerate(conditions):
        path = os.path.join('C:\\Users\\Ana\\Desktop\\Data OPTO', network, condition + '\\')
        import online_tracking_class
        otrack_class = online_tracking_class.otrack_class(path)
        import locomotion_class
        loco = locomotion_class.loco_class(path)
        for count_a, animal in enumerate(animals):
            trials = otrack_class.get_trials(animal)
            # LOAD PROCESSED DATA
            [otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, laser_on] = otrack_class.load_processed_files(animal)
            # LOAD DATA FOR BENCHMARK ANALYSIS
            [st_led_on, sw_led_on, frame_counter_session] = otrack_class.load_benchmark_files(animal)
            # READ OFFLINE PAW EXCURSIONS
            [final_tracks_trials, st_strides_trials, sw_strides_trials] = otrack_class.get_offtrack_paws(loco, animal, session)
            
            st_duration_loop = np.zeros(len(trials))
            sw_duration_loop = np.zeros(len(trials))
            for count_t, trial in enumerate(trials):
                st_duration_loop[count_t] = np.nanmean(st_strides_trials[count_t][0][:, 1, 0]-st_strides_trials[count_t][0][:, 0, 0])/1000
                sw_duration_loop[count_t] = np.nanmean(st_strides_trials[count_t][0][:, 1, 0]-sw_strides_trials[count_t][0][:, 0, 0])/1000
            st_duration_trials = np.vstack([st_duration_trials, st_duration_loop])
            sw_duration_trials = np.vstack([sw_duration_trials, sw_duration_loop])

st_duration_trials_corr = st_duration_trials[1:, :]
sw_duration_trials_corr = sw_duration_trials[1:, :]

trials_reshape = np.reshape(np.arange(1, 11), (5, 2))
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
for i in range(len(trials_reshape)):
    data_plot = st_duration_trials_corr[:, trials_reshape[i, :]-1].flatten()
    ax.scatter(np.repeat(trials_reshape[i, 0], len(data_plot))+np.random.rand(len(data_plot)), data_plot, s=10, color='black')
ax.set_xticks(trials_reshape[:, 0]+0.5)
ax.set_xticklabels(['0.175', '0.275', '0.375', '0.175 left\n0.375 right', '0.375 left\n0.175 right'], fontsize=14)
ax.set_title('Stance duration', fontsize=16)
ax.set_ylabel('Stance duration (s)', fontsize=14)
ax.set_xlabel('Speed (m/s)', fontsize=14)
ax.set_ylim([0.05, 0.32])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('C:\\Users\\Ana\\Desktop\\Data OPTO\\Benchmark plots\\stance_duration.png', dpi=128)

fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
for i in range(len(trials_reshape)):
    data_plot = sw_duration_trials_corr[:, trials_reshape[i, :]-1].flatten()
    ax.scatter(np.repeat(trials_reshape[i, 0], len(data_plot))+np.random.rand(len(data_plot)), data_plot, s=10, color='black')
ax.set_xticks(trials_reshape[:, 0]+0.5)
ax.set_xticklabels(['0.175', '0.275', '0.375', '0.175 left\n0.375 right', '0.375 left\n0.175 right'], fontsize=14)
ax.set_title('Swing duration', fontsize=16)
ax.set_ylabel('Swing duration (s)', fontsize=14)
ax.set_xlabel('Speed (m/s)', fontsize=14)
ax.set_ylim([0.05, 0.16])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig('C:\\Users\\Ana\\Desktop\\Data OPTO\\Benchmark plots\\swing_duration.png', dpi=128)


