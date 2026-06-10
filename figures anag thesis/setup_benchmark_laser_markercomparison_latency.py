import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from bioinfokit.analys import stat
np.warnings.filterwarnings('ignore')

animals = ['VIV40922', 'VIV40923', 'VIV40924']
networks = ['CM tests', 'HR tests', 'Tailbase tests']
networks_label = ['Center of mass normalization', 'Hind right paw normalization', 'Base of the tail normalization']
conditions_plot = [['50percent', '50percent'], ['50percent', '50percent'], ['50percent', '50percent']]
speeds = ['0,275', 'right_fast', 'left_fast']
speeds_label = ['0.275 m/s', 'split right fast', 'split left fast']
trials_reshape = np.reshape(np.arange(1, 11), (5, 2)) #0.175, 0.275, 0.375, right fast, left fast
trials = trials_reshape[1, :].flatten()
colors_networks = ['black', 'teal', 'orange']
summary_path = 'J:\\Opto Benchmarks\\Benchmark plots\\For each speed comparison of marker for 50% threshold\\'

latency_on_st_array = np.zeros((6, 3, len(networks)))
latency_off_st_array = np.zeros((6, 3, len(networks)))
latency_on_sw_array = np.zeros((6, 3, len(networks)))
latency_off_sw_array = np.zeros((6, 3, len(networks)))
for count_n, n in enumerate(networks):
    c = conditions_plot[count_n]
    path_st = os.path.join('J:\\Opto Benchmarks\\', n, c[0])
    for count_s, s in enumerate(np.array([1, 3, 4])):
        trials = trials_reshape[s, :].flatten()
        import online_tracking_class
        otrack_class = online_tracking_class.otrack_class(path_st)
        import locomotion_class
        loco = locomotion_class.loco_class(path_st)
        latency_st = pd.read_csv(os.path.join(path_st, 'processed files', 'latency_data_st.csv'))
        latency_st_trials = latency_st[latency_st['trial'].isin(trials)]
        for count_a, animal in enumerate(animals):
            for count_t, t in enumerate(trials):
                latency_on_st_array[count_a+(3*count_t), count_s, count_n] = latency_st_trials.loc[(latency_st_trials['animal'] == animal) & (latency_st_trials['trial'] == t)].median()[1]
                latency_off_st_array[count_a+(3*count_t), count_s, count_n] = latency_st_trials.loc[(latency_st_trials['animal'] == animal) & (latency_st_trials['trial'] == t)].median()[2]
        path_sw = os.path.join('J:\\Opto Benchmarks', n, c[1])
        import online_tracking_class
        otrack_class = online_tracking_class.otrack_class(path_sw)
        import locomotion_class
        loco = locomotion_class.loco_class(path_sw)
        latency_sw = pd.read_csv(os.path.join(path_sw, 'processed files', 'latency_data_sw.csv'))
        latency_sw_trials = latency_sw[latency_sw['trial'].isin(trials)]
        for count_a, animal in enumerate(animals):
            for count_t, t in enumerate(trials):
                latency_on_sw_array[count_a+(3*count_t), count_s, count_n] = latency_sw_trials.loc[(latency_sw_trials['animal'] == animal) & (latency_sw_trials['trial'] == t)].median()[1]
                latency_off_sw_array[count_a+(3*count_t), count_s, count_n] = latency_sw_trials.loc[(latency_sw_trials['animal'] == animal) & (latency_sw_trials['trial'] == t)].median()[2]

#LARGER LATENCIES ARE DUE TO THE MATCH BETWEEN OTRACKS BEING ON AND THE LED TURING ON - CODE FINDS THE CLOSEST, DOESN'T MEAN IT IS THE CORRECT
#PROBABLY CAN HAPPEN THAT FOR VERY SHORT TRUE PULSES (2 FRAMES??) LED DOESN'T TURN ON
#FOR THESIS WILL CROP PLOT BUT THIS CAN BE LOOKED INTO

fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
for count_n, n in enumerate(networks_label):
    for a in range(6):
        ax.scatter(np.arange(0, 30, 10) + (np.ones(3) * count_n) + np.random.rand(3),
                   latency_on_st_array[a, :, count_n], s=10, color=colors_networks[count_n])
ax.set_xticks(np.arange(0, 30, 10)+1)
ax.set_xticklabels(speeds_label, fontsize=14)
ax.set_ylabel('Latency for stance\nLED on (s)', fontsize=14)
ax.set_ylim([0, 0.035])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(summary_path, 'latency_on_st'), dpi=128)
plt.savefig(os.path.join(summary_path, 'latency_on_st.svg'), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
for count_n, n in enumerate(networks_label):
    for a in range(6):
        ax.scatter(np.arange(0, 30, 10) + (np.ones(3) * count_n) + np.random.rand(3),
                   latency_on_sw_array[a, :, count_n], s=10, color=colors_networks[count_n])
ax.set_xticks(np.arange(0, 30, 10)+1)
ax.set_xticklabels(speeds_label, fontsize=14)
ax.set_ylabel('Latency for swing\nLED on (s)', fontsize=14)
ax.set_ylim([0, 0.035])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(summary_path, 'latency_on_sw'), dpi=128)
plt.savefig(os.path.join(summary_path, 'latency_on_sw.svg'), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
for count_n, n in enumerate(networks_label):
    for a in range(6):
        ax.scatter(np.arange(0, 30, 10) + (np.ones(3) * count_n) + np.random.rand(3),
                   latency_off_st_array[a, :, count_n], s=10, color=colors_networks[count_n])
ax.set_xticks(np.arange(0, 30, 10)+1)
ax.set_xticklabels(speeds_label, fontsize=14)
ax.set_ylabel('Latency for stance\nLED off (s)', fontsize=14)
ax.set_ylim([0, 0.035])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(summary_path, 'latency_off_st'), dpi=128)
plt.savefig(os.path.join(summary_path, 'latency_off_st.svg'), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
for count_n, n in enumerate(networks_label):
    for a in range(6):
        ax.scatter(np.arange(0, 30, 10) + (np.ones(3) * count_n) + np.random.rand(3),
                   latency_off_sw_array[a, :, count_n], s=10, color=colors_networks[count_n])
ax.set_xticks(np.arange(0, 30, 10)+1)
ax.set_xticklabels(speeds_label, fontsize=14)
ax.set_ylabel('Latency for swing\nLED off (s)', fontsize=14)
ax.set_ylim([0, 0.035])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(summary_path, 'latency_off_sw'), dpi=128)
plt.savefig(os.path.join(summary_path, 'latency_off_sw.svg'), dpi=128)
                
###### STATS #######
#Mann-Whitney on latency for the three computed speeds - network comparison stance on
latency_on_st_cm_hr = scipy.stats.mannwhitneyu(latency_on_st_array[:, 0, :].flatten(), latency_on_st_array[:, 1, :].flatten(), method='exact')
print(latency_on_st_cm_hr)
latency_on_st_cm_tb = scipy.stats.mannwhitneyu(latency_on_st_array[:, 0, :].flatten(), latency_on_st_array[:, 2, :].flatten(), method='exact')
print(latency_on_st_cm_tb)
latency_on_st_hr_tb = scipy.stats.mannwhitneyu(latency_on_st_array[:, 1, :].flatten(), latency_on_st_array[:, 2, :].flatten(), method='exact')
print(latency_on_st_hr_tb)

#Mann-Whitney on latency for the three computed speeds - network comparison swing on
latency_on_sw_cm_hr = scipy.stats.mannwhitneyu(latency_on_sw_array[:, 0, :].flatten(), latency_on_sw_array[:, 1, :].flatten(), method='exact')
print(latency_on_sw_cm_hr)
latency_on_sw_cm_tb = scipy.stats.mannwhitneyu(latency_on_sw_array[:, 0, :].flatten(), latency_on_sw_array[:, 2, :].flatten(), method='exact')
print(latency_on_sw_cm_tb)
latency_on_sw_hr_tb = scipy.stats.mannwhitneyu(latency_on_sw_array[:, 1, :].flatten(), latency_on_sw_array[:, 2, :].flatten(), method='exact')
print(latency_on_sw_hr_tb)

#Mann-Whitney on latency for the three computed speeds - network comparison stance off
latency_off_st_cm_hr = scipy.stats.mannwhitneyu(latency_off_st_array[:, 0, :].flatten(), latency_off_st_array[:, 1, :].flatten(), method='exact')
print(latency_off_st_cm_hr)
latency_off_st_cm_tb = scipy.stats.mannwhitneyu(latency_off_st_array[:, 0, :].flatten(), latency_off_st_array[:, 2, :].flatten(), method='exact')
print(latency_off_st_cm_tb)
latency_off_st_hr_tb = scipy.stats.mannwhitneyu(latency_off_st_array[:, 1, :].flatten(), latency_off_st_array[:, 2, :].flatten(), method='exact')
print(latency_off_st_hr_tb)

#Mann-Whitney on latency for the three computed speeds - network comparison swing off
latency_off_sw_cm_hr = scipy.stats.mannwhitneyu(latency_off_sw_array[:, 0, :].flatten(), latency_off_sw_array[:, 1, :].flatten(), method='exact')
print(latency_off_sw_cm_hr)
latency_off_sw_cm_tb = scipy.stats.mannwhitneyu(latency_off_sw_array[:, 0, :].flatten(), latency_off_sw_array[:, 2, :].flatten(), method='exact')
print(latency_off_sw_cm_tb)
latency_off_sw_hr_tb = scipy.stats.mannwhitneyu(latency_off_sw_array[:, 1, :].flatten(), latency_off_sw_array[:, 2, :].flatten(), method='exact')
print(latency_off_sw_hr_tb)
