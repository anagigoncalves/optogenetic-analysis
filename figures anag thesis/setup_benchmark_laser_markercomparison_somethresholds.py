import os
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import scipy.stats
from bioinfokit.analys import stat
np.warnings.filterwarnings('ignore')

networks = ['CM tests', 'HR tests', 'Tailbase tests']
networks_label = ['Center of mass normalization', 'Hind right paw normalization', 'Base of the tail normalization']
conditions_plot = [['50percent', '50percent'], ['50percent', '50percent'], ['50percent', '50percent']]
speeds = ['0,275', 'right_fast', 'left_fast']
speeds_label = ['0.275 m/s', 'split right fast', 'split left fast']
trials_reshape = np.reshape(np.arange(1, 11), (5, 2)) #0.175, 0.275, 0.375, right fast, left fast
measure_name = ['accuracy', 'f1_score', 'false_negatives', 'false_positives']
measure_name_label = ['Accuracy', 'F1 score', 'False negatives', 'False positives']
colors_networks = ['black', 'teal', 'orange']
summary_path = 'J:\\Opto Benchmarks\\Benchmark plots\\For each speed comparison of marker for 50% threshold\\'

stim_duration_st_net = []
light_onset_phase_st_net = []
light_offset_phase_st_net = []
stim_duration_sw_net = []
light_onset_phase_sw_net = []
light_offset_phase_sw_net = []
stim_nr_st_net = []
stride_nr_st_net = []
stim_nr_sw_net = []
stride_nr_sw_net = []
accuracy_measures_st = np.zeros((12, 4, 3, len(networks)))
accuracy_measures_sw = np.zeros((12, 4, 3, len(networks)))
frac_strides_st = np.zeros((12, len(networks), len(speeds)))
frac_strides_sw = np.zeros((12, len(networks), len(speeds)))
for count_n, n in enumerate(networks):
    c = conditions_plot[count_n]
    stim_duration_st_cond = []
    light_onset_phase_st_cond = []
    light_offset_phase_st_cond = []
    stim_duration_sw_cond = []
    light_onset_phase_sw_cond = []
    light_offset_phase_sw_cond = []
    stim_nr_st_cond = []
    stride_nr_st_cond = []
    stim_nr_sw_cond = []
    stride_nr_sw_cond = []
    for count_s, s in enumerate(np.array([1, 3, 4])):
        trials = trials_reshape[s, :].flatten()
        path_st = os.path.join('J:\\Opto Benchmarks', n, c[0])
        import online_tracking_class
        otrack_class = online_tracking_class.otrack_class(path_st)
        import locomotion_class
        loco = locomotion_class.loco_class(path_st)
        frac_strides_st[:, count_n, count_s] = np.load(os.path.join(path_st, 'processed files', 'frac_strides_st.npy'),
                                                         allow_pickle=True)[:, s, :].flatten()
        benchmark_accuracy_st = pd.read_csv(os.path.join(path_st, 'processed files', 'benchmark_accuracy.csv'))
        stim_duration_st_list = np.load(os.path.join(path_st, 'processed files', 'stim_duration_st.npy'), allow_pickle=True)
        light_onset_phase_st_list = np.load(os.path.join(path_st, 'processed files', 'light_onset_phase_st.npy'), allow_pickle=True)
        light_offset_phase_st_list = np.load(os.path.join(path_st, 'processed files', 'light_offset_phase_st.npy'), allow_pickle=True)
        stim_nr_st_list = np.load(os.path.join(path_st, 'processed files', 'stim_nr_st.npy'), allow_pickle=True)
        stride_nr_st_list = np.load(os.path.join(path_st, 'processed files', 'stride_nr_st.npy'), allow_pickle=True)
        stim_nr_st_cond.append(list(itertools.chain(*stim_nr_st_list[:, s])))
        stride_nr_st_cond.append(list(itertools.chain(*stride_nr_st_list[:, s])))
        stim_duration_st_cond.append(list(itertools.chain(*stim_duration_st_list[:, s])))
        light_onset_phase_st_cond.append(list(itertools.chain(*light_onset_phase_st_list[:, s])))
        light_offset_phase_st_cond.append(list(itertools.chain(*light_offset_phase_st_list[:, s])))
        accuracy_measures_st[:, :, count_s, count_n] = np.array(benchmark_accuracy_st[benchmark_accuracy_st['trial'].isin(trials)].iloc[:,3::2])
        path_sw = os.path.join('J:\\Opto Benchmarks', n, c[1])
        import online_tracking_class
        otrack_class = online_tracking_class.otrack_class(path_sw)
        import locomotion_class
        loco = locomotion_class.loco_class(path_sw)
        frac_strides_sw[:, count_n, count_s] = np.load(os.path.join(path_sw, 'processed files', 'frac_strides_sw.npy'),
                                                         allow_pickle=True)[:, s, :].flatten()
        benchmark_accuracy_sw = pd.read_csv(os.path.join(path_sw, 'processed files', 'benchmark_accuracy.csv'))
        stim_duration_sw_list = np.load(os.path.join(path_sw, 'processed files', 'stim_duration_sw.npy'), allow_pickle=True)
        light_onset_phase_sw_list = np.load(os.path.join(path_sw, 'processed files', 'light_onset_phase_sw.npy'), allow_pickle=True)
        light_offset_phase_sw_list = np.load(os.path.join(path_sw, 'processed files', 'light_offset_phase_sw.npy'), allow_pickle=True)
        stim_nr_sw_list = np.load(os.path.join(path_sw, 'processed files', 'stim_nr_st.npy'), allow_pickle=True)
        stride_nr_sw_list = np.load(os.path.join(path_sw, 'processed files', 'stride_nr_st.npy'), allow_pickle=True)
        stim_nr_sw_cond.append(list(itertools.chain(*stim_nr_sw_list[:, s])))
        stride_nr_sw_cond.append(list(itertools.chain(*stride_nr_sw_list[:, s])))
        stim_duration_sw_cond.append(list(itertools.chain(*stim_duration_sw_list[:, s])))
        light_onset_phase_sw_cond.append(list(itertools.chain(*light_onset_phase_sw_list[:, s])))
        light_offset_phase_sw_cond.append(list(itertools.chain(*light_offset_phase_sw_list[:, s])))
        accuracy_measures_sw[:, :, count_s, count_n] = np.array(benchmark_accuracy_sw[benchmark_accuracy_sw['trial'].isin(trials)].iloc[:,4::2])
    stim_duration_st_net.append(stim_duration_st_cond)
    light_onset_phase_st_net.append(light_onset_phase_st_cond)
    light_offset_phase_st_net.append(light_offset_phase_st_cond)
    stim_duration_sw_net.append(stim_duration_sw_cond)
    light_onset_phase_sw_net.append(light_onset_phase_sw_cond)
    light_offset_phase_sw_net.append(light_offset_phase_sw_cond)
    stim_nr_st_net.append(stim_nr_st_cond)
    stride_nr_st_net.append(stride_nr_st_cond)
    stim_nr_sw_net.append(stim_nr_sw_cond)
    stride_nr_sw_net.append(stride_nr_sw_cond)

# DURATION - VIOLIN PLOT
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
for n in range(len(networks)):
    violin_parts = ax.violinplot(stim_duration_st_net[n], positions=np.arange(0, 9, 3) + (0.5 * n),
        showextrema=False, showmeans=True)
    for pc in violin_parts['bodies']:
        pc.set_color(colors_networks[n])
    violin_parts['cmeans'].set_color(colors_networks[n])
ax.set_xticks(np.arange(0, 9, 3)+0.5)
ax.set_xticklabels(speeds_label, fontsize=14)
#ax.set_title('Stance stim duration', fontsize=16)
ax.set_ylabel('LED-on duration (s)', fontsize=14)
ax.set_ylim([0, 0.3])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(summary_path, 'stim_duration_st'), dpi=128)
plt.savefig(os.path.join(summary_path, 'stim_duration_st.svg'), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
for n in range(len(networks)):
    violin_parts = ax.violinplot(stim_duration_sw_net[n], positions=np.arange(0, 9, 3) + (0.5 * n),
        showextrema=False, showmeans=True)
    for pc in violin_parts['bodies']:
        pc.set_color(colors_networks[n])
    violin_parts['cmeans'].set_color(colors_networks[n])
ax.set_xticks(np.arange(0, 9, 3)+0.5)
ax.set_xticklabels(speeds_label, fontsize=14)
#ax.set_title('Swing stim duration', fontsize=16)
ax.set_ylabel('LED-on duration (s)', fontsize=14)
ax.set_ylim([0, 0.65])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(summary_path, 'stim_duration_sw'), dpi=128)
plt.savefig(os.path.join(summary_path, 'stim_duration_sw.svg'), dpi=128)

# DURATION - HISTOGRAM PLOT
for count_c, c in enumerate(speeds):
    fig, ax = plt.subplots(tight_layout=True, figsize=(7, 3))
    for n in range(len(networks)):
        ax.hist(stim_duration_st_net[n][count_c], bins=100, histtype='step', color=colors_networks[n], linewidth=4)
    ax.set_xlabel('LED-on duration (s)', fontsize=14)
    ax.set_ylabel('Counts', fontsize=14)
    ax.set_xlim([0, 0.5])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(summary_path, 'stim_duration_hist_st_' + networks[n] + '_' + c), dpi=128)
    plt.savefig(os.path.join(summary_path, 'stim_duration_hist_st_' + networks[n] + '_' + c + '.svg'), dpi=128)
    fig, ax = plt.subplots(tight_layout=True, figsize=(7, 3))
    for n in range(len(networks)):
        ax.hist(stim_duration_sw_net[n][count_c], bins=100, histtype='step', color=colors_networks[n], linewidth=4)
    ax.set_xlabel('LED-on duration (s)', fontsize=14)
    ax.set_ylabel('Counts', fontsize=14)
    ax.set_xlim([0, 0.5])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(summary_path, 'stim_duration_hist_sw_' + networks[n] + '_' + c), dpi=128)
    plt.savefig(os.path.join(summary_path, 'stim_duration_hist_sw_' + networks[n] + '_' + c + '.svg'), dpi=128)
plt.close('all')

# STIMULATION ONSETS AND OFFSETS
for count_n, n in enumerate(networks):
    for count_s, s in enumerate(speeds):
        [fraction_strides_stim_st_on, fraction_strides_stim_st_off] = \
            otrack_class.plot_laser_presentation_phase_benchmark(light_onset_phase_st_net[count_n][count_s],
                                                                 light_offset_phase_st_net[count_n][count_s], 'stance',
                                                                 16, np.sum(stim_nr_st_net[count_n][count_s]),
                                                                 np.sum(stride_nr_st_net[count_n][count_s]), 'Greys',
                                                                 summary_path,
                                                                 '\\light_stance_' + networks[count_n] + '_' +
                                                                 speeds[count_s])
        [fraction_strides_stim_sw_on, fraction_strides_stim_sw_off] = \
            otrack_class.plot_laser_presentation_phase_benchmark(light_onset_phase_sw_net[count_n][count_s],
                                                                 light_offset_phase_sw_net[count_n][count_s], 'swing',
                                                                 16, np.sum(stim_nr_sw_net[count_n][count_s]),
                                                                 np.sum(stride_nr_sw_net[count_n][count_s]), 'Greys',
                                                                 summary_path,
                                                                 '\\light_swing_' + networks[count_n] + '_' +
                                                                 speeds[count_s])
        otrack_class.plot_laser_presentation_phase_hist(light_onset_phase_st_net[count_n][count_s], light_offset_phase_st_net[count_n][count_s],
                                                        20,  summary_path,
                                                                 '\\light_stance_hist_' + networks[count_n] + '_' +
                                                                 speeds[count_s], 1)
        otrack_class.plot_laser_presentation_phase_hist(light_onset_phase_sw_net[count_n][count_s], light_offset_phase_sw_net[count_n][count_s],
                                                        20,  summary_path,
                                                                 '\\light_swing_hist_' + networks[count_n] + '_' +
                                                                 speeds[count_s], 1)
        plt.close('all')

# FRACTION OF STIMULATED STRIDES
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
for count_n, n in enumerate(networks_label):
    for a in range(12):
        if a == 0:
            ax.scatter(np.arange(0, 30, 10) + (np.ones(3) * count_n) + np.random.rand(3),
                       frac_strides_st[a, count_n, :],
                       s=10, color=colors_networks[count_n], label=n)
        else:
            ax.scatter(np.arange(0, 30, 10) + (np.ones(3) * count_n) + np.random.rand(3),
                       frac_strides_st[a, count_n, :],
                       s=10, color=colors_networks[count_n], label='_nolegend_')
    ax.scatter(np.arange(0, 30, 10) + 1.5*count_n, np.nanmean(frac_strides_st[:, count_n, :], axis=0), s=200, marker='_', color=colors_networks[count_n])
# ax.legend(networks_label, frameon=False, fontsize=14)
ax.set_xticks(np.arange(0, 30, 10) + 2)
ax.set_xticklabels(speeds_label, fontsize=14)
ax.set_ylabel('Fraction of LED-on\nstrides', fontsize=14)
ax.set_ylim([0, 1.15])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(summary_path, 'strides_stimulated_st_' + conditions_plot[0][0]), dpi=128)
plt.savefig(os.path.join(summary_path, 'strides_stimulated_st_' + conditions_plot[0][0] + '.svg'), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
for count_n, n in enumerate(networks_label):
    for a in range(12):
        if a == 0:
            ax.scatter(np.arange(0, 30, 10) + (np.ones(3) * count_n) + np.random.rand(3),
                       frac_strides_sw[a, count_n, :],
                       s=10, color=colors_networks[count_n], label=n)
        else:
            ax.scatter(np.arange(0, 30, 10) + (np.ones(3) * count_n) + np.random.rand(3),
                       frac_strides_sw[a, count_n, :],
                       s=10, color=colors_networks[count_n], label='_nolegend_')
    ax.scatter(np.arange(0, 30, 10) + 1.5 * count_n, np.nanmean(frac_strides_sw[:, count_n, :], axis=0), s=200,
               marker='_', color=colors_networks[count_n])
# ax.legend(networks_label, frameon=False, fontsize=14)
ax.set_xticks(np.arange(0, 30, 10) + 2)
ax.set_xticklabels(speeds_label, fontsize=14)
ax.set_ylabel('Fraction of LED-on\nstrides', fontsize=14)
ax.set_ylim([0, 1.15])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(summary_path, 'strides_stimulated_sw_' + conditions_plot[0][0]), dpi=128)
plt.savefig(os.path.join(summary_path, 'strides_stimulated_sw_' + conditions_plot[0][0] + '.svg'), dpi=128)

# ACCURACY
ylabel_names = ['fraction correct hits', '% F1 score', 'fraction false negatives', 'fraction false positives']
for i in range(len(measure_name)):
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
    for n in range(len(networks)):
        for a in range(12):
            if a == 0:
                ax.scatter(np.arange(0, 30, 10)+(np.ones(3)*n)+np.random.rand(3), accuracy_measures_st[a, i, :, n],
                           s=10, color=colors_networks[n], linewidth=2, label=networks[n])
            else:
                ax.scatter(np.arange(0, 30, 10)+(np.ones(3)*n)+np.random.rand(3), accuracy_measures_st[a, i, :, n],
                           s=10, color=colors_networks[n], linewidth=2, label='_nolegend_')
        ax.scatter(np.arange(0, 30, 10) + 1.5 * n, np.nanmean(accuracy_measures_st[:, i, :, n], axis=0), s=200,
                   marker='_', color=colors_networks[n])
    ax.set_xticks(np.arange(0, 30, 10)+2)
    ax.set_xticklabels(speeds_label, fontsize=14)
    ax.set_ylim([0, 1])
    #ax.set_title('Stance ' + measure_name[i].replace('_', ' '), fontsize=16)
    ax.set_ylabel(measure_name_label[i], fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(summary_path, measure_name[i] + '_st'), dpi=128)
    plt.savefig(os.path.join(summary_path, measure_name[i] + '_st.svg'), dpi=128)

    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
    for n in range(len(networks)):
        for a in range(12):
            if a == 0:
                ax.scatter(np.arange(0, 30, 10)+(np.ones(3)*n)+np.random.rand(3), accuracy_measures_sw[a, i, :, n],
                           s=10, color=colors_networks[n], linewidth=2, label=networks[n])
            else:
                ax.scatter(np.arange(0, 30, 10)+(np.ones(3)*n)+np.random.rand(3), accuracy_measures_sw[a, i, :, n],
                           s=10, color=colors_networks[n], linewidth=2, label='_nolegend_')
        ax.scatter(np.arange(0, 30, 10) + 1.5 * n, np.nanmean(accuracy_measures_sw[:, i, :, n], axis=0), s=200,
                   marker='_', color=colors_networks[n])
    ax.set_xticks(np.arange(0, 30, 10)+2)
    ax.set_xticklabels(speeds_label, fontsize=14)
    #ax.set_title('Swing ' + measure_name[i].replace('_', ' '), fontsize=16)
    ax.set_ylabel(measure_name_label[i], fontsize=14)
    ax.set_ylim([0, 1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(summary_path, measure_name[i] + '_sw'), dpi=128)
    plt.savefig(os.path.join(summary_path, measure_name[i] + '_sw.svg'), dpi=128)
    plt.close('all')

##### COEFFICIENT OF VARIATION ON TIMING ########
def cv(data):
    cv = np.nanstd(data)/np.abs(np.nanmean(data))
    return cv
# CV across speeds and networks - stance
onset_cm = [cv(light_onset_phase_st_net[0][0]), cv(light_onset_phase_st_net[0][1]), cv(light_onset_phase_st_net[0][2])]
offset_cm = [cv(light_offset_phase_st_net[0][0]), cv(light_offset_phase_st_net[0][1]), cv(light_offset_phase_st_net[0][2])]
onset_hr = [cv(light_onset_phase_st_net[1][0]), cv(light_onset_phase_st_net[1][1]), cv(light_onset_phase_st_net[1][2])]
offset_hr = [cv(light_offset_phase_st_net[1][0]), cv(light_offset_phase_st_net[1][1]), cv(light_offset_phase_st_net[1][2])]
onset_tb = [cv(light_onset_phase_st_net[2][0]), cv(light_onset_phase_st_net[2][1]), cv(light_onset_phase_st_net[2][2])]
offset_tb = [cv(light_offset_phase_st_net[2][0]), cv(light_offset_phase_st_net[2][1]), cv(light_offset_phase_st_net[2][2])]
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
ax.scatter(np.arange(3), onset_cm, s=120, c=colors_networks[0])
ax.scatter(np.arange(3), offset_cm, s=120, facecolors='none', edgecolors=colors_networks[0])
ax.scatter(np.arange(3), onset_hr, s=120, c=colors_networks[1])
ax.scatter(np.arange(3), offset_hr, s=120, facecolors='none', edgecolors=colors_networks[1])
ax.scatter(np.arange(3), onset_tb, s=120, c=colors_networks[2])
ax.scatter(np.arange(3), offset_tb, s=120, facecolors='none', edgecolors=colors_networks[2])
ax.set_xticks(np.arange(3))
ax.set_xticklabels(speeds_label, fontsize=16, rotation=15)
ax.set_ylabel('CV (abs. value) on\nstance onset condition', fontsize=16)
ax.set_xlabel('Speed', fontsize=16)
ax.set_ylim([0, 1.7])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(summary_path, 'cv_timing_st'), dpi=128)
plt.savefig(os.path.join(summary_path, 'cv_timing_st.svg'), dpi=128)
# CV across speeds and networks - swing
onset_cm = [cv(light_onset_phase_sw_net[0][0]), cv(light_onset_phase_sw_net[0][1]), cv(light_onset_phase_sw_net[0][2])]
offset_cm = [cv(light_offset_phase_sw_net[0][0]), cv(light_offset_phase_sw_net[0][1]), cv(light_offset_phase_sw_net[0][2])]
onset_hr = [cv(light_onset_phase_sw_net[1][0]), cv(light_onset_phase_sw_net[1][1]), cv(light_onset_phase_sw_net[1][2])]
offset_hr = [cv(light_offset_phase_sw_net[1][0]), cv(light_offset_phase_sw_net[1][1]), cv(light_offset_phase_sw_net[1][2])]
onset_tb = [cv(light_onset_phase_sw_net[2][0]), cv(light_onset_phase_sw_net[2][1]), cv(light_onset_phase_sw_net[2][2])]
offset_tb = [cv(light_offset_phase_sw_net[2][0]), cv(light_offset_phase_sw_net[2][1]), cv(light_offset_phase_sw_net[2][2])]
fig, ax = plt.subplots(tight_layout=True, figsize=(5, 5))
ax.scatter(np.arange(3), onset_cm, s=120, c=colors_networks[0])
ax.scatter(np.arange(3), offset_cm, s=120, facecolors='none', edgecolors=colors_networks[0])
ax.scatter(np.arange(3), onset_hr, s=120, c=colors_networks[1])
ax.scatter(np.arange(3), offset_hr, s=120, facecolors='none', edgecolors=colors_networks[1])
ax.scatter(np.arange(3), onset_tb, s=120, c=colors_networks[2])
ax.scatter(np.arange(3), offset_tb, s=120, facecolors='none', edgecolors=colors_networks[2])
ax.set_xticks(np.arange(3))
ax.set_xticklabels(speeds_label, fontsize=16, rotation=15)
ax.set_ylabel('CV (abs. value) on\nswing onset condition', fontsize=16)
ax.set_xlabel('Speed', fontsize=16)
ax.set_ylim([0, 1.7])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(summary_path, 'cv_timing_sw'), dpi=128)
plt.savefig(os.path.join(summary_path, 'cv_timing_sw.svg'), dpi=128)

# ###### STATS #######
# #Mann-Whitney on accuracy for the three computed speeds - network comparison stance
# accuracy_stats_st = accuracy_measures_st[:, 0, :, :].reshape((np.shape(accuracy_measures_st[:, 0, :, :])[0]*np.shape(accuracy_measures_st[:, 0, :, :])[1], np.shape(accuracy_measures_st[:, 0, :, :])[2]))
# res_accuracy_st_cm_hr = scipy.stats.mannwhitneyu(accuracy_stats_st[:, 0], accuracy_stats_st[:, 1], method='exact')
# print(res_accuracy_st_cm_hr)
# res_accuracy_st_cm_tb = scipy.stats.mannwhitneyu(accuracy_stats_st[:, 0], accuracy_stats_st[:, 2], method='exact')
# print(res_accuracy_st_cm_tb)
# res_accuracy_st_hr_tb = scipy.stats.mannwhitneyu(accuracy_stats_st[:, 1], accuracy_stats_st[:, 2], method='exact')
# print(res_accuracy_st_hr_tb)
#
# #Mann-Whitney on accuracy for the three computed speeds - network comparison swing
# accuracy_stats_sw = accuracy_measures_sw[:, 0, :, :].reshape((np.shape(accuracy_measures_sw[:, 0, :, :])[0]*np.shape(accuracy_measures_sw[:, 0, :, :])[1], np.shape(accuracy_measures_sw[:, 0, :, :])[2]))
# res_accuracy_sw_cm_hr = scipy.stats.mannwhitneyu(accuracy_stats_sw[:, 0], accuracy_stats_sw[:, 1], method='exact')
# print(res_accuracy_sw_cm_hr)
# res_accuracy_sw_cm_tb = scipy.stats.mannwhitneyu(accuracy_stats_sw[:, 0], accuracy_stats_sw[:, 2], method='exact')
# print(res_accuracy_sw_cm_tb)
# res_accuracy_sw_hr_tb = scipy.stats.mannwhitneyu(accuracy_stats_sw[:, 1], accuracy_stats_sw[:, 2], method='exact')
# print(res_accuracy_sw_hr_tb)

# #ANOVA on onset times across speeds - network comparison stance
# #CM
# fvalue_onset_cm_st, pvalue_onset_cm_st = scipy.stats.f_oneway(light_onset_phase_st_net[0][0], light_onset_phase_st_net[0][1], light_onset_phase_st_net[0][2])
# print(fvalue_onset_cm_st, pvalue_onset_cm_st)
# dict_onset_cm_st = {'value': np.concatenate((light_onset_phase_st_net[0][0], light_onset_phase_st_net[0][1], light_onset_phase_st_net[0][2])),
#     'network': np.concatenate((np.repeat('0,275', len(light_onset_phase_st_net[0][0])), np.repeat('splitright', len(light_onset_phase_st_net[0][1])),
#         np.repeat('splitleft', len(light_onset_phase_st_net[0][2]))))}
# df_onset_cm_st = pd.DataFrame(dict_onset_cm_st)
# res_onset_cm_st = stat()
# res_onset_cm_st.tukey_hsd(df=df_onset_cm_st, res_var='value', xfac_var='network', anova_model='value ~ C(network)')
# res_onset_cm_st.tukey_summary
# fvalue_offset_cm_st, pvalue_offset_cm_st = scipy.stats.f_oneway(light_offset_phase_st_net[0][0], light_offset_phase_st_net[0][1], light_offset_phase_st_net[0][2])
# print(fvalue_offset_cm_st, pvalue_offset_cm_st)
# dict_offset_cm_st = {'value': np.concatenate((light_offset_phase_st_net[0][0], light_offset_phase_st_net[0][1], light_offset_phase_st_net[0][2])),
#     'network': np.concatenate((np.repeat('0,275', len(light_offset_phase_st_net[0][0])), np.repeat('splitright', len(light_offset_phase_st_net[0][1])),
#         np.repeat('splitleft', len(light_offset_phase_st_net[0][2]))))}
# df_offset_cm_st = pd.DataFrame(dict_offset_cm_st)
# res_offset_cm_st = stat()
# res_offset_cm_st.tukey_hsd(df=df_offset_cm_st, res_var='value', xfac_var='network', anova_model='value ~ C(network)')
# res_offset_cm_st.tukey_summary
# #HR
# fvalue_onset_hr_st, pvalue_onset_hr_st = scipy.stats.f_oneway(light_onset_phase_st_net[1][0], light_onset_phase_st_net[1][1], light_onset_phase_st_net[1][2])
# print(fvalue_onset_hr_st, pvalue_onset_hr_st)
# dict_onset_hr_st = {'value': np.concatenate((light_onset_phase_st_net[1][0], light_onset_phase_st_net[1][1], light_onset_phase_st_net[1][2])),
#     'network': np.concatenate((np.repeat('0,275', len(light_onset_phase_st_net[1][0])), np.repeat('splitright', len(light_onset_phase_st_net[1][1])),
#         np.repeat('splitleft', len(light_onset_phase_st_net[1][2]))))}
# df_onset_hr_st = pd.DataFrame(dict_onset_hr_st)
# res_onset_hr_st = stat()
# res_onset_hr_st.tukey_hsd(df=df_onset_hr_st, res_var='value', xfac_var='network', anova_model='value ~ C(network)')
# res_onset_hr_st.tukey_summary
# fvalue_offset_hr_st, pvalue_offset_hr_st = scipy.stats.f_oneway(light_offset_phase_st_net[1][0], light_offset_phase_st_net[1][1], light_offset_phase_st_net[1][2])
# print(fvalue_offset_hr_st, pvalue_offset_hr_st)
# #TB
# fvalue_onset_tb_st, pvalue_onset_tb_st = scipy.stats.f_oneway(light_onset_phase_st_net[0][0], light_onset_phase_st_net[0][1], light_onset_phase_st_net[0][2])
# print(fvalue_onset_tb_st, pvalue_onset_tb_st)
# dict_onset_tb_st = {'value': np.concatenate((light_onset_phase_st_net[2][0], light_onset_phase_st_net[2][1], light_onset_phase_st_net[2][2])),
#     'network': np.concatenate((np.repeat('0,275', len(light_onset_phase_st_net[2][0])), np.repeat('splitright', len(light_onset_phase_st_net[2][1])),
#         np.repeat('splitleft', len(light_onset_phase_st_net[2][2]))))}
# df_onset_tb_st = pd.DataFrame(dict_onset_tb_st)
# res_onset_tb_st = stat()
# res_onset_tb_st.tukey_hsd(df=df_onset_tb_st, res_var='value', xfac_var='network', anova_model='value ~ C(network)')
# res_onset_tb_st.tukey_summary
# fvalue_offset_tb_st, pvalue_offset_tb_st = scipy.stats.f_oneway(light_offset_phase_st_net[2][0], light_offset_phase_st_net[2][1], light_offset_phase_st_net[2][2])
# print(fvalue_offset_tb_st, pvalue_offset_tb_st)
# dict_offset_tb_st = {'value': np.concatenate((light_offset_phase_st_net[2][0], light_offset_phase_st_net[2][1], light_offset_phase_st_net[2][2])),
#     'network': np.concatenate((np.repeat('0,275', len(light_offset_phase_st_net[2][0])), np.repeat('splitright', len(light_offset_phase_st_net[2][1])),
#         np.repeat('splitleft', len(light_offset_phase_st_net[2][2]))))}
# df_offset_tb_st = pd.DataFrame(dict_offset_tb_st)
# res_offset_tb_st = stat()
# res_offset_tb_st.tukey_hsd(df=df_offset_tb_st, res_var='value', xfac_var='network', anova_model='value ~ C(network)')
# res_offset_tb_st.tukey_summary
#
# #ANOVA on onset times for 0,275m/s - network comparison stance
# fvalue_onset_st, pvalue_onset_st = scipy.stats.f_oneway(light_onset_phase_st_net[0][0], light_onset_phase_st_net[1][0], light_onset_phase_st_net[2][0])
# print(fvalue_onset_st, pvalue_onset_st)
# #Tukey HSD test for multiple pairwise comparison
# median_networks_onset_st = np.round([np.nanmedian(light_onset_phase_st_net[0][0]), np.nanmedian(light_onset_phase_st_net[1][0]), np.nanmedian(light_onset_phase_st_net[2][0])], 3)
# print(median_networks_onset_st)
# dict_onset_st = {'value': np.concatenate((light_onset_phase_st_net[0][0], light_onset_phase_st_net[1][0], light_onset_phase_st_net[2][0])),
#     'network': np.concatenate((np.repeat('CM', len(light_onset_phase_st_net[0][0])), np.repeat('HR', len(light_onset_phase_st_net[1][0])),
#         np.repeat('TB', len(light_onset_phase_st_net[2][0]))))}
# df_onset_st = pd.DataFrame(dict_onset_st)
# res_onset_st = stat()
# res_onset_st.tukey_hsd(df=df_onset_st, res_var='value', xfac_var='network', anova_model='value ~ C(network)')
# res_onset_st.tukey_summary
#
# #ANOVA on offset times for 0,275m/s - network comparison stance
# fvalue_offset_st, pvalue_offset_st = scipy.stats.f_oneway(light_offset_phase_st_net[0][0], light_offset_phase_st_net[1][0], light_offset_phase_st_net[2][0])
# print(fvalue_offset_st, pvalue_offset_st)
# #Tukey HSD test for multiple pairwise comparison
# median_networks_offset_st = np.round([np.nanmedian(light_offset_phase_st_net[0][0]), np.nanmedian(light_offset_phase_st_net[1][0]), np.nanmedian(light_offset_phase_st_net[2][0])], 3)
# print(median_networks_offset_st)
# dict_offset_st = {'value': np.concatenate((light_offset_phase_st_net[0][0], light_offset_phase_st_net[1][0], light_offset_phase_st_net[2][0])),
#     'network': np.concatenate((np.repeat('CM', len(light_offset_phase_st_net[0][0])), np.repeat('HR', len(light_offset_phase_st_net[1][0])),
#         np.repeat('TB', len(light_offset_phase_st_net[2][0]))))}
# df_offset_st = pd.DataFrame(dict_offset_st)
# res_offset_st = stat()
# res_offset_st.tukey_hsd(df=df_offset_st, res_var='value', xfac_var='network', anova_model='value ~ C(network)')
# res_offset_st.tukey_summary
#
# #ANOVA on onset times across speeds - network comparison swing
# #CM
# fvalue_onset_cm_sw, pvalue_onset_cm_sw = scipy.stats.f_oneway(light_onset_phase_sw_net[0][0], light_onset_phase_sw_net[0][1], light_onset_phase_sw_net[0][2])
# print(fvalue_onset_cm_sw, pvalue_onset_cm_sw)
# dict_onset_cm_sw = {'value': np.concatenate((light_onset_phase_sw_net[0][0], light_onset_phase_sw_net[0][1], light_onset_phase_sw_net[0][2])),
#     'network': np.concatenate((np.repeat('0,275', len(light_onset_phase_sw_net[0][0])), np.repeat('splitright', len(light_onset_phase_sw_net[0][1])),
#         np.repeat('splitleft', len(light_onset_phase_sw_net[0][2]))))}
# df_onset_cm_sw = pd.DataFrame(dict_onset_cm_sw)
# res_onset_cm_sw = stat()
# res_onset_cm_sw.tukey_hsd(df=df_onset_cm_sw, res_var='value', xfac_var='network', anova_model='value ~ C(network)')
# res_onset_cm_sw.tukey_summary
# fvalue_offset_cm_sw, pvalue_offset_cm_sw = scipy.stats.f_oneway(light_offset_phase_sw_net[0][0], light_offset_phase_sw_net[0][1], light_offset_phase_sw_net[0][2])
# print(fvalue_offset_cm_sw, pvalue_offset_cm_sw)
# dict_offset_cm_sw = {'value': np.concatenate((light_offset_phase_sw_net[0][0], light_offset_phase_sw_net[0][1], light_offset_phase_sw_net[0][2])),
#     'network': np.concatenate((np.repeat('0,275', len(light_offset_phase_sw_net[0][0])), np.repeat('splitright', len(light_offset_phase_sw_net[0][1])),
#         np.repeat('splitleft', len(light_offset_phase_sw_net[0][2]))))}
# df_offset_cm_sw = pd.DataFrame(dict_offset_cm_sw)
# res_offset_cm_sw = stat()
# res_offset_cm_sw.tukey_hsd(df=df_offset_cm_sw, res_var='value', xfac_var='network', anova_model='value ~ C(network)')
# res_offset_cm_sw.tukey_summary
# #HR
# fvalue_onset_hr_sw, pvalue_onset_hr_sw = scipy.stats.f_oneway(light_onset_phase_sw_net[1][0], light_onset_phase_sw_net[1][1], light_onset_phase_sw_net[1][2])
# print(fvalue_onset_hr_sw, pvalue_onset_hr_sw)
# fvalue_offset_hr_sw, pvalue_offset_hr_sw = scipy.stats.f_oneway(light_offset_phase_sw_net[1][0], light_offset_phase_sw_net[1][1], light_offset_phase_sw_net[1][2])
# print(fvalue_offset_hr_sw, pvalue_offset_hr_sw)
# dict_offset_hr_sw = {'value': np.concatenate((light_offset_phase_sw_net[1][0], light_offset_phase_sw_net[1][1], light_offset_phase_sw_net[1][2])),
#     'network': np.concatenate((np.repeat('0,275', len(light_offset_phase_sw_net[1][0])), np.repeat('splitright', len(light_offset_phase_sw_net[1][1])),
#         np.repeat('splitleft', len(light_offset_phase_sw_net[1][2]))))}
# df_offset_hr_sw = pd.DataFrame(dict_offset_hr_sw)
# res_offset_hr_sw = stat()
# res_offset_hr_sw.tukey_hsd(df=df_offset_hr_sw, res_var='value', xfac_var='network', anova_model='value ~ C(network)')
# res_offset_hr_sw.tukey_summary
# #TB
# fvalue_onset_tb_sw, pvalue_onset_tb_sw = scipy.stats.f_oneway(light_onset_phase_sw_net[0][0], light_onset_phase_sw_net[0][1], light_onset_phase_sw_net[0][2])
# print(fvalue_onset_tb_sw, pvalue_onset_tb_sw)
# dict_onset_tb_sw = {'value': np.concatenate((light_onset_phase_sw_net[2][0], light_onset_phase_sw_net[2][1], light_onset_phase_sw_net[2][2])),
#     'network': np.concatenate((np.repeat('0,275', len(light_onset_phase_sw_net[2][0])), np.repeat('splitright', len(light_onset_phase_sw_net[2][1])),
#         np.repeat('splitleft', len(light_onset_phase_sw_net[2][2]))))}
# df_onset_tb_sw = pd.DataFrame(dict_onset_tb_sw)
# res_onset_tb_sw = stat()
# res_onset_tb_sw.tukey_hsd(df=df_onset_tb_sw, res_var='value', xfac_var='network', anova_model='value ~ C(network)')
# res_onset_tb_sw.tukey_summary
# fvalue_offset_tb_sw, pvalue_offset_tb_sw = scipy.stats.f_oneway(light_offset_phase_sw_net[2][0], light_offset_phase_sw_net[2][1], light_offset_phase_sw_net[2][2])
# print(fvalue_offset_tb_sw, pvalue_offset_tb_sw)
# dict_offset_tb_sw = {'value': np.concatenate((light_offset_phase_sw_net[2][0], light_offset_phase_sw_net[2][1], light_offset_phase_sw_net[2][2])),
#     'network': np.concatenate((np.repeat('0,275', len(light_offset_phase_sw_net[2][0])), np.repeat('splitright', len(light_offset_phase_sw_net[2][1])),
#         np.repeat('splitleft', len(light_offset_phase_sw_net[2][2]))))}
# df_offset_tb_sw = pd.DataFrame(dict_offset_tb_sw)
# res_offset_tb_sw = stat()
# res_offset_tb_sw.tukey_hsd(df=df_offset_tb_sw, res_var='value', xfac_var='network', anova_model='value ~ C(network)')
# res_offset_tb_sw.tukey_summary
#
# #ANOVA on onset times for 0,275m/s - network comparison swing
# fvalue_onset_sw, pvalue_onset_sw = scipy.stats.f_oneway(light_onset_phase_sw_net[0][0], light_onset_phase_sw_net[1][0], light_onset_phase_sw_net[2][0])
# print(fvalue_onset_sw, pvalue_onset_sw)
# #Tukey HSD test for multiple pairwise comparison
# median_networks_onset_sw = np.round([np.nanmedian(light_onset_phase_sw_net[0][0]), np.nanmedian(light_onset_phase_sw_net[1][0]), np.nanmedian(light_onset_phase_sw_net[2][0])], 3)
# print(median_networks_onset_sw)
# dict_onset_sw = {'value': np.concatenate((light_onset_phase_sw_net[0][0], light_onset_phase_sw_net[1][0], light_onset_phase_sw_net[2][0])),
#     'network': np.concatenate((np.repeat('CM', len(light_onset_phase_sw_net[0][0])), np.repeat('HR', len(light_onset_phase_sw_net[1][0])),
#         np.repeat('TB', len(light_onset_phase_sw_net[2][0]))))}
# df_onset_sw = pd.DataFrame(dict_onset_sw)
# res_onset_sw = stat()
# res_onset_sw.tukey_hsd(df=df_onset_sw, res_var='value', xfac_var='network', anova_model='value ~ C(network)')
# res_onset_sw.tukey_summary
#
# #ANOVA on offset times for 0,275m/s - network comparison swing
# fvalue_offset_sw, pvalue_offset_sw = scipy.stats.f_oneway(light_offset_phase_sw_net[0][0], light_offset_phase_sw_net[1][0], light_offset_phase_sw_net[2][0])
# print(fvalue_offset_sw, pvalue_offset_sw)
# #Tukey HSD test for multiple pairwise comparison
# median_networks_offset_sw = np.round([np.nanmedian(light_offset_phase_sw_net[0][0]), np.nanmedian(light_offset_phase_sw_net[1][0]), np.nanmedian(light_offset_phase_sw_net[2][0])], 3)
# print(median_networks_offset_sw)
# dict_offset_sw = {'value': np.concatenate((light_offset_phase_sw_net[0][0], light_offset_phase_sw_net[1][0], light_offset_phase_sw_net[2][0])),
#     'network': np.concatenate((np.repeat('CM', len(light_offset_phase_sw_net[0][0])), np.repeat('HR', len(light_offset_phase_sw_net[1][0])),
#         np.repeat('TB', len(light_offset_phase_sw_net[2][0]))))}
# df_offset_sw = pd.DataFrame(dict_offset_sw)
# res_offset_sw = stat()
# res_offset_sw.tukey_hsd(df=df_offset_sw, res_var='value', xfac_var='network', anova_model='value ~ C(network)')
# res_offset_sw.tukey_summary
