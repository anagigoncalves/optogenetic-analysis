import os
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
np.warnings.filterwarnings('ignore')

network = 'HR tests'
conditions = ['25percent', '50percent', '75percent']
conditions_name = ['25%', '50%', '75%']
speeds = ['0,175', '0,275', '0,375', 'right_fast', 'left_fast']
trials_reshape = np.reshape(np.arange(1, 11), (5, 2)) #0.175, 0.275, 0.375, right fast, left fast
measure_name = ['accuracy', 'f1_score', 'false_negatives', 'false_positives']
measure_name_label = ['Accuracy', 'F1 score', 'False negatives', 'False positives']
cmap_speeds = plt.get_cmap('magma')
colors_speeds = [cmap_speeds(i) for i in np.linspace(0, 1, int(np.floor(len(speeds) + 1)))]
summary_path = 'J:\\Opto Benchmarks\\Benchmark plots\\Selected plots\\'

for idx_speed in range(len(speeds)):
    stim_duration_st_cond = []
    stim_duration_sw_cond = []
    light_onset_phase_st_cond = []
    light_onset_phase_sw_cond = []
    light_offset_phase_st_cond = []
    light_offset_phase_sw_cond = []
    stim_nr_st_cond = []
    stim_nr_sw_cond = []
    stride_nr_st_cond = []
    stride_nr_sw_cond = []
    for count_c, c in enumerate(conditions):
        path = os.path.join('J:\\Opto Benchmarks', network, c)
        if not os.path.exists(os.path.join(path, 'plots')):
            os.mkdir(os.path.join(path, 'plots'))
        import online_tracking_class
        otrack_class = online_tracking_class.otrack_class(path)
        import locomotion_class
        loco = locomotion_class.loco_class(path)
        # frac_strides_st[:, count_c, idx_speed] = np.load(os.path.join(path, 'processed files', 'frac_strides_st.npy'), allow_pickle=True)[:, idx_speed, :].flatten()
        # frac_strides_sw[:, count_c, idx_speed] = np.load(
        #     os.path.join(path, 'processed files', 'frac_strides_sw.npy'), allow_pickle=True)[:, idx_speed, :].flatten()
        stim_duration_st_list = np.load(os.path.join(path, 'processed files', 'stim_duration_st.npy'), allow_pickle=True)
        stim_duration_st_cond.append(list(itertools.chain(*stim_duration_st_list[:, idx_speed])))
        stim_duration_sw_list = np.load(os.path.join(path, 'processed files', 'stim_duration_sw.npy'), allow_pickle=True)
        stim_duration_sw_cond.append(list(itertools.chain(*stim_duration_sw_list[:, idx_speed])))
        light_onset_phase_st_list = np.load(os.path.join(path, 'processed files', 'light_onset_phase_st.npy'), allow_pickle=True)
        light_onset_phase_st_cond.append(list(itertools.chain(*light_onset_phase_st_list[:, idx_speed])))
        light_onset_phase_sw_list = np.load(os.path.join(path, 'processed files', 'light_onset_phase_sw.npy'), allow_pickle=True)
        light_onset_phase_sw_cond.append(list(itertools.chain(*light_onset_phase_sw_list[:, idx_speed])))
        light_offset_phase_st_list = np.load(os.path.join(path, 'processed files', 'light_offset_phase_st.npy'), allow_pickle=True)
        light_offset_phase_st_cond.append(list(itertools.chain(*light_offset_phase_st_list[:, idx_speed])))
        light_offset_phase_sw_list = np.load(os.path.join(path, 'processed files', 'light_offset_phase_sw.npy'), allow_pickle=True)
        light_offset_phase_sw_cond.append(list(itertools.chain(*light_offset_phase_sw_list[:, idx_speed])))
        stim_nr_st_list = np.load(os.path.join(path, 'processed files', 'stim_nr_st.npy'), allow_pickle=True)
        stim_nr_st_cond.append(list(itertools.chain(*stim_nr_st_list[:, idx_speed])))
        stim_nr_sw_list = np.load(os.path.join(path, 'processed files', 'stim_nr_sw.npy'), allow_pickle=True)
        stim_nr_sw_cond.append(list(itertools.chain(*stim_nr_sw_list[:, idx_speed])))
        stride_nr_st_list = np.load(os.path.join(path, 'processed files', 'stride_nr_st.npy'), allow_pickle=True)
        stride_nr_st_cond.append(list(itertools.chain(*stride_nr_st_list[:, idx_speed])))
        stride_nr_sw_list = np.load(os.path.join(path, 'processed files', 'stride_nr_sw.npy'), allow_pickle=True)
        stride_nr_sw_cond.append(list(itertools.chain(*stride_nr_sw_list[:, idx_speed])))
        benchmark_accuracy = pd.read_csv(os.path.join(path, 'processed files', 'benchmark_accuracy.csv'))
        trials = trials_reshape[idx_speed, :]

    # DURATION - HISTOGRAM PLOT
    for count_c, c in enumerate(conditions):
        fig, ax = plt.subplots(tight_layout=True, figsize=(7, 3))
        ax.hist(stim_duration_st_cond[count_c], bins=100, histtype='step', color='orange', linewidth=4)
        ax.set_xlabel('Time (s)', fontsize=14)
        ax.set_ylabel('Counts', fontsize=14)
        ax.set_xlim([0, 0.3])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(os.path.join(summary_path, 'stim_duration_hist_st_' + network + '_' + c + speeds[idx_speed]), dpi=128)
        plt.savefig(os.path.join(summary_path, 'stim_duration_hist_st_' + network + '_' + c + speeds[idx_speed] + '.svg'), dpi=128)
        fig, ax = plt.subplots(tight_layout=True, figsize=(7, 3))
        ax.hist(stim_duration_sw_cond[count_c], bins=100, histtype='step', color='green', linewidth=4)
        ax.set_xlabel('Time (s)', fontsize=14)
        ax.set_ylabel('Counts', fontsize=14)
        ax.set_xlim([0, 0.3])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(os.path.join(summary_path, 'stim_duration_hist_sw_' + network + '_' + c + speeds[idx_speed]), dpi=128)
        plt.savefig(os.path.join(summary_path, 'stim_duration_hist_sw_' + network + '_' + c + speeds[idx_speed] + '.svg'), dpi=128)
    plt.close('all')


