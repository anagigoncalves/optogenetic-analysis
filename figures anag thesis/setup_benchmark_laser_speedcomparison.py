import os
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
np.warnings.filterwarnings('ignore')

networks = ['CM tests', 'HR tests', 'Tailbase tests']
conditions = ['25percent', '50percent', '75percent']
conditions_name = ['25%', '50%', '75%']
speeds = ['0,175', '0,275', '0,375', 'right_fast', 'left_fast']
trials_reshape = np.reshape(np.arange(1, 11), (5, 2)) #0.175, 0.275, 0.375, right fast, left fast
measure_name = ['accuracy', 'f1_score', 'false_negatives', 'false_positives']
measure_name_label = ['Accuracy', 'F1 score', 'False negatives', 'False positives']
cmap_speeds = plt.get_cmap('magma')
colors_speeds = [cmap_speeds(i) for i in np.linspace(0, 1, int(np.floor(len(speeds) + 1)))]
summary_path = 'J:\\Opto Benchmarks\\Benchmark plots\\For each marker comparison of speed and threshold\\'

for count_n, n in enumerate(networks):
    accuracy_measures_st = np.zeros((12, 4, len(conditions), len(speeds)))
    accuracy_measures_sw = np.zeros((12, 4, len(conditions), len(speeds)))
    frac_strides_st = np.zeros((12, len(conditions), len(speeds)))
    frac_strides_sw = np.zeros((12, len(conditions), len(speeds)))
    stim_duration_st_net = []
    stim_duration_sw_net = []
    light_onset_phase_st_net = []
    light_onset_phase_sw_net = []
    light_offset_phase_st_net = []
    light_offset_phase_sw_net = []
    stim_nr_st_net = []
    stim_nr_sw_net = []
    stride_nr_st_net = []
    stride_nr_sw_net = []
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
            path = os.path.join('J:\\Opto Benchmarks', n, c)
            if not os.path.exists(os.path.join(path, 'plots')):
                os.mkdir(os.path.join(path, 'plots'))
            import online_tracking_class
            otrack_class = online_tracking_class.otrack_class(path)
            import locomotion_class
            loco = locomotion_class.loco_class(path)
            frac_strides_st[:, count_c, idx_speed] = np.load(os.path.join(path, 'processed files', 'frac_strides_st.npy'), allow_pickle=True)[:, idx_speed, :].flatten()
            frac_strides_sw[:, count_c, idx_speed] = np.load(
                os.path.join(path, 'processed files', 'frac_strides_sw.npy'), allow_pickle=True)[:, idx_speed, :].flatten()
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
            accuracy_measures_st[:, :, count_c, idx_speed] = benchmark_accuracy[benchmark_accuracy['trial'].isin(trials)].iloc[:, 3::2]
            accuracy_measures_sw[:, :, count_c, idx_speed] = benchmark_accuracy[benchmark_accuracy['trial'].isin(trials)].iloc[:, 3::2]
        stim_duration_st_net.append(stim_duration_st_cond)
        stim_duration_sw_net.append(stim_duration_sw_cond)
        light_onset_phase_st_net.append(light_onset_phase_st_cond)
        light_onset_phase_sw_net.append(light_onset_phase_sw_cond)
        light_offset_phase_st_net.append(light_offset_phase_st_cond)
        light_offset_phase_sw_net.append(light_offset_phase_sw_cond)
        stim_nr_st_net.append(stim_nr_st_cond)
        stim_nr_sw_net.append(stim_nr_sw_cond)
        stride_nr_st_net.append(stride_nr_st_cond)
        stride_nr_sw_net.append(stride_nr_sw_cond)

    # STIMULATION ONSETS AND OFFSETS
    for count_c, c in enumerate(conditions):
        for i in range(len(speeds)):
            [fraction_strides_stim_st_on, fraction_strides_stim_st_off] = \
            otrack_class.plot_laser_presentation_phase_benchmark(light_onset_phase_st_net[i][count_c],
            light_offset_phase_st_net[i][count_c], 'stance', 16, np.sum(stim_nr_st_net[i][count_c]), np.sum(stride_nr_st_net[i][count_c]), 'Greys',
                    summary_path, '\\light_stance_'+n+'_'+speeds[i]+'_'+conditions[count_c])
            [fraction_strides_stim_sw_on, fraction_strides_stim_sw_off] = \
            otrack_class.plot_laser_presentation_phase_benchmark(light_onset_phase_sw_net[i][count_c],
            light_offset_phase_sw_net[i][count_c], 'swing', 16, np.sum(stim_nr_sw_net[i][count_c]), np.sum(stride_nr_sw_net[i][count_c]), 'Greys',
                    summary_path, '\\light_swing_'+n+'_'+speeds[i]+'_'+conditions[count_c])
            otrack_class.plot_laser_presentation_phase_hist(light_onset_phase_st_net[i][count_c],
                                                            light_offset_phase_st_net[i][count_c],
                                                            20, summary_path,
                                                            '\\light_stance_hist_'+n+'_'+speeds[i]+'_'+conditions[count_c], 1)
            otrack_class.plot_laser_presentation_phase_hist(light_onset_phase_sw_net[i][count_c],
                                                            light_offset_phase_sw_net[i][count_c],
                                                            20, summary_path,
                                                            '\\light_swing_hist_'+n+'_'+speeds[i]+'_'+conditions[count_c], 1)
            plt.close('all')

    # FRACTION OF STIMULATED STRIDES
    speeds_label = ['0.175 m/s', '0.275 m/s', '0.375 m/s', 'split right fast', 'split left fast']
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
    for s in range(len(speeds)):
        for a in range(12):
            if a == 0:
                ax.scatter(np.arange(0, 30, 10) + (np.ones(3) * s) + np.random.rand(3),
                           frac_strides_st[a, :, s],
                           s=10, color=colors_speeds[s], label=speeds[s])
            else:
                ax.scatter(np.arange(0, 30, 10) + (np.ones(3) * s) + np.random.rand(3),
                           frac_strides_st[a, :, s],
                           s=10, color=colors_speeds[s], label='_nolegend_')
        ax.scatter(np.arange(0, 30, 10) + 1.2*s, np.nanmean(frac_strides_st[:, :, s], axis=0), s=200, marker='_', color=colors_speeds[s])
    # ax.legend(speeds_label, frameon=False, fontsize=14)
    ax.set_xticks(np.arange(0, 30, 10) + 2.5)
    ax.set_xticklabels(conditions_name, fontsize=14)
    ax.set_ylabel('Fraction of LED-on\nstrides', fontsize=14)
    ax.set_ylim([0, 1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(summary_path, 'strides_stimulated_st_'+networks[count_n]), dpi=128)
    plt.savefig(os.path.join(summary_path, 'strides_stimulated_st_' + networks[count_n]+'.svg'), dpi=128)
    fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
    for s in range(len(speeds)):
        for a in range(12):
            if a == 0:
                ax.scatter(np.arange(0, 30, 10) + (np.ones(3) * s) + np.random.rand(3),
                           frac_strides_sw[a, :, s],
                           s=10, color=colors_speeds[s], label=speeds[s])
            else:
                ax.scatter(np.arange(0, 30, 10) + (np.ones(3) * s) + np.random.rand(3),
                           frac_strides_sw[a, :, s],
                           s=10, color=colors_speeds[s], label='_nolegend_')
        ax.scatter(np.arange(0, 30, 10) + 1.2 * s, np.nanmean(frac_strides_sw[:, :, s], axis=0), s=200, marker='_',
                   color=colors_speeds[s])
    # ax.legend(speeds, frameon=False, fontsize=12)
    ax.set_xticks(np.arange(0, 30, 10) + 2.5)
    ax.set_xticklabels(conditions_name, fontsize=14)
    ax.set_ylabel('Fraction of LED-on\nstrides', fontsize=14)
    ax.set_ylim([0, 1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(summary_path, 'strides_stimulated_sw_'+networks[count_n]), dpi=128)
    plt.savefig(os.path.join(summary_path, 'strides_stimulated_sw_' + networks[count_n]+'.svg'), dpi=128)

    # DURATION - VIOLIN PLOT
    fig, ax = plt.subplots(tight_layout=True, figsize=(7, 3))
    for s in range(len(speeds)):
        violin_parts = ax.violinplot(stim_duration_st_net[s], positions=np.arange(0, 9, 3) + (0.5 * s),
            showextrema=False, showmeans=True)
        for pc in violin_parts['bodies']:
            pc.set_color(colors_speeds[s])
        violin_parts['cmeans'].set_color(colors_speeds[s])
    ax.set_xticks(np.arange(0, 9, 3)+1)
    ax.set_xticklabels(conditions_name, fontsize=14)
    #ax.set_title('Stance stim duration ' + n, fontsize=16)
    ax.set_ylabel('LED-on duration (s)', fontsize=14)
    ax.set_ylim([0, 0.4])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(summary_path, 'stim_duration_st_' + n), dpi=128)
    plt.savefig(os.path.join(summary_path, 'stim_duration_st_' + n+'.svg'), dpi=128)
    fig, ax = plt.subplots(tight_layout=True, figsize=(7, 3))
    for s in range(len(speeds)):
        violin_parts = ax.violinplot(stim_duration_sw_net[s], positions=np.arange(0, 9, 3) + (0.5 * s),
            showextrema=False, showmeans=True)
        for pc in violin_parts['bodies']:
             pc.set_color(colors_speeds[s])
        violin_parts['cmeans'].set_color(colors_speeds[s])
    ax.set_xticks(np.arange(0, 9, 3)+1)
    ax.set_xticklabels(conditions_name, fontsize=14)
    #ax.set_title('Swing stim duration ' + n, fontsize=16)
    ax.set_ylabel('LED-on duration (s)', fontsize=14)
    ax.set_ylim([-0.1, 0.85])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(os.path.join(summary_path, 'stim_duration_sw_' + n), dpi=128)
    plt.savefig(os.path.join(summary_path, 'stim_duration_sw_' + n+'.svg'), dpi=128)

    # DURATION - HISTOGRAM PLOT
    for count_c, c in enumerate(conditions):
        fig, ax = plt.subplots(tight_layout=True, figsize=(7, 3))
        for s in range(len(speeds)):
            ax.hist(stim_duration_st_net[s][count_c], bins=100, histtype='step', color=colors_speeds[s], linewidth=4)
        ax.set_xlabel('LED-on duration (s)', fontsize=14)
        ax.set_ylabel('Counts', fontsize=14)
        ax.set_xlim([0, 0.5])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(os.path.join(summary_path, 'stim_duration_hist_st_' + n + '_' + c), dpi=128)
        plt.savefig(os.path.join(summary_path, 'stim_duration_hist_st_' + n + '_' + c + '.svg'), dpi=128)
        fig, ax = plt.subplots(tight_layout=True, figsize=(7, 3))
        for s in range(len(speeds)):
            ax.hist(stim_duration_sw_net[s][count_c], bins=100, histtype='step', color=colors_speeds[s], linewidth=4)
        ax.set_xlabel('LED-on duration (s)', fontsize=14)
        ax.set_ylabel('Counts', fontsize=14)
        ax.set_xlim([0, 0.5])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(os.path.join(summary_path, 'stim_duration_hist_sw_' + n + '_' + c), dpi=128)
        plt.savefig(os.path.join(summary_path, 'stim_duration_hist_sw_' + n + '_' + c + '.svg'), dpi=128)
    plt.close('all')

    # ACCURACY
    ylabel_names = ['fraction correct hits', '% F1 score', 'fraction false negatives', 'fraction false positives']
    for i in range(len(measure_name)):
        fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
        for s in range(len(speeds)):
            for a in range(12):
                if a == 0:
                    ax.scatter(np.arange(0, 30, 10)+(np.ones(3)*s)+np.random.rand(3), accuracy_measures_st[a, i, :, s],
                               s=10, color=colors_speeds[s], label=speeds[s])
                else:
                    ax.scatter(np.arange(0, 30, 10)+(np.ones(3)*s)+np.random.rand(3), accuracy_measures_st[a, i, :, s],
                               s=10, color=colors_speeds[s], label='_nolegend_')
            ax.scatter(np.arange(0, 30, 10) + 1.2 * s, np.nanmean(accuracy_measures_st[:, i, :, s], axis=0), s=200, marker='_',
                       color=colors_speeds[s])
        # ax.legend(speeds_label, frameon=False, fontsize=12)
        ax.set_xticks(np.arange(0, 30, 10)+2.5)
        ax.set_xticklabels(conditions_name, fontsize=14)
        #ax.set_title('Stance ' + measure_name[i].replace('_', ' ') + ' ' + n, fontsize=16)
        ax.set_ylabel(measure_name_label[i], fontsize=14)
        ax.set_ylim([0, 1.2])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(os.path.join(summary_path, measure_name[i] + '_st_' + n), dpi=128)
        plt.savefig(os.path.join(summary_path, measure_name[i] + '_st_' + n+'.svg'), dpi=128)

        fig, ax = plt.subplots(tight_layout=True, figsize=(5, 3))
        for s in range(len(speeds)):
            for a in range(12):
                if a == 0:
                    ax.scatter(np.arange(0, 30, 10)+(np.ones(3)*s)+np.random.rand(3), accuracy_measures_sw[a, i, :, s],
                               s=10, color=colors_speeds[s], label=speeds[s])
                else:
                    ax.scatter(np.arange(0, 30, 10)+(np.ones(3)*s)+np.random.rand(3), accuracy_measures_sw[a, i, :, s],
                               s=10, color=colors_speeds[s], label='_nolegend_')
            ax.scatter(np.arange(0, 30, 10) + 1.2 * s, np.nanmean(accuracy_measures_st[:, i, :, s], axis=0), s=200, marker='_',
                       color=colors_speeds[s])
        # ax.legend(speeds, frameon=False, fontsize=12)
        ax.set_xticks(np.arange(0, 30, 10)+2.5)
        ax.set_xticklabels(conditions_name, fontsize=14)
        #ax.set_title('Swing ' + measure_name[i].replace('_', ' ') + ' ' + n, fontsize=16)
        ax.set_ylim([0, 1.2])
        ax.set_ylabel(measure_name_label[i], fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig(os.path.join(summary_path, measure_name[i] + '_sw_' + n), dpi=128)
        plt.savefig(os.path.join(summary_path, measure_name[i] + '_sw_' + n+'.svg'), dpi=128)
        plt.close('all')

