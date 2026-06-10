import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.warnings.filterwarnings('ignore')

path = 'J:\\Opto Benchmarks\\Tailbase tests\\75percent\\'
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

accuracy_st = []
f1_score_st = []
fn_st = []
fp_st = []
accuracy_sw = []
f1_score_sw = []
fn_sw = []
fp_sw = []
animal_id = []
trial_id = []
condition_id = []
stim_duration_st = []
stim_duration_sw = []
light_onset_phase_st = []
light_offset_phase_st = []
light_onset_phase_sw = []
light_offset_phase_sw = []
stim_nr_st_all = []
stride_nr_st_all = []
stim_nr_sw_all = []
stride_nr_sw_all = []
for count_a, animal in enumerate(animals):
    print('Processing ' + animal)
    trials = otrack_class.get_trials(animal)
    # LOAD PROCESSED DATA
    [otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, laser_on] = otrack_class.load_processed_files(animal)
    # LOAD DATA FOR BENCHMARK ANALYSIS
    [st_led_on, sw_led_on, frame_counter_session] = otrack_class.load_benchmark_files(animal)
    # READ OFFLINE PAW EXCURSIONS
    [final_tracks_trials, st_strides_trials, sw_strides_trials] = otrack_class.get_offtrack_paws(loco, animal, session)
    final_tracks_phase = loco.final_tracks_phase(final_tracks_trials, trials, st_strides_trials, sw_strides_trials,
                                                 'st-sw-st')
    # LASER ACCURACY
    tp_st_laser = np.zeros(len(trials))
    tn_st_laser = np.zeros(len(trials))
    f1_st_laser = np.zeros(len(trials))
    fn_st_laser = np.zeros(len(trials))
    fp_st_laser = np.zeros(len(trials))
    event = 'stance'
    for count_t, trial in enumerate(trials):
        if animal == 'MC18089' or  animal == 'MC18090' or animal == 'MC18091':
            [tp_trial, fp_trial, tn_trial, fn_trial, precision_trial, recall_trial, f1_trial] = otrack_class.accuracy_laser_sync(trial, event, offtracks_st, offtracks_sw, laser_on, final_tracks_trials, timestamps_session, 0)
        else:
            [tp_trial, fp_trial, tn_trial, fn_trial, precision_trial, recall_trial, f1_trial] = otrack_class.accuracy_light(trial, event, offtracks_st, offtracks_sw, st_led_on, sw_led_on, final_tracks_trials, timestamps_session, 0)
        tp_st_laser[count_t] = tp_trial
        tn_st_laser[count_t] = tn_trial
        f1_st_laser[count_t] = f1_trial
        fn_st_laser[count_t] = fn_trial
        fp_st_laser[count_t] = fp_trial
    accuracy_st.extend(tp_st_laser+tn_st_laser)
    f1_score_st.extend(f1_st_laser)
    fn_st.extend(fn_st_laser)
    fp_st.extend(fp_st_laser)
    tp_sw_laser = np.zeros(len(trials))
    tn_sw_laser = np.zeros(len(trials))
    f1_sw_laser = np.zeros(len(trials))
    fn_sw_laser = np.zeros(len(trials))
    fp_sw_laser = np.zeros(len(trials))
    event = 'swing'
    for count_t, trial in enumerate(trials):
        if animal == 'MC18089' or animal == 'MC18090' or animal == 'MC18091':
            [tp_trial, fp_trial, tn_trial, fn_trial, precision_trial, recall_trial, f1_trial] = otrack_class.accuracy_light(trial, event, offtracks_st, offtracks_sw, st_led_on, sw_led_on, final_tracks_trials, timestamps_session, 0)
        else:
            [tp_trial, fp_trial, tn_trial, fn_trial, precision_trial, recall_trial, f1_trial] = otrack_class.accuracy_light(trial, event, offtracks_st, offtracks_sw,  st_led_on, sw_led_on, final_tracks_trials, timestamps_session, 0)
        tp_sw_laser[count_t] = tp_trial
        tn_sw_laser[count_t] = tn_trial
        f1_sw_laser[count_t] = f1_trial
        fn_sw_laser[count_t] = fn_trial
        fp_sw_laser[count_t] = fp_trial
    accuracy_sw.extend(tp_sw_laser+tn_sw_laser)
    f1_score_sw.extend(f1_sw_laser)
    fn_sw.extend(fn_sw_laser)
    fp_sw.extend(fp_sw_laser)
    trial_id.extend(trials)
    animal_id.extend(np.repeat(animal, len(trials)))
    condition_id.extend(np.repeat(condition, len(trials)))

    # STIMULATION DURATION OFFSETS AND ONSETS
    trials_reshape = np.reshape(np.arange(1, 11), (5, 2))
    stim_duration_st_animal = []
    stim_duration_sw_animal = []
    stim_nr_st_trials = np.zeros(np.shape(trials_reshape)[0])
    stride_nr_st_trials = np.zeros(np.shape(trials_reshape)[0])
    stim_nr_sw_trials = np.zeros(np.shape(trials_reshape)[0])
    stride_nr_sw_trials = np.zeros(np.shape(trials_reshape)[0])
    light_onset_phase_st_animal = []
    light_offset_phase_st_animal = []
    light_onset_phase_sw_animal = []
    light_offset_phase_sw_animal = []
    stim_nr_st_animal = []
    stride_nr_st_animal = []
    stim_nr_sw_animal = []
    stride_nr_sw_animal = []
    for i in range(len(trials_reshape)):
        stim_duration_st_trialtype = []
        stim_duration_sw_trialtype = []
        light_onset_phase_st_trialtype = []
        light_offset_phase_st_trialtype = []
        light_onset_phase_sw_trialtype = []
        light_offset_phase_sw_trialtype = []
        stim_nr_st_trial = np.zeros(len(trials_reshape[i, :]))
        stride_nr_st_trial = np.zeros(len(trials_reshape[i, :]))
        stim_nr_sw_trial = np.zeros(len(trials_reshape[i, :]))
        stride_nr_sw_trial = np.zeros(len(trials_reshape[i, :]))
        for count_t, trial in enumerate(trials_reshape[i, :]):
            #stim duration
            stim_duration_st_trialtype.extend((laser_on.loc[laser_on['trial'] == trial]['time_off']-laser_on.loc[laser_on['trial'] == trial]['time_on']))
            stim_duration_sw_trialtype.extend((sw_led_on.loc[sw_led_on['trial'] == trial]['time_off']-sw_led_on.loc[sw_led_on['trial'] == trial]['time_on']))
            #stim phase
            if animal[0] == 'M':
                [light_onset_phase_st_trial, light_offset_phase_st_trial, stim_nr_st, stride_nr_st] = \
                    otrack_class.laser_presentation_phase_all(trial, trials, 'stance', offtracks_st, offtracks_sw, laser_on,
                                                          timestamps_session, final_tracks_phase, 'FR')
            if animal[0] == 'V':
                [light_onset_phase_st_trial, light_offset_phase_st_trial, stim_nr_st, stride_nr_st] = \
                    otrack_class.light_presentation_phase_all(trial, trials, 'stance', offtracks_st, offtracks_sw, st_led_on, sw_led_on,
                                                          timestamps_session, final_tracks_phase, 'FR')
            [light_onset_phase_sw_trial, light_offset_phase_sw_trial, stim_nr_sw, stride_nr_sw] = \
                otrack_class.light_presentation_phase_all(trial, trials, 'swing', offtracks_st, offtracks_sw, st_led_on, sw_led_on,
                                                      timestamps_session, final_tracks_phase, 'FR')
            stim_nr_st_trial[count_t] = stim_nr_st
            stride_nr_st_trial[count_t] = stride_nr_st
            stim_nr_sw_trial[count_t] = stim_nr_sw
            stride_nr_sw_trial[count_t] = stride_nr_sw
            light_onset_phase_st_trialtype.extend(light_onset_phase_st_trial)
            light_offset_phase_st_trialtype.extend(light_offset_phase_st_trial)
            light_onset_phase_sw_trialtype.extend(light_onset_phase_sw_trial)
            light_offset_phase_sw_trialtype.extend(light_offset_phase_sw_trial)
        stim_duration_st_animal.append(stim_duration_st_trialtype)
        stim_duration_sw_animal.append(stim_duration_sw_trialtype)
        light_onset_phase_st_animal.append(light_onset_phase_st_trialtype)
        light_offset_phase_st_animal.append(light_offset_phase_st_trialtype)
        light_onset_phase_sw_animal.append(light_onset_phase_sw_trialtype)
        light_offset_phase_sw_animal.append(light_offset_phase_sw_trialtype)
        stim_nr_st_animal.append(stim_nr_st_trial)
        stride_nr_st_animal.append(stride_nr_st_trial)
        stim_nr_sw_animal.append(stim_nr_sw_trial)
        stride_nr_sw_animal.append(stride_nr_sw_trial)
    stim_duration_st.append(stim_duration_st_animal)
    stim_duration_sw.append(stim_duration_sw_animal)
    light_onset_phase_st.append(light_onset_phase_st_animal)
    light_offset_phase_st.append(light_offset_phase_st_animal)
    light_onset_phase_sw.append(light_onset_phase_sw_animal)
    light_offset_phase_sw.append(light_offset_phase_sw_animal)
    stim_nr_st_all.append(stim_nr_st_animal)
    stride_nr_st_all.append(stride_nr_st_animal)
    stim_nr_sw_all.append(stim_nr_sw_animal)
    stride_nr_sw_all.append(stride_nr_sw_animal)

benchmark_accuracy = pd.DataFrame(
    {'condition': condition_id, 'animal': animal_id, 'trial': trial_id, 'accuracy_st': accuracy_st,
     'accuracy_sw': accuracy_sw, 'f1_score_st': f1_score_st, 'f1_score_sw': f1_score_sw, 'fn_st': fn_st,
     'fn_sw': fn_sw, 'fp_st': fp_st, 'fp_sw': fp_sw})
benchmark_accuracy.to_csv(os.path.join(otrack_class.path, 'processed files', 'benchmark_accuracy.csv'), sep=',', index=False)

np.save(os.path.join(otrack_class.path, 'processed files', 'stim_duration_st.npy'), stim_duration_st, allow_pickle=True)
np.save(os.path.join(otrack_class.path, 'processed files', 'stim_duration_sw.npy'), stim_duration_sw, allow_pickle=True)
np.save(os.path.join(otrack_class.path, 'processed files', 'light_onset_phase_st.npy'), light_onset_phase_st, allow_pickle=True)
np.save(os.path.join(otrack_class.path, 'processed files', 'light_offset_phase_st.npy'), light_offset_phase_st, allow_pickle=True)
np.save(os.path.join(otrack_class.path, 'processed files', 'light_onset_phase_sw.npy'), light_onset_phase_sw, allow_pickle=True)
np.save(os.path.join(otrack_class.path, 'processed files', 'light_offset_phase_sw.npy'), light_offset_phase_sw, allow_pickle=True)
np.save(os.path.join(otrack_class.path, 'processed files', 'stim_nr_st.npy'), stim_nr_st_all, allow_pickle=True)
np.save(os.path.join(otrack_class.path, 'processed files', 'stride_nr_st.npy'), stride_nr_st_all, allow_pickle=True)
np.save(os.path.join(otrack_class.path, 'processed files', 'stim_nr_sw.npy'), stim_nr_sw_all, allow_pickle=True)
np.save(os.path.join(otrack_class.path, 'processed files', 'stride_nr_sw.npy'), stride_nr_sw_all, allow_pickle=True)

# STIMULATION ACCURACY MEASURES
trials_reshape = np.reshape(np.arange(1, 11), (5, 2))
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
for count_a, animal in enumerate(animals):
    f1_data = np.array(benchmark_accuracy.loc[benchmark_accuracy['animal'] == animal, 'f1_score_st'])
    trials_ave = np.zeros(len(trials_reshape))
    for i in range(len(trials_reshape)):
        ax.scatter(np.ones(2)*trials_reshape[i, 0], f1_data[trials_reshape[i, :]-1], s=80, color=colors_animals[count_a])
        trials_ave[i] = np.nanmedian(f1_data[trials_reshape[i, :]-1])
    ax.plot(trials_reshape[:, 0], trials_ave, color=colors_animals[count_a], linewidth=3)
ax.set_xticks(trials_reshape[:, 0])
ax.set_xticklabels(['0.175', '0.275', '0.375', 'split ipsi\nfast', 'split contra\nfast'], fontsize=14)
ax.set_title('Stance F1 score ' + condition, fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'f1_score_st_' + condition), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
for count_a, animal in enumerate(animals):
    f1_data = np.array(benchmark_accuracy.loc[benchmark_accuracy['animal'] == animal, 'f1_score_sw'])
    trials_ave = np.zeros(len(trials_reshape))
    for i in range(len(trials_reshape)):
        ax.scatter(np.ones(2)*trials_reshape[i, 0], f1_data[trials_reshape[i, :]-1], s=80, color=colors_animals[count_a])
        trials_ave[i] = np.nanmedian(f1_data[trials_reshape[i, :]-1])
    ax.plot(trials_reshape[:, 0], trials_ave, color=colors_animals[count_a], linewidth=3)
ax.set_xticks(trials_reshape[:, 0])
ax.set_xticklabels(['0.175', '0.275', '0.375', 'split ipsi\nfast', 'split contra\nfast'], fontsize=14)
ax.set_title('Swing F1 score ' + condition, fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'f1_score_sw_' + condition), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
for count_a, animal in enumerate(animals):
    accuracy_data = np.array(benchmark_accuracy.loc[benchmark_accuracy['animal'] == animal, 'accuracy_st'])
    trials_ave = np.zeros(len(trials_reshape))
    for i in range(len(trials_reshape)):
        ax.scatter(np.ones(2)*trials_reshape[i, 0], accuracy_data[trials_reshape[i, :]-1], s=80, color=colors_animals[count_a])
        trials_ave[i] = np.nanmedian(accuracy_data[trials_reshape[i, :]-1])
    ax.plot(trials_reshape[:, 0], trials_ave, color=colors_animals[count_a], linewidth=2)
ax.set_xticks(trials_reshape[:, 0])
ax.set_xticklabels(['0.175', '0.275', '0.375', 'split ipsi\nfast', 'split contra\nfast'], fontsize=14)
ax.set_title('Stance accuracy score ' + condition, fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'accuracy_st_' + condition), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
for count_a, animal in enumerate(animals):
    accuracy_data = np.array(benchmark_accuracy.loc[benchmark_accuracy['animal'] == animal, 'accuracy_sw'])
    trials_ave = np.zeros(len(trials_reshape))
    for i in range(len(trials_reshape)):
        ax.scatter(np.ones(2)*trials_reshape[i, 0], accuracy_data[trials_reshape[i, :]-1], s=80, color=colors_animals[count_a])
        trials_ave[i] = np.nanmedian(accuracy_data[trials_reshape[i, :]-1])
    ax.plot(trials_reshape[:, 0], trials_ave, color=colors_animals[count_a], linewidth=2)
ax.set_xticks(trials_reshape[:, 0])
ax.set_xticklabels(['0.175', '0.275', '0.375', 'split ipsi\nfast', 'split contra\nfast'], fontsize=14)
ax.set_title('Swing accuracy score ' + condition, fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'accuracy_sw_' + condition), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
for count_a, animal in enumerate(animals):
    fn_data = np.array(benchmark_accuracy.loc[benchmark_accuracy['animal'] == animal, 'fn_st'])
    trials_ave = np.zeros(len(trials_reshape))
    for i in range(len(trials_reshape)):
        ax.scatter(np.ones(2)*trials_reshape[i, 0], fn_data[trials_reshape[i, :]-1], s=80, color=colors_animals[count_a])
        trials_ave[i] = np.nanmedian(fn_data[trials_reshape[i, :]-1])
    ax.plot(trials_reshape[:, 0], trials_ave, color=colors_animals[count_a], linewidth=2)
ax.set_xticks(trials_reshape[:, 0])
ax.set_xticklabels(['0.175', '0.275', '0.375', 'split ipsi\nfast', 'split contra\nfast'], fontsize=14)
ax.set_title('Stance false negatives ' + condition, fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'fn_st_' + condition), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
for count_a, animal in enumerate(animals):
    fn_data = np.array(benchmark_accuracy.loc[benchmark_accuracy['animal'] == animal, 'fn_sw'])
    trials_ave = np.zeros(len(trials_reshape))
    for i in range(len(trials_reshape)):
        ax.scatter(np.ones(2)*trials_reshape[i, 0], fn_data[trials_reshape[i, :]-1], s=80, color=colors_animals[count_a])
        trials_ave[i] = np.nanmedian(fn_data[trials_reshape[i, :]-1])
    ax.plot(trials_reshape[:, 0], trials_ave, color=colors_animals[count_a], linewidth=2)
ax.set_xticks(trials_reshape[:, 0])
ax.set_xticklabels(['0.175', '0.275', '0.375', 'split ipsi\nfast', 'split contra\nfast'], fontsize=14)
ax.set_title('Swing false negatives ' + condition, fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'fn_sw_' + condition), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
for count_a, animal in enumerate(animals):
    fp_data = np.array(benchmark_accuracy.loc[benchmark_accuracy['animal'] == animal, 'fp_st'])
    trials_ave = np.zeros(len(trials_reshape))
    for i in range(len(trials_reshape)):
        ax.scatter(np.ones(2)*trials_reshape[i, 0], fp_data[trials_reshape[i, :]-1], s=80, color=colors_animals[count_a])
        trials_ave[i] = np.nanmedian(fp_data[trials_reshape[i, :]-1])
    ax.plot(trials_reshape[:, 0], trials_ave, color=colors_animals[count_a], linewidth=2)
ax.set_xticks(trials_reshape[:, 0])
ax.set_xticklabels(['0.175', '0.275', '0.375', 'split ipsi\nfast', 'split contra\nfast'], fontsize=14)
ax.set_title('Stance false positives ' + condition, fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'fp_st_' + condition), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
for count_a, animal in enumerate(animals):
    fp_data = np.array(benchmark_accuracy.loc[benchmark_accuracy['animal'] == animal, 'fp_sw'])
    trials_ave = np.zeros(len(trials_reshape))
    for i in range(len(trials_reshape)):
        ax.scatter(np.ones(2)*trials_reshape[i, 0], fp_data[trials_reshape[i, :]-1], s=80, color=colors_animals[count_a])
        trials_ave[i] = np.nanmedian(fp_data[trials_reshape[i, :]-1])
    ax.plot(trials_reshape[:, 0], trials_ave, color=colors_animals[count_a], linewidth=2)
ax.set_xticks(trials_reshape[:, 0])
ax.set_xticklabels(['0.175', '0.275', '0.375', 'split ipsi\nfast', 'split contra\nfast'], fontsize=14)
ax.set_title('Swing false positives ' + condition, fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'fp_sw_' + condition), dpi=128)
plt.close('all')

# STIMULATION DURATION - VIOLIN
xaxis = np.array([0, 3, 6, 9, 12])
fig, ax = plt.subplots(tight_layout=True, figsize=(15,5))
for count_a in range(len(animals)):
    violin_parts = ax.violinplot(stim_duration_st[count_a], positions=xaxis+(0.5*count_a))
    for pc in violin_parts['bodies']:
        pc.set_color(colors_animals[count_a])
    violin_parts['cbars'].set_color(colors_animals[count_a])
    violin_parts['cmins'].set_color(colors_animals[count_a])
    violin_parts['cmaxes'].set_color(colors_animals[count_a])
ax.set_xticks(xaxis+0.125)
ax.set_xticklabels(['0.175', '0.275', '0.375', 'split ipsi\nfast', 'split contra\nfast'], fontsize=14)
ax.set_ylabel('Light on\nduration (s)', fontsize=14)
ax.set_title('Stance stimulus duration ' + condition, fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'stim_duration_st_' + condition), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(15,5))
for count_a in range(len(animals)):
    violin_parts = ax.violinplot(stim_duration_sw[count_a], positions=xaxis+(0.5*count_a))
    for pc in violin_parts['bodies']:
        pc.set_color(colors_animals[count_a])
    violin_parts['cbars'].set_color(colors_animals[count_a])
    violin_parts['cmins'].set_color(colors_animals[count_a])
    violin_parts['cmaxes'].set_color(colors_animals[count_a])
ax.set_xticks(xaxis+0.125)
ax.set_xticklabels(['0.175', '0.275', '0.375', 'split ipsi\nfast', 'split contra\nfast'], fontsize=14)
ax.set_ylim([0, 0.7])
ax.set_ylabel('Light on\nduration (s)', fontsize=14)
ax.set_title('Swing stimulus duration ' + condition, fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'stim_duration_sw_' + condition), dpi=128)
plt.close('all')

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
                                          16, os.path.join(path, 'plots'), '\\stance_time_hist_'+speeds[i], 1)
    otrack_class.plot_laser_presentation_phase_hist(light_onset_phase_sw_plot_animals, light_offset_phase_sw_plot_animals,
                                          16, os.path.join(path, 'plots'), '\\swing_time_hist_'+speeds[i], 1)
plt.close('all')

# STIMULATION ONSET AND OFFSET IN %STRIDE - HISTOGRAM WITH DURATION
for i in range(len(trials_reshape)):
    light_onset_phase_st_plot_animals = []
    light_offset_phase_st_plot_animals = []
    light_onset_phase_sw_plot_animals = []
    light_offset_phase_sw_plot_animals = []
    stim_nr_st_animals = []
    stride_nr_st_animals = []
    stim_nr_sw_animals = []
    stride_nr_sw_animals = []
    for count_a in range(len(animals)):
        light_onset_phase_st_plot_animals.extend(light_onset_phase_st[count_a][i])
        light_offset_phase_st_plot_animals.extend(light_offset_phase_st[count_a][i])
        light_onset_phase_sw_plot_animals.extend(light_onset_phase_sw[count_a][i])
        light_offset_phase_sw_plot_animals.extend(light_offset_phase_sw[count_a][i])
        stim_nr_st_animals.append(np.sum(stim_nr_st_all[count_a][i]))
        stride_nr_st_animals.append(np.sum(stride_nr_st_all[count_a][i]))
        stim_nr_sw_animals.append(np.sum(stim_nr_sw_all[count_a][i]))
        stride_nr_sw_animals.append(np.sum(stride_nr_sw_all[count_a][i]))
    otrack_class.plot_laser_presentation_phase_benchmark(light_onset_phase_st_plot_animals, light_offset_phase_st_plot_animals, 'stance',
                16, np.sum(stim_nr_st_animals), np.sum(stride_nr_st_animals), 'Greys',
                os.path.join(path, 'plots'), '\\stance_'+speeds[i])
    otrack_class.plot_laser_presentation_phase_benchmark(light_onset_phase_sw_plot_animals, light_offset_phase_sw_plot_animals, 'swing',
                16, np.sum(stim_nr_sw_animals), np.sum(stride_nr_sw_animals), 'Greys',
                os.path.join(path, 'plots'), '\\swing_'+speeds[i])
plt.close('all')
