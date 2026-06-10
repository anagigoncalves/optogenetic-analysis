import os
import numpy as np
import matplotlib.pyplot as plt

paw_colors = ['red', 'magenta', 'blue', 'cyan']
paw_otrack = 'FR'
path_main = 'C:\\Users\\alice\\Dropbox (Personal)\\CareyLab\\materialAnaG\\240323 test thresholds\\' #'C:\\Users\\Ana\\Documents\\PhD\\Projects\\Online Stimulation Treadmill\\learning and threshold tests 140323\\'
subdir = 'cm\\' #'thresholds test 140323 cm\\'
path = os.path.join(path_main, subdir)
main_dir = path.split('\\')[:-2]
animal = 'MC16946CM'      #'MC16946CM'
session = 1
plot_data = 0
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)
trials = otrack_class.get_trials()

# LOAD PROCESSED DATA
[otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, st_led_on, sw_led_on] = otrack_class.load_processed_files()

# READ CAMERA TIMESTAMPS AND FRAME COUNTER
[camera_timestamps_session, camera_frames_kept, camera_frame_counter_session] = otrack_class.get_session_metadata(plot_data)

# READ SYNCHRONIZER SIGNALS
[timestamps_session, frame_counter_session, trial_signal_session, sync_signal_session, laser_signal_session, laser_trial_signal_session] = otrack_class.get_synchronizer_data(camera_frames_kept, plot_data)

# READ OFFLINE PAW EXCURSIONS
final_tracks_trials = otrack_class.get_offtrack_paws(loco, animal, session)

# READ OFFLINE DLC TRACKS
[offtracks_st, offtracks_sw] = otrack_class.get_offtrack_event_data(paw_otrack, loco, animal, session, timestamps_session)

# PROCESS SYNCHRONIZER LASER SIGNALS
laser_on = otrack_class.get_laser_on(laser_signal_session, timestamps_session)

# LATENCY OF OTRACK IN RELATION TO OFFTRACK
[tracks_hits_st, tracks_hits_sw, otrack_st_hits, otrack_sw_hits] = otrack_class.get_hits_swst_online(trials, otracks_st, otracks_sw, offtracks_st, offtracks_sw)
latency_st = []
for trial in trials:
    # Select current trial 
    current_trial_st = tracks_hits_st.loc[tracks_hits_st['trial'] == trial]
    # Select start of stance period based on the 4 frames condition
    onset_ind = np.where(np.diff(current_trial_st['otrack_frames'])>4)[0]+1
    # Compute difference between online and offline start of stance
    diff_otrack_offtrack = np.array(current_trial_st['otrack_times'])[onset_ind] - np.array(current_trial_st['offtrack_times'])[onset_ind]
    latency_st.append(np.array(diff_otrack_offtrack)*1000)
latency_sw = []
for trial in trials:
    # Select current trial
    current_trial_sw = tracks_hits_sw.loc[tracks_hits_sw['trial'] == trial]
    # Select start of swing period based on the 4 frames condition
    onset_ind = np.where(np.diff(current_trial_sw['otrack_frames'])>4)[0]+1
    # Compute difference between online and offline start of swing
    diff_otrack_offtrack = np.array(current_trial_sw['otrack_times'])[onset_ind] - np.array(current_trial_sw['offtrack_times'])[onset_ind]
    latency_sw.append(np.array(diff_otrack_offtrack)*1000)
# latency plot
plt.rcParams['font.size'] = 18
fig, ax = plt.subplots(2, len(trials), figsize=(20, 20), tight_layout=True)
for count_t, trial in enumerate(trials):
    ax[0, count_t].hist(latency_st[count_t], bins=100,  range=(-10, 300), color='black')
    ax[0, count_t].set_title('stance latency trial '+str(trial))
    ax[0, count_t].spines['right'].set_visible(False)
    ax[0, count_t].spines['top'].set_visible(False)
for count_t, trial in enumerate(trials):
    ax[1, count_t].hist(latency_sw[count_t], bins=100,  range=(-10, 800), color='black')
    ax[1, count_t].set_title('swing latency trial '+str(trial))
    ax[1, count_t].spines['right'].set_visible(False)
    ax[1, count_t].spines['top'].set_visible(False)
if not os.path.exists(otrack_class.path + 'plots'):
    os.mkdir(otrack_class.path + 'plots')
plt.savefig(os.path.join(otrack_class.path, 'plots', 'latency_offtrack_otrack.png'))
# hits plot
fig, ax = plt.subplots(2, 1, tight_layout=True, sharey=True)
ax = ax.ravel()
for count_t, trial in enumerate(trials):
    ax[0].bar(trials, np.array(otrack_st_hits), color='green')
    ax[0].bar(trials, [np.shape(offtracks_st.loc[offtracks_st['trial']==i])[0] for i in trials]-np.array(otrack_st_hits), bottom = np.array(otrack_st_hits), color='red')
ax[0].set_title('stance misses')
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
for count_t, trial in enumerate(trials):
    ax[1].bar(trials, np.array(otrack_sw_hits), color='green')
    ax[1].bar(trials,
              [np.shape(offtracks_sw.loc[offtracks_sw['trial'] == i])[0] for i in trials] - np.array(otrack_sw_hits),
              bottom = np.array(otrack_sw_hits), color='red')
ax[1].set_title('swing misses')
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
if not os.path.exists(otrack_class.path + 'plots'):
    os.mkdir(otrack_class.path + 'plots')
plt.savefig(os.path.join(otrack_class.path, 'plots', 'otrack_misses.png'))
# frames poorly detected
[detected_frames_bad_st, detected_frames_bad_sw] = otrack_class.frames_outside_st_sw(trials, offtracks_st, offtracks_sw, otracks_st, otracks_sw)
fig, ax = plt.subplots(2, 1, tight_layout=True)
ax = ax.ravel()
for count_t, trial in enumerate(trials):
    ax[0].bar(trials, detected_frames_bad_st, color='black')
ax[0].set_title('# frames where stance was assigned wrong')
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
for count_t, trial in enumerate(trials):
    ax[1].bar(trials, detected_frames_bad_sw, color='black')
ax[1].set_title('# frames where stance was assigned wrong')
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
if not os.path.exists(otrack_class.path + 'plots'):
    os.mkdir(otrack_class.path + 'plots')
plt.savefig(os.path.join(otrack_class.path, 'plots', 'frames_bad_detected.png'))

# LATENCY OF LIGHT IN RELATION TO OTRACK
latency_light_st = np.load(os.path.join(otrack_class.path, 'processed files', 'latency_light_st.npy'), allow_pickle=True)
latency_light_sw = np.load(os.path.join(otrack_class.path, 'processed files', 'latency_light_sw.npy'), allow_pickle=True)
fig, ax = plt.subplots(2, len(trials), figsize=(20, 10), tight_layout=True)
for count_t, trial in enumerate(trials):
    ax[0, count_t].hist(latency_light_st[count_t], bins=100, range=(-10, 300), color='black')
    ax[0, count_t].set_title('led stance latency trial ' + str(trial))
    ax[0, count_t].spines['right'].set_visible(False)
    ax[0, count_t].spines['top'].set_visible(False)
    ax[1, count_t].hist(latency_light_sw[count_t], bins=100, range=(-10, 300), color='black')
    ax[1, count_t].spines['right'].set_visible(False)
    ax[1, count_t].spines['top'].set_visible(False)
    ax[1, count_t].set_title('led swing latency trial ' + str(trial))
if not os.path.exists(otrack_class.path + 'plots'):
    os.mkdir(otrack_class.path + 'plots')
plt.savefig(os.path.join(otrack_class.path, 'plots', 'latency_otrack_ledon.png'))

# OVERLAP OF SYNCH SIGNAL LASER WITH LED ON FROM VIDEO
trial = 3
event = 'swing'
[signal_time_diff_onset, signal_time_diff_offset] = otrack_class.plot_led_synchronizer_signals(trial, event, st_led_on, sw_led_on, laser_signal_session, timestamps_session, plot_data)
fig, ax = plt.subplots(2, 1, tight_layout=True)
ax = ax.ravel()
ax[0].scatter(signal_time_diff_onset[0, :], signal_time_diff_onset[1, :]*1000, color='black')
ax[0].set_ylabel('time difference in ms')
ax[0].set_title('onset time difference')
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[1].scatter(signal_time_diff_offset[0, :], signal_time_diff_offset[1, :]*1000, color='black')
ax[1].set_title('offset time difference')
ax[1].set_ylabel('time difference in ms')
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
if not os.path.exists(otrack_class.path + 'plots'):
    os.mkdir(otrack_class.path + 'plots')
plt.savefig(os.path.join(otrack_class.path, 'plots', 'time_diff_led_laser_synch_trial' + str(trial) + '.png'))

# PERIODS WITH LIGHT ON
trial = 2
event = 'stance'
online_bool = 0
st_th = np.array([200, 110, 75, 85, 50])
sw_th = np.array([75, 40, 60, 55, 50])
otrack_class.plot_led_on_paws(timestamps_session, st_led_on, sw_led_on, final_tracks_trials, otracks, st_th[trial-1], sw_th[trial-1], trial, event, online_bool)

# ACCURACY OF LIGHT ON
event = 'stance'
plot_data = 0
led_st_hits = np.zeros(len(trials))
led_st_incomplete = np.zeros(len(trials))
led_st_misses = np.zeros(len(trials))
for count_t, trial in enumerate(trials):
    [full_hits, incomplete_hits, misses] = otrack_class.get_hit_light(trial, event, offtracks_st, offtracks_sw, st_led_on, sw_led_on, final_tracks_trials, timestamps_session, plot_data)
    led_st_hits[count_t] = full_hits
    led_st_incomplete[count_t] = incomplete_hits
    led_st_misses[count_t] = misses
fig, ax = plt.subplots(tight_layout=True, figsize=(5,3))
for count_t, trial in enumerate(trials):
    ax.bar(trials, led_st_hits, color='green')
    ax.bar(trials, led_st_incomplete, bottom = led_st_hits, color='orange')
    ax.bar(trials, led_st_misses, bottom = led_st_hits + led_st_incomplete, color='red')
ax.set_title('stance led misses')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
if not os.path.exists(otrack_class.path + 'plots'):
    os.mkdir(otrack_class.path + 'plots')
plt.savefig(os.path.join(otrack_class.path, 'plots', 'light_on_accuracy_stance.png'))

# LASER LIGHT ON TRACKS
laser_event = 'stance'
laser_hits = np.zeros(len(trials))
laser_incomplete = np.zeros(len(trials))
laser_misses = np.zeros(len(trials))
for count_t, trial in enumerate(trials):
    [full_hits, incomplete_hits, misses] = otrack_class.get_hit_laser_synch(trial, laser_event, offtracks_st, offtracks_sw, laser_on, final_tracks_trials, timestamps_session, 1)
    laser_hits[count_t] = full_hits
    laser_incomplete[count_t] = incomplete_hits
    laser_misses[count_t] = misses
# plot summaries
fig, ax = plt.subplots(tight_layout=True, figsize=(5,3))
ax.bar(trials, laser_hits, color='green')
ax.bar(trials, laser_incomplete, bottom = laser_hits, color='orange')
ax.bar(trials, laser_misses, bottom = laser_hits + laser_incomplete, color='red')
ax.set_title(laser_event + ' misses')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
if not os.path.exists(otrack_class.path + 'plots'):
    os.mkdir(otrack_class.path + 'plots')
plt.savefig(os.path.join(otrack_class.path, 'plots', 'laser_on_accuracy_' + laser_event + '.png'))
fig, ax = plt.subplots(tight_layout=True, figsize=(5,3))
ax.plot(trials, (laser_hits/(laser_hits+laser_misses+laser_incomplete))*100, '-o', color='green')
ax.plot(trials, (laser_incomplete/(laser_hits+laser_misses+laser_incomplete))*100, '-o', color='orange')
ax.set_title(laser_event + ' accuracy')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
if not os.path.exists(otrack_class.path + 'plots'):
    os.mkdir(otrack_class.path + 'plots')
plt.savefig(os.path.join(otrack_class.path, 'plots', 'laser_on_accuracy_' + laser_event + '_summary.png'))

event='swing'
plot_data = 1
led_sw_hits = np.zeros(len(trials))
led_sw_incomplete = np.zeros(len(trials))
led_sw_misses = np.zeros(len(trials))
for count_t, trial in enumerate(trials):
    [full_hits, incomplete_hits, misses] = otrack_class.get_hit_light(trial, event, offtracks_st, offtracks_sw, st_led_on, sw_led_on, final_tracks_trials, timestamps_session, plot_data)
    led_sw_hits[count_t] = full_hits
    led_sw_incomplete[count_t] = incomplete_hits
    led_sw_misses[count_t] = misses
fig, ax = plt.subplots(tight_layout=True, figsize=(5,3))
for count_t, trial in enumerate(trials):
    ax.bar(trials, led_sw_hits, color='green')
    ax.bar(trials, led_sw_incomplete, bottom = led_sw_hits, color='orange')
    ax.bar(trials, led_sw_misses, bottom = led_sw_hits + led_sw_incomplete, color='red')
ax.set_title('swing led misses')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
if not os.path.exists(otrack_class.path + 'plots'):
    os.mkdir(otrack_class.path + 'plots')
plt.savefig(os.path.join(otrack_class.path, 'plots', 'light_on_accuracy_swing.png'))
fig, ax = plt.subplots(tight_layout=True, figsize=(5,3))
ax.plot(trials, (led_sw_hits/(led_sw_hits+led_sw_misses+led_sw_incomplete))*100, '-o', color='green')
ax.plot(trials, (led_sw_incomplete/(led_sw_hits+led_sw_misses+led_sw_incomplete))*100, '-o', color='orange')
ax.set_title('swing accuracy')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
if not os.path.exists(otrack_class.path + 'plots'):
    os.mkdir(otrack_class.path + 'plots')
plt.savefig(os.path.join(otrack_class.path, 'plots', 'light_on_accuracy_swing_summary.png'))

#ACCURACY OF OTRACK
th_st = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200])
th_sw = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40])
for count_t, trial in enumerate(trials):
    fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
    ax.plot(otracks.loc[otracks['trial']==trial, 'time'], otracks.loc[otracks['trial']==trial, 'x'], color='black')
    ax.scatter(otracks_st.loc[otracks_st['trial']==trial, 'time'], otracks_st.loc[otracks_st['trial']==trial, 'x'], color='blue')
    ax.scatter(otracks_sw.loc[otracks_sw['trial']==trial, 'time'], otracks_sw.loc[otracks_sw['trial']==trial, 'x'], color='red')
    ax.axhline(th_st[count_t], color='blue')
    ax.axhline(th_sw[count_t], color='red')
    ax.set_title('trial' + str(trial))
    ax.set_ylim([-100, 300])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if not os.path.exists(otrack_class.path + 'plots'):
        os.mkdir(otrack_class.path + 'plots')
    plt.savefig(os.path.join(otrack_class.path, 'plots', 'otrack_file_delays_trial' + str(trial) + '.png'))

#STIMULATION DURATION
import seaborn as sns
fig, axes = plt.subplots(2, 1, figsize=(20, 10), tight_layout=True)
sns.boxplot(y = (laser_on['time_off']-laser_on['time_on'])*1000, x = 'trial', data = laser_on, ax = axes[0], showfliers=False)
sns.boxplot(y = (sw_led_on['time_off']-sw_led_on['time_on'])*1000, x = 'trial', data = sw_led_on, ax = axes[1], showfliers=False)
if not os.path.exists(otrack_class.path + 'plots'):
    os.mkdir(otrack_class.path + 'plots')
plt.savefig(os.path.join(otrack_class.path, 'plots', 'stim_duration.png'))

