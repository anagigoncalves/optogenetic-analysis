import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from decord import VideoReader
from decord import cpu

paw_colors = ['red', 'magenta', 'blue', 'cyan']
paw_otrack = 'FR'
path_main = 'C:\\Users\\alice\\Dropbox (Personal)\\CareyLab\\materialAnaG\\test sync pulse 200323\\' #'C:\\Users\\Ana\\Documents\\PhD\\Projects\\Online Stimulation Treadmill\\learning and threshold tests 140323\\'
subdir = '' #'thresholds test 140323 cm\\'
path = os.path.join(path_main, subdir)
main_dir = path.split('\\')[:-2]
animal = 'MC16947CM'      #'MC16946CM'
session = 1
plot_data = 0
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)
trials = otrack_class.get_trials()

# LOAD PROCESSED DATA
[otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, st_led_on, sw_led_on] = otrack_class.load_processed_files()

# READ SYNCHRONIZER SIGNALS
[timestamps_session, frame_counter_session, trial_signal_session, sync_signal_session, laser_signal_session, laser_trial_signal_session] = otrack_class.get_synchronizer_data(camera_frames_kept, plot_data)

# READ OFFLINE PAW EXCURSIONS
final_tracks_trials = otrack_class.get_offtrack_paws(loco, animal, session)

# LATENCY OF OTRACK IN RELATION TO OFFTRACK
[tracks_hits_st, tracks_hits_sw, otrack_st_hits, otrack_sw_hits] = otrack_class.get_hits_swst_online(trials, otracks_st, otracks_sw, offtracks_st, offtracks_sw)
latency_st = []
for trial in trials:
    diff_otrack_offtrack = tracks_hits_st.loc[tracks_hits_st['trial'] == trial, 'otrack_times']-tracks_hits_st.loc[tracks_hits_st['trial'] == trial, 'offtrack_times']
    diff_otrack_offtrack_peak_idx = find_peaks(-diff_otrack_offtrack)[0]
    latency_st.append(np.array(diff_otrack_offtrack)[diff_otrack_offtrack_peak_idx[0]]*1000)
latency_sw = []
for trial in trials:
    diff_otrack_offtrack = tracks_hits_sw.loc[tracks_hits_sw['trial'] == trial, 'otrack_times']-tracks_hits_sw.loc[tracks_hits_sw['trial'] == trial, 'offtrack_times']
    diff_otrack_offtrack_peak_idx = find_peaks(-diff_otrack_offtrack)[0]
    latency_sw.append(np.array(diff_otrack_offtrack)[diff_otrack_offtrack_peak_idx[0]]*1000)
# latency plot
fig, ax = plt.subplots(2, len(trials), figsize=(20, 20), tight_layout=True)
for count_t, trial in enumerate(trials):
    ax[0, count_t].hist(latency_st[count_t], bins=100, color='black')
    ax[0, count_t].set_title('stance latency trial '+str(trial))
    ax[0, count_t].spines['right'].set_visible(False)
    ax[0, count_t].spines['top'].set_visible(False)
for count_t, trial in enumerate(trials):
    ax[1, count_t].hist(latency_sw[count_t], bins=100, color='black')
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
plt.savefig(os.path.join(otrack_class.path, 'plots', 'misses.png'))
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
fig, ax = plt.subplots(2, len(trials), figsize=(20, 20), tight_layout=True)
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
trial = 1
#stance
st_led_trials = np.transpose(np.array(st_led_on.loc[st_led_on['trial'] == trial].iloc[:, 2:4]))
plt.figure()
for r in range(np.shape(st_led_trials)[1]):
    rectangle = plt.Rectangle((timestamps_session[trial-1][st_led_trials[0, r]], 0), timestamps_session[trial-1][st_led_trials[1, r]]-timestamps_session[trial-1][st_led_trials[0, r]], 1, fc='grey', alpha=0.3)
    plt.gca().add_patch(rectangle)
plt.plot(laser_signal_session.loc[laser_signal_session['trial']==trial, 'time'], laser_signal_session.loc[laser_signal_session['trial']==trial, 'signal'], color='black')
#swing
sw_led_trials = np.transpose(np.array(sw_led_on.loc[sw_led_on['trial'] == trial].iloc[:, 2:4]))
plt.figure()
for r in range(np.shape(sw_led_trials)[1]):
    rectangle = plt.Rectangle((timestamps_session[trial-1][sw_led_trials[0, r]], 0), timestamps_session[trial-1][sw_led_trials[1, r]]-timestamps_session[trial-1][sw_led_trials[0, r]], 1, fc='grey', alpha=0.3)
    plt.gca().add_patch(rectangle)
plt.plot(laser_signal_session.loc[laser_signal_session['trial']==trial, 'time'], laser_signal_session.loc[laser_signal_session['trial']==trial, 'signal'], color='black')

# PERIODS WITH LIGHT ON
trial = 3
event = 'stance'
online_bool = 0
st_th = np.array([100, 110, 75, 85, 50])
sw_th = np.array([50, 40, 60, 55, 50])
otrack_class.plot_led_on_paws_frames(timestamps_session, st_led_on, sw_led_on, final_tracks_trials, otracks, st_th[trial-1], sw_th[trial-1], trial, event, online_bool)