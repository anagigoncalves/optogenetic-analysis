import os
import numpy as np
import matplotlib.pyplot as plt

path = path_loco = 'C:\\Users\\Ana\\Desktop\\fixed duration tests\\100ms\\'

import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
if not os.path.exists(os.path.join(path, 'processed files')):
    os.mkdir(os.path.join(path, 'processed files'))
animal = 'VIV42377'

trials = otrack_class.get_trials(animal)

# READ CAMERA TIMESTAMPS AND FRAME COUNTER
[camera_timestamps_session, camera_frames_kept, camera_frame_counter_session] = otrack_class.get_session_metadata(animal, 0)

# READ SYNCHRONIZER SIGNALS
[timestamps_session, frame_counter_session, trial_signal_session, sync_signal_session, laser_signal_session, laser_trial_signal_session] = otrack_class.get_synchronizer_data(camera_frames_kept, animal, 0)

trial_start = np.where(trial_signal_session['signal'] == 1)[0][0]
trial_end = np.where(trial_signal_session['signal'] == 1)[0][-1]
sync_sr = np.int64((trial_end-trial_start)/camera_timestamps_session[0][-1])

# CHECK DURATION LENGTH
sig = np.array(laser_trial_signal_session['signal'])
sig_diff = np.where(np.diff(sig >= 1))[0]

# PLOT FIXED DURATION CHANNEL
plt.figure()
plt.plot(laser_trial_signal_session['signal'])
plt.scatter(sig_diff, np.repeat(1, len(sig_diff)), color='black')

plt.figure()
plt.scatter(np.arange(len(sig_diff)-1), (np.diff(sig_diff)-1)/sync_sr)
plt.ylim([0, 0.2])
