# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:59:15 2023
@author: Ana
"""
import os
paw_otrack = 'FR'
path_main = 'C:\\Users\\Ana\\Documents\\PhD\\Projects\\Online Stimulation Treadmill\\Tests\\'
subdir = '040423 mobile network crop bottom tests\\'
path = os.path.join(path_main, subdir)
main_dir = path.split('\\')[:-2]
animal = 'MC16946'
session = 1
plot_data = 0
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)
trials = otrack_class.get_trials()
# READ CAMERA TIMESTAMPS AND FRAME COUNTER
[camera_timestamps_session, camera_frames_kept, camera_frame_counter_session] = otrack_class.get_session_metadata(plot_data)

# READ SYNCHRONIZER SIGNALS
[timestamps_session, frame_counter_session, trial_signal_session, sync_signal_session, laser_signal_session, laser_trial_signal_session] = otrack_class.get_synchronizer_data(camera_frames_kept, plot_data)

# READ ONLINE DLC TRACKS
otracks = otrack_class.get_otrack_excursion_data(timestamps_session)
[otracks_st, otracks_sw] = otrack_class.get_otrack_event_data(timestamps_session)

# READ OFFLINE DLC TRACKS
[offtracks_st, offtracks_sw] = otrack_class.get_offtrack_event_data(paw_otrack, loco, animal, session, timestamps_session)

# READ OFFLINE PAW EXCURSIONS
final_tracks_trials = otrack_class.get_offtrack_paws(loco, animal, session)

# LATENCY OF LIGHT IN RELATION TO OTRACK
[latency_light_st, latency_light_sw, st_led_on, sw_led_on] = otrack_class.get_led_information_trials(trials, timestamps_session, otracks_st, otracks_sw)

# PROCESS SYNCHRONIZER LASER SIGNALS
laser_on = otrack_class.get_laser_on(laser_signal_session, timestamps_session)

# # OVERLAY WHEN LED SWING WAS ON
# for t in trials:
#     otrack_class.overlay_tracks_video(t, 'swing', final_tracks_trials, laser_on, st_led_on, sw_led_on)
