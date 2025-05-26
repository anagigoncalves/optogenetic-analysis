import os
import numpy as np

paw_otrack = 'FR'
path = path_loco = 'D:\\AliG\\climbing-opto-treadmill\\Experiments\\Tied belt sessions\\20240126 tied stance stim\\'
main_dir = path.split('\\')[:-2]
session = 1
plot_data = 0
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)
if not os.path.exists(os.path.join(path, 'processed files')):
    os.mkdir(os.path.join(path, 'processed files'))
animals = ['VIV44766', 'VIV44771', 'VIV45372', 'VIV45373']
corr_latency = [0, 0, 0, 0]

animal_session_list = loco.animals_within_session()
animal_list = []
for a in range(len(animal_session_list)):
    animal_list.append(animal_session_list[a][0])
session_list = []
for a in range(len(animal_session_list)):
    session_list.append(animal_session_list[a][1])

for count_a, animal in enumerate(animals):
    print('Processing ' + animal)
    trials = otrack_class.get_trials(animal)
    # READ CAMERA TIMESTAMPS AND FRAME COUNTER
    [camera_timestamps_session, camera_frames_kept, camera_frame_counter_session] = otrack_class.get_session_metadata(animal, plot_data)
    # READ SYNCHRONIZER SIGNALS
    # If MC16851 need to uncomment/comment some lines inside function
    [timestamps_session, frame_counter_session, trial_signal_session, sync_signal_session, laser_signal_session, laser_trial_signal_session] = otrack_class.get_synchronizer_data(camera_frames_kept, animal, plot_data)

    # READ ONLINE DLC TRACKS
    otracks = otrack_class.get_otrack_excursion_data(timestamps_session, animal)
    [otracks_st, otracks_sw] = otrack_class.get_otrack_event_data(timestamps_session, animal)

    # READ OFFLINE DLC TRACKS
    [offtracks_st, offtracks_sw] = otrack_class.get_offtrack_event_data(paw_otrack, loco, animal, np.int64(session_list[count_a]), timestamps_session, save_csv=True)

    ## READ OFFLINE PAW EXCURSIONS
    [final_tracks_trials, st_strides_trials, sw_strides_trials] = otrack_class.get_offtrack_paws(loco, animal, session)

    # PROCESS SYNCHRONIZER LASER SIGNALS
    #if 'ChR2'
    laser_on = otrack_class.get_laser_on_some_trials(animal, laser_trial_signal_session, timestamps_session, np.arange(9, 19))
    #if JAWS
    #laser_on = otrack_class.get_laser_on(animal, laser_signal_session, timestamps_session)

    # # GET LED INFORMATION
    # [st_led_on, sw_led_on] = otrack_class.get_led_information_trials(animal, timestamps_session, otracks_st, otracks_sw, corr_latency[count_a])

    # # OVERLAY WHEN LED SWING WAS ON
    # for t in trials:
    #     otrack_class.overlay_tracks_video(t, 'swing', final_tracks_trials, laser_on, st_led_on, sw_led_on)
