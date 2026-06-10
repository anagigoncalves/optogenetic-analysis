import os
import numpy as np

paw_otrack = 'FR'
path = 'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\tied stance stim\\'
#path = 'D:\\AliG\\climbing-opto-treadmill\\Experiments ChR2 RT\\LOW expression\\ALL_ANIMALS\\tied th100sw IO 50ms\\'
#path = 'C:\\Users\\Utilizador\Carey Lab Dropbox\\Alice Geminiani\\LocoCF-internal\\Tests setup\\26052023 HR test\\25percent\\'
main_dir = path.split('\\')[:-2]
session = 1
plot_data = 0
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)
if not os.path.exists(os.path.join(path, 'processed files')):
    os.mkdir(os.path.join(path, 'processed files'))
animals = ['MC19107']

#['VIV42906', 'VIV42908', 'VIV42974', 'VIV42985','VIV42987','VIV44766', 'VIV45372']  # ChR2

#['MC16848', 'MC16851', 'MC17319', 'MC17665', 'MC17666', 'MC17669', 'MC17670', 'MC19082',
#'MC19124', 'MC19214', 'VIV41329', 'VIV41330', 'VIV42375', 'VIV42428', 'VIV42429', 'VIV42430', 'VIV42376', 'MC19107']   # jaws


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

    if sync_signal_session is not None:
        # READ ONLINE DLC TRACKS
        otracks = otrack_class.get_otrack_excursion_data(timestamps_session, animal)
        [otracks_st, otracks_sw] = otrack_class.get_otrack_event_data(timestamps_session, animal)

        # READ OFFLINE DLC TRACKS
        [offtracks_st, offtracks_sw] = otrack_class.get_offtrack_event_data(paw_otrack, loco, animal, np.int64(session_list[count_a]), timestamps_session, save_csv=True)

        ## READ OFFLINE PAW EXCURSIONS
        # For mouse MC16851, set threshold for stride exclusion at 10 instead of 20 (line 570 and 576 of locomotion_class.py)
        [final_tracks_trials, st_strides_trials, sw_strides_trials] = otrack_class.get_offtrack_paws(loco, animal, session)

        # PROCESS SYNCHRONIZER LASER SIGNALS
        if 'ChR2' in path:
            laser_on = otrack_class.get_laser_on_some_trials(animal, laser_trial_signal_session, timestamps_session, np.arange(9, 19))
        elif 'JAWS' in path:
            laser_on = otrack_class.get_laser_on(animal, laser_signal_session, timestamps_session)
        else:
            print('Check path. Cannot process laser signal.')

        # # GET LED INFORMATION
        # [st_led_on, sw_led_on] = otrack_class.get_led_information_trials(animal, timestamps_session, otracks_st, otracks_sw, corr_latency[count_a])

        # # OVERLAY WHEN LED SWING WAS ON
        # for t in trials:
        #     otrack_class.overlay_tracks_video(t, 'swing', final_tracks_trials, laser_on, st_led_on, sw_led_on)

    else:
        print('No synchronizer signal found for ' + animal + '. Skipping...')
        continue