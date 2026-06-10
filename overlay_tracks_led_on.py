from decord import VideoReader
from decord import cpu
import cv2
import scipy.signal as sp
import numpy as np

path = 'J:\\Thesis\\Presentation\\video real-time tracking\\'
filename = 'VIV40923_68_19_0.275_0.275_tied_1_3.mp4'
animal = filename.split('_')[0]
trial = np.int64(filename.split('_')[-1][:-4])
session = np.int64(filename.split('_')[-2])

#Load video
sr = 333
frame_width = 1088
frame_height = 420
out = cv2.VideoWriter(path + filename[:-4] + '_overlay_mp4codec_test2.mp4', cv2.VideoWriter_fourcc(*'MP4V'), sr, (frame_width, frame_height))
vidObj = VideoReader(path + filename, ctx=cpu(0))  # read the video
frames_total = len(vidObj)

#Load LED-on times
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
[st_led_on, sw_led_on, frame_counter_session] = otrack_class.load_benchmark_files(animal)
led_st = st_led_on.loc[st_led_on['trial'] == trial]
led_sw = sw_led_on.loc[sw_led_on['trial'] == trial]

#Load offline tracking
import locomotion_class
loco = locomotion_class.loco_class(path)
[final_tracks_trials, st_strides_trials, sw_strides_trials] = otrack_class.get_offtrack_paws(loco, animal, session)
FR_z = sp.medfilt(otrack_class.inpaint_nans(np.array(final_tracks_trials[0][3, 0, :])), 11)
FR_y = sp.medfilt(otrack_class.inpaint_nans(np.array(final_tracks_trials[0][2, 0, :])), 11)

#Get paw position for LED-on stance
frames_st = []
for count_i, i in enumerate(led_st['frames_on']):
    frames_st.extend(np.arange(led_st['frames_on'].iloc[count_i], led_st['frames_off'].iloc[count_i]+1))
frames_st_arr = np.array(frames_st)
#Get paw position for LED-on swing
frames_sw = []
for count_i, i in enumerate(led_sw['frames_on']):
    frames_sw.extend(np.arange(led_sw['frames_on'].iloc[count_i], led_sw['frames_off'].iloc[count_i]+1))
frames_sw_arr = np.array(frames_sw)

# frame_np[:60, 980:1050, :]
# frame_np[:60, 1050:, :]

for frameNr in range(frames_total):
    frame = vidObj[frameNr]
    frame_np = frame.asnumpy()
    cv2.circle(frame_np, (1009, 28), radius=15, color=(0, 165, 255), thickness=2) #left LED
    cv2.circle(frame_np, (1061, 30), radius=12, color=(0, 255, 0), thickness=2) #right LED
    if frameNr in frames_st_arr:
        cv2.circle(frame_np, (np.int64(FR_y[frameNr]), np.int64(FR_z[frameNr])), radius=3, color=(0, 165, 255), thickness=5)
        out.write(frame_np)
    elif frameNr in frames_sw_arr:
        cv2.circle(frame_np, (np.int64(FR_y[frameNr]), np.int64(FR_z[frameNr])), radius=3, color=(0, 255, 0), thickness=5)
        out.write(frame_np)
    else:
        out.write(frame_np)
out.release()

