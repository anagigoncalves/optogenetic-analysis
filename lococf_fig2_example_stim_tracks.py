# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
np.warnings.filterwarnings('ignore')

path_save = 'J:\\LocoCF\\Fig2 - split swing stim jaws\\'
path = 'J:\\Opto JAWS Data\\split right fast swing stim\\'
condition = path.split('\\')[-2]
network = path.split('\\')[-3]
session = 1
if not os.path.exists(os.path.join(path, 'plots')):
    os.mkdir(os.path.join(path, 'plots'))
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)
paws = ['FR', 'HR', 'FL', 'HL']
paw_colors = ['#e52c27', '#ad4397', '#3854a4', '#6fccdf']

animal = 'MC17670'
trial = 9
trials = otrack_class.get_trials(animal)
# LOAD PROCESSED DATA
[otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, laser_on] = otrack_class.load_processed_files(animal)
# READ OFFLINE PAW EXCURSIONS
[final_tracks_trials, st_strides_trials, sw_strides_trials] = otrack_class.get_offtrack_paws(loco, animal, session)

# STIM ACCURACY - swing
time_on = 44.225
time_off = 44.9
offtrack_trial = offtracks_sw.loc[offtracks_sw['trial'] == trial]
led_trials = np.transpose(np.array(laser_on.loc[laser_on['trial'] == trial].iloc[:, 2:4]))
# tracks_norm_FR = otrack_class.inpaint_nans(final_tracks_trials[trial - 1][0, 0, :])-np.nanmean(otrack_class.inpaint_nans(final_tracks_trials[trial - 1][0, :4, :]), axis=0)
# tracks_norm_FL = otrack_class.inpaint_nans(final_tracks_trials[trial - 1][0, 2, :])-np.nanmean(otrack_class.inpaint_nans(final_tracks_trials[trial - 1][0, :4, :]), axis=0)
tracks_norm_FR = otrack_class.inpaint_nans(final_tracks_trials[trial - 1][0, 0, :])
tracks_norm_FL = otrack_class.inpaint_nans(final_tracks_trials[trial - 1][0, 2, :])
tracks_norm_HR = otrack_class.inpaint_nans(final_tracks_trials[trial - 1][0, 1, :])
tracks_norm_HL = otrack_class.inpaint_nans(final_tracks_trials[trial - 1][0, 3, :])
fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
for r in range(np.shape(led_trials)[1]):
    rectangle = plt.Rectangle((timestamps_session[trial - 1][led_trials[0, r]], 120),
                              timestamps_session[trial - 1][led_trials[1, r]] - timestamps_session[trial - 1][led_trials[0, r]], 100, fc='lightblue', alpha=0.3, zorder=0)
    plt.gca().add_patch(rectangle)
ax.plot(timestamps_session[trial - 1], tracks_norm_FR*loco.pixel_to_mm, color=paw_colors[0], linewidth=3)
ax.plot(timestamps_session[trial - 1], tracks_norm_FL*loco.pixel_to_mm, color=paw_colors[2], linewidth=3)
ax.plot(timestamps_session[trial - 1], tracks_norm_HR*loco.pixel_to_mm, color=paw_colors[1], linewidth=3)
ax.plot(timestamps_session[trial - 1], tracks_norm_HL*loco.pixel_to_mm, color=paw_colors[3], linewidth=3)
ax.set_xlabel('Time (s)', fontsize=20)
ax.set_ylabel('X position (mm)', fontsize=20)
ax.set_xticks([44.3, 44.55, 44.8])
ax.set_xlim([time_on, time_off])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.set_ylim([120, 220])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path_save, 'fig2_swing_stim_example_MC17670_trial9_44,225_44,9s.png'), dpi=256)
plt.savefig(os.path.join(path_save, 'fig2_swing_stim_example_MC17670_trial9_44,225_44,9s.svg'), dpi=256)