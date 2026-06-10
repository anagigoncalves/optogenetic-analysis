# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:30:33 2023

@author: Ana
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.warnings.filterwarnings('ignore')
summary_path = 'J:\\LocoCF\\timing test\\'
path = 'J:\\LocoCF\\timing test\\20240603 tied stance stim IOchr2 50ms\\'
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

animal = 'VIV44766'
trial = 9
trials = otrack_class.get_trials(animal)
# LOAD PROCESSED DATA
[otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, laser_on] = otrack_class.load_processed_files(animal)
# LOAD DATA FOR BENCHMARK ANALYSIS
#[st_led_on, sw_led_on, frame_counter_session] = otrack_class.load_benchmark_files(animal)
# READ OFFLINE PAW EXCURSIONS
[final_tracks_trials, st_strides_trials, sw_strides_trials] = otrack_class.get_offtrack_paws(loco, animal, session)

# LASER ACCURACY
time_on = 11
time_off = 15
yaxis = np.array([-100, 200])
offtrack_trial = offtracks_st.loc[offtracks_st['trial'] == trial]
light_trial = laser_on.loc[laser_on['trial'] == trial]
led_trials = np.transpose(np.array(laser_on.loc[laser_on['trial'] == trial]))
offtrack_trial_otherperiod = offtracks_sw.loc[offtracks_sw['trial'] == trial]
fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
for r in range(np.shape(led_trials)[1]):
    rectangle = plt.Rectangle((led_trials[0, r], yaxis[0]),
                              led_trials[1, r] - led_trials[0, r], yaxis[1]-yaxis[0], fc='grey', alpha=0.3)
    plt.gca().add_patch(rectangle)
mean_excursion = np.nanmean(final_tracks_trials[trial - 1][0, 0, :])
ax.plot(timestamps_session[trial - 1], final_tracks_trials[trial - 1][0, 0, :] - mean_excursion,
        color='red', linewidth=2)
ax.set_xlabel('Time (s)', fontsize=14)
ax.set_ylabel('FR paw excursion', fontsize=14)
ax.set_xlim([time_on, time_off])
ax.set_ylim(yaxis)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig(os.path.join(summary_path, 'examples_GOOD_' + network + '_' + condition + '_st'), dpi=128)

# # LED ACCURACY - stance
# time_on = 13.8
# time_off = 14.5
# yaxis = np.array([0, 270])
# offtrack_trial = offtracks_st.loc[offtracks_st['trial'] == trial]
# light_trial = st_led_on.loc[st_led_on['trial'] == trial]
# led_trials = np.transpose(np.array(st_led_on.loc[st_led_on['trial'] == trial].iloc[:, 2:4]))
# fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
# for r in range(np.shape(led_trials)[1]):
#     rectangle = plt.Rectangle((timestamps_session[trial - 1][led_trials[0, r]], yaxis[0]),
#                               timestamps_session[trial - 1][led_trials[1, r]] - timestamps_session[trial - 1][led_trials[0, r]], yaxis[1]-yaxis[0], fc='lightblue', alpha=0.3)
#     plt.gca().add_patch(rectangle)
# mean_excursion = np.nanmean(final_tracks_trials[trial - 1][0, 0, :])
# for p in range(4):
#     ax.plot(timestamps_session[trial - 1], otrack_class.inpaint_nans(final_tracks_trials[trial - 1][0, p, :])*loco.pixel_to_mm,
#         color=paw_colors[p], linewidth=3)
# ax.plot(otracks.loc[otracks['trial']==trial, 'time'], otracks.loc[otracks['trial']==trial, 'x']*loco.pixel_to_mm,
#         color='black', linewidth=4)
# ax.axhline(y=200*loco.pixel_to_mm, color='black', linestyle='dashed')
# #ax.scatter(offtrack_trial['time'], offtrack_trial['x'] - mean_excursion, s=20, color='black')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Time (s)', fontsize=20)
# ax.set_ylabel('Paw position (mm)', fontsize=20)
# ax.set_xlim([time_on, time_off])
# # ax.set_ylim(yaxis)
# ax.tick_params(axis='both', which='major', labelsize=20)
#plt.savefig('J:\\Thesis\\figuresChapter3\\fig3_1_example_' + animal + '_' + str(trial) + '_' + network.replace(' ', '_') + '_' + condition + '_st_LED', dpi=256)
#plt.savefig('J:\\Thesis\\figuresChapter3\\fig3_1_example_' + animal + '_' + str(trial) + '_' + network.replace(' ', '_') + '_' + condition + '_st_LED.svg', dpi=256)
# plt.savefig(os.path.join(summary_path, 'examples_GOOD_' + network + '_' + condition + '_st'), dpi=128)


# LED ACCURACY - swing
# time_on = 14.5
# time_off = 15.3
# time_on = 0
# time_off = 60
# yaxis = np.array([-100, 200])
# offtrack_trial = offtracks_sw.loc[offtracks_sw['trial'] == trial]
# light_trial = sw_led_on.loc[sw_led_on['trial'] == trial]
# led_trials = np.transpose(np.array(sw_led_on.loc[sw_led_on['trial'] == trial].iloc[:, 2:4]))
# fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
# for r in range(np.shape(led_trials)[1]):
#     rectangle = plt.Rectangle((timestamps_session[trial - 1][led_trials[0, r]], yaxis[0]),
#                               timestamps_session[trial - 1][led_trials[1, r]] - timestamps_session[trial - 1][led_trials[0, r]], yaxis[1]-yaxis[0], fc='grey', alpha=0.3)
#     plt.gca().add_patch(rectangle)
# mean_excursion = np.nanmean(final_tracks_trials[trial - 1][0, 0, :])
# ax.plot(timestamps_session[trial - 1], final_tracks_trials[trial - 1][0, 0, :] - mean_excursion,
#         color='red', linewidth=2)
# ax.scatter(offtrack_trial['time'], offtrack_trial['x'] - mean_excursion, s=20, color='black')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlabel('Time (s)', fontsize=14)
# ax.set_ylabel('FR paw excursion', fontsize=14)
# ax.set_xlim([time_on, time_off])
# ax.set_ylim(yaxis)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig(os.path.join(summary_path, 'examples_GOOD_' + network + '_' + condition + '_sw'), dpi=128)


