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
summary_path = 'C:\\Users\\Ana\\Documents\\PhD\\Projects\\Online Stimulation Treadmill\\Benchmark plots\\Examples\\'
path = 'C:\\Users\\Ana\\Documents\\PhD\\Projects\\Online Stimulation Treadmill\\Tests\\Tailbase tests\\50percent\\'
condition = path.split('\\')[-2]
network = path.split('\\')[-3]
session = 1
if not os.path.exists(os.path.join(path, 'plots')):
    os.mkdir(os.path.join(path, 'plots'))
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)

animal = 'MC18089'
trial = 1
trials = otrack_class.get_trials(animal)
# LOAD PROCESSED DATA
[otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session, laser_on] = otrack_class.load_processed_files(animal)
# LOAD DATA FOR BENCHMARK ANALYSIS
[st_led_on, sw_led_on, frame_counter_session] = otrack_class.load_benchmark_files(animal)
# READ OFFLINE PAW EXCURSIONS
final_tracks_trials = otrack_class.get_offtrack_paws(loco, animal, session)

# # LASER ACCURACY
# time_on = 11
# time_off = 12
# yaxis = np.array([-100, 200])
# offtrack_trial = offtracks_st.loc[offtracks_st['trial'] == trial]
# light_trial = laser_on.loc[laser_on['trial'] == trial]
# led_trials = np.transpose(np.array(laser_on.loc[laser_on['trial'] == trial]))
# offtrack_trial_otherperiod = offtracks_sw.loc[offtracks_sw['trial'] == trial]
# fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
# for r in range(np.shape(led_trials)[1]):
#     rectangle = plt.Rectangle((led_trials[0, r], yaxis[0]),
#                               led_trials[1, r] - led_trials[0, r], yaxis[1]-yaxis[0], fc='grey', alpha=0.3)
#     plt.gca().add_patch(rectangle)
# mean_excursion = np.nanmean(final_tracks_trials[trial - 1][0, 0, :])
# ax.plot(timestamps_session[trial - 1], final_tracks_trials[trial - 1][0, 0, :] - mean_excursion,
#         color='red', linewidth=2)
# ax.set_xlabel('Time (s)', fontsize=14)
# ax.set_ylabel('FR paw excursion', fontsize=14)
# ax.set_xlim([time_on, time_off])
# ax.set_ylim(yaxis)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# plt.savefig(os.path.join(summary_path, 'examples_GOOD_' + network + '_' + condition + '_st'), dpi=128)

# LED ACCURACY
# time_on = 11
# time_off = 12
yaxis = np.array([-100, 200])
offtrack_trial = offtracks_sw.loc[offtracks_sw['trial'] == trial]
light_trial = sw_led_on.loc[sw_led_on['trial'] == trial]
led_trials = np.transpose(np.array(sw_led_on.loc[sw_led_on['trial'] == trial].iloc[:, 2:4]))
fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
for r in range(np.shape(led_trials)[1]):
    rectangle = plt.Rectangle((timestamps_session[trial - 1][led_trials[0, r]], yaxis[0]),
                              timestamps_session[trial - 1][led_trials[1, r]] - timestamps_session[trial - 1][led_trials[0, r]], yaxis[1]-yaxis[0], fc='grey', alpha=0.3)
    plt.gca().add_patch(rectangle)
mean_excursion = np.nanmean(final_tracks_trials[trial - 1][0, 0, :])
ax.plot(timestamps_session[trial - 1], final_tracks_trials[trial - 1][0, 0, :] - mean_excursion,
        color='red', linewidth=2)
ax.scatter(offtrack_trial['time'], offtrack_trial['x'] - mean_excursion, s=20, color='black')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Time (s)', fontsize=14)
ax.set_ylabel('FR paw excursion', fontsize=14)
# ax.set_xlim([time_on, time_off])
# ax.set_ylim(yaxis)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# plt.savefig(os.path.join(summary_path, 'examples_GOOD_' + network + '_' + condition + '_sw'), dpi=128)

