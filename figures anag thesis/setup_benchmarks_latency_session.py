# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:59:15 2023
@author: Ana
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

paw_otrack = 'FR'
path = path = 'C:\\Users\\Ana\\Desktop\\Data OPTO\\HR tests\\25percent\\'
animals = ['VIV40922', 'VIV40923', 'VIV40924']
colors_animals = ['crimson', 'purple', 'gray']
session = 1
condition = path.split('\\')[-2]
main_dir = path.split('\\')[:-2]
import online_tracking_class
otrack_class = online_tracking_class.otrack_class(path)
import locomotion_class
loco = locomotion_class.loco_class(path)
if not os.path.exists(os.path.join(path, 'processed files')):
    os.mkdir(os.path.join(path, 'processed files'))
if not os.path.exists(os.path.join(path, 'plots')):
    os.mkdir(os.path.join(path, 'plots'))
th_latency_on_st = []
th_latency_off_st = []
animal_st_id = []
trial_st_id = []
condition_st_id = []
th_latency_on_sw = []
th_latency_off_sw = []
animal_sw_id = []
trial_sw_id = []
condition_sw_id = []
for animal in animals:
    print('Processing ' + animal)
    trials = otrack_class.get_trials(animal)
    # LOAD PROCESSED DATA
    [otracks, otracks_st, otracks_sw, offtracks_st, offtracks_sw, timestamps_session,
     laser_on] = otrack_class.load_processed_files(animal)
    # LOAD DATA FOR BENCHMARK ANALYSIS
    [st_led_on, sw_led_on, frame_counter_session] = otrack_class.load_benchmark_files(animal)
    # READ OFFLINE PAW EXCURSIONS
    final_tracks_trials = otrack_class.get_offtrack_paws(loco, animal, session)
    # GET LATENCY DATA STANCE
    [condition_id_st_singleanimal, trial_id_st_singleanimal, animal_id_st_singleanimal, th_latency_on_st_singleanimal,
     th_latency_off_st_singleanimal] = otrack_class.get_latency_data_led('stance',
        condition, animal, otracks, otracks_st, otracks_sw, st_led_on, sw_led_on, offtracks_st, offtracks_sw)
    th_latency_on_st.extend(th_latency_on_st_singleanimal)
    th_latency_off_st.extend(th_latency_off_st_singleanimal)
    animal_st_id.extend(animal_id_st_singleanimal)
    trial_st_id.extend(trial_id_st_singleanimal)
    condition_st_id.extend(condition_id_st_singleanimal)
    # GET LATENCY DATA SWING
    [condition_id_sw_singleanimal, trial_id_sw_singleanimal, animal_id_sw_singleanimal, th_latency_on_sw_singleanimal,
      th_latency_off_sw_singleanimal] = otrack_class.get_latency_data_led('swing',
         condition, animal, otracks, otracks_st, otracks_sw, st_led_on, sw_led_on, offtracks_st, offtracks_sw)
    # [condition_id_sw_singleanimal, trial_id_sw_singleanimal, animal_id_sw_singleanimal, th_latency_on_sw_singleanimal,
    #  th_latency_off_sw_singleanimal] = otrack_class.get_latency_data_laser('swing', np.repeat(150, 10),
    #     condition, animal, otracks, otracks_st, otracks_sw, laser_on, offtracks_st, offtracks_sw)
    th_latency_on_sw.extend(th_latency_on_sw_singleanimal)
    th_latency_off_sw.extend(th_latency_off_sw_singleanimal)
    animal_sw_id.extend(animal_id_sw_singleanimal)
    trial_sw_id.extend(trial_id_sw_singleanimal)
    condition_sw_id.extend(condition_id_sw_singleanimal)
latency_data_st = pd.DataFrame(
    {'condition': condition_st_id, 'animal': animal_st_id, 'trial': trial_st_id, 'th_latency_on': th_latency_on_st,
     'th_latency_off': th_latency_off_st})
latency_data_st.to_csv(os.path.join(otrack_class.path, 'processed files', 'latency_data_st.csv'), sep=',', index=False)

latency_data_sw = pd.DataFrame(
    {'condition': condition_sw_id, 'animal': animal_sw_id, 'trial': trial_sw_id, 'th_latency_on': th_latency_on_sw,
     'th_latency_off': th_latency_off_sw})
latency_data_sw.to_csv(os.path.join(otrack_class.path, 'processed files', 'latency_data_sw.csv'), sep=',', index=False)

# Latency summary
trials_reshape = np.reshape(np.arange(1, 11), (5, 2))
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
for count_a, animal in enumerate(animals):
    latency_data_animal = latency_data_st.loc[latency_data_st['animal'] == animal]
    trials_ave = np.zeros(len(trials_reshape))
    for i in range(len(trials_reshape)):
        data_t1 = np.nanmedian(latency_data_animal.loc[latency_data_animal['trial']==trials_reshape[i, 0], 'th_latency_on'])
        data_t2 = np.nanmedian(
            latency_data_animal.loc[latency_data_animal['trial'] == trials_reshape[i, 1], 'th_latency_on'])
        ax.scatter(np.ones(2)*trials_reshape[i, 0], np.array([data_t1, data_t2])*1000, s=80, color=colors_animals[count_a])
ax.set_xticks(trials_reshape[:, 0])
ax.set_xticklabels(['0.175', '0.275', '0.375', 'split ipsi\nfast', 'split contra\nfast'], fontsize=14)
ax.set_ylabel('Time (ms)', fontsize=14)
ax.set_title('Stance latency ' + condition, fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'latency_st_' + condition), dpi=128)
fig, ax = plt.subplots(tight_layout=True, figsize=(7,5))
for count_a, animal in enumerate(animals):
    latency_data_animal = latency_data_sw.loc[latency_data_sw['animal'] == animal]
    trials_ave = np.zeros(len(trials_reshape))
    for i in range(len(trials_reshape)):
        data_t1 = np.nanmedian(latency_data_animal.loc[latency_data_animal['trial']==trials_reshape[i, 0], 'th_latency_on'])
        data_t2 = np.nanmedian(
            latency_data_animal.loc[latency_data_animal['trial'] == trials_reshape[i, 1], 'th_latency_on'])
        ax.scatter(np.ones(2)*trials_reshape[i, 0], np.array([data_t1, data_t2])*1000, s=80, color=colors_animals[count_a])
ax.set_xticks(trials_reshape[:, 0])
ax.set_xticklabels(['0.175', '0.275', '0.375', 'split ipsi\nfast', 'split contra\nfast'], fontsize=14)
ax.set_ylabel('Time (ms)', fontsize=14)
ax.set_title('Swing latency ' + condition, fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(os.path.join(path, 'plots', 'latency_sw_' + condition), dpi=128)