#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 17:23:29 2020

@author: anagigoncalves
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob

class otrack_class:
    def __init__(self, path):
        self.path = path
        self.delim = self.path[-1]
        # self.pixel_to_mm = ?????
        self.sr = 333  # sampling rate of behavior camera for treadmill

    @staticmethod
    def converttime(time):
        # offset = time & 0xFFF
        cycle1 = (time >> 12) & 0x1FFF
        cycle2 = (time >> 25) & 0x7F
        seconds = cycle2 + cycle1 / 8000.
        return seconds

    @staticmethod
    def overlayDLCtracks(vidObj, otracks, frameNr, paw, frame_width, frame_height):
        if paw == 'FR':
            paw_color = (255, 0, 0)
        if paw == 'FL':
            paw_color = (0, 0, 255)
        vidObj.set(1, frameNr)
        cap, frame = vidObj.read()
        frame_otracks = cv2.circle(frame, (np.int64(otracks.iloc[frameNr, 2]), np.int64(otracks.iloc[frameNr, 3])),
                                   radius=5, color=paw_color, thickness=0)
        return frame_otracks

    @staticmethod
    def get_port_data(sync, port):
        sync_signal = sync.iloc[np.where(sync.iloc[:, 1] == 32)[0], :]
        sync_dev = []
        for i in range(len(sync_signal.iloc[:, 3])):
            bin_value = bin(sync_signal.iloc[i, 3])[2:]
            if bin_value == '1':
                sync_dev.append('00000000000001')
            else:
                sync_dev.append(bin_value)
        sync_dev_array = np.array(sync_dev)
        y = np.zeros(len(sync_dev_array))
        for i in range(len(sync_dev_array)-1):
            y[i] = int(sync_dev_array[i][-1 - port], 2)
        sync_signal_full = np.zeros(len(sync_signal) + 1)
        sync_signal_full[np.nonzero(np.diff(sync_signal.iloc[:, 2]))[0] + 1] = y[:-1]
        sync_timestamps = (sync_signal.iloc[:, 2] - sync_signal.iloc[0, 2]) * 1000  # in ms
        sync_timestamps_full = np.zeros(len(sync_timestamps) + 1)
        sync_timestamps_full[1:] = sync_timestamps
        sync_timestamps_full[0] = -0.1
        return sync_timestamps_full, sync_signal_full

    def get_session_metadata(self, plot_data):
        timestamps_session = []
        frame_counter_session = []
        frame_counter_0 = []
        timestamps_0 = []
        metadata_files = glob.glob(os.path.join(self.path,'*_meta.csv'))
        trial_order = []
        filelist = []
        for f in metadata_files:
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            filelist.append(f)
            trial_order.append(int(filename_split[7]))
        trial_ordered = np.sort(np.array(trial_order)) #reorder trials
        files_ordered = [] #order tif filenames by file order
        for f in range(len(filelist)):
            tr_ind = np.where(trial_ordered[f] == trial_order)[0][0]
            files_ordered.append(filelist[tr_ind])
        for trial, f in enumerate(files_ordered):
            metadata = pd.read_csv(os.path.join(self.path, f))
            cam_timestamps = [0]
            for t in np.arange(1, len(metadata.iloc[:,9])):
                cam_timestamps.append(self.converttime(metadata.iloc[t,9]-metadata.iloc[0,9]))
            print('FPS for camera acquisition ' + str(np.round(1/np.nanmedian(np.diff(cam_timestamps)), 2)))
            frame_counter = metadata.iloc[:,3]-metadata.iloc[0,3]
            frame_counter_0.append(metadata.iloc[0,3])
            timestamps_session.append(list(cam_timestamps))
            timestamps_0.append(metadata.iloc[0,9])
            frame_counter_session.append(list(frame_counter))
            if plot_data:
                plt.figure()
                plt.plot(list(cam_timestamps), metadata.iloc[:,3]-metadata.iloc[0,3])
                plt.title('Camera metadata for trial '+str(trial+1))
                plt.xlabel('Camera timestamps (s)')
                plt.ylabel('Frame counter')
        return timestamps_session, frame_counter_session, frame_counter_0, timestamps_0

    def get_synchronizer_data(self, plot_data):
        trial_signal_session = []
        sync_signal_session = []
        sync_files = glob.glob(os.path.join(self.path,'*_synch.csv'))
        trial_order = []
        filelist = []
        for f in sync_files:
            path_split = f.split(self.delim)
            filename_split = path_split[-2].split('_')
            filelist.append(f)
            trial_order.append(int(filename_split[7]))
        trial_ordered = np.sort(np.array(trial_order) ) #reorder trials
        files_ordered = [] #order tif filenames by file order
        for f in range(len(filelist)):
            tr_ind = np.where(trial_ordered[f] == trial_order)[0][0]
            files_ordered.append(filelist[tr_ind])
        for t, f in enumerate(files_ordered):
            sync_csv = pd.read_csv(os.path.join(self.path, f))
            [sync_timestamps_p0, sync_signal_p0] = self.get_port_data(sync_csv, 0)
            [sync_timestamps_p1, sync_signal_p1] = self.get_port_data(sync_csv, 1)
            trial_signal_session.append(np.concatenate((sync_timestamps_p0, sync_signal_p0)))
            sync_signal_session.append(np.concatenate((sync_timestamps_p1, sync_signal_p1)))
            if plot_data:
                plt.figure()
                plt.plot(sync_timestamps_p1, sync_signal_p1)
                plt.plot(sync_timestamps_p0, sync_signal_p0, linewidth=2)
                plt.title('Sync data for trial '+str(t+1))
                plt.xlabel('Time (ms)')
        return trial_signal_session, sync_signal_session

    def get_otrack_event_data(self, frame_counter_0_session, timestamps_0_session):
        sync_files = glob.glob(os.path.join(self.path,'*_otrack.csv'))
        trial_order = []
        filelist = []
        for f in sync_files:
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            filelist.append(f)
            trial_order.append(int(filename_split[7]))
        trial_ordered = np.sort(np.array(trial_order) ) #reorder trials
        files_ordered = [] #order tif filenames by file order
        for f in range(len(filelist)):
            tr_ind = np.where(trial_ordered[f] == trial_order)[0][0]
            files_ordered.append(filelist[tr_ind])
        otracks_st_time = []
        otracks_sw_time = []
        otracks_st_frames = []
        otracks_sw_frames = []
        otracks_st_trials = []
        otracks_sw_trials = []
        otracks_st_posx = []
        otracks_sw_posx = []
        otracks_st_posy = []
        otracks_sw_posy = []
        for trial, f in enumerate(files_ordered):
            otracks = pd.read_csv(os.path.join(self.path, f))
            stance_frames = np.where(otracks.iloc[:, 4]==True)[0]
            swing_frames = np.where(otracks.iloc[:, 4]==True)[0]
            otracks_frame_counter = []
            for i in range(len(otracks.iloc[:,1])):
                otracks_frame_counter.append((otracks.iloc[i,1]-frame_counter_0_session[trial]))
            otracks_timestamps = [0]
            for j in np.arange(1, len(otracks.iloc[:,0])):
                otracks_timestamps.append(self.converttime(np.array(otracks.iloc[j,0]-timestamps_0_session[trial])))
            otracks_st_time.extend(list(otracks_timestamps)[stance_frames])
            otracks_sw_time.extend(list(otracks_timestamps)[swing_frames])
            otracks_st_frames.extend(list(otracks_frame_counter)[stance_frames])
            otracks_sw_frames.extend(list(otracks_frame_counter)[swing_frames])
            otracks_st_trials.extend(list(np.ones(len(otracks_frame_counter))[stance_frames]*(trial+1)))
            otracks_sw_trials.extend(list(np.ones(len(otracks_frame_counter))[swing_frames]*(trial+1)))
            otracks_st_posx.extend(list(otracks.iloc[stance_frames, 2]))
            otracks_sw_posx.extend(list(otracks.iloc[swing_frames, 2]))
            otracks_st_posy.extend(list(otracks.iloc[stance_frames, 3]))
            otracks_sw_posy.extend(list(otracks.iloc[swing_frames, 3]))
        otracks_st = pd.DataFrame({'time': otracks_st_time, 'frames': otracks_st_frames, 'trial': otracks_st_trials,
            'x': otracks_st_posx, 'y': otracks_st_posy})
        print(otracks_st)
        otracks_sw = pd.DataFrame([otracks_sw_time.tolist(), otracks_sw_frames.tolist(), otracks_sw_trials.tolist(),
            otracks_sw_posx.tolist(), otracks_sw_posy.tolist()], columns=['time', 'frame', 'trial', 'x', 'y'])
        return otracks_st, otracks_sw

    def get_offtrack_event_data(self, paw, loco, animal, session):
        if paw == 'FR':
            p = 0
        if paw == 'FL':
            p = 2
        h5files = glob.glob(os.path.join(self.path, '*.h5'))
        filelist = []
        trial_order = []
        for f in h5files:
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            animal_name = filename_split[0][filename_split[0].find('M'):]
            session_nr = int(filename_split[6])
            if animal_name == animal and session_nr == session:
                filelist.append(path_split[-1])
                trial_order.append(int(filename_split[7][:-3]))
        trial_ordered = np.sort(np.array(trial_order))  # reorder trials
        files_ordered = []  # order tif filenames by file order
        for f in range(len(filelist)):
            tr_ind = np.where(trial_ordered[f] == trial_order)[0][0]
            files_ordered.append(filelist[tr_ind])
        offtracks_st_time = []
        offtracks_sw_time = []
        offtracks_st_frames = []
        offtracks_sw_frames = []
        offtracks_st_trials = []
        offtracks_sw_trials = []
        offtracks_st_posx = []
        offtracks_sw_posx = []
        offtracks_st_posy = []
        offtracks_sw_posy = []
        for f in files_ordered:
            path_split = f.split(self.delim)
            filename_split = path_split[-1].split('_')
            trial = int(filename_split[7][:-3])
            [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, 0.9, 0)
            [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
            # DO THIS AS A DATAFRAME
            offtracks_st_time.append(np.array(st_strides_mat[p][:, 0, 0] / 1000))
            offtracks_sw_time.append(np.array(sw_pts_mat[p][:, 0, 0] / 1000))
            offtracks_st_frames.append(np.array(st_strides_mat[p][:, 0, -1]))
            offtracks_sw_frames.append(np.array(sw_pts_mat[p][:, 0, -1]))
            offtracks_st_trials.append(np.ones(len(st_strides_mat[p][:, 0, 0])) * trial)
            offtracks_sw_trials.append(np.ones(len(sw_pts_mat[p][:, 0, -1])) * trial)
            offtracks_st_posx.append(final_tracks[0, p, np.int64(st_strides_mat[p][:, 0, -1])])
            offtracks_sw_posx.append(final_tracks[0, p, np.int64(sw_pts_mat[p][:, 0, -1])])
            offtracks_st_posy.append(final_tracks[1, p, np.int64(st_strides_mat[p][:, 0, -1])])
            offtracks_sw_posy.append(final_tracks[1, p, np.int64(sw_pts_mat[p][:, 0, -1])])
        offtracks_st = pd.DataFrame([offtracks_st_time.tolist(), offtracks_st_frames.tolist(), offtracks_st_trials.tolist(),
            offtracks_st_posx.tolist(), offtracks_st_posy.tolist()], columns=['time', 'frame', 'trial', 'x', 'y'])
        offtracks_sw = pd.DataFrame([offtracks_sw_time.tolist(), offtracks_sw_frames.tolist(), offtracks_sw_trials.tolist(),
            offtracks_sw_posx.tolist(), offtracks_sw_posy.tolist()], columns=['time', 'frame', 'trial', 'x', 'y'])
        return offtracks_st, offtracks_sw