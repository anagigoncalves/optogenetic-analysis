"""
Video Gold Standard Annotation Tool
====================================

A GUI application for manually annotating video frames to create gold standard 
ground truth data for tracking validation.

Purpose:
--------
This tool allows users to manually select and mark specific anatomical points 
(e.g., paw positions, stance/swing onsets) frame-by-frame in video recordings.
The annotated coordinates serve as ground truth for validating automated 
tracking algorithms.

Features:
---------
- Load video files (MP4, AVI, MOV formats)
- Navigate through frames using arrow keys (Left/Right or A/D)
- Select point type from predefined options:
    * FRbottom, FLbottom: Front right/left paw bottom positions
    * FRz, FLz: Front right/left Z coordinates
    * FRbottom_validation: Validation points for front right paw
    * STonset: Stance onset frames
    * SWonset: Swing onset frames
- Click on video frame to mark the selected point type
- Points are saved as NumPy arrays (.npy) with shape (n_frames, 2)

Usage:
------
1. Run the script to open the GUI
2. Select the point type you want to annotate from the dropdown
3. Click "Load Video" to open a video file
4. Navigate frames with Left/Right arrow keys (or A/D)
5. Click on the video to mark the current point location
6. Press 'Q' or 'Esc' to save and close the video
7. Output file is saved as: <video_name>_<point_type>_points.npy

Controls:
---------
- Left Arrow / A: Previous frame
- Right Arrow / D: Next frame
- Left Click: Mark point at cursor position
- Q / Esc: Save points and close video

Dependencies:
-------------
- cv2 (OpenCV)
- tkinter
- numpy
- keyboard

Author: Alice Geminiani, Copilot ChatGPT-5
Date: 2024
"""

import cv2
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import os
import sys
import keyboard  # Add keyboard library

# Initialize variables
current_frame = None
frame_count = 0
selected_points = None
total_frames = 0
video_name = ""
point_type = None

def select_point(event, x, y, flags, param):
    global selected_points, frame_count
    if event == cv2.EVENT_LBUTTONDOWN and point_type.get():
        selected_points[frame_count-1] = [x, y]
        print(f"Frame {frame_count}, {point_type.get()} Point: {x}, {y}")
        
        # Redraw frame with new point immediately
        display_frame = current_frame.copy()
        point = tuple([x, y])
        cv2.circle(display_frame, point, 3, (255, 0, 0), -1)
        cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames} - {point_type.get()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Video", display_frame)

def load_video():
    global current_frame, frame_count, selected_points, total_frames, video_name
    path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    if not path:
        return

    # Get video directory and name
    video_dir = os.path.dirname(path)
    video_name = os.path.splitext(os.path.basename(path))[0]

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # Get total frames and initialize NaN array
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_points = np.full((total_frames, 2), np.nan)
    all_frames = []  # Store all frames for navigation
    
    # Read all frames first
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    
    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", select_point)
    frame_count = 0

    def show_frame(idx):
        global current_frame, frame_count
        if 0 <= idx < total_frames:
            current_frame = all_frames[idx].copy()
            frame_count = idx + 1
            display_frame = current_frame.copy()
            
            if not np.isnan(selected_points[idx]).any():
                point = tuple(selected_points[idx].astype(int))
                cv2.circle(display_frame, point, 3, (255, 0, 0), -1)
                
            cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames} - {point_type.get()}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Video", display_frame)

    current_idx = 0
    show_frame(current_idx)

    while True:
        key = cv2.waitKey(1)  # Changed to non-blocking wait
        
        if keyboard.is_pressed('left') or keyboard.is_pressed('a'):
            current_idx = max(0, current_idx - 1)
            show_frame(current_idx)
            while keyboard.is_pressed('left') or keyboard.is_pressed('a'):
                pass
        elif keyboard.is_pressed('right') or keyboard.is_pressed('d'):
            current_idx = min(total_frames - 1, current_idx + 1)
            show_frame(current_idx)
            while keyboard.is_pressed('right') or keyboard.is_pressed('d'):
                pass
        elif keyboard.is_pressed('q') or keyboard.is_pressed('esc'):  # Added esc as alternative
            if point_type.get():
                output_file = os.path.join(video_dir, f'{video_name}_{point_type.get()}_points.npy')
                np.save(output_file, selected_points)
                print(f"Points saved to {output_file}")
            cap.release()
            cv2.destroyAllWindows()
            return  # Return to main window instead of break

def exit_application():
    try:
        root.destroy()
        cv2.destroyAllWindows()
    finally:
        os._exit(0)  # Force exit the process

# Create GUI
root = tk.Tk()
root.title("Video Frame Selector")

def on_closing():
    if tk.messagebox.askokcancel("Quit", "Do you want to quit?"):
        exit_application()
        
root.protocol("WM_DELETE_WINDOW", on_closing)

# Add point type selection
point_type = tk.StringVar()
point_label = tk.Label(root, text="Select Point Type:")
point_label.pack(pady=5)

point_combo = ttk.Combobox(root, textvariable=point_type, 
                          values=["FRbottom", "FLbottom", "FRz", "FLz", "FRbottom_validation", "STonset", "SWonset"],
                          state="readonly")
point_combo.pack(pady=5)

load_button = tk.Button(root, text="Load Video", command=load_video)
load_button.pack(pady=20)

exit_button = tk.Button(root, text="Exit", command=exit_application)
exit_button.pack(pady=20)

root.mainloop()