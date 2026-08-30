
import os
"""
compare_kinematics.py
Author: [Your Name]
Date: 2024-06-08
Description:
    This script compares the average and standard deviation of kinematic trajectories for multiple animals
    across two experimental conditions. It loads trajectory data from pickle files for each animal in two
    specified folders, plots the mean trajectory with shaded regions representing the standard deviation,
    and visualizes the comparison using matplotlib.
Usage:
    - Update the 'path' variable to point to the root directory containing the animal data folders.
    - Ensure the folder names and file naming conventions match those expected by the script.
    - Run the script to generate a plot comparing the average and standard deviation of trajectories
      for each animal under both conditions.
Dependencies:
    - numpy
    - matplotlib
    - pickle
    - os
"""
import pickle
import numpy as np

import matplotlib.pyplot as plt

# List of animal names
path = 'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\tied stance stim retracked with finetuned DLC\\'

folders = [path+'st_centered_force_centerFalse_plot_off_to_onFalse_tied_trials_nonstim\\', path+'st_centered_force_centerFalse_plot_off_to_onFalse_tied_trials\\']  
colors = ['gray', 'red']  # Colors for each folder
animal_files = [f for f in os.listdir(folders[0]) if f.endswith('_avg_trajectory_data.pkl')]
animals = [fname.split('_')[0] for fname in animal_files]

plt.figure(figsize=(8, 6))

for animal, folder in zip(animals, folders):
    file_path = os.path.join(folder, f"{animal}_avg_trajectory_data.pkl")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        avg = np.array(data['avg_trajectory'])  
        std = np.array(data['std_trajectory'])
        x = np.array(data['avg_time'])
        plt.plot(x, avg, color=colors[folders.index(folder)])
        plt.fill_between(x, avg - std, avg + std, color=colors[folders.index(folder)], alpha=0.2)

plt.xlabel('Time')
plt.ylabel('Trajectory')
plt.title('Average and Std Trajectories')
plt.tight_layout()
plt.show()