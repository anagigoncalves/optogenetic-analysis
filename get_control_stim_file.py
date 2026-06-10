import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = 'C:\\Users\\Ana\\Desktop\\test\\laser_on.csv'
laser_on_csv = pd.read_csv(filename)
plot_data = 0
trials = [2, 3]

#create boolean csv for each trial
for trial in trials:
    laser_on_csv_trial = laser_on_csv.loc[laser_on_csv['trial'] == trial]
    bool_vec = np.zeros(21000) #make it larger for the slightly different trial durations
    frames_on = []
    for r in range(len(laser_on_csv_trial)):
        frames_on.extend(np.arange(laser_on_csv_trial.iloc[r, 2], laser_on_csv_trial.iloc[r, 3]))
    bool_vec[np.array(frames_on)] = 1
    if plot_data:
        plt.figure()
        plt.plot(bool_vec)
    df_bool = pd.DataFrame(bool_vec)
    df_bool.to_csv(('\\').join(filename.split('\\')[:-1])+'\\laser_on_bonsai_trial'+str(trial)+'.csv', index=False)

