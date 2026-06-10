"""
Created on Wed Feb 12 09:16:35 2025

@author: Alice Geminiani
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import os
import scipy.stats as st
import colorsys
import matplotlib.colors as mc
import utils

# Plotting functions
# Locomotor adaptation
# Baselines

FIGSIZE = (7, 8)
LABEL_SIZE = 24
TICK_SIZE = 20
laser_color = 'lightblue'

def lighten_color(color, amount=0.5):
    """Return a lightened version of *color* (amount: 0=unchanged, 1=white)."""
    try:
        c = mc.cnames[color]
    except (KeyError, TypeError):
        c = color
    h, l, s = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(h, 1 - amount * (1 - l), s)


def darken_color(color, amount=0.5):
    """Return a darkened version of *color* (amount: 0=unchanged, 1=black)."""
    try:
        c = mc.cnames[color]
    except (KeyError, TypeError):
        c = color
    h, l, s = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(h, (1 - amount) * l, s)


def get_latinj_point_colors(path_animals, n_points, base_color, left_animals, right_animals):
    colors_arr = []
    for ai in range(n_points):
        animal_name = path_animals[ai] if ai < len(path_animals) else None
        if animal_name in left_animals:
            colors_arr.append(darken_color(base_color, 0.4))
        elif animal_name in right_animals:
            colors_arr.append(lighten_color(base_color, 0.55))
        else:
            colors_arr.append(base_color)
    return colors_arr

# STANCE PHASE
def plot_phase(phase_data, animals, paw_colors, intervals=None):
    ntrial = np.shape(phase_data)[2]
    # plot stance phase - polar - group mean
    #error bars in polar plot don't rotate well
    fig_group_mean = plt.figure(figsize=FIGSIZE, tight_layout=True)
    ax = fig_group_mean.add_subplot(111, projection='polar')
    for paw in range(3):
        data_mean = st.circmean(phase_data[paw, :, :], axis=0)
        ax.scatter(data_mean, np.arange(1, ntrial + 1), c=paw_colors[paw], s=30)
    ax.set_yticks([8.5, 16.5])
    ax.set_yticklabels(['', ''])
    ax.tick_params(axis='both', which='major', labelsize=20)

    # plot stance phase - polar - individual animals, for each paw



    # plot stance phase - linear - individual animals
    num_animals=np.shape(phase_data)[1]
    fig_ind_animals, ax = plt.subplots(2, int(np.ceil(num_animals/2)), figsize=(20, 20), tight_layout=True, sharey=True, sharex=True)
    ax = ax.ravel()
    for count_a in range(np.shape(phase_data)[1]):
        for p in range(4):
            if intervals:
                if 'split' in intervals.keys():
                    add_patch_interval(ax[count_a], intervals['split'], set_fc='lightgray')
                if 'stim' in intervals.keys():
                    add_patch_interval(ax[count_a], intervals['stim'], set_fc=laser_color)
                    add_start_end_interval(ax[count_a], intervals['stim'])
            ax[count_a].plot(np.arange(1, ntrial+1), np.rad2deg(phase_data[p, count_a, :]), color=paw_colors[p], linewidth=2)
            ax[count_a].spines['right'].set_visible(False)
            ax[count_a].spines['top'].set_visible(False)
            ax[count_a].tick_params(axis='x')
           # ax[count_a].set_ylim([70, 230])
            ax[count_a].tick_params(axis='y')
            ax[count_a].set_title(animals[count_a])

    # plot stance phase - linear - group mean
    fig_group_mean_linear, ax_group_mean_linear = plt.subplots(figsize=(10, 10), tight_layout=True)
    for paw in range(3):
        # # Convert data to % and remove discontinuities
        phase_data_perc = (phase_data/(2*np.pi))*100
        #phase_data_perc[phase_data_perc < -30] += 100
        #phase_data_perc[phase_data_perc > 86.111] -= 100
        data_mean = st.circmean(phase_data_perc[paw, :, :], axis=0)
        ax_group_mean_linear.plot(np.arange(1, ntrial + 1), data_mean, color=paw_colors[paw], linewidth=2)
        #ax_group_mean_linear.fill_between(np.linspace(1, ntrial, ntrial), 
        #            st.circmean(phase_data_perc[paw, :, :], axis=0)+np.nanstd(phase_data_perc[paw, :, :], axis=0)/np.sqrt(len(animals)), 
        #            st.circmean(phase_data_perc[paw, :, :], axis=0)-np.nanstd(phase_data_perc[paw, :, :], axis=0)/np.sqrt(len(animals)), 
        #            facecolor=paw_colors[paw], alpha=0.5)

    return fig_group_mean, fig_ind_animals, fig_group_mean_linear


# STANCE SPEED
def plot_stance_speed(data, animal, paw_colors, intervals=None):
    fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)
   
    for p in range(4):
        ax.plot(np.linspace(1,len(data[p,:]),len(data[p,:])), data[p,:], color = paw_colors[p], linewidth = 2)
    # Add split and stimulation intervals
    if intervals:
        if 'split' in intervals.keys():
            add_patch_interval(ax, intervals['split'], set_fc='lightgray')
        if 'stim' in intervals.keys():
            add_patch_interval(ax, intervals['stim'], set_fc=laser_color)
            add_start_end_interval(ax, intervals['stim'])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Trial', fontsize = LABEL_SIZE)
    ax.set_ylabel('Stance speed', fontsize = LABEL_SIZE)
    ax.tick_params(axis='x',labelsize = TICK_SIZE)
    ax.tick_params(axis='y',labelsize = TICK_SIZE)
    ax.set_title(animal,fontsize=LABEL_SIZE)

    return fig

# SYMMETRY LEARNING CURVES
# Individual animals
def plot_learning_curve_ind_animals(param_sym, current_param, labels_dic, animal_list, colors, intervals=None):
    """
    Plots the learning curve for individual animals, each one with the animal's color.
    Overlaps the start and end of split/stimulation intervals, if specified.
    
    Parameters:
    param_sym (numpy.ndarray): A 3D array containing the parameter values for each parameter, animal and trial.
    current_param (int): The index of the current parameter to plot.
    labels_dic (dict): Dictionary keys are the parameter names and values are the corresponding labels.
    animal_list (list of str): A list of animal identifiers.
    colors (dict): A dictionary mapping animal identifiers to colors.
    intervals (dict, optional): A dictionary containing 'split' and 'stim' intervals with their respective start and duration (in trials).

    Returns:
    fig (matplotlib.figure.Figure): The figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)


         
    # Add split and stimulation intervals
    if intervals:
        if 'split' in intervals.keys():
            add_patch_interval(ax, intervals['split'], set_fc='lightgray')
        if 'stim' in intervals.keys():
            add_patch_interval(ax, intervals['stim'], set_fc=laser_color)
            add_start_end_interval(ax, intervals['stim'])

    # Plot learning curves for each animal
    for a in range(np.shape(param_sym)[1]):  # Loop on all animals
        plt.plot(np.linspace(1, len(param_sym[current_param, a, :]), len(param_sym[current_param, a, :])), param_sym[current_param, a, :], color=colors[animal_list[a]],
                 label=animal_list[a], linewidth=2)
        
    set_symmetry_plot(ax, list(labels_dic.values())[current_param])

    return fig 


# Individual animals with average
def plot_learning_curve_ind_animals_avg(param_sym_avg, current_param, labels_dic, animal_list, included_animals, colors, experiment_name, intervals=None, ranges=[False, None]):
    '''
    Plot learning curve for individual animals with average.
    
    Parameters:
    param_sym_avg (numpy.ndarray): A 2D array containing the average parameter values for each animal and trial.
    current_param (int): The index of the current parameter to plot.
    labels_dic (dict): Dictionary keys are the parameter names and values are the corresponding labels.
    animal_list (list of str): A list of animal identifiers.
    included_animals (list): A list of two lists, the first containing the names/identifiers of the animals to include in the plot, and the second containing the corresponding indices within the all animals list.
    colors (list): A list of dictionaries, the first mapping animal identifiers to colors and the second mapping the experiments to colors.
    intervals (dict, optional): A dictionary containing 'split' and 'stim' intervals with their respective start and duration (in trials).
    experiment_name (str, optional): The name of the experiment for color mapping.
    ranges (list, optional): A list containing a boolean indicating whether to use uniform y-axis ranges and a dictionary containing y-axis ranges for each parameter.
    
    Returns:
    fig (matplotlib.figure.Figure): The figure object containing the plot.
    '''

    fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)

    # Add split and stimulation intervals
    if intervals:
        if 'split' in intervals.keys():
            add_patch_interval(ax, intervals['split'], set_fc='lightgray')
        if 'stim' in intervals.keys():
            add_patch_interval(ax, intervals['stim'], set_fc=laser_color)
            add_start_end_interval(ax, intervals['stim'])


    # Plot learning curves for each animal
    for a in range(np.shape(param_sym_avg)[0]):
        plt.plot(np.linspace(1, len(param_sym_avg[a, :]), len(param_sym_avg[a, :])), param_sym_avg[a, :], linewidth=1, color=colors[0][included_animals[0][a]], label=animal_list[included_animals[1][a]])
    ax.legend(frameon=False)
    
    # Plot average
    plt.plot(np.linspace(1, len(param_sym_avg[0, :]), len(param_sym_avg[0, :])), np.nanmean(param_sym_avg, axis=0), color=colors[1][experiment_name], linewidth=3)

    param_sym_names = list(labels_dic.keys())
    param_sym_labels = list(labels_dic.values())

    # Define y-axis limits
    if ranges[0] and (ranges[1] is not None):
        ax.set(ylim=ranges[1][param_sym_names[current_param]])
    else:
        ax.set(ylim=[np.nanmin(param_sym_avg[:, :].flatten()), np.nanmax(param_sym_avg[:, :].flatten())])


    set_symmetry_plot(ax, param_sym_labels[current_param])
   
    return fig


# Average with SEM for all experiments compared
def plot_learning_curve_avg_compared(param_sym_multi, current_param, labels_dic, included_animals_list_ids, experiment_colors_dict, experiment_names, intervals=None,  ranges=[False, None], use_median_iqr=False):
    """
    Plots the average learning curve compared across multiple experiments.

    Parameters
    ----------
    param_sym_multi : dict
        Dictionary containing the parameter symmetry data for multiple paths.
    current_param : int
        Index of the current parameter to be plotted.
    labels_dic : dict
        Dictionary where keys are the parameter names and values are the corresponding labels.
    included_animals_list_ids : list
        A list of two lists, the first containing the names/identifiers of the animals to include in the plot, 
        and the second containing the corresponding indices within the all animals list.
    experiment_colors_dict : dict
        Dictionary mapping experiment names to their corresponding colors.
    experiment_names : list, optional
        List of experiment names. Defaults to None.
    intervals : dict, optional
        Dictionary containing 'split' and 'stim' intervals. Defaults to None.
    ranges : list, optional
        List containing a boolean and a dictionary for y-axis limits. Defaults to [False, None].
    use_median_iqr : bool, optional
        If True, plot median with interquartile range (IQR). If False, plot mean with standard error of the mean (SEM). Defaults to False.

    Returns
    -------
    fig_multi : matplotlib.figure.Figure
        The resulting figure object.
    """
    
    fig_multi, ax_multi = plt.subplots(figsize=FIGSIZE, tight_layout=True)
    min_rect = 0
    max_rect = 0
    path_index = 0

    paths = list(param_sym_multi.keys())
    param_sym_names = list(labels_dic.keys())
    param_sym_labels = list(labels_dic.values())
    # Define y-axis limits
    if ranges[0] and (ranges[1] is not None):
        ax_multi.set(ylim=ranges[1][param_sym_names[current_param]])
    else:
        ax_multi.set(ylim=[min_rect, max_rect])

    # Add split and stimulation intervals
    if intervals:
        if 'split' in intervals.keys():
            add_patch_interval(ax_multi, intervals['split'], set_fc='lightgray')
            add_start_end_interval(ax_multi, intervals['split'])
        if 'stim' in intervals.keys():
            add_patch_interval(ax_multi, intervals['stim'], set_fc=laser_color)
            add_start_end_interval(ax_multi, intervals['stim'])
    
    for path in paths:
        ntrial = len(param_sym_multi[path][current_param][0,:])
        x = np.linspace(1, ntrial, ntrial)
        data = param_sym_multi[path][current_param]

        if use_median_iqr:
            center = np.nanmedian(data, axis=0)
            q25 = np.nanpercentile(data, 25, axis=0)
            q75 = np.nanpercentile(data, 75, axis=0)
            plt.plot(x, center, color=experiment_colors_dict[experiment_names[path_index]], linewidth=2, label=experiment_names[path_index])
            ax_multi.fill_between(x, q75, q25, facecolor=experiment_colors_dict[experiment_names[path_index]], alpha=0.5)
            min_rect = min(min_rect, np.nanmin(q25))
            max_rect = max(max_rect, np.nanmax(q75))
        else:
            center = np.nanmean(data, axis=0)
            sem = np.nanstd(data, axis=0) / np.sqrt(len(included_animals_list_ids[0]))
            plt.plot(x, center, color=experiment_colors_dict[experiment_names[path_index]], linewidth=2, label=experiment_names[path_index])
            ax_multi.fill_between(x, center + sem, center - sem, facecolor=experiment_colors_dict[experiment_names[path_index]], alpha=0.5)
            min_rect = min(min_rect, np.nanmin(center - sem))
            max_rect = max(max_rect, np.nanmax(center + sem))

        path_index += 1


    set_symmetry_plot(ax_multi, param_sym_labels[current_param])

    return fig_multi


# LEARNING PARAMETERS
# barplot of all learning parameters with average and SEM + scatterplot of ind animals in the middle (optional)
def plot_all_learning_params(learning_params, current_param_sym, included_animals_list, experiment_names, experiment_colors, animal_colors_dict, stat_learning_params=None, scatter_single_animals=False, ranges=[False, None]):
    """
    Plots all learning parameters in a bar plot with optional scatter plots for individual animals and statistical markers.
    Parameters:
    -----------
    learning_params : dict
        Dictionary where keys are parameter names and values are 2D arrays (experiments x animals) of learning parameter values.
    current_param_sym : list
        List of two elements: the parameter name and the corresponding label.
    included_animals_list : list
        List of animal identifiers included in the analysis.
    experiment_names : list
        List of names of the experiments.
    experiment_colors : list
        List of colors corresponding to each experiment.
    animal_colors_dict : dict
        Dictionary mapping animal identifiers to their respective colors.
    stat_learning_params : dict, optional
        Dictionary where keys are parameter names and values are lists of statistical test results (default is None).
    scatter_single_animals : bool, optional
        If True, scatter plots of individual animals will be added to the bar plots (default is False).
    ranges : list, optional
        List containing a boolean and a dictionary. If the boolean is True, the dictionary specifies the y-axis limits for each parameter (default is [False, None]).

    Returns:
    --------
    fig_bar : matplotlib.figure.Figure
        The figure object containing the bar plots.
    """
    
    nrows = len(learning_params)//3 + 1         # We want always 3 columns
    fig_bar, ax_bar = plt.subplots(nrows,3)
    ax_bar = ax_bar.flatten()

    current_param_sym_name = current_param_sym[0]
    current_param_sym_label = current_param_sym[1]

    for i, (lp_name, lp_values) in enumerate(learning_params.items()):
        if i<3:
            subplot_idx = i
        else:
            subplot_idx = i+1
        if not np.isnan(lp_values).all():              # Check for nans
            bars=ax_bar[subplot_idx].bar([0] + list(range(len(experiment_names) + 1, (len(experiment_names)) * 2)),
                        np.nanmean(lp_values, axis=1),
                        yerr=np.nanstd(lp_values, axis=1) / np.sqrt(len(lp_values)),
                        align='center', alpha=0.5, color=experiment_colors, ecolor='black', capsize=6)
        
        # Add scatterplot of individual animals in the middle
        if scatter_single_animals:
            for a in range(len(lp_values[0])):
                ax_bar[subplot_idx].plot(list(range(1,len(experiment_names)+1)),np.array(lp_values)[:,a],'-o', markersize=4, markerfacecolor=animal_colors_dict[included_animals_list[a]], color=animal_colors_dict[included_animals_list[a]], linewidth=1)
        # Add plots Statistics
        if len(stat_learning_params)>0:
            ax_bar[subplot_idx].plot(list(range(len(experiment_names)+1,(len(experiment_names))*2)),[max(max(np.nanmean(lp_values, axis=1)+np.nanstd(lp_values, axis=1)),0)*i if i==1 else math.nan*i for i in stat_learning_params[lp_name]],'*', color='black')
            print('stat '+lp_name+': '+str(stat_learning_params[lp_name]))

        # Set titles and labels
        if i==0:
            ax_bar[subplot_idx].set_ylabel(current_param_sym_label + ' asymmetry ')
        ax_bar[subplot_idx].set_title(lp_name, size=9)

        set_learning_param_plot(ax_bar[subplot_idx], current_param_sym_name, lp_name, ranges=ranges)
        
    # Remove empty subplots
    for ax in ax_bar:
        if not ax.has_data():
            fig_bar.delaxes(ax)
    
    fig_bar.suptitle(current_param_sym_label)
    fig_bar.tight_layout()
    fig_bar.legend(bars, experiment_names,
        loc="lower left",   
        borderaxespad=3
        )
    
    return fig_bar

# barplot of one selected learning parameter with average and SEM + scatterplot of ind animals in the middle (optional)
def plot_learning_param(learning_param, current_param_sym, lp_name, included_animals_list, experiment_names, experiment_colors, animal_colors_dict, stat_learning_params=None, scatter_single_animals=False, ranges=[False, None]):
    """
    Plots a single learning parameter in a bar plot with optional scatter plots for individual animals and statistical markers.
    Parameters:
    -----------
    learning_param : 2D array
        Array of learning parameter values (experiments x animals).
    current_param_sym : list
        List of two elements: the parameter name and the corresponding label.
    included_animals_list : list
        List of animal identifiers included in the analysis.
    experiment_names : list
        List of names of the experiments.
    experiment_colors : list
        List of colors corresponding to each experiment.
    animal_colors_dict : dict
        Dictionary mapping animal identifiers to their respective colors.
    stat_learning_params : list, optional
        List of statistical test results (default is None).
    scatter_single_animals : bool, optional
        If True, scatter plots of individual animals will be added to the bar plots (default is False).
    ranges : list, optional
        List containing a boolean and a dictionary. If the boolean is True, the dictionary specifies the y-axis limits for the parameter (default is [False, None]).

    Returns:
    --------
    fig_bar : matplotlib.figure.Figure
        The figure object containing the bar plot.
    """
    
    fig_bar, ax_bar = plt.subplots(figsize=FIGSIZE, tight_layout=True)
    
    current_param_sym_name = current_param_sym[0]
    current_param_sym_label = current_param_sym[1] 

    if not np.isnan(learning_param).all():              # Check for nans
        ax_bar.bar([0] + list(range(len(experiment_names) + 1, (len(experiment_names)) * 2)),
                np.nanmean(learning_param, axis=1),
                yerr=np.nanstd(learning_param, axis=1) / np.sqrt(len(learning_param)),
                align='center', alpha=0.5, color=experiment_colors, ecolor='black', capsize=6)
    # Add scatterplot of individual animals in the middle
    if scatter_single_animals:
        for a in range(len(learning_param[0])):
            ax_bar.plot(list(range(1,len(experiment_names)+1)),np.array(learning_param)[:,a],'-o', markersize=8, markerfacecolor=animal_colors_dict[included_animals_list[a]], color=animal_colors_dict[included_animals_list[a]], linewidth=1)
    # Add plots Statistics
    if len(stat_learning_params)>0:
        ax_bar.plot(list(range(len(experiment_names)+1,(len(experiment_names))*2)),[max(max(np.nanmean(learning_param, axis=1)+np.nanstd(learning_param, axis=1)),0)*i if i==1 else math.nan*i for i in stat_learning_params[lp_name]],'*', color='black')

    # Set titles and labels
    ax_bar.set_ylabel(current_param_sym_label + ' asymmetry ', fontsize=LABEL_SIZE)
    ax_bar.set_title(lp_name, fontsize=LABEL_SIZE)
    ax_bar.set_xticks([0]+list(range(len(experiment_names)+1,(len(experiment_names))*2)))
    ax_bar.set_xticklabels(experiment_names)

    set_learning_param_plot(ax_bar, current_param_sym_name, lp_name, ranges=ranges)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    
    return fig_bar

# average line + scatterplot (no animals color or with animals color)
def plot_learning_param_scatter(learning_param, current_param_sym, lp_name, included_animals_list, experiment_names, experiment_colors, animal_colors_dict=None, stat_learning_params=None, ranges=[False, None]):
    """
    Plots a single learning parameter with a scatter plot of individual animals.
    Parameters:
    -----------
    learning_param : 2D array
        Array of learning parameter values (experiments x animals).
    current_param_sym_name : list
        List of two elements: the parameter name and the corresponding label.
    included_animals_list : list
        List of animal identifiers included in the analysis.
    experiment_names : list
        List of names of the experiments.
    experiment_colors : list
        List of colors corresponding to each experiment.
    animal_colors_dict : dict
        Dictionary mapping animal identifiers to their respective colors.
    stat_learning_param : list, optional
        List of statistical test results (default is None).
    scatter_single_animals : bool, optional
        If True, scatter plots of individual animals will be added to the bar plots (default is False).
    ranges : list, optional
        List containing a boolean and a dictionary. If the boolean is True, the dictionary specifies the y-axis limits for the parameter (default is [False, None]).

    Returns:
    --------
    fig_scatter : matplotlib.figure.Figure
        The figure object containing the scatter plot.
    """
    
    fig_scatter, ax_scatter = plt.subplots(figsize=FIGSIZE, tight_layout=True)
    
    current_param_sym_name = current_param_sym[0]
    current_param_sym_label = current_param_sym[1] 

    x=np.linspace(1,len(experiment_names),len(experiment_names))

    # Add single animal data
    if animal_colors_dict is None:              # Plot all animal lines in lightblue
        for a in range(len(learning_param[0])):
            ax_scatter.plot(x,np.array(learning_param)[:,a],'-', color='lightblue', linewidth=1)
    else:
        for a in range(len(learning_param[0])):
            ax_scatter.plot(x,np.array(learning_param)[:,a],'-o', markersize=4, markerfacecolor=animal_colors_dict[included_animals_list[a]], color=animal_colors_dict[included_animals_list[a]], linewidth=1)
    for count_exp, exp in enumerate(experiment_names):
        if animal_colors_dict is None:
            ax_scatter.scatter([x[count_exp]] * len(learning_param[count_exp][:]), learning_param[count_exp][:], s=60, c=experiment_colors[count_exp])             # Plot all animal points in the corresponding experiment color
        # Add avg value
        ax_scatter.plot([x[count_exp]-0.15, x[count_exp]+0.15], [np.nanmean(learning_param[count_exp][:]), np.nanmean(learning_param[count_exp][:])], color=experiment_colors[count_exp], linewidth=4)
        
    # Add plots Statistics
    if len(stat_learning_params)>0:
        ax_scatter.plot(x[1:],[max(max(np.nanmean(learning_param, axis=1)+2*np.nanstd(learning_param, axis=1)),0)*i if i==1 else math.nan*i for i in stat_learning_params[lp_name]],'*', color='black')

    # Set titles and labels
    ax_scatter.set_ylabel(current_param_sym_label + ' asymmetry ', fontsize=LABEL_SIZE)
    ax_scatter.set_title(lp_name, fontsize=LABEL_SIZE)
    ax_scatter.set_xticks(list(range(1,len(experiment_names)+1)))
    ax_scatter.set_xticklabels(experiment_names)

    set_learning_param_plot(ax_scatter, current_param_sym_name, lp_name, ranges=ranges)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)

    return fig_scatter



# UTILS plotting functions
def add_patch_interval(ax, intervals, set_fc='lightblue'):
    """
    Adds split or stimulation intervals to a plot, as a patch light blue rectangle.
    
    Parameters:
    ax (matplotlib.axes.Axes): The axes object to add the intervals to.
    intervals (list): A list containing the start and duration (in trials) of the interval to plot as a patch.
    set_fc (str, optional): The color of the patch. Defaults to 'lightgray'.   
    """

    start, duration = intervals
    rectangle = plt.Rectangle((start - 0.5, ax.get_ylim()[0]), duration,
                                    ax.get_ylim()[1] - ax.get_ylim()[0],
                                    fc=set_fc, alpha=0.6)
    ax.add_patch(rectangle)
    
 
def add_start_end_interval(ax, intervals):
    """
    Adds split or stimulation intervals to a plot, as start and end lines.
    
    Parameters:
    ax (matplotlib.axes.Axes): The axes object to add the intervals to.
    intervals (list): A list containing the start and duration (in trials) of the interval to plot as a start and end line.
    """
    start, duration = intervals
    ax.axvline(x=start-0.5, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=start+duration-0.5, color='k', linestyle='-', linewidth=0.5)


def set_symmetry_plot(ax, param_name):
    # Add horizontal line at 0
    ax.axhline(y=0, color='grey', linestyle='--')

    # Set labels
    ax.set_xlabel('1-min trial', fontsize=LABEL_SIZE)
    ax.set_ylabel(param_name + ' asymmetry', fontsize=LABEL_SIZE)

    # Set ticks
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    # Hide right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def set_learning_param_plot(ax, param_sym_name, lp_name, ranges=[False, None]):

    # Add zero line and set ranges
    ax.axhline(y = 0, color = 'k', linestyle = '--', linewidth=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if ranges[0] and (ranges[1] is not None):
        ax.set(ylim= ranges[1][param_sym_name])      
        if 'change' in lp_name:      # Increased ranges for _sym_change parameters
            ax.set(ylim= list(30*np.array(ranges[1][param_sym_name])))
    

def plot_symmetry_scatterplot(data, experiment_names, experiment_colors, ylabel,
                              stat_results=None, ylim=None, point_colors=None, display_names=None,
                              stat_mode='between_paths', connect_points=True):
    """
    Scatter plot of a symmetry parameter across experiments, with individual animal lines
    and mean markers.

    Parameters
    ----------
    data : list of array-like
        One array per experiment, each of shape (n_animals,).
    experiment_names : list of str
        Names of the experiments (used to detect WT control).
    experiment_colors : list
        One color per experiment.
    ylabel : str
        Label for the y-axis.
    stat_results : list of bool, optional
        Statistical significance flags (length = len(data)-1). Stars are plotted
        for True entries.
    ylim : list or tuple, optional
        Y-axis limits [ymin, ymax].

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    scatter_figsize = (3, 4)
    scale = scatter_figsize[1] / (FIGSIZE[1]/1.5)
    label_size = LABEL_SIZE * scale
    tick_size = TICK_SIZE * scale

    fig, ax = plt.subplots(figsize=scatter_figsize, tight_layout=True)
    x_positions = list(range(1, len(data) + 1))

    # Connect individual animals across experiments
    if connect_points:
        skip_first = 'WT' in experiment_names and len(experiment_names) > 1
        if skip_first:
            for a in range(len(data[1])):
                ax.plot(x_positions[1:], np.array(data)[1:, a], '-', color='lightgray', linewidth=1)
        else:
            for a in range(len(data[0])):
                ax.plot(x_positions, np.array(data)[:, a], '-', color='lightgray', linewidth=1)

    # Scatter points and mean lines
    for i, x in enumerate(x_positions):
        _c = point_colors[i] if (point_colors is not None and point_colors[i] is not None) else [experiment_colors[i]] * len(data[i])
        ax.scatter([x] * len(data[i][:]), data[i][:], s=30, c=_c)
        ax.plot([x - 0.15, x + 0.15],
                [np.nanmean(data[i][:]), np.nanmean(data[i][:])],
                color=experiment_colors[i], linewidth=4)

    # Formatting
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel(ylabel, fontsize=label_size)
    ax.set(xlim=[x_positions[0] - 0.5, x_positions[-1] + 0.5])
    ax.set_xticks(x_positions)
    if display_names is None:
        display_names = ['non-inj' if name == 'WT' else name for name in experiment_names]
    ax.set_xticklabels(display_names)
    if ylim is not None:
        ax.set(ylim=ylim)

    # Statistics labels
    if stat_results is not None and len(stat_results) > 0:
        all_values = np.concatenate([d for d in data])
        y_max = np.nanmax(all_values[~np.isnan(all_values)]) if np.any(~np.isnan(all_values)) else 0
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        if stat_mode == 'between_paths':
            label_step = 0.14 if len(stat_results) > 1 else 0.08
            top_label_y = y_max + label_step * y_range * len(stat_results)
            if top_label_y > ax.get_ylim()[1]:
                ax.set_ylim(ax.get_ylim()[0], top_label_y + 0.05 * y_range)
            for ci, stat_val in enumerate(stat_results):
                if isinstance(stat_val, (bool, np.bool_)):
                    label = '*' if stat_val else 'n.s.'
                elif stat_val is None or (isinstance(stat_val, (float, np.floating)) and np.isnan(stat_val)):
                    label = 'n.s.'
                else:
                    pval = float(stat_val)
                    if pval < 0.001:
                        label = '**'
                    elif pval < 0.05:
                        label = '*'
                    else:
                        label = 'n.s.'

                x_mid = (x_positions[0] + x_positions[ci + 1]) / 2
                y_pos = y_max + label_step * y_range * (ci + 1)
                y_line_offset = 0.015 if label != 'n.s.' else 0.04
                y_line = y_pos - y_line_offset * y_range
                ax.plot([x_positions[0], x_positions[ci + 1]], [y_line, y_line], color='k', linewidth=0.6)
                ax.text(x_mid, y_pos, label, ha='center', va='center', fontsize=20)
        elif stat_mode == 'vs_zero':
            label_step = 0.10 if len(stat_results) > 1 else 0.08
            top_label_y = y_max + label_step * y_range * len(stat_results)
            if top_label_y > ax.get_ylim()[1]:
                ax.set_ylim(ax.get_ylim()[0], top_label_y + 0.05 * y_range)
            for ci, stat_val in enumerate(stat_results):
                if isinstance(stat_val, (bool, np.bool_)):
                    label = '*' if stat_val else 'n.s.'
                elif stat_val is None or (isinstance(stat_val, (float, np.floating)) and np.isnan(stat_val)):
                    label = 'n.s.'
                else:
                    pval = float(stat_val)
                    if pval < 0.001:
                        label = '**'
                    elif pval < 0.05:
                        label = '*'
                    else:
                        label = 'n.s.'

                y_pos = y_max + label_step * y_range * (ci + 1)
                ax.text(x_positions[ci], y_pos, label, ha='center', va='center', fontsize=20)
        else:
            raise ValueError(f'Unknown stat_mode: {stat_mode}')

    plt.xticks(fontsize=tick_size, rotation=30)
    plt.yticks(fontsize=tick_size)

    return fig


def save_plot(figure, path, param_name, plot_name='', bs_bool=False, dpi=128):
    """
    Saves the input plot as a .png file, and also in .eps and .svg formats (in separate subfolders).
    Parameters:
    figure (matplotlib.figure.Figure): The figure object to save.
    path (str): The path to save the figure.
    param_name (str): The name of the parameter represented in the figure.
    plot_name (str, optional): The name of the plot. Defaults to ''.
    bs_bool (bool, optional): A boolean indicating if the plot is for the baseline or non baseline subtracted parameter. Defaults to False.
    dpi (int, optional): The resolution of the saved figure. Defaults to 128.

    """
    if bs_bool:
        filename = param_name + '_sym_bs_' + plot_name
    else:
        filename = param_name + '_sym_non_bs_' + plot_name

    utils.save_figure_multi_format(figure, path, filename, dpi=dpi)