"""
Created on Wed Feb 12 09:16:35 2025

@author: Alice Geminiani
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import os
import scipy.stats as st

# Plotting functions
# Locomotor adaptation
# Baselines


# STANCE PHASE
def plot_phase(phase_data, animals, paw_colors, intervals=None):
    ntrial = np.shape(phase_data)[2]
    # plot stance phase - polar - group mean
    #error bars in polar plot don't rotate well
    fig_group_mean = plt.figure(figsize=(10, 10), tight_layout=True)
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
                    add_patch_interval(ax[count_a], intervals['split'], set_fc='lightblue')
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
    fig, ax = plt.subplots(figsize=(7,10), tight_layout=True)
   
    for p in range(4):
        ax.plot(np.linspace(1,len(data[p,:]),len(data[p,:])), data[p,:], color = paw_colors[p], linewidth = 2)
    # Add split and stimulation intervals
    if intervals:
        if 'split' in intervals.keys():
            add_patch_interval(ax, intervals['split'], set_fc='lightgray')
        if 'stim' in intervals.keys():
            add_patch_interval(ax, intervals['split'], set_fc='lightblue')
            add_start_end_interval(ax, intervals['stim'])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Trial', fontsize = 24)
    ax.set_ylabel('Stance speed', fontsize = 24)
    ax.tick_params(axis='x',labelsize = 20)
    ax.tick_params(axis='y',labelsize = 20)
    ax.set_title(animal,fontsize=18)

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
    fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)


         
    # Add split and stimulation intervals
    if intervals:
        if 'split' in intervals.keys():
            add_patch_interval(ax, intervals['split'], set_fc='lightgray')
        if 'stim' in intervals.keys():
            add_patch_interval(ax, intervals['split'], set_fc='lightblue')
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

    fig, ax = plt.subplots(figsize=(7, 10), tight_layout=True)

    # Add split and stimulation intervals
    if intervals:
        if 'split' in intervals.keys():
            add_patch_interval(ax, intervals['split'], set_fc='lightgray')
        if 'stim' in intervals.keys():
            add_patch_interval(ax, intervals['split'], set_fc='lightblue')
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
def plot_learning_curve_avg_compared(param_sym_multi, current_param, labels_dic, included_animals_list_ids, experiment_colors_dict, experiment_names, intervals=None,  ranges=[False, None]):
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

    Returns
    -------
    fig_multi : matplotlib.figure.Figure
        The resulting figure object.
    """
    
    fig_multi, ax_multi = plt.subplots(figsize=(7, 10), tight_layout=True)
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
        if 'stim' in intervals.keys():
            add_patch_interval(ax_multi, intervals['split'], set_fc='lightblue')
            add_start_end_interval(ax_multi, intervals['stim'])
    
    for path in paths:
        ntrial = len(param_sym_multi[path][current_param][0,:])
        plt.plot(np.linspace(1, ntrial, ntrial), np.nanmean(param_sym_multi[path][current_param], axis = 0), 
                 color=experiment_colors_dict[experiment_names[path_index]],  linewidth=2, label=experiment_names[path_index])
        # Add SE of each session
        ax_multi.fill_between(np.linspace(1, ntrial, ntrial), 
                    np.nanmean(param_sym_multi[path][current_param], axis = 0)+np.nanstd(param_sym_multi[path][current_param], axis = 0)/np.sqrt(len(included_animals_list_ids[0])), 
                    np.nanmean(param_sym_multi[path][current_param], axis = 0)-np.nanstd(param_sym_multi[path][current_param], axis = 0)/np.sqrt(len(included_animals_list_ids[0])), 
                    facecolor=experiment_colors_dict[experiment_names[path_index]], alpha=0.5)
        min_rect = min(min_rect,np.nanmin(np.nanmean(param_sym_multi[path][current_param], axis = 0)-np.nanstd(param_sym_multi[path][current_param], axis = 0)))
        max_rect = max(max_rect,np.nanmax(np.nanmean(param_sym_multi[path][current_param], axis=0)+np.nanstd(param_sym_multi[path][current_param], axis = 0)))
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
    
    fig_bar, ax_bar = plt.subplots(figsize=(7, 10), tight_layout=True)
    
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
    ax_bar.set_ylabel(current_param_sym_label + ' asymmetry ', fontsize=24)
    ax_bar.set_title(lp_name, fontsize=24)
    ax_bar.set_xticks([0]+list(range(len(experiment_names)+1,(len(experiment_names))*2)))
    ax_bar.set_xticklabels(experiment_names)

    set_learning_param_plot(ax_bar, current_param_sym_name, lp_name, ranges=ranges)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=20)
    
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
    
    fig_scatter, ax_scatter = plt.subplots(figsize=(7, 10), tight_layout=True)
    
    current_param_sym_name = current_param_sym[0]
    current_param_sym_label = current_param_sym[1] 

    x=np.linspace(1,len(experiment_names),len(experiment_names))

    # Add single animal data
    if animal_colors_dict is None:              # Plot all animal lines in lightgray
        for a in range(len(learning_param[0])):
            ax_scatter.plot(x,np.array(learning_param)[:,a],'-', color='lightgray', linewidth=1)
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
    ax_scatter.set_ylabel(current_param_sym_label + ' asymmetry ', fontsize=24)
    ax_scatter.set_title(lp_name, fontsize=24)
    ax_scatter.set_xticks(list(range(1,len(experiment_names)+1)))
    ax_scatter.set_xticklabels(experiment_names)

    set_learning_param_plot(ax_scatter, current_param_sym_name, lp_name, ranges=ranges)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=20)

    return fig_scatter



# UTILS plotting functions
def add_patch_interval(ax, intervals, set_fc='lightgray'):
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
                                    fc=set_fc, alpha=0.3)
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
    ax.set_xlabel('1-min trial', fontsize=24)
    ax.set_ylabel(param_name + ' asymmetry', fontsize=24)

    # Set ticks
    ax.tick_params(axis='both', which='major', labelsize=20)

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
    

def save_plot(figure, path, param_name, plot_name='', bs_bool=False, dpi=128):
    """
    Saves the input plot as a .png file.
    Parameters:
    figure (matplotlib.figure.Figure): The figure object to save.
    path (str): The path to save the figure.
    param_name (str): The name of the parameter represented in the figure.
    plot_name (str, optional): The name of the plot. Defaults to ''.
    bs_bool (bool, optional): A boolean indicating if the plot is for the baseline or non baseline subtracted parameter. Defaults to False.
    dpi (int, optional): The resolution of the saved figure. Defaults to 128.

    """
    if not os.path.exists(path):
        os.mkdir(path)
    if bs_bool:
        figure.savefig(path + param_name + '_sym_bs_'+ plot_name, dpi=dpi)
    else:
        figure.savefig(path + param_name + '_sym_non_bs_'+ plot_name, dpi=dpi)