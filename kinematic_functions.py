"""
Created on Tue Oct 15 14:42:12 2024

@author: Alice Geminiani
"""               

# Kinematic analysis functions
import numpy as np
import matplotlib.pyplot as plt

def resample_strides_position(position, st_matrix, sw_points, paw, num_samples, center = 'sw', force_center=0):
    """
    Interpolate stride position to be st-sw-st/sw-st-sw with or without forcing swing/stance in the middle or at 66-33%.
        Input: position - x or y or z paw excursion centered
               st_matrix (stridesx2x5) - stance matrix with info on start and end stride timing, position and idx
               sw_points (stridesx1x5) - swing points matrix
               paw - index of the current paw to analyze
               num_samples - total number of new samples
               center - 'st' or 'sw' to center the stride around stance or swing; default is 'sw'
               force_center - if True, the center of the stride will be forced to be at the middle of the stance/swing; default is False
        Output: strides_resampled (strides x num_samples)
    """
    
    strides_resampled = np.empty((len(st_matrix[paw]),num_samples))
    for s in range(len(st_matrix[paw])-1):
        if center == 'st':
            current_stride = position[paw][int(sw_points[paw][s,0,-1]):int(sw_points[paw][s+1,0,-1])]
            x_stride = np.linspace(sw_points[paw][s,0,0],sw_points[paw][s+1,0,0], len(current_stride))
        elif center == 'sw':
            current_stride = position[paw][int(st_matrix[paw][s,0,-1]):int(st_matrix[paw][s,1,-1])]
            x_stride = np.linspace(st_matrix[paw][s,0,0],st_matrix[paw][s,1,0], len(current_stride))
                
        # New x values for resampled signal with num_samples data points
        if force_center!=0:
            if center == 'st':
                x_stride_resampled = np.linspace(sw_points[paw][s,0,0],st_matrix[paw][s+1,0,0], int(num_samples*force_center/2))
                x_stride_resampled = np.append(x_stride_resampled, np.linspace(st_matrix[paw][s+1,0,0],sw_points[paw][s+1,0,0], int(num_samples*force_center/2)))
            elif center == 'sw':
                x_stride_resampled = np.linspace(st_matrix[paw][s,0,0],sw_points[paw][s,0,0], int(num_samples*force_center))
                x_stride_resampled = np.append(x_stride_resampled, np.linspace(sw_points[paw][s,0,0],st_matrix[paw][s,1,0], int(num_samples*force_center)))
            else:
                raise ValueError("center must be 'st' or 'sw'")
        else:
            if center == 'st':
                x_stride_resampled = np.linspace(sw_points[paw][s,0,0],sw_points[paw][s+1,0,0], num_samples)
            elif center == 'sw':
                x_stride_resampled = np.linspace(st_matrix[paw][s,0,0],st_matrix[paw][s,1,0], num_samples)
            else:
                raise ValueError("center must be 'st' or 'sw'")
        
        
        # Linear interpolation to resample the signal
        current_stride_resampled = np.interp(x_stride_resampled, x_stride, current_stride)

        # Add to array
        strides_resampled[s,:] = current_stride_resampled

    return strides_resampled

def plot_resampled_position(strides_resampled, variable_name, paw_color, paw_name, animal, path_save, center = 'sw', force_center=False):
    fig, ax = plt.subplots(tight_layout=True, figsize=(5,5))
    # Add to plot
    for s in range(strides_resampled.shape[0]):
        plt.plot(strides_resampled[s,:], paw_color, linewidth=1, alpha=0.2)
    # Add the average
    plt.plot(np.nanmean(strides_resampled,axis=0), paw_color, linewidth=2)
    set_kinematic_plot_style(ax, variable_name, paw_name, center, force_center)
    # Save figure
    plt.savefig(path_save + animal + '_strides_' +variable_name+'_' +paw_name, dpi=128)


def plot_resampled_position_all_trials(strides_resampled_trials, variable_name, paw_name, animal, path_save, center = 'sw', force_center=0):
    fig, ax = plt.subplots(tight_layout=True, figsize=(7,10))
    main_trials_color = ['lightgray', 'gray', 'lightblue', 'blue', 'green', 'lightgreen']  # Colors for main trials
    index_trial = 0
    # Add to plot
    for trial in [0, 7, 8, 15, 18, -1]:    #   range(len(avg_strides_resampled_trials)):
        #color = plt.cm.viridis(trial / len(avg_strides_resampled_trials))              # For all trials
        color = main_trials_color[index_trial]         # For main trials
        plt.plot(np.nanmean(strides_resampled_trials[trial],axis=0), color=color, linewidth=2)
        ax.fill_between(np.linspace(1, 360, 360), 
                    np.nanmean(strides_resampled_trials[trial],axis=0)+np.nanstd(strides_resampled_trials[trial],axis=0), 
                    np.nanmean(strides_resampled_trials[trial],axis=0)-np.nanstd(strides_resampled_trials[trial],axis=0), 
                    facecolor=color, alpha=0.5)
        
        index_trial += 1

    set_kinematic_plot_style(ax, variable_name, paw_name, center, force_center)
    # Save figure
    plt.savefig(path_save + animal + '_strides_all_trials_' +variable_name+'_' +paw_name, dpi=128)

def plot_resampled_position_avg_all(strides_resampled_avg_all, variable_name, paw_name, trials, path_save, center = 'sw', force_center=0):
    fig, ax = plt.subplots(tight_layout=True, figsize=(7,10))
    # Add to plot
    #color = plt.cm.viridis(trial / len(avg_strides_resampled_trials))              # For all trials
    #color = main_trials_color[index_trial]         # For main trials
    plt.plot(np.nanmean(np.nanmean(strides_resampled_avg_all[:,trials,:],axis=0),axis=0), color='red', linewidth=2)

    # ax.fill_between(np.linspace(1, 360, 360), 
    #            np.nanmean(strides_resampled_trials[trial],axis=0)+np.nanstd(strides_resampled_trials[trial],axis=0), 
    #            np.nanmean(strides_resampled_trials[trial],axis=0)-np.nanstd(strides_resampled_trials[trial],axis=0), 
        #           facecolor=color, alpha=0.5)
    

    set_kinematic_plot_style(ax, variable_name, paw_name, center, force_center)
    # Save figure
    plt.savefig(path_save + '_strides_avg_all_trials_from_' +str(trials[0])+'to'+str(trials[-1])+variable_name+'_' +paw_name, dpi=128)

def set_kinematic_plot_style(ax, variable_name, paw_name, center = 'sw', force_center=0):
    """
    Set the style for kinematic plots.
    """
    ax.set_ylabel(variable_name+' position (mm)')
    ax.set_xlabel('stride time')
    ax.set_xlim([0, 360])
    if force_center!=0:
        ax.set_xticks([0,360*force_center,360])
        if center == 'st':
            ax.set_xticklabels(['sw','st','sw'])
        else:
            ax.set_xticklabels(['st','sw','st'])
    else:
        ax.set_xticks([0,360])
        if center == 'st':
            ax.set_xticklabels(['sw','sw'])
        else:
            ax.set_xticklabels(['st','st'])
    if 'F' in paw_name and variable_name=='x':
        ax.set_ylim([-10,40])
    elif 'H' in paw_name and variable_name=='x':
        ax.set_ylim([-60,20])
    elif 'L' in paw_name and variable_name=='y':
        ax.set_ylim([-20, 0])
    elif 'R' in paw_name and variable_name=='y':    
        ax.set_ylim([0, 20])
    else:
        ax.set_ylim([-7, 7])


def plot_stacked_stride_positions(
    ax,
    padded_positions,
    stride_laser_onsets,
    stride_laser_offsets,
    min_time,
    max_time,
    time_range,
    axis,
    paw_name,
    paw_color,
    laser_color,
    center,
    title=None,
    show_colorbar=True,
    show_labels=True,
    vmin=0,
    vmax=40,
    show_xlabel=None,
    show_ylabel=None,
    show_legend=None,
    legend_loc='upper right',
    legend_bbox=(1.02, 1.02),
):
    """
    Plot stacked stride positions as a 2D grayscale image with laser intervals overlaid.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    padded_positions : np.ndarray
        2D array of stride positions (strides x time samples)
    stride_laser_onsets : list
        List of laser onset times for each stride
    stride_laser_offsets : list
        List of laser offset times for each stride
    min_time, max_time : float
        Time range for the extent of the image
    time_range : list
        [min, max] time range for x-axis limits
    axis : str
        Axis label (e.g., 'x', 'y', 'z')
    paw_name : str
        Name of the paw
    paw_color : str
        Color for paw-related elements
    laser_color : str
        Color for laser interval shading
    center : str
        'st' or 'sw' for stance or swing centering
    title : str, optional
        Plot title
    show_colorbar : bool
        Whether to show the colorbar
    show_labels : bool
        Backward compatible flag for enabling labels/legend together
    vmin, vmax : float
        Color scale limits
    show_xlabel, show_ylabel, show_legend : bool or None
        Fine-grained control over label/legend visibility (defaults follow show_labels)
    legend_loc : str
        Matplotlib legend location string
    legend_bbox : tuple
        bbox_to_anchor tuple for legend placement
    
    Returns:
    --------
    im : matplotlib.image.AxesImage
        The image object
    """
    # Use the requested time_range for a consistent x-axis extent across plots
    im = ax.imshow(padded_positions, aspect='equal', cmap='gray',
                   extent=[time_range[0], time_range[1], 0, len(padded_positions)],
                   origin='lower', vmin=vmin, vmax=vmax, interpolation='nearest')
    
    # Add shaded areas between onset and offset times of laser
    # Iterate over all strides (rows in padded_positions)
    n_strides = len(padded_positions)
    n_laser = len(stride_laser_onsets)
    
    if n_strides != n_laser:
        print(f"Warning: positions has {n_strides} strides but laser data has {n_laser} entries")
    
    for t in range(n_strides):
        # Only draw laser patch if we have laser data for this stride
        if t < n_laser:
            onset = stride_laser_onsets[t]
            offset = stride_laser_offsets[t]
            # Check for valid (non-NaN, non-None) values
            if onset is not None and offset is not None:
                try:
                    if not (np.isnan(onset) or np.isnan(offset)):
                        ax.fill_betweenx(
                            y=[t, t+1],
                            x1=onset,
                            x2=offset,
                            color=laser_color,
                            alpha=0.3, lw=0.1
                        )
                except (TypeError, ValueError):
                    # Skip if values can't be checked for NaN
                    pass
    
    ax.set_xlim(time_range[0], time_range[1])
    ax.set_ylim(0, len(padded_positions))
    ax.axvline(x=0, color=paw_color, linestyle='--', linewidth=1, label=center + ' onset')
    
    # Resolve per-element visibility (defaults follow legacy show_labels)
    if show_xlabel is None:
        show_xlabel = show_labels
    if show_ylabel is None:
        show_ylabel = show_labels
    if show_legend is None:
        show_legend = show_labels

    if title:
        ax.set_title(title, fontsize=10)

    if show_xlabel:
        ax.set_xlabel('Time (ms)')
    if show_ylabel:
        ax.set_ylabel('Stride Index')
    if show_legend:
        ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox, fontsize=8, frameon=False)
    
    return im