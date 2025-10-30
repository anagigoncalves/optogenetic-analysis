import locomotion_class
import plotly.graph_objects as go
from scipy.signal import savgol_filter, find_peaks, medfilt
from scipy.interpolate import CubicSpline
import numpy as np
import os
from plotly.subplots import make_subplots
from utils import compute_rmse

#TODO: APPLY EVERYTHING TO THE REL SIGNALS!!!
confidence = 0.8  # Confidence threshold for DLC tracking

def correct_signal_cubic_spline(signal, dt, thr_sig='velocity'):
    """
    Corrects a signal using cubic spline interpolation.
    """
    time = np.arange(len(signal)) * dt  # in seconds

    # Step 1: Compute velocity (difference)
    velocity = np.diff(signal, prepend=signal[0])
    acceleration = np.diff(velocity, prepend=velocity[0]) / dt  # Compute acceleration

    #TODO: threshold for velocity and buffer size can change things!!! Check! And also check the velocity signal. The bad points are basically the ones where the velocity if first positive and abruptly get nevative
    # or viceversa - maybe the threshold can be on ACCELERATION instead of VELOCITY?
    # Step 2: Detect outliers based on velocity threshold
    if thr_sig == 'velocity':
        v_std = np.nanstd(velocity)
        v_thresh = 3 * v_std
        print("Velocity threshold:", v_thresh)
        outliers_velocity = np.abs(velocity) > v_thresh
        # Check also for acceleration outliers, when acceleration is constant it is a bad tracking period!
        # Identify regions where velocity remains constant for more than 5 samples
        constant_velocity = np.abs(np.diff(velocity, prepend=velocity[0])) == 0
        constant_velocity_outliers = np.convolve(constant_velocity, np.ones(5, dtype=int), mode='same') >= 5

        outliers = outliers_velocity | constant_velocity_outliers  # Combine both conditions
    elif thr_sig == 'acceleration':
        v_std = np.nanstd(acceleration)
        v_thresh = 3 * v_std
        outliers = np.abs(acceleration) > v_thresh

    # Expand outlier region slightly to cover transition
    buffer = 5
    outlier_idx = np.where(outliers)[0]
    for idx in outlier_idx:
        outliers[max(0, idx-buffer):min(len(outliers), idx+buffer+1)] = True

    # Step 3: Perform cubic spline interpolation over good points
    good_idx = ~outliers
    signal_without_outliers = signal.copy()
    signal_without_outliers[~good_idx] = np.nan  # Set outliers to NaN for visualization
    
    # Remove NaN values for spline interpolation
    valid_idx = ~np.isnan(signal[good_idx])
    valid_time = time[good_idx][valid_idx]
    valid_signal = signal[good_idx][valid_idx]
    
    if len(valid_signal) > 0:
        cs = CubicSpline(valid_time, valid_signal, bc_type='clamped')
        corrected_signal = cs(time)
    else:
        corrected_signal = signal.copy()  # Return original signal if no valid points
    
    return velocity, acceleration, signal_without_outliers, corrected_signal 



path = 'D:\\AliG\\climbing-opto-treadmill\\Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\tied stance stim retracked with finetuned DLC\\'             # retracked with finetuned DLC
#path_retracked = 'D:\\AliG\\climbing-opto-treadmill\Experiments JAWS RT\\Tied belt sessions\\ALL_ANIMALS\\tied stance stim retracked with finetuned DLC\\'  # retracked with finetuned DLC
# path = 'D:\\AliG\\climbing-opto-treadmill\\Videos Coralie - miniscopeDLCmodel\\'
path_retracked = None
animal = 'MC16848'         #'RN-F20-RightF' 
session = 1

manual_track = True  # If True, load and compare with manual tracking

paws = ['FR']                      # 'HR', 'HL'  # Front and hind paws         


loco = locomotion_class.loco_class(path)
if path_retracked:
    loco_retracked = locomotion_class.loco_class(path_retracked)
    filelist_retracked = loco_retracked.get_track_files(animal, session)
dt = 1/loco.sr      #0.003  # 3 milliseconds = 0.003 seconds


filelist = loco.get_track_files(animal, session)



if manual_track:
    file_path_FRbottom = path+'MC16848_159_22_0.275_0.275_tied_1_1DLC_resnet50_tmLOCOAug29shuffle1_1000000_labeled_FRbottom_points.npy'
    file_path_FRbottom_validation = path+'MC16848_159_22_0.275_0.275_tied_1_1DLC_resnet50_tmLOCOAug29shuffle1_1000000_labeled_FRbottom_validation_points.npy'

    manual_track_FRbottom = np.load(file_path_FRbottom)
    manual_track_FRbottom_validation = np.load(file_path_FRbottom_validation)

for f in filelist[:3]:
    count_trial = int(f.split('DLC')[0].split('_')[-1])
    [final_tracks, tracks_tail, joints_wrist, joints_elbow, ear, bodycenter] = loco.read_h5(f, confidence, 0)
    [st_strides_mat, sw_pts_mat] = loco.get_sw_st_matrices(final_tracks, 1)
    paws_rel = {'x': loco.get_paws_rel(final_tracks, 'X'), 'y': loco.get_paws_rel(final_tracks, 'Y'), 'z': loco.get_paws_rel(final_tracks, 'Z')}
    if path_retracked:
        f_retracked = filelist_retracked[filelist.index(f)]
        [final_tracks_retracked, tracks_tail_retracked, joints_wrist_retracked, joints_elbow_retracked, ear_retracked, bodycenter_retracked] = loco_retracked.read_h5(f_retracked, 0.9, 0)
        paws_rel_retracked = {'x': loco_retracked.get_paws_rel(final_tracks_retracked, 'X'), 
                              'y': loco_retracked.get_paws_rel(final_tracks_retracked, 'Y'), 
                              'z': loco_retracked.get_paws_rel(final_tracks_retracked, 'Z')}
       
    for p in range(len(paws)):
        X = final_tracks[0,:,:]*loco.pixel_to_mm
        Y = final_tracks[1,:,:]*loco.pixel_to_mm
        Z = -final_tracks[3,:,:]*loco.pixel_to_mm
        
        # Represent all signals in a  plotly figure
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(y=X[p,:], name=paws[p]+' Xraw', line=dict(color='gray')))

        # Interpolate nans with linear interpolation
        X_interp = loco.inpaint_nans(X)
        Y_interp = loco.inpaint_nans(Y)
        Z_interp = loco.inpaint_nans(Z)
        fig.add_trace(go.Scatter(y=X_interp[p,:], name=paws[p]+' Xraw_interp', line=dict(color='black')))
        

        # Interpolate nans with cubic spline interpolation
        X = final_tracks[0,:,:]*loco.pixel_to_mm
        X_interp_spline = loco.inpaint_nans_cubic_spline(X)
        Y_interp_spline = loco.inpaint_nans_cubic_spline(Y)
        Z_interp_spline = loco.inpaint_nans_cubic_spline(Z)
        fig.add_trace(go.Scatter(y=X_interp_spline[p,:], name=paws[p]+' Xraw_interp_spline', line=dict(color='green')))

        if path_retracked:
            X_retracked = final_tracks_retracked[0,:,:]*loco.pixel_to_mm
            Y_retracked = final_tracks_retracked[1,:,:]*loco.pixel_to_mm
            Z_retracked = -final_tracks_retracked[3,:,:]*loco.pixel_to_mm
            X_interp_retracked = loco_retracked.inpaint_nans(X_retracked)
            Y_interp_retracked = loco_retracked.inpaint_nans(Y_retracked)
            Z_interp_retracked = loco_retracked.inpaint_nans(Z_retracked)
        # VALIDATED: the manual tracking is correct and matches the DLC tracks, so we can use it to compare with the automatic tracking
        #fig_validation = go.Figure()
        #fig_validation.add_trace(go.Scatter(y=manual_track_FRbottom_validation[:,0]*loco.pixel_to_mm, name=paws[p]+' X bottom validation track', line=dict(color='green')))
        #fig_validation.add_trace(go.Scatter(y=X[p,:], name=paws[p]+' Xraw', line=dict(color='gray')))
        #fig_validation.show()


        if path_retracked:
            fig.add_trace(go.Scatter(y=X_retracked[p,:], name=paws[p]+' Xraw retracked', line=dict(color='orange')))
        

        
        if manual_track:
            # Add manually tracked signal
            fig.add_trace(go.Scatter(y=manual_track_FRbottom[:,0]*loco.pixel_to_mm, name=paws[p]+' X bottom manual track', line=dict(color='blue')))
            X_manual_corrected = X[p,:].copy()
            ind_manual_track = np.where(~np.isnan(manual_track_FRbottom[:,0]))[0]
            X_manual_corrected[ind_manual_track] = manual_track_FRbottom[ind_manual_track,0]*loco.pixel_to_mm
            X_interp_manual_corrected = loco.inpaint_nans(X_manual_corrected)
            fig.add_trace(go.Scatter(y=X_interp_manual_corrected, name=paws[p]+' X bottom manual track corrected', line=dict(color='purple')))
        else:
            X_interp_manual_corrected = X_interp[p,:]
        
        ### Method 1: Savgol filter, window 5, order 1 - used in compute_trajectories
        fig.add_trace(go.Scatter(y=savgol_filter(X_interp[p,:],window_length = 5, polyorder = 1), name='X_filtered ord1', line=dict(color='#FFB6C1')))
        error_method1 = compute_rmse(X_interp_manual_corrected, savgol_filter(X_interp[p,:],window_length = 5, polyorder = 1))
        print("RMSE for method 1 (savgol filter, window 5, order 1):", error_method1)

        ### Method 2: Savgol filter, window 11, order 1 - used in sw/st detection
        fig.add_trace(go.Scatter(y=savgol_filter(X[p,:], window_length = 11, polyorder = 1), name='X_filtered_swst ord1', line=dict(color='red')))
        error_method2 = compute_rmse(X_interp_manual_corrected, savgol_filter(X[p,:], window_length = 11, polyorder = 1))
        print("RMSE for method 2 (savgol filter, window 11, order 1):", error_method2)

        ### For retracked data, check if the filtering is better or worse
        if path_retracked:
            fig.add_trace(go.Scatter(y=savgol_filter(X_retracked[p,:], window_length = 11, polyorder = 1), name='X_retracked_filtered_swst ord1', line=dict(color='pink')))
            #error_method2 = compute_rmse(X_interp_manual_corrected, savgol_filter(X[p,:], window_length = 11, polyorder = 1))
            #print("RMSE for method 2 (savgol filter, window 11, order 1):", error_method2)

        # Add sw and st points
        data_filt = savgol_filter(X[p,:], window_length = 11, polyorder = 1)
        peaks = find_peaks(data_filt)
        throughs = find_peaks(-data_filt)
        stance = peaks[0]
        swing = throughs[0]
        fig.add_trace(go.Scatter(x=stance, y=data_filt[stance], mode='markers', name='Stance Onset', marker=dict(symbol='circle-open',color='orange', size=5)))
        fig.add_trace(go.Scatter(x=swing, y=data_filt[swing], mode='markers', name='Swing Onset', marker=dict(symbol='circle-open',color='green', size=5)))

        # Add sw and st points from strides matrices
        if st_strides_mat is not None and sw_pts_mat is not None:
            st_strides = np.int64(st_strides_mat[p][:,0,-1])  # Last column is the frame number
            sw_pts = np.int64(sw_pts_mat[p][:,0,-1])
            fig.add_trace(go.Scatter(x=st_strides, y=data_filt[st_strides], mode='markers', name='Stance Onset after exclusion', marker=dict(color='orange', size=5)))
            fig.add_trace(go.Scatter(x=sw_pts, y=data_filt[sw_pts], mode='markers', name='Swing Onset after exclusion', marker=dict(color='green', size=5)))
  
        # Add sw and st points with cubic spline filtering
        data_filt = savgol_filter(X_interp_spline[p,:], window_length = 11, polyorder = 1)
        peaks = find_peaks(data_filt)
        throughs = find_peaks(-data_filt)
        stance = peaks[0]
        swing = throughs[0]
        fig.add_trace(go.Scatter(x=stance, y=data_filt[stance], mode='markers', name='Stance Onset', marker=dict(symbol='star-open',color='orange', size=10)))
        fig.add_trace(go.Scatter(x=swing, y=data_filt[swing], mode='markers', name='Swing Onset', marker=dict(symbol='star-open',color='green', size=10)))

        ### Method 3: Savgol filter, window 21, order 1 - smooths and delays too much
        fig.add_trace(go.Scatter(y=savgol_filter(X_interp[p,:],window_length = 50, polyorder = 3), name='X_filtered_swst w50 ord3', line=dict(color='lightblue')))
        error_method3 = compute_rmse(X_interp_manual_corrected, savgol_filter(X_interp[p,:],window_length = 50, polyorder = 3))
        print("RMSE for method 3 (savgol filter, window 50, order 3):", error_method3)


        ### Method 4: Median filter, kernel size 5 - not so different from savgol filter
        fig.add_trace(go.Scatter(y=medfilt(X_interp[p,:],kernel_size = 5), name='X_filtered_medianfil', line=dict(color='darkblue')))
        error_method4 = compute_rmse(X_interp_manual_corrected, medfilt(X_interp[p,:],kernel_size = 5))
        print("RMSE for method 4 (median filter, kernel size 5):", error_method4)


        ### Method 5: outlier detection and interpolation with cubic spline
        # Velocity and acceleration signals
        x_velocity_raw, x_acceleration_raw, x_without_outliers, corrected_x = correct_signal_cubic_spline(X_interp[p,:], dt, thr_sig='acceleration')
        y_velocity_raw, y_acceleration_raw, y_without_outliers, corrected_y = correct_signal_cubic_spline(Y_interp[p,:], dt, thr_sig='acceleration')
        z_velocity_raw, z_acceleration_raw, z_without_outliers, corrected_z = correct_signal_cubic_spline(Z_interp[p,:], dt, thr_sig='acceleration')

        fig.add_trace(go.Scatter(y=x_without_outliers, name=paws[p]+' Xraw with removed outliers velocity/acceleration', line=dict(color='salmon')))
        error_method5_without_outliers = compute_rmse(X_interp_manual_corrected, x_without_outliers)
        print("RMSE for method 5 (removal of outliers):", error_method5_without_outliers)
        num_outliers = np.sum(np.isnan(x_without_outliers))
        print("Number of outliers (NaNs) in x_without_outliers:", num_outliers)
        fig.add_trace(go.Scatter(y=corrected_x, name=paws[p]+' X_corrected spline', line=dict(color='#8B0000')))
        error_method5 = compute_rmse(X_interp_manual_corrected[100:], corrected_x[100:])
        print("RMSE for method 5 (cubic spline interpolation):", error_method5)

        # Add also the raw signals for velocity and acceleration used for filtering and correcting
        fig.add_trace(go.Scatter(y=x_velocity_raw, name=paws[p]+' X velocity raw', line=dict(color='green')), secondary_y=True)
        fig.add_trace(go.Scatter(y=x_acceleration_raw, name=paws[p]+' X acceleration raw', line=dict(color='lightgreen')), secondary_y=True)  


        # Update layout
        fig.update_layout(
            title=animal + ' trial '+str(count_trial)+' - '+paws[p]+' Tracking X Signals'+' confidence '+str(confidence),
            xaxis_title='Frame',
            yaxis_title='Position (mm)',
            #xaxis=dict(range=[13100, 14000]),
            yaxis=dict(range=[0, 250]),
           # yaxis2=dict(range=[-5, 5]),
            showlegend=True
        )
        
        fig.show()

        # Save as HTML
        fig.write_html(os.path.join(loco.path, animal+'_'+paws[p]+'_trial'+str(count_trial)+'_tracking_Xsignals.html'))


        ######## Plot relative signals
        fig_rel = make_subplots(specs=[[{"secondary_y": True}]])
        fig_rel.add_trace(go.Scatter(y=X_interp[p,:]-np.nanmean(X_interp[:4,:],axis=0), 
                            name=paws[p]+' Xraw_interp rel', line=dict(color='darkgray')))
        
        # Plot velocity signals from the corrected signal (X, X-Y, X-Y-Z) to check for the best to detect stance and swing)
        x_velocity_raw_rel, x_acceleration_raw_rel, x_without_outliers_rel, corrected_x_rel = correct_signal_cubic_spline(X_interp[p,:] - np.nanmean(X_interp[:4,:],axis=0), dt, thr_sig='acceleration')
        y_velocity_raw_rel, y_acceleration_raw_rel, y_without_outliers_rel, corrected_y_rel = correct_signal_cubic_spline(Y_interp[p,:] - np.nanmean(Y_interp[:4,:],axis=0), dt, thr_sig='acceleration')
        z_velocity_raw_rel, z_acceleration_raw_rel, z_without_outliers_rel, corrected_z_rel = correct_signal_cubic_spline(Z_interp[p,:] - np.nanmean(Z_interp[:4,:],axis=0), dt, thr_sig='acceleration')
        
        # Calculate velocity
        corrected_filtered_x = savgol_filter(corrected_x_rel, window_length = 11, polyorder = 1)
        corrected_filtered_y = savgol_filter(corrected_y_rel, window_length = 11, polyorder = 1)   
        corrected_filtered_z = savgol_filter(corrected_z_rel, window_length = 11, polyorder = 1)
        
        velocity_corrected_filtered_x = np.diff(corrected_filtered_x, prepend=corrected_filtered_x[0])
        velocity_corrected_filtered_y = np.diff(corrected_filtered_y, prepend=corrected_filtered_y[0])
        velocity_corrected_filtered_z = np.diff(corrected_filtered_z, prepend=corrected_filtered_z[0])


        # Add right side axes
        fig_rel.add_trace(go.Scatter(y=corrected_filtered_x, name='X_corrected spline and filtered rel'))
        fig_rel.add_trace(go.Scatter(y=velocity_corrected_filtered_x, name=paws[p]+' X velocity', line=dict(color='purple')), secondary_y=True)
        fig_rel.add_trace(go.Scatter(y=corrected_filtered_z, name='Z_corrected spline and filtered rel', line=dict(color='red')))
        fig_rel.add_trace(go.Scatter(y=velocity_corrected_filtered_z, name=paws[p]+' Z velocity', line=dict(color='green')), secondary_y=True)
        fig_rel.update_yaxes(title_text="Velocity (mm/s)", secondary_y=True)
        # Add the vector of X-Z velocity
        velocity_corrected_filtered_xz = np.sqrt(velocity_corrected_filtered_x**2 + velocity_corrected_filtered_z**2)               #np.sign(np.arctan2(velocity_corrected_filtered_x,velocity_corrected_filtered_z))*
        fig_rel.add_trace(go.Scatter(y=velocity_corrected_filtered_xz, name=paws[p]+' XZ velocity', line=dict(color='darkblue')), secondary_y=True)
        # Add the vector of X-Y-Z velocity
        velocity_corrected_filtered_xyz = np.sqrt(velocity_corrected_filtered_x**2 + velocity_corrected_filtered_y**2 + velocity_corrected_filtered_z**2)
        fig_rel.add_trace(go.Scatter(y=velocity_corrected_filtered_xyz, name=paws[p]+' XYZ velocity', line=dict(color='orange')), secondary_y=True)
        

       # fig_rel.add_trace(go.Scatter(y=Z_interp[p,:]-np.nanmean(Z_interp[:4,:],axis=0), name='Z_raw'))
       # fig_rel.add_trace(go.Scatter(y=corrected_z, name='Z_corrected spline', line=dict(color='black')))
        
        #fig_rel.add_trace(go.Scatter(y=savgol_filter(Z_interp[p,:],window_length = 5, polyorder = 1)-np.nanmean(Z_interp[:4,:],axis=0), name='Z_filtered'))
        fig_rel.add_trace(go.Scatter(x=stance, y=corrected_filtered_x[stance], mode='markers', name='Stance Onset', marker=dict(symbol='circle-open',color='orange', size=5)))
        fig_rel.add_trace(go.Scatter(x=swing, y=corrected_filtered_x[swing], mode='markers', name='Swing Onset', marker=dict(symbol='circle-open',color='green', size=5)))
        fig_rel.update_layout(
            title=animal + ' trial '+str(count_trial)+' - '+paws[p]+' Relative position and velocity signals',
            xaxis=dict(range=[13100, 14000], title='Frame'),
            yaxis=dict(range=[-10, 40], title="Position (mm)"),
            yaxis2=dict(range=[-4, 4], title="Velocity (mm/s)"),
            showlegend=True
        )

        fig_rel.show()
        # Save as HTML
        #fig_rel.write_html(os.path.join(loco.path, animal+'_'+paws[p]+'_trial'+str(count_trial)+'_rel_signals.html'))

        fig2 = go.Figure()
        for s in range(5,6):
            fig2.add_trace(go.Scatter(x=corrected_filtered_x[sw_pts[s]:sw_pts[s+1]], y=corrected_filtered_z[sw_pts[s]:sw_pts[s+1]], name=paws[p]+' XZ trajectory '+str(s), line=dict(color='red')))
            # Add velocity vectors at each time point along the trajectory
            for t in range(sw_pts[s], sw_pts[s+1]):
                fig2.add_trace(go.Scatter(
                    x=[corrected_filtered_x[t], corrected_filtered_x[t] +  velocity_corrected_filtered_x[t]],
                    y=[corrected_filtered_z[t], corrected_filtered_z[t] +  velocity_corrected_filtered_z[t]],
                    mode='lines',
                    line=dict(color='blue'),
                ))
                fig2.add_trace(go.Scatter(
                    x=[corrected_filtered_x[t], corrected_filtered_x[t] + velocity_corrected_filtered_x[t]],
                    y=[corrected_filtered_z[t], corrected_filtered_z[t]],
                    mode='lines',
                    line=dict(color='purple'),
                ))
                fig2.add_trace(go.Scatter(
                    x=[corrected_filtered_x[t], corrected_filtered_x[t]],
                    y=[corrected_filtered_z[t], corrected_filtered_z[t] + velocity_corrected_filtered_z[t]],
                    mode='lines',
                    line=dict(color='green'),
                ))
        fig_rel.update_layout(
            title=animal + ' trial '+str(count_trial)+' - '+paws[p]+' Trajectories with Velocity Vectors',
            xaxis_title='X Position (mm)',
            yaxis_title='Z Position (mm)',
            showlegend=True
        )
        fig2.show()
        # Save as HTML
        #fig2.write_html(os.path.join(loco.path, animal+'_'+paws[p]+'_trial'+str(count_trial)+'_trajectories.html'))



        # Plot each stride separately    - TO VERIFY CONSISTENCY WITH kinematic_analysis.py code
