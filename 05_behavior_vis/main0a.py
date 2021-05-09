## Example whisking and contacts
"""
2B	
    PLOT_EXAMPLE_WHISKING_TRACE_180119_KM131_66	
    N/A	
    Whisking traces and contact epochs from example trial

3A, left	
    example_frame_burn_180119_KM131_137393	
    N/A	
    Image showing whisk with and without contact
"""

import json
import os
import imageio
import numpy as np
import pandas
import matplotlib
import matplotlib.pyplot as plt
import whiskvid
import my
import my.plot


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    

## Plotting params    
DEGREE = chr(176)
my.plot.manuscript_defaults()
my.plot.font_embed()


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)


## Load data
# patterns data
big_tm = my.dataload.load_data_from_patterns(params, 
    'big_tm')
big_C2_tip_whisk_cycles = my.dataload.load_data_from_patterns(params, 
    'big_C2_tip_whisk_cycles')
big_cycle_features = my.dataload.load_data_from_patterns(params, 
    'big_cycle_features')
big_touching_df = my.dataload.load_data_from_patterns(params, 
    'big_touching_df')
big_tip_pos = my.dataload.load_data_from_patterns(params, 
    'big_tip_pos')

# logreg features
features = my.dataload.load_data_from_logreg(
    params, 'unobliviated_unaggregated_features', mouse2task=None)


## Choose example session and load data
session_name = '180119_KM131'

# Get handles
vs = whiskvid.django_db.VideoSession.from_name(session_name)

# Slice out tip pos
tip_angle = big_tip_pos.loc[
    session_name, 'tip'].xs('angle', level='metric', axis=1)

# Load joints
joints = vs.data.joints.load_data()

# Frame shape
frame_height = session_df.loc[session_name, 'frame_height']
frame_width = session_df.loc[session_name, 'frame_width']


## Extract edges
es = vs.data.edge_summary.load_data()

# Drop the flatter (or nothing) ones
es = es.loc[pandas.IndexSlice[:, :, [50, 150]], :].copy()

# Drop the rewside level which is not useful because we have stepper_pos
es.index = es.index.droplevel('rewside')
es.index = es.index.remove_unused_levels()
assert not es.index.duplicated().any()

# Normalize to a max of 1.0
norm_es = es.unstack('row').divide(
    es.unstack('row').max(axis=1), axis=0).stack('row')

# Binarize
# This fattens the edges a little
binary_norm_es = (norm_es > .0001).astype(np.int)    

# Mean over stimuli
esumm = binary_norm_es.mean(level='row')

# Reindex at frame resolution
esumm = esumm.reindex(range(0, frame_height)).reindex(range(0, frame_width), axis=1)
esumm = esumm.interpolate(
    limit_direction='both').interpolate(
    axis=1, limit_direction='both')

# Rebinarize
esumm = (esumm > .0001).astype(np.float)

# normalizing object
# actually not necessary since we just binarized
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

# Build custom colormap
cmap = my.misc.CustomCmap(
    list(matplotlib.colors.to_rgb('white')),
    list(matplotlib.colors.to_rgb('pink')),
    )
    
# Normalize and convert to rgba
esumm_rgba = cmap(norm(esumm.values))

# Drop unnecessary alpha
esumm_rgb = esumm_rgba[:, :, :3]

# Nullify zeros in esumm 
esumm.values[esumm.values < .0001] = np.nan
 


## Plot flags
PLOT_EXAMPLE_WHISKING_TRACE = True
PLOT_EXAMPLE_FRAME = True


## Plots
if PLOT_EXAMPLE_WHISKING_TRACE:
    ## Choose an example trial
    example_trial = 101 # convex
    example_trial = 66 # concave
    
    """Here's how to choose example trials with contacts on each whisker:
    cb = features.loc[session_name]['contact_binarized'].sum(level='trial')
    trials_with_contact_mask = (cb[['C1', 'C2', 'C3']] > 0).all(1)
    sliced_tm = big_tm.loc[session_name].loc[trials_with_contact_mask.index[trials_with_contact_mask.values]]
    my.pick(sliced_tm, rewside='left')
    """

    for example_trial in [66]:

        ## Slice out whisking angle on this trial
        trial_ta = tip_angle.loc[example_trial].unstack('whisker')

        # Drop C0 throughout
        trial_ta = trial_ta.drop('C0', 1)

        # Slice out the info about this trial
        trial_bwc = big_C2_tip_whisk_cycles.loc[session_name].loc[example_trial]
        trial_bcf = big_cycle_features.loc[session_name].loc[example_trial]


        ## Plot handles
        # Careful, if it gets too narrow, it won't plot the short touches
        f, axa = plt.subplots(
            2, 1, figsize=(5.3, 2.25), 
            gridspec_kw={'height_ratios': (.15, 1)},
            sharex=True)
        f.subplots_adjust(left=.05, right=.93, bottom=.25, top=1, hspace=.1)


        ## Iterate over whiskers
        whisker2color = {'C0': 'magenta', 'C1': 'b', 'C2': 'g', 'C3': 'r'}
        whisker_yspacing = 30
        for n_whisker, whisker in enumerate(['C1', 'C2', 'C3']):
            # Demean
            topl = trial_ta[whisker].copy()
            topl = topl - topl.mean()
            
            # Plot the individual whisker
            color = whisker2color[whisker]
            axa[1].plot(
                topl.index.values / 200.,
                topl.values + n_whisker * whisker_yspacing,
                color=color, lw=1)
            
            # Label
            axa[1].text(
                1.03, n_whisker * whisker_yspacing, whisker, 
                color=color, ha='left', va='center', size=12, clip_on=False)

        
        ## Plot the touching
        whisker_l = ['C1', 'C2', 'C3']
        for n_whisker, whisker in enumerate(whisker_l):
            # Get touching
            whisker_touching = big_touching_df.loc[
                session_name].loc[example_trial].loc[whisker]
            
            # Get each contiguous segment of ones
            starts = np.where(np.diff(whisker_touching) == 1)[0]
            stops = np.where(np.diff(whisker_touching) == -1)[0]
            assert len(starts) == len(stops)
            assert (starts < stops).all()
            
            # Plot
            color = whisker2color[whisker]
            yval = n_whisker
            for start, stop in zip(starts, stops):
                # If this is half-open, check that the plotting works
                if stop - start == 0:
                    1/0
                
                # Plot touching bar
                axa[0].plot(
                    trial_ta.index.values[[start, stop]] / 200., 
                    [yval, yval], 
                    color=color, lw=3.5, clip_on=False)

        # Scale
        axa[0].set_ylim((-.5, len(whisker_l)))
        
        
        ## Dashed line between whisking and contacts
        axa[0].plot([-2, 1], [-1.3, -1.3], 'k--', clip_on=False, lw=.75)
        axa[0].text(-1, -.4, 'contacts:', ha='right', va='bottom')


        ## Pretty
        axa[0].axis('off')
        my.plot.despine(ax=axa[1], which=('left', 'top', 'right'))
        axa[1].set_yticks([])
        axa[1].set_xlim((-2, 1))
        axa[1].set_xticks((-2, -1, 0, 1))
        axa[1].set_xlabel('time in trial (s)')

        # Scale bars
        t_start = -1.8
        y_start = 75
        t_len = .2
        y_len = 20
        axa[1].plot(
            [t_start, t_start], [y_start, y_start + y_len], 
            'k-', lw=.8, clip_on=False)
        axa[1].text(
            t_start - .01, np.mean([y_start, y_start + y_len]) + y_len / 10, 
            '{}{}'.format(y_len, DEGREE), 
            ha='right', va='center', rotation=90, size=12)


        ## Save
        f.savefig('PLOT_EXAMPLE_WHISKING_TRACE_{}_{}.svg'.format(
            session_name, example_trial))
        f.savefig('PLOT_EXAMPLE_WHISKING_TRACE_{}_{}.png'.format(
            session_name, example_trial), dpi=300)



if PLOT_EXAMPLE_FRAME:
    ## Choose an example trial
    example_trial = 101 # convex
    #~ example_trial = 66 # concave

    if example_trial == 101:
        # This one works fine
        example_cycle = 35
        peak_frame_offset = 1
    elif example_trial == 78:
        # This one is an opto trial so should not be used
        example_cycle = 46
        peak_frame_offset = 0
    else:
        # This one is concave and works okay but C0 is distracting
        example_trial = 66
        example_cycle = 48
        peak_frame_offset = 0        


    ## Slice out whisking angle on this trial
    trial_ta = tip_angle.loc[example_trial].unstack('whisker')

    # Drop C0 throughout
    trial_ta = trial_ta.drop('C0', 1)

    # Slice out the info about this trial
    trial_bwc = big_C2_tip_whisk_cycles.loc[session_name].loc[example_trial]
    trial_bcf = big_cycle_features.loc[session_name].loc[example_trial]

    
    ## Choose cycles to plot
    anti_contact_count = features.loc[session_name].loc[example_trial].loc[:, 'anti_contact_count']
    contact_binarized = features.loc[session_name].loc[example_trial].loc[:, 'contact_binarized']

    cycles_with_contacts = contact_binarized.index[contact_binarized.sum(1) > 0]
    cycles_with_anticontacts = anti_contact_count.index[anti_contact_count.sum(1) > 0]


    ## Choose cycle and get times and frames
    # Get frame numbers for this cycle
    start_frame = trial_bwc.loc[example_cycle, 'start_frame']
    stop_frame = trial_bwc.loc[example_cycle, 'stop_frame']
    peak_frame = trial_bwc.loc[example_cycle, 'peak_frame']
    t_wrt_rwin = (peak_frame - big_tm.loc[session_name].loc[
        example_trial, 'rwin_frame']) / 200.

    # the plot frame may not be the peak frame exactly
    plot_frame = peak_frame + peak_frame_offset

    # Get example frame
    frame = imageio.imread(os.path.join(
        params['example_frames_dir'], 
        '{}_{}.png'.format(session_name, plot_frame)))

    # Burn the edge summary into the frame
    burn_factor = 0.8
    burned_frame = np.array([frame] * 3).swapaxes(0, 2).swapaxes(0, 1) / 255.0
    burned_frame[~esumm.isnull().values] = (
        burn_factor * esumm_rgb[~esumm.isnull().values] +
        (1 - burn_factor) * burned_frame[~esumm.isnull().values]
        )    


    ## Plot
    for plot_meth in ['burn', 'no_burn']:
        f, ax = plt.subplots(figsize=(0.5 * np.array([6.4, 5.5])), dpi=100)
        f.subplots_adjust(left=0, right=1, bottom=0, top=1)

        if plot_meth == 'burn':
            # Plot frame
            my.plot.imshow(burned_frame, cmap=plt.cm.gray, clim=(0, 1), ax=ax,
                xd_range=(0, frame_width),
                yd_range=(0, frame_height),    
                axis_call='image',
                )
        else:
            # Plot frame
            my.plot.imshow(frame, cmap=plt.cm.gray, clim=(0, 255), ax=ax,
                xd_range=(0, frame_width),
                yd_range=(0, frame_height),    
                axis_call='image',
                )        
        
        # Plot contacts
        for whisker, in_contact in contact_binarized.loc[example_cycle].iteritems():
            if in_contact > 0:
                ax.plot(
                    [trial_bcf.loc[example_cycle, 'peak_tip_x'].loc[whisker]],
                    [trial_bcf.loc[example_cycle, 'peak_tip_y'].loc[whisker]],
                    marker='o', mec='yellow', mfc='yellow'
                )
        
        if plot_meth == 'burn':
            # Plot anti_contacts
            for whisker, in_contact in anti_contact_count.loc[example_cycle].iteritems():
                if in_contact > 0:
                    ax.plot(
                        [trial_bcf.loc[example_cycle, 'peak_tip_x'].loc[whisker]],
                        [trial_bcf.loc[example_cycle, 'peak_tip_y'].loc[whisker]],
                        marker='o', mec='yellow', mfc='none'
                    )        
        
        # Plot the whiskers
        whisker2color = {'C0': 'white', 'C1': 'b', 'C2': 'g', 'C3': 'r'}
        
        frame_joints = joints.loc[plot_frame]
        for whisker in ['C0', 'C1', 'C2', 'C3']:
            color = whisker2color[whisker]
            ax.plot(
                frame_joints.loc[whisker, 'c'],
                frame_joints.loc[whisker, 'r'],
                color=color, lw=1.5)
        
        # Pretty
        ax.axis('image')
        ax.set_xlim((frame_width, 0))
        ax.set_ylim((0, frame_height))
        ax.set_frame_on(False)

        f.savefig(
            'example_frame_{}_{}_{}.png'.format(
            plot_meth, session_name, plot_frame), dpi=200)

plt.show()