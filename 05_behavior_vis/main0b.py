## Plot the example frames (Fig 1B)
"""
1B, left
    example_3contact_frame_with_edge_180221_KF132_242546.png	
    Image showing tracked whiskers in contact with concave shape

1B, right	
    example_3contact_frame_with_edge_180221_KF132_490102.png	
    Image showing tracked whiskers in contact with concave shape
"""

import os
import json
import imageio
import pandas
import numpy as np
import matplotlib.pyplot as plt
import whiskvid
import my
import my.plot


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)

    
## Example session and frames
session_name = '180221_KF132'
convex_frame = 490102
concave_frame = 242546


## Get handles
vs = whiskvid.django_db.VideoSession.from_name(session_name)

# Frame shape
frame_height = session_df.loc[session_name, 'frame_height']
frame_width = session_df.loc[session_name, 'frame_width']


## Load joints for plotting whiskers
joints = vs.data.joints.load_data()


## Load edges for plotting
edges = vs.data.all_edges.load_data()

# Choose example edges at the further distances
trial_matrix = vs.data.trial_matrix.load_data().dropna()
shape2rwin_frame = trial_matrix.groupby(['rewside', 'servo_pos'])[
    'rwin_frame'].last().astype(np.int)


## Truncation
TRUNCATE_LATERAL = 15
shape2truncate_medial = shape2rwin_frame * 0 + 1
shape2truncate_medial.loc[('left', 1850)] = 10
shape2truncate_medial.loc[('left', 1760)] = 30
shape2truncate_medial.loc[('left', 1670)] = 35


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
    
    
## Plot each example frame
# The pixel size of the image is always the same
# But as DPI increases, the image is rendered smaller, while the lines stay
# the same widths
DPI = 500 
for rewside in ['left', 'right']:
    # Get the appropriate frame
    if rewside == 'left':
        frame_number = concave_frame
        other_rewside = 'right'
    else:
        frame_number = convex_frame
        other_rewside = 'left'

    # Get the frame
    frame = imageio.imread(os.path.join(
        params['example_frames_dir'], 
        '{}_{}.png'.format(session_name, frame_number)))

    # Create a figure with a single axis filling it
    figsize = (frame_width / float(DPI), frame_height / float(DPI))
    f = plt.figure(frameon=False, figsize=figsize)
    ax = f.add_subplot(position=[0, 0, 1, 1])
    ax.set_frame_on(False)
    
    # Display image
    im = my.plot.imshow(frame, ax=ax, cmap=plt.cm.gray, interpolation='bilinear')
    im.set_clim((0, 255))

    # Plot the edge at this time
    TRUNCATE_MEDIAL = shape2truncate_medial.loc[rewside].loc[1850]
    edge = edges[frame_number][TRUNCATE_LATERAL:-TRUNCATE_MEDIAL]
    ax.plot(edge[:, 1], edge[:, 0], color='pink', lw=1)
    
    # Plot the example edges
    example_edge_frames = shape2rwin_frame.loc[rewside].loc[[1670, 1760]]
    for servo_pos in example_edge_frames.index:
        example_edge_frame = example_edge_frames.loc[servo_pos]
        TRUNCATE_MEDIAL = shape2truncate_medial.loc[rewside].loc[servo_pos]
        edge = edges[example_edge_frame][TRUNCATE_LATERAL:-TRUNCATE_MEDIAL]
        ax.plot(edge[:, 1], edge[:, 0], color='cyan', lw=1)
        
    # Plot whiskers
    for whisker in ['C1', 'C2', 'C3']:
        color = {'C1': 'b', 'C2': 'g', 'C3': 'r'}[whisker]
        
        # Extract and plot joints
        try:
            whisker_joints = joints.loc[frame_number].loc[whisker].unstack().T
        except KeyError:
            continue
        ax.plot(whisker_joints['c'], whisker_joints['r'], color=color, lw=1)

        # Plot a yellow dot on the end for contact (since these were selected for
        # having a contact, although we may be off by a frame)
        ax.plot(
            whisker_joints['c'][0:1], whisker_joints['r'][0:1], 
            color='yellow', marker='o', ms=4)

    # Rotate into standard orientation
    ax.axis('image')
    ax.set_xlim((frame_width, 0))
    ax.set_ylim((0, frame_height))
    
    
    # Save
    f.savefig(
        'example_3contact_frame_with_edge_{}_{}.png'.format(
        session_name, frame_number), dpi=DPI)


plt.show()