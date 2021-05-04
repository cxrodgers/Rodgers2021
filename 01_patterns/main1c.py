## Generate transformation_df and consensus_edge_summary
# Looks like the concave shapes are slightly different for 228 and 231
# I fixed by careful thresholding
# TODO: move main1b and main1c to pipeline_prep, and hardcode these outputs

"""
Procedure
* Uses scaling_df to generate transformation_df for each session
  This forces everything to a fixed scale of 60px per servo position
  And puts everything in a consensus orientation (now: nose-up)
* Uses transformation_df to transform each edge_summary
  Extracts a 2d pixel window around the origin
* Aggregates edge_summary over sessions
  Binarizes and dilates
  Thresholds the mean over sessions
  This becomes consensus_edge_summary

Dumps in params['scaling_dir']:
  transformation_df
  consensus_edge_summary
"""

import json
import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas
import scipy.ndimage
import whiskvid
import MCwatch.behavior
import runner.models
import my, my.plot


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    

## Parameters
# Force consensus image to this scale
FORCE_SCALE = 60.0

# Extract this many pixels around the origin for the consensus image
EXTRACT_XMIN = -300
EXTRACT_XMAX = 300
EXTRACT_YMIN = -200
EXTRACT_YMAX = 400


## Behavioral datasets
gs_qs = runner.models.GrandSession.objects.filter(
    tags__name=params['decoding_tag'])
session_name_l = sorted(gs_qs.values_list('name', flat=True))


## Load data
scaling_df = pandas.read_pickle(
    os.path.join(params['scaling_dir'], 'scaling_df'))
keypoints_df = pandas.read_pickle(
    os.path.join(params['scaling_dir'], 'keypoints_df'))


## Extract transformations for each session
rec_l = []
for session_name in scaling_df.index:
    ## Rotation
    # Get travel direction in degrees
    # Positive is CCW in the image
    travel_direction = scaling_df.loc[session_name, 'theta']
    
    # Rotate by -theta to make the travel direction horizontal (left to right)
    # However, also take a negative here, to account for the fact that
    # CCW in the image will be CW in mathematical plane
    # So these negatives cancel out
    # Positive rotation_angle rotates the image CW
    rotation_angle = travel_direction * np.pi / 180
    
    # Rotate by additional 180 deg to get the image in "nose up" position
    rotation_angle += np.pi
    
    # Construct the rotation matrix
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)], 
        [np.sin(rotation_angle), np.cos(rotation_angle)]
        ])

    
    ## Scaling
    # Scale to 60.0 px / position
    scaling_value = FORCE_SCALE / scaling_df.loc[session_name, 'scale']
    scaling_matrix = np.eye(2) * scaling_value
    
    
    ## Shifting
    # This will shift the identified "origin" in the frame to (0, 0)
    shift_vector = -scaling_df.loc[session_name, ['origin_x', 'origin_y']].values
    
    
    ## Store
    rec_l.append({
        'session': session_name,
        'shift_x': shift_vector[0],
        'shift_y': shift_vector[1],
        'c00': rotation_matrix[0, 0],
        'c01': rotation_matrix[0, 1],
        'c10': rotation_matrix[1, 0],
        'c11': rotation_matrix[1, 1],
        })

# Concat
transformation_df = pandas.DataFrame.from_records(rec_l).set_index(
    'session').sort_index()


## Error check that origin transforms to zero
transformed_origin = my.misc.transform(
    scaling_df[['origin_x', 'origin_y']], transformation_df)
assert np.allclose(transformed_origin, np.zeros_like(transformed_origin))


## Transform the keypoints
transformed_keypoint_l = []
for keypoint in keypoints_df.columns.levels[0]:
    this_keypoint = keypoints_df[keypoint].rename(columns={'row': 'y', 'col': 'x'})
    transformed_keypoint = my.misc.transform(this_keypoint, transformation_df)
    transformed_keypoint_l.append(transformed_keypoint)

transformed_keypoints = pandas.concat(
    transformed_keypoint_l,
    keys=keypoints_df.columns.levels[0], axis=1)


## Build a consensus geometry in the warped space
# Warp the edge summaries for each session
# Build intersections
warped_es_df_keys_l = []
warped_es_df_l = []
for session_name in tqdm.tqdm(session_name_l):
    ## Load data from this session
    vs = whiskvid.django_db.VideoSession.from_name(session_name)
    trial_matrix = pandas.read_pickle(os.path.join(vs.session_path, 'trial_matrix'))
    
    
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

    
    ## Extract scaling params
    origin = scaling_df.loc[session_name, ['origin_x', 'origin_y']].rename(
        {'origin_x': 'col', 'origin_y': 'row'})
    theta = scaling_df.loc[session_name, 'theta']
    scale = scaling_df.loc[session_name, 'scale']


    ## Warp each edge summary
    warped_es_keys_l = []
    warped_es_l = []
    for (servo_pos, stepper_pos), this_bnes in binary_norm_es.groupby(
            ['servo_pos', 'stepper_pos']):
        
        ## Reindex the image into real pixels
        this_bnes = this_bnes.droplevel(['servo_pos', 'stepper_pos'])
        img0 = this_bnes.reindex(
            range(vs.frame_height)).interpolate(
            limit_direction='both').reindex(
            range(vs.frame_width), axis=1).interpolate(
            axis=1, limit_direction='both').values

        
        ## Translate so that origin is centered
        # TODO: Pad before translating so that data doesn't get shifted out of bounds
        # No wrapping, so data can get shifted out of bounds here
        shift_x = img0.shape[1] / 2.0 - origin['col']
        shift_y = img0.shape[0] / 2.0 - origin['row']
        img1 = scipy.ndimage.shift(img0, 
            np.array([shift_y, shift_x]),
            )
        
        # Rotate so that angle becomes zero (travel direction from left to right)
        # And then another 180 to get it in nose-up position
        rotation_angle = -theta + 180
        img2 = scipy.ndimage.rotate(img1, angle=rotation_angle)
        
        # Scale to 60px
        zoom_ratio = FORCE_SCALE / scale
        img3 = scipy.ndimage.zoom(img2, zoom_ratio)
        
        # Extract a consistently sized image without recentering
        img4 = pandas.DataFrame(img3)
        sz0 = img4.shape
        img4 = img4.reindex(
            range((sz0[0] // 2) + EXTRACT_YMIN, (sz0[0] // 2) + EXTRACT_YMAX),
            ).reindex(
            range((sz0[1] // 2) + EXTRACT_XMIN, (sz0[1] // 2) + EXTRACT_XMAX),
            axis=1).fillna(0)
        img4 = img4.values    
        
        # Threshold because binary became float
        img4 = (img4 > 0.5).astype(np.int)

        # DataFrame
        warped_es = pandas.DataFrame(
            img4,
            index=range(EXTRACT_YMIN, EXTRACT_YMAX),
            columns=range(EXTRACT_XMIN, EXTRACT_XMAX),
        )
        warped_es.index.name = 'row'
        warped_es.columns.name = 'col'

        # Store
        warped_es_l.append(warped_es)
        warped_es_keys_l.append((servo_pos, stepper_pos))

    
    ## Concat
    warped_es_df = pandas.concat(warped_es_l, keys=warped_es_keys_l, 
        names=['servo_pos', 'stepper_pos'])
    
    
    #~ ## Debug
    #~ f, ax = plt.subplots()
    #~ topl = warped_es_df.max(level='row')
    #~ my.plot.imshow(
        #~ topl,
        #~ xd_range=(topl.columns[0], topl.columns[-1]),
        #~ yd_range=(topl.index[0], topl.index[-1]),
        #~ ax=ax,
        #~ axis_call='image',
        #~ )
    
    #~ # Plot the keypoints
    #~ session_keypoints = transformed_keypoints.loc[session_name].unstack('coord')
    #~ ax.plot([0], [0], 'ko')    
    #~ ax.plot(
        #~ [session_keypoints.loc['keypoint_cv', 'x']],
        #~ [session_keypoints.loc['keypoint_cv', 'y']],
        #~ 'bo')
    #~ ax.plot(
        #~ [session_keypoints.loc['keypoint_x0', 'x']],
        #~ [session_keypoints.loc['keypoint_x0', 'y']],
        #~ 'go')    
    #~ ax.plot(
        #~ [session_keypoints.loc['keypoint_x1', 'x']],
        #~ [session_keypoints.loc['keypoint_x1', 'y']],
        #~ 'ro')    

    #~ plt.show()
    #~ 1/0
    
    
    
    ## Store
    warped_es_df_l.append(warped_es_df)
    warped_es_df_keys_l.append(session_name)


## Concat
big_warped = pandas.concat(
    warped_es_df_l, keys=warped_es_df_keys_l, names=['session'])


## Dilate each session * servo_pos * stepper_pos
grouping_keys = ['session', 'servo_pos', 'stepper_pos']
dilated_keys_l = []
dilated_l = []
for grouped_keys, this_warped in big_warped.groupby(grouping_keys):
    this_warped = this_warped.droplevel(grouping_keys)

    # Dilate
    dilated = scipy.ndimage.binary_dilation(this_warped, iterations=6).astype(np.int)

    # DataFrame
    dilated_df = pandas.DataFrame(
        dilated, 
        index=this_warped.index, 
        columns=this_warped.columns,
        )
    
    # Store
    dilated_l.append(dilated_df)
    dilated_keys_l.append(grouped_keys)

big_dilated = pandas.concat(dilated_l, keys=dilated_keys_l, names=grouping_keys)


## Threshold each servo_pos * stepper_pos over sessions
grouping_keys = ['servo_pos', 'stepper_pos']
thresholded_l = []
thresholded_keys_l = []
for grouped_keys, this_dilated in big_dilated.groupby(grouping_keys):
    this_dilated = this_dilated.droplevel(grouping_keys)
    
    # Mean over sessions
    meaned_dilated = this_dilated.mean(level='row')
    
    # Threshold
    if grouped_keys[0] == 50:
        # convex
        thresh = .4
    else:
        thresh = .35
    thresholded = (meaned_dilated > thresh).astype(np.int)

    # Gap fill
    gapfilled = scipy.ndimage.binary_dilation(thresholded, iterations=5)
    gapfilled = scipy.ndimage.binary_erosion(gapfilled, iterations=5)
    gapfilled = pandas.DataFrame(gapfilled.astype(np.int),
        index=thresholded.index, columns=thresholded.columns)

    # Store
    thresholded_l.append(gapfilled)
    thresholded_keys_l.append(grouped_keys)

# Concat
big_thresholded = pandas.concat(thresholded_l, keys=thresholded_keys_l, names=grouping_keys)


## Plot
grouping_keys = ['servo_pos', 'stepper_pos']
servo_pos_l = [1670, 1760, 1850]
stepper_pos_l = [50, 150]
f, axa = plt.subplots(len(stepper_pos_l), len(servo_pos_l), figsize=(12, 7.5))
for grouped_keys, this_thresholded in big_thresholded.groupby(grouping_keys):
    # Get ax
    ax = axa[
        stepper_pos_l.index(grouped_keys[1]), 
        servo_pos_l.index(grouped_keys[0]),
        ]

    # Plot
    im = my.plot.imshow(this_thresholded, ax=ax)
    ax.axis('image')
my.plot.harmonize_clim_in_subplots(fig=f)


## Max the consensus edge summary over sessions and visualize it
topl = big_thresholded.max(level='row')
f, ax = plt.subplots()
my.plot.imshow(
    topl,
    xd_range=(topl.columns[0], topl.columns[-1]),
    yd_range=(topl.index[0], topl.index[-1]),
    ax=ax,
    )

# Plot the keypoints
global_keypoints = transformed_keypoints.mean().unstack('coord')
ax.plot([0], [0], 'ko')    
ax.plot(
    [global_keypoints.loc['keypoint_cv', 'x']],
    [global_keypoints.loc['keypoint_cv', 'y']],
    'bo')
ax.plot(
    [global_keypoints.loc['keypoint_x0', 'x']],
    [global_keypoints.loc['keypoint_x0', 'y']],
    'go')    
ax.plot(
    [global_keypoints.loc['keypoint_x1', 'x']],
    [global_keypoints.loc['keypoint_x1', 'y']],
    'ro')  
plt.show()


## Dump
transformation_df.to_pickle(
    os.path.join(params['scaling_dir'], 'transformation_df'))
big_thresholded.to_pickle(
    os.path.join(params['scaling_dir'], 'consensus_edge_summary'))
