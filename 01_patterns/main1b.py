## main1b,c : normalizing shape space to consensus space
# main1b: generates scaling_df and keypoints_df
# TODO: replace this with hardcoded data files
# Note that scaling_df is used to convert px to mm for kappa

"""
Procedure
* Binarize the edge_summary
* Define travel_vector as the best translation between adjacent pairs of CC/CV
  scale and theta are the length and angle of this vector
* Find keypoint_cv as the closest point on the closest CV along travel_vector
  This depends on travel_vector
* Find keypoint_x0 and keypoint_x1 as medial intersections between CC/CV
  (Would be nice to have a lateral keypoint but these aren't always visible)
  This does not depend on travel_vector
* Define origin as the mean of all keypoints
* Now we can use theta to rotate the image, scale to zoom the image, and
  origin to recenter the image.

Dumps in params['scaling_dir']:
  scaling_df 
  keypoints_df
"""
import json
import os
import tqdm
import pandas
import numpy as np
import skimage.feature
import scipy.ndimage
import matplotlib.pyplot as plt
import whiskvid
import MCwatch.behavior
import runner.models
import my
import my.plot


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)

# Create if doesn't exist
if not os.path.exists(params['scaling_dir']):
    os.mkdir(params['scaling_dir'])


## This is the spatial downsampling of the edge_summary (checked below)
ES_INTERVAL = 2


## Behavioral datasets
gs_qs = runner.models.GrandSession.objects.filter(
    tags__name=params['decoding_tag'])
session_name_l = sorted(list(gs_qs.values_list('name', flat=True)))


## Iterate over sessions
scaling_res_l = []
normalized_image_l = []
keypoints_l = []
for session_name in tqdm.tqdm(session_name_l):
    ## Load data from this session
    vs = whiskvid.django_db.VideoSession.from_name(session_name)
    trial_matrix = vs.data.trial_matrix.load_data()
    
    
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

    # Assert that this is every other pixel
    bne_meanstim = binary_norm_es.mean(level='row')
    idxdiffs = np.unique(np.diff(bne_meanstim.index.values))
    coldiffs = np.unique(np.diff(bne_meanstim.columns.values))
    assert len(idxdiffs) == 1
    assert len(coldiffs) == 1
    assert idxdiffs[0] == ES_INTERVAL
    assert coldiffs[0] == ES_INTERVAL
    
    
    ## Identify the travel direction in the image
    # Do this by correlating pairs of convex and concave at each distance
    # The use of a convex/concave pair makes the solution more unique
    rec_l = []
    srvpos_l = [1670, 1760, 1850]
    for n_srvpos, srvpos in enumerate(srvpos_l[:-1]):
        # Extract data for this (further) srvpos
        H = binary_norm_es.loc[srvpos].loc[[50, 150]].mean(level='row').values
        
        # Extract data for the next closer srvpos
        next_srvpos = srvpos_l[n_srvpos + 1]
        H_next = binary_norm_es.loc[next_srvpos].loc[[50, 150]].mean(level='row').values
        
        # Find displacement from H to H_next
        displacement = my.misc.find_image_shift(H, H_next)

        # Store the results
        rec_l.append({
            'srvpos': srvpos,
            'dr': displacement[0],
            'dc': displacement[1],
        })
    
    
    # Concat the results across comparisons
    displacement_df = pandas.DataFrame.from_records(rec_l).set_index(
        ['srvpos']).sort_index()
    
    # Account for ES_INTERVAL to get this into pixels in the original frame
    displacement_df = displacement_df * ES_INTERVAL
    
    # Mean the results over comparisons
    travel_vector = displacement_df.mean().reindex(['dr', 'dc'])


    ## Parameterize the image based on this travel_vector
    # Scale is the length of it, which we should know in mm
    scale = np.sqrt((travel_vector ** 2).sum())

    # Rotation is the angle of the vector traveling from further to closer
    # invert the row index to convert it into "y-coordinates"
    theta = np.arctan2(-travel_vector['dr'], travel_vector['dc']) * 180 / np.pi


    ## Define a consistent location on the shape and call that the origin
    # Could do the closest point, but that might depend on accuracy of this
    # travel vector
    # Could also do some intersection between CC and CV
    
    # Get this edge
    closest_cv = binary_norm_es.loc[1850].loc[50]
    
    # The proximity vector is 45 degrees CCW of the travel vector
    # Negative is CCW in the image (CW in the mathematical plane)
    beta = -45 / 180. * np.pi
    rot = np.array([
        [np.cos(beta), -np.sin(beta)], 
        [np.sin(beta), np.cos(beta)]
        ])
    prox_vector = pandas.Series(
        np.dot(rot, travel_vector.reindex(['dc', 'dr'])), # x, y order
        index=['dc', 'dr']).reindex(['dr', 'dc'])
    
    # Find the point along this edge that is the furthest along the travel vector
    closest_cv_pts = closest_cv.stack()
    closest_cv_pts = closest_cv_pts[closest_cv_pts > 0]
    closest_cv_pts = closest_cv_pts.index.to_frame().reset_index(drop=True)
    
    # Project
    projected = (closest_cv_pts * 
        prox_vector.rename({'dr': 'row', 'dc': 'col'})).sum(1).sort_values()
    
    # Max (the units are real pixels, not indices into norm_es)
    keypoint_cv = closest_cv_pts.loc[projected.idxmax()]
    

    ## Finally find some keypoints defined by intersections between cc and cv
    # Only use the medial intersections because the lateral intersections aren't
    # always visible
    es_cv1760 = norm_es.loc[1850].loc[50]
    es_cc1850 = norm_es.loc[1850].loc[150]
    summed = es_cv1760 + es_cc1850

    # Truncate image to keep only medial portion to make peak unambiguous
    start_col = int(keypoint_cv['col'] - 100)
    summed.loc[:, :start_col] = 0
    
    # Smooth
    smoothed = scipy.ndimage.gaussian_filter(summed, sigma=2)
    
    # Find peaks
    coordinates = skimage.feature.peak_local_max(
        smoothed, min_distance=10, num_peaks=1)
    coord = coordinates[0]
    
    # Convert to real pixel values
    keypoint_x0 = pandas.Series([
        summed.index[coord[0]],
        summed.columns[coord[1]],
        ], index=['row', 'col'])
    
    # Error check
    if keypoint_x0['col'] in [start_col]:
        1/0   
    

    ## Finally find some keypoints defined by intersections between cc and cv
    # Only use the medial intersections because the lateral intersections aren't
    # always visible
    es_cv1670 = norm_es.loc[1670].loc[50]
    es_cc1760 = norm_es.loc[1670].loc[150]
    summed = es_cv1670 + es_cc1760

    # Truncate image to keep only medial portion to make peak unambiguous
    start_col = int(keypoint_cv['col'] - 100)
    summed.loc[:, :start_col] = 0
    
    # Smooth
    smoothed = scipy.ndimage.gaussian_filter(summed, sigma=1)
    
    # Find peaks
    coordinates = skimage.feature.peak_local_max(
        smoothed, min_distance=10, num_peaks=1)
    coord = coordinates[0]
    
    # Convert to real pixel values
    keypoint_x1 = pandas.Series([
        summed.index[coord[0]],
        summed.columns[coord[1]],
        ], index=['row', 'col'])

    # Error check
    if keypoint_x1['col'] in [start_col]:
        1/0    
    
    
    ## Error check
    # The difference between the keypoints should be nearly 2x the travel_vector
    keydiff = keypoint_x0 - keypoint_x1
    tv2 = travel_vector * 2
    error = np.sqrt((
        (keydiff['row'] - tv2['dr']) ** 2 + 
        (keydiff['col'] - tv2['dc']) ** 2
        ).sum())
    assert error < 15 # px

    
    #~ ## Debug plot
    #~ f, ax = plt.subplots()
    #~ my.plot.imshow(norm_es.mean(level='row'), 
        #~ xd_range=(0, vs.frame_width),
        #~ yd_range=(0, vs.frame_height),
        #~ ax=ax)
    #~ ax.axis('image')
    #~ ax.scatter(
        #~ np.array([keypoint_cv, keypoint_x0, keypoint_x1])[:, 1], 
        #~ np.array([keypoint_cv, keypoint_x0, keypoint_x1])[:, 0], 
        #~ c='yellow')
    #~ f.suptitle(session_name)
    
    
    ## Define the origin as the mean of all keypoints
    origin = (keypoint_cv + keypoint_x0 + keypoint_x1) / 3.0
    

    ## Transform the edge summary by these parameters as a test
    # Generate test image
    img0 = binary_norm_es.mean(level='row')
    img0 = img0.reindex(
        range(vs.frame_height)).interpolate(
        limit_direction='both').reindex(
        range(vs.frame_width), axis=1).interpolate(
        axis=1, limit_direction='both').values

    # Translate so that origin is centered
    # TODO: Pad before translating so that data doesn't get shifted out of bounds
    # No wrapping, so data can get shifted out of bounds here
    shift_x = img0.shape[1] / 2.0 - origin['col']
    shift_y = img0.shape[0] / 2.0 - origin['row']
    img1 = scipy.ndimage.shift(img0, 
        np.array([shift_y, shift_x]),
        )
    
    # Rotate so that angle becomes zero (travel direction from left to right)
    rotation_angle = -theta
    img2 = scipy.ndimage.rotate(img1, angle=rotation_angle)
    
    # Scale to 30px
    zoom_ratio = 30.0 / scale
    img3 = scipy.ndimage.zoom(img2, zoom_ratio)
    
    # Extract a consistently sized image without recentering
    img4 = pandas.DataFrame(img3)
    sz0 = img4.shape
    img4 = img4.reindex(
        range((sz0[0] // 2) - 400, (sz0[0] // 2) + 400)).reindex(
        range((sz0[1] // 2) - 400, (sz0[1] // 2) + 400), axis=1).fillna(0)
    img4 = img4.values


    ## Store
    res = {
        'origin_x': origin['col'],
        'origin_y': origin['row'],
        'theta': theta,
        'scale': scale,
        'session': session_name,
    }
    res2 = {
        'origin': origin,
        'keypoint_cv': keypoint_cv,
        'keypoint_x0': keypoint_x0,
        'keypoint_x1': keypoint_x1,
    }

    scaling_res_l.append(res)
    keypoints_l.append(res2)
    normalized_image_l.append(img4)


## Concat over sessions
# The scaling parameters
scaling_df = pandas.DataFrame.from_records(
    scaling_res_l)

# The regularized images
all_imgs = np.array(normalized_image_l)

# The keypoints
keypoints_df = pandas.concat([
    pandas.concat(keypoints, names=['keypoint', 'coord']) 
    for keypoints in keypoints_l], 
    axis=1, keys=scaling_df['session']).T.sort_index()

# Set index
scaling_df = scaling_df.set_index('session').sort_index()


## Dump
scaling_df.to_pickle(os.path.join(params['scaling_dir'], 'scaling_df'))
keypoints_df.to_pickle(os.path.join(params['scaling_dir'], 'keypoints_df'))


## Debug plot the warped edge summaries
DEBUG_PLOT = True
if DEBUG_PLOT:
    f, ax = plt.subplots()
    my.plot.imshow(all_imgs.mean(0), ax=ax)
    ax.axis('image')

    plt.show()
