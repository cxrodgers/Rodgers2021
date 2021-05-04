## main2* is all about prepping whisker positions
# This file prepares the whisker angle data for Hilbert analysis.
#
# Procedure:
#   extracts joints
#   interpolates joints in cartesian space
#   calculate angle wrt mean follicle position
#
# Dumps for each session:
#   joint_location_each_session
# Also dumps:
#   mean_follicle

import json
import os
import tqdm
import numpy as np
import whiskvid
import pandas
import MCwatch.behavior
import runner.models


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Behavioral datasets
gs_qs = runner.models.GrandSession.objects.filter(
    tags__name=params['decoding_tag'])
session_name_l = sorted(list(gs_qs.values_list('name', flat=True)))


## Create joint_location_dir
joint_location_dir = os.path.join(
    params['patterns_dir'], 'joint_location_each_session')
if not os.path.exists(joint_location_dir):
    os.mkdir(joint_location_dir)


## Hilbert each session
mean_follicle_l = []
mean_follicle_keys_l = []
metrics_l = []
metrics_keys_l = []
for session_name in tqdm.tqdm(session_name_l):
    ## Load colorized results and pivot by whisker
    vs = whiskvid.django_db.VideoSession.from_name(session_name)
    cwe = vs.data.colorized_whisker_ends.load_data()
    
    # Ensure consistent naming
    assert cwe['whisker'].isin(['C0', 'C1', 'C2', 'C3']).all()
    
    # Load the joints (for extracting root angle)
    joints = pandas.read_pickle(os.path.join(vs.session_path, 'joints'))
    
    # error check that joints matches cwe
    assert np.allclose(
        joints.xs(0, level=1, axis=1)[['c', 'r']].values, 
        cwe[['tip_x', 'tip_y']].values,
    )    
    
    
    ## Define consistent mean follicle
    mean_follicle = cwe.groupby('whisker')[['fol_x', 'fol_y']].mean()
    mean_follicle = mean_follicle.rename(columns={'fol_x': 'x', 'fol_y': 'y'})
    mean_follicle.columns.name = 'coord'
    
    # Store
    mean_follicle_l.append(mean_follicle)
    mean_follicle_keys_l.append(session_name)

    
    ## Extract joints
    # 0=tip, 7=fol, 6=root_angle
    # Now I'm only using tip, but retain this level for compatibility
    extracted_joints = joints.loc[
        :, pandas.IndexSlice[['c', 'r'], [0]]]
    extracted_joints = extracted_joints.rename(
        columns={'r': 'y', 'c': 'x'}, level=0).rename(
        columns={0: 'tip', 7: 'fol', 6: 'fol1'}, level=1)
    extracted_joints.columns = extracted_joints.columns.rename(
        'coord', level=0)
    
    
    ## Get frames alone on index
    joint_location_by_whisker = extracted_joints.unstack('whisker')

    # We should only be including sessions with C0-C3 by this point
    assert (
        joint_location_by_whisker.columns.levels[2] == 
        ['C0', 'C1', 'C2', 'C3']).all()
    
    
    ## Deal with missing data
    # Reindex across all frames, even those without whiskers
    new_index = np.arange(
        joint_location_by_whisker.index.min(), 
        joint_location_by_whisker.index.max() + 1, 
        dtype=np.int)
    joint_location_by_whisker = joint_location_by_whisker.reindex(new_index)

    # Keep track of which frames were missing data
    frames_without_data = joint_location_by_whisker.stack(
        'whisker', dropna=False).isnull().any(1).unstack('whisker')

    # Interpolate over all missing frames
    joint_location_by_whisker = joint_location_by_whisker.interpolate(
        method='linear', limit_direction='both')
    assert not joint_location_by_whisker.isnull().any().any()


    ## Recalculate the angle for each joint wrt to the fol
    # Shortcuts
    tip_data = joint_location_by_whisker.xs('tip', level='joint', axis=1)
    
    # Apply arctan2 to the consistent follicle
    tip_angle = np.arctan2(
        -(mean_follicle['y'] - tip_data['y']),
        mean_follicle['x'] - tip_data['x'],
        ) * 180 / np.pi
    
    # Concat this angle onto tip_pos_by_whisker
    angles = pandas.concat(
        [tip_angle], axis=1, keys=['tip'], 
        verify_integrity=True)
    angles.columns = pandas.MultiIndex.from_tuples(
        [('angle', joint, whisker) for joint, whisker in angles.columns],
        names=['coord', 'joint', 'whisker'])
    joint_location_by_whisker = pandas.concat(
        [joint_location_by_whisker, angles], axis=1, verify_integrity=True,
        ).sort_index(axis=1)
    

    ## Save
    # Save tip_pos
    joint_location_filename = os.path.join(joint_location_dir, vs.name)
    joint_location_by_whisker.to_pickle(joint_location_filename)


## Concat
# Mean follicle
mean_follicle_df = pandas.concat(
    mean_follicle_l, keys=mean_follicle_keys_l, names=['session'])

# Dump
mean_follicle_df.to_pickle(
    os.path.join(params['patterns_dir'], 'mean_follicle'))