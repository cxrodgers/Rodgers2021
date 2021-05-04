## main3* is all about prepping contacts
# This ones adds kinematic data as columns to ccs, and dumps to
# DATA_DIR/ccs_with_kinematics/SESSION_NAME
#
# Contact angle is recalculated from mean_follicle.

import json
import tqdm
import numpy as np
import whiskvid
import my
import pandas
import os
import MCwatch.behavior
import runner.models


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## How to lock
LOCKING_COLUMN = 'rwin_frame'


## Where to save pattern data
if not os.path.exists(
        os.path.join(params['patterns_dir'], 'ccs_with_kinematics')):
    os.mkdir(os.path.join(params['patterns_dir'], 'ccs_with_kinematics'))


## Sessions
# Behavioral datasets
gs_qs = runner.models.GrandSession.objects.filter(
    tags__name=params['decoding_tag'])
session_name_l = sorted(list(gs_qs.values_list('name', flat=True)))


## Load kappa
kp = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'kappa_parameterized'))


## Load mean_follicle_df, for recalculating contact angle
mean_follicle_df = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'mean_follicle'))


## Load phases, for assigning phase of contact
whisking_parameters_by_frame = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'whisking_parameters_by_frame'))


## Iterate over sessions
mean_bout_l = []
mean_bout_keys_l = []
for session_name in tqdm.tqdm(session_name_l):
    ## Load session data
    vs = whiskvid.django_db.VideoSession.from_name(session_name)


    ## Load data
    # Load trial matrix
    # Use the original one, not big_tm, with all the trial numbers
    trial_matrix = vs.data.trial_matrix.load_data()
    
    # Load tip_pos_by_whisker, for calculating pre-contact velocity
    tip_pos_filename = os.path.join(
        params['patterns_dir'], 'joint_location_each_session', vs.name)
    tip_pos = pandas.read_pickle(tip_pos_filename)
    
    # Get phases
    session_whisking_params = whisking_parameters_by_frame.loc[session_name]
    session_whisking_phase = session_whisking_params['phase']
    

    ## Get ccs
    # This adds a trial number based on exact_start_frame, but no other
    # trial data
    ccs = vs.data.colorized_contacts_summary.load_data(
        trial_matrix=trial_matrix,
        locking_column='exact_start_frame',
        add_trial_info=True, columns_to_join=[],
        )
    assert 'angle' in ccs

    # Lock the contact times in frames
    ccs['locked_frame'] = ccs['frame_start'] - ccs['trial'].map(
        trial_matrix[LOCKING_COLUMN])

    # Convert to seconds using hardcoded frame rate
    ccs['locked_t'] = ccs['locked_frame'] / vs.frame_rate

    # Reset index
    ccs = ccs.reset_index()

    # Drop contacts in the first two frames, for which we can't calculate
    # a velocity
    ccs = ccs.loc[ccs['frame_start'] >= 2].copy()


    ## Recalculate contact angle from tip_pos so that it uses consistent follicle location
    # Follicle of this session
    mean_follicle = mean_follicle_df.loc[session_name]
    
    # Drop follicle location and old angle
    ccs = ccs.drop(['fol_x', 'fol_y', 'angle', 'angle_range'], 1)
    
    # Recalculate angle of each contact using mean_follicle
    to_join1 = ccs[['cluster', 'whisker', 'tip_x', 'tip_y']]
    to_join1 = to_join1.join(mean_follicle, on='whisker')
    to_join1['angle'] = np.arctan2(
        -(to_join1['y'] - to_join1['tip_y']),
        to_join1['x'] - to_join1['tip_x'],
        ) * 180 / np.pi    
    
    # Join this angle back onto ccs
    ccs = ccs.join(
        to_join1[['cluster', 'angle']].set_index('cluster'), on='cluster')

    # Check
    assert not ccs.isnull().any().any()
    

    ## Calculate pre-contact velocity
    # Get the frame numbers of the contact and previous frames
    frame_number_starts = ccs[['cluster', 'whisker', 'frame_start']].set_index(
        ['whisker', 'cluster'])['frame_start'].rename('frame')
    frame_numbers = pandas.concat([
        frame_number_starts, frame_number_starts - 1, frame_number_starts - 2], 
        axis=1, keys=[0, 1, 2], verify_integrity=True, names='lag'
        ).stack().rename('frame').reset_index()
    
    # Join on tip_pos
    joined = frame_numbers.join(tip_pos['angle'].stack('whisker'), 
        on=['frame', 'whisker'])
    
    # Diff the angle along the lag axis
    # Positive values mean angle was increasing
    diffed = -joined.drop('frame', 1).set_index(
        ['whisker', 'cluster', 'lag']).unstack(
        ['whisker', 'cluster']).diff().drop(0)
    diffed.columns.names = ['joint', 'whisker', 'cluster']
    diffed = diffed.unstack().unstack(['lag', 'joint']).sort_index(axis=1)
    
    # Calculate velocity in a few ways
    # velocity1 is angle(0) - angle(-1), e.g. velocity over one preceding frame
    #   This is the diff at lag 1
    # velocity2 is (angle(0) - angle(-2)) / 2, e.g. velocity over two preceding frames
    #   This is the mean of the diff at lag 1 and at lag 2
    # velocity3 is angle(-1) - angle(-2), e.g. ignoring contact frame itself
    #   This is the diff at lag 2
    velocities = pandas.concat([
        diffed[1], diffed[[1, 2]].mean(level='joint', axis=1), diffed[2]],
        axis=1, keys=['velocity1', 'velocity2', 'velocity3'], 
        verify_integrity=True)
    
    # Put velocities onto ccs
    velocities.columns = ['%s_%s' % (vel, joint) for vel, joint in velocities.columns]
    ccs = ccs.join(velocities, on=['whisker', 'cluster'])
    ccs = ccs.set_index('cluster').sort_index()


    ## Join the kappa columns on ccs
    to_join = kp.loc[session_name, ['min', 'max', 'std']].rename(
        columns={'min': 'kappa_min', 'max': 'kappa_max', 'std': 'kappa_std'})
    ccs = ccs.join(to_join)


    ## Check null
    assert not ccs.isnull().any().any()


    ## Join phase at frame_start of each contact
    # It will be null for contacts outside of the whisking analysis window
    # Stack
    to_join = session_whisking_phase.stack().rename(
        'phase').reset_index().rename(columns={'frame': 'locked_frame'})    
    
    # Join
    ccs = ccs.join(
        to_join.set_index(['trial', 'locked_frame']), 
        on=['trial', 'locked_frame'])
    

    ## Save
    ccs.to_pickle(
        os.path.join(params['patterns_dir'], 
        'ccs_with_kinematics', session_name))
    
    
    