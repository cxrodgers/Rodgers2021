## main2* is all about prepping whisker positions
# This file slices tip pos around trial times and concatenates across sessions.
# Dumps in patterns_dir:
#   big_tm
#   big_tip_pos
#
# It also chooses the trials that will be analyzed:
#   random trials,
#   non-munged (therefore have video data),
#   excluding flatter shapes,
#   excluding "late" trials where choice_time_wrt_rwin > 0.5
#   excluding trials following a spoil
#
# Codes opto in the trial_matrix as 0=no opto, or sham; 2=laser-off; 3=laser-on
#
# Cuts a wider window to avoid edge effects later

import my
import my.dataload
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

# How to lock
LOCKING_COLUMN = 'rwin_frame'

# This much data will be extracted around the locking time from each
# trial, and put into big_tip_pos
EXTRACTION_START_TIME = params['extraction_start_time']
EXTRACTION_STOP_TIME = params['extraction_stop_time']

# Drop trials where response is after this time
LATE_RESPONSE_THRESHOLD = params['late_response_threshold']


## Paths
joint_location_dir = os.path.join(
    params['patterns_dir'], 'joint_location_each_session')


## Behavioral datasets
gs_qs = runner.models.GrandSession.objects.filter(
    tags__name=params['decoding_tag'])
session_name_l = sorted(list(gs_qs.values_list('name', flat=True)))


## Load session_df with opto information
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)


## Iterate over sessions
sliced_phases_df_l = []
sliced_tip_pos_df_l = []
subtm_l = []
keys_l = []

for session_name in tqdm.tqdm(session_name_l):
    ## Get data
    vs = whiskvid.django_db.VideoSession.from_name(session_name)
    

    ## Load tip_pos_by_whisker
    tip_pos_filename = os.path.join(joint_location_dir, vs.name)
    if not os.path.exists(tip_pos_filename):
        print("warning: cannot find tip pos for %s" % vs.name)
        continue
    tip_pos = pandas.read_pickle(tip_pos_filename)
    
    
    ## Load trial matrix
    trial_matrix = vs.data.trial_matrix.load_data()


    ## Add trial history columns to trial_matrix
    # Add prev_choice and prev_rewside by simply shifting
    trial_matrix['prev_choice'] = trial_matrix['choice'].shift()
    trial_matrix['prev_rewside'] = trial_matrix['rewside'].shift()
    
    # Previous hit is coded as the previous rewside if it was a hit, otherwise nan
    trial_matrix['prev_hit'] = trial_matrix['rewside'].copy()
    trial_matrix.loc[trial_matrix['outcome'] != 'hit', 'prev_hit'] = np.nan
    trial_matrix['prev_hit'] = trial_matrix['prev_hit'].shift()
    
    # Previous error is coded as the previous rewside if it was a error, otherwise nan
    trial_matrix['prev_err'] = trial_matrix['rewside'].copy()
    trial_matrix.loc[trial_matrix['outcome'] != 'error', 'prev_err'] = np.nan
    trial_matrix['prev_err'] = trial_matrix['prev_err'].shift()


    ## Additional trial_matrix processing
    # If this was no-opto or sham-opto, code opto as 0
    # Otherwise, leave it as it is (2 or 3, shouldn't be any 4 or 5)
    if (
            (not session_df.loc[session_name, 'opto']) or 
            session_df.loc[session_name, 'sham']):
        trial_matrix['opto'] = 0

    # Skip the ones with the old servo_pos
    assert not trial_matrix['servo_pos'].isin([1770, 1690]).any()
    
    # For 2shapes discrimniation, drop trials with E2 shapes
    if session_df.loc[session_name, 'twoshapes']:            
        trial_matrix = trial_matrix.loc[
            trial_matrix['stepper_pos'].isin([50, 150]), :].copy()
    
    # Drop "late" response trials
    trial_matrix = trial_matrix.loc[
        trial_matrix['rt'] < LATE_RESPONSE_THRESHOLD, :]
    
    # Keep only random, non-munged trials; also drop those following a spoil
    trial_matrix = trial_matrix.loc[
        trial_matrix['isrnd']
        & ~trial_matrix['munged']
        & (trial_matrix['prev_choice'] != 'nogo')
    ]
    assert trial_matrix['prev_choice'].isin(['left', 'right']).all()
    
    # Check that only these specified columns can contain nulls
    assert not trial_matrix.drop(
        ['prev_hit', 'prev_err', 'shape_stop_frame', 'shape_stop_time'], 
        1).isnull().any().any()
    
    # Intify the locking frames
    assert not trial_matrix['rwin_frame'].isnull().any()
    trial_matrix['rwin_frame'] = trial_matrix['rwin_frame'].astype(np.int)

    # This is not true for detection sessions
    #~ assert not trial_matrix['shape_stop_frame'].isnull().any()
    #~ trial_matrix['shape_stop_frame'] = trial_matrix['shape_stop_frame'].astype(np.int)


    ## Slice and extract
    # Window length
    dstart = int(np.rint(vs.frame_rate * EXTRACTION_START_TIME))
    dstop = int(np.rint(vs.frame_rate * EXTRACTION_STOP_TIME))
    t = np.arange(dstart, dstop) / vs.frame_rate

    # Drop triggers with no data
    trial_matrix = trial_matrix.loc[
        (trial_matrix[LOCKING_COLUMN] > -dstart) &
        (trial_matrix[LOCKING_COLUMN] < len(tip_pos) - dstop)
    ].copy()

    # Slice by triggers
    sliced_tip_pos_l = []
    for trial, trigger in trial_matrix[LOCKING_COLUMN].iteritems():
        # Slice
        sliced_tip_pos = tip_pos.loc[            
            trigger + dstart:trigger + dstop - 1]

        # Label (basically replacing coord and param with metric here)
        sliced_tip_pos.columns.names = ['metric', 'joint', 'whisker']
        sliced_tip_pos = sliced_tip_pos.reorder_levels(
            ['joint', 'whisker', 'metric'], axis=1).sort_index(axis=1)

        
        ## Reindex relative to trigger
        sliced_tip_pos.index = (
            sliced_tip_pos.index.values.astype(np.int) - trigger)
        
        
        ## Store
        sliced_tip_pos_l.append(sliced_tip_pos)

    
    ## Concat
    concatted_tip_pos = pandas.concat(sliced_tip_pos_l, 
        keys=trial_matrix[LOCKING_COLUMN].index)
    concatted_tip_pos.index.names = ['trial', 'frame']
    
    # Trials on index
    concatted_tip_pos = concatted_tip_pos.unstack('frame')

    # Store
    sliced_tip_pos_df_l.append(concatted_tip_pos)
    subtm_l.append(trial_matrix)
    keys_l.append(session_name)
    
    if len(concatted_tip_pos) != len(trial_matrix):
        1/0


## Concat
big_tip_pos = pandas.concat(sliced_tip_pos_df_l, keys=keys_l, 
    names=['session'], verify_integrity=True)
big_tm = pandas.concat(subtm_l, keys=keys_l, 
    names=['session'], verify_integrity=True)


## Dump
print("dumping")
big_tip_pos.to_hdf(os.path.join(params['patterns_dir'], 'big_tip_pos'), 
    key='big_tip_pos', mode='w')
big_tm.to_pickle(os.path.join(params['patterns_dir'], 'big_tm'))
