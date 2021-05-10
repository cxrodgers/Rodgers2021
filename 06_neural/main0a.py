## Analyze neural data on individual whisks
# First lock spikes to the whisks they came from
#
# This one just loads spikes, converts them to frames, bins in single 
# frames, and dumps. Later this can be indexed in a manner exactly 
# analogous to how big_tip_pos etc are sliced.
#
# Laser trials are excluded
# Munged, FA trials etc are excluded by including only video analysis trials
# 
# This dumps:
#   neural_big_tm
#   big_binned_spikes
# Both of which exclude laser-on, munged, FA, non-video trials.

import json
import pandas
import numpy as np
import os
import kkpandas
import MCwatch
import whiskvid
import tqdm
import my.neural


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    

## Paths
# Create the neural dir
my.misc.create_dir_if_does_not_exist(params['neural_dir'])
    

## Timing
# This has to match the values in 01_patterns/main2b.py
EXTRACTION_START_TIME = params['extraction_start_time']
EXTRACTION_STOP_TIME = params['extraction_stop_time']
EXTRACTION_START_FRAME = int(np.rint(EXTRACTION_START_TIME * 200))
EXTRACTION_STOP_FRAME = int(np.rint(EXTRACTION_STOP_TIME * 200))


## Load metadata about sessions
neural_session_df = pandas.read_pickle(
    os.path.join(params['pipeline_input_dir'], 'neural_session_df'))


## Load data
# Directly load, instead of applying no_opto filter, because we'll do that
# explicitly below
big_tm = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_tm'))
C2_whisk_cycles = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_C2_tip_whisk_cycles'))


## Iterate over sessions and slice spikes from each whisk
# Iterate over sessions    
trial_matrix_l = []
sliced_bsdf_l = []
keys_l = []
for session_name in tqdm.tqdm(neural_session_df.index):
    ## Get data
    session_wc = C2_whisk_cycles.loc[session_name].copy()
    
    
    ## Get session objects
    vs = whiskvid.django_db.VideoSession.from_name(session_name)
    frame_rate = neural_session_df.loc[session_name, 'frame_rate']


    ## Load spikes
    spikes = pandas.read_pickle(
        os.path.join(vs.session_path, 'spikes'))
    included_clusters = np.sort(spikes['cluster'].unique())


    ## Get trial matrix and timings
    # Load the one from big_tm that has "opto" fixed, plus FA and munged
    # and no-video trials removed
    trial_matrix = big_tm.loc[session_name].copy()
    trial_timings_neural = pandas.read_pickle(
        os.path.join(vs.session_path, 'neural_trial_timings'))

    # Join
    trial_matrix = trial_matrix.join(trial_timings_neural)
    
    # Check that every row in trial_matrix is in trial_timings_neural
    # And that every trial was synced with neural
    assert (~trial_matrix['start_time_nbase'].isnull()).all()
    assert (~trial_matrix['rwin_time_nbase'].isnull()).all()

    # And that the rwin isn't too close to the recording ends
    assert (
        (trial_matrix['rwin_time_nbase'] - 2) > (spikes['time'].min() + .01)
        ).all()
    assert (
        (trial_matrix['rwin_time_nbase'] + 1) < (spikes['time'].max() - .01)
        ).all()

    # Check that we have video data for all of these trials
    # This should already have been enforced in big_tm
    video_included_trials = session_wc.index.get_level_values('trial').unique()
    assert trial_matrix.index.isin(video_included_trials).all()

    # Filtering by big_tm (presently) also excludes FA and munged trials
    assert trial_matrix['isrnd'].all()
    assert (~trial_matrix['munged']).all()
    assert (~trial_matrix['rwin_frame'].isnull()).all()

    # Include only data from no-laser trials
    trial_matrix = trial_matrix.loc[trial_matrix['opto'].isin([0, 2])].copy()
    

    ## Convert spike times to frames
    # Identify trial label of each spike
    # -1 means before the first trial
    # These values will be really high for dropped trials, but we'll drop
    # those spikes below anyway
    spikes['trial_idx'] = np.searchsorted(
        trial_matrix['start_time_nbase'].values, 
        spikes['time']) - 1

    # Index the ones before the first trial by the first trial
    spikes.loc[spikes['trial_idx'] == -1, 'trial_idx'] = 0

    # Convert to trial number
    spikes['trial'] = trial_matrix.index[spikes['trial_idx'].values].values
    spikes = spikes.drop('trial_idx', 1)

    # Re-index each spike time to the start of the trial
    spikes['t_wrt_start'] = (spikes['time'] - 
        spikes['trial'].map(trial_matrix['start_time_nbase']))

    # Convert to frames
    spikes['frame_wrt_start'] = spikes['t_wrt_start'] * frame_rate

    # Add in the start frame
    spikes['frame'] = spikes['frame_wrt_start'] + spikes['trial'].map(
        trial_matrix['exact_start_frame'])
    
    # Resort by 'frame', because this procedure can cause slight realignment
    spikes = spikes.sort_values('frame')
    
    assert not spikes['frame'].isnull().any()
    

    ## Bin each unit by frame
    # A frame range that includes all spikes
    binning_frames = np.arange(
        np.floor(spikes['frame'].min()),
        np.ceil(spikes['frame'].max()),
        1, dtype=np.int)
    
    # Bin
    counts_l = []
    counts_keys_l = []
    for unit, sub_spikes in spikes.groupby('cluster'):
        counts, junk = np.histogram(sub_spikes['frame'], bins=binning_frames)
        counts_l.append(counts)
        counts_keys_l.append(unit)
    
    # Concat
    binned_spikes_df = pandas.DataFrame(
        np.transpose(counts_l), 
        columns=pandas.Index(counts_keys_l, name='neuron'), 
        index=pandas.Index(binning_frames[:-1], name='frame'),
        )
    
    
    ## Now slice by trial
    # Get the rwin frame of every trial
    rwin_frame_by_trial = trial_matrix['rwin_frame'].astype(np.int)
    
    # Add shifts around every rwin frame
    shift_wrt_trial_index = pandas.Index(
        range(EXTRACTION_START_FRAME, EXTRACTION_STOP_FRAME), 
        name='frame')
    rwin_frame_with_shifts = pandas.concat([
        rwin_frame_by_trial + shift for shift in shift_wrt_trial_index
        ], keys=shift_wrt_trial_index).swaplevel().sort_index().rename('frame')
    
    # Slice binned_spikes_df accordingly
    sliced_bsdf = binned_spikes_df.loc[rwin_frame_with_shifts]
    sliced_bsdf.index = rwin_frame_with_shifts.index
    assert not sliced_bsdf.isnull().any().any()
    
    
    ## Save
    trial_matrix_l.append(trial_matrix)
    sliced_bsdf_l.append(sliced_bsdf.stack().rename('spikes'))
    keys_l.append(session_name)


## Concat
big_binned_spikes = pandas.concat(
    sliced_bsdf_l, keys=keys_l, names=['session'])
neural_big_tm = pandas.concat(
    trial_matrix_l, keys=keys_l, names=['session'])


## Error check consistent session * trial included
st1 = big_binned_spikes.index.to_frame()[
    ['session', 'trial']].reset_index(drop=True).drop_duplicates().sort_values(
    ['session', 'trial']).reset_index(drop=True)
st2 = neural_big_tm.index.to_frame().reset_index(drop=True).sort_values(
    ['session', 'trial']).reset_index(drop=True)
assert st1.equals(st2)


## Dump
big_binned_spikes.to_pickle(
    os.path.join(params['neural_dir'], 'big_binned_spikes'))
neural_big_tm.to_pickle(
    os.path.join(params['neural_dir'], 'neural_big_tm'))