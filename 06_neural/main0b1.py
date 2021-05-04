## Slices spikes according to their whisk cycle
# tip_pos, touching, etc are also sliced, but this is not really necessary
# except for visualization
#
# Dumps:
#   big_sliced_touching
#   big_sliced_tip_pos
#   big_psth
#   big_alfws
#   spikes_by_cycle
#
# Because these operate on big_binned_spikes, they also exclude laser-on,
# FA, munged, and non-video trials.


import json
import pandas
import numpy as np
import os
import kkpandas
import MCwatch
import whiskvid
import runner.models
import tqdm
import my.neural


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Load spikes from main0a
big_binned_spikes = pandas.read_pickle(
    os.path.join(params['neural_dir'], 'big_binned_spikes'))


## Behavioral datasets
gs_qs = runner.models.GrandSession.objects.filter(
    tags__name=params['neural_tag'])
session_name_l = sorted(list(gs_qs.values_list('name', flat=True)))


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)


## Load data
# patterns data
C2_whisk_cycles = my.dataload.load_data_from_patterns(
    params, 'big_C2_tip_whisk_cycles')
big_tip_pos = my.dataload.load_data_from_patterns(
    params, 'big_tip_pos')
big_touching_df = my.dataload.load_data_from_patterns(
    params, 'big_touching_df')
    

## This much data around each peak
# TODO: Change this to -7, +7 or so
dstart = -10
dstop = 10
shift_index_halfopen = pandas.Index(range(dstart, dstop), name='shift')
shift_index_closed = pandas.Index(range(dstart, dstop + 1), name='shift')


## Iterate over sessions and slice spikes from each whisk
all_locking_frames_wshifts_l = []
sliced_tip_pos_l = []
sliced_touching_l = []
sliced_spikes_l = []
keys_l = []
    
# Iterate over sessions    
for session_name in tqdm.tqdm(session_name_l):
    
    ## Get data
    session_wc = C2_whisk_cycles.loc[session_name].copy()
    session_tip_pos = big_tip_pos.loc[session_name].copy()
    session_touching = big_touching_df.loc[session_name].copy()   
    session_spikes = big_binned_spikes.loc[session_name].copy()


    ## Lock the frame numbers to the whisk cycles
    # Take the peak frame
    all_locking_frames = session_wc['peak_frame_wrt_rwin']
    
    # Choose dstart and stop for each lock
    # Data will be blanked before/after this
    all_dstart_frames = session_wc['start_frame'] - session_wc['rwin_frame']
    all_dstop_frames = session_wc['stop_frame'] - session_wc['rwin_frame']
    
    # Get the frame of every shift
    all_locking_frames_wshifts = pandas.concat(
        [all_locking_frames + shift for shift in shift_index_halfopen],
        keys=shift_index_halfopen, axis=1)
    
    # Blank out shifted indices before dstart and after dstop
    blank_mask1 = all_locking_frames_wshifts.sub(all_dstart_frames, axis=0) < 0
    blank_mask2 = all_locking_frames_wshifts.sub(all_dstop_frames, axis=0) >= 0
    all_locking_frames_wshifts = all_locking_frames_wshifts.astype(np.float)
    all_locking_frames_wshifts[blank_mask1 | blank_mask2] = np.nan
    
    # Stack to remove nulls
    all_locking_frames_wshifts = all_locking_frames_wshifts.stack().astype(
        np.int).rename('frame')


    ## Include only trials in session_spikes (which itself only includes
    ## trials with video data)
    include_trials = session_spikes.index.get_level_values('trial').unique()
    all_locking_frames_wshifts = all_locking_frames_wshifts.loc[include_trials]
    
    
    ## Generate the frame numbers to slice, indexed by their cycle
    slicing_idx = pandas.MultiIndex.from_frame(
        all_locking_frames_wshifts.reset_index()[['trial', 'frame']])
    

    ## Slice whisking data
    # Slice session_tip_pos
    to_slice = session_tip_pos.xs(
        'angle', axis=1, level='metric').xs(
        'tip', axis=1, level='joint').stack(
        'frame')
    sliced_tip_pos = to_slice.reindex(slicing_idx)    
    sliced_tip_pos.index = all_locking_frames_wshifts.index
    
    # Also slice touching
    to_slice = session_touching.stack('frame')
    sliced_touching = to_slice.reindex(slicing_idx)    
    sliced_touching.index = all_locking_frames_wshifts.index    

    
    ## Slice psth
    sliced_spikes = session_spikes.unstack('neuron').loc[slicing_idx]
    assert not sliced_spikes.isnull().any().any()
    sliced_spikes.index = all_locking_frames_wshifts.index    
    sliced_spikes = sliced_spikes.stack().rename('spikes')
    
    
    ## Store
    all_locking_frames_wshifts_l.append(all_locking_frames_wshifts)
    sliced_tip_pos_l.append(sliced_tip_pos)
    sliced_touching_l.append(sliced_touching)
    sliced_spikes_l.append(sliced_spikes)
    keys_l.append(session_name)
    
    
## Concatenate across sessions
big_sliced_touching = pandas.concat(
    sliced_touching_l,
    keys=keys_l, names=['session'], axis=0)
big_sliced_tip_pos = pandas.concat(
    sliced_tip_pos_l,
    keys=keys_l, names=['session'], axis=0)
big_psth = pandas.concat(
    sliced_spikes_l,
    keys=keys_l, names=['session'], axis=0)
big_alfws = pandas.concat(
    all_locking_frames_wshifts_l, 
    keys=keys_l, names=['session'], axis=0)


## Sum over shifts within cycle
spikes_by_cycle = big_psth.sum(level=['session', 'trial', 'cycle', 'neuron'])


## Dump
big_sliced_touching.to_pickle(
    os.path.join(params['neural_dir'], 'big_sliced_touching'))
big_sliced_tip_pos.to_pickle(
    os.path.join(params['neural_dir'], 'big_sliced_tip_pos'))
big_psth.to_pickle(
    os.path.join(params['neural_dir'], 'big_psth'))
big_alfws.to_pickle(
    os.path.join(params['neural_dir'], 'big_alfws'))    
spikes_by_cycle.to_pickle(
    os.path.join(params['neural_dir'], 'spikes_by_cycle'))