## main2* is all about prepping whisker positions
# main2d extracts peak and start data about each whisk
# Something like this was already calculated in main2c but not saved
# This is slightly different because it's each individual whisker's peak,
# not its location at the time of the C2 cyle peak.
# This is dumped as big_cycle_features

import json
import pandas
import numpy as np
import os
import tqdm


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Load data
big_tip_pos = pandas.read_hdf(
    os.path.join(params['patterns_dir'], 'big_tip_pos'))
C2_whisk_cycles = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_C2_tip_whisk_cycles'))


## Iterate over sessions and slice out parameters of each whisk cycle
keys_l = []
parameterized_l = []
for session_name in tqdm.tqdm(C2_whisk_cycles.index.levels[0]):
    ## Get data
    session_wc = C2_whisk_cycles.loc[session_name].copy()
    session_tip_pos = big_tip_pos.loc[session_name].copy()
    
    
    ## Select whisks to lock on
    locking_whisks = session_wc.copy()


    ## Extract locking frames of each whisk
    # Choose frames to lock on
    all_locking_frames = locking_whisks['peak_frame_wrt_rwin'].astype(np.int)
    
    # Choose dstart and stop for each lock
    # Data will be blanked before/after this
    all_dstart_frames = locking_whisks['start_frame_wrt_rwin'].astype(np.int)
    all_dstop_frames = (all_dstart_frames + 
        locking_whisks['duration'].astype(np.int))
    
    # Define shifts
    shift_index = pandas.Index(range(-12, 12), name='shift')
    
    # Get the frame of every shift
    all_locking_frames_wshifts = pandas.concat(
        [all_locking_frames + shift for shift in shift_index],
        keys=shift_index, axis=1)
    
    # Blank out shifted indices before dstart and after dstop
    blank_mask1 = all_locking_frames_wshifts.sub(all_dstart_frames, axis=0) < 0
    blank_mask2 = all_locking_frames_wshifts.sub(all_dstop_frames, axis=0) >= 0
    all_locking_frames_wshifts = all_locking_frames_wshifts.astype(np.float)
    all_locking_frames_wshifts[blank_mask1 | blank_mask2] = np.nan
    
    # Stack to remove nulls
    all_locking_frames_wshifts = all_locking_frames_wshifts.stack().astype(
        np.int).rename('frame')

    
    ## Slice tip_pos accordingly
    # Index by trial * frame
    to_slice = session_tip_pos.stack('frame')
    slicing_idx = pandas.MultiIndex.from_frame(
        all_locking_frames_wshifts.reset_index()[['trial', 'frame']])

    # Slice and set index from all_locking_frames_wshifts
    sliced_tip_pos = to_slice.reindex(slicing_idx)    
    sliced_tip_pos.index = all_locking_frames_wshifts.index
    

    ## Parameters about the start of each whisk
    start_params = sliced_tip_pos.groupby(['trial', 'cycle']).first()
    start_params = start_params.stack('whisker')
    
    # Flatten the column names, e.g. ('tip', 'x') to 'tip_x')
    start_params.columns = ['%s_%s' % (joint, metric) 
        for joint, metric in start_params.columns]
    
    # Get the index back to trial * cycle
    start_params = start_params.unstack('whisker').swaplevel(
        axis=1).sort_index(axis=1)    
    
    
    ## Peak parameters
    # Take data from the peak frame of each individual whisker separately. 
    # Would have been simpler just to take the position at the C2 peak frame, 
    # but I guess this makes slightly more sense, and also I've already done the
    # work to parse out each whisk separately here (in case of other 
    # parameterizations). Likely makes barely any difference.
    
    # Identify the frame where the tip angle of each whisker reaches its
    # own peak on each whisk cycle
    stacked = sliced_tip_pos.stack('whisker')
    stacked_tip_angle = stacked.loc[:, 'tip'].loc[:, 'angle']
    peak_angle_by_whisk = stacked_tip_angle.sort_values().reset_index(
        ).drop_duplicates(
        ['trial', 'cycle', 'whisker'], keep='last').set_index(
        ['trial', 'cycle', 'shift', 'whisker'])[
        'angle'].sort_index()
    
    # Use that peak frame to index stacked again, to extract data from
    # each whisker's own peak frame on each cycle
    peak_params = stacked.loc[peak_angle_by_whisk.index].droplevel('shift')
    
    # Flatten the column names, e.g. ('tip', 'x') to 'tip_x')
    peak_params.columns = ['%s_%s' % (joint, metric) 
        for joint, metric in peak_params.columns]
    
    # Get the index back to trial * cycle
    peak_params = peak_params.unstack('whisker').swaplevel(
        axis=1).sort_index(axis=1)
    

    ## Concat the params
    whisk_cycle_params = pandas.concat([start_params, peak_params], 
        keys=['start', 'peak'], axis=1, verify_integrity=True)
    whisk_cycle_params.columns = pandas.MultiIndex.from_tuples([
        ('%s_%s' % (typ, metric), whisker) 
        for typ, whisker, metric in whisk_cycle_params.columns],
        names=['metric', 'whisker'])
    whisk_cycle_params = whisk_cycle_params.sort_index(axis=1)


    ## Store
    parameterized_l.append(whisk_cycle_params)
    keys_l.append(session_name)


## Concat
whisking_parameterized = pandas.concat(parameterized_l, keys=keys_l, 
    names=['session'])


## Dump
whisking_parameterized.to_pickle(
    os.path.join(params['patterns_dir'], 'big_cycle_features'))