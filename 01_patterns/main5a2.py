## Process anti contacts, e.g. add cycle
import json
import os
import pandas
import numpy as np
import tqdm
import whiskvid
import my
import my.dataload


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)
session_name_l = sorted(session_df.index)


## Load patterns data
big_tm = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_tm'))
big_anti_ccs_df = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_anti_ccs_df'))
C2_whisk_cycles = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_C2_tip_whisk_cycles'))


## Dump the ones that overlap with a contact
big_anti_ccs_df = big_anti_ccs_df[big_anti_ccs_df['overlaps'] == False]


## Dump columns we don't need
big_anti_ccs_df = big_anti_ccs_df.drop(
    ['angle_range', 'overlapping_ccs_start_index', 
    'overlapping_ccs_stop_index', 'overlaps'], 1)


## Iterate over sessions and join on cycle
ccs_l = []
matching_cycles_l = []
keys_l = []
for session_name in tqdm.tqdm(session_name_l):
    ## Get anti_ccs
    session_ccs = big_anti_ccs_df.loc[session_name]


    ## Match up contact to its whisk cycle
    # Get the session whisk cycle
    session_wc = C2_whisk_cycles.loc[session_name]

    # Find the matching cycles
    matching_cycle_idx = my.misc.find_interval(
        event_times=session_ccs['frame_start'], 
        interval_starts=session_wc['start_frame'], 
        interval_stops=session_wc['stop_frame'],
    )
    nonnullmask = ~np.isnan(matching_cycle_idx)
    
    # Extract the corresponding indices into whisk_cycles
    sCw_idxs = session_wc.index[
        matching_cycle_idx[nonnullmask].astype(np.int)]
    
    # Index them like the session_ccs
    to_join = sCw_idxs.to_frame()
    to_join.index = session_ccs.index[nonnullmask]
    
    
    ## Join the cycle onto the ccs
    to_join = to_join.rename(columns={'trial': 'trial2'})
    session_ccs = session_ccs.join(to_join, on='cluster')
    
    # Error check
    nullmask = session_ccs['trial2'].isnull()
    assert (
        session_ccs.loc[~nullmask, 'trial'] == 
        session_ccs.loc[~nullmask, 'trial2']).all()
    session_ccs = session_ccs.drop('trial2', axis=1)
    
    
    ## Store
    ccs_l.append(session_ccs)
    keys_l.append(session_name)


## Concat
big_anti_ccs_df2 = pandas.concat(
    ccs_l, keys=keys_l, names=['session'], 
    verify_integrity=True, sort=True)


## Drop null cycles (those outside trial window)
big_anti_ccs_df2 = big_anti_ccs_df2.dropna(subset=['cycle'])
big_anti_ccs_df2['cycle'] = big_anti_ccs_df2['cycle'].astype(np.int)

assert not big_anti_ccs_df2.isnull().any().any()


## Calculate contact time wrt cycle peak
big_anti_ccs_df2 = big_anti_ccs_df2.join(
    C2_whisk_cycles['peak_frame'].rename('cycle_peak_frame'), 
    on=['session', 'trial', 'cycle'])
big_anti_ccs_df2['frame_start_wrt_peak'] = (
    big_anti_ccs_df2['frame_start'] - big_anti_ccs_df2['cycle_peak_frame'])
big_anti_ccs_df2['frame_stop_wrt_peak'] = (
    big_anti_ccs_df2['frame_stop'] - big_anti_ccs_df2['cycle_peak_frame'])


## Dump
big_anti_ccs_df2.to_pickle(
    os.path.join(params['patterns_dir'], 'big_anti_ccs_df2'))
