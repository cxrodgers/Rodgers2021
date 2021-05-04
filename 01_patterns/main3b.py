## main3* is all about prepping contacts
# This one joins together all ccs into big_ccs_df, 
# identifies the whisking cycle of each contact, 
# drops contacts without matching whisk cycles (because outside the trial window),
# parameterizes the grasps (e.g., which whiskers, cross-whisker stuff),
# and dumps big_ccs_df and big_grasp_df in the data_directory.

import json
import tqdm
import numpy as np
import whiskvid
import my
import pandas
import os
import matplotlib.pyplot as plt
import MCwatch.behavior
import runner.models


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Sessions
# Behavioral datasets
gs_qs = runner.models.GrandSession.objects.filter(
    tags__name=params['decoding_tag'])
session_name_l = sorted(list(gs_qs.values_list('name', flat=True)))


## Load whisk cycles to define grasps
C2_whisk_cycles = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_C2_tip_whisk_cycles'))


## Iterate over sessions
ccs_l = []
matching_cycles_l = []
keys_l = []
for session_name in tqdm.tqdm(session_name_l):
    ## Load sesion params
    vs = whiskvid.django_db.VideoSession.from_name(session_name)


    ## Get ccs with kinematics
    session_ccs = pandas.read_pickle(
        os.path.join(params['patterns_dir'], 'ccs_with_kinematics', 
        session_name))


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
big_ccs_df = pandas.concat(
    ccs_l, keys=keys_l, names=['session'], 
    verify_integrity=True, sort=True)


## Drop null cycles (those outside trial window)
# This *should* drop all null contact phases as well
big_ccs_df = big_ccs_df.dropna(subset=['cycle'])
big_ccs_df['cycle'] = big_ccs_df['cycle'].astype(np.int)

assert not big_ccs_df.isnull().any().any()


## Calculate contact time wrt cycle peak
big_ccs_df = big_ccs_df.join(
    C2_whisk_cycles['peak_frame'].rename('cycle_peak_frame'), 
    on=['session', 'trial', 'cycle'])
big_ccs_df['frame_start_wrt_peak'] = (
    big_ccs_df['frame_start'] - big_ccs_df['cycle_peak_frame'])
big_ccs_df['frame_stop_wrt_peak'] = (
    big_ccs_df['frame_stop'] - big_ccs_df['cycle_peak_frame'])


## Parameterize grasps
# frame_start, and locked_t at beginning of grasp
big_grasp_df = big_ccs_df.groupby(
    ['session', 'trial', 'cycle'])[
    ['frame_start', 'locked_t', 'frame_start_wrt_peak']].min()

# Add in the end of the grasp
big_grasp_df = big_grasp_df.join(big_ccs_df.groupby(
    ['session', 'trial', 'cycle'])[
    ['frame_stop', 'frame_stop_wrt_peak']].max())


## Assign a label to each grasp type
# Count contacts by whisker within each grasp
# Double-hits within a cycle are ~2% of the total, so hopefully we can neglect
contact_counts_by_grasp = big_ccs_df.groupby(
    ['session', 'trial', 'cycle', 'whisker']).size()

# Unstack and binarize
contact_counts_by_grasp = contact_counts_by_grasp.unstack(
    ).fillna(0).astype(np.int)
grasp_wic = (contact_counts_by_grasp > 0).astype(np.int)

# Group by whiskers in contact
def label_by_wic(grasp_wic, exclude_C0=False):
    """Label each grasp by the whiskers it contains
    
    grasp_wic : DataFrame
        Index: grasp keys. Columns: whisker. Values: binarized contact.
    
    exclude_C0 : bool
        If False, group by all whiskers.
        If True, ignore C0, and group only by C1, C2, and C3.
            But label C0-only contacts as C0.
    """
    # Set grouping_whiskers
    if exclude_C0:
        grouping_whiskers = ['C1', 'C2', 'C3']
    else:
        grouping_whiskers = ['C0', 'C1', 'C2', 'C3']
    
    # Init return variable
    res = pandas.Series(
        ['blank'] * len(grasp_wic), index=grasp_wic.index).rename('label')
    
    # Group
    gobj = grasp_wic[['C0', 'C1', 'C2', 'C3']].groupby(grouping_whiskers)
    for included_mask, sub_grasp_wic in gobj:
        # Generate label by joining all whiskers in this group
        label = '-'.join(
            [w for w, w_in in zip(grouping_whiskers, included_mask) if w_in])
        
        if label == '':
            # This should only happen if exclude_C0 and on the C0 group
            assert exclude_C0
            assert (sub_grasp_wic['C0'] == 1).all()
            assert (sub_grasp_wic.drop('C0', 1) == 0).all().all()
            
            # So label it C0
            label = 'C0'
        
        # Assign 
        res.loc[sub_grasp_wic.index] = label
    
    # Error check
    assert 'blank' not in res.values
    
    return res
    
# Label with and without C0
big_grasp_df['label'] = label_by_wic(grasp_wic, exclude_C0=False)
big_grasp_df['label_noC0'] = label_by_wic(grasp_wic, exclude_C0=True)


## Store nwic
big_grasp_df['nwic'] = grasp_wic.sum(1)


## Parameterize cross-whisker onset latency
frame_start_by_whisker = big_ccs_df.groupby(
    ['session', 'trial', 'whisker', 'cycle'])[
    'frame_start'].min().unstack('whisker')

# Do only C2 vs other
cross_whisker_onset_latencies = pandas.concat([
    frame_start_by_whisker['C1'] - frame_start_by_whisker['C2'],
    frame_start_by_whisker['C3'] - frame_start_by_whisker['C2'],
    ], axis=1, keys=['C1vC2', 'C3vC2'])


## Parameterize cross-whisker offset latency
frame_stop_by_whisker = big_ccs_df.groupby(
    ['session', 'trial', 'whisker', 'cycle'])[
    'frame_stop'].max().unstack('whisker')

# Do only C2 vs other
cross_whisker_offset_latencies = pandas.concat([
    frame_stop_by_whisker['C1'] - frame_stop_by_whisker['C2'],
    frame_stop_by_whisker['C3'] - frame_stop_by_whisker['C2'],
    ], axis=1, keys=['C1vC2', 'C3vC2'])


## Parameterize cross-whisker angle
contact_angle_by_whisker = big_ccs_df.groupby(
    ['session', 'trial', 'whisker', 'cycle'])[
    'angle'].mean().unstack('whisker')

# Do only C2 vs other
cross_whisker_angles = pandas.concat([
    contact_angle_by_whisker['C1'] - contact_angle_by_whisker['C2'],
    contact_angle_by_whisker['C3'] - contact_angle_by_whisker['C2'],
    ], axis=1, keys=['C1vC2', 'C3vC2'])


## Parameterize cross-whisker durations
contact_duration_by_whisker = big_ccs_df.groupby(
    ['session', 'trial', 'whisker', 'cycle'])[
    'duration'].sum().unstack('whisker')

# Do only C2 vs other
cross_whisker_durations = pandas.concat([
    contact_duration_by_whisker['C1'] - contact_duration_by_whisker['C2'],
    contact_duration_by_whisker['C3'] - contact_duration_by_whisker['C2'],
    ], axis=1, keys=['C1vC2', 'C3vC2'])


## Concatenate cross-whisker stuff onto big_grasp_df
# Concat
cwdata = pandas.concat([
    cross_whisker_onset_latencies,
    cross_whisker_offset_latencies,
    cross_whisker_angles,
    cross_whisker_durations,
    ], keys=['latency_on', 'latency_off', 'angle', 'duration'], 
    axis=1, verify_integrity=True)

# Flatten columns
cwdata.columns = [
    '{}_{}'.format(whiskers, metric) 
    for metric, whiskers in cwdata.columns]

# Concat onto big_grasp_df
big_grasp_df = big_grasp_df.join(cwdata)


## Dump
big_ccs_df.to_pickle(os.path.join(params['patterns_dir'], 'big_ccs_df'))
big_grasp_df.to_pickle(os.path.join(params['patterns_dir'], 'big_grasp_df'))