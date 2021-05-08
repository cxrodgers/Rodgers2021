## Identify "anti-contacts", where whiskers pass the closest frontier
# This is done using the frontier at each frame
# An alternative would be to just use the frontier on the last frame
# when it is closest. This would make distance to frontier more interpretable
# because the frontier is stationary. But it would ignore any temporal
# component in the strategy.
#
# This takes several hours. It should be broken into parallelizable steps.

import json
import tqdm
import numpy as np
import whiskvid
import my
import my.dataload
import pandas
import os
import matplotlib.pyplot as plt
from whiskvid.Handlers.TacHandler import label_greedy


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)

N_POINTS = 100


## Load metadata about sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)
session_name_l = sorted(session_df.index)


## Load patterns data
big_tm = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_tm'))
big_tip_pos = pandas.read_hdf(
    os.path.join(params['patterns_dir'], 'big_tip_pos'))
big_ccs_df = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_ccs_df'))


## Iterate over sessions
fbf_each_l = []
fbf_any_l = []
dbw_l = []
session_keys_l = []
anti_ccs_l = []
anti_ccs_keys_l = []
for session_name in tqdm.tqdm(session_name_l):
    ## Load session data
    vs = whiskvid.django_db.VideoSession.from_name(session_name)

    # Tip pos
    session_tip_pos = big_tip_pos.loc[session_name]
    
    # Real contacts
    session_ccs = big_ccs_df.loc[session_name]
    
    # Trial matrix
    trial_matrix = big_tm.loc[session_name]
    
    
    ## Load frontier
    # For this anti_contact analysis, just use frontier_all_stim
    frontier_all_stim = pandas.read_pickle(os.path.join(
        vs.session_path, 'frontier_all_stim'))
    frontier_all_stim.index = frontier_all_stim.index.remove_unused_levels()
    
    
    ## Interpolate between locking_frames
    ## First enforce an equal number of points per frontier
    reindexed_l = []
    reindexed_keys_l = []
    for locking_frame in frontier_all_stim.index.levels[0]:
        # Frontier on this locking frame
        this_frontier = frontier_all_stim.loc[locking_frame].copy()
        
        # Sort values geometrically to make this curve nice and the
        # interpolation valid
        this_frontier = this_frontier.sort_values(['col', 'row'])
        this_frontier.index = range(len(this_frontier))
        
        # Index evenly
        new_index = np.linspace(
            this_frontier.index[0], this_frontier.index[-1], N_POINTS)
        reindexed = this_frontier.reindex(
            this_frontier.index.union(new_index)).interpolate().loc[new_index]
        
        # Intify the index
        reindexed.index = range(len(reindexed))
        reindexed.index.name = 'index'
        
        # Store
        reindexed_l.append(reindexed)
        reindexed_keys_l.append(locking_frame)
    
    # Concat
    reindexed_frontier = pandas.concat(
        reindexed_l, keys=reindexed_keys_l, names=['locking_frame'])
    
    
    ## Now temporally interpolate
    reindexed_frontier = reindexed_frontier.unstack('index')
    assert not reindexed_frontier.isnull().any().any()
    
    # New index goes from the first identified frontier frame, to the
    # last frame per trial in session_tip_pos
    last_tip_pos_frame = session_tip_pos.columns.get_level_values('frame').max()
    new_temporal_index = np.arange(
        reindexed_frontier.index[0], last_tip_pos_frame + 1, dtype=np.int)
    
    # Reindex temporally
    # This should forward fill the last identified frontier to the end
    # of the trial, I think
    reindexed_frontier = reindexed_frontier.reindex(
        new_temporal_index).interpolate()
    assert not reindexed_frontier.isnull().any().any()
    reindexed_frontier = reindexed_frontier.stack('index').sort_index()


    ## Identify when tips cross frontier
    # Get trial, whisker, frame, tip_x, tip_y on columns
    stacked_tip_pos = session_tip_pos['tip'].stack(
        ['whisker', 'frame'])[['x', 'y']].reset_index()
    
    # Rename for compatibility
    stacked_tip_pos = stacked_tip_pos.rename(
        columns={'x': 'tip_x', 'y': 'tip_y'})
    
    # Include only frames for which we have frontier data
    stacked_tip_pos = stacked_tip_pos.loc[
        stacked_tip_pos['frame'].isin(reindexed_frontier.index.levels[0])
        ].copy()
    
    # Merge on frame
    merged = pandas.merge(reindexed_frontier.reset_index(), stacked_tip_pos,
        left_on='locking_frame', right_on='frame',
    )
    
    # Identify rows where whisker crosses frontier
    merged['edge_above_right_whisker'] = (
        (merged['row'] < merged['tip_y']) &
        (merged['col'] > merged['tip_x'])
    )

    # Identify frames where each whisker crosses frontier
    frames_beyond_frontier_each_whisker = merged.groupby(
        ['trial', 'frame', 'whisker'])['edge_above_right_whisker'
        ].any().unstack('whisker')    
    
    # Identify frames where any whisker crosses frontier
    frames_beyond_frontier_any_whisker = frames_beyond_frontier_each_whisker.any(1)
    
    #~ print frames_beyond_frontier_any_whisker.any(level='trial').mean()


    ## Measure distance from whiskers to frontier
    merged['dist'] = np.sqrt(
        (merged['row'] - merged['tip_y']) ** 2 +
        (merged['col'] - merged['tip_x']) ** 2
    )
    
    # Identify minimum distance of each whisker to frontier on each frame
    # This is unsigned (always positive) regardless of which side it's on
    distance_by_whisker = merged.groupby(['trial', 'frame', 'whisker'])[
        'dist'].min().unstack('whisker')


    ## Convert this into a representation matching "tac"
    # Extract an index (trial, frame, whisker) when the whiskers are past the frontier
    # From now on this is only frames * whiskers where it is past the frontier
    tac_fw = frames_beyond_frontier_each_whisker.stack()
    tac_fw = tac_fw.index[tac_fw.values]
    
    # Extract tip pos at those indexes
    # This is a little different than the real ctac, which operated on
    # colorized_whisker_ends, and here we're acting on big_tip_pos.
    anti_tac = session_tip_pos['tip'].stack(['frame', 'whisker']
        ).sort_index().loc[tac_fw]
    
    # Add the distance beyond frontier in case it is useful
    anti_tac['dist_beyond_frontier'] = distance_by_whisker.stack().loc[tac_fw]
    
    # Reset index to conform with tac
    anti_tac = anti_tac.reset_index()

    # Rename for compatibility
    anti_tac = anti_tac.rename(
        columns={'x': 'tip_x', 'y': 'tip_y'})
        
    
    ## Threshold
    # Could optionally only include frames where it exceeds the frontier
    # by some amount. But since it is now an unbiased measure, I think it's
    # best to use it as is. Plus, below I'll exclude anti_contacts that
    # overlap with actual contacts, which is one important reason to do this.
    #~ anti_tac = anti_tac.loc[anti_tac['dist_beyond_frontier'] > 4].copy()
    
    
    ## Greedily clump contacts by whisker and trial
    ## Have to clump separately by trial because frame is wrt rwin
    # We clump contacts within 5 frames, on the assumption that it's better
    # to combine two close contacts than to arbitrarily split one big one
    # x_contig should just be infinite now that we've already labeled the
    # whiskers, otherwise we could get two simultaneous contacts by the same
    # whisker
    clumping_n_contig = 5
    ctac_l = []
    ctac_keys_l = []
    group_key_offset = 0
    for (trial, whisker), whisker_tac in anti_tac.groupby(['trial', 'whisker']):
        whisker_tac2 = whisker_tac.sort_values('frame')
        
        # Add a "group" column
        whisker_ctac = label_greedy(
            whisker_tac2.drop(['trial', 'whisker'], 1),
            n_contig=clumping_n_contig, x_contig=10000)
        
        # Ensure distinct group keys for each whisker
        assert (whisker_ctac['group'] != 0).all()
        whisker_ctac['group'] += group_key_offset
        group_key_offset = whisker_ctac['group'].max()
        
        # Store
        ctac_l.append(whisker_ctac.set_index('frame'))
        ctac_keys_l.append((trial, whisker))
    
    # Concat
    anti_ctac = pandas.concat(ctac_l, keys=ctac_keys_l, names=['trial', 'whisker'])
    anti_ctac = anti_ctac.reset_index()


    ## Summarize contacts
    grouped_ctac = anti_ctac.groupby(['trial', 'whisker', 'group'])    
    anti_ccs = pandas.concat([
        grouped_ctac[['tip_x', 'tip_y', 'angle', 'dist_beyond_frontier']].mean(),
        grouped_ctac['frame'].min().rename('frame_start'),
        grouped_ctac['frame'].max().rename('frame_stop'),
        (grouped_ctac['angle'].max() - grouped_ctac['angle'].min()
        ).rename('angle_range'),
        grouped_ctac['angle'].max().rename('angle_max'),
        ], axis=1, verify_integrity=True)
    anti_ccs['duration'] = anti_ccs['frame_stop'] - anti_ccs['frame_start'] + 1

    # Index by group key
    anti_ccs = anti_ccs.reset_index().set_index('group').sort_index()
    assert not anti_ccs.index.duplicated().any()
    anti_ccs.index.name = 'cluster'


    ## Convert frame_wrt_rwin into absolute frame
    anti_ccs['frame_start_wrt_rwin'] = anti_ccs['frame_start']
    anti_ccs['frame_stop_wrt_rwin'] = anti_ccs['frame_stop']
    
    anti_ccs['frame_start'] = (
        anti_ccs['frame_start_wrt_rwin'] + 
        anti_ccs['trial'].map(trial_matrix['rwin_frame'])
        )
    anti_ccs['frame_stop'] = (
        anti_ccs['frame_stop_wrt_rwin'] + 
        anti_ccs['trial'].map(trial_matrix['rwin_frame'])
        )

    
    ## Identify anti-contacts that contain a real contact
    # All real contacts are currently contained within anti-contacts, 
    # unless they were on the frontier
    matching_res_l = []
    matching_res_keys_l = []
    for whisker in ['C0', 'C1', 'C2', 'C3']:
        # Extract by whisker
        whisker_anti_ccs = anti_ccs.loc[anti_ccs['whisker'] == whisker]
        whisker_ccs = session_ccs.loc[session_ccs['whisker'] == whisker]
        
        
        ## Find contact starts that are contained within an anti-contact
        # For each start, the index of the anti-start that it is equal to or after
        start_ss1 = np.searchsorted(whisker_anti_ccs['frame_start'] - 1, 
            whisker_ccs['frame_start']) - 1
        
        # For each start, the index of the anti-stop that it is equal to or before
        start_ss2 = np.searchsorted(whisker_anti_ccs['frame_stop'], 
            whisker_ccs['frame_start'])

        # Where ss1 == ss2, the contact is within the inclusive interval
        # defined by some corresponding anti_ccs_start and anti_ccs_stop
        start_overlaps_with_anti_contact_mask = start_ss1 == start_ss2
        
        
        ## Find contact stops that are contained within an anti-contact
        # For each stop, the index of the anti-start that it is equal to or after
        stop_ss1 = np.searchsorted(whisker_anti_ccs['frame_start'] - 1, 
            whisker_ccs['frame_stop']) - 1
        
        # For each stop, the index of the anti-stop that it is equal to or before
        stop_ss2 = np.searchsorted(whisker_anti_ccs['frame_stop'], 
            whisker_ccs['frame_stop'])

        # Where ss1 == ss2, the anti_contact is within the inclusive interval
        # defined by some corresponding ccs_start and ccs_stop
        stop_overlaps_with_anti_contact_mask = stop_ss1 == stop_ss2

        
        ## Store the results of this process
        res = pandas.DataFrame.from_dict({
            'start_overlaps': start_overlaps_with_anti_contact_mask, 
            'stop_overlaps': stop_overlaps_with_anti_contact_mask, 
            'start_iloc_into_anti_ccs': start_ss1,
            'stop_iloc_into_anti_ccs': stop_ss1,
        })
        res.index = whisker_ccs.index
        
        # Mask out the indexes that are not matching
        res.loc[~res['start_overlaps'], 'start_iloc_into_anti_ccs'] = np.nan
        res.loc[~res['stop_overlaps'], 'stop_iloc_into_anti_ccs'] = np.nan
        
        # Replace the integer index into anti_ccs with the actual index value
        res['start_anti_ccs_index'] = np.nan
        res['stop_anti_ccs_index'] = np.nan
        mask = res['start_overlaps'].values
        res.loc[mask, 'start_anti_ccs_index'] = whisker_anti_ccs.index[
            res.loc[mask, 'start_iloc_into_anti_ccs'].values.astype(np.int)]
        mask = res['stop_overlaps'].values
        res.loc[mask, 'stop_anti_ccs_index'] = whisker_anti_ccs.index[
            res.loc[mask, 'stop_iloc_into_anti_ccs'].values.astype(np.int)]
            
        # Store
        matching_res_l.append(res)
        matching_res_keys_l.append(whisker)

    # Concat the matching results
    matching_df = pandas.concat(matching_res_l, keys=matching_res_keys_l, 
        names=['whisker'])
    assert not matching_df['start_overlaps'].isnull().any()
    assert not matching_df['stop_overlaps'].isnull().any()
    
    # Extract the indexes into anti_ccs_df that are ever matched
    mask = matching_df['start_overlaps'].values
    anti_ccs.loc[
        matching_df.loc[mask, 'start_anti_ccs_index'], 
        'overlapping_ccs_start_index'
        ] = matching_df.index.get_level_values('cluster')[mask]

    mask = matching_df['stop_overlaps'].values
    anti_ccs.loc[
        matching_df.loc[mask, 'stop_anti_ccs_index'], 
        'overlapping_ccs_stop_index'
        ] = matching_df.index.get_level_values('cluster')[mask]

    # Generate an "overlaps" column for either start or stop
    anti_ccs['overlaps'] = (
        ~anti_ccs['overlapping_ccs_start_index'].isnull() |
        ~anti_ccs['overlapping_ccs_stop_index'].isnull())


    ## Store
    anti_ccs_l.append(anti_ccs)
    anti_ccs_keys_l.append(session_name)
    

## Concat    
big_anti_ccs_df = pandas.concat(
    anti_ccs_l, keys=anti_ccs_keys_l, names=['session'])


## Save
big_anti_ccs_df.to_pickle(
    os.path.join(params['patterns_dir'], 'big_anti_ccs_df'))