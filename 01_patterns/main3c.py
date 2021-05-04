## Calculates "touching" representation: 1 during contact, 0 otherwise
# This will be constructed to match big_tip_pos in index (session * trial)
# and columns (whisker * frame_wrt_rwin)
# Writes out big_touching_df to patterns_dir
# And big_binarized_touching_by_cycle

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


## How to lock
LOCKING_COLUMN = 'rwin_frame'


## Load trial matrix
big_tm = pandas.read_pickle(os.path.join(params['patterns_dir'], 'big_tm'))
big_ccs_df = pandas.read_pickle(os.path.join(params['patterns_dir'], 'big_ccs_df'))
C2_whisk_cycles = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_C2_tip_whisk_cycles'))

# These will be used to index the result
whisker_names = ['C0', 'C1', 'C2', 'C3']

# Choose frames that go into big_touching_df
# This is the same as for big_tip_pos
dstart = int(np.rint(200 * params['extraction_start_time']))
dstop = int(np.rint(200 * params['extraction_stop_time']))
frame_index = pandas.Index(np.arange(dstart, dstop, dtype=np.int), name='frame')

    
## Iterate over sessions
rec_l = []
binarized_touching_by_cycle_l = []
keys_l = []
for session in tqdm.tqdm(big_tm.index.levels[0]):
    # This session
    session_tm = big_tm.loc[session]
    session_contacts = big_ccs_df.loc[session]
    
    for trial in session_tm.index:
        # Get rwin_frame
        locked_frame = session_tm.loc[trial, LOCKING_COLUMN]
        
        # Find contacts for this session * trial
        trial_contacts = session_contacts.loc[session_contacts['trial'] == trial]

        # Start with an all zero DataFrame
        touching_df = pandas.DataFrame(
            np.zeros((len(frame_index), len(whisker_names)), dtype=np.int),
            index=frame_index, columns=whisker_names,
        )

        # Add each
        for idx, row in trial_contacts.iterrows():
            whisker = row['whisker']

            # Non-Pythonic, which is good, because the shortest possible contact
            # has frame_start == frame_stop
            start_wrt_rwin = row['frame_start'] - locked_frame
            stop_wrt_rwin = row['frame_stop'] - locked_frame
            touching_df.loc[start_wrt_rwin:stop_wrt_rwin, whisker] += 1        
        
        # Assign cycle to every frame of touching_df
        trial_cycles = C2_whisk_cycles.loc[session].loc[trial].copy()
        first_frame = trial_cycles['start_frame_wrt_rwin'].min()
        last_frame = (
            trial_cycles['start_frame_wrt_rwin'] + trial_cycles['duration']
            ).max()
        frame2cycle = trial_cycles['start_frame_wrt_rwin'].reset_index(
            ).rename(columns={'start_frame_wrt_rwin': 'frame'}).set_index(
            'frame')['cycle'].reindex(
            range(first_frame, last_frame)
            ).ffill().astype(np.int)

        # Error check
        if (touching_df > 1).any().any():
            print("warning: impossible simultaneous contacts in trial %d" % trial)
            touching_df[touching_df > 1] = 1
        
        # Group touching by cycle
        joined = touching_df.join(frame2cycle).dropna()
        joined['cycle'] = joined['cycle'].astype(np.int)
        grouped = joined.groupby('cycle')[whisker_names].sum()
        binarized_touching_by_cycle = (grouped > 0).astype(np.int)
        
        # Store
        rec_l.append(touching_df)
        binarized_touching_by_cycle_l.append(binarized_touching_by_cycle)
        keys_l.append((session, trial))

# Concat
big_touching_df = pandas.concat(rec_l, axis=1, keys=keys_l, 
    names=['session', 'trial', 'whisker']).T.unstack('whisker').swaplevel(
    'whisker', 'frame', axis=1).sort_index(axis=1)
big_binarized_touching_by_cycle = pandas.concat(
    binarized_touching_by_cycle_l, axis=0, keys=keys_l, 
    names=['session', 'trial']).sort_index()
    
# Save
big_touching_df.to_pickle(
    os.path.join(params['patterns_dir'], 'big_touching_df'))
big_binarized_touching_by_cycle.to_pickle(
    os.path.join(params['patterns_dir'], 'big_binarized_touching_by_cycle'))