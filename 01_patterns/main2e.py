## Extracts kappa around contacts and parameterizes it
# Dumps in patterns_dir:
#   peri_contact_kappa
#   kappa_parameterized

import json
import tqdm
import os
import pandas
import numpy as np
import whiskvid
import my
import my.dataload

## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Sessions
session_df, task2mouse, mouse2task = my.dataload.load_session_metadata(params)
session_name_l = list(session_df.index)


## Extraction params
# Extract this much data around each contact (Pythonic)
EXTRACT_START = -5
EXTRACT_STOP = 20 # Note that mean C3 duration is 10!
extract_t = range(EXTRACT_START, EXTRACT_STOP)

# This will be the baseline period (Pythonic)
BASELINE_START = -5
BASELINE_STOP = 0

# Additionally include this many frames after the contact offset in the parameterization
# If it's 2, then it's the first 2 frames after detaching, so the
# "frame_stop" frame and the one after it.
# Pythonically the window will be 
# frame_start:frame_start + duration + POST_CONTACT_FRAMES
# If the duration is 1 (only one frame in contact), then the window
# includes this frame and the 2 afterward (3 total)
POST_CONTACT_FRAMES = 2


## Load the scaling information to convert to mm
scaling_df = pandas.read_pickle(
    os.path.join(params['scaling_dir'], 'scaling_df'))
conversion_px_per_mm = scaling_df['scale'] / 2.7


## Iterate over sessions
peri_kappa_l = []
contact_metadata_l = []
keys_l = []
for session_name in tqdm.tqdm(session_name_l):
    ## Load data
    vs = whiskvid.django_db.VideoSession.from_name(session_name)
    
    # Load cwek
    cwek = vs.data.cwe_with_kappa.load_data()
    
    # Load contacts summary
    ccs = vs.data.colorized_contacts_summary.load_data(add_trial_info=False)
    
    # Index by whisker and frame
    curv = cwek.set_index(['whisker', 'frame'])['kappa'].sort_index()
    
    
    ## Iterate over whiskers
    reindexed_kappa_d = {}
    for whisker, whisker_ccs in ccs.groupby('whisker'):
        # Get whisker curv
        whisker_curv = curv.loc[whisker]
        whisker_curv = whisker_curv.reindex(
            np.arange(whisker_curv.index.min(), whisker_curv.index.max(), 
            dtype=np.int))
        
        # Get contacts
        contact_frames = whisker_ccs['frame_start']
        
        # Get window of data around every contact frame
        reindexed_kappa = pandas.DataFrame([
            whisker_curv.loc[
            frame_start + EXTRACT_START:frame_start + EXTRACT_STOP - 1].values
            for frame_start in contact_frames.values],
            index=whisker_ccs.index,
            columns=pandas.Index(extract_t, name='frame_wrt_contact')
        )
        reindexed_kappa_d[whisker] = reindexed_kappa
    
    
    ## Concat over whiskers
    peri_kappa = pandas.concat(reindexed_kappa_d, verify_integrity=True,
        names=['whisker'])

    # Store
    peri_kappa_l.append(peri_kappa)
    contact_metadata_l.append(ccs[['whisker', 'duration']])
    keys_l.append(session_name)


## Concat over sessions
# Concat peri_kappa
bigpk = pandas.concat(
    peri_kappa_l, keys=keys_l, names=['session'],
    verify_integrity=True)

# Concat metadata
big_metadata = pandas.concat(
    contact_metadata_l, keys=keys_l, names=['session'],
    verify_integrity=True)

# Drop the redundant whisker level
bigpk.index = bigpk.index.droplevel('whisker')
big_metadata = big_metadata.drop('whisker', 1)

# Interpolate over nans
bigpk = bigpk.interpolate(axis=1, limit_direction='both', method='linear')
assert not bigpk.isnull().any().any()


## Align with the duration
# Add metadata columns
joined = bigpk.join(big_metadata, on=['session', 'cluster'])
assert not joined.isnull().any().any()

# Ceiling the duration
# The window can't be any longer than EXTRACT_STOP
joined['duration_ceil'] = joined['duration'].copy()
joined.loc[ 
    joined['duration_ceil'] > (EXTRACT_STOP - POST_CONTACT_FRAMES), 
    'duration_ceil'] = EXTRACT_STOP - POST_CONTACT_FRAMES


## Reindex by duration_ceil, duration, session, cluster
joined = joined.reset_index()
joined = joined.set_index(
    ['duration_ceil', 'duration', 'session', 'cluster']).sort_index()

# Intify the columns now that the metadata has been moved to index
joined.columns = joined.columns.astype(np.int)


## Baseline each one
bjoined = joined.sub(joined.loc[:, -5:-1].mean(1), axis=0)


## Convert to 1/mm
bjoined = bjoined.mul(conversion_px_per_mm, axis=0, level='session')


## Parameterize each duration group
this_parameterized_l = []
this_parameterized_keys_l = []
for duration_ceil in bjoined.index.levels[0]:
    this_duration = bjoined.loc[duration_ceil]
    
    # Window the data around the contact period
    # Non-pythonic
    assert duration_ceil + POST_CONTACT_FRAMES - 1 in this_duration.columns
    windowed = this_duration.loc[:, 0:(duration_ceil + POST_CONTACT_FRAMES - 1)]
    
    # Parameterize
    this_parameterized = pandas.concat([
        windowed.max(1),
        windowed.min(1),
        windowed.std(1),
        ], axis=1, verify_integrity=True,
        keys=['max', 'min', 'std']
    )
    
    # Store
    this_parameterized_l.append(this_parameterized)
    this_parameterized_keys_l.append(duration_ceil)

# Concat
parameterized = pandas.concat(this_parameterized_l, 
    keys=this_parameterized_keys_l, names=['duration_ceil'], 
    verify_integrity=True)

# Reindex
parameterized = parameterized.reset_index().set_index(['session', 'cluster']
    ).sort_index()

# Drop duration_ceil
parameterized = parameterized.drop('duration_ceil', 1)


## Store
bjoined.to_pickle(
    os.path.join(params['patterns_dir'], 'peri_contact_kappa'))
parameterized.to_pickle(
    os.path.join(params['patterns_dir'], 'kappa_parameterized'))