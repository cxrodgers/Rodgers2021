## main4a: Extract lick times on each trial
# Dumps big_licks in current directory

import json
import tqdm
import MCwatch.behavior
import ArduFSM
import my
import numpy as np
import datetime
import pandas
import kkpandas
import runner.models
import whiskvid
import os


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)
    

## How to lock
LOCKING_COLUMN = 'rwin_time'
EXTRACTION_START_TIME = params['extraction_start_time']
EXTRACTION_STOP_TIME = params['extraction_stop_time']

lick_bins = np.linspace(-2, 1, 31)


## Load data
big_tm = pandas.read_pickle(os.path.join(params['patterns_dir'], 'big_tm'))


## Sessions
# Behavioral datasets
gs_qs = runner.models.GrandSession.objects.filter(
    tags__name=params['decoding_tag'])
session_name_l = sorted(list(gs_qs.values_list('name', flat=True)))


## Helper function
def looks_like_ir(s):
    if (
        (s.startswith('L') or s.startswith('R')) and
        s.endswith('.') and
        (s.count(';') == 2)
        ):
        return True
    else:
        return False


## Iterate over sessions
binned_l = []
binned_keys_l = []
for session_name in tqdm.tqdm(session_name_l):
    ## Get the triggered whisker angle by whisker
    vs = whiskvid.django_db.VideoSession.from_name(session_name)
    
    
    ## Process trial matrix
    # Load trial matrix
    trial_matrix = big_tm.loc[session_name]
    
    # Also get the pldf and use that to get lick times
    ldf = ArduFSM.TrialSpeak.read_logfile_into_df(
        os.path.join(vs.session_path, 'behavioral_logfile'))
    
    # Determine if it's an IR session
    # It looks like KM91, KM100, KM101, KF104, KM127 are capacitive and the rest are IR
    mean_dbgs = ldf.loc[ldf['command'] == 'DBG', 'arg0'].dropna().apply(
        looks_like_ir).mean()
    mouse_name = session_name.split('_')[1]
    if mouse_name in ['KM91', 'KM100', 'KM101', 'KF104', 'KM127']:
        assert mean_dbgs < .01
    else:
        assert mean_dbgs > .99
    
    # Get the lick times
    lick_times = ArduFSM.TrialSpeak.get_commands_from_parsed_lines(ldf, 'TCH')
        
    # Keep only lick types 1 and 2
    lick_times = lick_times[lick_times.arg0.isin([1, 2])]

    # Lock on each lick type
    for lick_type, licks in lick_times.groupby('arg0'):
        folded = kkpandas.Folded.from_flat(
            licks['time'].values / 1000.,
            centers=trial_matrix[LOCKING_COLUMN].values,
            labels=trial_matrix.index.values,
            dstart=EXTRACTION_START_TIME, dstop=EXTRACTION_STOP_TIME,
        )

        # Bin by trial
        binned = kkpandas.Binned.from_folded_by_trial(folded, bins=lick_bins)
        
        # Store
        binned_l.append(binned)
        binned_keys_l.append((session_name, lick_type))

## Concat
big_licks = pandas.concat(
    [binned.rate for binned in binned_l], axis=1,
    keys=binned_keys_l, names=['session', 'lick', 'trial']
).astype(np.int)
big_licks.index = binned.t


## Save
big_licks.to_pickle(os.path.join(params['patterns_dir'], 'big_licks'))