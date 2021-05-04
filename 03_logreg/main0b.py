## This whole directory runs behavioral decoders
# This script obliviates and aggregates the features.
#   Defines bins and adds to C2_whisk_cycles
#   Obliviates all features after the choice time
#   Adds bin to features (unobliviated and obliviated)
#   Aggregates obliviated features by mean or sum, depending
#   Unstacks analysis_bin and fillna with zero where relevant
#   Sum counts and anti-counts over time and whisker (to aggregated_features)
#   Add task features (to aggregated_features)
#   
# Dumps:
#     obliviated_aggregated_features
#        This will be sliced for each dataset and used for decoding.
#        This also has some additional columns, like summed_count and task features.
#        This will have null analysis_bin for features summed over trial
#     obliviated_unaggregated_features
#     obliviated_unaggregated_features_with_bin
#     unobliviated_unaggregated_features_with_bin
# 
# The unaggregated data is still by cycle, not aggregated into bins. It
# may not contain every possible analysis_bin for every trial, if there were
# no whisk cycles in that bin on that trial.
#
# The obliviated data contains no information after the choice time, which
# is never more than 0.5s by selection.


import json
import pandas
import numpy as np
import my
import my.decoders
import os


## Parameters
with open('../parameters') as fi:
    params = json.load(fi)


## Load data
big_tm = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_tm'))
C2_whisk_cycles = pandas.read_pickle(
    os.path.join(params['patterns_dir'], 'big_C2_tip_whisk_cycles'))

# Results from main0a
unobliviated_features = pandas.read_pickle(
    os.path.join(params['logreg_dir'],
    'unobliviated_unaggregated_features'))
    
    
## Define temporal bins to be used throughout
BINS = {
    'start_t': params['logreg_start_time'],
    'stop_t': params['logreg_stop_time'],
    'bin_width_t': params['logreg_bin_width'],
}
BINS['n_bins'] = int(np.rint(
    float(BINS['stop_t'] - BINS['start_t']) / BINS['bin_width_t']))
BINS['bin_edges_t'] = np.linspace(
    BINS['start_t'], BINS['stop_t'], BINS['n_bins'] + 1)
BINS['bin_centers_t'] = 0.5 * (
    BINS['bin_edges_t'][1:] + 
    BINS['bin_edges_t'][:-1]
)
BINS['start_frame'] = int(np.rint(200 * BINS['start_t']))
BINS['stop_frame'] = int(np.rint(200 * BINS['stop_t']))
BINS['bin_width_frames'] = int(np.rint(200 * BINS['bin_width_t']))
BINS['bin_edges_frames'] = 200 * BINS['bin_edges_t']
BINS['bin_centers_frames'] = 200 * BINS['bin_centers_t']


## Identify whisk cycles to include in analysis (those before choice)
# Add a locked_t column to C2_whisk_cycles to use for binning
# We use peak frame
C2_whisk_cycles['locked_t'] = C2_whisk_cycles['peak_frame_wrt_rwin'] / 200.0

# Join on RT
assert (big_tm['rt'] < BINS['stop_t']).all()
C2_whisk_cycles = C2_whisk_cycles.join(big_tm['rt'], on=['session', 'trial'])
assert (C2_whisk_cycles['rt'] < BINS['stop_t']).all()

# Drop ones where no RT is available (which is only those that are not in big_tm)
assert not C2_whisk_cycles['rt'].isnull().any()


## Obliviate
# Obliviate the features after the RT
obliviated_features = unobliviated_features.loc[C2_whisk_cycles.index[
    (C2_whisk_cycles['locked_t'] < C2_whisk_cycles['rt'])
    ]].copy()


## Bin into analysis bins
# Add analysis_bin to each variable
# This will drop rows that are not contained by a bin
obliviated_features_with_bin = my.decoders.bin_features_into_analysis_bins(
    obliviated_features, C2_whisk_cycles, BINS)
unobliviated_features_with_bin = my.decoders.bin_features_into_analysis_bins(
    unobliviated_features, C2_whisk_cycles, BINS)
    
# More error checks
assert not obliviated_features.isnull().all(1).any()
assert not obliviated_features_with_bin.isnull().all(1).any()


## Aggregate within analysis bin
## From here on we only work with aggregated_features
aggregated_features = obliviated_features_with_bin.mean(
    level=['session', 'trial', 'analysis_bin']
)

# Some metrics need to be aggregated by sum, not mean
# It's a bit weird that contact_binarized is no longer binary
# But I think it makes sense, it's binary only on a whisk-basis, not a bin-basis
metrics_to_sum = [
    'anti_contact_count',
    'contact_binarized',
    'contact_interaction',
    'contact_surplus',
    'touching',
]
for metric in metrics_to_sum:
    aggregated_features[metric] = obliviated_features_with_bin[metric].sum(
        level=['session', 'trial', 'analysis_bin'])

# Error check
assert not aggregated_features.isnull().all(1).any()


## Unstack analysis_bin
# This generates some completely null columns: e.g., contacts in the first bin
# Also some null columns for some trials: e.g., 
#   obliviated features in the last bin,
#   analysis_bins that don't contain a whisk cycle
# And potentially some null columns for all trials in some sessions
# levels: metric, label, analysis_bin
aggregated_features = aggregated_features.unstack('analysis_bin')

# Drop columns that are always null
aggregated_features = aggregated_features.loc[
    :, ~aggregated_features.isnull().all(0)].copy()
aggregated_features.columns = aggregated_features.columns.remove_unused_levels()


## Fillna and intify some metrics
# For these metrics, zero has a meaning
metrics_to_fillna_and_intify = [
    'anti_contact_count',
    'contact_binarized',
    'contact_interaction',
    'contact_surplus',
    'touching',
]

for metric in metrics_to_fillna_and_intify:
    aggregated_features[metric] = (
        aggregated_features[metric].fillna(0).astype(np.int))


## Add a summed_count metric
# Sum contacts by whisker over time
contact_count_by_whisker = (
    aggregated_features.loc[:, 'contact_binarized'].sum(level='label', axis=1) +
    aggregated_features.loc[:, 'contact_surplus'].sum(level='label', axis=1)
)

# Sum contacts by time over whisker
contact_count_by_time = (
    aggregated_features.loc[:, 'contact_binarized'].sum(level='analysis_bin', axis=1) +
    aggregated_features.loc[:, 'contact_surplus'].sum(level='analysis_bin', axis=1)
)

# Grand sum over whisker and time
contact_count_total = contact_count_by_whisker.sum(1).to_frame()

# Sum contact_interaction by label over time
# I don't think it make sense to also sum these over label by time
contact_interaction_count_by_label = (
    aggregated_features.loc[:, 'contact_interaction'].sum(level='label', axis=1)
)

# Relevel to match the other features: metric, label, analysis_bin
contact_count_by_time.columns = pandas.MultiIndex.from_tuples([
    ('contact_count_by_time', 'all', bin) 
    for bin in contact_count_by_time.columns],
    names=['metric', 'label', 'analysis_bin'])
contact_count_by_whisker.columns = pandas.MultiIndex.from_tuples([
    ('contact_count_by_whisker', whisker, None) 
    for whisker in contact_count_by_whisker.columns],
    names=['metric', 'label', 'analysis_bin'])
contact_count_total.columns = pandas.MultiIndex.from_tuples([
    ('contact_count_total', 'all', None)],
    names=['metric', 'label', 'analysis_bin'])
contact_interaction_count_by_label.columns = pandas.MultiIndex.from_tuples([
    ('contact_interaction_count_by_label', label, None) 
    for label in contact_interaction_count_by_label.columns],
    names=['metric', 'label', 'analysis_bin'])
    
# Concat    
concatted = pandas.concat([
    aggregated_features,
    contact_count_by_whisker, 
    contact_count_by_time, 
    contact_count_total, 
    contact_interaction_count_by_label,
    ], 
    axis=1, sort=True, verify_integrity=True).sort_index(axis=1)
assert len(concatted) == len(aggregated_features)
aggregated_features = concatted.copy()


## Add a summed_anti_count metric
# Sum anti_contacts by whisker over time
anti_contact_count_by_whisker = (
    aggregated_features.loc[:, 'anti_contact_count'].sum(level='label', axis=1)
)

# Sum contacts by time over whisker
anti_contact_count_by_time = (
    aggregated_features.loc[:, 'anti_contact_count'].sum(level='analysis_bin', axis=1)
)

# Grand sum over whisker and time
anti_contact_count_total = anti_contact_count_by_whisker.sum(1).to_frame()

# Relevel to match the other features: metric, label, analysis_bin
anti_contact_count_by_time.columns = pandas.MultiIndex.from_tuples([
    ('anti_contact_count_by_time', 'all', bin) 
    for bin in anti_contact_count_by_time.columns],
    names=['metric', 'label', 'analysis_bin'])
anti_contact_count_by_whisker.columns = pandas.MultiIndex.from_tuples([
    ('anti_contact_count_by_whisker', whisker, None) 
    for whisker in anti_contact_count_by_whisker.columns],
    names=['metric', 'label', 'analysis_bin'])
anti_contact_count_total.columns = pandas.MultiIndex.from_tuples([
    ('anti_contact_count_total', 'all', None)],
    names=['metric', 'label', 'analysis_bin'])
    
# Concat    
concatted = pandas.concat([
    aggregated_features,
    anti_contact_count_by_whisker, 
    anti_contact_count_by_time, 
    anti_contact_count_total, 
    ], 
    axis=1, sort=True, verify_integrity=True).sort_index(axis=1)
assert len(concatted) == len(aggregated_features)
aggregated_features = concatted.copy()


## Join on task features
task_features = big_tm[['prev_rewside', 'prev_choice', 'prev_hit']].copy()
task_features['prev_hit'] = (~task_features['prev_hit'].isnull()).astype(np.int)
task_features['prev_rewside'] = task_features['prev_rewside'].replace(
    {'left': 0, 'right': 1}).astype(np.int)
task_features['prev_choice'] = task_features['prev_choice'].replace(
    {'left': 0, 'right': 1}).astype(np.int)

# Set up multi-level columns for concatting
task_features.columns = pandas.MultiIndex.from_tuples([
    ('task', label, None) 
    for label in task_features.columns],
    names=['metric', 'label', 'analysis_bin'])

# Concat
concatted = pandas.concat(
    [aggregated_features, task_features], axis=1, 
    verify_integrity=True).sort_index(axis=1)
assert len(concatted) == len(aggregated_features)
aggregated_features = concatted.copy()


## Error check that we still have no all-null columns
assert not aggregated_features.isnull().all().any()


## Dump
# Unobliviated features, now with bin
unobliviated_features_with_bin.to_pickle(os.path.join(params['logreg_dir'],
    'unobliviated_unaggregated_features_with_bin'))

# The obliviated, aggregated features
# This is for all trials, not just a particular dataset
aggregated_features.to_pickle(os.path.join(params['logreg_dir'],
    'obliviated_aggregated_features'))

# The obliviated, unaggregated features
# With or without an analysis_bin column
obliviated_features.to_pickle(os.path.join(params['logreg_dir'],
    'obliviated_unaggregated_features'))
obliviated_features_with_bin.to_pickle(os.path.join(params['logreg_dir'],
    'obliviated_unaggregated_features_with_bin'))

# Dump the bins
my.misc.pickle_dump(BINS, os.path.join(params['logreg_dir'], 'BINS'))

    
